"""End-to-end pipeline orchestrator for the methylation knowledge graph.

Runs all pipeline steps in order, with resumability: each step checks
whether its output already exists and skips if so (unless ``--force``).

Usage::

    # Run full pipeline on dev subset
    python -m epigraph.pipeline --mode dev

    # Run full pipeline on production data
    python -m epigraph.pipeline --mode production

    # Run specific steps
    python -m epigraph.pipeline --steps clinical,annotations,mapping

    # Force re-run of all steps
    python -m epigraph.pipeline --mode dev --force

    # Show pipeline state
    python -m epigraph.pipeline --status
"""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click
import polars as pl

from epigraph.common.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Pipeline state tracking
# ---------------------------------------------------------------------------

STATE_FILE = Path("data/.pipeline_state.json")


class PipelineState:
    """Track pipeline step completion with timestamps.

    Reads and writes a JSON state file to enable resumability and
    provide visibility into pipeline progress.
    """

    def __init__(self, mode: str, state_path: Path = STATE_FILE) -> None:
        self.mode = mode
        self.state_path = state_path
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load state from disk, returning empty state if missing."""
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    data = json.load(f)
                if data.get("mode") == self.mode:
                    return data
            except (json.JSONDecodeError, OSError):
                log.warning("pipeline_state_corrupted", path=str(self.state_path))
        return {"mode": self.mode, "steps": {}}

    def _save(self) -> None:
        """Persist current state to disk."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self._data, f, indent=2)

    def is_completed(self, step: str) -> bool:
        """Check whether a step has been recorded as completed."""
        step_info = self._data.get("steps", {}).get(step, {})
        return step_info.get("status") == "completed"

    def record_completion(self, step: str, elapsed_s: float) -> None:
        """Record that a step completed successfully."""
        self._data.setdefault("steps", {})[step] = {
            "status": "completed",
            "timestamp": datetime.now(UTC).isoformat(),
            "elapsed_s": round(elapsed_s, 2),
        }
        self._save()

    def print_status(self) -> None:
        """Print current pipeline state to stdout."""
        click.echo(f"Pipeline mode: {self._data.get('mode', 'unknown')}")
        steps = self._data.get("steps", {})
        if not steps:
            click.echo("No steps have been recorded.")
            return
        click.echo(f"{'Step':<20} {'Status':<12} {'Elapsed (s)':<12} {'Timestamp'}")
        click.echo("-" * 70)
        for step_name, info in steps.items():
            status = info.get("status", "unknown")
            elapsed = info.get("elapsed_s", "")
            ts = info.get("timestamp", "")
            click.echo(f"{step_name:<20} {status:<12} {str(elapsed):<12} {ts}")


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEP_ORDER = [
    "clinical",
    "annotations",
    "convert",
    "coverage",
    "mapping",
    "islands",
    "aggregate",
    "hypermethylation",
    "compare",
    "enrichment",
    "report",
    "visualise",
    "stats",
]


def _output_exists(path: str | Path) -> bool:
    """Check if an output file or directory with files exists."""
    p = Path(path)
    if p.is_dir():
        return any(p.iterdir())
    return p.exists()


# ---------------------------------------------------------------------------
# Individual pipeline steps
# ---------------------------------------------------------------------------


def step_clinical(cfg: dict[str, Any]) -> None:
    """Parse clinical metadata from CLIN_* worksheets."""
    output = Path(cfg["clinical_output"])
    xlsx = Path(cfg["clinical_xlsx"])

    from epigraph.db_build.load_clinical_metadata import merge_clinical_sheets

    result = merge_clinical_sheets(xlsx)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(output)
    log.info("step_clinical_complete", n_samples=len(result))


def step_annotations(cfg: dict[str, Any]) -> None:
    """Download and parse genomic annotations."""
    from epigraph.db_build.parse_annotations import (
        parse_cpg_islands,
        parse_gencode_gtf,
        parse_goa_gaf,
        parse_reactome,
    )

    ext = Path(cfg["external_dir"])

    gtf = ext / "gencode.v45.annotation.gtf.gz"
    if gtf.exists():
        parse_gencode_gtf(gtf, ext / "genes.parquet")

    gaf = ext / "goa_human.gaf.gz"
    if gaf.exists():
        parse_goa_gaf(gaf, ext / "go_annotations.parquet")

    pw = ext / "ReactomePathways.txt"
    gp = ext / "Ensembl2Reactome.txt"
    if pw.exists() and gp.exists():
        parse_reactome(pw, gp, ext / "reactome_pathways.parquet", ext / "reactome_gene_pathway.parquet")

    cpgi = ext / "cpgIslandExt.txt.gz"
    if cpgi.exists():
        parse_cpg_islands(cpgi, ext / "cpg_islands.parquet")

    _build_symbol_mappings(ext)

    log.info("step_annotations_complete")


def _build_symbol_mappings(ext: Path) -> None:
    """Build gene_symbol -> pathway/term mappings from Ensembl-keyed data."""
    genes_pq = ext / "genes.parquet"
    reactome_pq = ext / "reactome_gene_pathway.parquet"

    if genes_pq.exists() and reactome_pq.exists():
        genes = pl.read_parquet(genes_pq)
        reactome = pl.read_parquet(reactome_pq)
        reactome_genes = reactome.filter(pl.col("gene_id").str.starts_with("ENSG"))
        joined = reactome_genes.join(genes.select("gene_id", "gene_symbol"), on="gene_id", how="inner")
        joined.select("gene_symbol", "pathway_id").unique().write_parquet(
            ext / "reactome_symbol_pathway.parquet"
        )

    go_pq = ext / "go_annotations.parquet"
    if go_pq.exists():
        go = pl.read_parquet(go_pq)
        go.select("gene_symbol", pl.col("go_id").alias("term_id")).unique().write_parquet(
            ext / "go_symbol_term.parquet"
        )


def step_convert(cfg: dict[str, Any]) -> None:
    """Convert full beta matrix CSV to per-chromosome Parquet."""
    from epigraph.db_build.convert_beta_to_parquet import convert_single_pass

    convert_single_pass(
        csv_path=Path(cfg["beta_csv"]),
        output_dir=Path(cfg["beta_chrom_dir"]),
    )
    log.info("step_convert_complete")


def step_coverage(cfg: dict[str, Any]) -> None:
    """Compute per-CpG coverage across all samples."""
    chrom_dir = Path(cfg["beta_chrom_dir"])
    output = Path(cfg["coverage_output"])

    all_coverage = []
    for pq_file in sorted(chrom_dir.glob("beta_chr*.parquet")):
        df = pl.read_parquet(pq_file)
        cpg_cols = [c for c in df.columns if c != "sample_id"]
        n_total = len(df)
        null_counts = df.select(cpg_cols).null_count()
        cov = pl.DataFrame({
            "cpg_id": cpg_cols,
            "n_present": [n_total - null_counts[col][0] for col in cpg_cols],
            "n_total": [n_total] * len(cpg_cols),
        }).with_columns((pl.col("n_present") / n_total).alias("coverage"))
        all_coverage.append(cov)
        del df

    full = pl.concat(all_coverage)
    output.parent.mkdir(parents=True, exist_ok=True)
    full.write_parquet(output)

    n_pass = full.filter(pl.col("coverage") >= 0.95).height
    log.info("step_coverage_complete", n_cpgs=len(full), n_pass_95pct=n_pass)


def step_mapping(cfg: dict[str, Any]) -> None:
    """Map CpGs to genes using coordinate overlap."""
    from epigraph.db_build.map_cpg_to_genes import build_gene_index, map_cpgs_to_genes

    genes = pl.read_parquet(Path(cfg["external_dir"]) / "genes.parquet")
    gene_index = build_gene_index(genes)

    coverage_path = Path(cfg["coverage_output"])
    if coverage_path.exists():
        cpg_ids = pl.read_parquet(coverage_path)["cpg_id"].to_list()
    else:
        from epigraph.common.io import read_beta_header
        header = read_beta_header(cfg["beta_csv"])
        cpg_ids = header[1:]

    mapping = map_cpgs_to_genes(cpg_ids, gene_index, report_intergenic=True)
    output = Path(cfg["mapping_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    mapping.write_parquet(output)
    log.info("step_mapping_complete", n_records=len(mapping))


def step_islands(cfg: dict[str, Any]) -> None:
    """Map CpGs to CpG island context."""
    from epigraph.db_build.map_cpg_to_islands import map_cpgs_to_island_context

    map_cpgs_to_island_context(
        cpg_mapping_path=cfg["mapping_output"],
        islands_path=str(Path(cfg["external_dir"]) / "cpg_islands.parquet"),
        output_path=cfg["island_context_output"],
    )
    log.info("step_islands_complete")


def step_aggregate(cfg: dict[str, Any]) -> None:
    """Aggregate CpG betas to gene-level features."""
    from epigraph.analysis.aggregate_by_chrom import aggregate_genes_by_chromosome

    mapping = pl.read_parquet(cfg["mapping_output"])
    mapping = mapping.filter(
        (pl.col("overlap_type") != "intergenic")
        & (pl.col("gene_symbol") != "")
        & (pl.col("gene_symbol").is_not_null())
    )

    gene_matrix = aggregate_genes_by_chromosome(
        beta_chrom_dir=cfg["beta_chrom_dir"],
        cpg_gene_mapping=mapping,
        method="mean",
        min_cpgs_per_gene=cfg.get("min_cpgs_per_gene", 3),
    )

    output = Path(cfg["gene_features_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    gene_matrix.write_parquet(output)
    log.info("step_aggregate_complete", shape=gene_matrix.shape)


def step_hypermethylation(cfg: dict[str, Any]) -> None:
    """Run hypermethylation scoring at multiple quantiles."""
    from epigraph.analysis.hypermethylation import run_hypermethylation_analysis

    run_hypermethylation_analysis(
        gene_features_path=cfg["gene_features_output"],
        metadata_path=cfg["clinical_output"],
        output_dir=cfg["hypermethylation_dir"],
    )
    log.info("step_hypermethylation_complete")


def step_compare(cfg: dict[str, Any]) -> None:
    """Run cohort comparisons at gene level."""
    from epigraph.analysis.cohort_comparison import run_all_comparisons

    gene_matrix = pl.read_parquet(cfg["gene_features_output"])
    metadata = pl.read_parquet(cfg["clinical_output"])

    results = run_all_comparisons(
        feature_matrix=gene_matrix,
        metadata=metadata,
        sample_col="barcode",
        group_col="clinical_category",
    )

    out_dir = Path(cfg["comparisons_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for label, df in results.items():
        df.write_parquet(out_dir / f"{label}.parquet")
        n_sig = df.filter(pl.col("significant")).height if "significant" in df.columns else 0
        log.info("comparison_result", label=label, n_genes=df.height, n_significant=n_sig)


def step_enrichment(cfg: dict[str, Any]) -> None:
    """Run pathway enrichment on comparison results."""
    from epigraph.analysis.pathway_enrichment import run_pathway_enrichment

    ext = Path(cfg["external_dir"])
    reactome = ext / "reactome_symbol_pathway.parquet"
    comparisons_dir = Path(cfg["comparisons_dir"])
    enrichment_dir = Path(cfg["enrichment_dir"])
    enrichment_dir.mkdir(parents=True, exist_ok=True)

    if not reactome.exists():
        log.warning("reactome_mapping_not_found", path=str(reactome))
        return

    reactome_df = pl.read_parquet(reactome)

    for comp_file in sorted(comparisons_dir.glob("*.parquet")):
        label = comp_file.stem
        diff = pl.read_parquet(comp_file)

        result = run_pathway_enrichment(
            diff_results=diff,
            gene_pathway_mapping=reactome_df,
            method="fisher",
            feature_col="feature",
            gene_col="gene_symbol",
            pathway_col="pathway_id",
        )
        result.write_parquet(enrichment_dir / f"{label}_reactome_fisher.parquet")
        n_sig = result.filter(pl.col("significant")).height if "significant" in result.columns else 0
        log.info("enrichment_result", label=label, method="reactome_fisher", n_pathways=result.height, n_sig=n_sig)

    go_mapping = ext / "go_symbol_term.parquet"
    primary_comp = comparisons_dir / "CRC_vs_Control.parquet"
    if go_mapping.exists() and primary_comp.exists():
        go_df = pl.read_parquet(go_mapping)
        diff = pl.read_parquet(primary_comp)
        result = run_pathway_enrichment(
            diff_results=diff,
            gene_pathway_mapping=go_df,
            method="fisher",
            feature_col="feature",
            gene_col="gene_symbol",
            pathway_col="term_id",
        )
        result.write_parquet(enrichment_dir / "CRC_vs_Control_go_fisher.parquet")
        n_sig = result.filter(pl.col("significant")).height if "significant" in result.columns else 0
        log.info("enrichment_result", label="CRC_vs_Control", method="go_fisher", n_terms=result.height, n_sig=n_sig)


def step_report(cfg: dict[str, Any]) -> None:
    """Generate biomarker candidate report."""
    from epigraph.analysis.biomarker_candidates import (
        generate_biomarker_report,
        load_gene_island_context,
        rank_gene_biomarkers,
        rank_pathway_biomarkers,
    )

    comparisons_dir = Path(cfg["comparisons_dir"])
    enrichment_dir = Path(cfg["enrichment_dir"])
    reports_dir = Path(cfg["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    crc_path = comparisons_dir / "CRC_vs_Control.parquet"
    if not crc_path.exists():
        log.warning("primary_comparison_not_found", path=str(crc_path))
        return

    crc_results = pl.read_parquet(crc_path)
    crc_sig = crc_results.filter(pl.col("significant"))

    gene_candidates = rank_gene_biomarkers(crc_sig, n_top=100)

    fisher_path = enrichment_dir / "CRC_vs_Control_reactome_fisher.parquet"
    pathway_candidates = None
    if fisher_path.exists():
        pathway_candidates = rank_pathway_biomarkers(
            pl.read_parquet(fisher_path), n_top=20
        )

    island_ctx = None
    island_path = Path(cfg["island_context_output"])
    mapping_path = Path(cfg["mapping_output"])
    if island_path.exists() and mapping_path.exists():
        island_ctx = load_gene_island_context(str(island_path), str(mapping_path))

    hms_scores = None
    hms_dir = Path(cfg["hypermethylation_dir"])
    for q in ["0.99", "0.95", "0.999"]:
        hms_path = hms_dir / f"hms_scores_q{q.replace('.', '_')}.parquet"
        if hms_path.exists():
            hms_scores = pl.read_parquet(hms_path)
            break

    report = generate_biomarker_report(
        cpg_candidates=None,
        gene_candidates=gene_candidates,
        pathway_candidates=pathway_candidates,
        cpg_island_context=island_ctx,
        hypermethylation_scores=hms_scores,
    )
    report.write_parquet(reports_dir / "biomarker_report.parquet")
    report.write_csv(reports_dir / "biomarker_report.csv")
    log.info("step_report_complete", n_entries=report.height)


def step_visualise(cfg: dict[str, Any]) -> None:
    """Generate analysis visualisations."""
    from epigraph.analysis.visualise import (
        gene_heatmap,
        hms_distribution,
        pathway_dot_plot,
        volcano_plot,
    )

    figures_dir = Path(cfg["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir = Path(cfg["comparisons_dir"])
    enrichment_dir = Path(cfg["enrichment_dir"])
    ext = Path(cfg["external_dir"])

    for comp_file in sorted(comparisons_dir.glob("*.parquet")):
        label = comp_file.stem
        volcano_plot(
            str(comp_file),
            str(figures_dir / f"volcano_{label}.png"),
            f"{label.replace('_', ' ')} — Gene-level Differential Methylation",
        )

    fisher_path = enrichment_dir / "CRC_vs_Control_reactome_fisher.parquet"
    pw_names_path = ext / "reactome_pathways.parquet"
    if fisher_path.exists() and pw_names_path.exists():
        pathway_dot_plot(
            str(fisher_path),
            str(pw_names_path),
            str(figures_dir / "dotplot_reactome_CRC_vs_Control.png"),
            "CRC vs Control — Reactome Pathway Enrichment",
        )

    hms_dir = Path(cfg["hypermethylation_dir"])
    for q in ["0.95", "0.99", "0.999"]:
        hms_path = hms_dir / f"hms_scores_q{q.replace('.', '_')}.parquet"
        if hms_path.exists():
            hms_distribution(
                str(hms_path),
                str(figures_dir / f"hms_distribution_q{q}.png"),
                f"Hypermethylation Score Distribution (q={q})",
            )

    crc_comp = comparisons_dir / "CRC_vs_Control.parquet"
    gene_feat = Path(cfg["gene_features_output"])
    if crc_comp.exists() and gene_feat.exists():
        gene_heatmap(
            str(gene_feat),
            cfg["clinical_output"],
            str(figures_dir / "heatmap_CRC_vs_Control.png"),
            n_top_genes=50,
            comparison_path=str(crc_comp),
        )

    log.info("step_visualise_complete", output_dir=str(figures_dir))


def step_stats(cfg: dict[str, Any]) -> None:
    """Print dataset statistics."""
    from click.testing import CliRunner

    from epigraph.db_build.dataset_stats import main as stats_main

    runner = CliRunner()
    result = runner.invoke(stats_main, [])
    click.echo(result.output)


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------

STEPS: dict[str, dict[str, Any]] = {
    "clinical": {"fn": step_clinical, "output_key": "clinical_output"},
    "annotations": {"fn": step_annotations, "output_key": "external_dir"},
    "convert": {"fn": step_convert, "output_key": "beta_chrom_dir"},
    "coverage": {"fn": step_coverage, "output_key": "coverage_output"},
    "mapping": {"fn": step_mapping, "output_key": "mapping_output"},
    "islands": {"fn": step_islands, "output_key": "island_context_output"},
    "aggregate": {"fn": step_aggregate, "output_key": "gene_features_output"},
    "hypermethylation": {"fn": step_hypermethylation, "output_key": "hypermethylation_dir"},
    "compare": {"fn": step_compare, "output_key": "comparisons_dir"},
    "enrichment": {"fn": step_enrichment, "output_key": "enrichment_dir"},
    "report": {"fn": step_report, "output_key": "reports_dir"},
    "visualise": {"fn": step_visualise, "output_key": "figures_dir"},
    "stats": {"fn": step_stats, "output_key": None},
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _build_config(mode: str) -> dict[str, Any]:
    """Build pipeline configuration for the given mode.

    Paths are read from environment variables where set, with sensible
    project-relative defaults. See ``.env.example`` for the full list.
    """
    beta_csv = os.environ.get("BETA_MATRIX_PATH", "data/raw/beta_matrix.csv")
    clinical_xlsx = os.environ.get("CLINICAL_METADATA_PATH", "data/raw/clinical_metadata.xlsx")

    if mode == "dev":
        return {
            "beta_csv": beta_csv,
            "clinical_xlsx": clinical_xlsx,
            "external_dir": "data/external",
            "beta_chrom_dir": "data/dev",
            "clinical_output": "data/processed/clinical_metadata.parquet",
            "coverage_output": "data/processed/cpg_coverage_dev.parquet",
            "mapping_output": "data/processed/cpg_gene_mapping_dev.parquet",
            "island_context_output": "data/processed/cpg_island_context_dev.parquet",
            "gene_features_output": "data/processed/gene_features_dev.parquet",
            "hypermethylation_dir": "data/processed/hypermethylation",
            "comparisons_dir": "data/processed/comparisons_dev",
            "enrichment_dir": "data/processed/enrichment_dev",
            "reports_dir": "data/processed/reports_dev",
            "figures_dir": "data/processed/figures_dev",
            "min_cpgs_per_gene": 1,
        }
    else:  # production
        return {
            "beta_csv": beta_csv,
            "clinical_xlsx": clinical_xlsx,
            "external_dir": "data/external",
            "beta_chrom_dir": "data/processed/beta_by_chrom",
            "clinical_output": "data/processed/clinical_metadata.parquet",
            "coverage_output": "data/processed/cpg_coverage_full.parquet",
            "mapping_output": "data/processed/cpg_gene_mapping_full.parquet",
            "island_context_output": "data/processed/cpg_island_context.parquet",
            "gene_features_output": "data/processed/gene_features_full.parquet",
            "hypermethylation_dir": "data/processed/hypermethylation",
            "comparisons_dir": "data/processed/comparisons_full",
            "enrichment_dir": "data/processed/enrichment_full",
            "reports_dir": "data/processed/reports",
            "figures_dir": "data/processed/figures",
            "min_cpgs_per_gene": 3,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("run-pipeline")
@click.option(
    "--mode",
    type=click.Choice(["dev", "production"]),
    default="production",
    help="Pipeline mode: dev (small subset) or production (full data).",
)
@click.option(
    "--steps",
    default=None,
    help="Comma-separated list of steps to run (default: all). "
    f"Available: {', '.join(STEP_ORDER)}",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force re-run even if outputs exist.",
)
@click.option(
    "--start-from",
    default=None,
    type=click.Choice(STEP_ORDER),
    help="Start from this step (skip earlier steps).",
)
@click.option(
    "--status",
    "show_status",
    is_flag=True,
    default=False,
    help="Print the current pipeline state and exit without running.",
)
def main(
    mode: str,
    steps: str | None,
    force: bool,
    start_from: str | None,
    show_status: bool,
) -> None:
    """Run the methylation knowledge graph pipeline.

    Processes: clinical metadata -> annotations -> beta conversion ->
    coverage -> CpG-gene mapping -> CpG island context -> gene aggregation ->
    hypermethylation scoring -> cohort comparison -> pathway enrichment -> stats.
    """
    state = PipelineState(mode)

    if show_status:
        state.print_status()
        return

    cfg = _build_config(mode)

    if steps:
        step_list = [s.strip() for s in steps.split(",")]
    else:
        step_list = STEP_ORDER

    if start_from:
        idx = STEP_ORDER.index(start_from)
        step_list = [s for s in step_list if STEP_ORDER.index(s) >= idx]

    log.info("pipeline_start", mode=mode, steps=step_list, force=force)
    total_start = time.time()

    for step_name in step_list:
        if step_name not in STEPS:
            log.error("unknown_step", step=step_name)
            continue

        step_info = STEPS[step_name]
        output_key = step_info["output_key"]

        if not force:
            if output_key and _output_exists(cfg.get(output_key, "")):
                log.info("step_skipped", step=step_name, reason="output_exists")
                # Record completion so ``--status`` reflects the true pipeline
                # state on re-runs (otherwise steps skipped via existing
                # outputs never appear in the state file).
                if not state.is_completed(step_name):
                    state.record_completion(step_name, 0.0)
                continue
            if state.is_completed(step_name):
                log.info("step_skipped", step=step_name, reason="state_completed")
                continue

        log.info("step_start", step=step_name)
        t0 = time.time()
        try:
            step_info["fn"](cfg)
            elapsed = time.time() - t0
            state.record_completion(step_name, elapsed)
            log.info("step_complete", step=step_name, elapsed_s=round(elapsed, 1))
        except Exception:
            log.exception("step_failed", step=step_name)
            raise

    total = time.time() - total_start
    log.info("pipeline_complete", elapsed_s=round(total, 1))


if __name__ == "__main__":
    main()
