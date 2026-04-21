"""Identify and rank biomarker candidates across CpG, gene, and pathway levels.

Combines differential methylation statistics with pathway enrichment evidence
to produce ranked lists of biomarker candidates.  Results can be exported to
Parquet and written back to TypeDB as ``is_biomarker_candidate`` flags.

Scoring approach::

    CpG score   = |effect_size| * -log10(q_value)
    Gene score  = diff_score + enrichment_bonus
    Pathway score = -log10(enrichment_q) * mean_|effect|
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np
import polars as pl

from epigraph.common.logging import get_logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# CpG-level ranking
# ---------------------------------------------------------------------------


def rank_cpg_biomarkers(
    cpg_diff_results: pl.DataFrame,
    n_top: int = 100,
    *,
    effect_col: str = "cohens_d",
    q_value_col: str = "q_value",
    feature_col: str = "feature",
    annotation_df: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Rank CpG sites by biomarker potential.

    The combined score is ``|effect_size| * -log10(q_value)``, giving
    priority to CpGs with both large effect sizes and high statistical
    significance.

    Args:
        cpg_diff_results: Differential methylation results with at least
            columns *feature_col*, *effect_col*, and *q_value_col*.
        n_top: Number of top candidates to return.
        effect_col: Column containing effect size (e.g. Cohen's d).
        q_value_col: Column containing FDR-adjusted p-values.
        feature_col: Column containing CpG identifiers.
        annotation_df: Optional DataFrame with CpG annotations (columns:
            ``cpg_id``, ``gene_symbol``, ``region``, ``cpg_island_status``).
            If provided, annotations are joined to the output.

    Returns:
        Top *n_top* CpG biomarker candidates sorted by descending score,
        with columns: ``cpg_id``, ``effect_size``, ``q_value``, ``score``,
        and any joined annotations.
    """
    log.info("rank_cpg_biomarkers_start", n_features=cpg_diff_results.height, n_top=n_top)

    df = cpg_diff_results.select([
        pl.col(feature_col).alias("cpg_id"),
        pl.col(effect_col).alias("effect_size"),
        pl.col(q_value_col).alias("q_value"),
    ])

    # Compute combined score: |effect_size| * -log10(q_value)
    # Clamp q_value to avoid log(0) and extremely inflated scores
    df = df.with_columns(
        pl.when(pl.col("q_value") > 0)
        .then(pl.col("q_value"))
        .otherwise(pl.lit(1e-300))
        .alias("q_value_safe"),
    )

    df = df.with_columns(
        (
            pl.col("effect_size").abs()
            * (-pl.col("q_value_safe").log(base=10))
        ).alias("score"),
    ).drop("q_value_safe")

    # Sort by score descending, take top N
    df = df.sort("score", descending=True, nulls_last=True).head(n_top)

    # Join annotations if provided
    if annotation_df is not None:
        annotation_cols = [c for c in annotation_df.columns if c != "cpg_id"]
        if annotation_cols:
            df = df.join(
                annotation_df.select(["cpg_id", *annotation_cols]),
                on="cpg_id",
                how="left",
            )

    log.info("rank_cpg_biomarkers_complete", n_candidates=df.height)
    return df


# ---------------------------------------------------------------------------
# Gene-level ranking
# ---------------------------------------------------------------------------


def rank_gene_biomarkers(
    gene_diff_results: pl.DataFrame,
    gene_enrichment: pl.DataFrame | None = None,
    n_top: int = 50,
    *,
    effect_col: str = "cohens_d",
    q_value_col: str = "q_value",
    feature_col: str = "feature",
    enrichment_bonus_weight: float = 1.0,
    hypermethylation_scores: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Rank genes combining differential methylation and enrichment evidence.

    The gene score is computed as::

        diff_score = |effect_size| * -log10(q_value)
        enrichment_bonus = sum(-log10(pathway_q)) for pathways containing gene
        final_score = diff_score + enrichment_bonus_weight * enrichment_bonus

    Compatible with the output of :func:`cohort_comparison.compare_groups`
    (after FDR correction), which produces columns: ``feature``,
    ``mean_group1``, ``mean_group2``, ``delta_mean``, ``cohens_d``,
    ``p_value``, ``q_value``, ``significant``.

    Args:
        gene_diff_results: Differential methylation results at the gene level.
        gene_enrichment: Optional pathway enrichment results (from
            :mod:`pathway_enrichment`).  Must have columns ``pathway``,
            ``q_value``, and overlap gene information.  If *None*, only
            differential methylation scores are used.
        n_top: Number of top candidates to return.
        effect_col: Column containing effect size.
        q_value_col: Column containing FDR-adjusted p-values.
        feature_col: Column containing gene identifiers.
        enrichment_bonus_weight: Weight applied to the enrichment bonus
            component of the score.
        hypermethylation_scores: Optional DataFrame with columns
            ``sample_id``, ``hms_count``, and ``clinical_category``.
            If provided, summary HMS stats are joined to results.

    Returns:
        Top *n_top* gene biomarker candidates with columns: ``gene``,
        ``effect_size``, ``q_value``, ``diff_score``, ``enrichment_bonus``,
        ``score``, and optionally ``delta_mean``, ``significant``.
    """
    log.info("rank_gene_biomarkers_start", n_features=gene_diff_results.height, n_top=n_top)

    # Build core selection -- include delta_mean and significant if available
    select_exprs: list[Any] = [
        pl.col(feature_col).alias("gene"),
        pl.col(effect_col).alias("effect_size"),
        pl.col(q_value_col).alias("q_value"),
    ]

    has_delta_mean = "delta_mean" in gene_diff_results.columns
    has_significant = "significant" in gene_diff_results.columns

    if has_delta_mean:
        select_exprs.append(pl.col("delta_mean"))
    if has_significant:
        select_exprs.append(pl.col("significant"))

    df = gene_diff_results.select(select_exprs)

    # Diff score component
    df = df.with_columns(
        pl.when(pl.col("q_value") > 0)
        .then(pl.col("q_value"))
        .otherwise(pl.lit(1e-300))
        .alias("q_value_safe"),
    )
    df = df.with_columns(
        (
            pl.col("effect_size").abs()
            * (-pl.col("q_value_safe").log(base=10))
        ).alias("diff_score"),
    ).drop("q_value_safe")

    # Enrichment bonus: aggregate pathway evidence per gene
    if gene_enrichment is not None and gene_enrichment.height > 0:
        log.debug("enrichment_bonus_computation", n_enrichment_rows=gene_enrichment.height)

        # If enrichment results contain overlap_genes, compute per-gene bonus
        if "overlap_genes" in gene_enrichment.columns:
            # Explode pathway-gene pairs and sum -log10(q) per gene
            enrich_q_col = (
                "q_value" if "q_value" in gene_enrichment.columns else "p_value"
            )
            enrich_exploded = gene_enrichment.select([
                pl.col("overlap_genes"),
                pl.col(enrich_q_col).alias("_enrich_q"),
            ]).explode("overlap_genes")

            enrich_exploded = enrich_exploded.with_columns(
                pl.when(pl.col("_enrich_q") > 0)
                .then(pl.col("_enrich_q"))
                .otherwise(pl.lit(1e-300))
                .alias("_enrich_q_safe"),
            )

            gene_bonus = enrich_exploded.group_by("overlap_genes").agg(
                (-pl.col("_enrich_q_safe").log(base=10)).sum().alias("enrichment_bonus"),
            ).rename({"overlap_genes": "gene"})

            df = df.join(gene_bonus, on="gene", how="left")
            df = df.with_columns(
                pl.col("enrichment_bonus").fill_null(0.0),
            )
        else:
            df = df.with_columns(pl.lit(0.0).alias("enrichment_bonus"))
    else:
        df = df.with_columns(pl.lit(0.0).alias("enrichment_bonus"))

    df = df.with_columns(
        (pl.col("diff_score") + enrichment_bonus_weight * pl.col("enrichment_bonus")).alias(
            "score"
        ),
    )

    df = df.sort("score", descending=True, nulls_last=True).head(n_top)

    log.info("rank_gene_biomarkers_complete", n_candidates=df.height)
    return df


# ---------------------------------------------------------------------------
# Pathway-level ranking
# ---------------------------------------------------------------------------


def rank_pathway_biomarkers(
    pathway_enrichment_results: pl.DataFrame,
    n_top: int = 20,
    *,
    q_value_col: str = "q_value",
    pathway_col: str = "pathway",
) -> pl.DataFrame:
    """Rank pathways by enrichment significance.

    Score is ``-log10(q_value)`` optionally weighted by effect magnitude
    if an ``odds_ratio`` or ``nes`` column is available.

    Args:
        pathway_enrichment_results: Enrichment results from
            :func:`pathway_enrichment.run_pathway_enrichment`.
        n_top: Number of top pathways to return.
        q_value_col: Column name for FDR-adjusted p-values.
        pathway_col: Column name for pathway identifiers.

    Returns:
        Top *n_top* pathway biomarker candidates with columns: ``pathway``,
        ``q_value``, ``score``, and any available effect columns.
    """
    log.info(
        "rank_pathway_biomarkers_start",
        n_pathways=pathway_enrichment_results.height,
        n_top=n_top,
    )

    # Select core columns plus any effect-size columns
    core_cols = [pathway_col, q_value_col, "p_value"]
    candidate_effect_names = ["odds_ratio", "nes", "es", "n_overlap", "n_hits"]
    effect_cols = [
        c for c in candidate_effect_names if c in pathway_enrichment_results.columns
    ]
    select_cols = [c for c in core_cols + effect_cols if c in pathway_enrichment_results.columns]

    df = pathway_enrichment_results.select(select_cols)

    # Rename pathway column for consistency
    if pathway_col != "pathway":
        df = df.rename({pathway_col: "pathway"})

    # Compute score
    df = df.with_columns(
        pl.when(pl.col(q_value_col) > 0)
        .then(pl.col(q_value_col))
        .otherwise(pl.lit(1e-300))
        .alias("q_value_safe"),
    )

    if "odds_ratio" in df.columns:
        # Fisher-based: score = log2(odds_ratio) * -log10(q_value)
        df = df.with_columns(
            (
                pl.col("odds_ratio").log(base=2).abs()
                * (-pl.col("q_value_safe").log(base=10))
            ).alias("score"),
        )
    elif "nes" in df.columns:
        # GSEA-based: score = |NES| * -log10(q_value)
        df = df.with_columns(
            (
                pl.col("nes").abs()
                * (-pl.col("q_value_safe").log(base=10))
            ).alias("score"),
        )
    else:
        # Fall back to just significance
        df = df.with_columns(
            (-pl.col("q_value_safe").log(base=10)).alias("score"),
        )

    df = df.drop("q_value_safe")
    df = df.sort("score", descending=True, nulls_last=True).head(n_top)

    log.info("rank_pathway_biomarkers_complete", n_candidates=df.height)
    return df


# ---------------------------------------------------------------------------
# Island context loader
# ---------------------------------------------------------------------------


def load_gene_island_context(
    island_context_path: str | Path,
    cpg_gene_mapping_path: str | Path,
) -> pl.DataFrame:
    """Load per-gene CpG island context distribution for biomarker reports.

    Reads the CpG island context file (columns: cpg_id, chromosome, position,
    context, island_id) and joins with the CpG-gene mapping file (columns:
    cpg_id, gene_symbol) to produce a per-gene island context distribution
    suitable for :func:`generate_biomarker_report`.

    Args:
        island_context_path: Path to ``cpg_island_context.parquet`` with
            columns ``cpg_id``, ``chromosome``, ``position``, ``context``,
            ``island_id``.
        cpg_gene_mapping_path: Path to ``cpg_gene_mapping.parquet`` with
            columns ``cpg_id``, ``gene_symbol``.

    Returns:
        DataFrame with columns ``gene_symbol`` and ``cpg_island_status``
        (the context value: island/shore/shelf/open_sea), one row per
        CpG-gene pair.
    """
    island_ctx = pl.read_parquet(
        Path(island_context_path),
        columns=["cpg_id", "context"],
    )
    cpg_genes = pl.read_parquet(
        Path(cpg_gene_mapping_path),
        columns=["cpg_id", "gene_symbol"],
    )

    joined = cpg_genes.join(island_ctx, on="cpg_id", how="inner")

    result = joined.select(
        pl.col("gene_symbol"),
        pl.col("context").alias("cpg_island_status"),
    )

    log.info(
        "load_gene_island_context_complete",
        n_rows=result.height,
        n_genes=result["gene_symbol"].n_unique(),
    )
    return result


# ---------------------------------------------------------------------------
# Consolidated report
# ---------------------------------------------------------------------------


def generate_biomarker_report(
    cpg_candidates: pl.DataFrame | None,
    gene_candidates: pl.DataFrame | None,
    pathway_candidates: pl.DataFrame | None,
    *,
    enrichment_results: pl.DataFrame | None = None,
    cpg_island_context: pl.DataFrame | None = None,
    hypermethylation_scores: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Create a consolidated multi-level biomarker report.

    Combines CpG, gene, and pathway candidates into a single long-format
    DataFrame for review and export.  Optionally enriches gene-level
    candidates with pathway memberships, CpG island context distribution,
    and hypermethylation status information.

    Args:
        cpg_candidates: Ranked CpG candidates from :func:`rank_cpg_biomarkers`.
        gene_candidates: Ranked gene candidates from :func:`rank_gene_biomarkers`.
        pathway_candidates: Ranked pathway candidates from
            :func:`rank_pathway_biomarkers`.
        enrichment_results: Optional pathway enrichment results with columns
            ``pathway`` and ``overlap_genes`` (list of gene symbols).  Used
            to annotate gene candidates with their pathway memberships.
        cpg_island_context: Optional DataFrame with columns ``gene_symbol``
            and ``cpg_island_status`` (or ``relation_to_island``).  Used to
            annotate gene candidates with CpG island context distribution.
        hypermethylation_scores: Optional DataFrame with columns
            ``sample_id``, ``hms_count``, and ``clinical_category``.  Used
            to compute and attach hypermethylation status summary.

    Returns:
        Long-format DataFrame with columns: ``level``, ``identifier``,
        ``score``, ``effect_size``, ``q_value``, ``rank``, and optionally
        ``pathways``, ``cpg_island_context``, ``delta_mean``, ``significant``.
    """
    log.info(
        "generate_biomarker_report",
        n_cpg=cpg_candidates.height if cpg_candidates is not None else 0,
        n_gene=gene_candidates.height if gene_candidates is not None else 0,
        n_pathway=pathway_candidates.height if pathway_candidates is not None else 0,
    )

    # Build gene -> pathways lookup from enrichment results
    gene_pathways: dict[str, list[str]] = {}
    if enrichment_results is not None and "overlap_genes" in enrichment_results.columns:
        for row in enrichment_results.iter_rows(named=True):
            pathway_name = row.get("pathway", "")
            overlap = row.get("overlap_genes", [])
            if overlap is None:
                continue
            for gene in overlap:
                gene_pathways.setdefault(gene, []).append(pathway_name)

    # Build gene -> CpG island context distribution
    gene_island_ctx: dict[str, str] = {}
    if cpg_island_context is not None:
        island_col = (
            "cpg_island_status"
            if "cpg_island_status" in cpg_island_context.columns
            else "relation_to_island"
            if "relation_to_island" in cpg_island_context.columns
            else None
        )
        gene_sym_col = (
            "gene_symbol"
            if "gene_symbol" in cpg_island_context.columns
            else "gene"
            if "gene" in cpg_island_context.columns
            else None
        )
        if island_col is not None and gene_sym_col is not None:
            # Build gene -> list[context] mapping, then summarise as a
            # short "ctx:count/total" string per gene.
            for row in cpg_island_context.select([gene_sym_col, island_col]).group_by(
                gene_sym_col
            ).agg(pl.col(island_col)).iter_rows(named=True):
                gene_name = row[gene_sym_col]
                contexts = row[island_col]
                if contexts:
                    # Count occurrences
                    from collections import Counter
                    counts = Counter(contexts)
                    total = sum(counts.values())
                    parts = [
                        f"{ctx}:{cnt}/{total}"
                        for ctx, cnt in sorted(counts.items(), key=lambda x: -x[1])
                    ]
                    gene_island_ctx[gene_name] = "; ".join(parts)

    # Hypermethylation summary string
    hms_summary = ""
    if hypermethylation_scores is not None and hypermethylation_scores.height > 0:
        group_stats = hypermethylation_scores.group_by("clinical_category").agg(
            pl.col("hms_count").mean().alias("mean_hms"),
        )
        parts = []
        for row in group_stats.sort("clinical_category").iter_rows(named=True):
            parts.append(f"{row['clinical_category']}={row['mean_hms']:.1f}")
        hms_summary = "; ".join(parts)

    rows: list[dict[str, Any]] = []

    # CpG level
    if cpg_candidates is not None:
        for rank_idx, row in enumerate(cpg_candidates.iter_rows(named=True), start=1):
            rows.append({
                "level": "cpg",
                "identifier": row.get("cpg_id", ""),
                "score": row.get("score", np.nan),
                "effect_size": row.get("effect_size", np.nan),
                "q_value": row.get("q_value", np.nan),
                "rank": rank_idx,
                "pathways": "",
                "cpg_island_context": "",
                "hms_summary": "",
            })

    # Gene level
    if gene_candidates is None:
        gene_candidates = pl.DataFrame()
    for rank_idx, row in enumerate(gene_candidates.iter_rows(named=True), start=1):
        gene_name = row.get("gene", "")
        rows.append({
            "level": "gene",
            "identifier": gene_name,
            "score": row.get("score", np.nan),
            "effect_size": row.get("effect_size", np.nan),
            "q_value": row.get("q_value", np.nan),
            "rank": rank_idx,
            "pathways": "; ".join(gene_pathways.get(gene_name, [])),
            "cpg_island_context": gene_island_ctx.get(gene_name, ""),
            "hms_summary": hms_summary,
        })

    # Pathway level
    if pathway_candidates is None:
        pathway_candidates = pl.DataFrame()
    for rank_idx, row in enumerate(pathway_candidates.iter_rows(named=True), start=1):
        rows.append({
            "level": "pathway",
            "identifier": row.get("pathway", ""),
            "score": row.get("score", np.nan),
            "effect_size": row.get("odds_ratio", row.get("nes", np.nan)),
            "q_value": row.get("q_value", np.nan),
            "rank": rank_idx,
            "pathways": "",
            "cpg_island_context": "",
            "hms_summary": "",
        })

    if not rows:
        return pl.DataFrame({
            "level": [],
            "identifier": [],
            "score": [],
            "effect_size": [],
            "q_value": [],
            "rank": [],
            "pathways": [],
            "cpg_island_context": [],
            "hms_summary": [],
        })

    report = pl.DataFrame(rows)
    log.info("biomarker_report_generated", total_candidates=report.height)
    return report


# ---------------------------------------------------------------------------
# TypeDB export
# ---------------------------------------------------------------------------


def export_to_typedb(
    candidates: pl.DataFrame,
    typedb_driver: Any,
    *,
    database: str = "epigraph",
) -> int:
    """Write biomarker candidates back to TypeDB.

    Sets the ``is_biomarker_candidate`` attribute on matching entities and
    stores the biomarker score and rank.

    Args:
        candidates: Biomarker report DataFrame (from
            :func:`generate_biomarker_report`).
        typedb_driver: TypeDB driver connection (``typedb.driver.TypeDB``).
        database: TypeDB database name.

    Returns:
        Number of entities updated in TypeDB.

    Raises:
        ConnectionError: If the TypeDB connection fails.
    """
    raise NotImplementedError("TypeDB export not yet implemented")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("biomarker-candidates")
@click.option(
    "--cpg-diff",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet file with CpG-level differential methylation results.",
)
@click.option(
    "--gene-diff",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet file with gene-level differential methylation results.",
)
@click.option(
    "--pathway-enrichment",
    "pathway_enrichment_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet file with pathway enrichment results.",
)
@click.option(
    "--cpg-annotations",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet file with CpG annotations (gene, region, island status).",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to write biomarker candidate files.",
)
@click.option("--n-top-cpg", type=int, default=100, show_default=True, help="Top CpG candidates.")
@click.option("--n-top-gene", type=int, default=50, show_default=True, help="Top gene candidates.")
@click.option(
    "--n-top-pathway", type=int, default=20, show_default=True, help="Top pathway candidates."
)
def main(
    cpg_diff: str | None,
    gene_diff: str | None,
    pathway_enrichment_path: str | None,
    cpg_annotations: str | None,
    output_dir: str,
    n_top_cpg: int,
    n_top_gene: int,
    n_top_pathway: int,
) -> None:
    """Identify and rank biomarker candidates at CpG, gene, and pathway levels.

    Reads differential methylation and enrichment results, computes
    combined scores, and produces ranked candidate lists plus a
    consolidated report.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    log.info("biomarker_candidates_start", output_dir=str(output))

    cpg_candidates = pl.DataFrame()
    gene_candidates = pl.DataFrame()
    pathway_candidates = pl.DataFrame()

    # --- CpG-level ---
    if cpg_diff is not None:
        cpg_df = pl.read_parquet(cpg_diff)
        ann_df = pl.read_parquet(cpg_annotations) if cpg_annotations else None
        cpg_candidates = rank_cpg_biomarkers(
            cpg_df, n_top=n_top_cpg, annotation_df=ann_df
        )
        cpg_candidates.write_parquet(output / "cpg_biomarker_candidates.parquet")
        log.info("cpg_candidates_written", n=cpg_candidates.height)

    # --- Gene-level ---
    if gene_diff is not None:
        gene_df = pl.read_parquet(gene_diff)
        pw_enrich = None
        if pathway_enrichment_path is not None:
            pw_enrich = pl.read_parquet(pathway_enrichment_path)
        gene_candidates = rank_gene_biomarkers(
            gene_df, gene_enrichment=pw_enrich, n_top=n_top_gene
        )
        gene_candidates.write_parquet(output / "gene_biomarker_candidates.parquet")
        log.info("gene_candidates_written", n=gene_candidates.height)

    # --- Pathway-level ---
    if pathway_enrichment_path is not None:
        pw_df = pl.read_parquet(pathway_enrichment_path)
        pathway_candidates = rank_pathway_biomarkers(pw_df, n_top=n_top_pathway)
        pathway_candidates.write_parquet(output / "pathway_biomarker_candidates.parquet")
        log.info("pathway_candidates_written", n=pathway_candidates.height)

    # --- Consolidated report ---
    if cpg_candidates.height > 0 or gene_candidates.height > 0 or pathway_candidates.height > 0:
        report = generate_biomarker_report(cpg_candidates, gene_candidates, pathway_candidates)
        report.write_parquet(output / "biomarker_report.parquet")
        log.info("biomarker_report_written", n=report.height)

    log.info("biomarker_candidates_complete")


if __name__ == "__main__":
    main()
