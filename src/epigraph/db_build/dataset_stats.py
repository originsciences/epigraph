"""Dataset overview statistics for the methylation knowledge graph.

Provides a CLI command that reports key metrics about the beta matrix,
clinical metadata, and (when available) annotation data. Designed to be
run at any stage of the pipeline to understand the current state of
processed data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import polars as pl
import pyarrow.parquet as pq

from epigraph.common.genome_coords import parse_cpg_id
from epigraph.common.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Beta matrix stats (from Parquet or CSV header)
# ---------------------------------------------------------------------------


def _beta_stats_from_parquet(path: Path) -> dict:
    """Compute stats from a Parquet beta matrix (dev subset or full)."""
    meta = pq.read_metadata(path)
    schema = pq.read_schema(path)
    col_names = schema.names

    cpg_cols = [c for c in col_names if c != "sample_id"]
    n_samples = meta.num_rows
    n_cpgs = len(cpg_cols)

    # Compute per-CpG missingness and value stats
    df = pl.read_parquet(path)
    cpg_nulls = df.select(cpg_cols).null_count()
    total_nulls = sum(cpg_nulls.row(0))
    total_cells = n_samples * n_cpgs

    # Per-CpG coverage (fraction of non-null).  Guard against empty input:
    # an empty Parquet has n_samples == 0, which would divide by zero.
    if n_samples > 0:
        null_fracs = [cpg_nulls[c][0] / n_samples for c in cpg_cols]
        coverage_fracs = [1.0 - f for f in null_fracs]
    else:
        coverage_fracs = [0.0 for _ in cpg_cols]

    # Coverage distribution
    n_95pct = sum(1 for f in coverage_fracs if f >= 0.95)
    n_90pct = sum(1 for f in coverage_fracs if f >= 0.90)
    n_80pct = sum(1 for f in coverage_fracs if f >= 0.80)
    n_50pct = sum(1 for f in coverage_fracs if f >= 0.50)

    # Chromosome distribution
    chrom_counts: dict[str, int] = {}
    for cpg in cpg_cols:
        try:
            chrom, _ = parse_cpg_id(cpg)
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
        except ValueError:
            pass

    # Value distribution (sample from first 100 CpGs for speed)
    sample_cpgs = cpg_cols[:min(100, len(cpg_cols))]
    vals = df.select(sample_cpgs).to_numpy().flatten()
    import numpy as np
    vals = vals[~np.isnan(vals)]

    return {
        "n_samples": n_samples,
        "n_cpgs": n_cpgs,
        "total_cells": total_cells,
        "total_nulls": total_nulls,
        "null_fraction": total_nulls / total_cells if total_cells > 0 else 0,
        "cpg_coverage_gte_95pct": n_95pct,
        "cpg_coverage_gte_90pct": n_90pct,
        "cpg_coverage_gte_80pct": n_80pct,
        "cpg_coverage_gte_50pct": n_50pct,
        "n_chromosomes": len(chrom_counts),
        "chrom_counts": chrom_counts,
        "beta_mean": float(np.mean(vals)) if len(vals) > 0 else None,
        "beta_median": float(np.median(vals)) if len(vals) > 0 else None,
        "beta_std": float(np.std(vals)) if len(vals) > 0 else None,
    }


def _beta_stats_from_csv_header(path: Path) -> dict:
    """Quick stats from CSV header only (no data loading)."""
    import csv

    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)

    cpg_cols = header[1:]  # skip empty first cell

    # Count rows (samples) by counting lines
    n_samples = 0
    with open(path, "rb") as fh:
        fh.readline()  # skip header
        for _ in fh:
            n_samples += 1

    chrom_counts: dict[str, int] = {}
    for cpg in cpg_cols:
        try:
            chrom, _ = parse_cpg_id(cpg)
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
        except ValueError:
            pass

    return {
        "n_samples": n_samples,
        "n_cpgs": len(cpg_cols),
        "total_cells": n_samples * len(cpg_cols),
        "n_chromosomes": len(chrom_counts),
        "chrom_counts": chrom_counts,
        "note": "CSV header-only stats; run on Parquet for full analysis",
    }


# ---------------------------------------------------------------------------
# Clinical metadata stats
# ---------------------------------------------------------------------------


def _clinical_stats(path: Path) -> dict:
    """Compute stats from clinical metadata Parquet."""
    df = pl.read_parquet(path)
    cat_counts = (
        df.group_by("clinical_category")
        .agg(pl.len().alias("count"))
        .sort("clinical_category")
    )
    return {
        "n_samples": len(df),
        "n_categories": cat_counts.height,
        "categories": {
            row["clinical_category"]: row["count"]
            for row in cat_counts.iter_rows(named=True)
        },
        "columns": df.columns,
    }


# ---------------------------------------------------------------------------
# Annotation stats
# ---------------------------------------------------------------------------


def _annotation_stats(external_dir: Path) -> dict[str, Any]:
    """Report which annotation files are present and their sizes."""
    annotations: dict[str, Any] = {}

    files_to_check = {
        "gencode_gtf": "gencode.v45.annotation.gtf.gz",
        "gencode_genes_parquet": "genes.parquet",
        "go_obo": "go-basic.obo",
        "goa_gaf": "goa_human.gaf.gz",
        "goa_parquet": "go_annotations.parquet",
        "reactome_pathways": "ReactomePathways.txt",
        "reactome_gene_map": "Ensembl2Reactome.txt",
        "reactome_parquet": "reactome_gene_pathway.parquet",
        "cpg_islands": "cpgIslandExt.txt.gz",
        "cpg_islands_parquet": "cpg_islands.parquet",
    }

    for key, filename in files_to_check.items():
        fpath = external_dir / filename
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            annotations[key] = {"present": True, "size_mb": round(size_mb, 2)}
        else:
            annotations[key] = {"present": False}

    # If parsed gene parquet exists, count genes
    gene_pq = external_dir / "genes.parquet"
    if gene_pq.exists():
        genes = pl.read_parquet(gene_pq)
        annotations["n_genes"] = len(genes)
        if "biotype" in genes.columns:
            biotype_counts = (
                genes.group_by("biotype")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
            )
            annotations["top_biotypes"] = {
                row["biotype"]: row["count"]
                for row in biotype_counts.head(10).iter_rows(named=True)
            }

    # CpG-gene mapping
    mapping_pq = Path("data/processed/cpg_gene_mapping.parquet")
    if mapping_pq.exists():
        mapping = pl.read_parquet(mapping_pq)
        annotations["cpg_gene_mapping"] = {
            "n_mappings": len(mapping),
            "n_unique_cpgs": mapping["cpg_id"].n_unique(),
            "n_unique_genes": mapping["gene_symbol"].n_unique()
            if "gene_symbol" in mapping.columns
            else None,
        }

    return annotations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("dataset-stats")
@click.option(
    "--beta-parquet",
    type=click.Path(exists=True),
    default=None,
    help="Path to beta matrix Parquet (dev subset or full).",
)
@click.option(
    "--beta-csv",
    type=click.Path(exists=True),
    default=None,
    help="Path to raw beta matrix CSV (header-only stats, fast).",
)
@click.option(
    "--clinical",
    type=click.Path(exists=True),
    default=None,
    help="Path to clinical metadata Parquet.",
)
@click.option(
    "--external-dir",
    type=click.Path(exists=True),
    default="data/external",
    help="Path to external annotation directory.",
)
def main(
    beta_parquet: str | None,
    beta_csv: str | None,
    clinical: str | None,
    external_dir: str,
) -> None:
    """Report overview statistics on the dataset.

    Shows: number of CpG sites, samples, genes, clinical categories,
    coverage distribution, annotation availability, and basic value stats.
    """
    # Auto-detect paths if not specified
    if beta_parquet is None:
        for candidate in ["data/dev/beta_subset.parquet", "data/processed/beta_matrix.parquet"]:
            if Path(candidate).exists():
                beta_parquet = candidate
                break

    if clinical is None:
        candidate = "data/processed/clinical_metadata.parquet"
        if Path(candidate).exists():
            clinical = candidate

    click.echo("=" * 60)
    click.echo("METHYLATION KNOWLEDGE GRAPH — DATASET STATISTICS")
    click.echo("=" * 60)

    # Beta matrix stats
    if beta_parquet:
        click.echo(f"\n--- Beta Matrix (Parquet): {beta_parquet} ---")
        stats = _beta_stats_from_parquet(Path(beta_parquet))
        click.echo(f"  Samples:           {stats['n_samples']}")
        click.echo(f"  CpG sites:         {stats['n_cpgs']:,}")
        click.echo(f"  Total cells:       {stats['total_cells']:,}")
        click.echo(f"  Missing cells:     {stats['total_nulls']:,} ({stats['null_fraction']:.3%})")
        click.echo(f"  Chromosomes:       {stats['n_chromosomes']}")
        click.echo(f"  CpGs >= 95% cov:   {stats['cpg_coverage_gte_95pct']:,}")
        click.echo(f"  CpGs >= 90% cov:   {stats['cpg_coverage_gte_90pct']:,}")
        click.echo(f"  CpGs >= 80% cov:   {stats['cpg_coverage_gte_80pct']:,}")
        if stats.get("beta_mean") is not None:
            click.echo(f"  Beta mean:         {stats['beta_mean']:.4f}")
            click.echo(f"  Beta median:       {stats['beta_median']:.4f}")
            click.echo(f"  Beta std:          {stats['beta_std']:.4f}")
        click.echo("  Chromosome distribution:")
        from epigraph.common.genome_coords import CHROMOSOME_ORDER

        for ch in sorted(stats["chrom_counts"].keys(), key=lambda c: CHROMOSOME_ORDER.get(c, 99)):
            click.echo(f"    {ch:6s} {stats['chrom_counts'][ch]:>6,}")
    elif beta_csv:
        click.echo(f"\n--- Beta Matrix (CSV header only): {beta_csv} ---")
        stats = _beta_stats_from_csv_header(Path(beta_csv))
        click.echo(f"  Samples:           {stats['n_samples']}")
        click.echo(f"  CpG sites:         {stats['n_cpgs']:,}")
        click.echo(f"  Total cells:       {stats['total_cells']:,}")
        click.echo(f"  Chromosomes:       {stats['n_chromosomes']}")
    else:
        click.echo("\n--- Beta Matrix: not found ---")

    # Clinical metadata stats
    if clinical:
        click.echo(f"\n--- Clinical Metadata: {clinical} ---")
        cstats = _clinical_stats(Path(clinical))
        click.echo(f"  Samples:           {cstats['n_samples']}")
        click.echo(f"  Categories:        {cstats['n_categories']}")
        for cat, count in sorted(cstats["categories"].items()):
            click.echo(f"    {cat:20s} {count:>5}")
    else:
        click.echo("\n--- Clinical Metadata: not found ---")

    # Annotation stats
    ext_path = Path(external_dir)
    if ext_path.exists():
        click.echo(f"\n--- Annotations: {external_dir} ---")
        astats = _annotation_stats(ext_path)
        for key, info in astats.items():
            if isinstance(info, dict) and "present" in info:
                status = "OK" if info["present"] else "MISSING"
                size = f" ({info.get('size_mb', '?')} MB)" if info.get("present") else ""
                click.echo(f"  {key:30s} [{status}]{size}")
            elif key == "n_genes":
                click.echo(f"  Total genes:       {info:,}")
            elif key == "top_biotypes":
                click.echo("  Top biotypes:")
                for bt, cnt in info.items():
                    click.echo(f"    {bt:30s} {cnt:>6,}")
            elif key == "cpg_gene_mapping":
                click.echo(f"  CpG-gene mappings: {info['n_mappings']:,}")
                click.echo(f"    Unique CpGs:     {info['n_unique_cpgs']:,}")
                if info.get("n_unique_genes"):
                    click.echo(f"    Unique genes:    {info['n_unique_genes']:,}")

    click.echo("\n" + "=" * 60)


if __name__ == "__main__":
    main()
