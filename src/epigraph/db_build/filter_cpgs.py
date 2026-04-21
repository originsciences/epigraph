"""Filter CpG sites by data completeness (coverage).

Removes CpG columns that have too much missing data across samples.
Default: keep only CpGs with at least 95% of samples having a value.

This filter is applied BEFORE any downstream analysis to ensure that
gene-level and pathway-level aggregations are based on reliable data.
"""

from __future__ import annotations

from pathlib import Path

import click
import polars as pl

from epigraph.common.logging import get_logger

log = get_logger(__name__)


def compute_cpg_coverage(beta_path: Path) -> pl.DataFrame:
    """Compute per-CpG coverage (fraction of non-null samples).

    Uses vectorized ``null_count()`` across all columns in a single call
    rather than iterating column-by-column.

    Args:
        beta_path: Path to beta matrix Parquet file.

    Returns:
        DataFrame with columns: cpg_id, n_present, n_total, coverage.
    """
    df = pl.read_parquet(beta_path)
    cpg_cols = [c for c in df.columns if c != "sample_id"]
    n_total = len(df)

    # Vectorized: null_count() returns a single-row DF with one column per CpG
    null_counts = df.select(cpg_cols).null_count()

    coverage_df = pl.DataFrame({
        "cpg_id": cpg_cols,
        "n_present": [n_total - null_counts[col][0] for col in cpg_cols],
        "n_total": [n_total] * len(cpg_cols),
    }).with_columns(
        (pl.col("n_present") / pl.lit(n_total)).alias("coverage")
        if n_total > 0
        else pl.lit(0.0).alias("coverage"),
    )

    log.info(
        "cpg_coverage_computed",
        n_cpgs=len(cpg_cols),
        n_samples=n_total,
        mean_coverage=coverage_df["coverage"].mean(),
    )
    return coverage_df


def filter_by_coverage(
    beta_path: Path,
    output_path: Path,
    min_coverage: float = 0.95,
    coverage_stats_path: Path | None = None,
) -> tuple[int, int]:
    """Filter beta matrix to keep only CpGs with sufficient coverage.

    Args:
        beta_path: Path to input beta matrix Parquet.
        output_path: Path for filtered output Parquet.
        min_coverage: Minimum fraction of non-null values (default 0.95).
        coverage_stats_path: Optional path to write coverage stats Parquet.

    Returns:
        Tuple of (n_kept, n_dropped).
    """
    log.info("filter_cpgs_start", input=str(beta_path), min_coverage=min_coverage)

    df = pl.read_parquet(beta_path)
    cpg_cols = [c for c in df.columns if c != "sample_id"]
    n_total_samples = len(df)

    # Compute coverage for each CpG
    null_counts = df.select(cpg_cols).null_count()
    coverages = {
        col: 1.0 - (null_counts[col][0] / n_total_samples)
        for col in cpg_cols
    }

    # Filter
    keep_cols = ["sample_id"] + [
        col for col, cov in coverages.items() if cov >= min_coverage
    ]
    drop_cols = [col for col, cov in coverages.items() if cov < min_coverage]

    n_kept = len(keep_cols) - 1  # subtract sample_id
    n_dropped = len(drop_cols)

    log.info(
        "filter_cpgs_result",
        n_input=len(cpg_cols),
        n_kept=n_kept,
        n_dropped=n_dropped,
        min_coverage=min_coverage,
    )

    # Write filtered matrix
    filtered = df.select(keep_cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_parquet(output_path)

    # Write coverage stats if requested
    if coverage_stats_path is not None:
        stats_records = [
            {"cpg_id": col, "coverage": cov, "kept": cov >= min_coverage}
            for col, cov in coverages.items()
        ]
        stats_df = pl.DataFrame(stats_records)
        coverage_stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df.write_parquet(coverage_stats_path)
        log.info("coverage_stats_written", path=str(coverage_stats_path))

    return n_kept, n_dropped


def filter_cpg_list_by_coverage(
    beta_path: Path,
    min_coverage: float = 0.95,
) -> list[str]:
    """Return list of CpG IDs that pass the coverage filter.

    Useful for filtering the full matrix header without loading all data.

    Args:
        beta_path: Path to beta matrix Parquet.
        min_coverage: Minimum coverage fraction.

    Returns:
        List of CpG IDs passing the filter.
    """
    df = pl.read_parquet(beta_path)
    cpg_cols = [c for c in df.columns if c != "sample_id"]
    n_total = len(df)

    null_counts = df.select(cpg_cols).null_count()
    passing = [
        col for col in cpg_cols
        if (1.0 - null_counts[col][0] / n_total) >= min_coverage
    ]

    log.info("cpg_list_filtered", n_input=len(cpg_cols), n_passing=len(passing))
    return passing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("filter-cpgs")
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    default="data/dev/beta_subset.parquet",
    help="Input beta matrix Parquet.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Output filtered Parquet. Defaults to input with '_filtered' suffix.",
)
@click.option(
    "--min-coverage",
    type=float,
    default=0.95,
    help="Minimum fraction of non-null values per CpG (default: 0.95).",
)
@click.option(
    "--stats-output",
    type=click.Path(),
    default="data/processed/cpg_coverage_stats.parquet",
    help="Path to write per-CpG coverage stats.",
)
def main(
    input_path: str,
    output_path: str | None,
    min_coverage: float,
    stats_output: str,
) -> None:
    """Filter CpG sites by data completeness.

    Removes CpGs where fewer than --min-coverage fraction of samples have
    data. Default keeps only CpGs with >= 95% data present.
    """
    inp = Path(input_path)

    if output_path is None:
        output_path = str(inp.parent / f"{inp.stem}_filtered{inp.suffix}")

    n_kept, n_dropped = filter_by_coverage(
        beta_path=inp,
        output_path=Path(output_path),
        min_coverage=min_coverage,
        coverage_stats_path=Path(stats_output),
    )

    click.echo(f"Filtered CpGs: kept {n_kept:,}, dropped {n_dropped:,}")
    click.echo(f"  Min coverage threshold: {min_coverage:.0%}")
    click.echo(f"  Output: {output_path}")
    click.echo(f"  Stats: {stats_output}")


if __name__ == "__main__":
    main()
