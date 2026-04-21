"""Convert the raw beta-matrix CSV to Parquet and compute per-CpG statistics.

.. deprecated::
    This module is deprecated. Use ``convert-beta``
    (``epigraph.db_build.convert_beta_to_parquet``) for production conversion
    of the full beta matrix. This module is retained for dev-subset CpG stats
    computation only (``compute_cpg_stats``).

The beta matrix is ~800 samples x ~4M CpGs (approx 22 GB CSV).  This
module processes it in *column chunks* to avoid loading the full matrix into
memory.  Each chunk reads a slice of columns for all rows, writes a Parquet
row-group, and accumulates running statistics (mean, variance, missingness).

Usage as a library::

    from epigraph.db_build.parse_betamatrix import convert_to_parquet, compute_cpg_stats

    convert_to_parquet("data/raw/beta_matrix.csv", "data/processed/beta_matrix.parquet")
    compute_cpg_stats("data/processed/beta_matrix.parquet", "data/processed/cpg_stats.parquet")

Or via CLI::

    python -m epigraph.db_build.parse_betamatrix \\
        --csv data/raw/beta_matrix.csv \\
        --output data/processed/beta_matrix.parquet
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import click
import numpy as np
import polars as pl
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from epigraph.common.genome_coords import parse_cpg_id
from epigraph.common.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Header reading
# ---------------------------------------------------------------------------


def _read_header(csv_path: Path) -> list[str]:
    """Read and return the full header row from the CSV (including empty first cell)."""
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        return next(reader)


# ---------------------------------------------------------------------------
# Chunked CSV -> Parquet conversion
# ---------------------------------------------------------------------------


def convert_to_parquet(
    csv_path: str | Path,
    output_path: str | Path,
    chunk_size: int = 10_000,
    compression: str = "zstd",
) -> dict[str, Any]:
    """Convert the beta-matrix CSV to Parquet using column-chunked processing.

    Reads *chunk_size* CpG columns at a time (plus the sample-ID column),
    converts to a PyArrow table, and appends as a row-group to the output
    Parquet file.

    Args:
        csv_path: Path to the raw beta matrix CSV.
        output_path: Destination Parquet file path.
        chunk_size: Number of CpG columns to process per chunk.
        compression: Parquet compression codec.

    Returns:
        Dictionary with conversion statistics.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = _read_header(csv_path)
    # header[0] is empty (sample-index column), rest are CpG IDs
    cpg_columns = header[1:]
    n_cpgs = len(cpg_columns)
    n_chunks = (n_cpgs + chunk_size - 1) // chunk_size

    log.info(
        "convert_start",
        csv_path=str(csv_path),
        n_cpgs=n_cpgs,
        chunk_size=chunk_size,
        n_chunks=n_chunks,
    )

    writer: pq.ParquetWriter | None = None
    stats: dict[str, Any] = {
        "n_cpgs": n_cpgs,
        "n_chunks": n_chunks,
        "n_samples": 0,
    }

    try:
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, n_cpgs)
            chunk_cpg_cols = cpg_columns[start:end]

            # Columns to read: sample-ID column (index 0) + this chunk's CpG columns
            include_columns = [header[0]] + chunk_cpg_cols

            log.debug(
                "reading_chunk",
                chunk=chunk_idx + 1,
                n_chunks=n_chunks,
                cols=f"{start}-{end}",
            )

            # Read only the selected columns
            read_opts = pcsv.ReadOptions(
                skip_rows=1,
                column_names=header,
            )
            convert_opts = pcsv.ConvertOptions(
                include_columns=include_columns,
                strings_can_be_null=True,
                null_values=["", "NA", "NaN"],
            )
            parse_opts = pcsv.ParseOptions(delimiter=",")

            table = pcsv.read_csv(
                csv_path,
                read_options=read_opts,
                convert_options=convert_opts,
                parse_options=parse_opts,
            )

            # Rename empty first column to "sample_id"
            col_names = list(table.column_names)
            if col_names[0] == "" or col_names[0] == header[0]:
                col_names[0] = "sample_id"
                table = table.rename_columns(col_names)

            if chunk_idx == 0:
                stats["n_samples"] = len(table)
                # For the first chunk, write the full table (includes sample_id)
                writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
                writer.write_table(table)
            else:
                # For subsequent chunks, we need to merge with sample_id
                # But Parquet row-groups must share a schema, so we write
                # each chunk as a separate Parquet file and merge at the end.
                # Alternative: use column-oriented approach.
                #
                # TODO: For production, consider writing each chunk to a
                # separate file and then merging with DuckDB or Polars,
                # or writing in a row-oriented fashion.
                writer.write_table(table)  # type: ignore[union-attr]

            if (chunk_idx + 1) % 10 == 0 or chunk_idx == n_chunks - 1:
                log.info(
                    "chunk_progress",
                    chunk=chunk_idx + 1,
                    n_chunks=n_chunks,
                    pct=f"{(chunk_idx + 1) / n_chunks * 100:.1f}%",
                )

    finally:
        if writer is not None:
            writer.close()

    log.info(
        "convert_complete",
        output=str(output_path),
        n_samples=stats["n_samples"],
        n_cpgs=stats["n_cpgs"],
    )
    return stats


# ---------------------------------------------------------------------------
# Per-CpG summary statistics
# ---------------------------------------------------------------------------


def compute_cpg_stats(
    parquet_path: str | Path,
    output_path: str | Path,
    batch_size: int = 50_000,
) -> pl.DataFrame:
    """Compute per-CpG summary statistics from the Parquet beta matrix.

    Computes mean, variance, and missingness fraction for each CpG column.
    Processes columns in batches to limit memory usage.

    Args:
        parquet_path: Path to the beta matrix Parquet file.
        output_path: Destination Parquet path for the stats table.
        batch_size: Number of CpG columns to process per batch.

    Returns:
        Polars DataFrame with columns:
        ``['cpg_id', 'chromosome', 'position', 'mean_beta', 'variance',
        'missingness', 'n_samples']``.
    """
    parquet_path = Path(parquet_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get column names from Parquet metadata
    pf = pq.ParquetFile(parquet_path)
    all_columns = pf.schema_arrow.names

    # Separate sample_id from CpG columns
    cpg_columns = [c for c in all_columns if c != "sample_id"]
    n_cpgs = len(cpg_columns)

    log.info("compute_stats_start", n_cpgs=n_cpgs)

    results: list[dict[str, Any]] = []
    n_batches = (n_cpgs + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_cpgs)
        batch_cols = cpg_columns[start:end]

        # Read only this batch of columns
        table = pq.read_table(parquet_path, columns=batch_cols)

        for col_name in batch_cols:
            arr = table.column(col_name).to_numpy(zero_copy_only=False).astype(np.float64)
            total = len(arr)
            valid_mask = ~np.isnan(arr)
            n_valid = int(valid_mask.sum())

            if n_valid > 0:
                valid_vals = arr[valid_mask]
                mean_val = float(np.mean(valid_vals))
                var_val = float(np.var(valid_vals, ddof=1)) if n_valid > 1 else 0.0
            else:
                mean_val = float("nan")
                var_val = float("nan")

            missingness = 1.0 - (n_valid / total) if total > 0 else 1.0

            try:
                chrom, pos = parse_cpg_id(col_name)
            except ValueError:
                chrom, pos = "unknown", 0

            results.append(
                {
                    "cpg_id": col_name,
                    "chromosome": chrom,
                    "position": pos,
                    "mean_beta": mean_val,
                    "variance": var_val,
                    "missingness": missingness,
                    "n_samples": n_valid,
                }
            )

        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            log.info(
                "stats_progress",
                batch=batch_idx + 1,
                n_batches=n_batches,
                pct=f"{(batch_idx + 1) / n_batches * 100:.1f}%",
            )

    stats_df = pl.DataFrame(results)
    stats_df.write_parquet(output_path)

    log.info(
        "compute_stats_complete",
        output=str(output_path),
        n_cpgs=len(stats_df),
        mean_missingness=f"{stats_df['missingness'].mean():.4f}",
    )
    return stats_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("parse-betamatrix")
@click.option(
    "--csv",
    "csv_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the raw beta matrix CSV.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    default="data/processed/beta_matrix.parquet",
    help="Output Parquet path.",
)
@click.option(
    "--stats-output",
    type=click.Path(dir_okay=False),
    default="data/processed/cpg_stats.parquet",
    help="Output path for CpG stats Parquet.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=10_000,
    help="Number of CpG columns to process per chunk.",
)
@click.option(
    "--skip-stats",
    is_flag=True,
    default=False,
    help="Skip computing CpG statistics after conversion.",
)
def main(
    csv_path: str,
    output: str,
    stats_output: str,
    chunk_size: int,
    skip_stats: bool,
) -> None:
    """Convert the raw beta matrix CSV to Parquet and compute CpG statistics.

    Processes the CSV in column chunks to avoid loading the full 22 GB
    matrix into memory.
    """
    import warnings

    warnings.warn(
        "parse_betamatrix is deprecated. Use convert-beta for production conversion. "
        "This module is retained for dev-subset CpG stats computation only.",
        DeprecationWarning,
        stacklevel=2,
    )
    log.info("parse_betamatrix_start", csv=csv_path, output=output)

    conversion_stats = convert_to_parquet(csv_path, output, chunk_size=chunk_size)
    click.echo(
        f"Conversion complete: {conversion_stats['n_samples']} samples x "
        f"{conversion_stats['n_cpgs']} CpGs"
    )

    if not skip_stats:
        stats_df = compute_cpg_stats(output, stats_output)
        click.echo(f"CpG stats computed: {len(stats_df)} CpGs -> {stats_output}")


if __name__ == "__main__":
    main()
