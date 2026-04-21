"""Convert the full beta matrix CSV to partitioned Parquet files.

The beta matrix is ~800 samples x ~4M CpGs = 22 GB CSV.

**Why not PyArrow read_csv?**
PyArrow's ``include_columns`` still parses every column in every row to
find the selected ones, buffering ~32 MB per row internally.  With ~800
rows this exceeds available RAM and gets OOM-killed.

**Strategy: batched single-pass line-by-line extraction.**
1. Read the header once to build column-index → chromosome mapping.
2. Open one ``ParquetWriter`` per chromosome up-front.
3. Read the CSV line-by-line in *batches* of ``batch_size`` samples.
   Each batch gets its own ``(batch_size, total_CpGs)`` float32 buffer.
4. After each batch is full (or at end-of-file), slice the buffer by
   chromosome and ``write_table`` one row group per chromosome.

Memory: peak ≈ ``batch_size × total_CpGs × 4 B`` plus PyArrow's
per-writer row-group buffers. With ``batch_size = 64`` and 4 M CpGs
that is ~1 GB — well within a commodity cloud VM. The historical
unbatched implementation allocated the full sample × CpG array at once
(~13 GB for 800 × 4 M) which OOM'd on 16 GB instances.

I/O: single pass through the 22 GB file.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from epigraph.common.genome_coords import CHROMOSOME_ORDER, parse_cpg_id
from epigraph.common.io import read_beta_header
from epigraph.common.logging import get_logger

log = get_logger(__name__)

DEFAULT_BATCH_SIZE = 64


def _read_header(csv_path: Path) -> list[str]:
    """Read the CSV header and return column names with sample_id renamed.

    Delegates to :func:`epigraph.common.io.read_beta_header`.
    """
    return read_beta_header(csv_path)


def _build_chrom_index(
    header: list[str],
    chrom_filter: set[str] | None = None,
) -> dict[str, list[tuple[int, str]]]:
    """Map each chromosome to a list of (column_index, cpg_name) pairs.

    Args:
        header: Full header list (first element is "sample_id").
        chrom_filter: If set, only include these chromosomes.

    Returns:
        Dict[chrom] -> [(col_idx, cpg_name), ...]
    """
    by_chrom: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for col_idx in range(1, len(header)):
        cpg = header[col_idx]
        try:
            chrom, _ = parse_cpg_id(cpg)
        except ValueError:
            continue
        if chrom_filter and chrom not in chrom_filter:
            continue
        by_chrom[chrom].append((col_idx, cpg))
    return dict(by_chrom)


def _parse_row_into_buffer(
    raw_line: bytes,
    csv_col_indices: np.ndarray,
    buffer: np.ndarray,
    buffer_row: int,
) -> tuple[str, int]:
    """Parse one CSV row into ``buffer[buffer_row, :]``.

    Returns the parsed sample_id and the number of cells that failed float
    conversion (stored as NaN).
    """
    comma = raw_line.find(b",")
    sample_id = raw_line[:comma].decode("utf-8").strip()

    fields = raw_line.decode("utf-8", errors="replace").rstrip("\n").split(",")
    row_arr = np.array(fields, dtype=object)

    valid_col_mask = csv_col_indices < len(fields)
    usable_indices = csv_col_indices[valid_col_mask]
    flat_positions = np.where(valid_col_mask)[0]

    needed = row_arr[usable_indices]
    mask = (needed != "") & (needed != "NA") & (needed != "NaN") & (needed != None)  # noqa: E711
    valid_flat = flat_positions[mask]
    valid_vals = needed[mask]

    parse_failures = 0
    if len(valid_flat) > 0:
        try:
            buffer[buffer_row, valid_flat] = np.asarray(valid_vals, dtype=np.float32)
        except ValueError:
            # Fall back to per-cell conversion when one cell is unparsable.
            for idx, val in zip(valid_flat, valid_vals):
                try:
                    buffer[buffer_row, idx] = float(val)
                except ValueError:
                    parse_failures += 1

    return sample_id, parse_failures


def convert_single_pass(
    csv_path: Path,
    output_dir: Path,
    chrom_filter: set[str] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, int]:
    """Convert the full beta matrix CSV to per-chromosome Parquet files.

    Single-pass, memory-safe: reads each line once, accumulates into a
    fixed-size ``(batch_size, total_CpGs)`` buffer, and flushes one row
    group per chromosome to disk each time the buffer fills.

    Args:
        csv_path: Path to the beta matrix CSV.
        output_dir: Directory for per-chromosome Parquet output.
        chrom_filter: Optional set of chromosomes to include.
        batch_size: Number of samples buffered in RAM between flushes.
            Peak memory ≈ ``batch_size × total_CpGs × 4 B``.

    Returns:
        Dict mapping chromosome to number of CpGs written.
    """
    header = _read_header(csv_path)
    log.info("header_read", n_columns=len(header))

    chrom_index = _build_chrom_index(header, chrom_filter)
    sorted_chroms = sorted(chrom_index.keys(), key=lambda c: CHROMOSOME_ORDER.get(c, 99))
    total_cpgs = sum(len(v) for v in chrom_index.values())
    log.info(
        "index_built",
        n_chromosomes=len(chrom_index),
        total_cpgs=total_cpgs,
        chroms=sorted_chroms,
    )

    # Build flat CpG layout: concatenation of chromosomes in sorted order.
    # This lets one contiguous (batch_size, total_cpgs) buffer cover every
    # chromosome, and per-chromosome flushes are sliced out of it.
    all_cpg_indices: list[int] = []
    chrom_flat_ranges: dict[str, tuple[int, int]] = {}
    flat_pos = 0
    for chrom in sorted_chroms:
        start = flat_pos
        for col_idx, _ in chrom_index[chrom]:
            all_cpg_indices.append(col_idx)
            flat_pos += 1
        chrom_flat_ranges[chrom] = (start, flat_pos)

    n_flat_cols = flat_pos
    csv_col_indices = np.array(all_cpg_indices, dtype=np.int64)

    peak_gb = batch_size * n_flat_cols * 4 / 1e9
    log.info(
        "allocating_batch_buffer",
        batch_size=batch_size,
        shape=f"{batch_size}x{n_flat_cols}",
        peak_gb=round(peak_gb, 2),
    )
    batch_data = np.full((batch_size, n_flat_cols), np.nan, dtype=np.float32)
    batch_sample_ids: list[str] = []

    # Open one ParquetWriter per chromosome up-front so each batch flush is
    # a single ``write_table`` call per chromosome.
    output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, pq.ParquetWriter] = {}
    chrom_cpg_names: dict[str, list[str]] = {}
    for chrom in sorted_chroms:
        cpg_names = [name for _, name in chrom_index[chrom]]
        chrom_cpg_names[chrom] = cpg_names
        schema = pa.schema(
            [("sample_id", pa.string())]
            + [(name, pa.float32()) for name in cpg_names]
        )
        writers[chrom] = pq.ParquetWriter(
            output_dir / f"beta_{chrom}.parquet",
            schema,
            compression="zstd",
        )

    def _flush_batch(buf: np.ndarray, n_rows: int) -> None:
        """Flush the first ``n_rows`` of ``buf`` as one row group per chromosome."""
        if n_rows == 0:
            return
        sample_id_arr = pa.array(batch_sample_ids, type=pa.string())
        for chrom in sorted_chroms:
            start, _end = chrom_flat_ranges[chrom]
            cpg_names = chrom_cpg_names[chrom]
            cols: dict[str, pa.Array] = {"sample_id": sample_id_arr}
            for j, name in enumerate(cpg_names):
                # buf[:n_rows, start + j] is already float32; pa.array
                # creates a zero-copy view where possible.
                cols[name] = pa.array(buf[:n_rows, start + j], type=pa.float32())
            writers[chrom].write_table(pa.table(cols))

    log.info("starting_single_pass", csv=str(csv_path))
    start = time.time()
    rows_in_batch = 0
    rows_total = 0
    parse_failures = 0

    try:
        with open(csv_path, "rb") as fh:
            fh.readline()  # skip header
            for raw_line in fh:
                sample_id, row_fails = _parse_row_into_buffer(
                    raw_line, csv_col_indices, batch_data, rows_in_batch
                )
                batch_sample_ids.append(sample_id)
                parse_failures += row_fails
                rows_in_batch += 1
                rows_total += 1

                if rows_in_batch == batch_size:
                    _flush_batch(batch_data, rows_in_batch)
                    batch_data.fill(np.nan)
                    batch_sample_ids.clear()
                    rows_in_batch = 0

                if rows_total % 100 == 0:
                    elapsed = time.time() - start
                    log.info("rows_processed", n=rows_total, elapsed_s=round(elapsed))

        # Final partial batch.
        _flush_batch(batch_data, rows_in_batch)
    finally:
        for writer in writers.values():
            writer.close()

    elapsed = time.time() - start
    log.info("pass_complete", n_samples=rows_total, elapsed_s=round(elapsed))

    if parse_failures:
        log.warning(
            "float_parse_failures",
            count=parse_failures,
            hint="These cells could not be converted to float and were stored as NaN.",
        )

    results: dict[str, int] = {}
    for chrom in sorted_chroms:
        n_cpgs = len(chrom_index[chrom])
        out_path = output_dir / f"beta_{chrom}.parquet"
        size_mb = out_path.stat().st_size / (1024 * 1024)
        log.info("chromosome_written", chrom=chrom, n_cpgs=n_cpgs, size_mb=round(size_mb, 1))
        results[chrom] = n_cpgs

    return results


@click.command("convert-beta")
@click.option(
    "--csv-path",
    type=click.Path(exists=True),
    default="data/raw/beta_matrix.csv",
    help="Path to the full beta matrix CSV.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="data/processed/beta_by_chrom",
    help="Directory for per-chromosome Parquet files.",
)
@click.option(
    "--chromosomes",
    default=None,
    help="Comma-separated list of chromosomes to convert (default: all).",
)
@click.option(
    "--batch-size",
    default=DEFAULT_BATCH_SIZE,
    show_default=True,
    type=int,
    help="Samples buffered in RAM between Parquet flushes. "
    "Peak memory ≈ batch_size × total_CpGs × 4 bytes.",
)
def main(
    csv_path: str,
    output_dir: str,
    chromosomes: str | None,
    batch_size: int,
) -> None:
    """Convert the full beta matrix CSV to per-chromosome Parquet files.

    Batched single-pass conversion: reads the 22 GB CSV once, buffers
    ``--batch-size`` samples at a time, and streams per-chromosome row
    groups to Parquet.
    """
    chrom_filter = None
    if chromosomes:
        chrom_filter = {c.strip() for c in chromosomes.split(",")}

    total_start = time.time()
    results = convert_single_pass(
        csv_path=Path(csv_path),
        output_dir=Path(output_dir),
        chrom_filter=chrom_filter,
        batch_size=batch_size,
    )

    total_elapsed = time.time() - total_start
    total_cpgs = sum(results.values())
    click.echo(f"\nDone: {total_cpgs:,} CpGs across {len(results)} chromosomes in {total_elapsed:.0f}s")
    for chrom in sorted(results.keys(), key=lambda c: CHROMOSOME_ORDER.get(c, 99)):
        click.echo(f"  {chrom:6s} {results[chrom]:>8,} CpGs")


if __name__ == "__main__":
    main()
