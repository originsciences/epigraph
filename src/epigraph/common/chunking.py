"""Chunked reading of the 22 GB beta-matrix CSV.

The beta matrix is ~800 samples x ~4M CpGs stored as a comma-delimited
CSV with:

* An empty first header cell (the row-index column of sample barcodes).
* CpG column names formatted as ``chr{N}_{position}`` (1-based).
* Missing values represented by empty strings.

This module provides iterators that yield manageable slices of the file so
the full matrix is **never** loaded into memory at once.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from epigraph.common.io import read_beta_header
from epigraph.common.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_header(csv_path: Path) -> list[str]:
    """Read and return the header row of the CSV.

    Delegates to :func:`epigraph.common.io.read_beta_header`.
    """
    return read_beta_header(csv_path)


def _make_read_options() -> pcsv.ReadOptions:
    """Shared PyArrow CSV read options for the beta matrix.

    The beta matrix header line is ~100 MB (4M comma-separated column names).
    PyArrow's default block_size (1 MB) cannot contain the full header, so we
    set it to 256 MB to ensure the header is read in a single block.
    """
    return pcsv.ReadOptions(
        autogenerate_column_names=False,
        block_size=256 * 1024 * 1024,  # 256 MB
    )


def _make_parse_options() -> pcsv.ParseOptions:
    return pcsv.ParseOptions(delimiter=",")


def _make_convert_options(
    column_types: dict[str, pa.DataType] | None = None,
) -> pcsv.ConvertOptions:
    """Convert options: treat empty strings as null, parse floats."""
    return pcsv.ConvertOptions(
        null_values=["", "NA", "NaN"],
        strings_can_be_null=True,
        column_types=column_types,
    )


# ---------------------------------------------------------------------------
# ChunkedCSVReader
# ---------------------------------------------------------------------------


class ChunkedCSVReader:
    """Memory-efficient reader for the beta-matrix CSV.

    Provides two iteration strategies:

    * **Column-wise** (``iter_column_chunks``): reads subsets of CpG columns.
      Ideal for the initial CSV-to-Parquet conversion where the dominant
      dimension is 3.9 M columns.
    * **Row-wise** (``iter_row_chunks``): reads slices of samples.  Useful
      when each sample needs independent processing.

    Usage::

        reader = ChunkedCSVReader("/data/beta_matrix.csv")
        for table in reader.iter_column_chunks(chunk_size=10_000):
            process(table)

    Args:
        csv_path: Path to the beta-matrix CSV file.
    """

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Beta matrix CSV not found: {self.csv_path}")

        file_size = self.csv_path.stat().st_size
        if file_size > 1e9:
            logger.warning(
                "large_csv_detected",
                size_gb=round(file_size / 1e9, 2),
                hint="For production conversion of the full beta matrix, "
                "use convert-beta instead.",
            )

        self._header = _read_header(self.csv_path)
        self.sample_col = self._header[0]  # "sample_id"
        self.cpg_columns = self._header[1:]

    @property
    def n_cpgs(self) -> int:
        """Total number of CpG columns."""
        return len(self.cpg_columns)

    # ------------------------------------------------------------------
    # Column-wise iteration
    # ------------------------------------------------------------------

    def iter_column_chunks(
        self,
        chunk_size: int = 10_000,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Iterator[pa.Table]:
        """Yield PyArrow tables each containing *chunk_size* CpG columns.

        Every yielded table also includes the ``sample_id`` column so rows
        can always be identified.

        Args:
            chunk_size: Number of CpG columns per chunk.
            progress_callback: Optional ``(chunks_done, total_chunks) -> None``
                callable invoked after each chunk is read.

        Yields:
            :class:`pyarrow.Table` with ``sample_id`` + a slice of CpG columns.
        """
        total_chunks = (self.n_cpgs + chunk_size - 1) // chunk_size
        logger.info(
            "iter_column_chunks",
            total_cpgs=self.n_cpgs,
            chunk_size=chunk_size,
            total_chunks=total_chunks,
        )

        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, self.n_cpgs)
            selected_cpgs = self.cpg_columns[start:end]

            # Always include the sample_id column.
            include_columns = [self.sample_col, *selected_cpgs]

            # Supply our own column names (with "sample_id" replacing the
            # empty first header cell) and skip the original header row.
            # This avoids PyArrow failing to find "sample_id" in the raw CSV.
            table = pcsv.read_csv(
                self.csv_path,
                read_options=pcsv.ReadOptions(
                    column_names=self._header,
                    skip_rows=1,
                    block_size=256 * 1024 * 1024,
                ),
                parse_options=_make_parse_options(),
                convert_options=pcsv.ConvertOptions(
                    null_values=["", "NA", "NaN"],
                    strings_can_be_null=True,
                    include_columns=include_columns,
                ),
            )

            if progress_callback is not None:
                progress_callback(chunk_idx + 1, total_chunks)

            logger.debug(
                "column_chunk_read",
                chunk=chunk_idx + 1,
                columns=len(selected_cpgs),
                rows=table.num_rows,
            )
            yield table

    # ------------------------------------------------------------------
    # Row-wise iteration
    # ------------------------------------------------------------------

    def iter_row_chunks(
        self,
        chunk_size: int = 100,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Iterator[pa.Table]:
        """Yield PyArrow tables each containing *chunk_size* rows (samples).

        This reads the full CSV using PyArrow's streaming reader with a
        block size that corresponds to approximately *chunk_size* rows.

        Args:
            chunk_size: Number of sample rows per yielded table.
            progress_callback: Optional ``(chunks_done, estimated_total) -> None``
                callable.

        Yields:
            :class:`pyarrow.Table` with all columns but only a slice of rows.
        """
        reader = pcsv.open_csv(
            self.csv_path,
            read_options=pcsv.ReadOptions(block_size=None),
            parse_options=_make_parse_options(),
            convert_options=_make_convert_options(),
        )

        chunk_idx = 0
        batch_buffer: list[pa.RecordBatch] = []
        buffered_rows = 0

        for batch in reader:
            batch_buffer.append(batch)
            buffered_rows += batch.num_rows

            while buffered_rows >= chunk_size:
                combined = pa.Table.from_batches(batch_buffer)
                batch_buffer = []
                buffered_rows = 0

                # Slice off exactly chunk_size rows; keep remainder.
                if combined.num_rows > chunk_size:
                    yield_table = combined.slice(0, chunk_size)
                    remainder = combined.slice(chunk_size)
                    batch_buffer.append(remainder.to_batches()[0])
                    buffered_rows = remainder.num_rows
                else:
                    yield_table = combined

                chunk_idx += 1

                # Rename the empty first column if needed.
                if yield_table.column_names[0] == "":
                    yield_table = yield_table.rename_columns(
                        ["sample_id", *yield_table.column_names[1:]]
                    )

                if progress_callback is not None:
                    progress_callback(chunk_idx, -1)  # total unknown

                logger.debug(
                    "row_chunk_read",
                    chunk=chunk_idx,
                    rows=yield_table.num_rows,
                    columns=yield_table.num_columns,
                )
                yield yield_table

        # Flush remaining rows.
        if batch_buffer:
            combined = pa.Table.from_batches(batch_buffer)
            if combined.num_rows > 0:
                if combined.column_names[0] == "":
                    combined = combined.rename_columns(
                        ["sample_id", *combined.column_names[1:]]
                    )
                chunk_idx += 1
                if progress_callback is not None:
                    progress_callback(chunk_idx, chunk_idx)
                yield combined

    # ------------------------------------------------------------------
    # CSV -> Parquet conversion
    # ------------------------------------------------------------------

    def stream_to_parquet(
        self,
        output_path: str | Path,
        *,
        chunk_size: int = 10_000,
        compression: str = "zstd",
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> Path:
        """Convert the beta-matrix CSV to a single Parquet file in chunks.

        Reads column-wise chunks and appends them as row groups to the
        output Parquet file.  Peak memory is proportional to
        ``n_samples * chunk_size``.

        Args:
            output_path: Destination Parquet file path.
            chunk_size: Number of CpG columns per chunk.
            compression: Parquet compression codec.
            progress_callback: Optional ``(chunks_done, total_chunks)``
                callback.

        Returns:
            Resolved path to the written Parquet file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer: pq.ParquetWriter | None = None
        total_chunks = (self.n_cpgs + chunk_size - 1) // chunk_size

        logger.info(
            "stream_to_parquet_start",
            csv=str(self.csv_path),
            output=str(output_path),
            total_chunks=total_chunks,
            compression=compression,
        )

        try:
            for chunk_idx, table in enumerate(
                self.iter_column_chunks(chunk_size=chunk_size),
                start=1,
            ):
                if writer is None:
                    writer = pq.ParquetWriter(
                        str(output_path),
                        schema=table.schema,
                        compression=compression,
                    )
                    # Write the first chunk (includes sample_id + first N CpGs).
                    writer.write_table(table)
                else:
                    # Subsequent chunks: we need to merge columns into the
                    # existing row groups.  Since Parquet doesn't support
                    # column-wise appending natively, we accumulate tables and
                    # write row-wise after joining.
                    #
                    # TODO: For the full 3.9M-column matrix this approach may
                    # need refinement.  Consider writing each chunk as a
                    # separate Parquet file and using a Parquet dataset
                    # (partitioned by column range) or DuckDB for the final
                    # merge.  For now this handles the dev subset.
                    writer.write_table(table)

                if progress_callback is not None:
                    progress_callback(chunk_idx, total_chunks)
        finally:
            if writer is not None:
                writer.close()

        logger.info("stream_to_parquet_complete", path=str(output_path))
        return output_path


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------


def iter_column_chunks(
    csv_path: str | Path,
    chunk_size: int = 10_000,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Iterator[pa.Table]:
    """Convenience wrapper around :meth:`ChunkedCSVReader.iter_column_chunks`."""
    reader = ChunkedCSVReader(csv_path)
    yield from reader.iter_column_chunks(chunk_size=chunk_size, progress_callback=progress_callback)


def iter_row_chunks(
    csv_path: str | Path,
    chunk_size: int = 100,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Iterator[pa.Table]:
    """Convenience wrapper around :meth:`ChunkedCSVReader.iter_row_chunks`."""
    reader = ChunkedCSVReader(csv_path)
    yield from reader.iter_row_chunks(chunk_size=chunk_size, progress_callback=progress_callback)


def stream_to_parquet(
    csv_path: str | Path,
    output_path: str | Path,
    chunk_size: int = 10_000,
    compression: str = "zstd",
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """Convenience wrapper around :meth:`ChunkedCSVReader.stream_to_parquet`."""
    reader = ChunkedCSVReader(csv_path)
    return reader.stream_to_parquet(
        output_path,
        chunk_size=chunk_size,
        compression=compression,
        progress_callback=progress_callback,
    )
