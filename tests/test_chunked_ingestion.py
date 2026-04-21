"""Tests for chunked CSV reading of the beta matrix.

Exercises ``ChunkedCSVReader``, ``iter_column_chunks``, ``iter_row_chunks``,
header parsing, and null handling.

Note: ``iter_column_chunks`` uses PyArrow ``include_columns`` which reads
column names from the raw CSV header. The ``ChunkedCSVReader`` renames the
empty first cell to ``sample_id`` internally, so column-chunk iteration
requires the CSV to have ``sample_id`` as the literal first header cell.
The ``tmp_beta_csv_named_header`` fixture provides this variant, while
``tmp_beta_csv`` (empty first cell) is used for row-chunk tests which
handle the rename after reading.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from epigraph.common.chunking import ChunkedCSVReader, iter_column_chunks

# =========================================================================
# ChunkedCSVReader construction
# =========================================================================


class TestChunkedCSVReaderInit:
    """Tests for ``ChunkedCSVReader`` initialisation and header parsing."""

    def test_constructs_from_valid_csv(self, tmp_beta_csv: Path) -> None:
        reader = ChunkedCSVReader(tmp_beta_csv)
        assert reader.csv_path == tmp_beta_csv

    def test_sample_col_is_sample_id(self, tmp_beta_csv: Path) -> None:
        reader = ChunkedCSVReader(tmp_beta_csv)
        assert reader.sample_col == "sample_id"

    def test_cpg_columns_parsed(self, tmp_beta_csv: Path) -> None:
        reader = ChunkedCSVReader(tmp_beta_csv)
        assert reader.n_cpgs == 10
        assert "chr1_100" in reader.cpg_columns
        assert "chrY_50" in reader.cpg_columns

    def test_empty_first_header_cell_handled(self, tmp_beta_csv: Path) -> None:
        """The empty first header cell should become 'sample_id'."""
        reader = ChunkedCSVReader(tmp_beta_csv)
        assert reader._header[0] == "sample_id"

    def test_named_header_also_works(
        self, tmp_beta_csv_named_header: Path
    ) -> None:
        """CSV with explicit ``sample_id`` header should also parse correctly."""
        reader = ChunkedCSVReader(tmp_beta_csv_named_header)
        assert reader.sample_col == "sample_id"
        assert reader.n_cpgs == 10

    def test_nonexistent_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ChunkedCSVReader(tmp_path / "nonexistent.csv")


# =========================================================================
# iter_column_chunks
# =========================================================================


class TestIterColumnChunks:
    """Tests for column-wise iteration.

    Uses ``tmp_beta_csv_named_header`` because ``iter_column_chunks``
    passes the internal ``sample_id`` column name to PyArrow's
    ``include_columns``, which requires it to exist in the raw CSV header.
    """

    def test_single_chunk_all_columns(
        self, tmp_beta_csv_named_header: Path
    ) -> None:
        """When chunk_size >= n_cpgs, a single chunk is yielded."""
        reader = ChunkedCSVReader(tmp_beta_csv_named_header)
        chunks = list(reader.iter_column_chunks(chunk_size=100))
        assert len(chunks) == 1

    def test_multiple_chunks(self, tmp_beta_csv_named_header: Path) -> None:
        """With 10 CpGs and chunk_size=3, expect ceil(10/3) = 4 chunks."""
        reader = ChunkedCSVReader(tmp_beta_csv_named_header)
        chunks = list(reader.iter_column_chunks(chunk_size=3))
        assert len(chunks) == 4

    def test_each_chunk_has_sample_id(
        self, tmp_beta_csv_named_header: Path
    ) -> None:
        reader = ChunkedCSVReader(tmp_beta_csv_named_header)
        for table in reader.iter_column_chunks(chunk_size=3):
            assert "sample_id" in table.column_names

    def test_chunk_column_counts(
        self, tmp_beta_csv_named_header: Path
    ) -> None:
        """Each chunk should have sample_id + chunk_size CpG columns (or fewer for last)."""
        reader = ChunkedCSVReader(tmp_beta_csv_named_header)
        chunks = list(reader.iter_column_chunks(chunk_size=4))
        # 10 CpGs / 4 = 3 chunks: 4+1, 4+1, 2+1 columns
        assert chunks[0].num_columns == 5  # sample_id + 4 cpgs
        assert chunks[1].num_columns == 5
        assert chunks[2].num_columns == 3  # sample_id + 2 cpgs

    def test_all_rows_present_in_each_chunk(
        self, tmp_beta_csv_named_header: Path
    ) -> None:
        reader = ChunkedCSVReader(tmp_beta_csv_named_header)
        for table in reader.iter_column_chunks(chunk_size=5):
            assert table.num_rows == 5  # 5 samples

    def test_progress_callback_called(
        self, tmp_beta_csv_named_header: Path
    ) -> None:
        reader = ChunkedCSVReader(tmp_beta_csv_named_header)
        calls: list[tuple[int, int]] = []
        list(
            reader.iter_column_chunks(
                chunk_size=5,
                progress_callback=lambda done, total: calls.append(
                    (done, total)
                ),
            )
        )
        assert len(calls) == 2  # ceil(10/5) = 2 chunks
        assert calls[-1][0] == calls[-1][1]  # last call: done == total

    def test_empty_strings_become_null(
        self, tmp_beta_csv_named_header: Path
    ) -> None:
        """Empty string values in the CSV should be read as null."""
        reader = ChunkedCSVReader(tmp_beta_csv_named_header)
        tables = list(reader.iter_column_chunks(chunk_size=100))
        table = tables[0]
        # Check that at least some nulls exist (we set ~15% missingness)
        total_nulls = sum(
            table.column(name).null_count
            for name in table.column_names
            if name != "sample_id"
        )
        assert total_nulls > 0


# =========================================================================
# iter_column_chunks (module-level convenience)
# =========================================================================


class TestIterColumnChunksConvenience:
    """Tests for the module-level ``iter_column_chunks`` function."""

    def test_yields_tables(self, tmp_beta_csv_named_header: Path) -> None:
        tables = list(
            iter_column_chunks(tmp_beta_csv_named_header, chunk_size=5)
        )
        assert len(tables) == 2
        for t in tables:
            assert isinstance(t, pa.Table)


# =========================================================================
# iter_row_chunks
# =========================================================================


class TestIterRowChunks:
    """Tests for row-wise iteration.

    Uses ``tmp_beta_csv`` (empty first header cell) because
    ``iter_row_chunks`` renames the column after reading each batch.
    """

    def test_single_chunk_all_rows(self, tmp_beta_csv: Path) -> None:
        reader = ChunkedCSVReader(tmp_beta_csv)
        chunks = list(reader.iter_row_chunks(chunk_size=100))
        total_rows = sum(t.num_rows for t in chunks)
        assert total_rows == 5

    def test_multiple_row_chunks(self, tmp_beta_csv: Path) -> None:
        reader = ChunkedCSVReader(tmp_beta_csv)
        chunks = list(reader.iter_row_chunks(chunk_size=2))
        total_rows = sum(t.num_rows for t in chunks)
        assert total_rows == 5

    def test_all_columns_present(self, tmp_beta_csv: Path) -> None:
        reader = ChunkedCSVReader(tmp_beta_csv)
        for table in reader.iter_row_chunks(chunk_size=3):
            # Should have sample_id + 10 CpG columns = 11
            assert table.num_columns == 11


# =========================================================================
# stream_to_parquet
# =========================================================================


class TestStreamToParquet:
    """Tests for CSV-to-Parquet streaming conversion.

    Uses the named-header fixture since ``stream_to_parquet`` delegates
    to ``iter_column_chunks``.
    """

    def test_produces_parquet_file(
        self, tmp_beta_csv_named_header: Path, tmp_path: Path
    ) -> None:
        reader = ChunkedCSVReader(tmp_beta_csv_named_header)
        out_path = tmp_path / "output.parquet"
        # Use a single chunk to avoid schema mismatch across row groups
        # (each chunk has different CpG columns; multi-chunk Parquet
        # writing is a known limitation noted in the source TODO).
        result = reader.stream_to_parquet(out_path, chunk_size=100)
        assert result.exists()
        assert result.suffix == ".parquet"

    def test_parquet_has_sample_id_column(
        self, tmp_beta_csv_named_header: Path, tmp_path: Path
    ) -> None:
        reader = ChunkedCSVReader(tmp_beta_csv_named_header)
        out_path = tmp_path / "output.parquet"
        reader.stream_to_parquet(out_path, chunk_size=100)

        import pyarrow.parquet as pq

        pf = pq.ParquetFile(out_path)
        assert "sample_id" in pf.schema_arrow.names
