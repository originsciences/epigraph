"""Tests for the convert_beta_to_parquet module.

Validates per-chromosome partitioning, Parquet output correctness, and
sample ID preservation using a small synthetic beta matrix CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from epigraph.db_build.convert_beta_to_parquet import (
    _build_chrom_index,
    _read_header,
    convert_single_pass,
)

# ---------------------------------------------------------------------------
# Constants for the fixture
# ---------------------------------------------------------------------------

_SAMPLE_IDS = ["S001", "S002", "S003", "S004", "S005"]

# 20 CpGs spread across 3 chromosomes
_CPG_IDS = [
    # chr1 -- 8 CpGs
    "chr1_100",
    "chr1_200",
    "chr1_300",
    "chr1_400",
    "chr1_500",
    "chr1_600",
    "chr1_700",
    "chr1_800",
    # chr2 -- 7 CpGs
    "chr2_150",
    "chr2_250",
    "chr2_350",
    "chr2_450",
    "chr2_550",
    "chr2_650",
    "chr2_750",
    # chr3 -- 5 CpGs
    "chr3_110",
    "chr3_220",
    "chr3_330",
    "chr3_440",
    "chr3_550",
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def beta_csv(tmp_path: Path) -> Path:
    """Create a small CSV file (5 samples x 20 CpGs across 3 chromosomes).

    Matches the beta matrix format: empty first header cell, CpG IDs as
    ``chr{N}_{position}``, and float values in [0, 1].
    """
    rng = np.random.default_rng(99)
    csv_path = tmp_path / "beta_matrix.csv"

    header = "," + ",".join(_CPG_IDS)
    rows: list[str] = [header]
    for sample_id in _SAMPLE_IDS:
        values: list[str] = []
        for _ in _CPG_IDS:
            if rng.random() < 0.1:
                values.append("")  # ~10% missing
            else:
                values.append(f"{rng.random():.6f}")
        rows.append(sample_id + "," + ",".join(values))

    csv_path.write_text("\n".join(rows) + "\n")
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_partition_cpgs_by_chromosome(beta_csv: Path) -> None:
    """Verify _build_chrom_index groups CpGs correctly by chromosome."""
    header = _read_header(beta_csv)
    chrom_index = _build_chrom_index(header)

    assert set(chrom_index.keys()) == {"chr1", "chr2", "chr3"}
    assert len(chrom_index["chr1"]) == 8
    assert len(chrom_index["chr2"]) == 7
    assert len(chrom_index["chr3"]) == 5

    # Each entry should be (col_idx, cpg_name) and col_idx > 0
    for _col_idx, cpg_name in chrom_index["chr1"]:
        assert cpg_name.startswith("chr1_")


def test_convert_single_pass(beta_csv: Path, tmp_path: Path) -> None:
    """Verify per-chromosome Parquet output: file existence, row counts, CpG column names."""
    output_dir = tmp_path / "output"
    results = convert_single_pass(beta_csv, output_dir)

    assert set(results.keys()) == {"chr1", "chr2", "chr3"}
    assert results["chr1"] == 8
    assert results["chr2"] == 7
    assert results["chr3"] == 5

    for chrom, expected_cpgs in [("chr1", 8), ("chr2", 7), ("chr3", 5)]:
        parquet_path = output_dir / f"beta_{chrom}.parquet"
        assert parquet_path.exists(), f"Missing Parquet for {chrom}"

        table = pq.read_table(parquet_path)
        # Rows = number of samples
        assert table.num_rows == len(_SAMPLE_IDS)
        # Columns = sample_id + CpG columns
        assert table.num_columns == expected_cpgs + 1

        cpg_cols = [c for c in table.column_names if c != "sample_id"]
        assert len(cpg_cols) == expected_cpgs
        for col_name in cpg_cols:
            assert col_name.startswith(f"{chrom}_")


def test_sample_ids_preserved(beta_csv: Path, tmp_path: Path) -> None:
    """Verify sample IDs in output match input."""
    output_dir = tmp_path / "output"
    convert_single_pass(beta_csv, output_dir)

    # Check any one chromosome file -- sample_ids should be the same in all
    table = pq.read_table(output_dir / "beta_chr1.parquet")
    output_sample_ids = table.column("sample_id").to_pylist()
    assert output_sample_ids == _SAMPLE_IDS


def test_batched_flush_preserves_sample_order(tmp_path: Path) -> None:
    """Force multiple batch flushes and confirm sample order + values survive.

    The ``batch_size=2`` invocation against 5 input samples exercises the
    "flush then refill then final partial flush" path of
    ``convert_single_pass`` that a single-batch run never hits.
    """
    cpg_ids = ["chr1_10", "chr1_20", "chr2_15"]
    samples = [
        ("S01", ["0.10", "0.20", "0.30"]),
        ("S02", ["0.11", "0.21", "0.31"]),
        ("S03", ["0.12", "0.22", "0.32"]),
        ("S04", ["0.13", "0.23", "0.33"]),
        ("S05", ["0.14", "0.24", "0.34"]),
    ]
    csv_path = tmp_path / "beta.csv"
    lines = ["," + ",".join(cpg_ids)]
    for sample_id, vals in samples:
        lines.append(f"{sample_id}," + ",".join(vals))
    csv_path.write_text("\n".join(lines) + "\n")

    output_dir = tmp_path / "output"
    convert_single_pass(csv_path, output_dir, batch_size=2)

    chr1 = pq.read_table(output_dir / "beta_chr1.parquet")
    chr2 = pq.read_table(output_dir / "beta_chr2.parquet")
    assert chr1.column("sample_id").to_pylist() == [s for s, _ in samples]
    assert chr2.column("sample_id").to_pylist() == [s for s, _ in samples]

    # Values in correct (sample, CpG) positions
    expected = np.array([[float(v) for v in row] for _, row in samples], dtype=np.float32)
    chr1_vals = np.column_stack(
        [chr1.column(c).to_numpy() for c in ["chr1_10", "chr1_20"]]
    )
    chr2_vals = chr2.column("chr2_15").to_numpy()
    np.testing.assert_allclose(chr1_vals, expected[:, :2], atol=1e-6)
    np.testing.assert_allclose(chr2_vals, expected[:, 2], atol=1e-6)


def test_values_round_trip(tmp_path: Path) -> None:
    """Every non-empty cell must land in the correct sample × CpG slot.

    Pins the values that get written to Parquet, so a refactor of
    ``convert_single_pass`` that changes batching, row-group size, or the
    in-memory buffer layout still has to preserve the data.
    """
    # Handcrafted matrix: 3 samples × 6 CpGs (2 chromosomes).  A few cells
    # are deliberately missing (empty / NA).
    cpg_ids = ["chr1_10", "chr1_20", "chr1_30", "chr2_15", "chr2_25", "chr2_35"]
    rows = [
        ("S1",   ["0.10",  "0.20",  "0.30",  "0.40",  "",      "0.60"]),
        ("S2",   ["0.11",  "NA",    "0.31",  "0.41",  "0.51",  "0.61"]),
        ("S3",   ["",      "0.22",  "0.32",  "",      "0.52",  "NaN"]),
    ]
    csv_path = tmp_path / "beta.csv"
    lines = ["," + ",".join(cpg_ids)]
    for sample_id, vals in rows:
        lines.append(f"{sample_id}," + ",".join(vals))
    csv_path.write_text("\n".join(lines) + "\n")

    output_dir = tmp_path / "output"
    convert_single_pass(csv_path, output_dir)

    # Expected values per chromosome, indexed [sample_i, cpg_j].
    expected_chr1 = np.array(
        [
            [0.10, 0.20, 0.30],
            [0.11, np.nan, 0.31],
            [np.nan, 0.22, 0.32],
        ],
        dtype=np.float32,
    )
    expected_chr2 = np.array(
        [
            [0.40, np.nan, 0.60],
            [0.41, 0.51, 0.61],
            [np.nan, 0.52, np.nan],
        ],
        dtype=np.float32,
    )

    chr1_table = pq.read_table(output_dir / "beta_chr1.parquet")
    chr2_table = pq.read_table(output_dir / "beta_chr2.parquet")

    assert chr1_table.column("sample_id").to_pylist() == ["S1", "S2", "S3"]
    assert chr2_table.column("sample_id").to_pylist() == ["S1", "S2", "S3"]

    def _col_matrix(table: pa.Table) -> np.ndarray:
        cols = [c for c in table.column_names if c != "sample_id"]
        return np.column_stack([table.column(c).to_numpy(zero_copy_only=False) for c in cols])

    np.testing.assert_array_equal(
        np.isnan(_col_matrix(chr1_table)), np.isnan(expected_chr1)
    )
    np.testing.assert_allclose(
        np.nan_to_num(_col_matrix(chr1_table)),
        np.nan_to_num(expected_chr1),
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        np.isnan(_col_matrix(chr2_table)), np.isnan(expected_chr2)
    )
    np.testing.assert_allclose(
        np.nan_to_num(_col_matrix(chr2_table)),
        np.nan_to_num(expected_chr2),
        atol=1e-6,
    )
