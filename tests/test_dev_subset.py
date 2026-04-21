"""Tests for dev subset creation logic.

Exercises CpG stratification, sample selection, subset extraction,
and ID-list file writing without requiring the full 22 GB dataset.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from epigraph.db_build.create_dev_subset import (
    extract_subset,
    read_header,
    read_sample_ids,
    select_cpgs_stratified,
    select_samples_stratified,
    write_id_list,
)

from .conftest import MOCK_CPG_IDS, MOCK_SAMPLE_IDS

# =========================================================================
# read_header / read_sample_ids
# =========================================================================


class TestHeaderAndSampleReading:
    """Tests for memory-safe header and sample ID reading."""

    def test_read_header_returns_cpg_columns(self, tmp_beta_csv: Path) -> None:
        cpg_cols = read_header(tmp_beta_csv)
        assert len(cpg_cols) == 10
        assert cpg_cols[0] == "chr1_100"
        assert cpg_cols[-1] == "chrY_50"

    def test_read_header_excludes_first_cell(self, tmp_beta_csv: Path) -> None:
        """The empty first header cell should NOT be in the returned list."""
        cpg_cols = read_header(tmp_beta_csv)
        assert "" not in cpg_cols
        assert "sample_id" not in cpg_cols

    def test_read_sample_ids(self, tmp_beta_csv: Path) -> None:
        sample_ids = read_sample_ids(tmp_beta_csv)
        assert len(sample_ids) == 5
        assert set(sample_ids) == set(MOCK_SAMPLE_IDS)


# =========================================================================
# select_cpgs_stratified
# =========================================================================


class TestSelectCpgsStratified:
    """Tests for stratified CpG selection across chromosomes."""

    def test_returns_requested_count(self) -> None:
        selected = select_cpgs_stratified(MOCK_CPG_IDS, n_cpgs=5, seed=42)
        assert len(selected) == 5

    def test_result_is_sorted(self) -> None:
        selected = select_cpgs_stratified(MOCK_CPG_IDS, n_cpgs=5, seed=42)
        assert selected == sorted(selected)

    def test_covers_multiple_chromosomes(self) -> None:
        selected = select_cpgs_stratified(MOCK_CPG_IDS, n_cpgs=8, seed=42)
        chroms = {cpg.split("_")[0] for cpg in selected}
        assert len(chroms) > 1

    def test_at_least_one_per_chromosome(self) -> None:
        """With enough budget, every chromosome gets at least one CpG."""
        selected = select_cpgs_stratified(MOCK_CPG_IDS, n_cpgs=10, seed=42)
        chroms = {cpg.split("_")[0] for cpg in selected}
        # Our mock has chr1, chr2, chr3, chrX, chrY
        assert len(chroms) == 5

    def test_chromosome_filter(self) -> None:
        """When filtering to a single chromosome, only that chromosome's CpGs
        should appear in the primary selection.  If n_cpgs exceeds available
        CpGs on the chromosome, padding may add CpGs from other chromosomes.
        """
        # Request only 3 (chr1 has 3 CpGs in mock data)
        selected = select_cpgs_stratified(
            MOCK_CPG_IDS, n_cpgs=3, seed=42, chromosomes=["chr1"]
        )
        for cpg in selected:
            chrom = cpg.split("_")[0]
            assert chrom == "chr1"

    def test_reproducible_with_same_seed(self) -> None:
        a = select_cpgs_stratified(MOCK_CPG_IDS, n_cpgs=5, seed=99)
        b = select_cpgs_stratified(MOCK_CPG_IDS, n_cpgs=5, seed=99)
        assert a == b

    def test_empty_input(self) -> None:
        result = select_cpgs_stratified([], n_cpgs=5, seed=42)
        assert result == []

    def test_request_more_than_available(self) -> None:
        selected = select_cpgs_stratified(MOCK_CPG_IDS, n_cpgs=100, seed=42)
        assert len(selected) <= len(MOCK_CPG_IDS)


# =========================================================================
# select_samples_stratified
# =========================================================================


class TestSelectSamplesStratified:
    """Tests for stratified sample selection."""

    def test_random_fallback_without_metadata(self) -> None:
        selected = select_samples_stratified(
            MOCK_SAMPLE_IDS, n_samples=3, seed=42
        )
        assert len(selected) == 3
        assert all(s in MOCK_SAMPLE_IDS for s in selected)

    def test_with_clinical_metadata(self) -> None:
        clin = pl.DataFrame(
            {
                "barcode": ["SAMPLE_0001", "SAMPLE_0002", "SAMPLE_0003", "SAMPLE_0004", "SAMPLE_0005"],
                "clinical_category": ["CRC", "CRC", "Control", "Control", "polyps"],
            }
        )
        selected = select_samples_stratified(
            MOCK_SAMPLE_IDS,
            n_samples=4,
            clinical_metadata=clin,
            min_per_category=1,
            seed=42,
        )
        assert len(selected) == 4

    def test_request_more_than_available(self) -> None:
        selected = select_samples_stratified(
            MOCK_SAMPLE_IDS, n_samples=100, seed=42
        )
        assert len(selected) == len(MOCK_SAMPLE_IDS)


# =========================================================================
# extract_subset
# =========================================================================


class TestExtractSubset:
    """Tests for extracting a subset from the beta matrix CSV."""

    def test_produces_parquet_output(
        self, tmp_beta_csv: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "subset.parquet"
        extract_subset(
            tmp_beta_csv,
            selected_cpgs=["chr1_100", "chr2_300"],
            selected_samples=["SAMPLE_0001", "SAMPLE_0002"],
            output_path=out,
        )
        assert out.exists()

    def test_output_dimensions(self, tmp_beta_csv: Path, tmp_path: Path) -> None:
        out = tmp_path / "subset.parquet"
        extract_subset(
            tmp_beta_csv,
            selected_cpgs=["chr1_100", "chr2_300"],
            selected_samples=["SAMPLE_0001", "SAMPLE_0002"],
            output_path=out,
        )
        df = pl.read_parquet(out)
        assert df.height == 2  # 2 samples
        # sample_id + 2 CpG columns = 3
        assert df.width == 3
        assert "sample_id" in df.columns

    def test_no_matching_samples_no_output(
        self, tmp_beta_csv: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "empty_subset.parquet"
        extract_subset(
            tmp_beta_csv,
            selected_cpgs=["chr1_100"],
            selected_samples=["NONEXISTENT"],
            output_path=out,
        )
        # The function returns without writing when no samples match
        assert not out.exists()


# =========================================================================
# write_id_list
# =========================================================================


class TestWriteIdList:
    """Tests for writing ID lists to text files."""

    def test_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "ids.txt"
        write_id_list(["id1", "id2", "id3"], out)
        assert out.exists()

    def test_one_id_per_line(self, tmp_path: Path) -> None:
        ids = ["SAMPLE_0001", "SAMPLE_0002", "SAMPLE_0003"]
        out = tmp_path / "ids.txt"
        write_id_list(ids, out)
        lines = out.read_text().strip().split("\n")
        assert lines == ids

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "dir" / "ids.txt"
        write_id_list(["a"], out)
        assert out.exists()
