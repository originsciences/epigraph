"""Tests for genome_coords module: CpG ID parsing, sorting, and interval operations."""

from __future__ import annotations

import pandas as pd
import pytest

from epigraph.common.genome_coords import (
    CHROMOSOME_ORDER,
    find_overlapping_genes,
    make_cpg_id,
    overlaps,
    parse_cpg_id,
    sort_cpg_ids,
)

# =========================================================================
# parse_cpg_id
# =========================================================================


class TestParseCpgId:
    """Tests for ``parse_cpg_id``."""

    @pytest.mark.parametrize(
        ("cpg_id", "expected_chrom", "expected_pos"),
        [
            ("chr1_10469", "chr1", 10469),
            ("chr22_100", "chr22", 100),
            ("chrX_500", "chrX", 500),
            ("chrY_42", "chrY", 42),
            ("chrM_100", "chrM", 100),
            ("chr1_1", "chr1", 1),
        ],
        ids=[
            "chr1-standard",
            "chr22-boundary",
            "chrX",
            "chrY",
            "chrM",
            "min-position",
        ],
    )
    def test_valid_inputs(
        self, cpg_id: str, expected_chrom: str, expected_pos: int
    ) -> None:
        chrom, pos = parse_cpg_id(cpg_id)
        assert chrom == expected_chrom
        assert pos == expected_pos

    @pytest.mark.parametrize(
        "cpg_id",
        [
            "chr0_100",       # chromosome 0 invalid
            "chr23_100",      # chromosome 23 invalid
            "1_100",          # missing chr prefix
            "chr1100",        # missing underscore
            "chr1_",          # missing position
            "chr1_-5",        # negative position
            "",               # empty string
            "chrW_100",       # unknown chromosome letter
            "chr1_10_20",     # extra underscore
        ],
        ids=[
            "chr0",
            "chr23",
            "no-prefix",
            "no-underscore",
            "no-position",
            "negative-pos",
            "empty",
            "unknown-chrom",
            "extra-underscore",
        ],
    )
    def test_invalid_inputs_raise(self, cpg_id: str) -> None:
        with pytest.raises(ValueError, match="Invalid CpG identifier"):
            parse_cpg_id(cpg_id)


# =========================================================================
# make_cpg_id
# =========================================================================


class TestMakeCpgId:
    """Tests for ``make_cpg_id``."""

    def test_basic_construction(self) -> None:
        assert make_cpg_id("chr1", 10469) == "chr1_10469"

    def test_round_trip_with_parse(self) -> None:
        original = "chrX_999"
        chrom, pos = parse_cpg_id(original)
        reconstructed = make_cpg_id(chrom, pos)
        assert reconstructed == original

    @pytest.mark.parametrize(
        "chrom",
        ["chr1", "chr22", "chrX", "chrY", "chrM"],
    )
    def test_all_valid_chromosomes(self, chrom: str) -> None:
        result = make_cpg_id(chrom, 1)
        assert result == f"{chrom}_1"

    def test_invalid_chromosome_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown chromosome"):
            make_cpg_id("chr99", 100)

    def test_zero_position_raises(self) -> None:
        with pytest.raises(ValueError, match="Position must be >= 1"):
            make_cpg_id("chr1", 0)

    def test_negative_position_raises(self) -> None:
        with pytest.raises(ValueError, match="Position must be >= 1"):
            make_cpg_id("chr1", -10)


# =========================================================================
# sort_cpg_ids
# =========================================================================


class TestSortCpgIds:
    """Tests for ``sort_cpg_ids``."""

    def test_chromosome_then_position_order(self) -> None:
        unsorted = ["chr2_300", "chr1_200", "chr1_100", "chrX_50"]
        result = sort_cpg_ids(unsorted)
        assert result == ["chr1_100", "chr1_200", "chr2_300", "chrX_50"]

    def test_same_chromosome_sorted_by_position(self) -> None:
        unsorted = ["chr1_999", "chr1_1", "chr1_500"]
        result = sort_cpg_ids(unsorted)
        assert result == ["chr1_1", "chr1_500", "chr1_999"]

    def test_all_chromosomes_sorted(self) -> None:
        ids = [f"chr{c}_1" for c in ["X", "1", "22", "M", "Y", "2"]]
        result = sort_cpg_ids(ids)
        expected_chroms = ["chr1", "chr2", "chr22", "chrX", "chrY", "chrM"]
        assert [r.split("_")[0] for r in result] == expected_chroms

    def test_invalid_ids_pushed_to_end(self) -> None:
        ids = ["chr1_100", "bad_id", "chr1_50"]
        result = sort_cpg_ids(ids)
        assert result == ["chr1_50", "chr1_100", "bad_id"]

    def test_empty_list(self) -> None:
        assert sort_cpg_ids([]) == []


# =========================================================================
# overlaps
# =========================================================================


class TestOverlaps:
    """Tests for ``overlaps`` (half-open interval check)."""

    def test_overlapping_intervals(self) -> None:
        assert overlaps(0, 10, 5, 15) is True

    def test_identical_intervals(self) -> None:
        assert overlaps(5, 10, 5, 10) is True

    def test_contained_interval(self) -> None:
        assert overlaps(0, 100, 10, 20) is True

    def test_non_overlapping_intervals(self) -> None:
        assert overlaps(0, 5, 10, 15) is False

    def test_edge_touching_is_not_overlap(self) -> None:
        # Half-open [0,5) and [5,10) share no position
        assert overlaps(0, 5, 5, 10) is False

    def test_reversed_non_overlap(self) -> None:
        assert overlaps(10, 15, 0, 5) is False

    def test_single_base_overlap(self) -> None:
        # [5,6) and [5,10) share position 5
        assert overlaps(5, 6, 5, 10) is True

    def test_adjacent_single_base_intervals(self) -> None:
        # [5,6) and [6,7) do not overlap (half-open)
        assert overlaps(5, 6, 6, 7) is False


# =========================================================================
# find_overlapping_genes
# =========================================================================


class TestFindOverlappingGenes:
    """Tests for ``find_overlapping_genes``."""

    def test_cpg_in_gene_body(self, sample_genes_pandas_df: pd.DataFrame) -> None:
        # Position 300 is inside GENE_A [100, 600)
        result = find_overlapping_genes("chr1", 300, sample_genes_pandas_df)
        assert len(result) == 1
        assert result.iloc[0]["gene_symbol"] == "GENE_A"

    def test_cpg_at_gene_start(self, sample_genes_pandas_df: pd.DataFrame) -> None:
        # Position 100: inside [100, 600) because start < 101 and end > 100
        result = find_overlapping_genes("chr1", 100, sample_genes_pandas_df)
        assert len(result) == 1

    def test_cpg_at_gene_end_exclusive(
        self, sample_genes_pandas_df: pd.DataFrame
    ) -> None:
        # Position 600: gene end is 600, so 600 < 601 is True and
        # 600 > 600 is False => no overlap (interval is [start, end))
        result = find_overlapping_genes("chr1", 600, sample_genes_pandas_df)
        assert len(result) == 0

    def test_cpg_intergenic(self, sample_genes_pandas_df: pd.DataFrame) -> None:
        # Position 50000 on chr1 -- beyond any gene
        result = find_overlapping_genes("chr1", 50000, sample_genes_pandas_df)
        assert len(result) == 0

    def test_cpg_on_different_chromosome(
        self, sample_genes_pandas_df: pd.DataFrame
    ) -> None:
        # Position 500 on chr2 is inside GENE_C [200, 1000)
        result = find_overlapping_genes("chr2", 500, sample_genes_pandas_df)
        assert len(result) == 1
        assert result.iloc[0]["gene_symbol"] == "GENE_C"

    def test_cpg_no_genes_on_chromosome(
        self, sample_genes_pandas_df: pd.DataFrame
    ) -> None:
        result = find_overlapping_genes("chr3", 100, sample_genes_pandas_df)
        assert len(result) == 0

    def test_missing_column_raises(self) -> None:
        bad_df = pd.DataFrame({"chrom": ["chr1"], "start": [100]})
        with pytest.raises(KeyError, match="Missing required columns"):
            find_overlapping_genes("chr1", 150, bad_df)

    def test_custom_column_names(self) -> None:
        df = pd.DataFrame(
            {
                "chr": ["chr1"],
                "begin": [100],
                "finish": [500],
            }
        )
        result = find_overlapping_genes(
            "chr1",
            200,
            df,
            chrom_col="chr",
            start_col="begin",
            end_col="finish",
        )
        assert len(result) == 1


# =========================================================================
# CHROMOSOME_ORDER sanity check
# =========================================================================


class TestChromosomeOrder:
    """Verify chromosome ordering constant."""

    def test_chr1_is_first(self) -> None:
        assert CHROMOSOME_ORDER["chr1"] == 0

    def test_chr_m_is_last(self) -> None:
        assert CHROMOSOME_ORDER["chrM"] == 24

    def test_all_25_chromosomes_present(self) -> None:
        # 1-22 + X + Y + M = 25
        assert len(CHROMOSOME_ORDER) == 25
