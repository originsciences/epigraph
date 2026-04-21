"""Tests for CpG-to-gene mapping logic.

Exercises the ``GeneInterval``, ``ChromosomeIndex``, ``build_gene_index``,
and ``map_cpgs_to_genes`` functions with controlled gene coordinates.
"""

from __future__ import annotations

import polars as pl
import pytest

from epigraph.db_build.map_cpg_to_genes import (
    PROMOTER_DOWNSTREAM,
    PROMOTER_UPSTREAM,
    ChromosomeIndex,
    GeneInterval,
    build_gene_index,
    map_cpgs_to_genes,
)

# =========================================================================
# GeneInterval
# =========================================================================


class TestGeneInterval:
    """Tests for ``GeneInterval`` dataclass and promoter boundary computation."""

    def test_plus_strand_tss(self) -> None:
        g = GeneInterval(
            gene_id="G1",
            gene_symbol="SYM",
            chrom="chr1",
            start=10000,
            end=20000,
            strand="+",
        )
        assert g.tss == 10000
        assert g.promoter_start == 10000 - PROMOTER_UPSTREAM  # 8500
        assert g.promoter_end == 10000 + PROMOTER_DOWNSTREAM  # 10500

    def test_minus_strand_tss(self) -> None:
        g = GeneInterval(
            gene_id="G2",
            gene_symbol="SYM2",
            chrom="chr1",
            start=10000,
            end=20000,
            strand="-",
        )
        assert g.tss == 20000  # TSS at gene end for minus strand
        assert g.promoter_start == 20000 - PROMOTER_DOWNSTREAM  # 19500
        assert g.promoter_end == 20000 + PROMOTER_UPSTREAM  # 21500

    def test_promoter_start_clipped_to_1(self) -> None:
        """Promoter start should not go below 1."""
        g = GeneInterval(
            gene_id="G3",
            gene_symbol="SYM3",
            chrom="chr1",
            start=100,
            end=500,
            strand="+",
        )
        assert g.promoter_start == 1  # max(1, 100 - 1500)


# =========================================================================
# ChromosomeIndex
# =========================================================================


class TestChromosomeIndex:
    """Tests for ``ChromosomeIndex`` binary-search based overlap finding."""

    @pytest.fixture()
    def chr1_index(self) -> ChromosomeIndex:
        """Build an index with two genes on chr1."""
        idx = ChromosomeIndex(chrom="chr1")
        idx.genes = [
            GeneInterval("G1", "A", "chr1", 1000, 5000, "+"),
            GeneInterval("G2", "B", "chr1", 10000, 20000, "-"),
        ]
        idx.build()
        return idx

    def test_build_sorts_by_start(self, chr1_index: ChromosomeIndex) -> None:
        starts = [g.start for g in chr1_index.genes]
        assert starts == sorted(starts)

    def test_find_in_gene_body(self, chr1_index: ChromosomeIndex) -> None:
        results = chr1_index.find_overlapping(3000)
        gene_ids = [g.gene_id for g, _ in results]
        assert "G1" in gene_ids

    def test_find_in_promoter(self, chr1_index: ChromosomeIndex) -> None:
        """Position in G2's promoter region (minus strand: TSS=20000, promoter extends to 21500)."""
        results = chr1_index.find_overlapping(21000)
        assert len(results) >= 1
        overlap_types = [ot for _, ot in results]
        assert "promoter" in overlap_types

    def test_intergenic_returns_empty(self, chr1_index: ChromosomeIndex) -> None:
        # Position far from any gene or promoter
        results = chr1_index.find_overlapping(500000)
        assert results == []

    def test_gene_body_vs_promoter_priority(self) -> None:
        """Position in promoter but NOT in gene body should be classified as promoter."""
        idx = ChromosomeIndex(chrom="chr1")
        # Gene on + strand: TSS=10000, promoter [8500, 10500), gene body [10000, 20000]
        idx.genes = [GeneInterval("G1", "A", "chr1", 10000, 20000, "+")]
        idx.build()

        # Position 9000 is in promoter region [8500, 10500) but before gene start
        results = idx.find_overlapping(9000)
        assert len(results) == 1
        assert results[0][1] == "promoter"


# =========================================================================
# build_gene_index
# =========================================================================


class TestBuildGeneIndex:
    """Tests for ``build_gene_index``."""

    def test_builds_index_from_dataframe(self, sample_genes_df: pl.DataFrame) -> None:
        index = build_gene_index(sample_genes_df)
        assert isinstance(index, dict)
        assert "chr1" in index
        assert "chr2" in index
        assert "chrX" in index

    def test_genes_per_chromosome(self, sample_genes_df: pl.DataFrame) -> None:
        index = build_gene_index(sample_genes_df)
        assert len(index["chr1"].genes) == 2  # GENE_A, GENE_B
        assert len(index["chr2"].genes) == 1  # GENE_C
        assert len(index["chrX"].genes) == 1  # GENE_D

    def test_missing_chromosome_not_in_index(
        self, sample_genes_df: pl.DataFrame
    ) -> None:
        index = build_gene_index(sample_genes_df)
        assert "chr3" not in index


# =========================================================================
# map_cpgs_to_genes
# =========================================================================


class TestMapCpgsToGenes:
    """Tests for the full CpG-to-gene mapping pipeline."""

    @pytest.fixture()
    def gene_index(self) -> dict[str, ChromosomeIndex]:
        genes_df = pl.DataFrame(
            {
                "gene_id": ["ENSG001", "ENSG002"],
                "gene_symbol": ["GENE_X", "GENE_Y"],
                "chrom": ["chr1", "chr2"],
                "start": [1000, 5000],
                "end": [5000, 10000],
                "strand": ["+", "-"],
            }
        )
        return build_gene_index(genes_df)

    def test_cpg_in_gene_body(
        self, gene_index: dict[str, ChromosomeIndex]
    ) -> None:
        result = map_cpgs_to_genes(["chr1_3000"], gene_index)
        assert len(result) == 1
        assert result["overlap_type"][0] == "gene_body"
        assert result["gene_id"][0] == "ENSG001"

    def test_cpg_in_promoter(
        self, gene_index: dict[str, ChromosomeIndex]
    ) -> None:
        """CpG upstream of + strand gene TSS (1000) within promoter window."""
        # Promoter for ENSG001 (+strand): [max(1, 1000-1500), 1000+500) = [1, 1500)
        # But the promoter check is <= and >=, so pos in [1, 1500]
        result = map_cpgs_to_genes(["chr1_200"], gene_index)
        assert len(result) == 1
        assert result["overlap_type"][0] == "promoter"

    def test_intergenic_excluded_by_default(
        self, gene_index: dict[str, ChromosomeIndex]
    ) -> None:
        result = map_cpgs_to_genes(
            ["chr1_999999"], gene_index, report_intergenic=False
        )
        assert len(result) == 0

    def test_intergenic_included_when_requested(
        self, gene_index: dict[str, ChromosomeIndex]
    ) -> None:
        result = map_cpgs_to_genes(
            ["chr1_999999"], gene_index, report_intergenic=True
        )
        assert len(result) == 1
        assert result["overlap_type"][0] == "intergenic"

    def test_cpg_on_unmapped_chromosome(
        self, gene_index: dict[str, ChromosomeIndex]
    ) -> None:
        result = map_cpgs_to_genes(
            ["chr3_100"], gene_index, report_intergenic=True
        )
        assert len(result) == 1
        assert result["overlap_type"][0] == "intergenic"

    def test_multiple_cpgs_mixed(
        self, gene_index: dict[str, ChromosomeIndex]
    ) -> None:
        cpgs = ["chr1_3000", "chr2_7000", "chr1_999999"]
        result = map_cpgs_to_genes(cpgs, gene_index, report_intergenic=True)
        types = set(result["overlap_type"].to_list())
        assert "gene_body" in types
        assert "intergenic" in types

    def test_invalid_cpg_id_skipped(
        self, gene_index: dict[str, ChromosomeIndex]
    ) -> None:
        result = map_cpgs_to_genes(["bad_id", "chr1_3000"], gene_index)
        # bad_id skipped, chr1_3000 mapped
        assert len(result) == 1

    def test_output_columns(
        self, gene_index: dict[str, ChromosomeIndex]
    ) -> None:
        result = map_cpgs_to_genes(["chr1_3000"], gene_index)
        expected_cols = {
            "cpg_id",
            "chromosome",
            "position",
            "gene_id",
            "gene_symbol",
            "overlap_type",
        }
        assert set(result.columns) == expected_cols


# =========================================================================
# Promoter definition validation
# =========================================================================


class TestPromoterDefinition:
    """Verify TSS +/- 1500/500 bp promoter boundaries."""

    def test_promoter_constants(self) -> None:
        assert PROMOTER_UPSTREAM == 1500
        assert PROMOTER_DOWNSTREAM == 500

    def test_plus_strand_promoter_boundaries(self) -> None:
        """For a + strand gene at position 10000, promoter should span [8500, 10500]."""
        g = GeneInterval("G", "S", "chr1", 10000, 20000, "+")
        assert g.promoter_start == 8500
        assert g.promoter_end == 10500

    def test_minus_strand_promoter_boundaries(self) -> None:
        """For a - strand gene ending at 20000, promoter should span [19500, 21500]."""
        g = GeneInterval("G", "S", "chr1", 10000, 20000, "-")
        assert g.promoter_start == 19500
        assert g.promoter_end == 21500

    def test_cpg_just_outside_promoter_is_not_promoter(self) -> None:
        """A CpG 1 bp outside the promoter boundary should not be classified as promoter."""
        idx = ChromosomeIndex(chrom="chr1")
        # + strand gene: TSS=10000, promoter [8500, 10500]
        idx.genes = [GeneInterval("G1", "A", "chr1", 10000, 20000, "+")]
        idx.build()

        # Position 8499 is 1 bp before promoter_start (8500)
        results = idx.find_overlapping(8499)
        overlap_types = [ot for _, ot in results]
        assert "promoter" not in overlap_types
