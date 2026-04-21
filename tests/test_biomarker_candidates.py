"""Tests for biomarker candidate ranking in epigraph.analysis.biomarker_candidates."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from epigraph.analysis.biomarker_candidates import (
    export_to_typedb,
    generate_biomarker_report,
    rank_cpg_biomarkers,
    rank_gene_biomarkers,
    rank_pathway_biomarkers,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cpg_diff_df() -> pl.DataFrame:
    """Mock CpG-level differential methylation results."""
    return pl.DataFrame({
        "feature": [f"cg{i:08d}" for i in range(10)],
        "cohens_d": [2.5, -1.8, 0.3, 1.2, -0.9, 3.0, 0.1, -2.0, 1.5, 0.7],
        "q_value": [1e-10, 1e-8, 0.5, 1e-4, 0.01, 1e-12, 0.9, 1e-6, 1e-3, 0.1],
    })


@pytest.fixture()
def gene_diff_df() -> pl.DataFrame:
    """Mock gene-level differential methylation results."""
    return pl.DataFrame({
        "feature": ["TP53", "BRCA1", "EGFR", "MYC", "KRAS"],
        "cohens_d": [2.0, -1.5, 0.8, 1.3, -0.4],
        "q_value": [1e-8, 1e-6, 0.01, 1e-4, 0.3],
    })


@pytest.fixture()
def pathway_enrichment_df() -> pl.DataFrame:
    """Mock pathway enrichment results."""
    return pl.DataFrame({
        "pathway": ["Apoptosis", "Cell Cycle", "DNA Repair", "Metabolism"],
        "p_value": [1e-6, 1e-4, 0.001, 0.05],
        "q_value": [1e-5, 1e-3, 0.01, 0.1],
        "odds_ratio": [5.0, 3.2, 2.1, 1.3],
        "n_overlap": [15, 10, 8, 4],
    })


# ---------------------------------------------------------------------------
# rank_cpg_biomarkers
# ---------------------------------------------------------------------------


class TestRankCpgBiomarkers:
    """Tests for CpG-level biomarker ranking."""

    def test_returns_expected_columns(self, cpg_diff_df: pl.DataFrame) -> None:
        result = rank_cpg_biomarkers(cpg_diff_df, n_top=5)
        expected_cols = {"cpg_id", "effect_size", "q_value", "score"}
        assert expected_cols.issubset(set(result.columns))

    def test_respects_n_top(self, cpg_diff_df: pl.DataFrame) -> None:
        result = rank_cpg_biomarkers(cpg_diff_df, n_top=3)
        assert result.height == 3

    def test_sorted_by_score_descending(self, cpg_diff_df: pl.DataFrame) -> None:
        result = rank_cpg_biomarkers(cpg_diff_df, n_top=10)
        scores = result["score"].to_list()
        assert scores == sorted(scores, reverse=True)

    def test_score_formula(self) -> None:
        """Verify score = |effect_size| * -log10(q_value)."""
        df = pl.DataFrame({
            "feature": ["cg_test"],
            "cohens_d": [2.0],
            "q_value": [0.001],
        })
        result = rank_cpg_biomarkers(df, n_top=1)
        expected_score = 2.0 * 3.0  # |2.0| * -log10(0.001) = 2.0 * 3.0
        assert abs(result["score"].item() - expected_score) < 1e-6

    def test_zero_q_value_clamped(self) -> None:
        """q_value=0 should not produce inf score."""
        df = pl.DataFrame({
            "feature": ["cg_zero"],
            "cohens_d": [1.0],
            "q_value": [0.0],
        })
        result = rank_cpg_biomarkers(df, n_top=1)
        assert np.isfinite(result["score"].item())

    def test_annotation_join(self, cpg_diff_df: pl.DataFrame) -> None:
        """Annotations should be joined when provided."""
        ann = pl.DataFrame({
            "cpg_id": ["cg00000005"],
            "gene_symbol": ["TP53"],
            "region": ["promoter"],
        })
        result = rank_cpg_biomarkers(cpg_diff_df, n_top=10, annotation_df=ann)
        assert "gene_symbol" in result.columns
        assert "region" in result.columns

    def test_n_top_exceeds_input(self, cpg_diff_df: pl.DataFrame) -> None:
        """Requesting more candidates than available should return all."""
        result = rank_cpg_biomarkers(cpg_diff_df, n_top=100)
        assert result.height == cpg_diff_df.height


# ---------------------------------------------------------------------------
# rank_gene_biomarkers
# ---------------------------------------------------------------------------


class TestRankGeneBiomarkers:
    """Tests for gene-level biomarker ranking."""

    def test_returns_expected_columns(self, gene_diff_df: pl.DataFrame) -> None:
        result = rank_gene_biomarkers(gene_diff_df, n_top=3)
        expected_cols = {"gene", "effect_size", "q_value", "diff_score", "enrichment_bonus", "score"}
        assert expected_cols == set(result.columns)

    def test_respects_n_top(self, gene_diff_df: pl.DataFrame) -> None:
        result = rank_gene_biomarkers(gene_diff_df, n_top=2)
        assert result.height == 2

    def test_sorted_descending(self, gene_diff_df: pl.DataFrame) -> None:
        result = rank_gene_biomarkers(gene_diff_df, n_top=5)
        scores = result["score"].to_list()
        assert scores == sorted(scores, reverse=True)

    def test_enrichment_bonus_zero_without_enrichment(self, gene_diff_df: pl.DataFrame) -> None:
        result = rank_gene_biomarkers(gene_diff_df, gene_enrichment=None)
        bonuses = result["enrichment_bonus"].to_list()
        assert all(b == 0.0 for b in bonuses)

    def test_score_equals_diff_score_without_enrichment(self, gene_diff_df: pl.DataFrame) -> None:
        result = rank_gene_biomarkers(gene_diff_df)
        for row in result.iter_rows(named=True):
            assert abs(row["score"] - row["diff_score"]) < 1e-10


# ---------------------------------------------------------------------------
# rank_pathway_biomarkers
# ---------------------------------------------------------------------------


class TestRankPathwayBiomarkers:
    """Tests for pathway-level biomarker ranking."""

    def test_returns_expected_columns(self, pathway_enrichment_df: pl.DataFrame) -> None:
        result = rank_pathway_biomarkers(pathway_enrichment_df, n_top=4)
        assert "pathway" in result.columns
        assert "score" in result.columns
        assert "q_value" in result.columns

    def test_respects_n_top(self, pathway_enrichment_df: pl.DataFrame) -> None:
        result = rank_pathway_biomarkers(pathway_enrichment_df, n_top=2)
        assert result.height == 2

    def test_sorted_descending(self, pathway_enrichment_df: pl.DataFrame) -> None:
        result = rank_pathway_biomarkers(pathway_enrichment_df, n_top=4)
        scores = result["score"].to_list()
        assert scores == sorted(scores, reverse=True)

    def test_odds_ratio_scoring(self, pathway_enrichment_df: pl.DataFrame) -> None:
        """When odds_ratio is available, score uses log2(OR) * -log10(q)."""
        result = rank_pathway_biomarkers(pathway_enrichment_df, n_top=1)
        top = result.row(0, named=True)
        # Apoptosis: OR=5.0, q=1e-5 => log2(5)*5 ~ 2.322*5 = 11.61
        expected = np.log2(5.0) * (-np.log10(1e-5))
        assert abs(top["score"] - expected) < 0.01

    def test_fallback_significance_only(self) -> None:
        """Without effect columns, score = -log10(q_value)."""
        df = pl.DataFrame({
            "pathway": ["PathA", "PathB"],
            "p_value": [0.001, 0.01],
            "q_value": [0.01, 0.1],
        })
        result = rank_pathway_biomarkers(df, n_top=2)
        top = result.row(0, named=True)
        expected = -np.log10(0.01)
        assert abs(top["score"] - expected) < 1e-6

    def test_nes_scoring(self) -> None:
        """When NES column is available, score = |NES| * -log10(q)."""
        df = pl.DataFrame({
            "pathway": ["PathA"],
            "p_value": [0.001],
            "q_value": [0.01],
            "nes": [-2.5],
        })
        result = rank_pathway_biomarkers(df, n_top=1)
        expected = 2.5 * (-np.log10(0.01))
        assert abs(result["score"].item() - expected) < 1e-6


# ---------------------------------------------------------------------------
# generate_biomarker_report
# ---------------------------------------------------------------------------


class TestGenerateBiomarkerReport:
    """Tests for consolidated biomarker report generation."""

    def test_report_has_all_levels(
        self,
        cpg_diff_df: pl.DataFrame,
        gene_diff_df: pl.DataFrame,
        pathway_enrichment_df: pl.DataFrame,
    ) -> None:
        cpg = rank_cpg_biomarkers(cpg_diff_df, n_top=3)
        gene = rank_gene_biomarkers(gene_diff_df, n_top=2)
        pathway = rank_pathway_biomarkers(pathway_enrichment_df, n_top=2)

        report = generate_biomarker_report(cpg, gene, pathway)

        levels = set(report["level"].to_list())
        assert levels == {"cpg", "gene", "pathway"}

    def test_report_columns(
        self,
        cpg_diff_df: pl.DataFrame,
        gene_diff_df: pl.DataFrame,
        pathway_enrichment_df: pl.DataFrame,
    ) -> None:
        cpg = rank_cpg_biomarkers(cpg_diff_df, n_top=2)
        gene = rank_gene_biomarkers(gene_diff_df, n_top=2)
        pathway = rank_pathway_biomarkers(pathway_enrichment_df, n_top=2)

        report = generate_biomarker_report(cpg, gene, pathway)
        expected_cols = {
            "level", "identifier", "score", "effect_size", "q_value", "rank",
            "pathways", "cpg_island_context", "hms_summary",
        }
        assert set(report.columns) == expected_cols

    def test_report_row_count(
        self,
        cpg_diff_df: pl.DataFrame,
        gene_diff_df: pl.DataFrame,
        pathway_enrichment_df: pl.DataFrame,
    ) -> None:
        cpg = rank_cpg_biomarkers(cpg_diff_df, n_top=3)
        gene = rank_gene_biomarkers(gene_diff_df, n_top=2)
        pathway = rank_pathway_biomarkers(pathway_enrichment_df, n_top=1)

        report = generate_biomarker_report(cpg, gene, pathway)
        assert report.height == 3 + 2 + 1

    def test_empty_report(self) -> None:
        empty = pl.DataFrame()
        report = generate_biomarker_report(empty, empty, empty)
        assert report.height == 0

    def test_report_with_enrichment_and_hms(
        self,
        cpg_diff_df: pl.DataFrame,
        gene_diff_df: pl.DataFrame,
        pathway_enrichment_df: pl.DataFrame,
    ) -> None:
        """Report includes pathway memberships and HMS summary when provided."""
        cpg = rank_cpg_biomarkers(cpg_diff_df, n_top=2)
        gene = rank_gene_biomarkers(gene_diff_df, n_top=3)
        pathway = rank_pathway_biomarkers(pathway_enrichment_df, n_top=2)

        enrichment = pl.DataFrame({
            "pathway": ["Apoptosis", "Cell Cycle"],
            "q_value": [1e-5, 1e-3],
            "overlap_genes": [["TP53", "BRCA1"], ["MYC", "KRAS"]],
        })
        hms = pl.DataFrame({
            "sample_id": ["S1", "S2", "S3"],
            "hms_count": [100, 200, 50],
            "clinical_category": ["CRC", "CRC", "Control"],
        })

        report = generate_biomarker_report(
            cpg, gene, pathway,
            enrichment_results=enrichment,
            hypermethylation_scores=hms,
        )
        # Gene-level rows should have pathway info
        gene_rows = report.filter(pl.col("level") == "gene")
        tp53_row = gene_rows.filter(pl.col("identifier") == "TP53")
        if tp53_row.height > 0:
            assert "Apoptosis" in tp53_row["pathways"].item()
        # HMS summary should be present for gene rows
        assert any(
            "CRC" in str(r["hms_summary"])
            for r in gene_rows.iter_rows(named=True)
        )


# ---------------------------------------------------------------------------
# export_to_typedb
# ---------------------------------------------------------------------------


class TestExportToTypedb:
    """Tests for the TypeDB export stub."""

    def test_raises_not_implemented(self) -> None:
        dummy_df = pl.DataFrame({"level": ["cpg"], "identifier": ["cg00000001"]})
        with pytest.raises(NotImplementedError, match="TypeDB export not yet implemented"):
            export_to_typedb(dummy_df, typedb_driver=None)
