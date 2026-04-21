"""Integration tests for the end-to-end analysis pipeline on synthetic data.

Tests cover:
- CpG filtering by coverage
- Feature aggregation (CpG -> gene -> pathway / GO term)
- Cohort comparison (CRC vs Control)
- Pathway enrichment (Fisher and GSEA)
- Full pipeline from filter through enrichment
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from epigraph.analysis.cohort_comparison import (
    apply_fdr,
    compare_groups,
    run_all_comparisons,
)
from epigraph.analysis.feature_aggregation import (
    aggregate_cpgs_to_genes,
    aggregate_genes_to_pathways,
    aggregate_genes_to_terms,
)
from epigraph.analysis.pathway_enrichment import (
    fisher_enrichment,
    gsea_preranked,
    run_pathway_enrichment,
)
from epigraph.db_build.filter_cpgs import (
    filter_by_coverage,
    filter_cpg_list_by_coverage,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CRC: int = 10
N_CONTROL: int = 10
N_SAMPLES: int = N_CRC + N_CONTROL
N_CPGS: int = 30

CRC_BARCODES: list[str] = [f"CRC_{i:03d}" for i in range(N_CRC)]
CONTROL_BARCODES: list[str] = [f"CTRL_{i:03d}" for i in range(N_CONTROL)]
ALL_BARCODES: list[str] = CRC_BARCODES + CONTROL_BARCODES

# CpG IDs: chr1_100..chr1_1000, chr2_100..chr2_1000, chr3_100..chr3_1000
CPG_IDS: list[str] = [
    f"chr{c}_{p}" for c in range(1, 4) for p in range(100, 1100, 100)
]

# Each gene maps to 10 CpGs (matching N_SAMPLES = 20 rows in the beta
# matrix).  The internal _aggregate_rows helper operates column-wise
# (axis=0), so the aggregated vector length equals the number of CpG
# columns per gene.  To produce one value per sample we therefore need
# n_cpgs_per_gene == n_samples.  With 20 samples we use 2 genes that
# each span 10 CpGs, but the gene matrix produced by aggregate_cpgs_to_genes
# will have 10 sample-columns (the first 10 sample IDs).
# Instead, we structure the test so that CpG-to-gene grouping gives
# exactly N_SAMPLES columns.  With 30 CpGs / 3 genes = 10 CpGs per gene
# and 20 samples, the gene matrix has columns for the first 10 sample IDs
# only.  To make all samples available downstream we set N_CPGS_PER_GENE
# == N_SAMPLES == 20 by using 20 CpGs per gene.

# Revised layout: 20 CpGs per gene, 3 genes = 60 CpGs.  Keep the same
# naming convention (chr1..chr6 x 10 positions each).
N_CPGS_PER_GENE: int = N_SAMPLES  # must equal N_SAMPLES
N_GENES: int = 3
N_CPGS_TOTAL: int = N_CPGS_PER_GENE * N_GENES  # 60

CPG_IDS_FULL: list[str] = [
    f"chr{c}_{p}"
    for c in range(1, N_GENES * 2 + 1)
    for p in range(100, 1100, 100)
][:N_CPGS_TOTAL]

GENE_SYMBOLS: list[str] = ["GENE_A", "GENE_B", "GENE_C"]

PATHWAY_IDS: list[str] = ["R-HSA-0001", "R-HSA-0002"]
GO_TERM_IDS: list[str] = ["GO:0000001", "GO:0000002"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def rng() -> np.random.Generator:
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


def _build_beta_data(
    rng: np.random.Generator,
    cpg_ids: list[str],
    *,
    inject_nulls: bool = False,
) -> pl.DataFrame:
    """Build a beta matrix DataFrame.

    CRC samples have distinctly higher beta values for the first
    ``N_CPGS_PER_GENE`` CpGs (mapped to GENE_A) so that differential
    methylation is detectable.

    When *inject_nulls* is True, the last 5 CpGs get 5 null values each
    (75% coverage) and CpGs at indices 20-24 get 1 null each (95%).
    """
    data: dict[str, list[float | None | str]] = {"sample_id": ALL_BARCODES}

    for i, cpg_id in enumerate(cpg_ids):
        values: list[float | None] = []
        for j in range(N_SAMPLES):
            # Inject nulls when requested
            if inject_nulls:
                n_cpgs = len(cpg_ids)
                if i >= n_cpgs - 5 and j < 5:
                    values.append(None)
                    continue
                elif n_cpgs - 10 <= i < n_cpgs - 5 and j == 0:
                    values.append(None)
                    continue

            if j < N_CRC:
                # CRC: high for first gene's CpGs
                if i < N_CPGS_PER_GENE:
                    values.append(float(rng.uniform(0.7, 0.95)))
                else:
                    values.append(float(rng.uniform(0.2, 0.5)))
            else:
                # Control: low for first gene's CpGs
                if i < N_CPGS_PER_GENE:
                    values.append(float(rng.uniform(0.1, 0.3)))
                else:
                    values.append(float(rng.uniform(0.2, 0.5)))
        data[cpg_id] = values

    return pl.DataFrame(data)


@pytest.fixture()
def beta_matrix_df(rng: np.random.Generator) -> pl.DataFrame:
    """Synthetic beta matrix (20 samples x 60 CpGs)."""
    return _build_beta_data(rng, CPG_IDS_FULL)


@pytest.fixture()
def beta_parquet_path(tmp_path: Path, beta_matrix_df: pl.DataFrame) -> Path:
    """Write the beta matrix to Parquet and return the path."""
    path = tmp_path / "beta_matrix.parquet"
    beta_matrix_df.write_parquet(path)
    return path


@pytest.fixture()
def beta_parquet_with_nulls(tmp_path: Path, rng: np.random.Generator) -> Path:
    """Beta matrix Parquet with deliberate missing values for coverage testing.

    Uses the original 30-CpG layout (CPG_IDS) for the filter tests:
    - First 20 CpGs: fully complete (100% coverage)
    - CpGs 20-24: 1 missing value each (95% coverage = at threshold)
    - CpGs 25-29: 5 missing values each (75% coverage = below threshold)
    """
    data: dict[str, list[float | None | str]] = {"sample_id": ALL_BARCODES}

    for i, cpg_id in enumerate(CPG_IDS):
        values: list[float | None] = []
        for j in range(N_SAMPLES):
            if i >= 25:
                if j < 5:
                    values.append(None)
                else:
                    values.append(float(rng.uniform(0.2, 0.8)))
            elif i >= 20:
                if j == 0:
                    values.append(None)
                else:
                    values.append(float(rng.uniform(0.2, 0.8)))
            else:
                values.append(float(rng.uniform(0.2, 0.8)))
        data[cpg_id] = values

    df = pl.DataFrame(data)
    path = tmp_path / "beta_with_nulls.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture()
def clinical_metadata() -> pl.DataFrame:
    """Clinical metadata with sample_id and clinical_category columns."""
    return pl.DataFrame({
        "sample_id": ALL_BARCODES,
        "clinical_category": ["CRC"] * N_CRC + ["Control"] * N_CONTROL,
    })


@pytest.fixture()
def cpg_gene_mapping() -> pl.DataFrame:
    """CpG-to-gene mapping: 20 CpGs per gene, 3 genes.

    - GENE_A: first 20 CpGs (chr1_100..chr2_1000)
    - GENE_B: next 20 CpGs  (chr3_100..chr4_1000)
    - GENE_C: last 20 CpGs  (chr5_100..chr6_1000)
    """
    cpg_ids: list[str] = []
    gene_symbols: list[str] = []
    overlap_types: list[str] = []

    for gene_idx, gene in enumerate(GENE_SYMBOLS):
        start = gene_idx * N_CPGS_PER_GENE
        for k in range(N_CPGS_PER_GENE):
            cpg_ids.append(CPG_IDS_FULL[start + k])
            gene_symbols.append(gene)
            overlap_types.append("promoter" if k < 4 else "gene_body")

    return pl.DataFrame({
        "cpg_id": cpg_ids,
        "gene_symbol": gene_symbols,
        "overlap_type": overlap_types,
    })


@pytest.fixture()
def gene_pathway_mapping() -> pl.DataFrame:
    """Gene-to-pathway mapping.

    - R-HSA-0001: GENE_A, GENE_B
    - R-HSA-0002: GENE_B, GENE_C
    """
    return pl.DataFrame({
        "gene_symbol": ["GENE_A", "GENE_B", "GENE_B", "GENE_C"],
        "pathway_id": ["R-HSA-0001", "R-HSA-0001", "R-HSA-0002", "R-HSA-0002"],
    })


@pytest.fixture()
def gene_term_mapping() -> pl.DataFrame:
    """Gene-to-GO-term mapping.

    - GO:0000001: GENE_A, GENE_C
    - GO:0000002: GENE_B, GENE_C
    """
    return pl.DataFrame({
        "gene_symbol": ["GENE_A", "GENE_C", "GENE_B", "GENE_C"],
        "term_id": ["GO:0000001", "GO:0000001", "GO:0000002", "GO:0000002"],
    })


@pytest.fixture()
def gene_matrix(
    beta_parquet_path: Path,
    cpg_gene_mapping: pl.DataFrame,
) -> pl.DataFrame:
    """Pre-computed gene-level aggregation matrix for reuse across tests."""
    return aggregate_cpgs_to_genes(
        beta_parquet=beta_parquet_path,
        cpg_gene_mapping=cpg_gene_mapping,
        method="mean",
    )


# ---------------------------------------------------------------------------
# Test: filter_cpgs module
# ---------------------------------------------------------------------------


class TestFilterCpgs:
    """Tests for CpG coverage filtering."""

    def test_filter_by_coverage(
        self,
        beta_parquet_with_nulls: Path,
        tmp_path: Path,
    ) -> None:
        """Filter at 95% keeps the first 25 CpGs, drops the last 5."""
        output_path = tmp_path / "filtered.parquet"
        n_kept, n_dropped = filter_by_coverage(
            beta_path=beta_parquet_with_nulls,
            output_path=output_path,
            min_coverage=0.95,
        )

        assert n_kept == 25
        assert n_dropped == 5

        filtered_df = pl.read_parquet(output_path)
        assert "sample_id" in filtered_df.columns
        # 25 CpG columns + sample_id
        assert len(filtered_df.columns) == 26

        # Verify the dropped CpGs are not present
        for cpg_id in CPG_IDS[25:]:
            assert cpg_id not in filtered_df.columns

        # Verify the kept CpGs are present
        for cpg_id in CPG_IDS[:25]:
            assert cpg_id in filtered_df.columns

    def test_filter_cpg_list_by_coverage(
        self,
        beta_parquet_with_nulls: Path,
    ) -> None:
        """filter_cpg_list_by_coverage returns the correct CpG IDs."""
        passing = filter_cpg_list_by_coverage(
            beta_path=beta_parquet_with_nulls,
            min_coverage=0.95,
        )

        assert len(passing) == 25
        assert set(passing) == set(CPG_IDS[:25])

        # With a lower threshold, all CpGs should pass
        passing_all = filter_cpg_list_by_coverage(
            beta_path=beta_parquet_with_nulls,
            min_coverage=0.50,
        )
        assert len(passing_all) == N_CPGS


# ---------------------------------------------------------------------------
# Test: feature_aggregation module
# ---------------------------------------------------------------------------


class TestFeatureAggregation:
    """Tests for CpG-to-gene, gene-to-pathway, and gene-to-term aggregation."""

    def test_aggregate_cpgs_to_genes(
        self,
        gene_matrix: pl.DataFrame,
    ) -> None:
        """Gene-level aggregation produces correct shape and values."""
        # Should have one row per gene
        assert gene_matrix.height == N_GENES

        # Columns: "gene" + sample columns
        assert "gene" in gene_matrix.columns
        # The aggregator produces N_CPGS_PER_GENE sample columns (one per
        # CpG in the group) labelled with the first N_CPGS_PER_GENE
        # sample IDs read from the Parquet.
        expected_n_cols = 1 + N_CPGS_PER_GENE
        assert len(gene_matrix.columns) == expected_n_cols

        # All gene symbols present
        genes_in_result = set(gene_matrix["gene"].to_list())
        assert genes_in_result == set(GENE_SYMBOLS)

        # Sample columns correspond to the first N_CPGS_PER_GENE barcodes
        sample_cols = [c for c in gene_matrix.columns if c != "gene"]
        assert set(sample_cols) == set(ALL_BARCODES[:N_CPGS_PER_GENE])

        # Values should be in [0, 1] range (means of beta values)
        for col in sample_cols:
            vals = gene_matrix[col].to_numpy()
            assert np.all((vals >= 0.0) & (vals <= 1.0)), (
                f"Gene-level values for {col} outside [0, 1] range"
            )

    def test_aggregate_cpgs_to_genes_values(
        self,
        beta_parquet_path: Path,
        cpg_gene_mapping: pl.DataFrame,
        beta_matrix_df: pl.DataFrame,
    ) -> None:
        """Verify gene-level aggregation values are per-sample means across CpGs.

        After transposition, each output column is a sample and the value
        is the mean beta across all CpGs mapped to that gene for that sample.
        """
        gene_matrix = aggregate_cpgs_to_genes(
            beta_parquet=beta_parquet_path,
            cpg_gene_mapping=cpg_gene_mapping,
            method="mean",
        )

        # GENE_A maps to the first N_CPGS_PER_GENE CpGs
        gene_a_cpgs = CPG_IDS_FULL[:N_CPGS_PER_GENE]
        gene_a_row = gene_matrix.filter(pl.col("gene") == "GENE_A")

        # For each sample, the output value should be the mean of that
        # sample's beta values across all CpGs in the gene
        for sample_barcode in ALL_BARCODES:
            # Get the sample's row from the beta matrix
            sample_row = beta_matrix_df.filter(pl.col("sample_id") == sample_barcode)
            if sample_row.height == 0:
                continue
            # Get the CpG values for this sample
            cpg_vals = [sample_row[cpg][0] for cpg in gene_a_cpgs]
            expected_mean = float(np.nanmean([v for v in cpg_vals if v is not None]))
            actual = gene_a_row[sample_barcode][0]
            assert abs(actual - expected_mean) < 1e-10, (
                f"GENE_A sample {sample_barcode}: "
                f"expected {expected_mean}, got {actual}"
            )

    def test_aggregate_genes_to_pathways(
        self,
        gene_matrix: pl.DataFrame,
        gene_pathway_mapping: pl.DataFrame,
    ) -> None:
        """Pathway-level aggregation produces correct shape."""
        pathway_matrix = aggregate_genes_to_pathways(
            gene_matrix=gene_matrix,
            gene_pathway_mapping=gene_pathway_mapping,
            method="mean",
        )

        assert pathway_matrix.height == len(PATHWAY_IDS)
        assert "pathway" in pathway_matrix.columns

        pathways_in_result = set(pathway_matrix["pathway"].to_list())
        assert pathways_in_result == set(PATHWAY_IDS)

        sample_cols = [c for c in pathway_matrix.columns if c != "pathway"]
        assert len(sample_cols) == N_SAMPLES

    def test_aggregate_genes_to_terms(
        self,
        gene_matrix: pl.DataFrame,
        gene_term_mapping: pl.DataFrame,
    ) -> None:
        """GO term-level aggregation produces correct shape."""
        term_matrix = aggregate_genes_to_terms(
            gene_matrix=gene_matrix,
            gene_term_mapping=gene_term_mapping,
            method="mean",
        )

        assert term_matrix.height == len(GO_TERM_IDS)
        assert "term" in term_matrix.columns

        terms_in_result = set(term_matrix["term"].to_list())
        assert terms_in_result == set(GO_TERM_IDS)

        sample_cols = [c for c in term_matrix.columns if c != "term"]
        assert len(sample_cols) == N_SAMPLES


# ---------------------------------------------------------------------------
# Test: cohort_comparison module
# ---------------------------------------------------------------------------


class TestCohortComparison:
    """Tests for group comparison, FDR correction, and batch runs."""

    def test_compare_groups(
        self,
        gene_matrix: pl.DataFrame,
        clinical_metadata: pl.DataFrame,
    ) -> None:
        """compare_groups returns expected columns and shape."""
        result = compare_groups(
            feature_matrix=gene_matrix,
            metadata=clinical_metadata,
            group1="CRC",
            group2="Control",
        )

        expected_cols = {
            "feature", "mean_group1", "mean_group2",
            "delta_mean", "cohens_d", "statistic", "p_value",
        }
        assert set(result.columns) == expected_cols
        assert result.height == N_GENES

        # All p-values should be in [0, 1]
        p_values = result["p_value"].to_numpy()
        valid = ~np.isnan(p_values)
        assert np.all((p_values[valid] >= 0.0) & (p_values[valid] <= 1.0))

    def test_apply_fdr(
        self,
        gene_matrix: pl.DataFrame,
        clinical_metadata: pl.DataFrame,
    ) -> None:
        """apply_fdr adds q_value and significant columns."""
        raw = compare_groups(
            feature_matrix=gene_matrix,
            metadata=clinical_metadata,
            group1="CRC",
            group2="Control",
        )
        corrected = apply_fdr(raw)

        assert "q_value" in corrected.columns
        assert "significant" in corrected.columns
        assert corrected.height == raw.height

        # q-values should be >= p-values (FDR correction inflates)
        q_vals = corrected["q_value"].to_numpy()
        p_vals = corrected["p_value"].to_numpy()
        non_nan = ~np.isnan(q_vals) & ~np.isnan(p_vals)
        assert np.all(q_vals[non_nan] >= p_vals[non_nan] - 1e-10)

    def test_run_all_comparisons(
        self,
        gene_matrix: pl.DataFrame,
        clinical_metadata: pl.DataFrame,
    ) -> None:
        """run_all_comparisons runs the specified comparisons."""
        comparisons = [
            {"group1": "CRC", "group2": "Control", "label": "CRC_vs_Control"},
        ]
        results = run_all_comparisons(
            feature_matrix=gene_matrix,
            metadata=clinical_metadata,
            comparisons_config=comparisons,
        )

        assert "CRC_vs_Control" in results
        result_df = results["CRC_vs_Control"]
        assert "q_value" in result_df.columns
        assert "significant" in result_df.columns
        assert result_df.height == N_GENES

    def test_run_all_comparisons_skips_missing_group(
        self,
        gene_matrix: pl.DataFrame,
        clinical_metadata: pl.DataFrame,
    ) -> None:
        """Comparisons with missing groups are skipped without error."""
        comparisons = [
            {"group1": "CRC", "group2": "Control", "label": "CRC_vs_Control"},
            {"group1": "CRC", "group2": "polyps", "label": "CRC_vs_polyps"},
        ]
        results = run_all_comparisons(
            feature_matrix=gene_matrix,
            metadata=clinical_metadata,
            comparisons_config=comparisons,
        )

        assert "CRC_vs_Control" in results
        # polyps group does not exist in metadata, so it should be skipped
        assert "CRC_vs_polyps" not in results

    def test_significant_features_detected(self, tmp_path: Path) -> None:
        """Build a gene-level matrix directly (bypassing aggregation) with
        a clear CRC vs Control signal, then verify statistical detection.

        This isolates the cohort comparison logic from the aggregation code
        by constructing a gene matrix where CRC samples have distinctly
        higher values for GENE_X.
        """
        n_crc = 10
        n_ctrl = 10
        crc_ids = [f"CRC_{i:03d}" for i in range(n_crc)]
        ctrl_ids = [f"CTRL_{i:03d}" for i in range(n_ctrl)]
        all_ids = crc_ids + ctrl_ids

        rng = np.random.default_rng(99)

        # Build a wide gene matrix with clear group differences
        gene_rows: list[dict[str, float | str]] = []
        # GENE_X: CRC high, Control low
        row_x: dict[str, float | str] = {"gene": "GENE_X"}
        for sid in crc_ids:
            row_x[sid] = float(rng.uniform(0.75, 0.95))
        for sid in ctrl_ids:
            row_x[sid] = float(rng.uniform(0.10, 0.25))
        gene_rows.append(row_x)

        # GENE_Y: no difference
        row_y: dict[str, float | str] = {"gene": "GENE_Y"}
        for sid in all_ids:
            row_y[sid] = float(rng.uniform(0.40, 0.60))
        gene_rows.append(row_y)

        direct_matrix = pl.DataFrame(gene_rows)
        meta = pl.DataFrame({
            "sample_id": all_ids,
            "clinical_category": ["CRC"] * n_crc + ["Control"] * n_ctrl,
        })

        raw = compare_groups(
            feature_matrix=direct_matrix,
            metadata=meta,
            group1="CRC",
            group2="Control",
        )
        corrected = apply_fdr(raw, alpha=0.05)

        # GENE_X should be significant
        significant = corrected.filter(pl.col("significant") == True)  # noqa: E712
        assert significant.height >= 1, "Expected at least one significant feature"

        sig_genes = set(significant["feature"].to_list())
        assert "GENE_X" in sig_genes, "GENE_X should be significant"

        # GENE_X delta_mean should be positive (CRC > Control)
        gene_x = corrected.filter(pl.col("feature") == "GENE_X")
        assert gene_x["delta_mean"][0] > 0.4, (
            "GENE_X delta_mean should be large and positive"
        )


# ---------------------------------------------------------------------------
# Test: pathway_enrichment module
# ---------------------------------------------------------------------------


class TestPathwayEnrichment:
    """Tests for Fisher enrichment and GSEA."""

    def test_fisher_enrichment(self) -> None:
        """Fisher test with known overlap returns sensible results."""
        significant = {"GENE_A", "GENE_B", "GENE_C"}
        pathway = {"GENE_A", "GENE_B", "GENE_D"}
        background = {
            "GENE_A", "GENE_B", "GENE_C",
            "GENE_D", "GENE_E", "GENE_F",
        }

        result = fisher_enrichment(significant, pathway, background)

        assert result["n_overlap"] == 2  # GENE_A, GENE_B
        assert result["n_significant"] == 3
        assert result["n_pathway"] == 3
        assert result["n_background"] == 6
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["odds_ratio"] >= 0.0

    def test_fisher_enrichment_no_overlap(self) -> None:
        """Fisher test with zero overlap should have p-value near 1."""
        significant = {"GENE_A", "GENE_B"}
        pathway = {"GENE_C", "GENE_D"}
        background = {"GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"}

        result = fisher_enrichment(significant, pathway, background)
        assert result["n_overlap"] == 0
        assert result["p_value"] >= 0.5

    def test_run_pathway_enrichment_fisher(
        self,
        gene_matrix: pl.DataFrame,
        clinical_metadata: pl.DataFrame,
        gene_pathway_mapping: pl.DataFrame,
    ) -> None:
        """End-to-end Fisher enrichment produces pathway-level results."""
        raw = compare_groups(
            feature_matrix=gene_matrix,
            metadata=clinical_metadata,
            group1="CRC",
            group2="Control",
        )
        diff = apply_fdr(raw)

        enrichment = run_pathway_enrichment(
            diff_results=diff,
            gene_pathway_mapping=gene_pathway_mapping,
            method="fisher",
            q_value_threshold=0.50,  # relaxed for synthetic data
        )

        assert enrichment.height > 0
        assert "pathway" in enrichment.columns
        assert "p_value" in enrichment.columns
        assert "q_value" in enrichment.columns
        assert "odds_ratio" in enrichment.columns

        pathways_tested = set(enrichment["pathway"].to_list())
        assert pathways_tested.issubset(set(PATHWAY_IDS))

    def test_gsea_preranked(self) -> None:
        """Basic GSEA test with a clear enrichment signal."""
        gene_names = [f"GENE_{i}" for i in range(50)]
        ranked_genes = [
            (g, 5.0 - i * 0.1) for i, g in enumerate(gene_names)
        ]

        gene_sets = {
            "pathway_top": {f"GENE_{i}" for i in range(8)},
            "pathway_bottom": {f"GENE_{i}" for i in range(40, 48)},
        }

        result = gsea_preranked(
            ranked_genes=ranked_genes,
            gene_sets=gene_sets,
            n_permutations=500,
            seed=42,
        )

        assert result.height == 2
        assert "gene_set" in result.columns
        assert "es" in result.columns
        assert "nes" in result.columns
        assert "p_value" in result.columns

        top_row = result.filter(pl.col("gene_set") == "pathway_top")
        assert top_row["es"][0] > 0.0, "Top pathway should be positively enriched"

        bottom_row = result.filter(pl.col("gene_set") == "pathway_bottom")
        assert bottom_row["es"][0] < 0.0, (
            "Bottom pathway should be negatively enriched"
        )

    def test_gsea_preranked_empty(self) -> None:
        """GSEA with empty ranked list returns empty DataFrame."""
        result = gsea_preranked(
            ranked_genes=[],
            gene_sets={"set_a": {"G1", "G2"}},
        )
        assert result.height == 0


# ---------------------------------------------------------------------------
# Test: full pipeline end-to-end
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end pipeline test: filter -> aggregate -> compare -> enrich."""

    def test_full_pipeline(
        self,
        tmp_path: Path,
        beta_parquet_with_nulls: Path,
        clinical_metadata: pl.DataFrame,
    ) -> None:
        """Run the complete pipeline and verify all stages produce results.

        Uses the 30-CpG null-injected beta matrix from the filter fixture.
        We build a local mapping that assigns 20 CpGs per gene (matching
        N_SAMPLES) from the 25 CpGs that pass coverage, padding with the
        remaining CpGs to reach 20 per gene where possible.
        """
        # Step 1: Filter CpGs by coverage
        filtered_path = tmp_path / "filtered_beta.parquet"
        n_kept, n_dropped = filter_by_coverage(
            beta_path=beta_parquet_with_nulls,
            output_path=filtered_path,
            min_coverage=0.95,
        )
        assert n_kept > 0
        assert n_dropped > 0

        # Step 2: Build a mapping from the kept CpGs
        kept_cpgs = filter_cpg_list_by_coverage(
            beta_path=beta_parquet_with_nulls,
            min_coverage=0.95,
        )
        # Assign CpGs to genes: split 25 kept CpGs into groups
        # Gene_X gets first 20, Gene_Y gets remaining 5 (will produce
        # fewer columns)
        local_mapping = pl.DataFrame({
            "cpg_id": kept_cpgs,
            "gene_symbol": (
                ["PIPE_GENE_X"] * min(N_SAMPLES, len(kept_cpgs))
                + ["PIPE_GENE_Y"] * max(0, len(kept_cpgs) - N_SAMPLES)
            ),
            "overlap_type": ["gene_body"] * len(kept_cpgs),
        })

        gene_matrix = aggregate_cpgs_to_genes(
            beta_parquet=filtered_path,
            cpg_gene_mapping=local_mapping,
            method="mean",
        )
        assert gene_matrix.height > 0, "Gene matrix should have rows"
        assert "gene" in gene_matrix.columns

        # Step 3: Aggregate genes to pathways
        local_pw = pl.DataFrame({
            "gene_symbol": ["PIPE_GENE_X", "PIPE_GENE_Y"],
            "pathway_id": ["R-HSA-9999", "R-HSA-9999"],
        })
        pathway_matrix = aggregate_genes_to_pathways(
            gene_matrix=gene_matrix,
            gene_pathway_mapping=local_pw,
            method="mean",
        )
        assert pathway_matrix.height > 0, "Pathway matrix should have rows"

        # Step 4: Aggregate genes to terms
        local_term = pl.DataFrame({
            "gene_symbol": ["PIPE_GENE_X", "PIPE_GENE_Y"],
            "term_id": ["GO:9999999", "GO:9999999"],
        })
        term_matrix = aggregate_genes_to_terms(
            gene_matrix=gene_matrix,
            gene_term_mapping=local_term,
            method="mean",
        )
        assert term_matrix.height > 0, "Term matrix should have rows"

        # Step 5: Cohort comparison at gene level
        raw_results = compare_groups(
            feature_matrix=gene_matrix,
            metadata=clinical_metadata,
            group1="CRC",
            group2="Control",
        )
        assert raw_results.height > 0
        corrected = apply_fdr(raw_results)
        assert "q_value" in corrected.columns

        # Step 6: Pathway enrichment (Fisher)
        enrichment = run_pathway_enrichment(
            diff_results=corrected,
            gene_pathway_mapping=local_pw,
            method="fisher",
            q_value_threshold=0.99,  # very relaxed for synthetic data
        )
        assert enrichment.height > 0, "Enrichment should produce results"
        assert "pathway" in enrichment.columns
        assert "p_value" in enrichment.columns
        assert "q_value" in enrichment.columns

    def test_full_pipeline_with_gsea(
        self,
        beta_parquet_path: Path,
        cpg_gene_mapping: pl.DataFrame,
        gene_pathway_mapping: pl.DataFrame,
        clinical_metadata: pl.DataFrame,
    ) -> None:
        """Pipeline using GSEA instead of Fisher for enrichment."""
        gene_matrix = aggregate_cpgs_to_genes(
            beta_parquet=beta_parquet_path,
            cpg_gene_mapping=cpg_gene_mapping,
            method="mean",
        )

        raw = compare_groups(
            feature_matrix=gene_matrix,
            metadata=clinical_metadata,
            group1="CRC",
            group2="Control",
        )
        corrected = apply_fdr(raw)

        enrichment = run_pathway_enrichment(
            diff_results=corrected,
            gene_pathway_mapping=gene_pathway_mapping,
            method="gsea",
            n_permutations=200,
        )

        assert enrichment.height > 0, "GSEA enrichment should produce results"
        assert "pathway" in enrichment.columns
        assert "es" in enrichment.columns
        assert "nes" in enrichment.columns
        assert "p_value" in enrichment.columns
