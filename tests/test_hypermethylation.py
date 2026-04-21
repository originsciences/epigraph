"""Tests for epigraph.analysis.hypermethylation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from epigraph.analysis.hypermethylation import (
    compute_gene_thresholds,
    run_hypermethylation_analysis,
    score_hypermethylation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def gene_matrix() -> pl.DataFrame:
    """Gene matrix with 5 genes and 6 samples (3 control, 3 CRC)."""
    return pl.DataFrame({
        "gene": ["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"],
        "CTRL_1": [0.1, 0.2, 0.3, 0.4, 0.5],
        "CTRL_2": [0.12, 0.22, 0.32, 0.42, 0.52],
        "CTRL_3": [0.11, 0.21, 0.31, 0.41, 0.51],
        "CRC_1": [0.5, 0.8, 0.9, 0.4, 0.5],
        "CRC_2": [0.6, 0.9, 0.85, 0.42, 0.55],
        "CRC_3": [0.55, 0.85, 0.88, 0.43, 0.48],
    })


@pytest.fixture()
def metadata() -> pl.DataFrame:
    """Clinical metadata matching the gene matrix fixture."""
    return pl.DataFrame({
        "barcode": ["CTRL_1", "CTRL_2", "CTRL_3", "CRC_1", "CRC_2", "CRC_3"],
        "clinical_category": [
            "Control", "Control", "Control", "CRC", "CRC", "CRC",
        ],
    })


# ---------------------------------------------------------------------------
# compute_gene_thresholds
# ---------------------------------------------------------------------------


class TestComputeGeneThresholds:
    """Tests for per-gene threshold computation."""

    def test_returns_series_with_correct_length(
        self, gene_matrix: pl.DataFrame, metadata: pl.DataFrame
    ) -> None:
        thresholds = compute_gene_thresholds(gene_matrix, metadata, quantile=0.99)
        assert len(thresholds) == gene_matrix.height

    def test_threshold_values_are_finite(
        self, gene_matrix: pl.DataFrame, metadata: pl.DataFrame
    ) -> None:
        thresholds = compute_gene_thresholds(gene_matrix, metadata, quantile=0.99)
        assert all(np.isfinite(thresholds.to_numpy()))

    def test_threshold_at_quantile_1_is_max(
        self, gene_matrix: pl.DataFrame, metadata: pl.DataFrame
    ) -> None:
        """At quantile=1.0, threshold should be the max control value per gene."""
        thresholds = compute_gene_thresholds(gene_matrix, metadata, quantile=1.0)
        control_cols = ["CTRL_1", "CTRL_2", "CTRL_3"]
        control_data = gene_matrix.select(control_cols).to_numpy()
        expected_max = np.max(control_data, axis=1)
        np.testing.assert_allclose(thresholds.to_numpy(), expected_max, rtol=1e-10)

    def test_threshold_at_quantile_0_is_min(
        self, gene_matrix: pl.DataFrame, metadata: pl.DataFrame
    ) -> None:
        """At quantile=0.0, threshold should be the min control value per gene."""
        thresholds = compute_gene_thresholds(gene_matrix, metadata, quantile=0.0)
        control_cols = ["CTRL_1", "CTRL_2", "CTRL_3"]
        control_data = gene_matrix.select(control_cols).to_numpy()
        expected_min = np.min(control_data, axis=1)
        np.testing.assert_allclose(thresholds.to_numpy(), expected_min, rtol=1e-10)

    def test_raises_on_no_controls(self, gene_matrix: pl.DataFrame) -> None:
        """Should raise if no control samples match."""
        bad_metadata = pl.DataFrame({
            "barcode": ["CTRL_1", "CTRL_2", "CTRL_3"],
            "clinical_category": ["CRC", "CRC", "CRC"],
        })
        with pytest.raises(ValueError, match="No control samples"):
            compute_gene_thresholds(gene_matrix, bad_metadata)

    def test_uses_only_control_samples(
        self, gene_matrix: pl.DataFrame, metadata: pl.DataFrame
    ) -> None:
        """Thresholds should be based only on control sample values."""
        thresholds = compute_gene_thresholds(gene_matrix, metadata, quantile=0.5)
        control_cols = ["CTRL_1", "CTRL_2", "CTRL_3"]
        control_data = gene_matrix.select(control_cols).to_numpy()
        expected_median = np.median(control_data, axis=1)
        np.testing.assert_allclose(thresholds.to_numpy(), expected_median, rtol=1e-10)


# ---------------------------------------------------------------------------
# score_hypermethylation
# ---------------------------------------------------------------------------


class TestScoreHypermethylation:
    """Tests for sample-level hypermethylation scoring."""

    def test_returns_expected_columns(
        self, gene_matrix: pl.DataFrame, metadata: pl.DataFrame
    ) -> None:
        thresholds = compute_gene_thresholds(gene_matrix, metadata, quantile=0.99)
        scores = score_hypermethylation(gene_matrix, thresholds)
        assert set(scores.columns) == {"sample_id", "hms_count"}

    def test_returns_all_samples(
        self, gene_matrix: pl.DataFrame, metadata: pl.DataFrame
    ) -> None:
        thresholds = compute_gene_thresholds(gene_matrix, metadata, quantile=0.99)
        scores = score_hypermethylation(gene_matrix, thresholds)
        sample_cols = [c for c in gene_matrix.columns if c != "gene"]
        assert scores.height == len(sample_cols)

    def test_control_samples_have_low_scores_at_high_quantile(
        self, gene_matrix: pl.DataFrame, metadata: pl.DataFrame
    ) -> None:
        """At max-based threshold, controls should not exceed any gene."""
        # Use quantile=1.0 so threshold equals max control value per gene.
        # No control sample can exceed its own max, so HMS should be 0.
        thresholds = compute_gene_thresholds(gene_matrix, metadata, quantile=1.0)
        scores = score_hypermethylation(gene_matrix, thresholds)
        ctrl_scores = scores.filter(
            pl.col("sample_id").is_in(["CTRL_1", "CTRL_2", "CTRL_3"])
        )
        assert all(s == 0 for s in ctrl_scores["hms_count"].to_list())

    def test_crc_samples_have_higher_scores(
        self, gene_matrix: pl.DataFrame, metadata: pl.DataFrame
    ) -> None:
        """CRC samples with hypermethylated genes should have higher HMS."""
        thresholds = compute_gene_thresholds(gene_matrix, metadata, quantile=0.99)
        scores = score_hypermethylation(gene_matrix, thresholds)

        ctrl_mean = scores.filter(
            pl.col("sample_id").is_in(["CTRL_1", "CTRL_2", "CTRL_3"])
        )["hms_count"].mean()
        crc_mean = scores.filter(
            pl.col("sample_id").is_in(["CRC_1", "CRC_2", "CRC_3"])
        )["hms_count"].mean()

        assert crc_mean > ctrl_mean

    def test_raises_on_mismatched_length(self, gene_matrix: pl.DataFrame) -> None:
        """Should raise if threshold length != gene matrix height."""
        bad_thresholds = pl.Series("threshold", [0.5, 0.5])
        with pytest.raises(ValueError, match="Threshold length"):
            score_hypermethylation(gene_matrix, bad_thresholds)

    def test_all_below_threshold(self) -> None:
        """When all values are below threshold, counts should be zero."""
        gm = pl.DataFrame({
            "gene": ["G1", "G2"],
            "S1": [0.1, 0.2],
            "S2": [0.15, 0.25],
        })
        thresholds = pl.Series("threshold", [0.5, 0.5])
        scores = score_hypermethylation(gm, thresholds)
        assert all(c == 0 for c in scores["hms_count"].to_list())

    def test_all_above_threshold(self) -> None:
        """When all values exceed threshold, count equals number of genes."""
        gm = pl.DataFrame({
            "gene": ["G1", "G2"],
            "S1": [0.9, 0.9],
            "S2": [0.8, 0.8],
        })
        thresholds = pl.Series("threshold", [0.5, 0.5])
        scores = score_hypermethylation(gm, thresholds)
        assert all(c == 2 for c in scores["hms_count"].to_list())


# ---------------------------------------------------------------------------
# run_hypermethylation_analysis
# ---------------------------------------------------------------------------


class TestRunHypermethylationAnalysis:
    """Tests for the high-level analysis runner."""

    def test_writes_output_files(
        self, gene_matrix: pl.DataFrame, metadata: pl.DataFrame, tmp_path: str
    ) -> None:
        """Runner should produce threshold and score files."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            gene_path = Path(tmpdir) / "gene_features.parquet"
            meta_path = Path(tmpdir) / "metadata.parquet"
            out_dir = Path(tmpdir) / "output"

            gene_matrix.write_parquet(gene_path)
            metadata.write_parquet(meta_path)

            result = run_hypermethylation_analysis(
                gene_features_path=gene_path,
                metadata_path=meta_path,
                output_dir=out_dir,
                quantiles=[0.95, 0.99],
            )

            assert (out_dir / "gene_thresholds.parquet").exists()
            assert (out_dir / "hms_scores_q0_95.parquet").exists()
            assert (out_dir / "hms_scores_q0_99.parquet").exists()

            assert "thresholds" in result
            assert "scores" in result
            assert 0.95 in result["scores"]
            assert 0.99 in result["scores"]

    def test_scores_have_clinical_category(
        self, gene_matrix: pl.DataFrame, metadata: pl.DataFrame
    ) -> None:
        """Score output should include clinical_category from metadata join."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            gene_path = Path(tmpdir) / "gene_features.parquet"
            meta_path = Path(tmpdir) / "metadata.parquet"
            out_dir = Path(tmpdir) / "output"

            gene_matrix.write_parquet(gene_path)
            metadata.write_parquet(meta_path)

            result = run_hypermethylation_analysis(
                gene_features_path=gene_path,
                metadata_path=meta_path,
                output_dir=out_dir,
                quantiles=[0.99],
            )

            scores = result["scores"][0.99]
            assert "clinical_category" in scores.columns

    def test_threshold_table_columns(
        self, gene_matrix: pl.DataFrame, metadata: pl.DataFrame
    ) -> None:
        """Threshold table should have gene + one column per quantile."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            gene_path = Path(tmpdir) / "gene_features.parquet"
            meta_path = Path(tmpdir) / "metadata.parquet"
            out_dir = Path(tmpdir) / "output"

            gene_matrix.write_parquet(gene_path)
            metadata.write_parquet(meta_path)

            result = run_hypermethylation_analysis(
                gene_features_path=gene_path,
                metadata_path=meta_path,
                output_dir=out_dir,
                quantiles=[0.95, 0.99, 0.999],
            )

            thresh_df = result["thresholds"]
            assert "gene" in thresh_df.columns
            assert "q0.95" in thresh_df.columns
            assert "q0.99" in thresh_df.columns
            assert "q0.999" in thresh_df.columns
