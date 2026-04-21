"""Tests for the visualisation module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture()
def diff_results(tmp_path: Path) -> Path:
    """Create mock differential methylation results."""
    rng = np.random.default_rng(42)
    n = 100
    df = pl.DataFrame({
        "feature": [f"GENE_{i}" for i in range(n)],
        "mean_group1": rng.uniform(0.3, 0.7, n).tolist(),
        "mean_group2": rng.uniform(0.3, 0.7, n).tolist(),
        "delta_mean": rng.uniform(-0.1, 0.1, n).tolist(),
        "cohens_d": rng.uniform(-1.0, 1.0, n).tolist(),
        "statistic": rng.uniform(0, 100, n).tolist(),
        "p_value": rng.uniform(0.0001, 1.0, n).tolist(),
        "q_value": rng.uniform(0.001, 1.0, n).tolist(),
        "significant": ([True] * 20 + [False] * 80),
    })
    path = tmp_path / "diff_results.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture()
def enrichment_results(tmp_path: Path) -> Path:
    """Create mock enrichment results."""
    df = pl.DataFrame({
        "pathway": [f"R-HSA-{i}" for i in range(20)],
        "odds_ratio": [3.0, 2.5, 2.0] + [1.5] * 17,
        "p_value": [0.001, 0.005, 0.01] + [0.5] * 17,
        "q_value": [0.01, 0.05, 0.1] + [0.8] * 17,
        "n_overlap": [15, 20, 10] + [5] * 17,
        "n_significant": [100] * 20,
        "n_pathway": [50, 60, 40] + [30] * 17,
        "n_background": [5000] * 20,
        "significant": ([True] * 2 + [False] * 18),
    })
    path = tmp_path / "enrichment.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture()
def pathway_names(tmp_path: Path) -> Path:
    """Create mock pathway names."""
    df = pl.DataFrame({
        "pathway_id": [f"R-HSA-{i}" for i in range(20)],
        "pathway_name": [f"Pathway {i}" for i in range(20)],
        "pathway_source": ["Reactome"] * 20,
    })
    path = tmp_path / "pathway_names.parquet"
    df.write_parquet(path)
    return path


@pytest.fixture()
def hms_scores(tmp_path: Path) -> Path:
    """Create mock HMS scores."""
    rng = np.random.default_rng(42)
    categories = ["CRC"] * 10 + ["Control"] * 15 + ["polyps"] * 5
    df = pl.DataFrame({
        "sample_id": [f"PD{i:06d}" for i in range(30)],
        "hms_count": rng.integers(50, 500, 30).tolist(),
        "clinical_category": categories,
    })
    path = tmp_path / "hms_scores.parquet"
    df.write_parquet(path)
    return path


class TestVolcanoPlot:
    def test_creates_file(self, diff_results: Path, tmp_path: Path) -> None:
        from epigraph.analysis.visualise import volcano_plot

        output = tmp_path / "volcano.png"
        volcano_plot(str(diff_results), str(output), "Test Volcano")
        assert output.exists()
        assert output.stat().st_size > 1000

    def test_custom_thresholds(self, diff_results: Path, tmp_path: Path) -> None:
        from epigraph.analysis.visualise import volcano_plot

        output = tmp_path / "volcano2.png"
        volcano_plot(
            str(diff_results), str(output), "Test",
            q_threshold=0.1, effect_threshold=0.05,
        )
        assert output.exists()


class TestPathwayDotPlot:
    def test_creates_file(
        self,
        enrichment_results: Path,
        pathway_names: Path,
        tmp_path: Path,
    ) -> None:
        from epigraph.analysis.visualise import pathway_dot_plot

        output = tmp_path / "dotplot.png"
        pathway_dot_plot(
            str(enrichment_results), str(pathway_names),
            str(output), "Test Dot Plot",
        )
        assert output.exists()
        assert output.stat().st_size > 1000


class TestHmsDistribution:
    def test_creates_file(self, hms_scores: Path, tmp_path: Path) -> None:
        from epigraph.analysis.visualise import hms_distribution

        output = tmp_path / "hms.png"
        hms_distribution(str(hms_scores), str(output), "Test HMS")
        assert output.exists()
        assert output.stat().st_size > 1000


class TestGeneHeatmap:
    def test_creates_file(self, tmp_path: Path) -> None:
        from epigraph.analysis.visualise import gene_heatmap

        # Create mock gene features
        rng = np.random.default_rng(42)
        samples = [f"PD{i:06d}" for i in range(20)]
        genes = [f"GENE_{i}" for i in range(30)]
        data = {"gene": genes}
        for s in samples:
            data[s] = rng.uniform(0, 1, len(genes)).tolist()
        gene_df = pl.DataFrame(data)
        gene_path = tmp_path / "gene_features.parquet"
        gene_df.write_parquet(gene_path)

        # Create mock metadata
        meta = pl.DataFrame({
            "barcode": samples,
            "clinical_category": ["CRC"] * 10 + ["Control"] * 10,
        })
        meta_path = tmp_path / "metadata.parquet"
        meta.write_parquet(meta_path)

        output = tmp_path / "heatmap.png"
        gene_heatmap(str(gene_path), str(meta_path), str(output), n_top_genes=10)
        assert output.exists()
        assert output.stat().st_size > 1000
