"""Smoke tests for ``epigraph.db_build.dataset_stats``.

Covers the four stat-computation helpers and the CLI entry point, on
minimal synthetic fixtures.  Previously 0% coverage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from click.testing import CliRunner

from epigraph.db_build.dataset_stats import (
    _annotation_stats,
    _beta_stats_from_csv_header,
    _beta_stats_from_parquet,
    _clinical_stats,
    main,
)


def _write_beta_parquet(path: Path, n_samples: int = 4, missing_mask: list[list[bool]] | None = None) -> None:
    """Write a deterministic beta parquet to ``path``."""
    cpg_ids = ["chr1_100", "chr1_200", "chr2_150"]
    data: dict[str, list] = {"sample_id": [f"S{i}" for i in range(n_samples)]}
    rng = np.random.default_rng(7)
    for j, cpg in enumerate(cpg_ids):
        col: list[float | None] = []
        for i in range(n_samples):
            if missing_mask and missing_mask[i][j]:
                col.append(None)
            else:
                col.append(float(rng.random()))
        data[cpg] = col
    pl.DataFrame(data).write_parquet(path)


class TestBetaStatsFromParquet:
    def test_counts_and_coverage(self, tmp_path: Path) -> None:
        beta = tmp_path / "beta.parquet"
        # 4 samples × 3 CpGs; sample 0 missing col 0, sample 1 missing col 2
        _write_beta_parquet(
            beta,
            n_samples=4,
            missing_mask=[
                [True, False, False],
                [False, False, True],
                [False, False, False],
                [False, False, False],
            ],
        )
        stats = _beta_stats_from_parquet(beta)
        assert stats["n_samples"] == 4
        assert stats["n_cpgs"] == 3
        assert stats["total_cells"] == 12
        assert stats["total_nulls"] == 2
        assert stats["null_fraction"] == 2 / 12
        # chr1_100 coverage = 3/4 = 0.75, chr1_200 = 1.0, chr2_150 = 0.75
        assert stats["cpg_coverage_gte_95pct"] == 1
        assert stats["cpg_coverage_gte_80pct"] == 1  # only chr1_200 >= 0.8
        assert stats["cpg_coverage_gte_50pct"] == 3
        assert set(stats["chrom_counts"].keys()) == {"chr1", "chr2"}
        assert stats["chrom_counts"]["chr1"] == 2
        assert stats["chrom_counts"]["chr2"] == 1
        # value stats are computed; just verify they are within sane bounds
        assert 0.0 <= stats["beta_mean"] <= 1.0

    def test_empty_cells_branch(self, tmp_path: Path) -> None:
        """Zero samples: total_cells = 0 must not divide by zero."""
        beta = tmp_path / "beta.parquet"
        pl.DataFrame({"sample_id": [], "chr1_100": [], "chr2_200": []}).write_parquet(beta)
        stats = _beta_stats_from_parquet(beta)
        assert stats["n_samples"] == 0
        assert stats["null_fraction"] == 0


class TestBetaStatsFromCsvHeader:
    def test_counts_samples_and_chroms(self, tmp_path: Path) -> None:
        csv = tmp_path / "beta.csv"
        csv.write_text(
            ",chr1_100,chr1_200,chr2_300\n"
            "S1,0.1,0.2,0.3\n"
            "S2,0.4,0.5,0.6\n"
        )
        stats = _beta_stats_from_csv_header(csv)
        assert stats["n_samples"] == 2
        assert stats["n_cpgs"] == 3
        assert stats["total_cells"] == 6
        assert stats["chrom_counts"] == {"chr1": 2, "chr2": 1}


class TestClinicalStats:
    def test_category_counts(self, tmp_path: Path) -> None:
        p = tmp_path / "clinical.parquet"
        pl.DataFrame(
            {
                "barcode": ["S1", "S2", "S3", "S4"],
                "clinical_category": ["CRC", "CRC", "Control", "polyps"],
            }
        ).write_parquet(p)
        stats = _clinical_stats(p)
        assert stats["n_samples"] == 4
        assert stats["n_categories"] == 3
        assert stats["categories"] == {"CRC": 2, "Control": 1, "polyps": 1}
        assert "barcode" in stats["columns"]


class TestAnnotationStats:
    def test_reports_missing_files(self, tmp_path: Path) -> None:
        stats = _annotation_stats(tmp_path)  # empty directory
        for key in ["gencode_gtf", "goa_gaf", "reactome_pathways", "cpg_islands"]:
            assert stats[key] == {"present": False}

    def test_reports_present_files_with_sizes(self, tmp_path: Path) -> None:
        (tmp_path / "gencode.v45.annotation.gtf.gz").write_bytes(b"x" * 1024)
        stats = _annotation_stats(tmp_path)
        assert stats["gencode_gtf"]["present"] is True
        assert "size_mb" in stats["gencode_gtf"]

    def test_reports_gene_count_and_biotypes_when_parquet_present(self, tmp_path: Path) -> None:
        genes_pq = tmp_path / "genes.parquet"
        pl.DataFrame(
            {
                "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
                "biotype": ["protein_coding", "protein_coding", "lncRNA"],
            }
        ).write_parquet(genes_pq)
        stats = _annotation_stats(tmp_path)
        assert stats["n_genes"] == 3
        assert stats["top_biotypes"] == {"protein_coding": 2, "lncRNA": 1}


class TestCLISmoke:
    def test_runs_on_synthetic_inputs(self, tmp_path: Path) -> None:
        beta = tmp_path / "beta.parquet"
        clinical = tmp_path / "clinical.parquet"
        external = tmp_path / "external"
        external.mkdir()

        _write_beta_parquet(beta, n_samples=3)
        pl.DataFrame(
            {"barcode": ["S0", "S1", "S2"], "clinical_category": ["CRC", "Control", "polyps"]}
        ).write_parquet(clinical)

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--beta-parquet", str(beta),
                "--clinical", str(clinical),
                "--external-dir", str(external),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "DATASET STATISTICS" in result.output
        assert "Samples:" in result.output
