"""Smoke tests for ``epigraph.analysis.generate_report``.

Covers ``_load_comparison``, ``_load_enrichment``, and the end-to-end
``build_report`` function on minimal synthetic inputs.  Previously 0%
coverage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from epigraph.analysis.generate_report import (  # noqa: E402
    _load_comparison,
    _load_enrichment,
    build_report,
)


def _write_comparison(path: Path) -> None:
    pl.DataFrame(
        {
            "feature": ["GENE_A", "GENE_B", "GENE_C"],
            "gene_symbol": ["GENE_A", "GENE_B", "GENE_C"],
            "p_value": [0.001, 0.04, 0.5],
            "q_value": [0.003, 0.06, 0.6],
            "cohens_d": [1.3, 0.5, 0.1],
            "delta_mean": [0.3, 0.1, 0.02],
            "significant": [True, False, False],
        }
    ).write_parquet(path)


def _write_enrichment(path: Path, include_name: str = "reactome") -> None:
    pl.DataFrame(
        {
            "pathway": ["R-HSA-1", "R-HSA-2"],
            "q_value": [0.01, 0.3],
            "odds_ratio": [3.0, 1.2],
            "n_overlap": [5, 2],
            "significant": [True, False],
            "p_value": [0.001, 0.1],
        }
    ).write_parquet(path / f"CRC_vs_Control_{include_name}_fisher.parquet")


class TestLoadComparison:
    def test_reports_significant_and_top_hits(self, tmp_path: Path) -> None:
        p = tmp_path / "CRC_vs_Control.parquet"
        _write_comparison(p)
        info = _load_comparison(p)
        assert info["label"] == "CRC_vs_Control"
        assert info["n_genes"] == 3
        assert info["n_significant"] == 1
        assert info["n_nominal"] == 2
        assert len(info["top_hits"]) >= 1
        assert info["top_hits"][0]["gene"] == "GENE_A"


class TestLoadEnrichment:
    def test_returns_significant_pathways(self, tmp_path: Path) -> None:
        enr = tmp_path / "enr"
        enr.mkdir()
        _write_enrichment(enr)
        path = enr / "CRC_vs_Control_reactome_fisher.parquet"
        info = _load_enrichment(path)
        assert info["n_tested"] == 2
        assert info["n_significant"] == 1
        assert len(info["pathways"]) == 1

    def test_joins_pathway_names_when_provided(self, tmp_path: Path) -> None:
        enr_dir = tmp_path / "enr"
        enr_dir.mkdir()
        _write_enrichment(enr_dir)
        enr_path = enr_dir / "CRC_vs_Control_reactome_fisher.parquet"

        names_path = tmp_path / "names.parquet"
        pl.DataFrame(
            {
                "pathway_id": ["R-HSA-1", "R-HSA-2"],
                "pathway_name": ["DNA Repair", "Cell Cycle"],
            }
        ).write_parquet(names_path)

        info = _load_enrichment(enr_path, names_path)
        # Only significant pathway (R-HSA-1) is returned
        assert info["pathways"][0]["name"] == "DNA Repair"


class TestBuildReportEndToEnd:
    def test_writes_docx(self, tmp_path: Path) -> None:
        comp_dir = tmp_path / "comparisons"
        comp_dir.mkdir()
        _write_comparison(comp_dir / "CRC_vs_Control.parquet")

        enr_dir = tmp_path / "enrichment"
        enr_dir.mkdir()
        _write_enrichment(enr_dir)

        hms_dir = tmp_path / "hms"
        hms_dir.mkdir()
        pl.DataFrame(
            {
                "sample_id": ["S1", "S2", "S3"],
                "hms_count": [10, 20, 30],
                "clinical_category": ["Control", "CRC", "CRC"],
            }
        ).write_parquet(hms_dir / "hms_scores_q0_99.parquet")

        ext = tmp_path / "external"
        ext.mkdir()
        pl.DataFrame(
            {"pathway_id": ["R-HSA-1"], "pathway_name": ["DNA Repair"]}
        ).write_parquet(ext / "reactome_pathways.parquet")

        mapping = tmp_path / "mapping.parquet"
        pl.DataFrame(
            {
                "cpg_id": ["c1", "c2"],
                "gene_symbol": ["GENE_A", "GENE_B"],
                "overlap_type": ["gene_body", "promoter"],
            }
        ).write_parquet(mapping)

        island = tmp_path / "island.parquet"
        pl.DataFrame({"context": ["island", "open_sea", "shore"]}).write_parquet(island)

        clinical = tmp_path / "clinical.parquet"
        pl.DataFrame(
            {
                "barcode": ["S1", "S2", "S3"],
                "clinical_category": ["Control", "CRC", "CRC"],
            }
        ).write_parquet(clinical)

        # One small figure so the figures-section code path runs
        figures = tmp_path / "figures"
        figures.mkdir()
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        fig.savefig(figures / "volcano_CRC_vs_Control.png")
        plt.close(fig)

        out_path = tmp_path / "report.docx"
        build_report(
            output_path=out_path,
            comparisons_dir=comp_dir,
            enrichment_dir=enr_dir,
            hms_dir=hms_dir,
            external_dir=ext,
            mapping_path=mapping,
            island_path=island,
            figures_dir=figures,
            clinical_path=clinical,
        )

        assert out_path.exists()
        # Must be a real docx (ZIP archive) of non-trivial size
        assert out_path.stat().st_size > 10_000
        assert out_path.read_bytes()[:2] == b"PK"
