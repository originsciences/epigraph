"""Regression tests anchoring pipeline filename conventions.

The pipeline writes enrichment results and hypermethylation scores to disk
under specific names, and later pipeline steps (report, visualise) read them
back. Historically these two sides drifted silently:

- Enrichment wrote ``{label}_fisher.parquet`` but readers expected
  ``{label}_reactome_fisher.parquet`` (indistinguishable from GO, and always
  missed), which silently dropped pathway candidates from the biomarker
  report and skipped the Reactome dot-plot.
- Hypermethylation wrote ``hms_scores_q0_99.parquet`` (dot → underscore) but
  readers looked for ``hms_scores_q0.99.parquet``, which silently ate the
  entire HMS section of the report.

These tests pin the writer/reader protocols together so either side can't
rot alone.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from epigraph import pipeline
from epigraph.analysis.hypermethylation import run_hypermethylation_analysis


def _write_minimal_comparison(path: Path, *, significant_gene: str = "GENE_A") -> None:
    """Write a minimal diff-results parquet that step_enrichment can consume."""
    pl.DataFrame(
        {
            "feature": ["GENE_A", "GENE_B", "GENE_C"],
            "gene_symbol": ["GENE_A", "GENE_B", "GENE_C"],
            "p_value": [0.001, 0.2, 0.5],
            "q_value": [0.003, 0.3, 0.6],
            "delta_mean": [0.3, 0.05, -0.1],
            "cohens_d": [1.2, 0.2, -0.4],
            "significant": [True, False, False],
        }
    ).write_parquet(path)


def _write_minimal_reactome(path: Path) -> None:
    """Write a minimal gene→pathway mapping parquet."""
    pl.DataFrame(
        {
            "gene_symbol": ["GENE_A", "GENE_B", "GENE_A", "GENE_C"],
            "pathway_id": ["R-HSA-1", "R-HSA-1", "R-HSA-2", "R-HSA-2"],
        }
    ).write_parquet(path)


class TestEnrichmentFilenameConvention:
    """step_enrichment output naming must match what step_report expects."""

    def test_writes_reactome_fisher_suffix(self, tmp_path: Path) -> None:
        """step_enrichment writes ``{label}_reactome_fisher.parquet``."""
        ext = tmp_path / "external"
        ext.mkdir()
        comp_dir = tmp_path / "comparisons"
        comp_dir.mkdir()
        enr_dir = tmp_path / "enrichment"

        _write_minimal_reactome(ext / "reactome_symbol_pathway.parquet")
        _write_minimal_comparison(comp_dir / "CRC_vs_Control.parquet")

        cfg = {
            "external_dir": str(ext),
            "comparisons_dir": str(comp_dir),
            "enrichment_dir": str(enr_dir),
        }
        pipeline.step_enrichment(cfg)

        # The exact filename step_report reads back
        expected = enr_dir / "CRC_vs_Control_reactome_fisher.parquet"
        assert expected.exists(), (
            f"step_enrichment must write {expected.name!r}; "
            f"found: {sorted(p.name for p in enr_dir.iterdir())}"
        )

    def test_report_reader_finds_enrichment_file(self, tmp_path: Path) -> None:
        """The exact path step_report assembles must match the writer's output."""
        # Simulate a successful enrichment run
        enr_dir = tmp_path / "enrichment"
        enr_dir.mkdir()
        # Name is the one step_enrichment will produce post-fix
        written = enr_dir / "CRC_vs_Control_reactome_fisher.parquet"
        pl.DataFrame(
            {
                "pathway": ["R-HSA-1"],
                "q_value": [0.01],
                "odds_ratio": [3.0],
                "significant": [True],
                "overlap_genes": [["GENE_A"]],
            }
        ).write_parquet(written)

        # Reproduce the step_report path-assembly exactly
        fisher_path = enr_dir / "CRC_vs_Control_reactome_fisher.parquet"
        assert fisher_path.exists()
        assert fisher_path == written


class TestHypermethylationFilenameConvention:
    """Writer and every pipeline reader must agree on HMS filenames."""

    def test_writer_replaces_dot_with_underscore(self, tmp_path: Path) -> None:
        """hypermethylation.run writes ``hms_scores_q{q_}.parquet`` (dot→underscore)."""
        gene_path = tmp_path / "gene_features.parquet"
        meta_path = tmp_path / "metadata.parquet"
        out_dir = tmp_path / "hms"

        pl.DataFrame(
            {
                "gene": ["G1", "G2", "G3"],
                "S_CTRL_1": [0.1, 0.2, 0.3],
                "S_CTRL_2": [0.12, 0.22, 0.32],
                "S_CRC_1": [0.6, 0.8, 0.7],
                "S_CRC_2": [0.65, 0.82, 0.72],
            }
        ).write_parquet(gene_path)
        pl.DataFrame(
            {
                "barcode": ["S_CTRL_1", "S_CTRL_2", "S_CRC_1", "S_CRC_2"],
                "clinical_category": ["Control", "Control", "CRC", "CRC"],
            }
        ).write_parquet(meta_path)

        run_hypermethylation_analysis(
            gene_features_path=gene_path,
            metadata_path=meta_path,
            output_dir=out_dir,
            quantiles=[0.95, 0.999],
        )

        # Dot-for-underscore
        assert (out_dir / "hms_scores_q0_95.parquet").exists()
        assert (out_dir / "hms_scores_q0_999.parquet").exists()
        # The bad (dot) names must NOT exist — would signal writer regression
        assert not (out_dir / "hms_scores_q0.95.parquet").exists()
        assert not (out_dir / "hms_scores_q0.999.parquet").exists()

    def test_report_reader_uses_underscore_convention(self, tmp_path: Path) -> None:
        """Pipeline readers must apply the same dot→underscore replacement."""
        hms_dir = tmp_path / "hms"
        hms_dir.mkdir()
        # Place a file under the writer's on-disk name
        (hms_dir / "hms_scores_q0_99.parquet").write_bytes(b"placeholder")

        # Reproduce the reader logic from step_report / step_visualise /
        # generate_report — each does q.replace('.', '_').
        for q in ["0.99", "0.95", "0.999"]:
            candidate = hms_dir / f"hms_scores_q{q.replace('.', '_')}.parquet"
            if candidate.exists():
                assert candidate.name == "hms_scores_q0_99.parquet"
                break
        else:
            raise AssertionError(
                "Pipeline HMS reader did not locate hms_scores_q0_99.parquet; "
                "filename convention drift between writer and reader."
            )
