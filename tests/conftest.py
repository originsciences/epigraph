"""Shared pytest fixtures for the epigraph test suite.

Provides mock data files (beta CSV, clinical XLSX) and DataFrames that
mirror the real project formats without requiring external data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# CpG and sample constants used across fixtures
# ---------------------------------------------------------------------------

MOCK_CPG_IDS: list[str] = [
    "chr1_100",
    "chr1_200",
    "chr1_500",
    "chr2_300",
    "chr2_800",
    "chr3_150",
    "chr3_900",
    "chrX_400",
    "chrX_700",
    "chrY_50",
]

MOCK_SAMPLE_IDS: list[str] = [
    "SAMPLE_0001",
    "SAMPLE_0002",
    "SAMPLE_0003",
    "SAMPLE_0004",
    "SAMPLE_0005",
]

# ---------------------------------------------------------------------------
# Beta matrix CSV fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_beta_csv(tmp_path: Path) -> Path:
    """Create a small mock beta matrix CSV (5 samples x 10 CpGs).

    Format mirrors the real file:
    - Empty first header cell (comma-delimited)
    - PD barcodes as row identifiers
    - Empty strings for missing values
    - Values in [0, 1] range
    """
    rng = np.random.default_rng(42)
    csv_path = tmp_path / "beta_matrix.csv"

    # Header: empty first cell, then CpG IDs
    header = "," + ",".join(MOCK_CPG_IDS)

    rows: list[str] = [header]
    for sample_id in MOCK_SAMPLE_IDS:
        values: list[str] = []
        for _ in MOCK_CPG_IDS:
            if rng.random() < 0.15:  # ~15% missingness
                values.append("")
            else:
                values.append(f"{rng.random():.6f}")
        rows.append(sample_id + "," + ",".join(values))

    csv_path.write_text("\n".join(rows) + "\n")
    return csv_path


@pytest.fixture()
def tmp_beta_csv_named_header(tmp_path: Path) -> Path:
    """Beta matrix CSV with ``sample_id`` as the explicit first header cell.

    The ``ChunkedCSVReader.iter_column_chunks`` method renames the empty
    first cell to ``sample_id`` internally then uses that name to select
    columns via PyArrow.  PyArrow reads the raw header from the file, so
    column-chunk iteration requires ``sample_id`` to appear literally in
    the CSV header.  This fixture provides that variant.
    """
    rng = np.random.default_rng(42)
    csv_path = tmp_path / "beta_matrix_named.csv"

    header = "sample_id," + ",".join(MOCK_CPG_IDS)

    rows: list[str] = [header]
    for sample_id in MOCK_SAMPLE_IDS:
        values: list[str] = []
        for _ in MOCK_CPG_IDS:
            if rng.random() < 0.15:
                values.append("")
            else:
                values.append(f"{rng.random():.6f}")
        rows.append(sample_id + "," + ",".join(values))

    csv_path.write_text("\n".join(rows) + "\n")
    return csv_path


# ---------------------------------------------------------------------------
# Clinical metadata XLSX fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_clinical_xlsx(tmp_path: Path) -> Path:
    """Create a mock Excel file with CLIN_CRC and CLIN_Control worksheets.

    Each sheet has ``barcode`` and ``clinical_category`` columns. A third
    non-matching sheet (``summary``) is included to verify filtering.
    """
    xlsx_path = tmp_path / "clinical_metadata.xlsx"

    crc_df = pd.DataFrame(
        {
            "barcode": ["SAMPLE_0001", "SAMPLE_0002"],
            "clinical_category": ["CRC", "CRC"],
        }
    )
    control_df = pd.DataFrame(
        {
            "barcode": ["SAMPLE_0003", "SAMPLE_0004", "SAMPLE_0005"],
            "clinical_category": ["Control", "Control", "Control"],
        }
    )
    summary_df = pd.DataFrame({"info": ["This sheet should be ignored"]})

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        crc_df.to_excel(writer, sheet_name="CLIN_CRC", index=False)
        control_df.to_excel(writer, sheet_name="CLIN_Control", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)

    return xlsx_path


# ---------------------------------------------------------------------------
# Gene records fixture (Polars DataFrame)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_genes_df() -> pl.DataFrame:
    """Mock gene records with known coordinates for overlap testing.

    Contains genes on chr1, chr2, and chrX with various strands to exercise
    promoter boundary calculations on both + and - strand genes.
    """
    return pl.DataFrame(
        {
            "gene_id": [
                "ENSG00000000001",
                "ENSG00000000002",
                "ENSG00000000003",
                "ENSG00000000004",
            ],
            "gene_symbol": ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
            "chrom": ["chr1", "chr1", "chr2", "chrX"],
            "start": [100, 5000, 200, 300],
            "end": [600, 8000, 1000, 900],
            "strand": ["+", "-", "+", "+"],
        }
    )


# ---------------------------------------------------------------------------
# Pandas gene DataFrame (for find_overlapping_genes which uses pandas)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_genes_pandas_df() -> pd.DataFrame:
    """Pandas version of gene records for ``find_overlapping_genes``."""
    return pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2", "chrX"],
            "start": [100, 5000, 200, 300],
            "end": [600, 8000, 1000, 900],
            "gene_symbol": ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
        }
    )


# ---------------------------------------------------------------------------
# CpG-to-gene mapping fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_cpg_gene_mapping() -> pl.DataFrame:
    """Mock CpG-to-gene mapping with various overlap types."""
    return pl.DataFrame(
        {
            "cpg_id": [
                "chr1_150",
                "chr1_5500",
                "chr2_500",
                "chrX_400",
                "chr3_99999",
            ],
            "chromosome": ["chr1", "chr1", "chr2", "chrX", "chr3"],
            "position": [150, 5500, 500, 400, 99999],
            "gene_id": [
                "ENSG00000000001",
                "ENSG00000000002",
                "ENSG00000000003",
                "ENSG00000000004",
                "",
            ],
            "gene_symbol": ["GENE_A", "GENE_B", "GENE_C", "GENE_D", ""],
            "overlap_type": [
                "gene_body",
                "gene_body",
                "gene_body",
                "gene_body",
                "intergenic",
            ],
        }
    )
