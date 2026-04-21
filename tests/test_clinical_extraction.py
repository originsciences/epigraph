"""Tests for clinical metadata extraction from Excel workbooks.

Exercises the sheet discovery, barcode extraction, merging, deduplication,
and non-CLIN sheet filtering logic in ``load_clinical_metadata``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from epigraph.db_build.load_clinical_metadata import (
    discover_clinical_sheets,
    merge_clinical_sheets,
    parse_clinical_sheet,
)

# =========================================================================
# discover_clinical_sheets
# =========================================================================


class TestDiscoverClinicalSheets:
    """Tests for ``discover_clinical_sheets``."""

    def test_finds_clin_sheets(self, tmp_clinical_xlsx: Path) -> None:
        sheets = discover_clinical_sheets(tmp_clinical_xlsx)
        assert "CLIN_CRC" in sheets
        assert "CLIN_Control" in sheets

    def test_ignores_non_clin_sheets(self, tmp_clinical_xlsx: Path) -> None:
        sheets = discover_clinical_sheets(tmp_clinical_xlsx)
        assert "summary" not in sheets

    def test_returns_sorted(self, tmp_clinical_xlsx: Path) -> None:
        sheets = discover_clinical_sheets(tmp_clinical_xlsx)
        assert sheets == sorted(sheets)

    def test_custom_pattern(self, tmp_clinical_xlsx: Path) -> None:
        sheets = discover_clinical_sheets(tmp_clinical_xlsx, pattern="summary")
        assert sheets == ["summary"]

    def test_no_matching_sheets(self, tmp_clinical_xlsx: Path) -> None:
        sheets = discover_clinical_sheets(
            tmp_clinical_xlsx, pattern="NONEXISTENT_*"
        )
        assert sheets == []


# =========================================================================
# parse_clinical_sheet
# =========================================================================


class TestParseClinicalSheet:
    """Tests for ``parse_clinical_sheet``."""

    def test_parses_crc_sheet(self, tmp_clinical_xlsx: Path) -> None:
        df = parse_clinical_sheet(tmp_clinical_xlsx, "CLIN_CRC")
        assert df is not None
        assert len(df) == 2
        assert "barcode" in df.columns
        assert "clinical_category" in df.columns
        assert "source_sheet" in df.columns

    def test_barcodes_normalised(self, tmp_clinical_xlsx: Path) -> None:
        df = parse_clinical_sheet(tmp_clinical_xlsx, "CLIN_CRC")
        assert df is not None
        barcodes = df["barcode"].to_list()
        # normalize_barcode uppercases, so all should be uppercase
        for bc in barcodes:
            assert bc == bc.strip().upper()

    def test_source_sheet_column_populated(self, tmp_clinical_xlsx: Path) -> None:
        df = parse_clinical_sheet(tmp_clinical_xlsx, "CLIN_CRC")
        assert df is not None
        assert all(s == "CLIN_CRC" for s in df["source_sheet"].to_list())

    def test_different_column_names(self, tmp_path: Path) -> None:
        """Sheets with alternative column names (e.g. ``Sample_Barcode``)."""
        xlsx_path = tmp_path / "alt_columns.xlsx"
        alt_df = pd.DataFrame(
            {
                "Sample_Barcode": ["SAMPLE_0006", "SAMPLE_0007"],
                "diagnosis": ["crc", "control"],
            }
        )
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            alt_df.to_excel(writer, sheet_name="CLIN_alt", index=False)

        result = parse_clinical_sheet(xlsx_path, "CLIN_alt")
        assert result is not None
        assert len(result) == 2
        # Category normalisation should have been applied
        categories = set(result["clinical_category"].to_list())
        assert "CRC" in categories
        assert "Control" in categories

    def test_missing_barcode_column_returns_none(self, tmp_path: Path) -> None:
        xlsx_path = tmp_path / "no_barcode.xlsx"
        bad_df = pd.DataFrame({"unrelated": [1, 2, 3]})
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            bad_df.to_excel(writer, sheet_name="CLIN_bad", index=False)

        result = parse_clinical_sheet(xlsx_path, "CLIN_bad")
        assert result is None


# =========================================================================
# merge_clinical_sheets
# =========================================================================


class TestMergeClinicalSheets:
    """Tests for ``merge_clinical_sheets``."""

    def test_merges_all_clin_sheets(self, tmp_clinical_xlsx: Path) -> None:
        result = merge_clinical_sheets(tmp_clinical_xlsx, exclude_samples_path=None)
        # 2 CRC + 3 Control = 5 unique barcodes
        assert len(result) == 5

    def test_deduplication_by_barcode(self, tmp_path: Path) -> None:
        """When a barcode appears in multiple sheets, keep first occurrence."""
        xlsx_path = tmp_path / "dup_barcodes.xlsx"
        sheet1 = pd.DataFrame(
            {
                "barcode": ["SAMPLE_0001", "SAMPLE_0002"],
                "clinical_category": ["CRC", "CRC"],
            }
        )
        sheet2 = pd.DataFrame(
            {
                "barcode": ["SAMPLE_0001", "SAMPLE_0003"],
                "clinical_category": ["Control", "Control"],
            }
        )
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            sheet1.to_excel(writer, sheet_name="CLIN_sheet1", index=False)
            sheet2.to_excel(writer, sheet_name="CLIN_sheet2", index=False)

        result = merge_clinical_sheets(xlsx_path, exclude_samples_path=None)
        # SAMPLE_0001 appears in both but should only appear once
        barcodes = result["barcode"].to_list()
        assert barcodes.count("SAMPLE_0001") == 1
        assert len(result) == 3  # SAMPLE_0001, SAMPLE_0002, SAMPLE_0003

    def test_output_schema(self, tmp_clinical_xlsx: Path) -> None:
        result = merge_clinical_sheets(tmp_clinical_xlsx, exclude_samples_path=None)
        assert set(result.columns) == {"barcode", "clinical_category", "source_sheet"}

    def test_no_matching_sheets_returns_empty(self, tmp_path: Path) -> None:
        xlsx_path = tmp_path / "no_clin.xlsx"
        df = pd.DataFrame({"data": [1]})
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="other", index=False)

        result = merge_clinical_sheets(xlsx_path, exclude_samples_path=None)
        assert len(result) == 0
        assert "barcode" in result.columns

    def test_non_clin_sheets_excluded(self, tmp_clinical_xlsx: Path) -> None:
        """The ``summary`` sheet should not contribute any rows."""
        result = merge_clinical_sheets(tmp_clinical_xlsx, exclude_samples_path=None)
        source_sheets = set(result["source_sheet"].to_list())
        assert "summary" not in source_sheets
