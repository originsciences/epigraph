"""Tests for the identifiers module: barcode and clinical category normalisation."""

from __future__ import annotations

import pytest

from epigraph.common.identifiers import (
    CLINICAL_CATEGORIES,
    normalize_barcode,
    normalize_clinical_category,
    validate_barcode,
)

# =========================================================================
# normalize_barcode
# =========================================================================


class TestNormalizeBarcode:
    """Tests for ``normalize_barcode``."""

    def test_strips_whitespace(self) -> None:
        assert normalize_barcode("  SAMPLE_0001  ") == "SAMPLE_0001"

    def test_uppercases(self) -> None:
        assert normalize_barcode("sample_0001") == "SAMPLE_0001"

    def test_strips_and_uppercases(self) -> None:
        assert normalize_barcode(" sample_0001\t") == "SAMPLE_0001"

    def test_already_normalised(self) -> None:
        assert normalize_barcode("SAMPLE_0001") == "SAMPLE_0001"

    def test_mixed_alphanumeric(self) -> None:
        assert normalize_barcode("s-12345a") == "S-12345A"


# =========================================================================
# validate_barcode
# =========================================================================


class TestValidateBarcode:
    """Tests for ``validate_barcode``."""

    @pytest.mark.parametrize(
        "barcode",
        [
            "SAMPLE_0001",
            "S001",
            "ABC-123",
            "sample_with_suffix_1a",
            "X12345",
        ],
        ids=[
            "sample-with-underscore",
            "short-alpha-numeric",
            "with-hyphen",
            "long-mixed",
            "single-letter-prefix",
        ],
    )
    def test_valid_barcodes(self, barcode: str) -> None:
        assert validate_barcode(barcode) is True

    @pytest.mark.parametrize(
        "barcode",
        [
            "ab",           # too short (< 3 chars)
            "",             # empty
            "has space",    # whitespace not allowed
            "with.dot",     # dot not allowed
            "with/slash",   # slash not allowed
        ],
        ids=[
            "too-short",
            "empty",
            "whitespace",
            "dot",
            "slash",
        ],
    )
    def test_invalid_barcodes(self, barcode: str) -> None:
        assert validate_barcode(barcode) is False

    def test_normalised_then_validated(self) -> None:
        """Common workflow: normalize first, then validate."""
        raw = "  sample_0001  "
        normalised = normalize_barcode(raw)
        assert validate_barcode(normalised) is True


# =========================================================================
# normalize_clinical_category
# =========================================================================


class TestNormalizeClinicalCategory:
    """Tests for ``normalize_clinical_category``."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # CRC synonyms
            ("crc", "CRC"),
            ("CRC", "CRC"),
            ("colorectal cancer", "CRC"),
            ("colorectal", "CRC"),
            # Control synonyms
            ("control", "Control"),
            ("Control", "Control"),
            ("ctrl", "Control"),
            ("normal", "Control"),
            ("healthy", "Control"),
            # polyps synonyms
            ("polyps", "polyps"),
            ("POLYPS", "polyps"),
            ("polyp", "polyps"),
            ("adenoma", "polyps"),
            ("Polyps", "polyps"),
        ],
        ids=[
            "crc-lower",
            "CRC-upper",
            "colorectal-cancer",
            "colorectal",
            "control-lower",
            "Control-title",
            "ctrl",
            "normal",
            "healthy",
            "polyps-lower",
            "POLYPS-upper",
            "polyp-singular",
            "adenoma",
            "Polyps-title",
        ],
    )
    def test_known_synonyms(self, raw: str, expected: str) -> None:
        assert normalize_clinical_category(raw) == expected

    def test_whitespace_stripping(self) -> None:
        assert normalize_clinical_category("  CRC  ") == "CRC"
        assert normalize_clinical_category(" control ") == "Control"

    def test_unknown_category_returned_stripped(self) -> None:
        result = normalize_clinical_category("  unknown_thing  ")
        assert result == "unknown_thing"

    def test_unknown_category_case_preserved(self) -> None:
        """Unknown categories are returned with original casing (but stripped)."""
        result = normalize_clinical_category("SomethingElse")
        assert result == "SomethingElse"


# =========================================================================
# CLINICAL_CATEGORIES constant
# =========================================================================


class TestClinicalCategories:
    """Verify the canonical set of clinical categories."""

    def test_contains_expected(self) -> None:
        assert CLINICAL_CATEGORIES == frozenset({
            "CRC", "Control", "polyps", "HGD", "other_cancer", "excluded",
        })

    def test_is_frozenset(self) -> None:
        assert isinstance(CLINICAL_CATEGORIES, frozenset)
