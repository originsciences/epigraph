"""Sample barcode and clinical category normalisation utilities.

Sample barcodes are treated as opaque alphanumeric identifiers (e.g.
``SAMPLE_0001``). Clinical categories map to a canonical set suitable
for CRC vs. Control vs. polyps comparisons.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLINICAL_CATEGORIES: frozenset[str] = frozenset({
    "CRC", "Control", "polyps", "HGD", "other_cancer", "excluded",
})
"""Canonical set of clinical categories used throughout the project.

- CRC: confirmed colorectal cancer
- Control: healthy / normal colonoscopy
- polyps: adenoma / non-HGD polyps
- HGD: high-grade dysplasia (kept separate from polyps for analysis)
- other_cancer: non-CRC cancers
- excluded: QC-failed or otherwise non-analysable
"""

_BARCODE_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{3,}$")
"""Regex for valid sample barcodes: alphanumeric with underscores/hyphens, >=3 chars."""

# Map common synonyms / variants to the canonical form.
_CATEGORY_SYNONYMS: dict[str, str] = {
    # CRC
    "crc": "CRC",
    "colorectal cancer": "CRC",
    "colorectal": "CRC",
    # Control
    "control": "Control",
    "ctrl": "Control",
    "normal": "Control",
    "healthy": "Control",
    # polyps (non-HGD)
    "polyps": "polyps",
    "polyp": "polyps",
    "adenoma": "polyps",
    # HGD — kept separate from polyps for analysis
    "hgd": "HGD",
    # Non-CRC cancers
    "other cancer": "other_cancer",
    # Excluded / non-analysable
    "excluded": "excluded",
    "pending": "excluded",
}

# ---------------------------------------------------------------------------
# Barcode helpers
# ---------------------------------------------------------------------------


def normalize_barcode(barcode: str) -> str:
    """Normalise a sample barcode by stripping whitespace and upper-casing.

    Args:
        barcode: Raw barcode string from any source.

    Returns:
        Cleaned barcode string ready for lookup / comparison.
    """
    return barcode.strip().upper()


def validate_barcode(barcode: str) -> bool:
    """Check whether *barcode* is a plausible sample identifier.

    Validation is applied **after** normalisation, so the caller should pass
    the output of :func:`normalize_barcode` for consistent results.

    Args:
        barcode: Barcode string (ideally already normalised).

    Returns:
        ``True`` if the barcode is a non-empty alphanumeric token of length
        >= 3 (underscores and hyphens permitted).
    """
    return _BARCODE_PATTERN.match(barcode) is not None


# ---------------------------------------------------------------------------
# Clinical category helpers
# ---------------------------------------------------------------------------


def normalize_clinical_category(category: str) -> str:
    """Map a clinical category string to its canonical form.

    The lookup is case-insensitive and strips surrounding whitespace.  Unknown
    values are returned unchanged (but stripped) so downstream code can decide
    how to handle them.

    Args:
        category: Raw category value, e.g. ``"crc"``, ``"Control "``.

    Returns:
        Canonical category string from :data:`CLINICAL_CATEGORIES`, or the
        stripped input if no synonym mapping exists.

    Examples:
        >>> normalize_clinical_category("crc")
        'CRC'
        >>> normalize_clinical_category(" Control ")
        'Control'
        >>> normalize_clinical_category("polyp")
        'polyps'
    """
    cleaned = category.strip()
    return _CATEGORY_SYNONYMS.get(cleaned.lower(), cleaned)
