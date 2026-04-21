"""Common utilities for the epigraph package.

Re-exports frequently used functions and constants so callers can do::

    from epigraph.common import parse_cpg_id, get_logger, normalize_barcode
"""

from epigraph.common.genome_coords import (
    CHROMOSOME_ORDER,
    find_overlapping_genes,
    make_cpg_id,
    overlaps,
    parse_cpg_id,
    parse_cpg_id_fast,
    sort_cpg_ids,
)
from epigraph.common.identifiers import (
    CLINICAL_CATEGORIES,
    normalize_barcode,
    normalize_clinical_category,
    validate_barcode,
)
from epigraph.common.logging import get_logger

__all__ = [
    # genome_coords
    "CHROMOSOME_ORDER",
    "find_overlapping_genes",
    "make_cpg_id",
    "overlaps",
    "parse_cpg_id",
    "parse_cpg_id_fast",
    "sort_cpg_ids",
    # identifiers
    "CLINICAL_CATEGORIES",
    "normalize_barcode",
    "normalize_clinical_category",
    "validate_barcode",
    # logging
    "get_logger",
]
