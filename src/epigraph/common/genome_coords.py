"""Genomic coordinate utilities for CpG site identifiers.

CpG column names in the beta matrix follow the format ``chr{N}_{position}``
where N is the chromosome number (1-22, X, Y, M) and position is 1-based.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHROM_NAMES: list[str] = [
    *(f"chr{i}" for i in range(1, 23)),
    "chrX",
    "chrY",
    "chrM",
]

CHROMOSOME_ORDER: dict[str, int] = {name: idx for idx, name in enumerate(_CHROM_NAMES)}
"""Mapping from chromosome name to sort order (chr1=0 ... chrM=24)."""

_CPG_PATTERN = re.compile(r"^(chr(?:[1-9]|1[0-9]|2[0-2]|X|Y|M))_(\d+)$")

# ---------------------------------------------------------------------------
# CpG ID parsing
# ---------------------------------------------------------------------------


def parse_cpg_id_fast(cpg_id: str) -> tuple[str, int]:
    """Fast CpG ID parser -- no validation, for inner loops over millions of IDs.

    Uses a simple ``rfind("_")`` split instead of regex matching.  Suitable
    for batch processing where IDs have already been validated upstream.

    Args:
        cpg_id: CpG column name, e.g. ``"chr1_10469"``.

    Returns:
        Tuple of (chromosome, position) where position is an integer.
    """
    i = cpg_id.rfind("_")
    return cpg_id[:i], int(cpg_id[i + 1 :])


def parse_cpg_id(cpg_id: str) -> tuple[str, int]:
    """Parse a CpG identifier into chromosome and position.

    Args:
        cpg_id: CpG column name, e.g. ``"chr1_10469"``.

    Returns:
        Tuple of (chromosome, position) where position is a 1-based integer.

    Raises:
        ValueError: If *cpg_id* does not match the expected format.
    """
    m = _CPG_PATTERN.match(cpg_id)
    if m is None:
        raise ValueError(
            f"Invalid CpG identifier {cpg_id!r}. "
            "Expected format 'chr{{N}}_{{position}}' (e.g. 'chr1_10469')."
        )
    return m.group(1), int(m.group(2))


def make_cpg_id(chrom: str, pos: int) -> str:
    """Create a CpG identifier from chromosome and position.

    Args:
        chrom: Chromosome name, e.g. ``"chr1"``.
        pos: 1-based genomic position.

    Returns:
        CpG identifier string.

    Raises:
        ValueError: If *chrom* is not a recognised chromosome name or *pos* < 1.
    """
    if chrom not in CHROMOSOME_ORDER:
        raise ValueError(
            f"Unknown chromosome {chrom!r}. "
            f"Must be one of {', '.join(_CHROM_NAMES)}."
        )
    if pos < 1:
        raise ValueError(f"Position must be >= 1, got {pos}.")
    return f"{chrom}_{pos}"


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def sort_cpg_ids(ids: list[str]) -> list[str]:
    """Sort CpG identifiers by chromosome order then by position.

    Args:
        ids: List of CpG identifier strings.

    Returns:
        New sorted list. Invalid identifiers are placed at the end, sorted
        lexicographically.
    """

    def _sort_key(cpg_id: str) -> tuple[int, int, str]:
        try:
            chrom, pos = parse_cpg_id(cpg_id)
            return (0, CHROMOSOME_ORDER[chrom], pos)  # type: ignore[return-value]
        except ValueError:
            # Push invalid IDs to the end, sorted lexicographically.
            return (1, 0, cpg_id)  # type: ignore[return-value]

    return sorted(ids, key=_sort_key)


# ---------------------------------------------------------------------------
# Interval operations
# ---------------------------------------------------------------------------


def overlaps(start1: int, end1: int, start2: int, end2: int) -> bool:
    """Check whether two half-open intervals ``[start, end)`` overlap.

    Args:
        start1: Start of the first interval (inclusive).
        end1: End of the first interval (exclusive).
        start2: Start of the second interval (inclusive).
        end2: End of the second interval (exclusive).

    Returns:
        ``True`` if the intervals share at least one base position.
    """
    return start1 < end2 and start2 < end1


def find_overlapping_genes(
    cpg_chrom: str,
    cpg_pos: int,
    genes_df: pd.DataFrame,
    chrom_col: str = "chrom",
    start_col: str = "start",
    end_col: str = "end",
) -> pd.DataFrame:
    """Find genes whose genomic interval contains a CpG position.

    The CpG is treated as a single-base interval ``[cpg_pos, cpg_pos + 1)``.

    Args:
        cpg_chrom: Chromosome of the CpG site, e.g. ``"chr1"``.
        cpg_pos: 1-based position of the CpG site.
        genes_df: DataFrame with at least *chrom_col*, *start_col*, and
            *end_col* columns.  Coordinates are expected to be 0-based
            half-open (BED convention).
        chrom_col: Name of the chromosome column.
        start_col: Name of the start-position column.
        end_col: Name of the end-position column.

    Returns:
        Filtered DataFrame containing only rows that overlap the CpG position.

    Raises:
        KeyError: If required columns are missing from *genes_df*.
    """
    missing = {chrom_col, start_col, end_col} - set(genes_df.columns)
    if missing:
        raise KeyError(f"Missing required columns in genes_df: {missing}")

    mask = (
        (genes_df[chrom_col] == cpg_chrom)
        & (genes_df[start_col] < cpg_pos + 1)
        & (genes_df[end_col] > cpg_pos)
    )
    return genes_df.loc[mask].copy()
