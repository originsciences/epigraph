"""Parse clinical metadata from an Excel workbook.

The workbook is expected to contain multiple worksheets matching the
pattern ``CLIN_*``. Each sheet has barcode and clinical-category columns
(column names may vary across sheets). This module merges all sheets,
deduplicates by barcode, normalizes values, and writes a single Parquet
file.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any

import click
import openpyxl
import polars as pl
import yaml

from epigraph.common.identifiers import normalize_barcode, normalize_clinical_category
from epigraph.common.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

# Common column name variations for barcode and clinical category
_BARCODE_ALIASES: list[str] = [
    "barcode",
    "Barcode",
    "BARCODE",
    "sample_barcode",
    "Sample_Barcode",
    "sample_id",
    "Sample_ID",
    "SampleID",
    "patient_id",
    "PD_Barcode",
    "pd_barcode",
]

_CATEGORY_ALIASES: list[str] = [
    "clinical_category",
    "Clinical_Category",
    "CLINICAL_CATEGORY",
    "category",
    "Category",
    "diagnosis",
    "Diagnosis",
    "condition",
    "group",
    "Group",
    "disease_status",
    "status",
]


def _load_settings() -> dict[str, Any]:
    """Load settings.yaml if it exists."""
    settings_path = Path("config/settings.yaml")
    if settings_path.exists():
        with open(settings_path) as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _resolve_path(raw: str) -> str:
    """Resolve a shell-variable-style path like ``${VAR:-default}``."""
    import os

    m = re.match(r"\$\{([^:}]+):-(.+)\}", raw)
    if m:
        return os.environ.get(m.group(1), m.group(2))
    return raw


# ---------------------------------------------------------------------------
# Sample exclusion
# ---------------------------------------------------------------------------


def load_excluded_barcodes_from_sheet(
    xlsx_path: Path,
    sheet_name: str,
    barcode_column: str = "sample_barcode",
) -> set[str]:
    """Load barcodes from a dedicated "excluded samples" worksheet.

    Use this to read a list of QC-failed or otherwise excluded samples from
    a named worksheet inside the same workbook that carries the clinical
    metadata.

    Args:
        xlsx_path: Path to the clinical metadata Excel workbook.
        sheet_name: Name of the worksheet containing excluded barcodes.
        barcode_column: Column in the worksheet containing sample barcodes.

    Returns:
        Set of normalised barcode strings to exclude.
    """
    import pandas as pd

    try:
        pdf = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    except (ValueError, KeyError):
        log.warning("exclusion_sheet_not_found", sheet=sheet_name)
        return set()

    if barcode_column not in pdf.columns:
        for col in pdf.columns:
            if col.strip().lower() == barcode_column.lower():
                barcode_column = col
                break
        else:
            log.warning("exclusion_barcode_column_not_found", columns=list(pdf.columns))
            return set()

    barcodes = {
        normalize_barcode(str(b))
        for b in pdf[barcode_column].dropna()
        if str(b).strip() and str(b).strip().lower() != "nan"
    }
    log.info("excluded_barcodes_loaded_from_sheet", n=len(barcodes), sheet=sheet_name)
    return barcodes


def load_excluded_barcodes_from_file(
    exclude_path: Path | str,
) -> set[str]:
    """Load excluded barcodes from a text file (one barcode per line).

    Args:
        exclude_path: Path to the exclude file.

    Returns:
        Set of normalised barcode strings to exclude. Empty if the file
        does not exist.
    """
    exclude_path = Path(exclude_path)
    if not exclude_path.exists():
        log.info("exclude_file_not_found", path=str(exclude_path))
        return set()

    barcodes = set()
    with open(exclude_path) as fh:
        for line in fh:
            barcode = line.strip()
            if barcode:
                barcodes.add(normalize_barcode(barcode))

    log.info("excluded_barcodes_loaded_from_file", n=len(barcodes), path=str(exclude_path))
    return barcodes


# ---------------------------------------------------------------------------
# Sheet discovery and parsing
# ---------------------------------------------------------------------------


def discover_clinical_sheets(
    xlsx_path: Path,
    pattern: str = "CLIN_*",
) -> list[str]:
    """Return worksheet names matching the given glob pattern.

    Args:
        xlsx_path: Path to the Excel workbook.
        pattern: Glob pattern to match sheet names.

    Returns:
        List of matching sheet names, sorted.
    """
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    try:
        sheets = [name for name in wb.sheetnames if fnmatch.fnmatch(name, pattern)]
    finally:
        wb.close()

    log.info("clinical_sheets_discovered", n_sheets=len(sheets), pattern=pattern, sheets=sheets)
    return sorted(sheets)


def _find_column(columns: list[str], aliases: list[str]) -> str | None:
    """Find the first column whose name matches any alias (case-insensitive)."""
    col_lower_map = {c.strip().lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in col_lower_map:
            return col_lower_map[alias.lower()]
    return None


def parse_clinical_sheet(
    xlsx_path: Path,
    sheet_name: str,
    barcode_aliases: list[str] | None = None,
    category_aliases: list[str] | None = None,
) -> pl.DataFrame | None:
    """Parse a single clinical worksheet and return a normalised DataFrame.

    Args:
        xlsx_path: Path to the Excel workbook.
        sheet_name: Name of the worksheet to parse.
        barcode_aliases: Column name aliases for the barcode column.
        category_aliases: Column name aliases for the clinical category column.

    Returns:
        DataFrame with columns ``['barcode', 'clinical_category', 'source_sheet']``
        or ``None`` if required columns cannot be found.
    """
    if barcode_aliases is None:
        barcode_aliases = _BARCODE_ALIASES
    if category_aliases is None:
        category_aliases = _CATEGORY_ALIASES

    # Read with pandas via openpyxl, then convert to Polars
    import pandas as pd

    pdf = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    if pdf.empty:
        log.warning("empty_sheet", sheet=sheet_name)
        return None

    columns = list(pdf.columns.astype(str))
    barcode_col = _find_column(columns, barcode_aliases)
    category_col = _find_column(columns, category_aliases)

    if barcode_col is None:
        log.warning("barcode_column_not_found", sheet=sheet_name, columns=columns)
        return None

    # Build result DataFrame
    barcodes = pdf[barcode_col].astype(str).tolist()

    if category_col is not None:
        categories = pdf[category_col].astype(str).tolist()
    else:
        log.warning("category_column_not_found", sheet=sheet_name, columns=columns)
        categories = ["unknown"] * len(barcodes)

    # Normalize
    norm_barcodes = [normalize_barcode(b) for b in barcodes]
    norm_categories = [normalize_clinical_category(c) for c in categories]

    df = pl.DataFrame(
        {
            "barcode": norm_barcodes,
            "clinical_category": norm_categories,
            "source_sheet": [sheet_name] * len(norm_barcodes),
        }
    )

    # Drop rows where barcode is empty or null-like
    df = df.filter(
        pl.col("barcode").is_not_null()
        & (pl.col("barcode") != "")
        & (pl.col("barcode") != "nan")
        & (pl.col("barcode") != "None")
    )

    log.info("sheet_parsed", sheet=sheet_name, rows=len(df))
    return df


# ---------------------------------------------------------------------------
# Merge and deduplicate
# ---------------------------------------------------------------------------


def merge_clinical_sheets(
    xlsx_path: Path,
    sheet_pattern: str = "CLIN_*",
    exclude_samples_path: Path | str | None = None,
    exclude_sheet: str | None = None,
) -> pl.DataFrame:
    """Parse all clinical sheets and merge into a single deduplicated DataFrame.

    When a barcode appears in multiple sheets, the first occurrence (by sheet
    sort order) is kept.

    Args:
        xlsx_path: Path to the Excel workbook.
        sheet_pattern: Glob pattern for clinical worksheets.
        exclude_samples_path: Optional path to a newline-delimited text file
            of barcodes to exclude.
        exclude_sheet: Optional name of a worksheet inside ``xlsx_path`` that
            lists barcodes to exclude (in a ``sample_barcode`` column).

    Returns:
        Deduplicated DataFrame with columns
        ``['barcode', 'clinical_category', 'source_sheet']``.
    """
    sheets = discover_clinical_sheets(xlsx_path, pattern=sheet_pattern)
    if not sheets:
        log.error("no_clinical_sheets_found", path=str(xlsx_path), pattern=sheet_pattern)
        return pl.DataFrame(
            schema={"barcode": pl.Utf8, "clinical_category": pl.Utf8, "source_sheet": pl.Utf8}
        )

    frames: list[pl.DataFrame] = []
    for sheet in sheets:
        df = parse_clinical_sheet(xlsx_path, sheet)
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        log.error("no_valid_clinical_data")
        return pl.DataFrame(
            schema={"barcode": pl.Utf8, "clinical_category": pl.Utf8, "source_sheet": pl.Utf8}
        )

    merged = pl.concat(frames)

    # Deduplicate: keep first occurrence per barcode
    deduped = merged.unique(subset=["barcode"], keep="first", maintain_order=True)

    # Optional exclusion lists: sheet-based and/or file-based
    all_exclusions: set[str] = set()

    from_sheet: set[str] = set()
    if exclude_sheet:
        from_sheet = load_excluded_barcodes_from_sheet(xlsx_path, exclude_sheet)
        all_exclusions.update(from_sheet)

    from_file: set[str] = set()
    if exclude_samples_path is not None:
        from_file = load_excluded_barcodes_from_file(exclude_samples_path)
        all_exclusions.update(from_file)

    if all_exclusions:
        n_before = deduped.height
        deduped = deduped.filter(~pl.col("barcode").is_in(list(all_exclusions)))
        n_excluded = n_before - deduped.height
        log.info(
            "samples_excluded",
            n_from_sheet=len(from_sheet),
            n_from_file=len(from_file),
            n_total_exclusions=len(all_exclusions),
            n_excluded_from_data=n_excluded,
            n_remaining=deduped.height,
        )

    log.info(
        "clinical_data_merged",
        total_rows=len(merged),
        unique_barcodes=len(deduped),
        n_sheets=len(frames),
    )
    return deduped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("ingest-clinical")
@click.option(
    "--xlsx-path",
    type=click.Path(exists=True),
    default=None,
    help="Path to the clinical metadata Excel file. "
    "Defaults to the value in config/settings.yaml.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    default="data/processed/clinical_metadata.parquet",
    help="Output Parquet path.",
)
@click.option(
    "--sheet-pattern",
    default="CLIN_*",
    help="Glob pattern for clinical worksheets.",
)
def main(xlsx_path: str | None, output: str, sheet_pattern: str) -> None:
    """Parse clinical metadata from an Excel workbook.

    Discovers all worksheets matching the pattern, extracts barcode and
    clinical category, normalizes, deduplicates, and writes to Parquet.
    """
    if xlsx_path is None:
        settings = _load_settings()
        raw = settings.get("paths", {}).get(
            "clinical_metadata", "data/raw/clinical_metadata.xlsx"
        )
        xlsx_path = _resolve_path(raw)

    path = Path(xlsx_path)
    log.info("load_clinical_start", xlsx_path=str(path))

    result = merge_clinical_sheets(path, sheet_pattern=sheet_pattern)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out_path)

    log.info(
        "load_clinical_complete",
        output=str(out_path),
        n_samples=len(result),
        categories=result["clinical_category"].unique().to_list(),
    )

    # Print summary
    click.echo(f"Clinical metadata written to {out_path}")
    click.echo(f"  Total samples: {len(result)}")
    for cat in sorted(result["clinical_category"].unique().to_list()):
        n = len(result.filter(pl.col("clinical_category") == cat))
        click.echo(f"  {cat}: {n}")


if __name__ == "__main__":
    main()
