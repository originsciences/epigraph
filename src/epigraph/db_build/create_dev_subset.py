"""Create a small, representative subset of the full beta matrix for development.

The subset is stratified by chromosome (for CpGs) and by clinical category
(for samples) so that downstream pipelines can be exercised quickly without
loading the full 22 GB matrix into memory.

The implementation is *memory-safe*: it reads only the header row and first
column to plan the subset, then uses PyArrow / Polars column-selection to
materialise only the required cells.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import click
import polars as pl
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq
import yaml

from epigraph.common.genome_coords import CHROMOSOME_ORDER, parse_cpg_id
from epigraph.common.io import read_beta_header
from epigraph.common.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _load_config(config_path: Path) -> dict[str, Any]:
    """Load and return the dev-subset configuration from YAML."""
    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}
    return raw.get("subset", raw)


# ---------------------------------------------------------------------------
# Header and sample-ID reading (memory-safe)
# ---------------------------------------------------------------------------


def read_header(csv_path: Path) -> list[str]:
    """Read only the first line of the CSV and return CpG column names.

    The first cell of the header is expected to be empty (row-index column).
    This returns only the CpG columns, *not* the empty first cell.

    Delegates header parsing to :func:`epigraph.common.io.read_beta_header`
    and strips the leading ``"sample_id"`` column.

    Args:
        csv_path: Path to the full beta matrix CSV.

    Returns:
        List of CpG column names (e.g. ``['chr1_10469', ...]``).
    """
    full_header = read_beta_header(csv_path)
    # read_beta_header returns ["sample_id", cpg1, cpg2, ...]; strip first.
    cpg_columns = full_header[1:]
    log.info("header_read", n_cpg_columns=len(cpg_columns))
    return cpg_columns


def read_sample_ids(csv_path: Path) -> list[str]:
    """Read the first column of every row (after the header) to get sample IDs.

    For large files this avoids parsing millions of comma-separated columns
    per line by only reading up to the first comma on each line.

    Args:
        csv_path: Path to the full beta matrix CSV.

    Returns:
        List of sample barcode strings.
    """
    sample_ids: list[str] = []
    with open(csv_path, "rb") as fh:
        fh.readline()  # skip header (fast binary read)
        for raw_line in fh:
            # Only decode up to the first comma — avoids parsing 4M fields
            comma_pos = raw_line.find(b",")
            if comma_pos > 0:
                sample_ids.append(raw_line[:comma_pos].decode("utf-8").strip())
    log.info("sample_ids_read", n_samples=len(sample_ids))
    return sample_ids


# ---------------------------------------------------------------------------
# Stratified selection
# ---------------------------------------------------------------------------


def select_cpgs_stratified(
    cpg_columns: list[str],
    n_cpgs: int,
    seed: int = 42,
    chromosomes: list[str] | None = None,
) -> list[str]:
    """Select CpGs spread evenly across chromosomes.

    Args:
        cpg_columns: All CpG column names from the header.
        n_cpgs: Target number of CpGs to select.
        seed: Random seed for reproducibility.
        chromosomes: If provided, restrict to these chromosomes.

    Returns:
        Sorted list of selected CpG column names.
    """
    rng = random.Random(seed)

    # Group CpGs by chromosome
    by_chrom: dict[str, list[str]] = defaultdict(list)
    for cpg in cpg_columns:
        try:
            chrom, _ = parse_cpg_id(cpg)
            if chromosomes and chrom not in chromosomes:
                continue
            by_chrom[chrom].append(cpg)
        except ValueError:
            continue

    if not by_chrom:
        log.warning("no_valid_cpgs_found")
        return []

    # Allocate proportionally, with at least 1 per chromosome
    total_valid = sum(len(v) for v in by_chrom.values())
    selected: list[str] = []
    chroms_sorted = sorted(by_chrom.keys(), key=lambda c: CHROMOSOME_ORDER.get(c, 99))

    for chrom in chroms_sorted:
        pool = by_chrom[chrom]
        # Proportional allocation, minimum 1
        alloc = max(1, round(len(pool) / total_valid * n_cpgs))
        alloc = min(alloc, len(pool))
        chosen = rng.sample(pool, alloc)
        selected.extend(chosen)

    # Trim or pad to hit target
    if len(selected) > n_cpgs:
        selected = rng.sample(selected, n_cpgs)
    elif len(selected) < n_cpgs:
        remaining = [c for c in cpg_columns if c not in set(selected)]
        extra = min(n_cpgs - len(selected), len(remaining))
        selected.extend(rng.sample(remaining, extra))

    log.info(
        "cpgs_selected",
        n_selected=len(selected),
        n_chromosomes=len(by_chrom),
    )
    return sorted(selected)


def select_samples_stratified(
    sample_ids: list[str],
    n_samples: int,
    clinical_metadata: pl.DataFrame | None = None,
    category_column: str = "clinical_category",
    barcode_column: str = "barcode",
    min_per_category: int = 5,
    seed: int = 42,
) -> list[str]:
    """Select samples, stratifying by clinical category when metadata is available.

    Args:
        sample_ids: All sample barcodes from the beta matrix.
        n_samples: Target number of samples.
        clinical_metadata: Optional DataFrame with barcode and category columns.
        category_column: Name of the clinical category column.
        barcode_column: Name of the barcode column.
        min_per_category: Minimum samples per category.
        seed: Random seed.

    Returns:
        List of selected sample barcodes.
    """
    rng = random.Random(seed)

    if clinical_metadata is not None and len(clinical_metadata) > 0:
        # Build barcode -> category mapping
        meta_barcodes = set(clinical_metadata[barcode_column].to_list())
        sample_set = set(sample_ids)
        matched = meta_barcodes & sample_set

        if matched:
            by_category: dict[str, list[str]] = defaultdict(list)
            for row in clinical_metadata.iter_rows(named=True):
                bc = row[barcode_column]
                if bc in sample_set:
                    cat = row.get(category_column, "unknown")
                    by_category[cat or "unknown"].append(bc)

            selected: list[str] = []
            for cat, barcodes in sorted(by_category.items()):
                alloc = max(min_per_category, round(len(barcodes) / len(matched) * n_samples))
                alloc = min(alloc, len(barcodes))
                selected.extend(rng.sample(barcodes, alloc))

            if len(selected) > n_samples:
                selected = rng.sample(selected, n_samples)
            elif len(selected) < n_samples:
                remaining = [s for s in sample_ids if s not in set(selected)]
                extra = min(n_samples - len(selected), len(remaining))
                selected.extend(rng.sample(remaining, extra))

            log.info(
                "samples_selected_stratified",
                n_selected=len(selected),
                n_categories=len(by_category),
            )
            return selected

    # Fallback: random selection
    n = min(n_samples, len(sample_ids))
    selected = rng.sample(sample_ids, n)
    log.info("samples_selected_random", n_selected=len(selected))
    return selected


# ---------------------------------------------------------------------------
# Subset extraction (memory-safe)
# ---------------------------------------------------------------------------


def extract_subset(
    csv_path: Path,
    selected_cpgs: list[str],
    selected_samples: list[str],
    output_path: Path,
) -> None:
    """Extract the subset from the CSV and write to Parquet.

    Uses PyArrow CSV reader with ``include_columns`` to read only the
    selected CpG columns (+ sample ID column) from the 4M-column file.
    This is memory-efficient because only ~1000 columns are materialised.

    The header line of the full beta matrix is ~100 MB (4M column names),
    which exceeds PyArrow's default streaming block size, so we use the
    non-streaming ``read_csv`` with a large enough block size.

    Args:
        csv_path: Path to the full beta matrix CSV.
        selected_cpgs: CpG columns to include.
        selected_samples: Sample barcodes to include.
        output_path: Destination Parquet path.
    """
    sample_set = set(selected_samples)

    # The first column header is empty in the CSV.  PyArrow will auto-name it
    # (e.g. "f0" or ""), so we read header manually and supply column_names.
    all_header = read_header(csv_path)
    full_header = ["sample_id"] + all_header  # rename empty first cell

    # Columns to include: sample_id + selected CpGs
    cpg_set = set(selected_cpgs)
    include_columns = ["sample_id"] + [c for c in all_header if c in cpg_set]

    log.info(
        "extracting_subset",
        n_cpg_cols=len(include_columns) - 1,
        n_target_samples=len(selected_samples),
    )

    # The header line is ~100 MB for 4M columns; set block_size large enough
    # to contain the entire header line.
    table = pcsv.read_csv(
        csv_path,
        read_options=pcsv.ReadOptions(
            column_names=full_header,
            skip_rows=1,  # skip the original header; we supply our own
            block_size=256 * 1024 * 1024,  # 256 MB to accommodate the wide header
        ),
        parse_options=pcsv.ParseOptions(delimiter=","),
        convert_options=pcsv.ConvertOptions(
            include_columns=include_columns,
            strings_can_be_null=True,
            null_values=["", "NA", "NaN"],
        ),
    )

    # Filter to selected samples
    sample_col = table.column("sample_id")
    mask = pa.array([s.as_py() in sample_set for s in sample_col], type=pa.bool_())
    table = table.filter(mask)

    if table.num_rows == 0:
        log.warning("no_matching_samples_found")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="zstd")
    log.info("subset_written", path=str(output_path), rows=len(table), cols=len(table.columns))


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------


def write_id_list(ids: list[str], output_path: Path) -> None:
    """Write a list of IDs to a text file, one per line.

    Args:
        ids: List of identifier strings.
        output_path: Destination file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(ids) + "\n")
    log.info("id_list_written", path=str(output_path), n_ids=len(ids))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_settings_path(key: str, fallback: str) -> str:
    """Read a path from config/settings.yaml, resolving shell-variable defaults."""
    import re as _re

    settings_path = Path("config/settings.yaml")
    if settings_path.exists():
        with open(settings_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        raw = cfg.get("paths", {}).get(key, "")
        m = _re.match(r"\$\{[^:}]+:-(.+)\}", raw)
        if m:
            return m.group(1)
        if raw:
            return raw
    return fallback


@click.command("create-dev-subset")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default="config/dev_subset.yaml",
    help="Path to dev subset config YAML.",
)
@click.option(
    "--beta-matrix",
    type=click.Path(exists=True),
    default=None,
    help="Path to the full beta matrix CSV. Defaults to settings.yaml value.",
)
@click.option(
    "--clinical-metadata",
    type=click.Path(exists=True),
    default=None,
    help="Path to clinical metadata Parquet (if available).",
)
def main(
    config_path: str,
    beta_matrix: str | None,
    clinical_metadata: str | None,
) -> None:
    """Create a small representative subset of the full beta matrix.

    Stratifies CpG selection across chromosomes and sample selection by
    clinical category (when metadata is available).  Never loads the full
    matrix into memory.
    """
    cfg = _load_config(Path(config_path))
    n_samples: int = cfg.get("n_samples", 50)
    n_cpgs: int = cfg.get("n_cpgs", 1000)
    seed: int = cfg.get("random_seed", 42)
    min_per_cat: int = cfg.get("min_per_category", 5)
    chroms: list[str] | None = cfg.get("chromosomes")

    # Resolve beta matrix path
    if beta_matrix is None:
        beta_matrix = _resolve_settings_path(
            "beta_matrix",
            "data/raw/beta_matrix.csv",
        )
    csv_path = Path(beta_matrix)

    log.info(
        "create_dev_subset_start",
        csv_path=str(csv_path),
        n_samples=n_samples,
        n_cpgs=n_cpgs,
    )

    # Step 1: Read header to get CpG column names
    cpg_columns = read_header(csv_path)

    # Step 2: Select CpGs spread across chromosomes
    selected_cpgs = select_cpgs_stratified(cpg_columns, n_cpgs, seed=seed, chromosomes=chroms)

    # Step 3: Read sample IDs from the first column
    sample_ids = read_sample_ids(csv_path)

    # Step 4: Load clinical metadata if available
    clin_df: pl.DataFrame | None = None
    if clinical_metadata:
        clin_path = Path(clinical_metadata)
        if clin_path.exists():
            clin_df = pl.read_parquet(clin_path)
            log.info("clinical_metadata_loaded", rows=len(clin_df))

    # Step 5: Select samples
    selected_samples = select_samples_stratified(
        sample_ids,
        n_samples,
        clinical_metadata=clin_df,
        min_per_category=min_per_cat,
        seed=seed,
    )

    # Step 6: Extract subset
    output_beta = Path(cfg.get("output_beta", "data/dev/beta_subset.parquet"))
    extract_subset(csv_path, selected_cpgs, selected_samples, output_beta)

    # Step 7: Write ID lists
    cpg_list_path = Path(cfg.get("output_cpg_list", "data/dev/cpg_list.txt"))
    sample_list_path = Path(cfg.get("output_sample_list", "data/dev/sample_list.txt"))
    write_id_list(selected_cpgs, cpg_list_path)
    write_id_list(selected_samples, sample_list_path)

    log.info(
        "create_dev_subset_complete",
        n_cpgs=len(selected_cpgs),
        n_samples=len(selected_samples),
        output=str(output_beta),
    )


if __name__ == "__main__":
    main()
