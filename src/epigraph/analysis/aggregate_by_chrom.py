"""Gene-level aggregation optimised for per-chromosome Parquet files.

The standard ``feature_aggregation.aggregate_cpgs_to_genes`` queries CpGs
across chromosome files via DuckDB — which OOMs on the full dataset because
each batch loads columns from multiple large Parquet files simultaneously.

This module takes a different approach: process **one chromosome at a time**.
For each chromosome file:
1. Load the full chromosome Parquet into memory (~1 GB for chr1)
2. Aggregate all genes whose CpGs are on this chromosome
3. Free the chromosome data
4. Move to the next chromosome

Genes spanning multiple chromosomes (rare but possible via overlapping loci)
are handled by accumulating partial per-chromosome contributions and averaging.

Peak memory: ~2 GB (one chromosome + gene accumulator).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from epigraph.common.genome_coords import CHROMOSOME_ORDER, parse_cpg_id_fast
from epigraph.common.logging import get_logger

log = get_logger(__name__)


def aggregate_genes_by_chromosome(
    beta_chrom_dir: str | Path,
    cpg_gene_mapping: pl.DataFrame,
    method: str = "mean",
    min_cpgs_per_gene: int = 1,
) -> pl.DataFrame:
    """Aggregate CpG beta values to gene-level features, one chromosome at a time.

    Args:
        beta_chrom_dir: Directory containing per-chromosome Parquet files
            (``beta_chr1.parquet``, etc.).
        cpg_gene_mapping: DataFrame with ``cpg_id`` and ``gene_symbol`` columns.
        method: ``"mean"`` or ``"median"``.
        min_cpgs_per_gene: Minimum CpGs per gene to produce a feature.

    Returns:
        DataFrame with ``gene`` column + one column per sample barcode.
    """
    chrom_dir = Path(beta_chrom_dir)

    # Build gene → {cpg_id} lookup (deduplicated)
    gene_cpgs: dict[str, set[str]] = {}
    for row in (
        cpg_gene_mapping.select("gene_symbol", "cpg_id")
        .unique()
        .iter_rows(named=True)
    ):
        gene_cpgs.setdefault(row["gene_symbol"], set()).add(row["cpg_id"])

    # Group CpGs by chromosome for efficient per-file processing
    cpg_to_chrom: dict[str, str] = {}
    for cpg in cpg_gene_mapping["cpg_id"].unique().to_list():
        try:
            chrom, _ = parse_cpg_id_fast(cpg)
            cpg_to_chrom[cpg] = chrom
        except (ValueError, IndexError):
            pass

    # For each gene, figure out which chromosomes its CpGs are on
    gene_chroms: dict[str, set[str]] = {}
    for gene, cpgs in gene_cpgs.items():
        chroms = {cpg_to_chrom[c] for c in cpgs if c in cpg_to_chrom}
        if chroms:
            gene_chroms[gene] = chroms

    # Discover available chromosome files
    chrom_files: dict[str, Path] = {}
    for pq_file in chrom_dir.glob("beta_chr*.parquet"):
        chrom = pq_file.stem.replace("beta_", "")
        chrom_files[chrom] = pq_file

    sorted_chroms = sorted(chrom_files.keys(), key=lambda c: CHROMOSOME_ORDER.get(c, 99))
    log.info(
        "aggregate_by_chrom_start",
        n_genes=len(gene_cpgs),
        n_chromosomes=len(sorted_chroms),
        method=method,
    )

    # Accumulators: for each gene, store sum and count per sample
    # so we can compute the mean across chromosomes.
    # Also track actual CpG count per gene for the min_cpgs_per_gene filter.
    sample_ids: list[str] | None = None
    gene_sum: dict[str, np.ndarray] = {}   # gene → sum of betas per sample
    gene_count: dict[str, np.ndarray] = {} # gene → count of non-NaN chromosome contributions
    gene_n_cpgs: dict[str, int] = {}       # gene → total number of CpGs across all chromosomes

    total_start = time.time()

    for chrom_idx, chrom in enumerate(sorted_chroms, 1):
        chrom_file = chrom_files[chrom]
        t0 = time.time()

        # Load chromosome data
        df = pl.read_parquet(chrom_file)
        if sample_ids is None:
            sample_ids = df["sample_id"].to_list()
            n_samples = len(sample_ids)

        cpg_cols = [c for c in df.columns if c != "sample_id"]

        # Convert the entire chromosome to a numpy matrix ONCE.
        # This avoids 5000+ calls to df.select().to_numpy() per chromosome.
        chrom_matrix = df.select(cpg_cols).to_numpy().astype(np.float32)
        # Build column-name → integer-index lookup for fast slicing
        col_to_idx = {col: i for i, col in enumerate(cpg_cols)}

        # Find genes with CpGs on this chromosome
        genes_on_chrom = [
            g for g, chroms in gene_chroms.items()
            if chrom in chroms
        ]

        n_genes_processed = 0
        for gene in genes_on_chrom:
            # Get column indices for this gene's CpGs on this chromosome
            gene_col_indices = [
                col_to_idx[c] for c in gene_cpgs[gene]
                if c in col_to_idx
            ]
            if not gene_col_indices:
                continue

            # Fast numpy integer indexing into the pre-built matrix
            vals = chrom_matrix[:, gene_col_indices]  # (n_samples, n_cpgs_for_gene)

            if method == "mean":
                with np.errstate(all="ignore"):
                    cpg_means = np.nanmean(vals, axis=1)
            elif method == "median":
                with np.errstate(all="ignore"):
                    cpg_means = np.nanmedian(vals, axis=1)
            else:
                raise ValueError(f"Unsupported method: {method}")

            # Accumulate: handle genes spanning multiple chromosomes
            if gene not in gene_sum:
                gene_sum[gene] = np.zeros(n_samples, dtype=np.float64)
                gene_count[gene] = np.zeros(n_samples, dtype=np.float64)

            valid_mask = ~np.isnan(cpg_means)
            gene_sum[gene][valid_mask] += cpg_means[valid_mask]
            gene_count[gene][valid_mask] += 1.0
            gene_n_cpgs[gene] = gene_n_cpgs.get(gene, 0) + len(gene_col_indices)
            n_genes_processed += 1

        # Free the chromosome matrix
        del chrom_matrix

        elapsed = time.time() - t0
        log.info(
            "chromosome_processed",
            chrom=chrom,
            n_cpgs=len(cpg_cols),
            n_genes=n_genes_processed,
            elapsed_s=round(elapsed, 1),
            progress=f"{chrom_idx}/{len(sorted_chroms)}",
        )

        # Free chromosome data
        del df

    # Compute final gene-level values
    if sample_ids is None:
        log.warning("no_chromosome_files_found")
        return pl.DataFrame({"gene": []})

    result_rows: list[dict[str, Any]] = []
    for gene in sorted(gene_sum.keys()):
        n_cpgs = gene_n_cpgs.get(gene, 0)
        if n_cpgs < min_cpgs_per_gene:
            continue

        with np.errstate(all="ignore"):
            final_values = np.where(
                gene_count[gene] > 0,
                gene_sum[gene] / gene_count[gene],
                np.nan,
            )

        row: dict[str, Any] = {"gene": gene}
        row.update({sid: float(v) for sid, v in zip(sample_ids, final_values)})
        result_rows.append(row)

    total_elapsed = time.time() - total_start
    log.info(
        "aggregate_by_chrom_complete",
        n_genes=len(result_rows),
        n_samples=len(sample_ids),
        elapsed_s=round(total_elapsed, 1),
    )

    return pl.DataFrame(result_rows)
