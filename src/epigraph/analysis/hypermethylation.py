"""Control-quantile-based hypermethylation scoring.

Ports the HypermethylationStatus logic from rules_based_classifier:
1. For each gene, compute a per-gene threshold as the Nth quantile
   of that gene's beta values across CONTROL samples only.
2. For each sample, count how many genes exceed their threshold.
3. This count is the "hypermethylation score" (HMS).

The approach is compatible with the existing rules_based_classifier pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np
import polars as pl

from epigraph.common.logging import get_logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_gene_thresholds(
    gene_matrix: pl.DataFrame,
    metadata: pl.DataFrame,
    *,
    control_label: str = "Control",
    quantile: float = 0.99,
    gene_col: str = "gene",
    sample_col: str = "barcode",
    group_col: str = "clinical_category",
) -> pl.Series:
    """Compute per-gene hypermethylation thresholds from control samples.

    For each gene (row), filters to control-sample columns only and
    computes the Nth quantile of that gene's beta values across controls.

    Args:
        gene_matrix: DataFrame with a gene identifier column and sample
            columns containing beta values (gene x samples layout, one
            row per gene).
        metadata: Clinical metadata with at least *sample_col* and
            *group_col* columns.
        control_label: Value in *group_col* identifying control samples.
        quantile: Quantile to compute (0-1).  Default 0.99 means the
            99th percentile of control betas defines the threshold.
        gene_col: Column name for gene identifiers.
        sample_col: Column name for sample identifiers in metadata.
        group_col: Column name for clinical group in metadata.

    Returns:
        A Polars Series indexed by gene name, containing the per-gene
        threshold value.  The Series name is ``"threshold"``.

    Raises:
        ValueError: If no control samples are found in the gene matrix.
    """
    # Identify control sample IDs
    control_ids = set(
        metadata.filter(pl.col(group_col) == control_label)[sample_col].to_list()
    )

    # Intersect with columns present in the gene matrix
    sample_cols = [c for c in gene_matrix.columns if c != gene_col]
    control_cols = [c for c in sample_cols if c in control_ids]

    if not control_cols:
        raise ValueError(
            f"No control samples (label={control_label!r}) found in gene matrix columns. "
            f"Checked {len(sample_cols)} sample columns against {len(control_ids)} control IDs."
        )

    log.info(
        "compute_gene_thresholds_start",
        n_genes=gene_matrix.height,
        n_control_samples=len(control_cols),
        quantile=quantile,
    )

    # Extract control-only data as numpy array (n_genes x n_control_samples)
    genes = gene_matrix[gene_col].to_list()
    control_data = gene_matrix.select(control_cols).to_numpy().astype(np.float64)

    # Compute quantile per gene (row-wise), ignoring NaN
    thresholds = np.nanquantile(control_data, quantile, axis=1)

    result = pl.Series("threshold", thresholds, dtype=pl.Float64)
    log.info(
        "compute_gene_thresholds_complete",
        n_genes=len(genes),
        median_threshold=float(np.nanmedian(thresholds)),
    )

    return result


def score_hypermethylation(
    gene_matrix: pl.DataFrame,
    thresholds: pl.Series,
    *,
    gene_col: str = "gene",
) -> pl.DataFrame:
    """Count per-sample genes exceeding their control-derived threshold.

    For each sample column, counts how many genes have beta > threshold.
    This count is the Hypermethylation Score (HMS).

    Args:
        gene_matrix: DataFrame with a gene identifier column and sample
            columns containing beta values.
        thresholds: Series of length ``gene_matrix.height`` with per-gene
            threshold values (from :func:`compute_gene_thresholds`).
        gene_col: Column name for gene identifiers.

    Returns:
        DataFrame with columns ``sample_id`` and ``hms_count``.
    """
    sample_cols = [c for c in gene_matrix.columns if c != gene_col]

    if len(thresholds) != gene_matrix.height:
        raise ValueError(
            f"Threshold length ({len(thresholds)}) != gene matrix height ({gene_matrix.height})."
        )

    log.info(
        "score_hypermethylation_start",
        n_genes=gene_matrix.height,
        n_samples=len(sample_cols),
    )

    # Extract data as numpy: shape (n_genes, n_samples)
    data = gene_matrix.select(sample_cols).to_numpy().astype(np.float64)
    thresh_arr = thresholds.to_numpy().astype(np.float64)

    # Broadcasting: compare each gene's value against its threshold
    # exceeds shape: (n_genes, n_samples), boolean
    exceeds = data > thresh_arr[:, np.newaxis]

    # Count per sample (column-wise sum), treating NaN comparisons as False
    # np.nan > threshold is False by default, which is correct behavior
    hms_counts = np.nansum(exceeds.astype(np.float64), axis=0).astype(int)

    result = pl.DataFrame({
        "sample_id": sample_cols,
        "hms_count": hms_counts.tolist(),
    })

    log.info(
        "score_hypermethylation_complete",
        n_samples=result.height,
        mean_hms=float(np.mean(hms_counts)),
        max_hms=int(np.max(hms_counts)),
    )

    return result


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------


def run_hypermethylation_analysis(
    gene_features_path: str | Path,
    metadata_path: str | Path,
    output_dir: str | Path,
    *,
    quantiles: list[float] | None = None,
    control_label: str = "Control",
    gene_col: str = "gene",
    sample_col: str = "barcode",
    group_col: str = "clinical_category",
) -> dict[str, Any]:
    """Run full hypermethylation analysis at multiple quantile thresholds.

    Loads gene-level features and clinical metadata, computes per-gene
    thresholds at each quantile, scores all samples, and writes results
    to the output directory.

    Args:
        gene_features_path: Path to gene features Parquet file (gene x samples).
        metadata_path: Path to clinical metadata Parquet file.
        output_dir: Directory to write output files.
        quantiles: List of quantiles to compute thresholds at.
            Defaults to ``[0.95, 0.99, 0.999]``.
        control_label: Clinical category label for control group.
        gene_col: Column name for gene identifiers.
        sample_col: Column name for sample identifiers in metadata.
        group_col: Column name for clinical group in metadata.

    Returns:
        Dict with keys ``"thresholds"`` (per-gene thresholds DataFrame)
        and ``"scores"`` (dict of quantile -> scores DataFrame).
    """
    if quantiles is None:
        quantiles = [0.95, 0.99, 0.999]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    log.info(
        "run_hypermethylation_analysis_start",
        gene_features_path=str(gene_features_path),
        metadata_path=str(metadata_path),
        quantiles=quantiles,
    )

    # Load data
    gene_matrix = pl.read_parquet(gene_features_path)
    metadata = pl.read_parquet(metadata_path)

    log.info(
        "data_loaded",
        n_genes=gene_matrix.height,
        n_samples=gene_matrix.width - 1,
        n_metadata_rows=metadata.height,
    )

    # Build threshold table: one column per quantile
    threshold_data: dict[str, Any] = {gene_col: gene_matrix[gene_col].to_list()}
    scores_by_quantile: dict[float, pl.DataFrame] = {}

    for q in quantiles:
        log.info("processing_quantile", quantile=q)

        thresholds = compute_gene_thresholds(
            gene_matrix,
            metadata,
            control_label=control_label,
            quantile=q,
            gene_col=gene_col,
            sample_col=sample_col,
            group_col=group_col,
        )
        threshold_data[f"q{q}"] = thresholds.to_list()

        scores = score_hypermethylation(gene_matrix, thresholds, gene_col=gene_col)

        # Join clinical category for convenience
        scores = scores.join(
            metadata.select([
                pl.col(sample_col).alias("sample_id"),
                pl.col(group_col).alias("clinical_category"),
            ]),
            on="sample_id",
            how="left",
        )

        q_label = str(q).replace(".", "_")
        scores.write_parquet(output / f"hms_scores_q{q_label}.parquet")
        scores_by_quantile[q] = scores

        # Log summary stats by group
        group_stats = scores.group_by("clinical_category").agg(
            pl.col("hms_count").mean().alias("mean_hms"),
            pl.col("hms_count").median().alias("median_hms"),
            pl.col("hms_count").max().alias("max_hms"),
        )
        for row in group_stats.iter_rows(named=True):
            log.info(
                "hms_group_summary",
                quantile=q,
                group=row["clinical_category"],
                mean_hms=row["mean_hms"],
                median_hms=row["median_hms"],
                max_hms=row["max_hms"],
            )

    # Write per-gene threshold table
    threshold_df = pl.DataFrame(threshold_data)
    threshold_df.write_parquet(output / "gene_thresholds.parquet")

    log.info(
        "run_hypermethylation_analysis_complete",
        output_dir=str(output),
        n_quantiles=len(quantiles),
    )

    return {
        "thresholds": threshold_df,
        "scores": scores_by_quantile,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("hypermethylation")
@click.option(
    "--gene-features",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet file with gene-level beta values (gene x samples).",
)
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet file with clinical metadata.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to write hypermethylation results.",
)
@click.option(
    "--quantiles",
    default="0.95,0.99,0.999",
    show_default=True,
    help="Comma-separated quantile values for threshold computation.",
)
@click.option(
    "--control-label",
    default="Control",
    show_default=True,
    help="Clinical category label for control samples.",
)
def main(
    gene_features: str,
    metadata: str,
    output_dir: str,
    quantiles: str,
    control_label: str,
) -> None:
    """Compute control-quantile hypermethylation scores.

    For each gene, computes a threshold from control sample beta values
    at the specified quantile(s).  Then counts how many genes exceed
    their threshold for each sample (the Hypermethylation Score).
    """
    quantile_list = [float(q.strip()) for q in quantiles.split(",")]

    run_hypermethylation_analysis(
        gene_features_path=gene_features,
        metadata_path=metadata,
        output_dir=output_dir,
        quantiles=quantile_list,
        control_label=control_label,
    )


if __name__ == "__main__":
    main()
