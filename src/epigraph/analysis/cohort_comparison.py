"""Statistical comparison between clinical cohorts.

Provides functions to compare methylation features (CpG, gene, pathway)
between clinical groups (CRC, Control, polyps) using non-parametric tests
with multiple-testing correction.

Typical usage::

    results = compare_groups(gene_features, metadata, "CRC", "Control")
    results = apply_fdr(results)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import click
import numpy as np
import polars as pl
from scipy import stats

from epigraph.common.logging import get_logger
from epigraph.common.parallel import get_n_workers
from epigraph.common.stats import apply_fdr_correction

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

TestMethod = Literal["mann_whitney", "t_test", "ks"]
FDRMethod = Literal["fdr_bh", "fdr_by", "bonferroni", "holm"]

# Default comparisons for the CRC / Control / polyps cohort study.
DEFAULT_COMPARISONS: list[dict[str, str]] = [
    {"group1": "CRC", "group2": "Control", "label": "CRC_vs_Control"},
    {"group1": "CRC", "group2": "polyps", "label": "CRC_vs_polyps"},
    {"group1": "polyps", "group2": "Control", "label": "polyps_vs_Control"},
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses the pooled standard deviation.  Returns 0.0 when both groups have
    zero variance to avoid division by zero.

    Args:
        x: Values for group 1.
        y: Values for group 2.

    Returns:
        Cohen's d (positive when mean(x) > mean(y)).
    """
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0

    var_x = np.nanvar(x, ddof=1)
    var_y = np.nanvar(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))

    if pooled_std == 0.0:
        return 0.0

    return float((np.nanmean(x) - np.nanmean(y)) / pooled_std)


def _run_test(
    x: np.ndarray,
    y: np.ndarray,
    test: TestMethod,
) -> tuple[float, float]:
    """Run a statistical test comparing two groups.

    Args:
        x: Values for group 1 (NaN-free).
        y: Values for group 2 (NaN-free).
        test: Statistical test to use.

    Returns:
        Tuple of (statistic, p_value).
    """
    if len(x) < 2 or len(y) < 2:
        return (np.nan, np.nan)

    if test == "mann_whitney":
        try:
            stat, pval = stats.mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            # All values are identical -- no ranking possible
            stat, pval = 0.0, 1.0
    elif test == "t_test":
        stat, pval = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    elif test == "ks":
        stat, pval = stats.ks_2samp(x, y)
    else:
        raise ValueError(f"Unknown test method: {test!r}")

    return (float(stat), float(pval))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_groups(
    feature_matrix: pl.DataFrame,
    metadata: pl.DataFrame,
    group1: str,
    group2: str,
    *,
    test: TestMethod = "mann_whitney",
    feature_col: str = "feature",
    sample_col: str = "sample_id",
    group_col: str = "clinical_category",
) -> pl.DataFrame:
    """Compare two clinical groups across all features.

    Args:
        feature_matrix: Wide-format DataFrame where the first column
            identifies the feature (gene / CpG / pathway) and remaining
            columns are sample values.  Alternatively, a long-format DataFrame
            with *feature_col* and sample columns.
        metadata: DataFrame with at least *sample_col* and *group_col*.
        group1: Clinical category label for the first group (e.g. ``"CRC"``).
        group2: Clinical category label for the second group (e.g. ``"Control"``).
        test: Statistical test to apply per feature.
        feature_col: Name of the column identifying features (wide format:
            first column name).
        sample_col: Name of the sample identifier column in *metadata*.
        group_col: Name of the clinical group column in *metadata*.

    Returns:
        DataFrame with columns: ``feature``, ``mean_group1``, ``mean_group2``,
        ``delta_mean``, ``cohens_d``, ``statistic``, ``p_value``, sorted by
        ascending ``p_value``.

    Raises:
        ValueError: If neither group has any samples in *metadata*.
    """
    # Resolve sample IDs for each group
    g1_samples = set(
        metadata.filter(pl.col(group_col) == group1)[sample_col].to_list()
    )
    g2_samples = set(
        metadata.filter(pl.col(group_col) == group2)[sample_col].to_list()
    )

    if not g1_samples:
        raise ValueError(f"No samples found for group {group1!r}.")
    if not g2_samples:
        raise ValueError(f"No samples found for group {group2!r}.")

    log.info(
        "compare_groups_start",
        group1=group1,
        group2=group2,
        n_group1=len(g1_samples),
        n_group2=len(g2_samples),
        test=test,
    )

    # Determine feature column: use feature_col if present, else fall back
    # to the first column.
    feat_col_name = (
        feature_col
        if feature_col in feature_matrix.columns
        else feature_matrix.columns[0]
    )
    all_sample_cols = [c for c in feature_matrix.columns if c != feat_col_name]

    g1_cols = [c for c in all_sample_cols if c in g1_samples]
    g2_cols = [c for c in all_sample_cols if c in g2_samples]

    if not g1_cols or not g2_cols:
        log.warning(
            "no_matching_sample_columns",
            g1_matched=len(g1_cols),
            g2_matched=len(g2_cols),
        )

    # ------------------------------------------------------------------
    # Pre-extract group numpy matrices ONCE (vectorized bulk extraction).
    # This avoids the O(n_features * n_cols) dict(zip(...)) overhead that
    # dominated runtime for wide matrices (800+ sample columns).
    # ------------------------------------------------------------------
    g1_data = feature_matrix.select(g1_cols).to_numpy().astype(np.float64)
    g2_data = feature_matrix.select(g2_cols).to_numpy().astype(np.float64)
    features = feature_matrix[feat_col_name].to_list()

    def _process_feature(row_idx: int) -> dict[str, Any]:
        """Compute stats for a single feature row."""
        feature_name = features[row_idx]
        x = g1_data[row_idx]
        x = x[~np.isnan(x)]
        y = g2_data[row_idx]
        y = y[~np.isnan(y)]

        mean_g1 = float(np.nanmean(x)) if len(x) > 0 else np.nan
        mean_g2 = float(np.nanmean(y)) if len(y) > 0 else np.nan

        stat, pval = _run_test(x, y, test)
        effect = _cohens_d(x, y)

        return {
            "feature": feature_name,
            "mean_group1": mean_g1,
            "mean_group2": mean_g2,
            "delta_mean": (
                mean_g1 - mean_g2
                if not (np.isnan(mean_g1) or np.isnan(mean_g2))
                else np.nan
            ),
            "cohens_d": effect,
            "statistic": stat,
            "p_value": pval,
        }

    n_features = len(features)
    row_indices = list(range(n_features))

    if n_features > 100:
        workers = get_n_workers()
        log.debug("parallel_feature_tests", n_features=n_features, n_workers=workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_process_feature, row_indices))
    else:
        results = [_process_feature(i) for i in row_indices]

    result_df = pl.DataFrame(results)

    # Sort by p_value ascending (NaN last)
    result_df = result_df.sort("p_value", nulls_last=True)

    log.info(
        "compare_groups_complete",
        n_features=len(results),
        n_significant_nominal=result_df.filter(pl.col("p_value") < 0.05).height,
    )

    return result_df


def apply_fdr(
    results: pl.DataFrame,
    method: FDRMethod = "fdr_bh",
    alpha: float = 0.05,
) -> pl.DataFrame:
    """Apply multiple-testing correction and add a q-value column.

    Args:
        results: DataFrame from :func:`compare_groups` containing a
            ``p_value`` column.
        method: Correction method passed to
            ``statsmodels.stats.multitest.multipletests``.
        alpha: Family-wise error rate or FDR level.

    Returns:
        Copy of *results* with additional columns ``q_value`` and
        ``significant`` (boolean at the given *alpha*).
    """
    pvals = results["p_value"].to_numpy()
    qvals_final, reject_final = apply_fdr_correction(pvals, method=method, alpha=alpha)

    result = results.with_columns(
        pl.Series("q_value", qvals_final),
        pl.Series("significant", reject_final),
    )

    n_sig = int(np.sum(reject_final))
    log.info("fdr_applied", method=method, alpha=alpha, n_significant=n_sig)

    return result


def run_all_comparisons(
    feature_matrix: pl.DataFrame,
    metadata: pl.DataFrame,
    comparisons_config: list[dict[str, str]] | None = None,
    *,
    test: TestMethod = "mann_whitney",
    fdr_method: FDRMethod = "fdr_bh",
    alpha: float = 0.05,
    sample_col: str = "sample_id",
    group_col: str = "clinical_category",
) -> dict[str, pl.DataFrame]:
    """Run all configured pairwise comparisons with FDR correction.

    Args:
        feature_matrix: Feature matrix (wide format).
        metadata: Sample metadata.
        comparisons_config: List of dicts with keys ``"group1"``,
            ``"group2"``, and ``"label"``.  Defaults to CRC/Control/polyps
            pairwise comparisons.
        test: Statistical test method.
        fdr_method: Multiple-testing correction method.
        alpha: Significance threshold.
        sample_col: Sample ID column name in metadata.
        group_col: Clinical group column name in metadata.

    Returns:
        Dict mapping comparison label to FDR-corrected results DataFrame.
    """
    if comparisons_config is None:
        comparisons_config = DEFAULT_COMPARISONS

    log.info("run_all_comparisons_start", n_comparisons=len(comparisons_config))

    results: dict[str, pl.DataFrame] = {}

    for comp in comparisons_config:
        label = comp["label"]
        log.info("running_comparison", label=label)

        try:
            comp_result = compare_groups(
                feature_matrix=feature_matrix,
                metadata=metadata,
                group1=comp["group1"],
                group2=comp["group2"],
                test=test,
                sample_col=sample_col,
                group_col=group_col,
            )
            comp_result = apply_fdr(comp_result, method=fdr_method, alpha=alpha)
            results[label] = comp_result
        except ValueError as exc:
            log.error("comparison_failed", label=label, error=str(exc))
            continue

    log.info("run_all_comparisons_complete", n_completed=len(results))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("run-analysis")
@click.option(
    "--feature-matrix",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the feature matrix Parquet file (gene/CpG/pathway level).",
)
@click.option(
    "--metadata",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to sample metadata Parquet file with clinical_category column.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to write comparison result Parquet files.",
)
@click.option(
    "--test",
    type=click.Choice(["mann_whitney", "t_test", "ks"]),
    default="mann_whitney",
    show_default=True,
    help="Statistical test method.",
)
@click.option(
    "--fdr-method",
    type=click.Choice(["fdr_bh", "fdr_by", "bonferroni", "holm"]),
    default="fdr_bh",
    show_default=True,
    help="Multiple testing correction method.",
)
@click.option(
    "--alpha",
    type=float,
    default=0.05,
    show_default=True,
    help="Significance threshold for FDR correction.",
)
def main(
    feature_matrix: str,
    metadata: str,
    output_dir: str,
    test: str,
    fdr_method: str,
    alpha: float,
) -> None:
    """Run cohort comparisons on a methylation feature matrix.

    Compares CRC vs Control, CRC vs polyps, and polyps vs Control using
    the specified statistical test, then applies FDR correction.
    """
    from pathlib import Path

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    log.info("cohort_comparison_start", output_dir=str(output))

    feat_df = pl.read_parquet(feature_matrix)
    meta_df = pl.read_parquet(metadata)

    all_results = run_all_comparisons(
        feature_matrix=feat_df,
        metadata=meta_df,
        test=test,  # type: ignore[arg-type]
        fdr_method=fdr_method,  # type: ignore[arg-type]
        alpha=alpha,
    )

    for label, result_df in all_results.items():
        out_path = output / f"{label}.parquet"
        result_df.write_parquet(out_path)
        log.info("comparison_written", label=label, path=str(out_path), rows=result_df.height)

    log.info("cohort_comparison_complete", n_comparisons=len(all_results))


if __name__ == "__main__":
    main()
