"""Pathway and GO term enrichment analysis for differentially methylated genes.

Tests whether differentially methylated genes are over-represented in
biological pathways or GO terms using Fisher's exact test or Gene Set
Enrichment Analysis (GSEA).

Typical workflow::

    diff_results = compare_groups(...)
    enriched = run_pathway_enrichment(diff_results, gene_pathway_mapping)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import click
import numpy as np
import polars as pl
from scipy import stats

from epigraph.common.logging import get_logger
from epigraph.common.stats import apply_fdr_correction

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

EnrichmentMethod = Literal["fisher", "gsea"]


# ---------------------------------------------------------------------------
# Fisher's exact test
# ---------------------------------------------------------------------------


def fisher_enrichment(
    significant_genes: set[str],
    pathway_genes: set[str],
    background_genes: set[str],
) -> dict[str, float]:
    """Test for over-representation of significant genes in a pathway.

    Constructs a 2x2 contingency table and applies Fisher's exact test
    (one-sided, testing for enrichment / over-representation).

    Args:
        significant_genes: Set of genes identified as differentially
            methylated (e.g. q_value < 0.05).
        pathway_genes: Set of genes belonging to the pathway of interest.
        background_genes: Universe of all tested genes.

    Returns:
        Dict with keys: ``odds_ratio``, ``p_value``, ``n_overlap``,
        ``n_significant``, ``n_pathway``, ``n_background``.
    """
    sig = significant_genes & background_genes
    pw = pathway_genes & background_genes
    bg = background_genes

    # Contingency table:
    #                   In pathway    Not in pathway
    # Significant       a             b
    # Not significant   c             d
    a = len(sig & pw)
    b = len(sig - pw)
    c = len(pw - sig)
    d = len(bg - sig - pw)

    table = np.array([[a, b], [c, d]])
    odds_ratio, p_value = stats.fisher_exact(table, alternative="greater")

    return {
        "odds_ratio": float(odds_ratio),
        "p_value": float(p_value),
        "n_overlap": a,
        "n_significant": len(sig),
        "n_pathway": len(pw),
        "n_background": len(bg),
    }


# ---------------------------------------------------------------------------
# GSEA (pre-ranked)
# ---------------------------------------------------------------------------


def gsea_preranked(
    ranked_genes: list[tuple[str, float]],
    gene_sets: dict[str, set[str]],
    n_permutations: int = 1000,
    *,
    seed: int = 42,
) -> pl.DataFrame:
    """Run Gene Set Enrichment Analysis on a pre-ranked gene list.

    Implements a permutation-based GSEA following the Subramanian et al.
    (2005) algorithm.  For each gene set, computes an enrichment score (ES)
    by walking down the ranked list and accumulating a running sum that
    increases when a gene is in the set and decreases otherwise.

    Args:
        ranked_genes: List of ``(gene_symbol, rank_metric)`` tuples sorted
            by the rank metric (e.g. signed ``-log10(p) * sign(delta)``).
            Must be pre-sorted in descending order.
        gene_sets: Dict mapping set name to a set of gene symbols.
        n_permutations: Number of permutations for p-value estimation.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: ``gene_set``, ``es`` (enrichment score),
        ``nes`` (normalised ES), ``p_value``, ``n_genes``, ``n_hits``.
    """
    rng = np.random.default_rng(seed)

    gene_names = [g for g, _ in ranked_genes]
    rank_values = np.array([v for _, v in ranked_genes], dtype=np.float64)
    n_genes = len(gene_names)

    if n_genes == 0:
        log.warning("gsea_empty_ranked_list")
        return pl.DataFrame({
            "gene_set": [],
            "es": [],
            "nes": [],
            "p_value": [],
            "n_genes": [],
            "n_hits": [],
        })

    gene_index = {g: i for i, g in enumerate(gene_names)}

    results: list[dict[str, Any]] = []

    for set_name, set_genes in sorted(gene_sets.items()):
        hits = np.zeros(n_genes, dtype=bool)
        for g in set_genes:
            idx = gene_index.get(g)
            if idx is not None:
                hits[idx] = True

        n_hits = int(hits.sum())
        if n_hits == 0 or n_hits == n_genes:
            continue

        es = _compute_enrichment_score(rank_values, hits)

        # Permutation test — shuffle in-place to avoid allocating a new array
        # per permutation.
        perm_hits = hits.copy()
        null_es = np.empty(n_permutations, dtype=np.float64)
        for perm_i in range(n_permutations):
            rng.shuffle(perm_hits)
            null_es[perm_i] = _compute_enrichment_score(rank_values, perm_hits)

        # Compute NES and p-value
        # TODO: Refine NES computation to separately normalise positive and
        # negative enrichment scores as in the original GSEA paper.
        mean_null = np.mean(np.abs(null_es))
        nes = es / mean_null if mean_null > 0 else 0.0

        if es >= 0:
            p_value = float(np.mean(null_es >= es))
        else:
            p_value = float(np.mean(null_es <= es))

        # Ensure p_value is not exactly 0 (use minimum possible given permutations)
        if p_value == 0.0:
            p_value = 1.0 / (n_permutations + 1)

        results.append({
            "gene_set": set_name,
            "es": float(es),
            "nes": float(nes),
            "p_value": p_value,
            "n_genes": len(set_genes),
            "n_hits": n_hits,
        })

    if not results:
        log.warning("gsea_no_results")
        return pl.DataFrame({
            "gene_set": [],
            "es": [],
            "nes": [],
            "p_value": [],
            "n_genes": [],
            "n_hits": [],
        })

    return pl.DataFrame(results).sort("p_value", nulls_last=True)


def _compute_enrichment_score(
    rank_values: np.ndarray,
    hits: np.ndarray,
) -> float:
    """Compute the enrichment score for a single gene set.

    Walking statistic: at each position in the ranked list, increment the
    running sum proportionally to the absolute rank metric if the gene is
    in the set, and decrement by a constant penalty otherwise.

    Args:
        rank_values: Array of rank metric values (same order as genes).
        hits: Boolean array indicating gene set membership.

    Returns:
        Enrichment score (the maximum deviation from zero).
    """
    n = len(rank_values)
    n_hits = hits.sum()
    n_miss = n - n_hits

    if n_hits == 0 or n_miss == 0:
        return 0.0

    # Weighted hit score: |r_i|^1 for genes in the set
    hit_weights = np.abs(rank_values) * hits
    hit_norm = hit_weights.sum()
    if hit_norm == 0:
        hit_norm = 1.0

    miss_penalty = 1.0 / n_miss

    running_sum = np.cumsum(
        np.where(hits, hit_weights / hit_norm, -miss_penalty)
    )

    # ES is the maximum deviation from zero
    max_dev = running_sum.max()
    min_dev = running_sum.min()

    if abs(max_dev) >= abs(min_dev):
        return float(max_dev)
    else:
        return float(min_dev)


# ---------------------------------------------------------------------------
# High-level enrichment runner
# ---------------------------------------------------------------------------


def run_pathway_enrichment(
    diff_results: pl.DataFrame,
    gene_pathway_mapping: pl.DataFrame,
    *,
    method: EnrichmentMethod = "fisher",
    q_value_threshold: float = 0.05,
    feature_col: str = "feature",
    q_value_col: str = "q_value",
    effect_col: str = "cohens_d",
    pathway_col: str = "pathway_id",
    gene_col: str = "gene_symbol",
    fdr_method: str = "fdr_bh",
    fdr_alpha: float = 0.05,
    n_permutations: int = 1000,
) -> pl.DataFrame:
    """Run enrichment analysis on differentially methylated genes.

    Args:
        diff_results: Results from :func:`cohort_comparison.compare_groups`
            (or after :func:`cohort_comparison.apply_fdr`).  Must have columns
            *feature_col*, *q_value_col*, and *effect_col*.
        gene_pathway_mapping: DataFrame with columns *gene_col* and
            *pathway_col*.
        method: ``"fisher"`` for Fisher's exact test or ``"gsea"`` for
            pre-ranked GSEA.
        q_value_threshold: Threshold on *q_value_col* to define significant
            genes (Fisher method only).
        feature_col: Column name for gene identifiers in *diff_results*.
        q_value_col: Column name for adjusted p-values.
        effect_col: Column name for effect sizes (used as rank metric for
            GSEA).
        pathway_col: Column name for pathway identifiers.
        gene_col: Column name for gene symbols in the mapping.
        fdr_method: FDR correction method for enrichment p-values.
        fdr_alpha: Significance threshold for FDR correction.
        n_permutations: Number of permutations (GSEA only).

    Returns:
        DataFrame with enrichment results sorted by ``p_value``, including
        ``q_value`` after FDR correction.
    """
    log.info(
        "run_pathway_enrichment_start",
        method=method,
        n_diff_features=diff_results.height,
    )

    # Build pathway -> gene set mapping
    pathway_genes: dict[str, set[str]] = {}
    for row in gene_pathway_mapping.iter_rows(named=True):
        pw = row[pathway_col]
        gene = row[gene_col]
        pathway_genes.setdefault(pw, set()).add(gene)

    # All tested genes (background)
    background_genes = set(diff_results[feature_col].to_list())

    if method == "fisher":
        # Identify significant genes
        sig_mask = diff_results[q_value_col] < q_value_threshold
        significant_genes = set(
            diff_results.filter(sig_mask)[feature_col].to_list()
        )
        log.info(
            "fisher_enrichment_start",
            n_significant=len(significant_genes),
            n_pathways=len(pathway_genes),
        )

        enrichment_rows: list[dict[str, Any]] = []
        for pw_id, pw_genes in sorted(pathway_genes.items()):
            result = fisher_enrichment(significant_genes, pw_genes, background_genes)
            enrichment_rows.append({"pathway": pw_id, **result})

        if not enrichment_rows:
            log.warning("no_enrichment_results")
            return pl.DataFrame({"pathway": []})

        enrichment_df = pl.DataFrame(enrichment_rows)

    elif method == "gsea":
        # Build ranked gene list: sign(effect) * -log10(p_value)
        # Filter out genes with NaN p-values — they produce NaN rank metrics
        # that propagate through the running sum and corrupt all ES values.
        ranked_genes: list[tuple[str, float]] = []
        n_nan_filtered = 0
        for row in diff_results.iter_rows(named=True):
            gene = row[feature_col]
            effect = row.get(effect_col, 0.0) or 0.0
            p_val = row.get("p_value", 1.0) or 1.0
            if p_val != p_val:  # NaN check
                n_nan_filtered += 1
                continue
            # Avoid log(0)
            p_val = max(p_val, 1e-300)
            rank_metric = math.copysign(-math.log10(p_val), effect)
            ranked_genes.append((gene, rank_metric))

        if n_nan_filtered > 0:
            log.info("gsea_nan_filtered", n_filtered=n_nan_filtered)

        # Sort descending by rank metric
        ranked_genes.sort(key=lambda x: x[1], reverse=True)

        enrichment_df = gsea_preranked(
            ranked_genes=ranked_genes,
            gene_sets=pathway_genes,
            n_permutations=n_permutations,
        )

        # Rename gene_set -> pathway for consistency
        if "gene_set" in enrichment_df.columns:
            enrichment_df = enrichment_df.rename({"gene_set": "pathway"})

    else:
        raise ValueError(f"Unknown enrichment method: {method!r}")

    # Apply FDR correction
    if enrichment_df.height > 0 and "p_value" in enrichment_df.columns:
        pvals = enrichment_df["p_value"].to_numpy()
        qvals_final, reject_final = apply_fdr_correction(
            pvals, method=fdr_method, alpha=fdr_alpha
        )

        enrichment_df = enrichment_df.with_columns(
            pl.Series("q_value", qvals_final),
            pl.Series("significant", reject_final),
        )

    enrichment_df = enrichment_df.sort("p_value", nulls_last=True)

    n_sig = (
        enrichment_df.filter(pl.col("significant")).height
        if "significant" in enrichment_df.columns
        else 0
    )
    log.info(
        "run_pathway_enrichment_complete",
        method=method,
        n_pathways_tested=enrichment_df.height,
        n_significant=n_sig,
    )

    return enrichment_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("pathway-enrichment")
@click.option(
    "--diff-results",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet file with differential methylation results (from cohort_comparison).",
)
@click.option(
    "--gene-pathway-mapping",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet file with gene_symbol -> pathway_id mapping.",
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False),
    help="Output Parquet file for enrichment results.",
)
@click.option(
    "--method",
    type=click.Choice(["fisher", "gsea"]),
    default="fisher",
    show_default=True,
    help="Enrichment method.",
)
@click.option(
    "--q-threshold",
    type=float,
    default=0.05,
    show_default=True,
    help="q-value threshold for defining significant genes (Fisher only).",
)
@click.option(
    "--n-permutations",
    type=int,
    default=1000,
    show_default=True,
    help="Number of permutations for GSEA.",
)
@click.option(
    "--fdr-method",
    type=click.Choice(["fdr_bh", "fdr_by", "bonferroni", "holm"]),
    default="fdr_bh",
    show_default=True,
    help="FDR correction method for enrichment p-values.",
)
def main(
    diff_results: str,
    gene_pathway_mapping: str,
    output: str,
    method: str,
    q_threshold: float,
    n_permutations: int,
    fdr_method: str,
) -> None:
    """Test pathway enrichment of differentially methylated genes.

    Reads differential methylation results and a gene-pathway mapping,
    then runs Fisher's exact test or GSEA to identify enriched pathways.
    """
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("pathway_enrichment_cli_start", method=method)

    diff_df = pl.read_parquet(diff_results)
    mapping_df = pl.read_parquet(gene_pathway_mapping)

    result = run_pathway_enrichment(
        diff_results=diff_df,
        gene_pathway_mapping=mapping_df,
        method=method,  # type: ignore[arg-type]
        q_value_threshold=q_threshold,
        n_permutations=n_permutations,
        fdr_method=fdr_method,
    )

    result.write_parquet(output_path)
    log.info(
        "pathway_enrichment_cli_complete",
        output=str(output_path),
        n_results=result.height,
    )


if __name__ == "__main__":
    main()
