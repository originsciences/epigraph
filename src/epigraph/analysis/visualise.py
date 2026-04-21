"""Visualisation module for methylation analysis results.

Generates publication-quality plots from cohort comparison, pathway
enrichment, and hypermethylation score data.  All figures are saved to
disk as 300 DPI PNGs using the non-interactive Agg backend.

Typical usage::

    from epigraph.analysis.visualise import volcano_plot
    volcano_plot("data/processed/comparisons_full/CRC_vs_Control.parquet",
                 "figures/volcano.png", "CRC vs Control")
"""

from __future__ import annotations

from pathlib import Path

import click
import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from epigraph.common.logging import get_logger  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
_FIGSIZE_WIDE = (10, 7)
_FIGSIZE_SQUARE = (8, 8)
_DPI = 300

PathLike = str | Path


# ---------------------------------------------------------------------------
# Volcano plot
# ---------------------------------------------------------------------------


def volcano_plot(
    diff_results_path: PathLike,
    output_path: PathLike,
    title: str,
    *,
    q_threshold: float = 0.05,
    effect_threshold: float = 0.3,
) -> Path:
    """Generate a volcano plot from cohort comparison results.

    Args:
        diff_results_path: Path to comparison Parquet with columns
            ``feature``, ``delta_mean``, ``p_value``, ``q_value``.
        output_path: Destination PNG path.
        title: Figure title.
        q_threshold: FDR significance cutoff.
        effect_threshold: Minimum absolute delta_mean to highlight.

    Returns:
        Resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(str(diff_results_path))
    log.info("volcano_plot", n_features=len(df), source=str(diff_results_path))

    # Compute -log10(p) and clamp infinite values
    neg_log_p = -np.log10(np.clip(df["p_value"].to_numpy(), a_min=1e-300, a_max=None))
    delta = df["delta_mean"].to_numpy()
    q_values = df["q_value"].to_numpy()
    features = df["feature"].to_list()

    # Categorise points
    sig = q_values < q_threshold
    large_effect = np.abs(delta) >= effect_threshold
    cat_colors = np.full(len(df), "grey", dtype=object)
    cat_colors[sig & ~large_effect] = "red"
    cat_colors[sig & large_effect] = "blue"

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    for colour, label in [
        ("grey", "Not significant"),
        ("red", f"FDR < {q_threshold}"),
        ("blue", f"FDR < {q_threshold} & |delta| >= {effect_threshold}"),
    ]:
        mask = cat_colors == colour
        ax.scatter(
            delta[mask],
            neg_log_p[mask],
            c=colour,
            s=6,
            alpha=0.5,
            label=label,
            edgecolors="none",
        )

    # Label top 10 genes by composite score (significance + effect)
    score = neg_log_p * np.abs(delta)
    top_idx = np.argsort(score)[-10:]
    for i in top_idx:
        ax.annotate(
            features[i],
            (delta[i], neg_log_p[i]),
            fontsize=7,
            ha="center",
            va="bottom",
            alpha=0.85,
        )

    ax.set_xlabel("Delta mean (group1 - group2)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.axhline(-np.log10(0.05), ls="--", lw=0.6, color="grey", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)
    log.info("volcano_plot saved", path=str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Pathway dot plot
# ---------------------------------------------------------------------------


def pathway_dot_plot(
    enrichment_path: PathLike,
    pathway_names_path: PathLike,
    output_path: PathLike,
    title: str,
    *,
    n_top: int = 20,
) -> Path:
    """Generate a dot plot of pathway enrichment results.

    Args:
        enrichment_path: Parquet with columns ``pathway``, ``q_value``,
            ``n_overlap``, ``odds_ratio``.
        pathway_names_path: Parquet mapping ``pathway_id`` to ``pathway_name``.
        output_path: Destination PNG path.
        title: Figure title.
        n_top: Number of most-significant pathways to display.

    Returns:
        Resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    enrich = pl.read_parquet(str(enrichment_path))
    names = pl.read_parquet(str(pathway_names_path))
    log.info("pathway_dot_plot", n_pathways=len(enrich), source=str(enrichment_path))

    # Join enrichment with human-readable names
    merged = enrich.join(names, left_on="pathway", right_on="pathway_id", how="left")
    merged = merged.with_columns(
        pl.when(pl.col("pathway_name").is_not_null())
        .then(pl.col("pathway_name"))
        .otherwise(pl.col("pathway"))
        .alias("display_name")
    )

    # Take top N by p_value (most significant first)
    top = merged.sort("p_value").head(n_top)
    if top.is_empty():
        log.warning("pathway_dot_plot: no pathways to plot")
        return output_path

    display_names = top["display_name"].to_list()
    neg_log_q = -np.log10(np.clip(top["q_value"].to_numpy(), a_min=1e-300, a_max=None))
    n_overlap = top["n_overlap"].to_numpy()
    odds = top["odds_ratio"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, max(4, n_top * 0.35)))

    scatter = ax.scatter(
        neg_log_q,
        range(len(display_names)),
        s=n_overlap * 12,
        c=odds,
        cmap="YlOrRd",
        edgecolors="black",
        linewidths=0.4,
        alpha=0.85,
    )

    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("-log10(q-value)")
    ax.set_title(title)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Odds ratio", fontsize=9)

    # Size legend
    for n_genes in [5, 15, 30]:
        ax.scatter([], [], s=n_genes * 12, c="grey", edgecolors="black", linewidths=0.4,
                   label=f"{n_genes} genes")
    ax.legend(title="Gene overlap", loc="lower right", fontsize=7, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("pathway_dot_plot saved", path=str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# HMS distribution
# ---------------------------------------------------------------------------


def hms_distribution(
    hms_scores_path: PathLike,
    output_path: PathLike,
    title: str,
) -> Path:
    """Generate a violin plot of hypermethylation score counts by clinical category.

    Args:
        hms_scores_path: Parquet with ``sample_id``, ``hms_count``,
            ``clinical_category``.
        output_path: Destination PNG path.
        title: Figure title.

    Returns:
        Resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(str(hms_scores_path))
    log.info("hms_distribution", n_samples=len(df), source=str(hms_scores_path))

    df = df.filter(
        pl.col("clinical_category").is_not_null()
        & (pl.col("clinical_category") != "excluded")
    )
    pdf = df.to_pandas()
    category_order = sorted(pdf["clinical_category"].unique())
    _base_palette = {
        "CRC": "#e74c3c",
        "Control": "#2ecc71",
        "polyps": "#3498db",
        "HGD": "#9b59b6",
        "other_cancer": "#95a5a6",
    }
    palette = {cat: _base_palette.get(cat, "#7f8c8d") for cat in category_order}

    fig, ax = plt.subplots(figsize=_FIGSIZE_WIDE)
    sns.violinplot(
        data=pdf,
        x="clinical_category",
        y="hms_count",
        hue="clinical_category",
        order=category_order,
        palette=palette,
        inner="box",
        cut=0,
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=pdf,
        x="clinical_category",
        y="hms_count",
        order=category_order,
        color="black",
        size=2,
        alpha=0.3,
        jitter=True,
        ax=ax,
    )

    ax.set_xlabel("Clinical Category")
    ax.set_ylabel("Hypermethylation Score (gene count)")
    ax.set_title(title)

    plt.tight_layout()
    fig.savefig(output_path, dpi=_DPI)
    plt.close(fig)
    log.info("hms_distribution saved", path=str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Gene heatmap
# ---------------------------------------------------------------------------


def gene_heatmap(
    gene_features_path: PathLike,
    metadata_path: PathLike,
    output_path: PathLike,
    *,
    n_top_genes: int = 50,
    comparison_path: PathLike | None = None,
) -> Path:
    """Generate a clustered heatmap of gene methylation values.

    Args:
        gene_features_path: Parquet with ``gene`` column and sample columns.
        metadata_path: Clinical metadata with ``barcode`` and
            ``clinical_category``.
        output_path: Destination PNG path.
        n_top_genes: Number of top genes to display.
        comparison_path: If provided, select genes by FDR significance from
            this comparison Parquet.

    Returns:
        Resolved output path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = pl.read_parquet(str(gene_features_path))
    meta = pl.read_parquet(str(metadata_path))
    log.info(
        "gene_heatmap",
        n_genes=len(features),
        n_samples=features.width - 1,
        source=str(gene_features_path),
    )

    # Filter to non-excluded samples present in both datasets
    meta = meta.filter(pl.col("clinical_category") != "excluded")
    sample_cols = [c for c in features.columns if c != "gene"]
    valid_samples = set(meta["barcode"].to_list()) & set(sample_cols)
    meta = meta.filter(pl.col("barcode").is_in(list(valid_samples)))

    # Select top genes
    if comparison_path is not None:
        comp = pl.read_parquet(str(comparison_path))
        top_genes = (
            comp.sort("q_value")
            .head(n_top_genes)["feature"]
            .to_list()
        )
    else:
        # Fallback: highest variance genes
        gene_names = features["gene"].to_list()
        data_matrix = features.select(list(valid_samples)).to_numpy()
        variances = np.nanvar(data_matrix, axis=1)
        top_idx = np.argsort(variances)[-n_top_genes:]
        top_genes = [gene_names[i] for i in top_idx]

    features = features.filter(pl.col("gene").is_in(top_genes))

    # Build sorted sample order by clinical category
    meta_sorted = meta.sort("clinical_category")
    ordered_samples = [s for s in meta_sorted["barcode"].to_list() if s in valid_samples]
    category_map = dict(
        zip(meta_sorted["barcode"].to_list(), meta_sorted["clinical_category"].to_list())
    )

    # Extract matrix and z-score per gene
    gene_labels = features["gene"].to_list()
    mat = features.select(ordered_samples).to_numpy().astype(float)
    row_mean = np.nanmean(mat, axis=1, keepdims=True)
    row_std = np.nanstd(mat, axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    z_mat = (mat - row_mean) / row_std
    z_mat = np.nan_to_num(z_mat, nan=0.0)

    # Clamp extreme z-scores for visual clarity
    z_mat = np.clip(z_mat, -3, 3)

    # Category colour bar
    palette = {"CRC": "#e74c3c", "Control": "#2ecc71", "polyps": "#3498db"}
    col_colors = [palette.get(category_map.get(s, ""), "#999999") for s in ordered_samples]

    fig, ax = plt.subplots(figsize=(14, max(6, n_top_genes * 0.22)))
    im = ax.imshow(z_mat, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)

    ax.set_yticks(range(len(gene_labels)))
    ax.set_yticklabels(gene_labels, fontsize=6)
    ax.set_xticks([])
    ax.set_xlabel("Samples (sorted by clinical category)")

    # Add colour bar for category at top
    for i, colour in enumerate(col_colors):
        ax.add_patch(plt.Rectangle((i - 0.5, -1.5), 1, 1, color=colour, clip_on=False))

    # Legend for categories
    from matplotlib.patches import Patch

    legend_handles = [Patch(facecolor=c, label=cat) for cat, c in palette.items()]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("Z-scored mean beta", fontsize=9)

    ax.set_title(f"Top {n_top_genes} Differentially Methylated Genes")
    plt.tight_layout()
    fig.savefig(output_path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("gene_heatmap saved", path=str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("visualise")
@click.option(
    "--output-dir",
    default="data/processed/figures",
    type=click.Path(),
    help="Directory where figures are saved.",
)
@click.option(
    "--comparisons-dir",
    default="data/processed/comparisons_full",
    type=click.Path(exists=True),
    help="Directory containing cohort comparison Parquet files.",
)
@click.option(
    "--enrichment-dir",
    default="data/processed/enrichment_full",
    type=click.Path(exists=True),
    help="Directory containing pathway enrichment Parquet files.",
)
@click.option(
    "--hms-dir",
    default="data/processed/hypermethylation",
    type=click.Path(exists=True),
    help="Directory containing HMS score Parquet files.",
)
@click.option(
    "--gene-features",
    default="data/processed/gene_features_full.parquet",
    type=click.Path(exists=True),
    help="Gene features matrix Parquet.",
)
@click.option(
    "--metadata",
    default="data/processed/clinical_metadata.parquet",
    type=click.Path(exists=True),
    help="Clinical metadata Parquet.",
)
@click.option(
    "--pathway-names",
    default="data/external/reactome_pathways.parquet",
    type=click.Path(exists=True),
    help="Pathway ID-to-name mapping Parquet.",
)
def main(
    output_dir: str,
    comparisons_dir: str,
    enrichment_dir: str,
    hms_dir: str,
    gene_features: str,
    metadata: str,
    pathway_names: str,
) -> None:
    """Generate all analysis visualisations from production data."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    comp_dir = Path(comparisons_dir)
    enrich_dir = Path(enrichment_dir)
    hms_d = Path(hms_dir)

    log.info("visualise.main", output_dir=str(out))

    # --- Volcano plots for every comparison ---
    for parquet in sorted(comp_dir.glob("*.parquet")):
        label = parquet.stem
        log.info("generating volcano", comparison=label)
        volcano_plot(
            parquet,
            out / f"volcano_{label}.png",
            f"{label.replace('_', ' ')} -- Gene-level Differential Methylation",
        )

    # --- Pathway dot plots for every enrichment ---
    for parquet in sorted(enrich_dir.glob("*.parquet")):
        label = parquet.stem
        log.info("generating pathway dot plot", enrichment=label)
        pathway_dot_plot(
            parquet,
            pathway_names,
            out / f"dotplot_{label}.png",
            f"{label.replace('_', ' ')}",
        )

    # --- HMS distributions ---
    for parquet in sorted(hms_d.glob("hms_scores_*.parquet")):
        label = parquet.stem
        q_tag = label.replace("hms_scores_", "")
        log.info("generating HMS distribution", quantile=q_tag)
        hms_distribution(
            parquet,
            out / f"hms_distribution_{q_tag}.png",
            f"Hypermethylation Score Distribution ({q_tag})",
        )

    # --- Gene heatmap for the primary comparison ---
    primary_comp = comp_dir / "CRC_vs_Control.parquet"
    if primary_comp.exists():
        log.info("generating gene heatmap")
        gene_heatmap(
            gene_features,
            metadata,
            out / "heatmap_CRC_vs_Control.png",
            n_top_genes=50,
            comparison_path=primary_comp,
        )

    log.info("visualise.main complete", output_dir=str(out))


if __name__ == "__main__":
    main()
