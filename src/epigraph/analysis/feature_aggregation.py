"""Aggregate CpG-level beta values to gene-level and pathway-level features.

This module bridges the raw beta matrix (stored as Parquet) with higher-level
biological features by mapping CpGs to genes (via genomic coordinates) and
genes to pathways / GO terms (via the TypeDB knowledge graph).

Typical pipeline::

    beta_parquet -> aggregate_cpgs_to_genes -> aggregate_genes_to_pathways
                                            -> aggregate_genes_to_terms

All outputs are Polars DataFrames and can be persisted to Parquet.
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any, Literal

import click
import duckdb
import numpy as np
import polars as pl

from epigraph.common.logging import get_logger

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

AggMethod = Literal["mean", "median", "weighted"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_GENE_BATCH_SIZE: int = 500
"""Number of genes processed per chunk to keep memory bounded."""

_CPG_COLUMN_RE = re.compile(r"^chr[0-9XYM]+_[0-9]+$")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_cpg_column_name(name: str) -> bool:
    """Check that a CpG column name matches the expected ``chr{N}_{pos}`` pattern.

    This prevents SQL injection via maliciously crafted column names that
    contain quote characters or other SQL metacharacters.

    Args:
        name: Column name to validate.

    Returns:
        ``True`` if the name is safe to interpolate into a DuckDB query.
    """
    return _CPG_COLUMN_RE.match(name) is not None


def _validate_agg_method(method: str) -> AggMethod:
    """Validate and return the aggregation method string.

    Args:
        method: One of ``"mean"``, ``"median"``, or ``"weighted"``.

    Returns:
        The validated method literal.

    Raises:
        ValueError: If *method* is not recognised.
    """
    valid: set[str] = {"mean", "median", "weighted"}
    if method not in valid:
        raise ValueError(f"Unknown aggregation method {method!r}. Choose from {valid}.")
    return method  # type: ignore[return-value]


def _aggregate_rows(
    matrix: pl.DataFrame,
    method: AggMethod,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Aggregate rows of a numeric matrix into a single row.

    Args:
        matrix: Polars DataFrame of shape (n_features, n_samples), all numeric.
        method: Aggregation strategy.
        weights: Optional weight vector of length ``n_features`` (only used
            when *method* is ``"weighted"``).

    Returns:
        1-D numpy array of length ``n_samples``.
    """
    arr = matrix.to_numpy()  # shape (n_features, n_samples)

    if arr.shape[0] == 0:
        return np.full(arr.shape[1], np.nan)

    if method == "mean":
        # All-NaN slices legitimately occur when a gene's CpGs are all missing
        # for a sample; nanmean's "Mean of empty slice" warning is noise —
        # the NaN result is the correct signal. Suppress locally.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(arr, axis=0)  # type: ignore[return-value]
    elif method == "median":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmedian(arr, axis=0)  # type: ignore[return-value]
    elif method == "weighted":
        if weights is None:
            raise ValueError("weights must be provided for method='weighted'.")
        if len(weights) != arr.shape[0]:
            raise ValueError(
                f"weights length ({len(weights)}) != number of features ({arr.shape[0]})."
            )
        # Weighted average ignoring NaNs
        masked = np.ma.MaskedArray(arr, mask=np.isnan(arr))
        w = np.asarray(weights, dtype=np.float64)
        weighted_sum = np.ma.average(masked, axis=0, weights=w)
        return np.asarray(weighted_sum, dtype=np.float64)  # type: ignore[return-value]
    else:
        raise ValueError(f"Unsupported method: {method!r}")


def _resolve_parquet_source(beta_path: str | Path) -> str:
    """Return a DuckDB-compatible parquet source expression.

    For a **single file**, returns the file path. For a **directory** of
    per-chromosome files, returns a glob — but note the glob only works for
    queries on shared columns (e.g. ``sample_id``), not for chromosome-specific
    CpG columns (which differ across files). Use :func:`_load_beta_columns`
    for CpG column queries.
    """
    path = Path(beta_path)
    if path.is_dir():
        return str(path / "*.parquet")
    return str(path)


def _build_chrom_file_index(beta_dir: Path) -> dict[str, Path]:
    """Map chromosome names to their Parquet file paths.

    Expects filenames like ``beta_chr1.parquet``, ``beta_chrX.parquet``.
    """
    index: dict[str, Path] = {}
    for pq_file in beta_dir.glob("beta_chr*.parquet"):
        chrom = pq_file.stem.replace("beta_", "")
        index[chrom] = pq_file
    return index


def _load_beta_columns(
    beta_path: str | Path,
    cpg_ids: list[str],
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    chrom_file_index: dict[str, Path] | None = None,
) -> pl.DataFrame:
    """Load a subset of CpG columns from beta Parquet file(s) using DuckDB.

    For a single Parquet file, selects columns directly. For a directory of
    per-chromosome files, groups requested CpGs by chromosome, queries each
    relevant file, and joins on sample_id.

    Args:
        beta_path: Single Parquet file or directory of per-chromosome files.
        cpg_ids: CpG column names to select.
        con: Optional DuckDB connection.
        chrom_file_index: Pre-built chrom→file mapping (avoids repeated glob).
    """
    if not cpg_ids:
        return pl.DataFrame()

    # Validate column names to prevent SQL injection
    safe_cpg_ids = [c for c in cpg_ids if _validate_cpg_column_name(c)]
    n_filtered = len(cpg_ids) - len(safe_cpg_ids)
    if n_filtered > 0:
        log.warning(
            "invalid_cpg_column_names_filtered",
            n_filtered=n_filtered,
            examples=[c for c in cpg_ids if not _validate_cpg_column_name(c)][:5],
        )
    if not safe_cpg_ids:
        return pl.DataFrame()
    cpg_ids = safe_cpg_ids

    own_con = con is None
    if con is None:
        con = duckdb.connect()
    # After this point ``con`` is guaranteed non-None — narrow the type.
    assert con is not None

    path = Path(beta_path)

    try:
        if not path.is_dir():
            # Single file: select directly
            quoted = ", ".join(f'"{c}"' for c in cpg_ids)
            query = f'SELECT {quoted} FROM read_parquet(\'{path}\')'  # noqa: S608
            return con.sql(query).pl()

        # Per-chromosome directory: group CpGs by chromosome, query each file
        if chrom_file_index is None:
            chrom_file_index = _build_chrom_file_index(path)

        from epigraph.common.genome_coords import parse_cpg_id_fast

        chrom_cpgs: dict[str, list[str]] = {}
        for cpg in cpg_ids:
            try:
                chrom, _ = parse_cpg_id_fast(cpg)
            except (ValueError, IndexError):
                continue
            chrom_cpgs.setdefault(chrom, []).append(cpg)

        # Query each chromosome file for its CpGs
        frames: list[pl.DataFrame] = []
        for chrom, cpg_list in chrom_cpgs.items():
            chrom_file = chrom_file_index.get(chrom)
            if chrom_file is None:
                continue
            quoted = ", ".join(f'"{c}"' for c in cpg_list)
            query = f'SELECT sample_id, {quoted} FROM read_parquet(\'{chrom_file}\')'  # noqa: S608
            frames.append(con.sql(query).pl())

        if not frames:
            return pl.DataFrame()

        # Join all chromosome results on sample_id
        result = frames[0]
        for frame in frames[1:]:
            result = result.join(frame, on="sample_id", how="inner")

        return result.drop("sample_id")

    finally:
        if own_con:
            con.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def aggregate_cpgs_to_genes(
    beta_parquet: str | Path,
    cpg_gene_mapping: pl.DataFrame,
    method: AggMethod = "mean",
    batch_size: int = DEFAULT_GENE_BATCH_SIZE,
    weights_col: str | None = None,
) -> pl.DataFrame:
    """Aggregate CpG-level beta values to gene-level methylation features.

    For each gene, the CpGs that overlap its genomic region are selected from
    the beta matrix and aggregated per sample using *method*.

    Args:
        beta_parquet: Path to the beta matrix Parquet file.  Columns are CpG
            IDs (``chr{N}_{pos}``), rows are samples.
        cpg_gene_mapping: DataFrame with at least columns ``cpg_id`` and
            ``gene_symbol``.  One row per CpG-gene pair (many-to-many).
        method: Aggregation strategy (``"mean"``, ``"median"``, or
            ``"weighted"``).
        batch_size: Number of genes to process per chunk to control memory
            usage.
        weights_col: Column name in *cpg_gene_mapping* containing per-CpG
            weights.  Required when *method* is ``"weighted"``.

    Returns:
        Polars DataFrame with columns ``["gene"] + sample_columns``, one row
        per gene.

    Raises:
        ValueError: On invalid *method* or missing weights column.
    """
    method = _validate_agg_method(method)
    beta_parquet = Path(beta_parquet)

    if method == "weighted" and weights_col is None:
        raise ValueError("weights_col must be specified when method='weighted'.")

    # Build gene -> [cpg_ids] lookup (deduplicated per gene)
    gene_cpgs: dict[str, list[str]] = {}
    for row in (
        cpg_gene_mapping.select("gene_symbol", "cpg_id")
        .unique()
        .group_by("gene_symbol")
        .agg(pl.col("cpg_id"))
        .iter_rows(named=True)
    ):
        gene_cpgs[row["gene_symbol"]] = row["cpg_id"]

    if method == "weighted" and weights_col is not None:
        cpg_weights: dict[str, float] = dict(
            zip(
                cpg_gene_mapping["cpg_id"].to_list(),
                cpg_gene_mapping[weights_col].to_list(),
            )
        )
    else:
        cpg_weights = {}

    genes = sorted(gene_cpgs.keys())
    n_batches = math.ceil(len(genes) / batch_size)
    log.info(
        "aggregating_cpgs_to_genes",
        n_genes=len(genes),
        method=method,
        batch_size=batch_size,
        n_batches=n_batches,
    )

    # Get sample IDs and available CpG columns from the Parquet file(s)
    con = duckdb.connect()
    chrom_file_index: dict[str, Path] | None = None
    try:
        if beta_parquet.is_dir():
            # Per-chromosome directory: collect all columns from all files,
            # read sample_ids from the first file
            chrom_file_index = _build_chrom_file_index(beta_parquet)
            all_columns: set[str] = set()
            sample_ids: list[str] = []
            for i, (chrom, pq_file) in enumerate(sorted(chrom_file_index.items())):
                schema_df = con.sql(
                    f"SELECT * FROM read_parquet('{pq_file}') LIMIT 0"  # noqa: S608
                ).pl()
                all_columns.update(schema_df.columns)
                if i == 0:
                    sample_ids = con.sql(
                        f"SELECT sample_id FROM read_parquet('{pq_file}')"  # noqa: S608
                    ).pl()["sample_id"].to_list()
            log.info("chrom_index_loaded", n_files=len(chrom_file_index), n_columns=len(all_columns))
        else:
            # Single file
            schema_df = con.sql(
                f"SELECT * FROM read_parquet('{beta_parquet}') LIMIT 0"  # noqa: S608
            ).pl()
            all_columns = set(schema_df.columns)
            sample_id_col = "sample_id" if "sample_id" in all_columns else schema_df.columns[0]
            sample_ids = con.sql(
                f'SELECT "{sample_id_col}" FROM read_parquet(\'{beta_parquet}\')'  # noqa: S608
            ).pl()[sample_id_col].to_list()

        result_rows: list[dict[str, Any]] = []

        for batch_idx in range(n_batches):
            batch_genes = genes[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            # Collect all unique CpG IDs needed for this batch
            batch_cpgs: set[str] = set()
            for gene in batch_genes:
                batch_cpgs.update(gene_cpgs[gene])

            # Filter to CpGs actually present in the beta matrix
            available_cpgs = [c for c in batch_cpgs if c in all_columns]
            if not available_cpgs:
                log.debug("batch_no_available_cpgs", batch=batch_idx)
                continue

            # Load the needed CpG columns
            beta_subset = _load_beta_columns(
                beta_parquet, available_cpgs, con=con,
                chrom_file_index=chrom_file_index,
            )

            beta_cols_set = set(beta_subset.columns)
            for gene in batch_genes:
                gene_cpg_ids = list(dict.fromkeys(
                    c for c in gene_cpgs[gene] if c in beta_cols_set
                ))
                if not gene_cpg_ids:
                    continue

                # beta_subset is (n_samples, n_cpgs); _aggregate_rows expects
                # (n_features, n_samples) so we transpose: CpGs become rows,
                # samples become columns.
                cpg_matrix = beta_subset.select(gene_cpg_ids).transpose()

                weights = None
                if method == "weighted":
                    weights = np.array([cpg_weights[c] for c in gene_cpg_ids])

                agg_values = _aggregate_rows(cpg_matrix, method, weights)

                row: dict[str, Any] = {"gene": gene}
                # Use actual sample IDs as column names so downstream cohort
                # comparison can match them to clinical metadata
                row.update(
                    {sid: float(v) for sid, v in zip(sample_ids, agg_values)}
                )
                result_rows.append(row)

            log.debug(
                "batch_complete",
                batch=batch_idx + 1,
                n_batches=n_batches,
                genes_in_batch=len(batch_genes),
            )
    finally:
        con.close()

    if not result_rows:
        log.warning("no_genes_aggregated")
        return pl.DataFrame({"gene": []})

    return pl.DataFrame(result_rows)


def aggregate_genes_to_pathways(
    gene_matrix: pl.DataFrame,
    gene_pathway_mapping: pl.DataFrame,
    method: AggMethod = "mean",
) -> pl.DataFrame:
    """Aggregate gene-level methylation features to pathway-level features.

    Args:
        gene_matrix: DataFrame with a ``"gene"`` column and sample columns,
            as returned by :func:`aggregate_cpgs_to_genes`.
        gene_pathway_mapping: DataFrame with columns ``"gene_symbol"`` and
            ``"pathway_id"`` (and optionally ``"pathway_name"``).
        method: Aggregation strategy.

    Returns:
        Polars DataFrame with columns ``["pathway"] + sample_columns``.
    """
    method = _validate_agg_method(method)

    sample_cols = [c for c in gene_matrix.columns if c != "gene"]
    pathway_genes: dict[str, list[str]] = {
        row[0]: row[1]
        for row in gene_pathway_mapping.group_by("pathway_id")
        .agg(pl.col("gene_symbol"))
        .iter_rows()
    }

    log.info("aggregating_genes_to_pathways", n_pathways=len(pathway_genes), method=method)

    gene_set = set(gene_matrix["gene"].to_list())
    result_rows: list[dict[str, Any]] = []

    for pathway_id, member_genes in sorted(pathway_genes.items()):
        available = [g for g in member_genes if g in gene_set]
        if not available:
            continue

        subset = gene_matrix.filter(pl.col("gene").is_in(available)).select(sample_cols)
        agg_values = _aggregate_rows(subset, method)

        row: dict[str, Any] = {"pathway": pathway_id}
        row.update({col: float(v) for col, v in zip(sample_cols, agg_values)})
        result_rows.append(row)

    if not result_rows:
        log.warning("no_pathways_aggregated")
        return pl.DataFrame({"pathway": []})

    return pl.DataFrame(result_rows)


def aggregate_genes_to_terms(
    gene_matrix: pl.DataFrame,
    gene_term_mapping: pl.DataFrame,
    method: AggMethod = "mean",
) -> pl.DataFrame:
    """Aggregate gene-level methylation features to GO / function term features.

    Args:
        gene_matrix: DataFrame with a ``"gene"`` column and sample columns,
            as returned by :func:`aggregate_cpgs_to_genes`.
        gene_term_mapping: DataFrame with columns ``"gene_symbol"`` and
            ``"term_id"`` (and optionally ``"term_name"``).
        method: Aggregation strategy.

    Returns:
        Polars DataFrame with columns ``["term"] + sample_columns``.
    """
    method = _validate_agg_method(method)

    sample_cols = [c for c in gene_matrix.columns if c != "gene"]
    term_genes: dict[str, list[str]] = {
        row[0]: row[1]
        for row in gene_term_mapping.group_by("term_id")
        .agg(pl.col("gene_symbol"))
        .iter_rows()
    }

    log.info("aggregating_genes_to_terms", n_terms=len(term_genes), method=method)

    gene_set = set(gene_matrix["gene"].to_list())
    result_rows: list[dict[str, Any]] = []

    for term_id, member_genes in sorted(term_genes.items()):
        available = [g for g in member_genes if g in gene_set]
        if not available:
            continue

        subset = gene_matrix.filter(pl.col("gene").is_in(available)).select(sample_cols)
        agg_values = _aggregate_rows(subset, method)

        row: dict[str, Any] = {"term": term_id}
        row.update({col: float(v) for col, v in zip(sample_cols, agg_values)})
        result_rows.append(row)

    if not result_rows:
        log.warning("no_terms_aggregated")
        return pl.DataFrame({"term": []})

    return pl.DataFrame(result_rows)


def write_results(
    df: pl.DataFrame,
    output_path: str | Path,
    *,
    typedb_driver: Any | None = None,
    feature_type: str = "gene",
) -> None:
    """Write aggregated feature matrix to Parquet and optionally to TypeDB.

    Args:
        df: Aggregated feature DataFrame.
        output_path: Destination Parquet file path.
        typedb_driver: Optional TypeDB driver connection for writing
            derived-methylation-feature relations.
        feature_type: Label for the feature type (``"gene"``, ``"pathway"``,
            or ``"term"``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)
    log.info("results_written", path=str(output_path), rows=df.height, feature_type=feature_type)

    if typedb_driver is not None:
        # TODO: Implement TypeDB write for derived-methylation-feature relations.
        # This should create relation instances linking each aggregated feature
        # value back to its constituent CpGs/genes and the sample.
        log.warning("typedb_export_not_implemented", feature_type=feature_type)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("derive-features")
@click.option(
    "--beta-parquet",
    required=True,
    type=click.Path(exists=True),
    help="Path to the beta values Parquet file or directory of per-chrom files.",
)
@click.option(
    "--cpg-gene-mapping",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet file with cpg_id -> gene_symbol mapping.",
)
@click.option(
    "--gene-pathway-mapping",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet file with gene_symbol -> pathway_id mapping.",
)
@click.option(
    "--gene-term-mapping",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Parquet file with gene_symbol -> term_id mapping.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory to write output Parquet files.",
)
@click.option(
    "--method",
    type=click.Choice(["mean", "median", "weighted"]),
    default="mean",
    show_default=True,
    help="Aggregation method.",
)
@click.option(
    "--batch-size",
    type=int,
    default=DEFAULT_GENE_BATCH_SIZE,
    show_default=True,
    help="Number of genes per processing batch.",
)
def main(
    beta_parquet: str,
    cpg_gene_mapping: str,
    gene_pathway_mapping: str | None,
    gene_term_mapping: str | None,
    output_dir: str,
    method: str,
    batch_size: int,
) -> None:
    """Aggregate CpG beta values to gene, pathway, and GO term features.

    Reads the beta matrix and CpG-gene mapping, computes gene-level
    methylation features, and optionally aggregates further to pathway
    and GO term levels.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    agg_method = _validate_agg_method(method)

    log.info("feature_aggregation_start", method=agg_method, output_dir=str(output))

    # --- Gene-level aggregation ---
    mapping_df = pl.read_parquet(cpg_gene_mapping)
    gene_matrix = aggregate_cpgs_to_genes(
        beta_parquet=beta_parquet,
        cpg_gene_mapping=mapping_df,
        method=agg_method,
        batch_size=batch_size,
    )
    write_results(gene_matrix, output / "gene_features.parquet", feature_type="gene")

    # --- Pathway-level aggregation ---
    if gene_pathway_mapping is not None:
        pathway_mapping_df = pl.read_parquet(gene_pathway_mapping)
        pathway_matrix = aggregate_genes_to_pathways(
            gene_matrix=gene_matrix,
            gene_pathway_mapping=pathway_mapping_df,
            method=agg_method,
        )
        write_results(pathway_matrix, output / "pathway_features.parquet", feature_type="pathway")

    # --- GO term-level aggregation ---
    if gene_term_mapping is not None:
        term_mapping_df = pl.read_parquet(gene_term_mapping)
        term_matrix = aggregate_genes_to_terms(
            gene_matrix=gene_matrix,
            gene_term_mapping=term_mapping_df,
            method=agg_method,
        )
        write_results(term_matrix, output / "term_features.parquet", feature_type="term")

    log.info("feature_aggregation_complete")


if __name__ == "__main__":
    main()
