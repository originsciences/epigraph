"""Import parsed data into TypeDB 3.x.

Loads genes, pathways, function terms, genomic regions, samples, CpGs, and
their relations into a TypeDB 3.x database.  Uses batched transactions for
performance and supports idempotent re-runs.

TypeDB 3.8+ driver API notes:
- Driver: ``TypeDB.driver(address=...)`` (not ``core_driver``).
- No sessions; transactions are opened directly on the driver.
- ``driver.transaction(database, transaction_type)``
- Unified query interface: ``tx.query(typeql_string)`` returns a QueryAnswer.
  Call ``.resolve()`` to materialise the result.
- Count aggregation uses ``reduce $count = count;`` (not ``get $e; count;``).

Import order:
1. Genes
2. Pathways
3. Function terms (GO)
4. Genomic regions (CpG islands)
5. Gene-pathway relations
6. Gene-function relations
7. Samples
8. CpGs (variance-filtered)
9. CpG-gene overlap relations
10. CpG-region overlap relations
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click
import polars as pl
import yaml

from epigraph.common.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _load_settings() -> dict[str, Any]:
    """Load TypeDB connection settings from config/settings.yaml."""
    settings_path = Path("config/settings.yaml")
    if settings_path.exists():
        with open(settings_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        return cfg
    return {}


def _resolve_env_default(raw: str) -> str:
    """Resolve ``${VAR:-default}`` patterns."""
    import os
    import re

    m = re.match(r"\$\{([^:}]+):-(.+)\}", raw)
    if m:
        return os.environ.get(m.group(1), m.group(2))
    return raw


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------


def _batched(iterable: list[Any], size: int) -> Iterator[list[Any]]:
    """Yield successive *size*-length slices from *iterable*."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


# ---------------------------------------------------------------------------
# TypeDB driver wrapper
# ---------------------------------------------------------------------------


class TypeDBImporter:
    """Manages a TypeDB connection and provides batch-insert methods.

    Attributes:
        address: TypeDB server address (host:port).
        database: TypeDB database name.
        batch_size: Number of entities per transaction.
        driver: TypeDB driver instance (lazy-initialised).
    """

    def __init__(
        self,
        address: str = "localhost:1729",
        database: str = "methylation_graph",
        batch_size: int = 1000,
    ) -> None:
        self.address = address
        self.database = database
        self.batch_size = batch_size
        self._driver: Any = None

    # -- Connection lifecycle -------------------------------------------------

    def connect(self) -> None:
        """Open a connection to the TypeDB server."""
        from typedb.driver import TypeDB

        log.info("typedb_connecting", address=self.address, database=self.database)
        self._driver = TypeDB.driver(address=self.address)

    def close(self) -> None:
        """Close the TypeDB connection."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            log.info("typedb_disconnected")

    def __enter__(self) -> TypeDBImporter:
        self.connect()
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    @property
    def driver(self) -> Any:
        """Return the active driver, raising if not connected."""
        if self._driver is None:
            raise RuntimeError("TypeDB driver not connected. Call connect() first.")
        return self._driver

    # -- Transaction helpers --------------------------------------------------

    def _write_tx(self) -> Any:
        """Open a write transaction on the configured database."""
        from typedb.driver import TransactionType

        return self.driver.transaction(self.database, TransactionType.WRITE)

    def _read_tx(self) -> Any:
        """Open a read transaction on the configured database."""
        from typedb.driver import TransactionType

        return self.driver.transaction(self.database, TransactionType.READ)

    def _batch_insert(
        self,
        queries: list[str],
        label: str,
    ) -> int:
        """Execute insert queries in batches.

        Args:
            queries: List of TypeQL insert statements.
            label: Label for progress logging.

        Returns:
            Number of successfully inserted queries.
        """
        total = len(queries)
        inserted = 0
        failed_batches = 0

        for batch_idx, batch in enumerate(_batched(queries, self.batch_size)):
            try:
                with self._write_tx() as tx:
                    for query in batch:
                        tx.query(query).resolve()
                    tx.commit()
                    inserted += len(batch)
            except Exception as exc:
                failed_batches += 1
                log.error(
                    "batch_insert_failed",
                    label=label,
                    batch=batch_idx,
                    batch_size=len(batch),
                    failed_batches=failed_batches,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

            if (batch_idx + 1) % 10 == 0 or inserted == total:
                log.info(
                    "import_progress",
                    label=label,
                    inserted=inserted,
                    total=total,
                    pct=f"{inserted / total * 100:.1f}%",
                )

        if failed_batches:
            log.warning(
                "batch_insert_summary",
                label=label,
                failed_batches=failed_batches,
                total_batches=(total + self.batch_size - 1) // self.batch_size,
                inserted=inserted,
                total=total,
            )

        return inserted

    # -- Entity importers -----------------------------------------------------

    def import_genes(self, genes_path: Path) -> int:
        """Import gene entities from Parquet.

        Args:
            genes_path: Path to genes.parquet.

        Returns:
            Number of genes inserted.
        """
        df = pl.read_parquet(genes_path)
        log.info("importing_genes", n=len(df))

        queries: list[str] = []
        for row in df.iter_rows(named=True):
            gene_id = row["gene_id"]
            symbol = row.get("gene_symbol", "")
            chrom = row.get("chrom", "")
            start = row.get("start", 0)
            end = row.get("end", 0)
            strand = row.get("strand", ".")
            biotype = row.get("biotype", "")

            q = (
                f'insert $g isa gene, '
                f'has ensembl_gene_id "{_escape(gene_id)}", '
                f'has gene_symbol "{_escape(symbol)}", '
                f'has chromosome "{_escape(chrom)}", '
                f'has start_position {start}, '
                f'has end_position {end}, '
                f'has strand "{_escape(strand)}", '
                f'has biotype "{_escape(biotype)}";'
            )
            queries.append(q)

        return self._batch_insert(queries, "genes")

    def import_pathways(self, pathways_path: Path) -> int:
        """Import pathway entities from Parquet.

        Args:
            pathways_path: Path to reactome_pathways.parquet.

        Returns:
            Number of pathways inserted.
        """
        df = pl.read_parquet(pathways_path)
        log.info("importing_pathways", n=len(df))

        queries: list[str] = []
        for row in df.iter_rows(named=True):
            pid = row["pathway_id"]
            name = _escape(row.get("pathway_name", ""))
            source = row.get("pathway_source", "Reactome")

            q = (
                f'insert $p isa pathway, '
                f'has pathway_id "{_escape(pid)}", '
                f'has pathway_name "{name}", '
                f'has pathway_source "{_escape(source)}";'
            )
            queries.append(q)

        return self._batch_insert(queries, "pathways")

    def import_function_terms(self, go_path: Path) -> int:
        """Import GO function-term entities from Parquet.

        Extracts unique GO terms and inserts them as function-term entities.

        Args:
            go_path: Path to go_annotations.parquet.

        Returns:
            Number of terms inserted.
        """
        df = pl.read_parquet(go_path)
        terms = df.select(["go_id", "aspect"]).unique(subset=["go_id"])
        log.info("importing_function_terms", n=len(terms))

        queries: list[str] = []
        for row in terms.iter_rows(named=True):
            tid = row["go_id"]
            namespace = row.get("aspect", "")

            q = (
                f'insert $t isa function-term, '
                f'has term_id "{_escape(tid)}", '
                f'has term_name "{_escape(tid)}", '
                f'has term_namespace "{_escape(namespace)}";'
            )
            queries.append(q)

        return self._batch_insert(queries, "function_terms")

    def import_genomic_regions(self, islands_path: Path) -> int:
        """Import CpG island genomic-region entities from Parquet.

        Args:
            islands_path: Path to cpg_islands.parquet.

        Returns:
            Number of regions inserted.
        """
        df = pl.read_parquet(islands_path)
        log.info("importing_genomic_regions", n=len(df))

        queries: list[str] = []
        for row in df.iter_rows(named=True):
            rid = row["region_id"]
            chrom = row["chrom"]
            start = row["start"]
            end = row["end"]

            q = (
                f'insert $r isa genomic-region, '
                f'has region_id "{_escape(rid)}", '
                f'has region_type "cpg_island", '
                f'has chromosome "{_escape(chrom)}", '
                f'has start_position {start}, '
                f'has end_position {end};'
            )
            queries.append(q)

        return self._batch_insert(queries, "genomic_regions")

    def import_gene_pathway_relations(self, mapping_path: Path) -> int:
        """Import gene-pathway-membership relations.

        Args:
            mapping_path: Path to reactome_gene_pathway.parquet.

        Returns:
            Number of relations inserted.
        """
        df = pl.read_parquet(mapping_path)
        log.info("importing_gene_pathway_relations", n=len(df))

        queries: list[str] = []
        for row in df.iter_rows(named=True):
            gene_id = row["gene_id"]
            pathway_id = row["pathway_id"]

            q = (
                f'match '
                f'$g isa gene, has ensembl_gene_id "{_escape(gene_id)}"; '
                f'$p isa pathway, has pathway_id "{_escape(pathway_id)}"; '
                f'insert '
                f'(member-gene: $g, containing-pathway: $p) isa gene-pathway-membership, '
                f'has annotation_source "Reactome";'
            )
            queries.append(q)

        return self._batch_insert(queries, "gene_pathway_relations")

    def import_gene_function_relations(self, go_path: Path) -> int:
        """Import gene-function-annotation relations from GO annotations.

        Args:
            go_path: Path to go_annotations.parquet.

        Returns:
            Number of relations inserted.
        """
        df = pl.read_parquet(go_path)
        log.info("importing_gene_function_relations", n=len(df))

        # We need to match genes by symbol since GOA uses symbols
        queries: list[str] = []
        for row in df.iter_rows(named=True):
            symbol = _escape(row["gene_symbol"])
            go_id = row["go_id"]
            evidence = row.get("evidence_code", "")

            q = (
                f'match '
                f'$g isa gene, has gene_symbol "{symbol}"; '
                f'$t isa function-term, has term_id "{_escape(go_id)}"; '
                f'insert '
                f'(annotated-gene: $g, annotating-term: $t) isa gene-function-annotation, '
                f'has evidence_code "{_escape(evidence)}", '
                f'has annotation_source "GOA";'
            )
            queries.append(q)

        return self._batch_insert(queries, "gene_function_relations")

    def import_samples(self, clinical_path: Path) -> int:
        """Import sample entities from clinical metadata Parquet.

        Args:
            clinical_path: Path to clinical_metadata.parquet.

        Returns:
            Number of samples inserted.
        """
        df = pl.read_parquet(clinical_path)
        log.info("importing_samples", n=len(df))

        queries: list[str] = []
        for row in df.iter_rows(named=True):
            barcode = row["barcode"]
            category = _escape(row.get("clinical_category", "unknown"))

            q = (
                f'insert $s isa sample, '
                f'has barcode "{_escape(barcode)}", '
                f'has clinical_category "{category}";'
            )
            queries.append(q)

        return self._batch_insert(queries, "samples")

    def import_cpgs(
        self,
        stats_path: Path,
        min_variance: float = 0.01,
        min_coverage: float = 0.5,
    ) -> int:
        """Import variance-filtered CpG entities from stats Parquet.

        Args:
            stats_path: Path to cpg_stats.parquet.
            min_variance: Minimum beta variance to include.
            min_coverage: Minimum fraction of non-missing samples.

        Returns:
            Number of CpGs inserted.
        """
        df = pl.read_parquet(stats_path)

        # Filter by variance and coverage
        filtered = df.filter(
            (pl.col("variance") >= min_variance)
            & (pl.col("missingness") <= (1.0 - min_coverage))
        )
        log.info("importing_cpgs", total=len(df), filtered=len(filtered))

        queries: list[str] = []
        for row in filtered.iter_rows(named=True):
            cpg_id = row["cpg_id"]
            chrom = row["chromosome"]
            pos = row["position"]
            mean_beta = row["mean_beta"]
            variance = row["variance"]
            missingness = row["missingness"]
            n_samples = row["n_samples"]

            q = (
                f'insert $c isa cpg, '
                f'has cpg_id "{_escape(cpg_id)}", '
                f'has chromosome "{_escape(chrom)}", '
                f'has position {pos}, '
                f'has mean_beta {mean_beta}, '
                f'has beta_variance {variance}, '
                f'has missingness_fraction {missingness}, '
                f'has n_samples {n_samples};'
            )
            queries.append(q)

        return self._batch_insert(queries, "cpgs")

    def import_cpg_gene_overlaps(self, mapping_path: Path) -> int:
        """Import CpG-gene overlap relations.

        Args:
            mapping_path: Path to cpg_gene_mapping.parquet.

        Returns:
            Number of relations inserted.
        """
        df = pl.read_parquet(mapping_path)
        # Only import overlaps for CpGs that were actually imported (non-intergenic)
        df = df.filter(pl.col("overlap_type") != "intergenic")
        log.info("importing_cpg_gene_overlaps", n=len(df))

        queries: list[str] = []
        for row in df.iter_rows(named=True):
            cpg_id = row["cpg_id"]
            gene_id = row["gene_id"]
            overlap_type = row["overlap_type"]

            q = (
                f'match '
                f'$c isa cpg, has cpg_id "{_escape(cpg_id)}"; '
                f'$g isa gene, has ensembl_gene_id "{_escape(gene_id)}"; '
                f'insert '
                f'(overlapping-cpg: $c, overlapped-gene: $g) isa cpg-gene-overlap, '
                f'has overlap_type "{_escape(overlap_type)}";'
            )
            queries.append(q)

        return self._batch_insert(queries, "cpg_gene_overlaps")

    def import_cpg_region_overlaps(
        self,
        cpg_stats_path: Path,
        islands_path: Path,
    ) -> int:
        """Import CpG-region overlap relations for CpG islands.

        For each imported CpG, checks whether it falls within a CpG island
        region and creates the overlap relation.

        Args:
            cpg_stats_path: Path to cpg_stats.parquet (for CpG coordinates).
            islands_path: Path to cpg_islands.parquet.

        Returns:
            Number of relations inserted.

        TODO: This is a naive O(n*m) approach.  For production, use an
        interval tree or pybedtools intersect for efficiency.
        """
        cpg_df = pl.read_parquet(cpg_stats_path)
        islands_df = pl.read_parquet(islands_path)

        log.info(
            "computing_cpg_region_overlaps",
            n_cpgs=len(cpg_df),
            n_islands=len(islands_df),
        )

        # Build island lookup by chromosome
        islands_by_chrom: dict[str, list[tuple[str, int, int]]] = {}
        for row in islands_df.iter_rows(named=True):
            chrom = row["chrom"]
            if chrom not in islands_by_chrom:
                islands_by_chrom[chrom] = []
            islands_by_chrom[chrom].append((row["region_id"], row["start"], row["end"]))

        # Sort each chromosome's islands by start
        for chrom in islands_by_chrom:
            islands_by_chrom[chrom].sort(key=lambda x: x[1])

        queries: list[str] = []
        for row in cpg_df.iter_rows(named=True):
            chrom = row["chromosome"]
            pos = row["position"]
            cpg_id = row["cpg_id"]

            chrom_islands = islands_by_chrom.get(chrom, [])
            for region_id, istart, iend in chrom_islands:
                if istart <= pos < iend:
                    q = (
                        f'match '
                        f'$c isa cpg, has cpg_id "{_escape(cpg_id)}"; '
                        f'$r isa genomic-region, has region_id "{_escape(region_id)}"; '
                        f'insert '
                        f'(overlapping-cpg: $c, overlapped-region: $r) isa cpg-region-overlap;'
                    )
                    queries.append(q)
                    break  # A CpG falls in at most one island
                elif istart > pos:
                    break  # sorted, no further islands can overlap

        log.info("cpg_region_overlaps_computed", n_overlaps=len(queries))
        return self._batch_insert(queries, "cpg_region_overlaps")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escape(value: str) -> str:
    """Escape special characters in a string for TypeQL double-quoted literals."""
    return (
        value
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("import-typedb")
@click.option(
    "--address",
    default=None,
    help="TypeDB server address (host:port). Defaults to settings.yaml.",
)
@click.option(
    "--database",
    default=None,
    help="TypeDB database name. Defaults to settings.yaml.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Entities per transaction. Defaults to settings.yaml.",
)
@click.option(
    "--data-dir",
    type=click.Path(exists=True, file_okay=False),
    default="data",
    help="Base data directory.",
)
@click.option(
    "--min-variance",
    type=float,
    default=None,
    help="Minimum CpG variance for import. Defaults to settings.yaml.",
)
@click.option(
    "--min-coverage",
    type=float,
    default=None,
    help="Minimum sample coverage fraction. Defaults to settings.yaml.",
)
@click.option(
    "--step",
    type=click.Choice([
        "all", "genes", "pathways", "function_terms", "genomic_regions",
        "gene_pathway", "gene_function", "samples", "cpgs",
        "cpg_gene_overlaps", "cpg_region_overlaps",
    ]),
    default="all",
    help="Run a single import step (default: all).",
)
def main(
    address: str | None,
    database: str | None,
    batch_size: int | None,
    data_dir: str,
    min_variance: float | None,
    min_coverage: float | None,
    step: str,
) -> None:
    """Import parsed data into TypeDB 3.x.

    Loads genes, pathways, GO terms, genomic regions, samples, CpGs, and
    their relations in the correct order.  Supports step-by-step execution
    for debugging.
    """
    settings = _load_settings()
    typedb_cfg = settings.get("typedb", {})
    proc_cfg = settings.get("processing", {})

    if address is None:
        address = _resolve_env_default(typedb_cfg.get("address", "localhost:1729"))
    if database is None:
        database = _resolve_env_default(typedb_cfg.get("database", "methylation_graph"))
    if batch_size is None:
        batch_size = typedb_cfg.get("batch_size", 1000)
    if min_variance is None:
        min_variance = proc_cfg.get("min_variance", 0.01)
    if min_coverage is None:
        min_coverage = proc_cfg.get("min_coverage", 0.5)

    data = Path(data_dir)
    ext = data / "external"
    proc = data / "processed"

    importer = TypeDBImporter(
        address=address,
        database=database,
        batch_size=batch_size,
    )

    steps_to_run: list[str]
    if step == "all":
        steps_to_run = [
            "genes", "pathways", "function_terms", "genomic_regions",
            "gene_pathway", "gene_function", "samples", "cpgs",
            "cpg_gene_overlaps", "cpg_region_overlaps",
        ]
    else:
        steps_to_run = [step]

    try:
        importer.connect()

        results: dict[str, int] = {}

        for s in steps_to_run:
            log.info("import_step_start", step=s)

            if s == "genes":
                results[s] = importer.import_genes(ext / "genes.parquet")
            elif s == "pathways":
                results[s] = importer.import_pathways(ext / "reactome_pathways.parquet")
            elif s == "function_terms":
                results[s] = importer.import_function_terms(ext / "go_annotations.parquet")
            elif s == "genomic_regions":
                results[s] = importer.import_genomic_regions(ext / "cpg_islands.parquet")
            elif s == "gene_pathway":
                results[s] = importer.import_gene_pathway_relations(
                    ext / "reactome_gene_pathway.parquet"
                )
            elif s == "gene_function":
                results[s] = importer.import_gene_function_relations(
                    ext / "go_annotations.parquet"
                )
            elif s == "samples":
                results[s] = importer.import_samples(proc / "clinical_metadata.parquet")
            elif s == "cpgs":
                results[s] = importer.import_cpgs(
                    proc / "cpg_stats.parquet",
                    min_variance=min_variance,
                    min_coverage=min_coverage,
                )
            elif s == "cpg_gene_overlaps":
                results[s] = importer.import_cpg_gene_overlaps(
                    proc / "cpg_gene_mapping.parquet"
                )
            elif s == "cpg_region_overlaps":
                results[s] = importer.import_cpg_region_overlaps(
                    proc / "cpg_stats.parquet",
                    ext / "cpg_islands.parquet",
                )

            click.echo(f"  {s}: {results.get(s, 0)} inserted")

        log.info("import_complete", results=results)
        click.echo("Import complete.")

    finally:
        importer.close()


if __name__ == "__main__":
    main()
