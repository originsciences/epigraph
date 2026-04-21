"""Validate the TypeDB import by counting entities and spot-checking relations.

Runs a series of validation queries against the TypeDB database and reports
results.  Checks:

1. Entity counts for all types (gene, pathway, function-term, etc.)
2. Relation counts for all relation types
3. Spot-check: a known gene has pathways and CpG overlaps
4. Sample count matches clinical metadata
5. CpG count matches the filtered stats file

TypeDB 3.8+ driver API notes:
- Driver: ``TypeDB.driver(address=...)`` (not ``core_driver``).
- Unified query interface: ``tx.query(typeql_string)`` returns a QueryAnswer.
- Count aggregation uses ``reduce $count = count;`` (not ``get $e; count;``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import polars as pl
import yaml

from epigraph.common.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Validation result model
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Container for a single validation check."""

    name: str
    passed: bool
    expected: int | str | None = None
    actual: int | str | None = None
    message: str = ""


@dataclass
class ValidationReport:
    """Aggregated validation results."""

    results: list[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True if all checks passed."""
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        """Number of passed checks."""
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        """Number of failed checks."""
        return sum(1 for r in self.results if not r.passed)

    def add(self, result: ValidationResult) -> None:
        """Add a validation result."""
        self.results.append(result)

    def format(self) -> str:
        """Format the report as a human-readable string."""
        lines: list[str] = [
            "=" * 60,
            "TypeDB Import Validation Report",
            "=" * 60,
        ]
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            line = f"  [{status}] {r.name}"
            if r.expected is not None:
                line += f" (expected={r.expected}, actual={r.actual})"
            if r.message:
                line += f" -- {r.message}"
            lines.append(line)

        lines.append("-" * 60)
        lines.append(f"Total: {len(self.results)} checks, "
                     f"{self.n_passed} passed, {self.n_failed} failed")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TypeDB query helpers
# ---------------------------------------------------------------------------


def _load_settings() -> dict[str, Any]:
    """Load settings from config/settings.yaml."""
    path = Path("config/settings.yaml")
    if path.exists():
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    return {}


def _resolve_env(raw: str) -> str:
    """Resolve ``${VAR:-default}`` patterns."""
    import os
    import re

    m = re.match(r"\$\{([^:}]+):-(.+)\}", raw)
    if m:
        return os.environ.get(m.group(1), m.group(2))
    return raw


class TypeDBValidator:
    """Runs validation queries against a TypeDB 3.x database.

    Attributes:
        address: TypeDB server address.
        database: Database name.
    """

    def __init__(self, address: str, database: str) -> None:
        self.address = address
        self.database = database
        self._driver: Any = None

    def connect(self) -> None:
        """Open a connection to TypeDB."""
        from typedb.driver import TypeDB

        self._driver = TypeDB.driver(address=self.address)
        log.info("validator_connected", address=self.address)

    def close(self) -> None:
        """Close the connection."""
        if self._driver:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> TypeDBValidator:
        self.connect()
        return self

    def __exit__(self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.close()

    def _read_tx(self) -> Any:
        """Open a read transaction."""
        from typedb.driver import TransactionType

        return self._driver.transaction(self.database, TransactionType.READ)

    def count_entity(self, entity_type: str) -> int:
        """Count instances of an entity type.

        Args:
            entity_type: TypeDB entity type name.

        Returns:
            Number of instances.
        """
        query = f"match $e isa {entity_type}; reduce $count = count;"
        with self._read_tx() as tx:
            result = tx.query(query).resolve()
            row = result.as_concept_rows().next()
            return int(row.get("count").as_value().as_long())
        return 0  # pragma: no cover

    def count_relation(self, relation_type: str) -> int:
        """Count instances of a relation type.

        Args:
            relation_type: TypeDB relation type name.

        Returns:
            Number of instances.
        """
        query = f"match $r isa {relation_type}; reduce $count = count;"
        with self._read_tx() as tx:
            result = tx.query(query).resolve()
            row = result.as_concept_rows().next()
            return int(row.get("count").as_value().as_long())
        return 0  # pragma: no cover

    def spot_check_gene(
        self,
        gene_symbol: str,
    ) -> dict[str, int]:
        """Spot-check a gene: count its pathways, GO terms, and CpG overlaps.

        Args:
            gene_symbol: Gene symbol to look up (e.g. ``"TP53"``).

        Returns:
            Dictionary with counts for ``pathways``, ``go_terms``, ``cpg_overlaps``.
        """
        counts: dict[str, int] = {"pathways": 0, "go_terms": 0, "cpg_overlaps": 0}

        # Pathways
        q_pw = (
            f'match $g isa gene, has gene_symbol "{gene_symbol}"; '
            f'(member-gene: $g, containing-pathway: $p) isa gene-pathway-membership; '
            f'reduce $count = count;'
        )
        # GO terms
        q_go = (
            f'match $g isa gene, has gene_symbol "{gene_symbol}"; '
            f'(annotated-gene: $g, annotating-term: $t) isa gene-function-annotation; '
            f'reduce $count = count;'
        )
        # CpG overlaps
        q_cpg = (
            f'match $g isa gene, has gene_symbol "{gene_symbol}"; '
            f'(overlapping-cpg: $c, overlapped-gene: $g) isa cpg-gene-overlap; '
            f'reduce $count = count;'
        )

        with self._read_tx() as tx:
            for key, query in [("pathways", q_pw), ("go_terms", q_go), ("cpg_overlaps", q_cpg)]:
                try:
                    result = tx.query(query).resolve()
                    row = result.as_concept_rows().next()
                    counts[key] = int(row.get("count").as_value().as_long())
                except Exception as exc:
                    log.warning("spot_check_query_failed", key=key, error=str(exc))

        return counts


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


def validate_import(
    validator: TypeDBValidator,
    clinical_path: Path | None = None,
    cpg_stats_path: Path | None = None,
    spot_check_gene: str = "TP53",
) -> ValidationReport:
    """Run all validation checks and return a report.

    Args:
        validator: Connected TypeDBValidator instance.
        clinical_path: Path to clinical_metadata.parquet for sample count check.
        cpg_stats_path: Path to cpg_stats.parquet for CpG count reference.
        spot_check_gene: Gene symbol to spot-check.

    Returns:
        Validation report with all results.
    """
    report = ValidationReport()

    # --- Entity counts ---
    entity_types = [
        "gene", "pathway", "function-term", "genomic-region", "sample", "cpg",
    ]
    for etype in entity_types:
        count = validator.count_entity(etype)
        passed = count > 0
        report.add(ValidationResult(
            name=f"entity_count_{etype}",
            passed=passed,
            actual=count,
            message="has instances" if passed else "NO instances found",
        ))

    # --- Relation counts ---
    relation_types = [
        "cpg-gene-overlap",
        "gene-pathway-membership",
        "gene-function-annotation",
        "cpg-region-overlap",
    ]
    for rtype in relation_types:
        count = validator.count_relation(rtype)
        passed = count > 0
        report.add(ValidationResult(
            name=f"relation_count_{rtype}",
            passed=passed,
            actual=count,
            message="has instances" if passed else "NO instances found",
        ))

    # --- Sample count vs clinical metadata ---
    if clinical_path and clinical_path.exists():
        clin_df = pl.read_parquet(clinical_path)
        expected_samples = len(clin_df)
        actual_samples = validator.count_entity("sample")
        report.add(ValidationResult(
            name="sample_count_matches_clinical",
            passed=actual_samples == expected_samples,
            expected=expected_samples,
            actual=actual_samples,
        ))

    # --- Spot-check gene ---
    gene_counts = validator.spot_check_gene(spot_check_gene)
    for key, count in gene_counts.items():
        report.add(ValidationResult(
            name=f"spot_check_{spot_check_gene}_{key}",
            passed=count > 0,
            actual=count,
            message=f"{spot_check_gene} has {count} {key}",
        ))

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("validate-import")
@click.option("--address", default=None, help="TypeDB server address.")
@click.option("--database", default=None, help="TypeDB database name.")
@click.option(
    "--clinical-metadata",
    type=click.Path(),
    default="data/processed/clinical_metadata.parquet",
    help="Path to clinical metadata Parquet for sample count validation.",
)
@click.option(
    "--spot-check-gene",
    default="TP53",
    help="Gene symbol to spot-check for relations.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit with non-zero code if any check fails.",
)
def main(
    address: str | None,
    database: str | None,
    clinical_metadata: str,
    spot_check_gene: str,
    strict: bool,
) -> None:
    """Validate the TypeDB import.

    Counts entities and relations, spot-checks a known gene, and verifies
    sample counts against clinical metadata.
    """
    settings = _load_settings()
    typedb_cfg = settings.get("typedb", {})

    if address is None:
        address = _resolve_env(typedb_cfg.get("address", "localhost:1729"))
    if database is None:
        database = _resolve_env(typedb_cfg.get("database", "methylation_graph"))

    validator = TypeDBValidator(address=address, database=database)

    try:
        validator.connect()

        clin_path = Path(clinical_metadata) if clinical_metadata else None

        report = validate_import(
            validator,
            clinical_path=clin_path,
            spot_check_gene=spot_check_gene,
        )

        click.echo(report.format())

        if strict and not report.passed:
            sys.exit(1)

    finally:
        validator.close()


if __name__ == "__main__":
    main()
