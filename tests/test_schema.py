"""Tests for TypeDB schema and function definitions.

Validates that ``schema.tql`` and ``functions.tql`` exist, contain valid
TypeQL constructs, and define all expected entity and relation types.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas" / "typedb"
SCHEMA_PATH = SCHEMA_DIR / "schema.tql"
FUNCTIONS_PATH = SCHEMA_DIR / "functions.tql"


# =========================================================================
# File existence
# =========================================================================


class TestSchemaFilesExist:
    """Verify that schema files are present in the repository."""

    def test_schema_tql_exists(self) -> None:
        assert SCHEMA_PATH.exists(), f"schema.tql not found at {SCHEMA_PATH}"

    def test_functions_tql_exists(self) -> None:
        assert FUNCTIONS_PATH.exists(), f"functions.tql not found at {FUNCTIONS_PATH}"


# =========================================================================
# schema.tql — basic syntax validation
# =========================================================================


class TestSchemaSyntax:
    """Basic regex checks for valid TypeQL schema syntax."""

    @pytest.fixture(autouse=True)
    def _load_schema(self) -> None:
        self.schema_text = SCHEMA_PATH.read_text()

    def test_starts_with_define(self) -> None:
        # Should begin with a comment or the 'define' keyword
        non_comment_lines = [
            line.strip()
            for line in self.schema_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert len(non_comment_lines) > 0
        assert non_comment_lines[0] == "define"

    def test_attribute_declarations(self) -> None:
        matches = re.findall(r"^attribute\s+\w+", self.schema_text, re.MULTILINE)
        assert len(matches) > 10, "Expected many attribute declarations"

    def test_entity_declarations(self) -> None:
        matches = re.findall(r"^entity\s+\w+", self.schema_text, re.MULTILINE)
        assert len(matches) >= 5

    def test_relation_declarations(self) -> None:
        matches = re.findall(r"^relation\s+[\w-]+", self.schema_text, re.MULTILINE)
        assert len(matches) >= 4

    def test_no_sub_entity_syntax(self) -> None:
        """TypeDB 3.x uses entity/relation directly, not 'sub entity'."""
        # Active schema should not use TypeDB 2.x 'sub entity' syntax
        active_lines = [
            line
            for line in self.schema_text.splitlines()
            if not line.strip().startswith("#")
        ]
        active_text = "\n".join(active_lines)
        assert "sub entity" not in active_text


# =========================================================================
# Expected entity types
# =========================================================================


class TestExpectedEntities:
    """Verify all required entity types are defined in the schema."""

    @pytest.fixture(autouse=True)
    def _load_schema(self) -> None:
        self.schema_text = SCHEMA_PATH.read_text()

    @pytest.mark.parametrize(
        "entity_name",
        [
            "sample",
            "cpg",
            "gene",
            "pathway",
            "function-term",
            "genomic-region",
            "cohort",
            "provenance",
        ],
    )
    def test_entity_defined(self, entity_name: str) -> None:
        pattern = rf"^entity\s+{re.escape(entity_name)}\b"
        assert re.search(
            pattern, self.schema_text, re.MULTILINE
        ), f"Entity '{entity_name}' not found in schema"


# =========================================================================
# Expected relation types
# =========================================================================


class TestExpectedRelations:
    """Verify all required relation types are defined in the schema."""

    @pytest.fixture(autouse=True)
    def _load_schema(self) -> None:
        self.schema_text = SCHEMA_PATH.read_text()

    @pytest.mark.parametrize(
        "relation_name",
        [
            "cpg-gene-overlap",
            "gene-pathway-membership",
            "gene-function-annotation",
            "cpg-region-overlap",
            "sample-cohort-membership",
            "gene-methylation-feature",
            "gene-differential-signal",
            "pathway-differential-signal",
            "term-differential-signal",
        ],
    )
    def test_relation_defined(self, relation_name: str) -> None:
        pattern = rf"^relation\s+{re.escape(relation_name)}\b"
        assert re.search(
            pattern, self.schema_text, re.MULTILINE
        ), f"Relation '{relation_name}' not found in schema"


# =========================================================================
# Expected key attributes on entities
# =========================================================================


class TestKeyAttributes:
    """Verify that key entities have @key attributes defined."""

    @pytest.fixture(autouse=True)
    def _load_schema(self) -> None:
        self.schema_text = SCHEMA_PATH.read_text()

    @pytest.mark.parametrize(
        ("entity", "key_attr"),
        [
            ("sample", "barcode"),
            ("cpg", "cpg_id"),
            ("gene", "ensembl_gene_id"),
            ("pathway", "pathway_id"),
        ],
    )
    def test_entity_has_key(self, entity: str, key_attr: str) -> None:
        # Look for pattern: owns <attr> @key
        pattern = rf"owns\s+{re.escape(key_attr)}\s+@key"
        assert re.search(
            pattern, self.schema_text
        ), f"Entity '{entity}' missing @key for '{key_attr}'"


# =========================================================================
# functions.tql
# =========================================================================


class TestFunctions:
    """Basic validation of functions.tql."""

    @pytest.fixture(autouse=True)
    def _load_functions(self) -> None:
        self.functions_text = FUNCTIONS_PATH.read_text()

    def test_starts_with_define(self) -> None:
        non_comment_lines = [
            line.strip()
            for line in self.functions_text.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        assert non_comment_lines[0] == "define"

    def test_fun_declarations_present(self) -> None:
        matches = re.findall(r"^fun\s+\w+", self.functions_text, re.MULTILINE)
        assert len(matches) >= 5, "Expected at least 5 function definitions"

    @pytest.mark.parametrize(
        "func_name",
        [
            "genes_in_pathway",
            "cpgs_for_gene",
            "cpgs_in_pathway",
            "pathways_for_gene",
            "go_terms_for_gene",
        ],
    )
    def test_expected_function_defined(self, func_name: str) -> None:
        assert (
            f"fun {func_name}" in self.functions_text
        ), f"Function '{func_name}' not found in functions.tql"
