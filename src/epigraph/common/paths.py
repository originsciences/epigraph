"""Project path resolution and settings management.

Paths are loaded from ``config/settings.yaml`` and resolved relative to the
project root.  The ``EPIGRAPH_ROOT`` environment variable can override the
auto-detected root.  Individual paths can also be overridden via environment
variables (see ``config/settings.yaml`` for the ``${VAR:-default}`` syntax).

Usage::

    from epigraph.common.paths import ProjectPaths

    paths = ProjectPaths.from_settings()
    df = pl.scan_parquet(paths.processed_dir / "beta.parquet")
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, model_validator

from epigraph.common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Env-var interpolation for YAML values
# ---------------------------------------------------------------------------

_ENV_PATTERN = re.compile(r"\$\{(\w+)(?::-(.*?))?\}")


def _resolve_env_vars(value: str) -> str:
    """Replace ``${VAR:-default}`` placeholders with environment values."""

    def _replace(m: re.Match[str]) -> str:
        var_name = m.group(1)
        default = m.group(2) if m.group(2) is not None else ""
        return os.environ.get(var_name, default)

    return _ENV_PATTERN.sub(_replace, value)


def _resolve_yaml_values(data: Any) -> Any:
    """Recursively resolve ``${VAR:-default}`` in all string values."""
    if isinstance(data, str):
        return _resolve_env_vars(data)
    if isinstance(data, dict):
        return {k: _resolve_yaml_values(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_yaml_values(v) for v in data]
    return data


# ---------------------------------------------------------------------------
# Root detection
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Walk upward from this file to find the project root.

    The root is identified by the presence of ``pyproject.toml``.
    """
    candidate = Path(__file__).resolve().parent
    for _ in range(10):
        if (candidate / "pyproject.toml").exists():
            return candidate
        candidate = candidate.parent
    raise RuntimeError(
        "Could not find project root (no pyproject.toml found). "
        "Set EPIGRAPH_ROOT explicitly."
    )


def get_project_root() -> Path:
    """Return the project root directory.

    Prefers ``EPIGRAPH_ROOT`` environment variable, falling back to
    auto-detection from the file tree.
    """
    env_root = os.getenv("EPIGRAPH_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return _find_project_root()


# ---------------------------------------------------------------------------
# ProjectPaths model
# ---------------------------------------------------------------------------


class ProjectPaths(BaseModel):
    """Resolved filesystem paths for the epigraph project.

    Attributes:
        root: Project root directory.
        beta_matrix: Path to the raw beta-matrix CSV.
        clinical_metadata: Path to the clinical metadata Excel file.
        data_dir: Top-level data directory.
        dev_subset_dir: Directory for the development-sized data subset.
        processed_dir: Directory for processed / intermediate outputs.
        external_dir: Directory for external reference data.
        config_dir: Directory containing YAML config files.
        schemas_dir: Directory containing TypeDB schema files.
    """

    root: Path
    beta_matrix: Path
    clinical_metadata: Path
    data_dir: Path
    dev_subset_dir: Path
    processed_dir: Path
    external_dir: Path
    config_dir: Path
    schemas_dir: Path

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _resolve_relative_paths(self) -> Self:
        """Make all relative paths absolute by prepending *root*."""
        for field_name in self.model_fields:
            if field_name == "root":
                continue
            value = getattr(self, field_name)
            if isinstance(value, Path) and not value.is_absolute():
                object.__setattr__(self, field_name, self.root / value)
        return self

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(
        cls,
        settings_path: str | Path | None = None,
        *,
        mode: str | None = None,
    ) -> ProjectPaths:
        """Load project paths from ``config/settings.yaml``.

        Args:
            settings_path: Explicit path to the settings file.  Defaults to
                ``<root>/config/settings.yaml``.
            mode: ``"dev"`` or ``"prod"``.  Currently unused but reserved for
                future environment-specific overrides.

        Returns:
            A fully resolved :class:`ProjectPaths` instance.
        """
        root = get_project_root()

        if settings_path is None:
            settings_path = root / "config" / "settings.yaml"
        else:
            settings_path = Path(settings_path)

        if not settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {settings_path}")

        raw = yaml.safe_load(settings_path.read_text())
        resolved = _resolve_yaml_values(raw)

        path_cfg: dict[str, Any] = resolved.get("paths", {})

        paths = cls(
            root=root,
            beta_matrix=Path(path_cfg.get("beta_matrix", "data/raw/beta_matrix.csv")),
            clinical_metadata=Path(
                path_cfg.get("clinical_metadata", "data/raw/clinical_metadata.xlsx")
            ),
            data_dir=Path(path_cfg.get("data_dir", "data")),
            dev_subset_dir=Path(path_cfg.get("dev_subset_dir", "data/dev")),
            processed_dir=Path(path_cfg.get("processed_dir", "data/processed")),
            external_dir=Path(path_cfg.get("external_dir", "data/external")),
            config_dir=root / "config",
            schemas_dir=root / "schemas",
        )

        logger.info(
            "project_paths_loaded",
            root=str(paths.root),
            beta_matrix=str(paths.beta_matrix),
            mode=mode or os.getenv("EPIGRAPH_ENV", "dev"),
        )
        return paths

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create all output directories that don't yet exist."""
        for dir_path in (
            self.data_dir,
            self.dev_subset_dir,
            self.processed_dir,
            self.external_dir,
        ):
            dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug("ensured_output_dirs")
