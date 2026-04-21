"""I/O utilities for Parquet, Excel, and S3 data access.

All heavy tabular I/O uses Polars backed by PyArrow.  Excel reading falls
back to openpyxl through Pandas (openpyxl does not have a Polars adapter).
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import polars as pl

from epigraph.common.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Beta matrix header helper
# ---------------------------------------------------------------------------


def read_beta_header(csv_path: str | Path) -> list[str]:
    """Read beta matrix CSV header, renaming empty first cell to ``'sample_id'``.

    The beta matrix CSV has an empty first header cell (the sample-index
    column).  This function reads only the first line and returns the full
    list of column names with that cell replaced by ``"sample_id"``.

    Args:
        csv_path: Path to the beta matrix CSV file.

    Returns:
        List of column names starting with ``"sample_id"``.
    """
    with open(csv_path) as fh:
        header_line = fh.readline().rstrip("\n")

    cols = header_line.split(",")
    if cols[0] == "":
        cols[0] = "sample_id"
    return cols


# ---------------------------------------------------------------------------
# Parquet helpers
# ---------------------------------------------------------------------------


def read_parquet_lazy(path: str | Path) -> pl.LazyFrame:
    """Return a Polars LazyFrame backed by a Parquet file.

    The file is memory-mapped, so only columns/rows that are actually
    collected will be loaded.

    Args:
        path: Filesystem path to a ``.parquet`` file.

    Returns:
        A :class:`polars.LazyFrame` ready for query composition.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    logger.info("opening_parquet_lazy", path=str(path))
    return pl.scan_parquet(path)


def write_parquet(
    df: pl.DataFrame | pl.LazyFrame,
    path: str | Path,
    *,
    compression: str = "zstd",
    row_group_size: int | None = None,
) -> Path:
    """Write a Polars DataFrame (or collected LazyFrame) to Parquet.

    Args:
        df: Data to persist.  A LazyFrame will be collected first.
        path: Destination file path.
        compression: Parquet compression codec (default ``"zstd"``).
        row_group_size: Optional number of rows per row group.

    Returns:
        Resolved :class:`Path` to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    kwargs: dict[str, Any] = {"compression": compression}
    if row_group_size is not None:
        kwargs["row_group_size"] = row_group_size

    df.write_parquet(path, **kwargs)
    logger.info(
        "wrote_parquet",
        path=str(path),
        rows=df.height,
        cols=df.width,
        compression=compression,
    )
    return path


# ---------------------------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------------------------


def read_xlsx_sheets(
    path: str | Path,
    pattern: str = "CLIN_*",
) -> dict[str, pl.DataFrame]:
    """Read worksheets whose names match *pattern* from an Excel workbook.

    Uses openpyxl via Pandas for the actual reading, then converts each sheet
    to a Polars DataFrame.

    Args:
        path: Path to an ``.xlsx`` file.
        pattern: ``fnmatch`` glob pattern applied to sheet names.

    Returns:
        Dict mapping sheet name to Polars DataFrame.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    import pandas as pd  # local import to keep module-level import light

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    xls = pd.ExcelFile(path, engine="openpyxl")
    matching = [s for s in xls.sheet_names if fnmatch.fnmatch(s, pattern)]
    logger.info(
        "reading_xlsx_sheets",
        path=str(path),
        pattern=pattern,
        matched=matching,
    )

    result: dict[str, pl.DataFrame] = {}
    for sheet in matching:
        pdf = xls.parse(sheet)
        result[sheet] = pl.from_pandas(pdf)

    return result


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

_S3_URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse ``s3://bucket/key`` into (bucket, key)."""
    m = _S3_URI_RE.match(uri)
    if m is None:
        raise ValueError(f"Invalid S3 URI: {uri!r}")
    return m.group(1), m.group(2)


def download_s3(s3_uri: str, local_path: str | Path) -> Path:
    """Download an object from S3 to a local file.

    Args:
        s3_uri: Full S3 URI, e.g. ``s3://my-bucket/data/file.parquet``.
        local_path: Destination on the local filesystem.

    Returns:
        Resolved :class:`Path` to the downloaded file.
    """
    import boto3  # local import to avoid hard dependency at import time

    bucket, key = _parse_s3_uri(s3_uri)
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("downloading_s3", bucket=bucket, key=key, dest=str(local_path))
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, str(local_path))
    logger.info("download_complete", path=str(local_path))
    return local_path


def resolve_path(
    path: str | Path,
    cache_dir: str | Path | None = None,
) -> Path:
    """Resolve a path that may be local or an S3 URI.

    If *path* is an S3 URI the object is downloaded to *cache_dir* (defaulting
    to ``~/.cache/epigraph/``) and the local cache path is returned.  A
    cached copy is reused if it already exists.

    Args:
        path: Local path or ``s3://`` URI.
        cache_dir: Directory used to cache S3 downloads.

    Returns:
        A :class:`Path` on the local filesystem.
    """
    path_str = str(path)

    if not path_str.startswith("s3://"):
        resolved = Path(path_str).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Local path does not exist: {resolved}")
        return resolved

    # S3 path -- download to cache if needed.
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "epigraph"
    cache_dir = Path(cache_dir).resolve()

    parsed = urlparse(path_str)
    # Build a cache path that mirrors the S3 key structure.
    cache_path = (cache_dir / parsed.netloc / parsed.path.lstrip("/")).resolve()

    # Containment check: reject keys like ``../../etc/passwd`` that would
    # write outside the cache directory once resolved.
    if cache_dir != cache_path and cache_dir not in cache_path.parents:
        raise ValueError(
            f"S3 URI {path_str!r} resolves to a cache path outside "
            f"{cache_dir!r}: {cache_path!r}"
        )

    if cache_path.exists():
        logger.debug("cache_hit", path=str(cache_path))
        return cache_path

    return download_s3(path_str, cache_path)
