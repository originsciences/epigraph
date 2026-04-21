"""Configurable parallelism for CPU-bound tasks."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TypeVar

from epigraph.common.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def get_n_workers(n_workers: int | None = None) -> int:
    """Return number of workers, defaulting to CPU count minus one.

    Args:
        n_workers: Explicit worker count.  If ``None`` or ``<= 0``,
            defaults to ``max(1, os.cpu_count() - 1)``.

    Returns:
        Positive integer number of workers.
    """
    if n_workers is not None and n_workers > 0:
        return n_workers
    return max(1, (os.cpu_count() or 1) - 1)


def parallel_map(
    fn: Callable,
    items: Iterable,
    n_workers: int | None = None,
    use_threads: bool = False,
    desc: str = "",
) -> list[R]:
    """Map *fn* over *items* using a process or thread pool.

    Falls back to sequential execution if *n_workers* resolves to 1.

    Args:
        fn: Callable applied to each item.
        items: Iterable of inputs.
        n_workers: Number of pool workers (see :func:`get_n_workers`).
        use_threads: If ``True`` use :class:`ThreadPoolExecutor`;
            otherwise use :class:`ProcessPoolExecutor`.
        desc: Human-readable label for log messages.

    Returns:
        List of results in the same order as *items*.
    """
    items_list = list(items)
    workers = get_n_workers(n_workers)

    if workers == 1 or len(items_list) <= 1:
        log.info("parallel_map_sequential", desc=desc, n_items=len(items_list))
        return [fn(item) for item in items_list]

    pool_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    log.info(
        "parallel_map_start",
        desc=desc,
        n_items=len(items_list),
        n_workers=workers,
        executor=pool_cls.__name__,
    )

    with pool_cls(max_workers=workers) as executor:
        results = list(executor.map(fn, items_list))

    log.info("parallel_map_complete", desc=desc, n_results=len(results))
    return results
