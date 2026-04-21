"""Tests for the parallel utility in epigraph.common.parallel."""

from __future__ import annotations

import os

from epigraph.common.parallel import get_n_workers, parallel_map

# ---------------------------------------------------------------------------
# get_n_workers
# ---------------------------------------------------------------------------


class TestGetNWorkers:
    """Tests for the worker count helper."""

    def test_explicit_value_returned(self) -> None:
        assert get_n_workers(4) == 4

    def test_default_is_positive(self) -> None:
        result = get_n_workers(None)
        assert result >= 1

    def test_default_at_most_cpu_count(self) -> None:
        cpu = os.cpu_count() or 1
        result = get_n_workers(None)
        assert result <= cpu

    def test_zero_falls_back_to_default(self) -> None:
        result = get_n_workers(0)
        assert result >= 1

    def test_negative_falls_back_to_default(self) -> None:
        result = get_n_workers(-2)
        assert result >= 1


# ---------------------------------------------------------------------------
# parallel_map with threads
# ---------------------------------------------------------------------------


def _double(x: int) -> int:
    return x * 2


class TestParallelMapThreads:
    """Tests for parallel_map using ThreadPoolExecutor."""

    def test_basic_thread_map(self) -> None:
        items = [1, 2, 3, 4, 5]
        result = parallel_map(_double, items, n_workers=2, use_threads=True)
        assert result == [2, 4, 6, 8, 10]

    def test_preserves_order(self) -> None:
        items = list(range(20))
        result = parallel_map(_double, items, n_workers=4, use_threads=True)
        assert result == [x * 2 for x in items]

    def test_empty_input(self) -> None:
        result = parallel_map(_double, [], n_workers=2, use_threads=True)
        assert result == []

    def test_single_item_runs_sequential(self) -> None:
        """A single item should trigger sequential fallback."""
        result = parallel_map(_double, [42], n_workers=4, use_threads=True)
        assert result == [84]


# ---------------------------------------------------------------------------
# parallel_map sequential fallback
# ---------------------------------------------------------------------------


class TestParallelMapSequential:
    """Tests for sequential fallback when n_workers=1."""

    def test_sequential_with_one_worker(self) -> None:
        items = [10, 20, 30]
        result = parallel_map(_double, items, n_workers=1, use_threads=False)
        assert result == [20, 40, 60]

    def test_sequential_preserves_order(self) -> None:
        items = list(range(10))
        result = parallel_map(_double, items, n_workers=1)
        assert result == [x * 2 for x in items]

    def test_sequential_with_empty(self) -> None:
        result = parallel_map(_double, [], n_workers=1)
        assert result == []
