"""Smoke tests for ``epigraph.db_build.map_cpg_to_islands``.

Covers ``IslandInterval``, ``ChromIslandIndex.classify`` (the four
context categories), ``_build_island_index``, and the end-to-end
``map_cpgs_to_island_context`` function.  Previously 0% coverage.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from epigraph.db_build.map_cpg_to_islands import (
    SHELF_BP,
    SHORE_BP,
    ChromIslandIndex,
    IslandInterval,
    _build_island_index,
    map_cpgs_to_island_context,
)


class TestIslandInterval:
    def test_shore_shelf_bounds(self) -> None:
        # start is > SHORE_BP + SHELF_BP so the max(0, ...) clamp doesn't fire
        iv = IslandInterval(island_id="CpGI_1", chrom="chr1", start=10_000, end=12_000)
        assert iv.shore_start == 10_000 - SHORE_BP
        assert iv.shore_end == 12_000 + SHORE_BP
        assert iv.shelf_start == 10_000 - SHORE_BP - SHELF_BP
        assert iv.shelf_end == 12_000 + SHORE_BP + SHELF_BP

    def test_shore_clamps_to_zero(self) -> None:
        iv = IslandInterval(island_id="CpGI_x", chrom="chr1", start=100, end=500)
        assert iv.shore_start == 0  # max(0, 100 - 2000)
        assert iv.shelf_start == 0


class TestChromIslandIndexClassify:
    def _make_index(self) -> ChromIslandIndex:
        idx = ChromIslandIndex(chrom="chr1")
        idx.islands = [
            IslandInterval(island_id="I1", chrom="chr1", start=10_000, end=11_000),
        ]
        idx.build()
        return idx

    def test_inside_island(self) -> None:
        ctx, iid = self._make_index().classify(10_500)
        assert ctx == "island"
        assert iid == "I1"

    def test_shore_region(self) -> None:
        # 500 bp upstream of island.start
        ctx, iid = self._make_index().classify(9_500)
        assert ctx == "shore"
        assert iid == "I1"

    def test_shelf_region(self) -> None:
        # 3000 bp upstream of island.start (in shelf, not shore)
        ctx, iid = self._make_index().classify(7_000)
        assert ctx == "shelf"
        assert iid == "I1"

    def test_open_sea(self) -> None:
        ctx, iid = self._make_index().classify(100_000)
        assert ctx == "open_sea"
        assert iid is None

    def test_island_priority_over_shore(self) -> None:
        # Two islands where pos is inside one and in the shore of the other
        idx = ChromIslandIndex(chrom="chr1")
        idx.islands = [
            IslandInterval(island_id="A", chrom="chr1", start=5_000, end=6_000),
            IslandInterval(island_id="B", chrom="chr1", start=7_500, end=8_500),
        ]
        idx.build()
        # pos 7_800 is INSIDE island B and in SHORE of island A
        ctx, iid = idx.classify(7_800)
        assert ctx == "island"
        assert iid == "B"


class TestBuildIslandIndex:
    def test_partitions_by_chromosome(self) -> None:
        df = pl.DataFrame(
            {
                "region_id": ["I1", "I2", "I3"],
                "chrom": ["chr1", "chr2", "chr1"],
                "start": [1000, 2000, 500],
                "end": [1500, 2500, 900],
            }
        )
        indices = _build_island_index(df)
        assert set(indices.keys()) == {"chr1", "chr2"}
        assert len(indices["chr1"].islands) == 2
        # chr1 islands should be sorted by start ascending
        assert indices["chr1"].islands[0].start == 500
        assert indices["chr1"].islands[1].start == 1000


class TestMapCpgsToIslandContextEndToEnd:
    def test_classifies_all_context_categories(self, tmp_path: Path) -> None:
        # Islands
        islands_pq = tmp_path / "islands.parquet"
        pl.DataFrame(
            {
                "region_id": ["I_chr1_10k"],
                "chrom": ["chr1"],
                "start": [10_000],  # 0-based half-open [10000, 11000)
                "end": [11_000],
            }
        ).write_parquet(islands_pq)

        # Four CpGs: one in each context (1-based positions; function
        # subtracts 1 internally)
        cpg_pq = tmp_path / "cpg_mapping.parquet"
        pl.DataFrame(
            {
                "cpg_id": ["cpg_island", "cpg_shore", "cpg_shelf", "cpg_open"],
                "chromosome": ["chr1", "chr1", "chr1", "chr1"],
                "position": [10_501, 9_501, 7_001, 100_001],  # 1-based
            }
        ).write_parquet(cpg_pq)

        out_pq = tmp_path / "out.parquet"
        result = map_cpgs_to_island_context(cpg_pq, islands_pq, out_pq)

        assert out_pq.exists()
        # Build cpg_id → context lookup from returned DataFrame
        mapping = dict(zip(result["cpg_id"].to_list(), result["context"].to_list()))
        assert mapping["cpg_island"] == "island"
        assert mapping["cpg_shore"] == "shore"
        assert mapping["cpg_shelf"] == "shelf"
        assert mapping["cpg_open"] == "open_sea"

    def test_unknown_chromosome_is_open_sea(self, tmp_path: Path) -> None:
        islands_pq = tmp_path / "islands.parquet"
        pl.DataFrame(
            {"region_id": ["I1"], "chrom": ["chr1"], "start": [1000], "end": [2000]}
        ).write_parquet(islands_pq)

        cpg_pq = tmp_path / "cpg_mapping.parquet"
        pl.DataFrame(
            {
                "cpg_id": ["cpg_Y_1"],
                "chromosome": ["chrY"],
                "position": [1500],
            }
        ).write_parquet(cpg_pq)

        out_pq = tmp_path / "out.parquet"
        result = map_cpgs_to_island_context(cpg_pq, islands_pq, out_pq)
        assert result["context"].to_list() == ["open_sea"]
        assert result["island_id"].to_list() == [None]
