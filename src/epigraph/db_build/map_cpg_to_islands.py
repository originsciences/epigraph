"""Map CpG sites to CpG island genomic context.

Assigns each CpG a context label relative to CpG islands:

- **island**: CpG falls within a CpG island
- **shore**: CpG is within 2 kb flanking an island
- **shelf**: CpG is within 2 kb flanking a shore (i.e. 2-4 kb from island)
- **open_sea**: CpG does not fall in any of the above

Uses a sorted-interval + binary-search approach for efficiency with
~4 M CpGs and ~32 K islands.

Coordinate conventions:
- CpG island coordinates (from UCSC) are 0-based half-open [start, end).
- CpG positions (from the Illumina manifest / mapping) are 1-based.
- We convert CpG positions to 0-based before comparison.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from pathlib import Path

import click
import polars as pl

from epigraph.common.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHORE_BP: int = 2000
"""Base pairs flanking an island that define a shore region."""

SHELF_BP: int = 2000
"""Base pairs flanking a shore that define a shelf region."""


# ---------------------------------------------------------------------------
# Data structures for interval lookup
# ---------------------------------------------------------------------------


@dataclass
class IslandInterval:
    """A CpG island's genomic interval (0-based half-open)."""

    island_id: str
    chrom: str
    start: int  # 0-based inclusive
    end: int  # 0-based exclusive

    @property
    def shore_start(self) -> int:
        """Start of the upstream shore region."""
        return max(0, self.start - SHORE_BP)

    @property
    def shore_end(self) -> int:
        """End of the downstream shore region."""
        return self.end + SHORE_BP

    @property
    def shelf_start(self) -> int:
        """Start of the upstream shelf region."""
        return max(0, self.start - SHORE_BP - SHELF_BP)

    @property
    def shelf_end(self) -> int:
        """End of the downstream shelf region."""
        return self.end + SHORE_BP + SHELF_BP


@dataclass
class ChromIslandIndex:
    """Sorted island intervals for a single chromosome with binary search."""

    chrom: str
    islands: list[IslandInterval] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)

    def build(self) -> None:
        """Sort islands by start position and build search array."""
        self.islands.sort(key=lambda i: i.start)
        self.starts = [i.start for i in self.islands]

    def classify(self, pos_0based: int) -> tuple[str, str | None]:
        """Classify a 0-based position relative to CpG islands.

        Args:
            pos_0based: 0-based genomic position.

        Returns:
            Tuple of (context, island_id). island_id is None for open_sea.
        """
        # Find candidates: islands whose shelf region could contain pos.
        # The maximum shelf reach is SHORE_BP + SHELF_BP = 4000 bp beyond
        # the island end. We need islands where shelf_start <= pos < shelf_end.
        #
        # shelf_start = island.start - 4000
        # shelf_end   = island.end + 4000
        #
        # So we need islands where island.start - 4000 <= pos, i.e.
        # island.start <= pos + 4000.
        max_reach = SHORE_BP + SHELF_BP
        right_idx = bisect.bisect_right(self.starts, pos_0based + max_reach)

        best_context: str = "open_sea"
        best_island_id: str | None = None
        # Priority: island > shore > shelf > open_sea
        context_priority = {"island": 3, "shore": 2, "shelf": 1, "open_sea": 0}

        for i in range(right_idx - 1, -1, -1):
            island = self.islands[i]

            # If the island's shelf_end is before our position, and the island
            # start is far enough left that no subsequent island can reach, stop.
            if island.shelf_end <= pos_0based:
                # Islands are sorted by start. If this island's shelf_end
                # is before pos, islands further left will also be too far.
                # But an island further left could have a larger end. In
                # practice, islands don't overlap and are reasonably sized,
                # so we use a generous cutoff.
                if island.start < pos_0based - max_reach - 500_000:
                    break
                continue

            if island.shelf_start > pos_0based:
                # This island is entirely to the right; skip.
                continue

            # Determine context
            if island.start <= pos_0based < island.end:
                ctx = "island"
            elif island.shore_start <= pos_0based < island.shore_end:
                ctx = "shore"
            elif island.shelf_start <= pos_0based < island.shelf_end:
                ctx = "shelf"
            else:
                continue

            if context_priority[ctx] > context_priority[best_context]:
                best_context = ctx
                best_island_id = island.island_id

            # island is highest priority — no need to keep looking
            if best_context == "island":
                break

        return best_context, best_island_id


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _build_island_index(
    islands_df: pl.DataFrame,
) -> dict[str, ChromIslandIndex]:
    """Build per-chromosome island indices from a Polars DataFrame.

    Args:
        islands_df: DataFrame with chrom, start, end columns (0-based).

    Returns:
        Dict mapping chromosome name to ChromIslandIndex.
    """
    indices: dict[str, ChromIslandIndex] = {}
    for row in islands_df.iter_rows(named=True):
        chrom = row["chrom"]
        island_id = f"{chrom}:{row['start']}-{row['end']}"
        interval = IslandInterval(
            island_id=island_id,
            chrom=chrom,
            start=row["start"],
            end=row["end"],
        )
        if chrom not in indices:
            indices[chrom] = ChromIslandIndex(chrom=chrom)
        indices[chrom].islands.append(interval)

    for idx in indices.values():
        idx.build()

    log.info(
        "island_index_built",
        n_chromosomes=len(indices),
        n_islands=sum(len(idx.islands) for idx in indices.values()),
    )
    return indices


def map_cpgs_to_island_context(
    cpg_mapping_path: str | Path,
    islands_path: str | Path,
    output_path: str | Path,
) -> pl.DataFrame:
    """Map CpG sites to CpG island context and write result to Parquet.

    Args:
        cpg_mapping_path: Path to the CpG-gene mapping Parquet with
            cpg_id, chromosome, position (1-based) columns.
        islands_path: Path to the CpG islands Parquet with
            chrom, start, end (0-based half-open) columns.
        output_path: Path to write the output Parquet.

    Returns:
        The resulting DataFrame with cpg_id, chromosome, position, context,
        island_id columns.
    """
    cpg_mapping_path = Path(cpg_mapping_path)
    islands_path = Path(islands_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load islands ---
    log.info("loading_cpg_islands", path=str(islands_path))
    islands_df = pl.read_parquet(islands_path)
    log.info("cpg_islands_loaded", n=len(islands_df))

    # --- Build index ---
    island_index = _build_island_index(islands_df)

    # --- Load CpGs (deduplicated by cpg_id) ---
    log.info("loading_cpg_positions", path=str(cpg_mapping_path))
    cpg_df = pl.read_parquet(
        cpg_mapping_path,
        columns=["cpg_id", "chromosome", "position"],
    )
    cpg_unique = cpg_df.unique(subset=["cpg_id"]).sort(["chromosome", "position"])
    log.info(
        "cpg_positions_loaded",
        n_records=len(cpg_df),
        n_unique_sites=len(cpg_unique),
    )

    # --- Classify each CpG ---
    log.info("classifying_cpgs")
    contexts: list[str] = []
    island_ids: list[str | None] = []

    total = len(cpg_unique)
    report_interval = max(total // 20, 1)

    for idx, row in enumerate(cpg_unique.iter_rows(named=True)):
        chrom = row["chromosome"]
        pos_1based = row["position"]
        # Convert 1-based CpG position to 0-based for comparison
        pos_0based = pos_1based - 1

        chrom_idx = island_index.get(chrom)
        if chrom_idx is None:
            contexts.append("open_sea")
            island_ids.append(None)
        else:
            ctx, iid = chrom_idx.classify(pos_0based)
            contexts.append(ctx)
            island_ids.append(iid)

        if (idx + 1) % report_interval == 0:
            log.info(
                "classify_progress",
                done=idx + 1,
                total=total,
                pct=round(100 * (idx + 1) / total, 1),
            )

    log.info("classification_complete")

    # --- Build result DataFrame ---
    result = pl.DataFrame(
        {
            "cpg_id": cpg_unique["cpg_id"],
            "chromosome": cpg_unique["chromosome"],
            "position": cpg_unique["position"],
            "context": contexts,
            "island_id": island_ids,
        }
    )

    # --- Log distribution ---
    dist = result.group_by("context").len().sort("len", descending=True)
    distribution = {
        row["context"]: {"count": row["len"], "pct": round(100 * row["len"] / len(result), 1)}
        for row in dist.iter_rows(named=True)
    }
    log.info("context_distribution", distribution=distribution)

    # --- Write output ---
    result.write_parquet(output_path)
    log.info("wrote_output", n_rows=len(result), path=str(output_path))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("map-cpg-islands")
@click.option(
    "--cpg-mapping",
    "cpg_mapping_path",
    type=click.Path(exists=True),
    default="data/processed/cpg_gene_mapping_full.parquet",
    help="Path to CpG-gene mapping Parquet.",
)
@click.option(
    "--islands",
    "islands_path",
    type=click.Path(exists=True),
    default="data/external/cpg_islands.parquet",
    help="Path to CpG islands Parquet.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default="data/processed/cpg_island_context.parquet",
    help="Output Parquet path.",
)
def main(cpg_mapping_path: str, islands_path: str, output_path: str) -> None:
    """Map CpG sites to CpG island context (island/shore/shelf/open_sea)."""
    map_cpgs_to_island_context(cpg_mapping_path, islands_path, output_path)


if __name__ == "__main__":
    main()
