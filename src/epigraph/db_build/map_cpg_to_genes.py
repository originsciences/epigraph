"""Map CpG positions to overlapping genes using coordinate intersection.

Assigns each CpG an overlap type relative to genes it intersects:

- **promoter**: TSS +/- 1500 bp upstream, 500 bp downstream
- **gene_body**: within the gene interval but not in promoter
- **exon / intron**: (future refinement with transcript models)
- **intergenic**: not overlapping any gene or promoter

For 4 M CpGs this must be efficient.  The approach:

1. Sort genes by chromosome and start position.
2. For each chromosome, build a sorted array of gene intervals.
3. For each CpG on that chromosome, use binary search to find
   candidate overlapping genes.

An alternative pybedtools-based approach is available for validation.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import polars as pl

from epigraph.common.genome_coords import parse_cpg_id
from epigraph.common.logging import get_logger
from epigraph.common.parallel import get_n_workers, parallel_map

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROMOTER_UPSTREAM: int = 1500
"""Base pairs upstream of TSS to define promoter region."""

PROMOTER_DOWNSTREAM: int = 500
"""Base pairs downstream of TSS to define promoter region."""


# ---------------------------------------------------------------------------
# Data structures for interval lookup
# ---------------------------------------------------------------------------


@dataclass
class GeneInterval:
    """A gene's genomic interval with promoter boundaries."""

    gene_id: str
    gene_symbol: str
    chrom: str
    start: int  # gene start (1-based)
    end: int    # gene end (1-based)
    strand: str
    tss: int = 0
    promoter_start: int = 0
    promoter_end: int = 0

    def __post_init__(self) -> None:
        """Compute TSS and promoter boundaries from strand."""
        if self.strand == "-":
            self.tss = self.end
            self.promoter_start = max(1, self.tss - PROMOTER_DOWNSTREAM)
            self.promoter_end = self.tss + PROMOTER_UPSTREAM
        else:
            # + strand or unknown
            self.tss = self.start
            self.promoter_start = max(1, self.tss - PROMOTER_UPSTREAM)
            self.promoter_end = self.tss + PROMOTER_DOWNSTREAM


@dataclass
class ChromosomeIndex:
    """Sorted gene intervals for a single chromosome, enabling binary search."""

    chrom: str
    genes: list[GeneInterval] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)
    ends: list[int] = field(default_factory=list)

    def build(self) -> None:
        """Sort genes by start position and build search arrays."""
        self.genes.sort(key=lambda g: g.start)
        self.starts = [g.start for g in self.genes]
        self.ends = [g.end for g in self.genes]

    def find_overlapping(self, pos: int) -> list[tuple[GeneInterval, str]]:
        """Find genes whose interval or promoter region contains *pos*.

        Args:
            pos: 1-based genomic position.

        Returns:
            List of (GeneInterval, overlap_type) tuples.
        """
        results: list[tuple[GeneInterval, str]] = []

        # Binary search: find genes that could overlap this position.
        # A gene can overlap if gene.start <= pos (start is before or at pos)
        # and gene.end >= pos (end is at or after pos).
        # But we also need to check promoter regions which extend beyond the gene.

        # Find rightmost gene whose start <= pos + PROMOTER_UPSTREAM
        # (the furthest a promoter could extend upstream)
        max_reach = pos + PROMOTER_UPSTREAM
        right = bisect.bisect_right(self.starts, max_reach)

        # Scan backwards from there
        for i in range(right - 1, -1, -1):
            gene = self.genes[i]

            # If this gene's promoter_end is before pos and gene end is before pos,
            # no genes further left can overlap either
            max_gene_reach = max(gene.end, gene.promoter_end)
            if max_gene_reach < pos:
                # Check if we can stop early: genes are sorted by start,
                # so all remaining genes have start <= this one.
                # But their ends could be larger, so we can't stop.
                # For efficiency with very large datasets, we limit scan depth.
                if gene.start < pos - PROMOTER_UPSTREAM - 1_000_000:
                    break
                continue

            # Check promoter overlap
            if gene.promoter_start <= pos <= gene.promoter_end:
                results.append((gene, "promoter"))
            # Check gene body overlap (not promoter)
            elif gene.start <= pos <= gene.end:
                results.append((gene, "gene_body"))

        return results


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def build_gene_index(genes_df: pl.DataFrame) -> dict[str, ChromosomeIndex]:
    """Build a chromosome-indexed lookup structure from the genes DataFrame.

    Args:
        genes_df: DataFrame with columns ``['gene_id', 'gene_symbol', 'chrom',
            'start', 'end', 'strand']``.

    Returns:
        Dictionary mapping chromosome name to a ``ChromosomeIndex``.
    """
    index: dict[str, ChromosomeIndex] = {}

    for row in genes_df.iter_rows(named=True):
        chrom = row["chrom"]
        if chrom not in index:
            index[chrom] = ChromosomeIndex(chrom=chrom)

        gene = GeneInterval(
            gene_id=row["gene_id"],
            gene_symbol=row.get("gene_symbol", ""),
            chrom=chrom,
            start=row["start"],
            end=row["end"],
            strand=row.get("strand", "+"),
        )
        index[chrom].genes.append(gene)

    for chrom_idx in index.values():
        chrom_idx.build()

    log.info(
        "gene_index_built",
        n_chromosomes=len(index),
        n_genes=sum(len(ci.genes) for ci in index.values()),
    )
    return index


# ---------------------------------------------------------------------------
# CpG-to-gene mapping
# ---------------------------------------------------------------------------


def _map_chromosome_batch(
    args: tuple[str, list[str], ChromosomeIndex | None, bool],
) -> dict[str, list[Any]]:
    """Map a batch of CpG IDs from a single chromosome to genes.

    This is the per-chromosome worker function used by :func:`map_cpgs_to_genes`
    for parallel execution.  Returns columnar lists instead of list-of-dicts
    to reduce per-record memory overhead.

    Args:
        args: Tuple of ``(chrom, cpg_ids, chrom_index, report_intergenic)``.

    Returns:
        Dict of column name to list of values (columnar format).
    """
    chrom, cpg_ids, chrom_idx, report_intergenic = args

    col_cpg_id: list[str] = []
    col_chromosome: list[str] = []
    col_position: list[int] = []
    col_gene_id: list[str] = []
    col_gene_symbol: list[str] = []
    col_overlap_type: list[str] = []

    for cpg_id in cpg_ids:
        try:
            _, pos = parse_cpg_id(cpg_id)
        except ValueError:
            continue

        if chrom_idx is None:
            if report_intergenic:
                col_cpg_id.append(cpg_id)
                col_chromosome.append(chrom)
                col_position.append(pos)
                col_gene_id.append("")
                col_gene_symbol.append("")
                col_overlap_type.append("intergenic")
            continue

        overlaps = chrom_idx.find_overlapping(pos)

        if overlaps:
            for gene, overlap_type in overlaps:
                col_cpg_id.append(cpg_id)
                col_chromosome.append(chrom)
                col_position.append(pos)
                col_gene_id.append(gene.gene_id)
                col_gene_symbol.append(gene.gene_symbol)
                col_overlap_type.append(overlap_type)
        elif report_intergenic:
            col_cpg_id.append(cpg_id)
            col_chromosome.append(chrom)
            col_position.append(pos)
            col_gene_id.append("")
            col_gene_symbol.append("")
            col_overlap_type.append("intergenic")

    return {
        "cpg_id": col_cpg_id,
        "chromosome": col_chromosome,
        "position": col_position,
        "gene_id": col_gene_id,
        "gene_symbol": col_gene_symbol,
        "overlap_type": col_overlap_type,
    }


def map_cpgs_to_genes(
    cpg_ids: list[str],
    gene_index: dict[str, ChromosomeIndex],
    report_intergenic: bool = False,
    n_workers: int | None = None,
) -> pl.DataFrame:
    """Map a list of CpG IDs to overlapping genes.

    When *n_workers* > 1, CpG IDs are grouped by chromosome and each
    chromosome batch is processed in parallel using a thread pool.
    The gene index is read-only after building, so sharing across threads
    is safe.

    Args:
        cpg_ids: List of CpG identifiers (``chr{N}_{position}`` format).
        gene_index: Pre-built chromosome index from ``build_gene_index``.
        report_intergenic: If ``True``, include CpGs with no gene overlap
            in the output with ``overlap_type='intergenic'``.
        n_workers: Number of parallel workers.  Defaults to
            ``cpu_count - 1``.  Set to ``1`` to disable parallelism.

    Returns:
        DataFrame with columns:
        ``['cpg_id', 'chromosome', 'position', 'gene_id', 'gene_symbol',
        'overlap_type']``.
    """
    # Group CpG IDs by chromosome
    by_chrom: dict[str, list[str]] = defaultdict(list)
    for cpg_id in cpg_ids:
        try:
            chrom, _ = parse_cpg_id(cpg_id)
        except ValueError:
            continue
        by_chrom[chrom].append(cpg_id)

    # Build work items: (chrom, cpg_list, chrom_index_or_none, report_intergenic)
    work_items: list[tuple[str, list[str], ChromosomeIndex | None, bool]] = [
        (chrom, ids, gene_index.get(chrom), report_intergenic)
        for chrom, ids in sorted(by_chrom.items())
    ]

    log.info(
        "mapping_start",
        total_cpgs=len(cpg_ids),
        n_chromosomes=len(work_items),
        n_workers=get_n_workers(n_workers),
    )

    batch_results = parallel_map(
        _map_chromosome_batch,
        work_items,
        n_workers=n_workers,
        use_threads=True,
        desc="cpg_to_gene_mapping",
    )

    # Merge columnar results from all chromosome batches
    merged: dict[str, list[Any]] = {
        "cpg_id": [],
        "chromosome": [],
        "position": [],
        "gene_id": [],
        "gene_symbol": [],
        "overlap_type": [],
    }
    for batch in batch_results:
        for col_name in merged:
            merged[col_name].extend(batch[col_name])

    n_total = len(merged["cpg_id"])
    n_intergenic = sum(1 for ot in merged["overlap_type"] if ot == "intergenic")
    n_mapped = n_total - n_intergenic

    log.info(
        "mapping_complete",
        total_cpgs=len(cpg_ids),
        mapped=n_mapped,
        intergenic=n_intergenic,
        total_records=n_total,
    )

    return pl.DataFrame(merged)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("map-cpg-to-genes")
@click.option(
    "--cpg-list",
    type=click.Path(exists=True),
    default=None,
    help="Text file with one CpG ID per line. If not provided, reads from "
    "the beta matrix header.",
)
@click.option(
    "--genes-parquet",
    type=click.Path(exists=True),
    default="data/external/genes.parquet",
    help="Path to the GENCODE genes Parquet file.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False),
    default="data/processed/cpg_gene_mapping.parquet",
    help="Output Parquet path for the mapping table.",
)
@click.option(
    "--include-intergenic",
    is_flag=True,
    default=False,
    help="Include CpGs with no gene overlap (marked as intergenic).",
)
@click.option(
    "--beta-matrix",
    type=click.Path(exists=True),
    default=None,
    help="Path to the beta matrix CSV to read CpG IDs from the header.",
)
@click.option(
    "--workers",
    type=int,
    default=None,
    help="Number of parallel workers (default: CPU count - 1).",
)
def main(
    cpg_list: str | None,
    genes_parquet: str,
    output: str,
    include_intergenic: bool,
    beta_matrix: str | None,
    workers: int | None,
) -> None:
    """Map CpG positions to overlapping genes.

    Uses coordinate intersection with promoter boundaries defined as
    TSS +/- 1500 bp upstream, 500 bp downstream.  Outputs a mapping table
    with overlap type (promoter, gene_body, intergenic).
    """
    # Load CpG IDs
    if cpg_list:
        cpg_ids = Path(cpg_list).read_text().strip().split("\n")
        log.info("cpg_ids_loaded_from_list", n=len(cpg_ids))
    elif beta_matrix:
        import csv

        with open(beta_matrix, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
        cpg_ids = header[1:]  # skip empty first cell
        log.info("cpg_ids_loaded_from_header", n=len(cpg_ids))
    else:
        click.echo("Provide either --cpg-list or --beta-matrix.", err=True)
        raise SystemExit(1)

    # Load gene coordinates
    genes_df = pl.read_parquet(genes_parquet)
    log.info("genes_loaded", n_genes=len(genes_df))

    # Build index
    gene_index = build_gene_index(genes_df)

    # Map
    mapping_df = map_cpgs_to_genes(
        cpg_ids,
        gene_index,
        report_intergenic=include_intergenic,
        n_workers=workers,
    )

    # Write output
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.write_parquet(out_path)

    click.echo(f"CpG-gene mapping: {len(mapping_df)} records -> {out_path}")

    # Summary by overlap type
    if len(mapping_df) > 0:
        summary = mapping_df.group_by("overlap_type").len().sort("len", descending=True)
        for row in summary.iter_rows(named=True):
            click.echo(f"  {row['overlap_type']}: {row['len']}")


if __name__ == "__main__":
    main()
