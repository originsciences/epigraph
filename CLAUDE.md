# epigraph — Methylation Knowledge Graph

## Project overview

Hybrid TypeDB + Parquet system for DNA methylation biomarker discovery.
Inputs: a dense beta-value matrix (samples × CpGs on GRCh38) and a
clinical metadata spreadsheet.

## Architecture

- **TypeDB 3.x**: biological knowledge graph (genes, pathways, GO, CpG
  – gene overlaps, derived features)
- **Parquet / DuckDB**: dense beta values, per-CpG stats, gene-level
  aggregations
- **Analysis layer**: bridges both — reads beta from Parquet, gene /
  pathway structure from TypeDB, computes features

## Key data formats

- **Beta matrix CSV**: comma-delimited, empty first header cell, CpG
  columns as `chr{N}_{position}` (1-based), sample IDs in the first
  column, missing values as empty string
- **Clinical metadata**: XLSX with `CLIN_*` worksheets, parsed to Parquet
- **CpG naming**: `chr1_10469` means chromosome 1, position 10469
  (1-based, converted from 0-based BED)

## Development workflow

1. Always work on the **dev subset** first (50 samples × 1000 CpGs in
   `data/dev/`).
2. Run `python -m pytest tests/ -q` before committing.
3. Use `dataset-stats` to check pipeline state.
4. The full beta matrix lives in per-chromosome Parquet under
   `data/processed/beta_by_chrom/`.

## Running the pipeline

```
python -m epigraph.pipeline --mode dev
python -m epigraph.pipeline --mode production
python -m epigraph.pipeline --status
```

The pipeline is resumable: each step skips when its output already
exists. Use `--force` to re-run and `--start-from` to resume from a
named step.

## Package structure

```
src/epigraph/
  common/     — shared utilities (genome_coords, identifiers, chunking, io, logging, paths)
  db_build/   — data ingestion, annotation parsing, TypeDB import
  analysis/   — statistical analysis (feature_aggregation, cohort_comparison, pathway_enrichment, ...)
```

## TypeDB notes

- TypeDB 3.8+ syntax: use anonymous relation syntax
  `rel-type (role: $var);` when no relation variable is needed.
- Use `$var isa rel-type, links (role: $player);` when a relation
  variable is needed.
- Use `== $param` for equality matching on function parameters in match
  clauses.
- Schema and functions are in `schemas/typedb/schema.tql` and
  `functions.tql`.

## Critical constraints

- **Never load the full beta matrix into RAM** — use chunked /
  streaming approaches.
- PyArrow `read_csv` with `include_columns` still parses every column,
  so large matrices must be read with raw line parsing.
- Gene-level aggregation transposes before row-aggregating (CpGs →
  rows, samples → columns).
- CpGs legitimately map to multiple overlapping genes — this is
  intentional, not a bug.
- TypeDB 3.x: `entity` / `relation` / `attribute` keywords, `fun` not
  `rule`, no sessions.

## Testing

```
python -m pytest tests/ -q          # all tests
python -m pytest tests/ -k schema   # schema tests only
```

## Annotation sources

- GENCODE (genes), GO / GOA (function terms), Reactome (pathways),
  UCSC CpG islands.
- All parsed to Parquet in `data/external/`.
- Versions pinned in `config/annotation_sources.yaml`.
