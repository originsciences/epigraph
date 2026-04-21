# epigraph

A hybrid [TypeDB](https://typedb.com) + columnar Parquet toolkit for DNA
methylation biomarker discovery.

`epigraph` integrates a dense beta-value matrix (samples × CpGs) with
genomic annotation (genes, pathways, GO / Reactome, regulatory regions,
CpG island context) to support multi-level biomarker analysis at CpG,
gene, pathway, and functional-category resolution.

- **TypeDB 3.x** stores the biological knowledge graph: samples, genes,
  pathways, GO terms, CpG islands, regulatory regions, and the relations
  between them.
- **Per-chromosome Parquet** stores the dense numerical beta matrix
  (one Parquet file per chromosome) and derived statistics.
- A **Python analysis layer** (Polars, scipy, statsmodels) bridges the
  two, computing differential methylation, gene-level aggregation,
  hypermethylation scoring, and pathway enrichment.

```
+---------------------+       +----------------------+       +------------------+
| Per-chrom Parquet   | <---> |   Python analysis    | <---> |    TypeDB 3.x    |
|                     |       |                      |       |                  |
| Beta matrix         |       |  Differential meth   |       |  Genes           |
| Per-chromosome      |       |  Gene aggregation    |       |  Pathways        |
| CpG stats           |       |  HMS scoring         |       |  GO terms        |
| Gene features       |       |  Pathway enrichment  |       |  CpG islands     |
| Clinical metadata   |       |  Biomarker ranking   |       |  Samples         |
+---------------------+       +----------------------+       +------------------+
```

See [docs/architecture.md](docs/architecture.md) for the full design
rationale.

## Quick start

### Prerequisites

- Python 3.12+
- TypeDB 3.x server (optional — only needed for the knowledge-graph steps)
- Enough RAM to hold one chromosome of the beta matrix at a time
  (~1 GB for a ~800-sample cohort)

### Install

```bash
git clone <your-fork-url> epigraph
cd epigraph
pip install -e ".[dev]"
```

Optional extras:

| Extra | Purpose |
|-------|---------|
| `typedb` | TypeDB 3.x Python driver (for graph ingest) |
| `report` | python-docx, for Word-format analysis reports |
| `dev` | pytest, ruff, mypy |
| `notebooks` | jupyter, plotly |
| `all` | everything above |

### Input data

Two inputs are required:

1. A **beta-matrix CSV** with:
   - Empty first header cell, CpG column names formatted as
     `chr{N}_{position}` (1-based), e.g. `chr1_10469`
   - Sample IDs in the first column
   - Missing values as empty string or `NA`
2. A **clinical metadata XLSX** containing one or more worksheets matching
   `CLIN_*` with at least a barcode column and a clinical-category column.

Paths are configured in `config/settings.yaml` or via environment
variables (see `.env.example`).

### Run the pipeline

The pipeline orchestrator runs all steps in order, skipping any whose
output already exists (unless `--force`):

```bash
# Full pipeline on a tiny dev subset (~1 min)
run-pipeline --mode dev

# Full pipeline on production data
run-pipeline --mode production

# Run specific steps only
run-pipeline --steps clinical,annotations,mapping

# Show which steps have completed
run-pipeline --status
```

Individual steps are also available as top-level CLI commands:

```bash
ingest-clinical -o data/processed/clinical_metadata.parquet
build-annotations --config config/annotation_sources.yaml
convert-beta --csv-path data/raw/beta_matrix.csv --output-dir data/processed/beta_by_chrom/
map-cpg-genes --genes data/external/genes.parquet \
              --cpg-source data/processed/beta_by_chrom/ \
              --output data/processed/cpg_gene_mapping.parquet
derive-features --beta-dir data/processed/beta_by_chrom/ \
                --cpg-gene-mapping data/processed/cpg_gene_mapping.parquet \
                --output-dir data/processed/features
hypermethylation --gene-matrix data/processed/features/gene_features.parquet \
                 --metadata data/processed/clinical_metadata.parquet \
                 --output-dir data/processed/features
run-analysis --feature-matrix data/processed/features/gene_features.parquet \
             --metadata data/processed/clinical_metadata.parquet \
             --output-dir data/processed/results
pathway-enrichment --results data/processed/results/CRC_vs_Control.parquet \
                   --output-dir data/processed/enrichment
biomarker-candidates --results-dir data/processed/results/ \
                     --output-dir data/processed/biomarkers
```

## CLI reference

| Command | Purpose |
|---------|---------|
| `run-pipeline` | End-to-end orchestrator with `--mode dev/production`, resumability, and step selection |
| `create-dev-subset` | Stratified small subset (samples × CpGs) for fast iteration |
| `ingest-clinical` | Parse `CLIN_*` worksheets from the clinical XLSX |
| `build-annotations` | Download and parse GENCODE, GO, Reactome, CpG islands into Parquet |
| `convert-beta` | Convert the full beta matrix CSV to per-chromosome Parquet (single-pass, memory-safe) |
| `dataset-stats` | Summary statistics for the beta matrix and clinical metadata |
| `filter-cpgs` | Filter CpGs by variance, coverage, or annotation overlap |
| `map-cpg-genes` | Map CpG sites to genes by genomic coordinate overlap |
| `import-typedb` | Batch-import genes, pathways, GO terms, CpG islands, samples, CpGs, and relations into TypeDB 3.x |
| `derive-features` | Aggregate CpG-level beta values to gene / pathway / GO-term features |
| `run-analysis` | Pairwise cohort comparisons (Mann-Whitney, effect size, BH-FDR) |
| `pathway-enrichment` | Fisher and GSEA pathway / GO term enrichment |
| `biomarker-candidates` | Multi-level biomarker candidate ranking |
| `hypermethylation` | Control-quantile-based hypermethylation scoring |
| `generate-report` | Produce a Word-format summary report from pipeline outputs |
| `visualise` | Volcano plots, pathway dot plots, HMS distributions, heatmaps |

## Package layout

```
epigraph/
├── config/                 Settings and annotation sources
├── data/                   Inputs, intermediates, and outputs (gitignored)
├── docs/                   Architecture and data-source docs
├── schemas/typedb/         TypeDB schema and functions
├── src/epigraph/
│   ├── common/             Shared utilities (paths, logging, I/O, chunking, stats)
│   ├── db_build/           Ingestion, parsing, and TypeDB import
│   ├── analysis/           Statistical analysis
│   └── pipeline.py         End-to-end orchestrator
└── tests/                  Unit, integration, and end-to-end tests
```

## Memory safety

The 22 GB CSV case: never call `pd.read_csv()` on the full matrix. The
codebase enforces memory-safe patterns throughout:

- **CSV → Parquet**: single-pass, line-by-line extraction with a
  pre-allocated `float32` array.
- **Parquet storage**: per-chromosome files so a single chromosome fits
  in RAM.
- **Gene aggregation**: one chromosome loaded at a time.
- **DuckDB**: out-of-core queries scan Parquet directly.
- **TypeDB**: batched inserts (1000 entities per transaction).

## Configuration

All paths and settings live in `config/settings.yaml` and support the
`${VAR:-default}` environment-variable syntax. Key variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `EPIGRAPH_ROOT` | Project root | auto-detected from `pyproject.toml` |
| `EPIGRAPH_ENV` | `dev` (pretty logs) or `prod` (JSON logs) | `dev` |
| `BETA_MATRIX_PATH` | Beta matrix CSV | `data/raw/beta_matrix.csv` |
| `CLINICAL_METADATA_PATH` | Clinical XLSX | `data/raw/clinical_metadata.xlsx` |
| `TYPEDB_ADDRESS` | TypeDB server | `localhost:1729` |
| `TYPEDB_DATABASE` | TypeDB database name | `methylation_graph` |

## Running the tests

```bash
python -m pytest tests/ -q
```

## Further reading

- [Architecture](docs/architecture.md) — hybrid design rationale, TypeDB schema, data flow
- [Annotation strategy](docs/annotation_strategy.md) — rationale for each annotation source
- [Data sources](docs/data_sources.md) — input formats and provenance

## Licence

MIT — see [LICENSE](LICENSE).
