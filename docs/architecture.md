# Methylation Knowledge Graph — Architecture

## A. Executive Summary

This system integrates a ~800-sample × ~4M-CpG DNA methylation beta matrix
with genomic annotation (genes, pathways, GO/KEGG/Reactome, regulatory regions)
to enable biomarker discovery at CpG, gene, pathway, and functional-category
resolution. The architecture is **hybrid**: TypeDB stores the biological knowledge
graph (entities, relationships, annotations, cohorts, derived features) while
raw beta values remain in Parquet/DuckDB for dense numerical queries. An analysis
layer bridges the two, materialising aggregated features that can be written back
into the graph.

### Why hybrid?

The beta matrix contains ~3.3 billion observations (~800 × ~4M). Storing
each as a TypeDB relation would:

- require ~3.3 billion `methylation-observation` relations, each with at least
  3 attributes (sample ref, CpG ref, beta value)
- produce an on-disk footprint of 200–500 GB in TypeDB (versus 22 GB CSV / ~4 GB
  Parquet)
- make bulk numerical operations (mean, variance, t-test across all samples)
  orders of magnitude slower than columnar engines
- offer no semantic benefit: beta values are dense numerical data, not graph
  relationships

TypeDB excels at traversing biological relationships (CpG → gene → pathway →
function) and answering questions like "which pathways have CpGs that are
differentially methylated in CRC?". It does not excel at computing the mean
beta across 4 million columns.

### What lives where

| Layer | Technology | Contents |
|-------|-----------|----------|
| Knowledge graph | TypeDB 3.x | Samples, CpGs (selected), genes, pathways, GO/KEGG/Reactome terms, regulatory regions, cohorts, derived features, provenance |
| Dense matrix store | Per-chromosome Parquet + DuckDB | Full beta matrix (25 files, ~11 GB total), per-CpG summary statistics, per-gene aggregated methylation, HMS scores |
| Analysis engine | Python (Polars/PyArrow/scipy) | Differential methylation, aggregation, pathway scoring, biomarker ranking |
| Clinical metadata | Parquet (derived from XLSX) | Barcodes, clinical categories, age, study metadata |
| Annotation cache | Parquet files in data/external/ | GENCODE genes, GO, KEGG, Reactome, CpG islands, ENCODE cCREs |

### Data flow

```
beta_matrix.csv (22 GB)
    │
    ├─► Per-chromosome Parquet (25 files, ~11 GB total)
    │       │
    │       ├─► Chromosome-at-a-time gene aggregation (54K genes × ~800 samples)
    │       │       │
    │       │       ├─► Per-gene features ──► Cohort comparisons + FDR
    │       │       │
    │       │       └─► HMS scoring (control-quantile thresholds) ──► Parquet
    │       │
    │       ├─► CpG island context mapping (island/shore/shelf/open_sea)
    │       │
    │       └─► DuckDB analytical queries (per-CpG stats) ──► TypeDB
    │
    ├─► Dev subset (1000 CpGs × 50 samples) ──► fast iteration
    │
clinical_metadata.xlsx
    │
    └─► CLIN_* sheets ──► sample metadata Parquet ──► TypeDB (sample entities)

Annotation sources (GENCODE, GO, Reactome, UCSC CpG Islands, ENCODE)
    │
    └─► Parsed ──► Parquet (data/external/) ──► TypeDB (genes, pathways, terms, regions)
```

---

## B. Recommended Annotation Stack for GRCh38

### Default stack

| Domain | Source | Format | Rationale |
|--------|--------|--------|-----------|
| **Genes / transcripts** | GENCODE v45 (GRCh38) | GTF | Gold standard for human gene annotation; superset of RefSeq; used by ENCODE and GTEx; stable versioned releases |
| **MANE Select transcripts** | NCBI MANE v1.3 | GFF3 | Consensus clinical-grade transcript set (GENCODE + RefSeq agreement); one canonical transcript per gene |
| **Gene coordinates** | GENCODE v45 GTF | GTF | Derived from gene annotation above; includes gene body, exons, UTRs, TSS |
| **Pathways** | Reactome (primary) + KEGG (secondary) | GMT / TSV | Reactome: open, well-curated, hierarchical; KEGG: complementary metabolic coverage |
| **GO annotations** | GOA/UniProt (human) | GAF 2.2 | Official GO Consortium annotations for human genes; IEA + curated evidence codes |
| **GO ontology** | Gene Ontology | OBO/OWL | For term hierarchy traversal (BP, MF, CC) |
| **MSigDB gene sets** | MSigDB v2024.1 (H, C2, C5) | GMT | Hallmark (H), curated pathways (C2:CP), GO sets (C5); widely used in GSEA |
| **CpG islands** | UCSC hg38 cpgIslandExt | BED | Standard CpG island definitions; essential for methylation interpretation |
| **Promoters** | Derived from GENCODE TSS | BED | TSS ± 1500/500 bp (or configurable); standard promoter definition |
| **Regulatory regions** | ENCODE cCREs v4 (GRCh38) | BED | Candidate cis-regulatory elements: promoters, enhancers, insulators, DNase-H3K4me3 |
| **CpG → gene mapping** | Computed from GENCODE + CpG islands | Internal | Overlap-based mapping using genomic coordinates |

### Why GENCODE over RefSeq?

- GENCODE is the reference annotation for GRCh38 in ENCODE, GTEx, and most
  large-scale genomics projects
- MANE Select provides RefSeq concordance where needed (clinical interpretation)
- GENCODE includes lncRNAs and other non-coding genes relevant to methylation

### Fallback options

- If KEGG licence restrictions apply: use Reactome + MSigDB C2:CP only
- If ENCODE cCREs are too large: use Roadmap Epigenomics chromatin states for
  colon/rectal tissue specifically
- For CRC-specific annotation: consider supplementing with CRC TCGA methylation
  clusters (CIMP-high, CIMP-low, etc.)

---

## C. Hybrid Architecture Detail

### TypeDB knowledge graph — what goes in

1. **Sample entities** — barcode, clinical_category, age, cohort, study
2. **CpG entities** — chromosome, position, CpG island overlap flag
   - Only CpGs that pass a variance/coverage filter (~50k–200k) get individual
     entities; the full 4M set is referenced by coordinate in Parquet
3. **Gene entities** — symbol, ensembl_id, chromosome, start, end, strand, biotype
4. **Pathway entities** — name, source (Reactome/KEGG), pathway_id
5. **FunctionTerm entities** — GO/KEGG/Reactome term ID, name, namespace (BP/MF/CC)
6. **GenomicRegion entities** — CpG islands, promoters, enhancers, ENCODE cCREs
7. **Cohort entities** — study name, version, date

### Relations in TypeDB

| Relation | Roles | Attributes |
|----------|-------|------------|
| `cpg-gene-overlap` | cpg:overlapping-cpg, gene:overlapped-gene | overlap_type (promoter/body/UTR/intergenic) |
| `gene-pathway-membership` | gene:member-gene, pathway:containing-pathway | source, evidence |
| `gene-function-annotation` | gene:annotated-gene, function-term:annotating-term | evidence_code, source |
| `cpg-region-overlap` | cpg:overlapping-cpg, genomic-region:overlapped-region | — |
| `sample-cohort-membership` | sample:member-sample, cohort:containing-cohort | — |
| `derived-methylation-feature` | sample:measured-sample, gene:measured-gene | mean_beta, median_beta, n_cpgs |
| `differential-signal` | gene:tested-gene, cohort:reference-cohort, cohort:comparison-cohort | effect_size, p_value, q_value |

### What stays outside TypeDB

- **Raw beta values** (3.3B observations) → Parquet, queried via DuckDB/Polars
- **Per-CpG summary statistics** → Parquet (mean, variance, missingness per CpG)
- **Intermediate analysis results** → Parquet (before promotion to TypeDB)

### Linking strategy

The bridge between TypeDB and Parquet is **coordinate identity**:

- TypeDB CpG entities have `chromosome` + `position` attributes
- Parquet columns are named `chr{N}_{position}` (1-based)
- Gene-level aggregation joins on genomic interval overlap
- Analysis code reads coordinates from TypeDB, selects columns from Parquet,
  computes features, writes results back to TypeDB as derived relations

### Why not store everything in TypeDB?

Pressure-testing the "all in TypeDB" alternative:

| Concern | Impact |
|---------|--------|
| **Ingestion time** | 3.3B inserts at ~5k/sec (optimistic) = ~7.6 days continuous |
| **Storage** | ~200–500 GB on disk for relations + indices |
| **Query performance** | Aggregating 4M beta values per sample requires full scan of relations; columnar engines do this 100–1000× faster |
| **Memory** | TypeDB loads indices into RAM; 3.3B relations would require >64 GB RAM |
| **Value proposition** | Beta values have no semantic structure; they are numbers, not relationships |

**Verdict**: Hybrid is the only practical architecture. TypeDB stores the
knowledge graph; Parquet/DuckDB stores the numbers.

### Per-chromosome Parquet storage

The full beta matrix (22 GB CSV) is converted into 25 per-chromosome Parquet
files (`beta_chr1.parquet` through `beta_chrY.parquet`), totalling approximately
11 GB. This design replaces the original single-file Parquet approach for two
reasons:

1. **Memory management**: The full matrix has ~4M columns. Loading it as
   a single Parquet file (even with column selection) requires substantial RAM.
   Per-chromosome files range from ~50 MB (chrY) to ~1.1 GB (chr1), making it
   feasible to load one at a time.

2. **DuckDB glob failure**: DuckDB's `read_parquet('*.parquet')` glob syntax
   was tested for cross-chromosome gene aggregation but failed on the production
   dataset. When a gene's CpGs span columns in multiple large Parquet files,
   DuckDB attempts to open and memory-map all files simultaneously, causing
   out-of-memory errors on machines with less than 64 GB RAM.

The conversion is implemented as a single-pass line-by-line reader
(`convert_beta_to_parquet.py`). It reads the CSV header to build a
column-index-to-chromosome mapping, then streams each row, distributing values
into per-chromosome numpy float32 arrays. After all ~800 rows are processed,
each array is written as a Parquet file. Peak memory is approximately 4 GB.

### Chromosome-at-a-time gene aggregation

The `aggregate_by_chrom.py` module implements production gene-level aggregation
by processing one chromosome file at a time:

1. Load `beta_chr{N}.parquet` into memory (~1 GB for chr1).
2. For each gene with CpGs on this chromosome, extract the relevant columns
   and compute the per-sample mean beta.
3. Free the chromosome data.
4. Repeat for all 25 chromosome files.

Genes whose CpGs span multiple chromosomes (rare) are handled by accumulating
partial per-chromosome contributions and averaging at the end.

**Production performance**: 54,289 genes across all samples aggregated in 97
seconds, with ~2 GB peak memory.

### Hypermethylation scoring

Hypermethylation scoring (HMS) is a derived feature that quantifies the degree
of aberrant hypermethylation per sample. The approach is ported from the
`rules_based_classifier` pipeline:

1. For each gene, compute a per-gene threshold as the Nth quantile (default
   q=0.99) of that gene's beta values across Control samples only.
2. For each sample, count how many genes exceed their gene-specific threshold.
3. This count is the hypermethylation score (HMS).

HMS is computed after gene-level aggregation and stored alongside gene features.
It serves as an input to the biomarker candidate ranking and can be used as a
standalone classification feature.

### Pipeline orchestrator

The `run-pipeline` CLI command (`pipeline.py`) is the primary entry point for
running the full analysis. It orchestrates all steps in dependency order with
built-in resumability:

- Each step checks whether its output files already exist and skips if so
  (unless `--force` is specified).
- State is tracked in `data/.pipeline_state.json` with timestamps.
- Two modes are supported: `--mode dev` (50 samples, 1000 CpGs) and
  `--mode production` (full full-scale matrix).
- Individual steps can be selected with `--steps step1,step2,...`.
- Current state can be inspected with `--status`.

### Selective CpG import to TypeDB

Not all 4M CpGs need individual TypeDB entities. Strategy:

1. **Variance filter**: Keep CpGs with variance > threshold across samples
   (typically reduces to 50k–200k informative CpGs)
2. **Annotation filter**: Keep CpGs that overlap annotated genes, promoters,
   or CpG islands
3. **Biomarker candidates**: CpGs identified by differential methylation analysis
4. **All genes get entities** regardless of CpG coverage

This means TypeDB has ~100k–200k CpG entities (manageable) rather than 4M.

---

## D. Operational Strategy

### Dev-first, production-second

1. **Phase 1 — Dev subset**: 50 samples × 1000 CpGs
   - Test schema, ingestion, coordinate parsing, clinical joins
   - Fast iteration (<1 min full pipeline)
2. **Phase 2 — Annotation integration**: Full gene/pathway/GO annotation
   - Test CpG-to-gene mapping on dev subset
   - Validate TypeDB traversals
3. **Phase 3 — Full matrix conversion**: Convert 22 GB CSV → per-chromosome Parquet (single-pass)
   - 25 output files, ~11 GB total
   - Validate column names, dtypes, missingness
   - No TypeDB import of raw values
4. **Phase 4 — Production analysis**: Full differential methylation + aggregation
   - Chromosome-at-a-time gene aggregation (54K genes, 97s)
   - Hypermethylation scoring (HMS)
   - Cohort comparisons with FDR correction
   - Pathway enrichment (Reactome, GO)
   - Biomarker candidate ranking

### Memory safety

- **CSV reading**: Single-pass line-by-line extraction with per-chromosome arrays
- **Parquet storage**: Per-chromosome files (25 files, ~11 GB total)
- **Gene aggregation**: One chromosome loaded at a time (~2 GB peak)
- **DuckDB**: Out-of-core by default; scans Parquet directly without loading
- **TypeDB ingestion**: Batch inserts (1000 entities per transaction)
- **Never**: `pd.read_csv("beta_matrix.csv")` on the full file

### Reproducibility

- Annotation versions pinned in `config/annotation_sources.yaml`
- Dev subset generation is deterministic (seeded random selection)
- All S3 paths configurable via `.env` / `config/settings.yaml`
- Provenance tracked in TypeDB (which annotation version, which run)

---

## E. Risks and Tradeoffs

| Risk | Mitigation |
|------|-----------|
| TypeDB 3.x is relatively new; Python driver may have bugs | Pin driver version; write integration tests; have fallback to Neo4j if critical bugs found |
| Full CpG-to-gene mapping for 4M CpGs is expensive | Pre-compute and cache as Parquet; only import annotated CpGs to TypeDB |
| KEGG licence restrictions may prevent redistribution | Default to Reactome; use MSigDB C2:CP as KEGG proxy |
| Clinical metadata in XLSX may have inconsistent formatting | Defensive parsing with explicit column mapping; validation tests |
| 22 GB CSV → Parquet conversion may take significant time | One-time operation; can be parallelised by chromosome |
| Gene-level aggregation depends on overlap definition (promoter vs body vs TSS) | Make overlap type configurable; default to promoter + body |
| Multiple-testing burden at pathway level with few samples (N) | Use permutation-based tests; report effect sizes alongside p-values |

## F. Implemented Capabilities

1. Repository scaffold, dev subset, clinical metadata parsing
2. Full annotation stack (GENCODE, GO, Reactome, CpG islands, ENCODE cCREs)
3. CpG-to-gene coordinate mapping and TypeDB schema
4. Per-chromosome Parquet conversion (single-pass, memory-safe)
5. Chromosome-at-a-time gene aggregation
6. Pairwise cohort comparisons with FDR correction
7. Pathway (Reactome) and GO term over-representation enrichment
8. Hypermethylation scoring (HMS) and biomarker candidate ranking
9. Pipeline orchestrator with dev/production modes
10. Unit, integration, and end-to-end test coverage
