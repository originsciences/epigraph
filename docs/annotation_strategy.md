# Annotation Strategy

This document describes the rationale, content, and handling for each genomic
annotation source used in the methylation knowledge graph. All annotations
target the **GRCh38** (hg38) genome build. Versions are pinned in
`config/annotation_sources.yaml`.

---

## 1. GENCODE v45 -- Gene Annotation

### Why GENCODE

GENCODE is the reference gene annotation for GRCh38, used by ENCODE, GTEx, and
the majority of large-scale genomics projects. It is a superset of RefSeq,
meaning it includes all RefSeq transcripts plus additional lncRNAs, antisense
genes, and other non-coding elements that are relevant to methylation biology.

Key advantages over RefSeq alone:

- Comprehensive non-coding gene coverage (lncRNAs, antisense, pseudogenes)
- Stable versioned releases with clear changelogs
- Consistent coordinate system aligned with the GRCh38 primary assembly
- Direct compatibility with ENCODE cCREs and GTEx expression data

### What we extract

From the GENCODE v45 comprehensive GTF, we extract **gene-level** records:

| Field | Source | Notes |
|-------|--------|-------|
| `gene_id` | `gene_id` attribute | Ensembl ID, version suffix stripped (e.g. `ENSG00000223972`) |
| `gene_symbol` | `gene_name` attribute | HGNC symbol (e.g. `DDX11L2`) |
| `chrom` | Column 1 | Chromosome (e.g. `chr1`) |
| `start` | Column 4 | 1-based inclusive (GTF convention) |
| `end` | Column 5 | 1-based inclusive |
| `strand` | Column 7 | `+` or `-` |
| `biotype` | `gene_type` attribute | e.g. `protein_coding`, `lncRNA`, `pseudogene` |

PAR (pseudoautosomal region) genes appearing on both chrX and chrY are
deduplicated, keeping the first occurrence.

### Format and download

- **URL**: `https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_45/gencode.v45.annotation.gtf.gz`
- **Format**: GTF 2.5 (tab-delimited, gzipped)
- **Size**: ~45 MB compressed
- **Output**: `data/external/genes.parquet`

---

## 2. MANE Select v1.3 -- Consensus Clinical Transcript

### Purpose

MANE (Matched Annotation from NCBI and EBI) Select designates one canonical
transcript per protein-coding gene, agreed upon by both GENCODE and RefSeq.
This is the recommended transcript for:

- Clinical variant interpretation
- Reporting methylation changes at specific gene regions
- Resolving ambiguity when a CpG maps to multiple transcripts of the same gene

### When to use

MANE Select is **not used for the primary CpG-to-gene mapping** (which operates
at the gene body level). It is used when:

1. A CpG falls in a promoter and the analysis needs a single canonical TSS
2. Reporting results to clinicians who expect RefSeq transcript IDs
3. Cross-referencing with ClinVar or other clinical databases

### Format and download

- **URL**: `https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/release_1.3/MANE.GRCh38.v1.3.ensembl_genomic.gff.gz`
- **Format**: GFF3 (gzipped)
- **Size**: ~5 MB compressed
- **Local**: `data/external/MANE.GRCh38.v1.3.gff.gz`

---

## 3. GO / GOA -- Gene Ontology Functional Annotations

### What GO provides

Gene Ontology annotations link genes to standardised functional terms across
three namespaces:

- **Biological Process (BP)**: e.g. "DNA methylation", "apoptotic process"
- **Molecular Function (MF)**: e.g. "DNA binding", "methyltransferase activity"
- **Cellular Component (CC)**: e.g. "nucleus", "chromatin"

### Evidence code filtering strategy

GOA annotations include an evidence code indicating how the annotation was
made. Our filtering approach:

| Evidence Type | Codes | Include? | Rationale |
|---------------|-------|----------|-----------|
| Experimental | EXP, IDA, IPI, IMP, IGI, IEP | Yes | Highest confidence |
| Computational | ISS, ISO, ISA, ISM, IGC, IBA, IBD, IKR, IRD, RCA | Yes | Useful for coverage |
| Author statement | TAS, NAS | Yes | Literature-supported |
| Electronic | IEA | Yes (default) | Provides broad coverage; can be excluded for stringent analysis |
| No data | ND | No | Explicitly "no data available" |

The default is to include all evidence codes except ND. For stringent
analyses, IEA annotations can be filtered out downstream using the
`evidence_code` column in the parsed Parquet file.

### Qualifier handling

The GAF `qualifier` field (column 4) can contain `NOT`, indicating a negative
annotation. These are preserved in the parsed output and should be filtered
out before pathway analysis.

### Format and download

- **Ontology**: `http://purl.obolibrary.org/obo/go/go-basic.obo` (OBO format)
- **Annotations**: `https://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz` (GAF 2.2)
- **Output**: `data/external/go_annotations.parquet`
- **Columns**: `gene_symbol`, `go_id`, `evidence_code`, `aspect`, `qualifier`

---

## 4. Reactome -- Pathway Database (Primary)

### Why Reactome as primary

Reactome is the primary pathway source because it is:

- **Open access**: no licence restrictions on redistribution or derived works
- **Well-curated**: expert-reviewed, with detailed reaction-level evidence
- **Hierarchical**: pathways nest within broader categories (e.g. "Signal
  Transduction" > "Signaling by WNT" > "beta-catenin degradation")
- **Cross-referenced**: links to UniProt, Ensembl, ChEBI, GO

### Pathway hierarchy handling

Reactome pathways form a directed acyclic graph (DAG), not a flat list. Our
approach:

1. Import all human pathways as flat entities in TypeDB
2. Store the pathway hierarchy as `pathway-hierarchy` relations (parent-child)
3. For enrichment analysis, use only **leaf pathways** (most specific) to avoid
   double-counting
4. For reporting, propagate significant leaf results up the hierarchy

### Files used

| File | Contents | URL |
|------|----------|-----|
| `ReactomePathways.txt` | Pathway ID, name, species | `https://reactome.org/download/current/ReactomePathways.txt` |
| `Ensembl2Reactome.txt` | Ensembl gene ID to pathway mapping | `https://reactome.org/download/current/Ensembl2Reactome.txt` |

### Output

- `data/external/reactome_pathways.parquet` -- pathway metadata
- `data/external/reactome_gene_pathway.parquet` -- gene-to-pathway mapping
- Columns: `gene_id` (Ensembl, version-stripped), `pathway_id`, `evidence_code`
- Only human pathways (filtered by "Homo sapiens")

---

## 5. KEGG -- Metabolic and Signaling Pathways (Secondary)

### Licence considerations

KEGG imposes licence restrictions on bulk download and redistribution for
non-academic use. To avoid licence complications:

- **Default**: Do not download KEGG directly
- **Proxy strategy**: Use MSigDB C2:CP:KEGG gene sets (see section 6), which
  are a frozen snapshot of KEGG pathways redistributed under MSigDB's terms
- **Alternative**: If the project has a KEGG licence, download via the KEGG
  REST API (`https://rest.kegg.jp/`)

### What KEGG adds

KEGG provides metabolic pathway coverage that Reactome handles less
comprehensively, particularly:

- Central carbon metabolism
- Amino acid biosynthesis and degradation
- Drug metabolism pathways
- Disease-specific pathway maps (e.g. "Colorectal cancer" KEGG map 05210)

### Configuration

In `config/annotation_sources.yaml`:

```yaml
kegg:
  use_msigdb_proxy: true
```

Set `use_msigdb_proxy: false` and provide API credentials if direct KEGG
access is available.

---

## 6. MSigDB -- Gene Set Collections

### Which collections

We use three MSigDB collections:

| Collection | Name | Gene Sets | Purpose |
|------------|------|-----------|---------|
| **H** | Hallmark | 50 | Well-defined biological states and processes; reduced redundancy; ideal for initial screening |
| **C2:CP** | Canonical Pathways | ~2,900 | Curated pathways from Reactome, KEGG, BioCarta, PID; serves as KEGG proxy when `use_msigdb_proxy: true` |
| **C5** | GO Gene Sets | ~15,000 | GO terms formatted as gene sets; alternative to direct GO analysis with pre-defined gene lists |

### Download instructions

MSigDB requires free registration at <https://www.gsea-msigdb.org/gsea/register.jsp>.

After registration, download GMT files:

```bash
# Hallmark
wget -O data/external/msigdb/h.all.v2024.1.Hs.symbols.gmt \
  "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/h.all.v2024.1.Hs.symbols.gmt"

# Canonical Pathways (includes KEGG, Reactome, BioCarta, PID)
wget -O data/external/msigdb/c2.cp.v2024.1.Hs.symbols.gmt \
  "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c2.cp.v2024.1.Hs.symbols.gmt"

# GO gene sets
wget -O data/external/msigdb/c5.all.v2024.1.Hs.symbols.gmt \
  "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs/c5.all.v2024.1.Hs.symbols.gmt"
```

Alternatively, use the `msigdbr` R package for programmatic access.

### GMT format

Each line in a GMT file is:

```
PATHWAY_NAME<TAB>URL_OR_DESCRIPTION<TAB>GENE1<TAB>GENE2<TAB>...
```

Gene identifiers are HGNC symbols.

---

## 7. CpG Islands -- UCSC cpgIslandExt

### What CpG islands provide

CpG islands are regions of elevated CG dinucleotide density, typically found
at gene promoters. A CpG site's relationship to CpG islands is one of the most
important features for methylation interpretation:

- **Island**: Within a CpG island (typically unmethylated in normal tissue)
- **Shore**: 0--2 kb flanking an island (where most cancer-related methylation
  changes occur)
- **Shelf**: 2--4 kb flanking an island
- **Open sea**: >4 kb from any island

### Source and format

The UCSC `cpgIslandExt` table provides:

| Column | Description |
|--------|-------------|
| `chrom` | Chromosome |
| `chromStart` | Start position (0-based) |
| `chromEnd` | End position (exclusive) |
| `name` | Island identifier (e.g. `CpG: 111`) |
| `length` | Island length in bp |
| `cpgNum` | Number of CpG dinucleotides |
| `gcNum` | Number of G+C bases |
| `perCpg` | Percentage of CpG |
| `perGc` | Percentage of GC content |
| `obsExp` | Observed/expected CpG ratio |

Coordinates are **0-based half-open** (BED convention).

### Download

- **URL**: `https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz`
- **Output**: `data/external/cpg_islands.parquet`

---

## 8. ENCODE cCREs v4 -- Candidate Cis-Regulatory Elements

### What cCREs add for methylation interpretation

ENCODE candidate cis-regulatory elements (cCREs) identify functional genomic
regions based on epigenomic signals (DNase-seq, H3K4me3, H3K27ac, CTCF).
They categorise regions as:

| cCRE Type | Abbreviation | Methylation Relevance |
|-----------|-------------|----------------------|
| Promoter-like | PLS | Methylation at PLS regions strongly correlates with gene silencing |
| Proximal enhancer-like | pELS | Enhancer methylation may affect distal gene regulation |
| Distal enhancer-like | dELS | Methylation changes at enhancers are frequent in cancer |
| DNase-H3K4me3 | DNase-H3K4me3 | Active regulatory regions |
| CTCF-only | CTCF-only | Insulator elements; methylation can disrupt boundary function |

Knowing whether a differentially methylated CpG falls within a cCRE provides
functional context that pure gene-body overlap cannot capture. A CpG in an
enhancer cCRE that is hypermethylated in CRC suggests a potential distal
regulatory mechanism.

### Download

- **URL**: `https://downloads.wenglab.org/V3/GRCh38-cCREs.bed`
- **Format**: BED (tab-delimited, no header)
- **Size**: ~50 MB
- **Local**: `data/external/GRCh38-cCREs.bed`

---

## Version Pinning

All annotation versions are recorded in `config/annotation_sources.yaml`. The
file specifies exact version numbers, URLs, and local paths. To see the current
versions:

```bash
cat config/annotation_sources.yaml
```

Current pinned versions:

| Source | Version | Date |
|--------|---------|------|
| GENCODE | v45 | 2023-06 |
| MANE Select | v1.3 | 2024-01 |
| GO/GOA | Rolling (date of download) | -- |
| Reactome | v89 | 2024 Q2 |
| MSigDB | v2024.1 | 2024-06 |
| UCSC CpG islands | Rolling (hg38) | -- |
| ENCODE cCREs | v4 | 2024 |

For GO/GOA and UCSC CpG islands, which do not have discrete version numbers,
record the download date in a `data/external/DOWNLOAD_LOG.txt` file.

---

## How to Update Annotations

### Step 1: Update version numbers in config

Edit `config/annotation_sources.yaml` to point to new URLs and version numbers.

### Step 2: Re-download

```bash
build-annotations --config config/annotation_sources.yaml --force
```

The `--force` flag re-downloads even if local files exist.

### Step 3: Re-import into TypeDB

```bash
# Drop and recreate the database, or use --step to re-import specific entity types
import-typedb --step genes
import-typedb --step pathways
import-typedb --step function_terms
import-typedb --step genomic_regions
import-typedb --step gene_pathway
import-typedb --step gene_function
```

### Step 4: Re-run analysis

Feature aggregation and cohort comparisons should be re-run after annotation
updates, as CpG-to-gene mappings may change.

### Step 5: Document the update

Record the update in `data/external/DOWNLOAD_LOG.txt` with the date, old
version, new version, and any notable changes.
