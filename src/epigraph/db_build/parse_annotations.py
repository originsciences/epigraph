"""Download and parse genomic annotations into Parquet files.

Handles four annotation sources:

1. **GENCODE GTF** -- gene coordinates, symbols, biotypes
2. **GO/GOA GAF** -- gene-to-GO-term functional annotations
3. **Reactome** -- gene-to-pathway mappings
4. **UCSC CpG islands** -- CpG-island coordinates

All outputs are written to ``data/external/`` as Parquet files.  Downloads
are incremental: existing files are reused unless ``--force`` is specified.
"""

from __future__ import annotations

import gzip
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import click
import polars as pl
import yaml

from epigraph.common.logging import get_logger

log = get_logger(__name__)

# Only these URL schemes are permitted for annotation downloads.  Blocks
# ``file://`` (local-file read / SSRF) and other surprising schemes while
# keeping the genuinely used HTTP(S) and FTP sources.
_ALLOWED_DOWNLOAD_SCHEMES = frozenset({"http", "https", "ftp"})


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _load_annotation_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load annotation source URLs and paths from YAML config."""
    if config_path is None:
        config_path = Path("config/annotation_sources.yaml")
    if config_path.exists():
        with open(config_path) as fh:
            return yaml.safe_load(fh) or {}
    log.warning("config_not_found", path=str(config_path))
    return {}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def download_if_missing(url: str, local_path: Path, force: bool = False) -> Path:
    """Download a file if it does not already exist locally.

    Args:
        url: Remote URL to download.
        local_path: Local destination path.
        force: Re-download even if the file exists.

    Returns:
        The local file path.
    """
    if local_path.exists() and not force:
        log.info("file_exists_skipping", path=str(local_path))
        return local_path

    scheme = urllib.parse.urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_DOWNLOAD_SCHEMES:
        raise ValueError(
            f"Refusing to download from URL with scheme {scheme!r}; "
            f"allowed schemes: {sorted(_ALLOWED_DOWNLOAD_SCHEMES)}"
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading", url=url, dest=str(local_path))

    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, str(local_path))  # noqa: S310 — scheme allow-listed above
            break
        except (urllib.error.URLError, OSError) as exc:
            if attempt == 2:
                raise
            log.warning("download_retry", attempt=attempt + 1, error=str(exc))

    size_mb = f"{local_path.stat().st_size / 1e6:.1f}"
    log.info("download_complete", path=str(local_path), size_mb=size_mb)
    return local_path


# ---------------------------------------------------------------------------
# GENCODE GTF parser
# ---------------------------------------------------------------------------


def parse_gencode_gtf(
    gtf_path: Path,
    output_path: Path,
    feature_type: str = "gene",
) -> pl.DataFrame:
    """Parse a GENCODE GTF file into a genes DataFrame.

    Extracts gene-level records with fields: gene_id, gene_symbol, chrom,
    start, end, strand, biotype.

    Args:
        gtf_path: Path to the (possibly gzipped) GTF file.
        output_path: Destination Parquet path.
        feature_type: GTF feature type to extract (default ``"gene"``).

    Returns:
        Polars DataFrame with gene records.
    """
    log.info("parsing_gencode_gtf", path=str(gtf_path))

    col_gene_id: list[str] = []
    col_gene_symbol: list[str] = []
    col_chrom: list[str] = []
    col_start: list[int] = []
    col_end: list[int] = []
    col_strand: list[str] = []
    col_biotype: list[str] = []

    opener = gzip.open if str(gtf_path).endswith(".gz") else open

    with opener(gtf_path, "rt") as fh:  # type: ignore[call-overload]
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[2] != feature_type:
                continue

            attrs = _parse_gtf_attributes(fields[8])

            col_gene_id.append(attrs.get("gene_id", "").split(".")[0])
            col_gene_symbol.append(attrs.get("gene_name", ""))
            col_chrom.append(fields[0])
            col_start.append(int(fields[3]))   # GTF is 1-based inclusive
            col_end.append(int(fields[4]))      # GTF end is 1-based inclusive
            col_strand.append(fields[6])
            col_biotype.append(attrs.get("gene_type", attrs.get("gene_biotype", "")))

    df = pl.DataFrame({
        "gene_id": col_gene_id,
        "gene_symbol": col_gene_symbol,
        "chrom": col_chrom,
        "start": col_start,
        "end": col_end,
        "strand": col_strand,
        "biotype": col_biotype,
    })

    # Deduplicate by gene_id (PAR genes appear on chrX and chrY)
    df = df.unique(subset=["gene_id"], keep="first", maintain_order=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    log.info("gencode_parsed", n_genes=len(df), output=str(output_path))
    return df


def _parse_gtf_attributes(attr_string: str) -> dict[str, str]:
    """Parse the GTF attribute column into a dict.

    GTF attributes look like: ``gene_id "ENSG00000223972"; gene_name "DDX11L2";``
    """
    attrs: dict[str, str] = {}
    for item in attr_string.strip().rstrip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(" ", 1)
        if len(parts) == 2:
            key = parts[0]
            val = parts[1].strip('"')
            attrs[key] = val
    return attrs


# ---------------------------------------------------------------------------
# GO/GOA GAF parser
# ---------------------------------------------------------------------------


def parse_goa_gaf(
    gaf_path: Path,
    output_path: Path,
) -> pl.DataFrame:
    """Parse a GOA GAF file into gene-to-GO-term mappings.

    GAF 2.2 format: tab-delimited, lines starting with ``!`` are comments.
    Key columns: 1=DB, 2=DB_Object_ID, 3=DB_Object_Symbol, 4=Qualifier,
    5=GO_ID, 7=Evidence_Code, 9=Aspect (F/P/C).

    Args:
        gaf_path: Path to the (possibly gzipped) GAF file.
        output_path: Destination Parquet path.

    Returns:
        Polars DataFrame with columns:
        ``['gene_symbol', 'go_id', 'evidence_code', 'aspect', 'qualifier']``.
    """
    log.info("parsing_goa_gaf", path=str(gaf_path))

    aspect_map = {"F": "molecular_function", "P": "biological_process", "C": "cellular_component"}

    col_gene_symbol: list[str] = []
    col_go_id: list[str] = []
    col_evidence_code: list[str] = []
    col_aspect: list[str] = []
    col_qualifier: list[str] = []

    opener = gzip.open if str(gaf_path).endswith(".gz") else open

    with opener(gaf_path, "rt") as fh:  # type: ignore[call-overload]
        for line in fh:
            if line.startswith("!"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 15:
                continue
            # Only keep human annotations (taxon:9606)
            taxon = fields[12]
            if "taxon:9606" not in taxon:
                continue

            col_gene_symbol.append(fields[2])
            col_go_id.append(fields[4])
            col_evidence_code.append(fields[6])
            col_aspect.append(aspect_map.get(fields[8], fields[8]))
            col_qualifier.append(fields[3])

    df = pl.DataFrame({
        "gene_symbol": col_gene_symbol,
        "go_id": col_go_id,
        "evidence_code": col_evidence_code,
        "aspect": col_aspect,
        "qualifier": col_qualifier,
    })
    df = df.unique()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    log.info("goa_parsed", n_annotations=len(df), output=str(output_path))
    return df


# ---------------------------------------------------------------------------
# Reactome parser
# ---------------------------------------------------------------------------


def parse_reactome(
    pathway_path: Path,
    gene_pathway_path: Path,
    output_pathways: Path,
    output_gene_map: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse Reactome pathway list and Ensembl-to-Reactome gene mapping.

    Args:
        pathway_path: Path to ``ReactomePathways.txt``.
        gene_pathway_path: Path to ``Ensembl2Reactome.txt``.
        output_pathways: Destination Parquet for pathway metadata.
        output_gene_map: Destination Parquet for gene-pathway mapping.

    Returns:
        Tuple of (pathways_df, gene_pathway_df).
    """
    log.info("parsing_reactome")

    # Parse pathways: ID \t name \t species
    pw_ids: list[str] = []
    pw_names: list[str] = []
    pw_sources: list[str] = []
    with open(pathway_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                # Only keep human pathways
                if "Homo sapiens" in parts[2]:
                    pw_ids.append(parts[0])
                    pw_names.append(parts[1])
                    pw_sources.append("Reactome")

    pathways_df = pl.DataFrame({
        "pathway_id": pw_ids,
        "pathway_name": pw_names,
        "pathway_source": pw_sources,
    })

    # Parse Ensembl2Reactome: gene_id \t pathway_id \t url \t pathway_name \t evidence \t species
    gp_gene_ids: list[str] = []
    gp_pathway_ids: list[str] = []
    gp_evidence_codes: list[str] = []
    with open(gene_pathway_path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 6:
                if "Homo sapiens" in parts[5]:
                    gp_gene_ids.append(parts[0].split(".")[0])  # strip version
                    gp_pathway_ids.append(parts[1])
                    gp_evidence_codes.append(parts[4] if len(parts) > 4 else "")

    gene_pathway_df = pl.DataFrame({
        "gene_id": gp_gene_ids,
        "pathway_id": gp_pathway_ids,
        "evidence_code": gp_evidence_codes,
    }).unique()

    for df, path in [(pathways_df, output_pathways), (gene_pathway_df, output_gene_map)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)

    log.info(
        "reactome_parsed",
        n_pathways=len(pathways_df),
        n_gene_mappings=len(gene_pathway_df),
    )
    return pathways_df, gene_pathway_df


# ---------------------------------------------------------------------------
# UCSC CpG islands parser
# ---------------------------------------------------------------------------


def parse_cpg_islands(
    bed_path: Path,
    output_path: Path,
) -> pl.DataFrame:
    """Parse UCSC CpG island annotations (cpgIslandExt table).

    The file is tab-delimited with columns:
    bin, chrom, chromStart, chromEnd, name, length, cpgNum, gcNum, perCpg, perGc,
    obsExp.

    Coordinates are 0-based half-open (BED convention).

    Args:
        bed_path: Path to the (possibly gzipped) cpgIslandExt file.
        output_path: Destination Parquet path.

    Returns:
        Polars DataFrame with columns:
        ``['region_id', 'chrom', 'start', 'end', 'cpg_count', 'gc_fraction',
        'obs_exp_ratio']``.
    """
    log.info("parsing_cpg_islands", path=str(bed_path))

    col_region_id: list[str] = []
    col_chrom: list[str] = []
    col_start: list[int] = []
    col_end: list[int] = []
    col_cpg_count: list[int] = []
    col_gc_fraction: list[float] = []
    col_obs_exp_ratio: list[float] = []

    opener = gzip.open if str(bed_path).endswith(".gz") else open

    with opener(bed_path, "rt") as fh:  # type: ignore[call-overload]
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 11:
                continue

            col_region_id.append(fields[4])      # name
            col_chrom.append(fields[1])
            col_start.append(int(fields[2]))      # 0-based
            col_end.append(int(fields[3]))        # exclusive
            col_cpg_count.append(int(fields[6]))
            col_gc_fraction.append(float(fields[9]) / 100.0 if fields[9] else 0.0)
            col_obs_exp_ratio.append(float(fields[10]) if fields[10] else 0.0)

    df = pl.DataFrame({
        "region_id": col_region_id,
        "chrom": col_chrom,
        "start": col_start,
        "end": col_end,
        "cpg_count": col_cpg_count,
        "gc_fraction": col_gc_fraction,
        "obs_exp_ratio": col_obs_exp_ratio,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)

    log.info("cpg_islands_parsed", n_islands=len(df), output=str(output_path))
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("build-annotations")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default="config/annotation_sources.yaml",
    help="Path to annotation sources YAML config.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False),
    default="data/external",
    help="Directory for annotation output files.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-download files even if they exist locally.",
)
@click.option(
    "--skip-download",
    is_flag=True,
    default=False,
    help="Skip downloading; parse only existing local files.",
)
def main(
    config_path: str,
    output_dir: str,
    force: bool,
    skip_download: bool,
) -> None:
    """Download and parse genomic annotations into Parquet files.

    Handles GENCODE GTF (genes), GO/GOA GAF (function terms), Reactome
    (pathways), and UCSC CpG islands.  Downloads are incremental by default.
    """
    cfg = _load_annotation_config(Path(config_path))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- GENCODE ---
    gencode_cfg = cfg.get("gencode", {})
    gtf_url = gencode_cfg.get("gtf_url", "")
    gtf_local = Path(gencode_cfg.get("local_path", out / "gencode.gtf.gz"))

    if not skip_download and gtf_url:
        download_if_missing(gtf_url, gtf_local, force=force)
    if gtf_local.exists():
        parse_gencode_gtf(gtf_local, out / "genes.parquet")
        click.echo(f"GENCODE genes -> {out / 'genes.parquet'}")
    else:
        click.echo("GENCODE GTF not found; skipping.", err=True)

    # --- GO/GOA ---
    go_cfg = cfg.get("go", {})
    gaf_url = go_cfg.get("annotation_url", "")
    gaf_local = Path(go_cfg.get("local_path_gaf", out / "goa_human.gaf.gz"))

    if not skip_download and gaf_url:
        download_if_missing(gaf_url, gaf_local, force=force)
    if gaf_local.exists():
        parse_goa_gaf(gaf_local, out / "go_annotations.parquet")
        click.echo(f"GO annotations -> {out / 'go_annotations.parquet'}")
    else:
        click.echo("GOA GAF not found; skipping.", err=True)

    # --- Reactome ---
    reactome_cfg = cfg.get("reactome", {})
    pw_url = reactome_cfg.get("pathway_url", "")
    gp_url = reactome_cfg.get("gene_pathway_url", "")
    pw_local = Path(reactome_cfg.get("local_path_pathways", out / "ReactomePathways.txt"))
    gp_local = Path(reactome_cfg.get("local_path_gene_map", out / "Ensembl2Reactome.txt"))

    if not skip_download:
        if pw_url:
            download_if_missing(pw_url, pw_local, force=force)
        if gp_url:
            download_if_missing(gp_url, gp_local, force=force)
    if pw_local.exists() and gp_local.exists():
        parse_reactome(
            pw_local,
            gp_local,
            out / "reactome_pathways.parquet",
            out / "reactome_gene_pathway.parquet",
        )
        click.echo(f"Reactome -> {out / 'reactome_pathways.parquet'}")
    else:
        click.echo("Reactome files not found; skipping.", err=True)

    # --- UCSC CpG islands ---
    cpgi_cfg = cfg.get("cpg_islands", {})
    cpgi_url = cpgi_cfg.get("url", "")
    cpgi_local = Path(cpgi_cfg.get("local_path", out / "cpgIslandExt.txt.gz"))

    if not skip_download and cpgi_url:
        download_if_missing(cpgi_url, cpgi_local, force=force)
    if cpgi_local.exists():
        parse_cpg_islands(cpgi_local, out / "cpg_islands.parquet")
        click.echo(f"CpG islands -> {out / 'cpg_islands.parquet'}")
    else:
        click.echo("UCSC CpG islands file not found; skipping.", err=True)

    click.echo("Annotation parsing complete.")


if __name__ == "__main__":
    main()
