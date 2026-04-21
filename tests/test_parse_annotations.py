"""Tests for annotation parsers in epigraph.db_build.parse_annotations."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest

from epigraph.db_build.parse_annotations import (
    _parse_gtf_attributes,
    download_if_missing,
    parse_cpg_islands,
    parse_gencode_gtf,
    parse_goa_gaf,
    parse_reactome,
)

# ---------------------------------------------------------------------------
# _parse_gtf_attributes
# ---------------------------------------------------------------------------


class TestParseGtfAttributes:
    """Unit tests for the GTF attribute string parser."""

    def test_standard_attribute_string(self) -> None:
        attr = 'gene_id "ENSG00000223972.5"; gene_name "DDX11L2"; gene_type "transcribed_unprocessed_pseudogene";'
        result = _parse_gtf_attributes(attr)
        assert result["gene_id"] == "ENSG00000223972.5"
        assert result["gene_name"] == "DDX11L2"
        assert result["gene_type"] == "transcribed_unprocessed_pseudogene"

    def test_empty_string(self) -> None:
        assert _parse_gtf_attributes("") == {}

    def test_single_attribute(self) -> None:
        result = _parse_gtf_attributes('gene_id "ENSG00000000001";')
        assert result == {"gene_id": "ENSG00000000001"}

    def test_whitespace_handling(self) -> None:
        attr = '  gene_id "A";  gene_name "B"  ;  '
        result = _parse_gtf_attributes(attr)
        assert result["gene_id"] == "A"
        assert result["gene_name"] == "B"


# ---------------------------------------------------------------------------
# parse_gencode_gtf
# ---------------------------------------------------------------------------


class TestParseGencodeGtf:
    """Tests for GENCODE GTF parsing."""

    @pytest.fixture()
    def tmp_gtf(self, tmp_path: Path) -> Path:
        """Create a small GTF file with gene and transcript records."""
        gtf_path = tmp_path / "test.gtf"
        lines = [
            "##description: test GTF",
            '#!genome-build GRCh38',
            'chr1\tHAVANA\tgene\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972.5"; gene_name "DDX11L2"; gene_type "transcribed_unprocessed_pseudogene";',
            'chr1\tHAVANA\ttranscript\t11869\t14409\t.\t+\t.\tgene_id "ENSG00000223972.5"; transcript_id "ENST00000456328.2";',
            'chr2\tENSEMBL\tgene\t38814\t46588\t.\t-\t.\tgene_id "ENSG00000227232.10"; gene_name "WASH7P"; gene_type "unprocessed_pseudogene";',
            'chrX\tHAVANA\tgene\t100\t5000\t.\t+\t.\tgene_id "ENSG00000000003.15"; gene_name "TSPAN6"; gene_type "protein_coding";',
        ]
        gtf_path.write_text("\n".join(lines) + "\n")
        return gtf_path

    def test_parse_produces_correct_columns(self, tmp_gtf: Path, tmp_path: Path) -> None:
        output = tmp_path / "genes.parquet"
        df = parse_gencode_gtf(tmp_gtf, output)
        expected_cols = {"gene_id", "gene_symbol", "chrom", "start", "end", "strand", "biotype"}
        assert set(df.columns) == expected_cols

    def test_parse_filters_to_gene_feature(self, tmp_gtf: Path, tmp_path: Path) -> None:
        output = tmp_path / "genes.parquet"
        df = parse_gencode_gtf(tmp_gtf, output)
        # The GTF has 3 gene lines and 1 transcript line; only genes are kept
        assert df.height == 3

    def test_gene_id_version_stripped(self, tmp_gtf: Path, tmp_path: Path) -> None:
        output = tmp_path / "genes.parquet"
        df = parse_gencode_gtf(tmp_gtf, output)
        gene_ids = df["gene_id"].to_list()
        assert "ENSG00000223972" in gene_ids
        # Versioned ID should not appear
        assert not any("." in gid for gid in gene_ids)

    def test_coordinates_and_strand(self, tmp_gtf: Path, tmp_path: Path) -> None:
        output = tmp_path / "genes.parquet"
        df = parse_gencode_gtf(tmp_gtf, output)
        wash7p = df.filter(pl.col("gene_symbol") == "WASH7P")
        assert wash7p.height == 1
        assert wash7p["start"].item() == 38814
        assert wash7p["end"].item() == 46588
        assert wash7p["strand"].item() == "-"

    def test_output_parquet_written(self, tmp_gtf: Path, tmp_path: Path) -> None:
        output = tmp_path / "genes.parquet"
        parse_gencode_gtf(tmp_gtf, output)
        assert output.exists()
        reloaded = pl.read_parquet(output)
        assert reloaded.height == 3

    def test_gzipped_gtf(self, tmp_path: Path) -> None:
        """Verify that gzipped GTF files are parsed correctly."""
        gz_path = tmp_path / "test.gtf.gz"
        line = 'chr1\tHAVANA\tgene\t100\t500\t.\t+\t.\tgene_id "ENSG00000000001.1"; gene_name "TESTG"; gene_type "protein_coding";'
        with gzip.open(gz_path, "wt") as fh:
            fh.write(line + "\n")
        output = tmp_path / "genes_gz.parquet"
        df = parse_gencode_gtf(gz_path, output)
        assert df.height == 1
        assert df["gene_id"].item() == "ENSG00000000001"


# ---------------------------------------------------------------------------
# parse_goa_gaf
# ---------------------------------------------------------------------------


class TestParseGoaGaf:
    """Tests for GO Annotation (GAF) parsing."""

    @pytest.fixture()
    def tmp_gaf(self, tmp_path: Path) -> Path:
        """Create a small GAF 2.2 file with human and non-human entries."""
        gaf_path = tmp_path / "test.gaf"
        # GAF columns (15 minimum): DB, DB_Object_ID, DB_Object_Symbol,
        # Qualifier, GO_ID, DB:Reference, Evidence_Code, With/From,
        # Aspect, DB_Object_Name, DB_Object_Synonym, DB_Object_Type,
        # Taxon, Date, Assigned_By
        human_line_1 = "\t".join([
            "UniProtKB", "A0A024R216", "TP53",
            "enables", "GO:0003677", "PMID:12345", "IDA", "",
            "F", "Tumor protein p53", "TP53", "protein",
            "taxon:9606", "20230101", "UniProt",
        ])
        human_line_2 = "\t".join([
            "UniProtKB", "P04637", "TP53",
            "involved_in", "GO:0006915", "PMID:12346", "IMP", "",
            "P", "Tumor protein p53", "TP53", "protein",
            "taxon:9606", "20230102", "UniProt",
        ])
        mouse_line = "\t".join([
            "UniProtKB", "Q00987", "Mdm2",
            "enables", "GO:0005515", "PMID:99999", "IPI", "",
            "F", "MDM2 protein", "Mdm2", "protein",
            "taxon:10090", "20230103", "UniProt",
        ])
        lines = [
            "!gaf-version: 2.2",
            "!date: 2023-01-01",
            human_line_1,
            mouse_line,
            human_line_2,
        ]
        gaf_path.write_text("\n".join(lines) + "\n")
        return gaf_path

    def test_filters_to_human_only(self, tmp_gaf: Path, tmp_path: Path) -> None:
        output = tmp_path / "go_annotations.parquet"
        df = parse_goa_gaf(tmp_gaf, output)
        # Only the 2 human lines should be kept
        assert df.height == 2

    def test_correct_columns(self, tmp_gaf: Path, tmp_path: Path) -> None:
        output = tmp_path / "go_annotations.parquet"
        df = parse_goa_gaf(tmp_gaf, output)
        expected = {"gene_symbol", "go_id", "evidence_code", "aspect", "qualifier"}
        assert set(df.columns) == expected

    def test_aspect_mapped(self, tmp_gaf: Path, tmp_path: Path) -> None:
        output = tmp_path / "go_annotations.parquet"
        df = parse_goa_gaf(tmp_gaf, output)
        aspects = set(df["aspect"].to_list())
        assert "molecular_function" in aspects
        assert "biological_process" in aspects

    def test_gene_symbol_extracted(self, tmp_gaf: Path, tmp_path: Path) -> None:
        output = tmp_path / "go_annotations.parquet"
        df = parse_goa_gaf(tmp_gaf, output)
        assert all(s == "TP53" for s in df["gene_symbol"].to_list())

    def test_output_parquet_written(self, tmp_gaf: Path, tmp_path: Path) -> None:
        output = tmp_path / "go_annotations.parquet"
        parse_goa_gaf(tmp_gaf, output)
        assert output.exists()


# ---------------------------------------------------------------------------
# parse_reactome
# ---------------------------------------------------------------------------


class TestParseReactome:
    """Tests for Reactome pathway and gene-pathway parsing."""

    @pytest.fixture()
    def tmp_reactome_files(self, tmp_path: Path) -> tuple[Path, Path]:
        """Create temp Reactome pathway and gene-pathway mapping files."""
        pw_path = tmp_path / "ReactomePathways.txt"
        gp_path = tmp_path / "Ensembl2Reactome.txt"

        pw_lines = [
            "R-HSA-109582\tHemostasis\tHomo sapiens",
            "R-HSA-1640170\tCell Cycle\tHomo sapiens",
            "R-MMU-109582\tHemostasis\tMus musculus",
        ]
        pw_path.write_text("\n".join(pw_lines) + "\n")

        gp_lines = [
            "ENSG00000000003.15\tR-HSA-109582\thttps://reactome.org\tHemostasis\tIEA\tHomo sapiens",
            "ENSG00000000005.6\tR-HSA-1640170\thttps://reactome.org\tCell Cycle\tTAS\tHomo sapiens",
            "ENSMUSG00000000001\tR-MMU-109582\thttps://reactome.org\tHemostasis\tIEA\tMus musculus",
        ]
        gp_path.write_text("\n".join(gp_lines) + "\n")

        return pw_path, gp_path

    def test_pathways_human_only(
        self, tmp_reactome_files: tuple[Path, Path], tmp_path: Path
    ) -> None:
        pw_path, gp_path = tmp_reactome_files
        pw_df, _ = parse_reactome(
            pw_path, gp_path,
            tmp_path / "pathways.parquet",
            tmp_path / "gene_map.parquet",
        )
        assert pw_df.height == 2  # only Homo sapiens

    def test_gene_map_human_only(
        self, tmp_reactome_files: tuple[Path, Path], tmp_path: Path
    ) -> None:
        pw_path, gp_path = tmp_reactome_files
        _, gp_df = parse_reactome(
            pw_path, gp_path,
            tmp_path / "pathways.parquet",
            tmp_path / "gene_map.parquet",
        )
        assert gp_df.height == 2  # only Homo sapiens

    def test_gene_id_version_stripped(
        self, tmp_reactome_files: tuple[Path, Path], tmp_path: Path
    ) -> None:
        pw_path, gp_path = tmp_reactome_files
        _, gp_df = parse_reactome(
            pw_path, gp_path,
            tmp_path / "pathways.parquet",
            tmp_path / "gene_map.parquet",
        )
        gene_ids = gp_df["gene_id"].to_list()
        assert "ENSG00000000003" in gene_ids
        assert not any("." in gid for gid in gene_ids)

    def test_pathway_columns(
        self, tmp_reactome_files: tuple[Path, Path], tmp_path: Path
    ) -> None:
        pw_path, gp_path = tmp_reactome_files
        pw_df, gp_df = parse_reactome(
            pw_path, gp_path,
            tmp_path / "pathways.parquet",
            tmp_path / "gene_map.parquet",
        )
        assert set(pw_df.columns) == {"pathway_id", "pathway_name", "pathway_source"}
        assert set(gp_df.columns) == {"gene_id", "pathway_id", "evidence_code"}

    def test_output_parquets_written(
        self, tmp_reactome_files: tuple[Path, Path], tmp_path: Path
    ) -> None:
        pw_path, gp_path = tmp_reactome_files
        out_pw = tmp_path / "pathways.parquet"
        out_gp = tmp_path / "gene_map.parquet"
        parse_reactome(pw_path, gp_path, out_pw, out_gp)
        assert out_pw.exists()
        assert out_gp.exists()


# ---------------------------------------------------------------------------
# parse_cpg_islands
# ---------------------------------------------------------------------------


class TestParseCpgIslands:
    """Tests for UCSC CpG island parsing."""

    @pytest.fixture()
    def tmp_cpg_island_file(self, tmp_path: Path) -> Path:
        """Create a temp cpgIslandExt file with 3 records."""
        cpgi_path = tmp_path / "cpgIslandExt.txt"
        # Columns: bin, chrom, chromStart, chromEnd, name, length, cpgNum,
        # gcNum, perCpg, perGc, obsExp
        lines = [
            "585\tchr1\t28735\t29810\tCpG:_111\t1075\t111\t731\t20.7\t68.0\t0.83",
            "585\tchr1\t135124\t135563\tCpG:_30\t439\t30\t295\t13.7\t67.2\t0.64",
            "586\tchr2\t321085\t321363\tCpG:_29\t278\t29\t199\t20.9\t71.6\t0.94",
        ]
        cpgi_path.write_text("\n".join(lines) + "\n")
        return cpgi_path

    def test_correct_row_count(self, tmp_cpg_island_file: Path, tmp_path: Path) -> None:
        output = tmp_path / "cpg_islands.parquet"
        df = parse_cpg_islands(tmp_cpg_island_file, output)
        assert df.height == 3

    def test_correct_columns(self, tmp_cpg_island_file: Path, tmp_path: Path) -> None:
        output = tmp_path / "cpg_islands.parquet"
        df = parse_cpg_islands(tmp_cpg_island_file, output)
        expected = {"region_id", "chrom", "start", "end", "cpg_count", "gc_fraction", "obs_exp_ratio"}
        assert set(df.columns) == expected

    def test_coordinate_values(self, tmp_cpg_island_file: Path, tmp_path: Path) -> None:
        output = tmp_path / "cpg_islands.parquet"
        df = parse_cpg_islands(tmp_cpg_island_file, output)
        first = df.row(0, named=True)
        assert first["chrom"] == "chr1"
        assert first["start"] == 28735
        assert first["end"] == 29810
        assert first["cpg_count"] == 111

    def test_gc_fraction_normalized(self, tmp_cpg_island_file: Path, tmp_path: Path) -> None:
        output = tmp_path / "cpg_islands.parquet"
        df = parse_cpg_islands(tmp_cpg_island_file, output)
        # perGc=68.0 should become 0.68
        first = df.row(0, named=True)
        assert abs(first["gc_fraction"] - 0.68) < 1e-6

    def test_comments_skipped(self, tmp_path: Path) -> None:
        cpgi_path = tmp_path / "cpg_with_comments.txt"
        lines = [
            "# this is a comment",
            "585\tchr1\t100\t200\tCpG:_10\t100\t10\t70\t20.0\t70.0\t0.90",
        ]
        cpgi_path.write_text("\n".join(lines) + "\n")
        output = tmp_path / "cpg_islands.parquet"
        df = parse_cpg_islands(cpgi_path, output)
        assert df.height == 1

    def test_output_parquet_written(self, tmp_cpg_island_file: Path, tmp_path: Path) -> None:
        output = tmp_path / "cpg_islands.parquet"
        parse_cpg_islands(tmp_cpg_island_file, output)
        assert output.exists()


# ---------------------------------------------------------------------------
# download_if_missing
# ---------------------------------------------------------------------------


class TestDownloadIfMissing:
    """Tests for the download helper's skip-existing logic."""

    def test_skips_existing_file(self, tmp_path: Path) -> None:
        """Existing files should not trigger a download."""
        existing = tmp_path / "already_here.txt"
        existing.write_text("content")

        result = download_if_missing("http://example.com/file", existing, force=False)
        assert result == existing
        # Content unchanged (no download happened)
        assert existing.read_text() == "content"

    def test_force_redownloads(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With force=True, even existing files should be re-downloaded."""
        existing = tmp_path / "already_here.txt"
        existing.write_text("old content")

        # Mock urlretrieve to write new content
        def fake_urlretrieve(url: str, filename: str) -> tuple[str, None]:
            Path(filename).write_text("new content")
            return filename, None

        monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
        result = download_if_missing("http://example.com/file", existing, force=True)
        assert result == existing
        assert existing.read_text() == "new content"
