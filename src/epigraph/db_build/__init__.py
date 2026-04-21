"""Database build pipeline for the methylation knowledge graph.

This sub-package contains CLI commands and library functions for:

- Inspecting the upstream beta-matrix generator source code
- Creating representative development subsets
- Loading and normalizing clinical metadata
- Parsing the beta matrix CSV into Parquet
- Downloading and parsing genomic annotations
- Mapping CpG sites to genes via coordinate intersection
- Importing all parsed data into TypeDB 3.x
- Validating the TypeDB import
"""

__all__: list[str] = []
