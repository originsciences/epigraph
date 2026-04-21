# epigraph -- reproducible test environment.
#
# Builds a slim Python 3.12 image with the package installed in editable
# mode along with the [dev] extras, so `docker run epigraph` executes
# the test suite out of the box.
#
# Usage:
#   docker build -t epigraph .
#   docker run --rm epigraph                 # runs the default test suite
#   docker run --rm epigraph pytest -q       # override the command
#   docker run --rm -it epigraph bash        # interactive shell

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# System deps needed by scientific-Python wheels at runtime (libgomp for
# scipy/numpy, libstdc++ for pyarrow/duckdb). Kept minimal so the image
# stays small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so changes under src/ don't invalidate the
# (expensive) dependency layer.
COPY pyproject.toml README.md ./
COPY src/ ./src/
# [dev] = test + lint + typecheck tooling; [report] = python-docx, needed
# by ``epigraph.analysis.generate_report`` (which the test suite exercises).
RUN pip install -e ".[dev,report]"

# Source of truth for the rest of the project.
COPY tests/ ./tests/
COPY config/ ./config/
COPY schemas/ ./schemas/

# Non-root runtime user.
RUN useradd --create-home --shell /bin/bash epigraph \
    && chown -R epigraph:epigraph /app
USER epigraph

# Default: run the full test suite. Override with `docker run ... <cmd>`.
CMD ["pytest", "tests/", "-q"]
