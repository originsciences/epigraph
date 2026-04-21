"""Structured logging configuration for epigraph.

Uses :mod:`structlog` with JSON output for production and a human-readable
Rich console renderer for development.  The mode is selected automatically
from the ``EPIGRAPH_ENV`` environment variable (``"dev"`` or ``"prod"``),
defaulting to ``"dev"``.

Usage::

    from epigraph.common.logging import get_logger

    logger = get_logger(__name__)
    logger.info("processing_started", n_samples=n)
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def _is_dev_mode() -> bool:
    """Return ``True`` when running in development mode."""
    return os.getenv("EPIGRAPH_ENV", "dev").lower() in ("dev", "development")


def _configure_structlog() -> None:
    """Idempotently configure structlog processors and output."""
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: structlog.types.Processor
    if _is_dev_mode():
        # Pretty console output via Rich (declared as a core dependency).
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Also configure the stdlib root logger so that structlog messages
    # emitted through the stdlib bridge are formatted consistently.
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
        root.setLevel(logging.DEBUG if _is_dev_mode() else logging.INFO)


# Run configuration once at import time.
_configure_structlog()


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog logger bound to *name*.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A :class:`structlog.stdlib.BoundLogger` instance.
    """
    return structlog.get_logger(name)
