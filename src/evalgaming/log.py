"""Structured logging for the whole project, built on structlog.

Call :func:`configure` once at the start of an experiment script, then use
:func:`get_logger` everywhere. Console output is rendered by structlog's dev renderer,
which uses rich for colors and tracebacks; ``json_logs=True`` emits one JSON object per
line for runs on remote GPU machines.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure(level: str = "INFO", *, json_logs: bool = False) -> None:
    """Configure structlog for the current process."""
    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a logger bound to ``name``; use ``__name__`` from library modules."""
    return structlog.get_logger(name)
