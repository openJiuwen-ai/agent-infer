"""
Structured logging configuration for vllm-evolve.

Wraps structlog with:
- JSON output to file (machine-readable, one JSON object per line)
- Coloured, human-friendly output to stderr (via ``structlog.dev.ConsoleRenderer``)
- Consistent timestamp, log-level, and logger-name fields across both sinks

Usage::

    from vllm_evolve.utils.logging import setup_logging

    setup_logging(level="DEBUG", log_file=Path("logs/run.jsonl"))
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------

_configured = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure structlog and the stdlib root logger.

    Calling this function more than once is safe; subsequent calls update the
    configuration in-place.

    Args:
        level:    Log level string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
                  ``"ERROR"``, ``"CRITICAL"``).  Case-insensitive.
        log_file: Optional path to a file that will receive JSON-formatted
                  log records (one JSON object per line, JSONL).  The parent
                  directory is created if it does not exist.  If ``None``,
                  only console output is produced.
    """
    global _configured

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # ------------------------------------------------------------------
    # Build shared pre-processors (run before the final renderer)
    # ------------------------------------------------------------------
    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
    handlers: list[logging.Handler] = []

    # Console handler – coloured, human-readable
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    handlers.append(console_handler)

    # File handler – JSON, machine-readable
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)

    # ------------------------------------------------------------------
    # Configure stdlib root logger
    # ------------------------------------------------------------------
    root_logger = logging.getLogger()
    # Remove existing handlers to avoid duplicates on re-configuration
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    for handler in handlers:
        root_logger.addHandler(handler)

    # ------------------------------------------------------------------
    # Configure structlog
    # ------------------------------------------------------------------
    # We use ProcessorFormatter so that stdlib loggers (e.g. from third-party
    # libraries) also get structured output.

    # Console formatter: coloured, human-readable
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=_stderr_supports_color()),
        ],
        foreign_pre_chain=shared_processors,
    )
    handlers[0].setFormatter(console_formatter)

    # File formatter: JSON
    if log_file is not None and len(handlers) > 1:
        json_formatter = structlog.stdlib.ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        )
        handlers[1].setFormatter(json_formatter)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    _configured = True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _stderr_supports_color() -> bool:
    """Return True if stderr appears to be a TTY that supports ANSI colours."""
    try:
        return sys.stderr.isatty()
    except AttributeError:
        return False
