"""Console and file logging configuration.

A minimal setup: one console handler plus an optional per-run file handler.
Secrets are never logged — adapters only log credential-free URLs and presence
flags.
"""

from __future__ import annotations

import logging
from pathlib import Path

PROGRESS = 15
logging.addLevelName(PROGRESS, "PROGRESS")

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def _level(value: str | int) -> int:
    if isinstance(value, int):
        return value
    return logging.getLevelName(value.upper())


def configure_logging(console_level: str = "INFO") -> None:
    """Install the console handler; clears any existing handlers."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(_level(console_level))
    console.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(console)
    # Quiet noisy third-party loggers.
    for name in ("httpx", "openai", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def attach_file_log(path: Path, level: str = "DEBUG") -> logging.Handler:
    """Add a file handler and return it (caller detaches after the run)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(_level(level))
    handler.setFormatter(logging.Formatter(_FORMAT))
    logging.getLogger().addHandler(handler)
    return handler


def detach_file_log(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()
