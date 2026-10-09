"""Console and file logging configuration.

A minimal setup: one console handler plus an optional per-run file handler.
Secrets are never logged — adapters only log credential-free URLs and presence
flags.

Concurrent replicates each attach their own file handler to the shared root
logger. A :class:`contextvars.ContextVar` records which run the current task
belongs to, and a handler filter keeps each task's records out of its siblings'
logs.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

PROGRESS = 15
logging.addLevelName(PROGRESS, "PROGRESS")

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Identifies the run whose logs a scoped handler should capture. A ContextVar
# (rather than a global) so concurrent tasks each see their own value.
_RUN_SCOPE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "renewal_run_scope", default=None
)


@contextmanager
def run_log_scope(key: str) -> Iterator[None]:
    """Bind ``key`` to the current task so run-scoped handlers capture it."""
    token = _RUN_SCOPE.set(key)
    try:
        yield
    finally:
        _RUN_SCOPE.reset(token)


class _ScopedFilter(logging.Filter):
    """Pass records from the matching run scope, or from no scope at all.

    Records emitted outside any scope (e.g. when a caller runs a single
    replicate directly) are allowed through so unscoped usage keeps working.
    """

    def __init__(self, key: str) -> None:
        super().__init__()
        self._key = key

    def filter(self, record: logging.LogRecord) -> bool:
        current = _RUN_SCOPE.get()
        return current is None or current == self._key


def run_scope_filter(key: str) -> logging.Filter:
    """Return a handler filter that captures only the run identified by ``key``."""
    return _ScopedFilter(key)


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


def attach_file_log(
    path: Path,
    level: str = "DEBUG",
    *,
    run_key: str | None = None,
) -> logging.Handler:
    """Add a file handler and return it (caller detaches after the run).

    With ``run_key`` set, only records logged inside the matching
    :func:`run_log_scope` are written, so concurrent runs keep separate logs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(_level(level))
    handler.setFormatter(logging.Formatter(_FORMAT))
    if run_key is not None:
        handler.addFilter(run_scope_filter(run_key))
    logging.getLogger().addHandler(handler)
    return handler


def detach_file_log(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()
