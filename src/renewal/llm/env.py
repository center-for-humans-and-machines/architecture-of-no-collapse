"""Resolve adapter settings from arguments or the environment.

Adapters take connection settings as optional constructor arguments and fall
back to provider-specific environment variables. Precedence is fixed: an
explicit non-None argument wins, then the environment variable, then the
caller's default. A repository ``.env`` file is loaded once at import.

Secrets are never logged — only presence (set/unset) and a credential-free
URL (``scheme://host:port``).
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

from dotenv import load_dotenv

# Load once at import so every adapter, CLI entry point, and test sees the
# same repository ``.env`` regardless of construction order.
load_dotenv()

LOGGER = logging.getLogger(__name__)


def setting(
    explicit: str | None,
    environment: str | None,
    default: str | None = None,
) -> str | None:
    """Resolve and trim one explicit or environment setting.

    Args:
        explicit: Value passed by the caller. Any non-None value wins,
            including a blank string, which the caller should then reject.
        environment: Variable consulted when ``explicit`` is None. Pass None
            to skip the environment lookup.
        default: Value used when the variable is unset.

    Returns:
        The resolved value with surrounding whitespace removed, or None when
        nothing was found. Whitespace-only values collapse to None so a stray
        space in a ``.env`` fails the adapter's own validation.
    """
    value = explicit
    source = "argument"
    if value is None:
        if environment is None:
            value = default
            source = "default"
        else:
            value = os.getenv(environment, default)
            source = "environment" if environment in os.environ else "default"
    resolved = value.strip() if value else None
    LOGGER.debug(
        "setting %s resolved from %s: %s",
        environment,
        source,
        "present" if resolved else "missing",
    )
    return resolved


def safe_url(value: str | None) -> str:
    """Describe a URL for logs without credentials, path, or query."""
    if not value:
        return "unset"
    parts = urlsplit(value)
    if not parts.hostname:
        return "set"
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname}{port}"
