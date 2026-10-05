"""Process-wide catalog of named implementations by (kind, name).

Plugins (interventions now, metrics in Step 2) opt in at import time with
``@Registry.register("intervention", "noop")``. A config file names a plugin by
its registered name; ``Registry.get`` resolves it back to the implementation.

Re-registering the identical target is idempotent (module reimports are
harmless); a second, different target under a taken name is an error so a YAML
name can never be ambiguous.
"""

from __future__ import annotations

from typing import Any


class Registry:
    """Class-level catalog keyed by ``(kind, name)``."""

    _entries: dict[tuple[str, str], Any] = {}

    @classmethod
    def register(cls, kind: str, name: str) -> Any:
        """Return a decorator that records a target under kind/name."""
        if not kind or not name:
            raise ValueError("registry kind and name must be non-empty")

        def decorator(target: Any) -> Any:
            key = (kind, name)
            existing = cls._entries.get(key)
            if existing is not None and existing is not target:
                raise ValueError(f"{kind} {name!r} is already registered")
            cls._entries[key] = target
            return target

        return decorator

    @classmethod
    def get(cls, kind: str, name: str) -> Any:
        """Return the target registered for kind/name, or raise KeyError."""
        try:
            return cls._entries[(kind, name)]
        except KeyError:
            known = sorted(n for (k, n) in cls._entries if k == kind)
            raise KeyError(f"no {kind} named {name!r}; known: {known}") from None

    @classmethod
    def names(cls, kind: str) -> list[str]:
        """Return the registered names for a kind, sorted."""
        return sorted(n for (k, n) in cls._entries if k == kind)
