"""The single plug-in interface for perturbing the loop.

An Intervention takes the current message history and returns a new Message to
append, or None for a no-op. Its internal logic is irrelevant to the loop. A
returned Message may set ``transient=True`` to be prompt-scoped only (fed to the
next agent's prompt and dropped, never stored in canonical history).
"""

from __future__ import annotations

from typing import Protocol

from renewal.core.message import Message


class Intervention(Protocol):
    name: str

    async def act(self, messages: list[Message]) -> Message | None:
        """Return a Message to append, or None to do nothing."""
        ...
