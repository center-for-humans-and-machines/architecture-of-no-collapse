"""The single plug-in interface for perturbing the loop.

An Intervention takes the current message history and returns a new Message to
append, or None for a no-op. Its internal logic is irrelevant to the loop. A
returned Message may set ``transient=True`` to be prompt-scoped only (fed to the
next agent's prompt and dropped, never stored in canonical history).

An intervention may additionally define ``prime`` — an optional opening hook the
loop calls once before the first agent turn, with no arguments. It returns a
Message to seed the conversation (or None), following the same ``transient``
semantics as ``act``. Implementations without ``prime`` are skipped by the loop.
"""

from __future__ import annotations

from typing import Protocol

from renewal.core.message import Message


class Intervention(Protocol):
    name: str

    async def act(self, messages: list[Message]) -> Message | None:
        """Return a Message to append, or None to do nothing."""
        ...

    async def prime(self) -> Message | None:
        """Return an opening Message to seed the conversation, or None."""
        ...
