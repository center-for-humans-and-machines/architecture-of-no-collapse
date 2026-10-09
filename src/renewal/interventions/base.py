"""The single plug-in interface for perturbing the loop.

An Intervention takes the current message history and returns a new Message to
append, or None for a no-op. Its internal logic is irrelevant to the loop. A
returned Message may set ``transient=True`` to be prompt-scoped only (fed to the
next agent's prompt and dropped, never stored in canonical history).

An intervention may instead return a :class:`HistoryReplacement` when it wants
to *compress* rather than append: the loop drops the canonical history from a
given turn on and appends the replacement messages in its place. This is how a
reflective scaffolder summarizes everything since the last topic into a single
recap, freeing the model's context without losing the recap itself.

An intervention may additionally define ``prime`` — an optional opening hook the
loop calls once before the first agent turn, with no arguments. It returns a
Message to seed the conversation (or None), following the same ``transient``
semantics as ``act``. Implementations without ``prime`` are skipped by the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from renewal.core.message import Message


@dataclass
class HistoryReplacement:
    """Ask the loop to replace a suffix of canonical history with messages.

    ``since`` is inclusive: every canonical history message whose ``turn_index``
    is ``>= since`` is dropped, then ``messages`` are appended in order. The
    dropped messages are gone from the loop's in-memory history (the model's
    context) but remain in the append-only transcript.
    """

    since: int
    messages: list[Message] = field(default_factory=list)


class Intervention(Protocol):
    name: str

    async def act(self, messages: list[Message]) -> Message | HistoryReplacement | None:
        """Return a Message to append, a HistoryReplacement, or None."""
        ...

    async def prime(self) -> Message | None:
        """Return an opening Message to seed the conversation, or None."""
        ...
