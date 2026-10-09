"""Memory store and surfacing algorithms for the memory scaffolder.

A :class:`Memory` is a distilled summary of a topic that has just ended. The
intervention layer creates one whenever a new topic is introduced; the
RESURFACE action brings an earlier memory back into the prompt. Each memory
keeps the fields a future selection algorithm needs:

* a sequential ``id``,
* the summary ``text``,
* a unit-norm ``embedding`` of that text,
* how often it was surfaced (``surfaced_count``) and when (``last_surfaced_turn``),
* the turn range it was distilled from.

Selection is delegated to a :class:`MemorySelector` so the surfacing algorithm
can evolve without touching the store or the intervention. The shipped
:class:`RandomUnsurfacedSelector` surfaces, uniformly at random, a memory that
has never been surfaced; once every memory has been seen at least once it falls
back to the least-surfaced ones so the action never becomes permanently dead.

The store optionally reports ``memory_created`` / ``memory_surfaced`` events to
a :class:`MemorySink` (the run recorder) so the memory set can be inspected
offline.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from renewal.llm.embed import EmbeddingClient

LOGGER = logging.getLogger(__name__)


class MemorySink(Protocol):
    """Receive append-only memory events for offline storage."""

    def record_memory(self, event: str, **fields: Any) -> None:
        ...


@dataclass
class Memory:
    """One distilled topic plus the bookkeeping needed to resurface it."""

    id: int
    text: str
    embedding: tuple[float, ...] | None
    created_turn: int
    since_turn: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    surfaced_count: int = 0
    last_surfaced_turn: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Full record, for provenance and the offline ``memories.jsonl``."""
        return {
            "id": self.id,
            "text": self.text,
            "embedding": (
                list(self.embedding) if self.embedding is not None else None
            ),
            "created_turn": self.created_turn,
            "since_turn": self.since_turn,
            "provenance": dict(self.provenance),
            "surfaced_count": self.surfaced_count,
            "last_surfaced_turn": self.last_surfaced_turn,
        }


class MemorySelector(Protocol):
    """Pick which memory to resurface (or ``None`` when there is no memory)."""

    name: str

    def select(
        self,
        memories: Sequence[Memory],
        *,
        rng: random.Random,
        turn_index: int,
        context: str | None = None,
    ) -> Memory | None:
        """Return the memory to resurface.

        ``context`` is the latest turn's text, available to future algorithms
        that want to score memories by embedding similarity to the current
        topic; the random selector ignores it.
        """
        ...


class RandomUnsurfacedSelector:
    """Prefer memories never surfaced; then the least-surfaced ones."""

    name = "random_unsurfaced"

    def select(
        self,
        memories: Sequence[Memory],
        *,
        rng: random.Random,
        turn_index: int,
        context: str | None = None,
    ) -> Memory | None:
        if not memories:
            return None
        unseen = [memory for memory in memories if memory.surfaced_count == 0]
        if unseen:
            return rng.choice(unseen)
        least = min(memory.surfaced_count for memory in memories)
        candidates = [
            memory for memory in memories if memory.surfaced_count == least
        ]
        return rng.choice(candidates)


class MemoryStore:
    """Ordered memory collection with pluggable selection and event reporting."""

    def __init__(
        self,
        embedder: EmbeddingClient,
        *,
        selector: MemorySelector | None = None,
        sink: MemorySink | None = None,
    ) -> None:
        self._embedder = embedder
        self._selector = selector or RandomUnsurfacedSelector()
        self._sink = sink
        self._memories: list[Memory] = []
        self._next_id = 1

    async def add(
        self,
        text: str,
        *,
        created_turn: int,
        since_turn: int | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> Memory:
        """Embed ``text`` (best-effort) and append it as a new memory."""
        embedding: tuple[float, ...] | None = None
        if text.strip():
            try:
                (vector,) = await self._embedder.embed([text])
                embedding = tuple(float(value) for value in vector)
            except Exception as exc:  # noqa: BLE001 - a missing vector must not break
                LOGGER.warning(
                    "memory embedding failed (%s); storing without a vector", exc
                )
        memory = Memory(
            id=self._next_id,
            text=text,
            embedding=embedding,
            created_turn=created_turn,
            since_turn=since_turn,
            provenance=dict(provenance or {}),
        )
        self._next_id += 1
        self._memories.append(memory)
        self._emit_created(memory)
        return memory

    def select(
        self,
        *,
        rng: random.Random,
        turn_index: int,
        context: str | None = None,
    ) -> Memory | None:
        """Run the configured selector over the current memories."""
        return self._selector.select(
            self._memories, rng=rng, turn_index=turn_index, context=context
        )

    def mark_surfaced(self, memory: Memory, turn_index: int) -> None:
        """Record that ``memory`` was surfaced on ``turn_index``."""
        memory.surfaced_count += 1
        memory.last_surfaced_turn = turn_index
        self._emit_surfaced(memory, turn_index)

    @property
    def memories(self) -> list[Memory]:
        """A shallow copy of the memories, oldest first."""
        return list(self._memories)

    def __len__(self) -> int:
        return len(self._memories)

    def _emit_created(self, memory: Memory) -> None:
        if self._sink is None:
            return
        self._sink.record_memory(
            "memory_created", selector=self._selector.name, **memory.as_dict()
        )

    def _emit_surfaced(self, memory: Memory, turn_index: int) -> None:
        if self._sink is None:
            return
        self._sink.record_memory(
            "memory_surfaced",
            id=memory.id,
            turn_index=turn_index,
            surfaced_count=memory.surfaced_count,
            last_surfaced_turn=memory.last_surfaced_turn,
            selector=self._selector.name,
        )
