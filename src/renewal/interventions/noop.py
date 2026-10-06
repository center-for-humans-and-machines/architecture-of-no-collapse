"""A no-op intervention, used as the default and for tests."""

from __future__ import annotations

from renewal.core.message import Message
from renewal.registry import Registry


@Registry.register("intervention", "noop")
class NoopIntervention:
    name = "noop"

    async def act(self, messages: list[Message]) -> Message | None:
        return None

    async def prime(self) -> Message | None:
        return None
