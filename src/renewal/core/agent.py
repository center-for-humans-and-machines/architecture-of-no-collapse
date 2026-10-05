"""Agent — a thin bundle of model, prompt, and params; it does not loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from renewal.core.message import Message
from renewal.llm.base import LLM


@dataclass
class Agent:
    name: str
    llm: LLM
    system_prompt: str
    params: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def _system_message(self) -> Message:
        return Message(
            role="system",
            speaker="system",
            content=self.system_prompt,
            turn_index=-1,
        )

    async def respond(
        self,
        history: list[Message],
        *,
        turn_index: int,
        round_index: int,
        position: int,
    ) -> Message:
        """Generate one turn from the full shared history."""
        prompt = [self._system_message(), *history]
        content = await self.llm.generate(prompt, self.params)
        return Message(
            role="assistant",
            speaker=self.name,
            content=content,
            turn_index=turn_index,
            meta={
                "round": round_index,
                "position": position,
                **self.meta,
            },
        )
