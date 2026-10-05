"""Deterministic offline adapter for tests and smoke runs.

Ignores the model and generation params entirely and returns a short string
that encodes the prompt length, so transcripts are deterministic and each turn
is distinct without any network or credentials.
"""

from __future__ import annotations

from typing import Any

from renewal.core.message import Message
from renewal.llm.base import BaseLLM


class FakeLLM(BaseLLM):
    def __init__(
        self,
        model: str | None = None,
        retry_delays: tuple[float, ...] = (0.5, 1.0),
    ) -> None:
        self.model = model or "fake"
        super().__init__(retry_delays=retry_delays)

    async def generate(self, messages: list[Message], params: dict[str, Any]) -> str:
        return f"fake reply #{len(messages)}"
