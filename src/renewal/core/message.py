"""The single data type that flows through the loop.

``Message`` is the only record of a turn. Roles are ``system`` (per-agent,
prompt-scoped only), ``user`` / ``assistant`` (agent turns), and
``intervention`` (messages appended by an Intervention plugin). There is no
transient flag: every message an intervention appends becomes canonical
history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "intervention"]


@dataclass
class Message:
    role: Role
    speaker: str
    content: str
    turn_index: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "speaker": self.speaker,
            "content": self.content,
            "turn_index": self.turn_index,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            speaker=data["speaker"],
            content=data["content"],
            turn_index=data["turn_index"],
            meta=dict(data.get("meta", {})),
        )
