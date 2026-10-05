"""PromptSource — load a prompt list and assign one system prompt per agent.

The global prompt file is a YAML list of ``{id, text}`` entries. Assignment is
layered (later wins): the seeded default selection, then ``prompts_override``
entries keyed by agent index (``"0"``) or name (``"agent_0"``), and finally the
code-level ``system_prompt`` fallback when no file is configured or no id
resolves.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import yaml

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prompt:
    id: str
    text: str


def load_prompt_file(path: Path | None) -> list[Prompt]:
    """Load and validate a prompt file, or return [] when no path is given."""
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or []
    if not isinstance(raw, list):
        raise ValueError(f"prompt file must be a list: {path}")
    prompts: list[Prompt] = []
    seen: set[str] = set()
    for entry in raw:
        if not isinstance(entry, dict) or "id" not in entry or "text" not in entry:
            raise ValueError(f"each prompt entry needs 'id' and 'text': {path}")
        pid = entry["id"]
        if not isinstance(pid, str) or not pid:
            raise ValueError(f"prompt id must be a non-empty string: {path}")
        if pid in seen:
            raise ValueError(f"duplicate prompt id {pid!r}: {path}")
        if not isinstance(entry["text"], str) or not entry["text"].strip():
            raise ValueError(f"prompt {pid!r} text must be non-empty: {path}")
        seen.add(pid)
        prompts.append(Prompt(id=pid, text=entry["text"].strip()))
    return prompts


class PromptSource:
    """Resolve one system prompt per agent from a file plus overrides."""

    def __init__(self, prompts: list[Prompt], default_text: str) -> None:
        self._by_id = {p.id: p.text for p in prompts}
        self._ids = [p.id for p in prompts]
        self.default_text = default_text

    @classmethod
    def load(cls, path: Path | None, default_text: str) -> "PromptSource":
        return cls(load_prompt_file(path), default_text)

    def assign(
        self,
        n: int,
        seed: int,
        *,
        overrides: dict[str, str] | None = None,
    ) -> list[str]:
        """Return n system-prompt strings.

        Seeded default selection shuffles the file's ids (cycling when there
        are fewer prompts than agents), then ``overrides`` (keyed by index
        string or agent name) replace individual entries. Unknown ids raise.
        """
        overrides = overrides or {}
        if self._ids:
            shuffled = list(self._ids)
            random.Random(seed).shuffle(shuffled)
            ids = [shuffled[i % len(shuffled)] for i in range(n)]
        else:
            ids = [None] * n
        for key, pid in overrides.items():
            ids[self._resolve_index(key, n)] = pid
        return [self._text(pid) for pid in ids]

    def _text(self, pid: str | None) -> str:
        if pid is None:
            return self.default_text
        if pid in self._by_id:
            return self._by_id[pid]
        raise ValueError(f"unknown prompt id {pid!r}")

    @staticmethod
    def _resolve_index(key: str, n: int) -> int:
        if key.isdigit():
            index = int(key)
            if index < 0 or index >= n:
                raise ValueError(f"prompt override index {key!r} out of range (n={n})")
            return index
        if key.startswith("agent_"):
            suffix = key[len("agent_") :]
            if suffix.isdigit():
                index = int(suffix)
                if 0 <= index < n:
                    return index
        raise ValueError(f"unknown agent key in prompts_override: {key!r}")
