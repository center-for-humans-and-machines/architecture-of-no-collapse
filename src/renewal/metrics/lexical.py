"""Lexical diversity — cumulative unique unigrams."""

from __future__ import annotations

from renewal.core.text import tokenize
from renewal.registry import Registry


@Registry.register("metric", "lexical_diversity")
class LexicalDiversity:
    name = "lexical_diversity"
    cadence = "per_window"
    needs_embedding = False

    def __init__(self) -> None:
        self._unique: set[str] = set()
        self._total = 0

    async def compute(self, view, index):
        tokens = tokenize(view.window_text(index))
        self._unique.update(tokens)
        self._total += len(tokens)
        if self._total == 0:
            return {"value": float("nan")}
        return {"value": len(self._unique) / self._total}
