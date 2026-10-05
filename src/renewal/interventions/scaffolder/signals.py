"""Per-turn adjacent-similarity signal for the scaffolder."""

from __future__ import annotations

import numpy as np

from renewal.llm.embed import EmbeddingClient


def _cosine(a: object, b: object) -> float:
    """Cosine similarity of two vectors (unit norm makes this a dot product)."""
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denom)


class AdjacentSimilaritySignal:
    """Measure cosine similarity between consecutive model turns.

    ``update`` embeds ``text`` and compares it to the previous turn's vector;
    the first call has nothing to compare against and returns ``None``.
    """

    def __init__(self, embedder: EmbeddingClient) -> None:
        self._embedder = embedder
        self._previous: list[float] | None = None

    async def update(self, text: str) -> float | None:
        (vector,) = await self._embedder.embed([text])
        if self._previous is None:
            self._previous = vector
            return None
        result = _cosine(vector, self._previous)
        self._previous = vector
        return result
