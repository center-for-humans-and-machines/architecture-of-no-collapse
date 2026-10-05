"""Semantic similarity metrics over window embeddings."""

from __future__ import annotations

from renewal.metrics.base import cosine
from renewal.registry import Registry


@Registry.register("metric", "semantic_anchor_sim")
class SemanticAnchorSim:
    name = "semantic_anchor_sim"
    cadence = "per_window"
    needs_embedding = True

    def __init__(self) -> None:
        self._v0 = None

    async def compute(self, view, index):
        text = view.window_text(index)
        if not text:
            if index == 0:
                self._v0 = None
            return {"value": float("nan")}
        vector = await view.embedding(text)
        if index == 0:
            self._v0 = vector
            return {"value": 1.0}
        if self._v0 is None:
            return {"value": float("nan")}
        return {"value": cosine(vector, self._v0)}


@Registry.register("metric", "semantic_adjacent_sim")
class SemanticAdjacentSim:
    name = "semantic_adjacent_sim"
    cadence = "per_window"
    needs_embedding = True

    def __init__(self) -> None:
        self._prev = None

    async def compute(self, view, index):
        text = view.window_text(index)
        if not text:
            self._prev = None
            return {"value": float("nan")}
        vector = await view.embedding(text)
        if self._prev is None:
            self._prev = vector
            return {"value": float("nan")}
        result = cosine(vector, self._prev)
        self._prev = vector
        return {"value": result}


@Registry.register("metric", "cross_run_sim")
class CrossRunSim:
    name = "cross_run_sim"
    cadence = "per_run"
    needs_embedding = True

    async def compute(self, view, index):
        raise NotImplementedError("cross_run_sim is computed by `renewal analyze`")
