"""Metric tools: interface, run view, windowing, and the incremental runner."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from renewal.artifacts import write_embeddings_parquet, write_metrics_parquet
from renewal.core.message import Message
from renewal.llm.embed import EmbeddingClient
from renewal.registry import Registry

LOGGER = logging.getLogger(__name__)

# The Kong-default set used when ``--metrics`` forces metrics on for a config
# whose ``metrics.enabled`` is empty.
DEFAULT_METRICS = [
    "lexical_diversity",
    "semantic_anchor_sim",
    "semantic_adjacent_sim",
    "cross_run_sim",
]


class MetricTool(Protocol):
    name: str
    cadence: Literal["per_turn", "per_window", "per_run"]
    needs_embedding: bool

    async def compute(self, view: "RunView", index: int) -> dict[str, float]: ...


def cosine(a: Any, b: Any) -> float:
    """Cosine similarity of two vectors (unit norm makes this a dot product)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


@dataclass(frozen=True)
class Window:
    index: int
    start_round: int
    end_round: int
    turns: tuple[Message, ...]

    @property
    def text(self) -> str:
        return "\n".join(turn.content for turn in self.turns)


def build_windows(turns: list[Message], window_size: int) -> list[Window]:
    """Group assistant turns into consecutive round-windows."""
    if not turns:
        return []
    int_rounds = [t.meta.get("round") for t in turns if isinstance(t.meta.get("round"), int)]
    if not int_rounds:
        return []
    max_round = max(int_rounds)
    n_windows = (max_round + 1 + window_size - 1) // window_size
    windows: list[Window] = []
    for index in range(n_windows):
        start = index * window_size
        end = min((index + 1) * window_size, max_round + 1)
        window_turns = tuple(
            t
            for t in turns
            if isinstance(t.meta.get("round"), int) and start <= t.meta["round"] < end
        )
        windows.append(Window(index, start, end, window_turns))
    return windows


class RunView:
    """Read-only view of one run's assistant turns, windows, and embedder."""

    def __init__(
        self,
        turns: list[Message],
        *,
        window_size: int,
        run_id: str,
        seed: int,
        experiment: str,
        condition: str | None,
        embedder: EmbeddingClient,
    ) -> None:
        self.turns = list(turns)
        self.window_size = window_size
        self.run_id = run_id
        self.seed = seed
        self.experiment = experiment
        self.condition = condition
        self.embedder = embedder
        self._windows = build_windows(self.turns, window_size)
        self._cache: dict[str, list[float]] = {}

    @property
    def n_windows(self) -> int:
        return len(self._windows)

    def window_turns(self, index: int) -> list[Message]:
        return list(self._windows[index].turns)

    def window_text(self, index: int) -> str:
        return self._windows[index].text

    def window_bounds(self, index: int) -> tuple[int, int]:
        return self._windows[index].start_round, self._windows[index].end_round

    async def embedding(self, text: str) -> list[float]:
        if text not in self._cache:
            (self._cache[text],) = await self.embedder.embed([text])
        return self._cache[text]


class MetricRunner:
    """Subscribes to loop window boundaries and computes metrics incrementally."""

    def __init__(
        self,
        *,
        run_id: str,
        seed: int,
        experiment: str,
        condition: str | None,
        window_size: int,
        embedder: EmbeddingClient,
        tools: list[MetricTool],
        need_embeddings: bool,
        run_dir: Path,
    ) -> None:
        self.run_id = run_id
        self.seed = seed
        self.experiment = experiment
        self.condition = condition
        self.window_size = window_size
        self.embedder = embedder
        self.tools = tools
        self.need_embeddings = need_embeddings
        self.run_dir = run_dir
        self.status = "running"
        self._turns: list[Message] = []
        self._metric_rows: list[dict[str, Any]] = []
        self._embedding_rows: list[dict[str, Any]] = []
        self._computed: set[int] = set()

    def _view(self) -> RunView:
        return RunView(
            self._turns,
            window_size=self.window_size,
            run_id=self.run_id,
            seed=self.seed,
            experiment=self.experiment,
            condition=self.condition,
            embedder=self.embedder,
        )

    async def on_window(self, index: int, turns: list[Message]) -> None:
        self._turns.extend(turns)
        await self._compute(self._view(), index)
        self._flush()

    async def finalize(self) -> str:
        if self._turns:
            view = self._view()
            last = view.n_windows - 1
            if last not in self._computed:
                await self._compute(view, last)
        self._flush()
        if self.status == "running":
            self.status = "completed"
        return self.status

    async def _compute(self, view: RunView, index: int) -> None:
        self._computed.add(index)
        start, end = view.window_bounds(index)
        text = view.window_text(index)

        if self.need_embeddings and text:
            try:
                vector = await view.embedding(text)
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("embedding failed for window %s: %s", index, error)
                self.status = "partial"
            else:
                self._embedding_rows.append(
                    {
                        "run_id": self.run_id,
                        "window_index": index,
                        "start_round": start,
                        "end_round": end,
                        "embedding": list(vector),
                    }
                )

        for tool in self.tools:
            try:
                result = await tool.compute(view, index)
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("%s failed for window %s: %s", tool.name, index, error)
                self.status = "partial"
                result = {"value": float("nan")}
            for key, value in result.items():
                metric = tool.name if key == "value" else f"{tool.name}.{key}"
                self._metric_rows.append(
                    {
                        "run_id": self.run_id,
                        "metric": metric,
                        "cadence": tool.cadence,
                        "index": index,
                        "value": float(value),
                        "window_start_round": start,
                        "window_end_round": end,
                        "seed": self.seed,
                        "experiment": self.experiment,
                        "condition": self.condition,
                    }
                )

    def _flush(self) -> None:
        write_metrics_parquet(self.run_dir / "metrics.parquet", self._metric_rows)
        write_embeddings_parquet(self.run_dir / "embeddings.parquet", self._embedding_rows)


def enabled_metrics(config: Any, force: bool = False) -> list[str]:
    """Resolve the effective metric set, applying the ``--metrics`` default."""
    enabled = list(config.metrics.enabled)
    if enabled:
        return enabled
    return list(DEFAULT_METRICS) if force else []


def build_metric_runner(
    config: Any,
    *,
    run_id: str,
    seed: int,
    experiment: str,
    condition: str | None,
    run_dir: Path,
    enabled: list[str],
) -> MetricRunner | None:
    """Build the incremental metric runner for one run, or None when disabled."""
    if not enabled:
        return None
    embedder = config.metrics.embedding.materialize()
    tools: list[MetricTool] = []
    need_embeddings = False
    for name in enabled:
        cls = Registry.get("metric", name)
        if cls.cadence == "per_window":
            tools.append(cls())
        if cls.needs_embedding:
            need_embeddings = True
    return MetricRunner(
        run_id=run_id,
        seed=seed,
        experiment=experiment,
        condition=condition,
        window_size=config.metrics.window_size,
        embedder=embedder,
        tools=tools,
        need_embeddings=need_embeddings,
        run_dir=run_dir,
    )
