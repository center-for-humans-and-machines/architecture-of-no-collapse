"""FastAPI app for the run viewer (JSON API)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from viewer.backend.index import (
    META_FILENAME,
    METRICS_FILENAME,
    TRANSCRIPT_FILENAME,
    RunSummary,
    build_index,
)

INDEX_TTL = 2.0


class Index:
    """TTL-cached scan of the run tree, so the list stays fresh while cheap."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self._summaries: list[RunSummary] = []
        self._by_id: dict[str, Path] = {}
        self._fetched = 0.0

    def summaries(self) -> list[RunSummary]:
        now = time.monotonic()
        if now - self._fetched > INDEX_TTL or not self._by_id:
            self._refresh()
        return self._summaries

    def run_dir(self, run_id: str) -> Path:
        self.summaries()
        run_dir = self._by_id.get(run_id)
        if run_dir is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id}")
        return run_dir

    def _refresh(self) -> None:
        self._summaries = build_index(self.base)
        self._by_id = {s.run_id: Path(s.run_dir) for s in self._summaries}
        self._fetched = time.monotonic()


def _clean_value(value: Any) -> Any:
    """Convert a DataFrame scalar into a JSON-safe value (NaN/NA → None)."""
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, list, dict)):
        value = value.item()
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, (list, dict)):
        return value
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _clean_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: _clean_value(v) for k, v in record.items()} for record in records]


def read_transcript(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / TRANSCRIPT_FILENAME
    if not path.exists():
        return []
    messages: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                messages.append(json.loads(line))
    return messages


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / METRICS_FILENAME
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    return _clean_records(df.to_dict(orient="records"))


def read_meta(run_dir: Path) -> dict[str, Any]:
    path = run_dir / META_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _model_names(summary: RunSummary) -> list[str]:
    return [m.get("model", "") for m in summary.models]


def _filter_summaries(
    summaries: list[RunSummary],
    *,
    model: list[str],
    rounds_min: int | None,
    rounds_max: int | None,
    temp_min: float | None,
    temp_max: float | None,
    max_tokens_min: int | None,
    max_tokens_max: int | None,
    window_size: int | None,
    since: str | None,
    until: str | None,
) -> list[RunSummary]:
    result: list[RunSummary] = []
    for s in summaries:
        if model and not set(_model_names(s)).intersection(model):
            continue
        if rounds_min is not None and s.rounds < rounds_min:
            continue
        if rounds_max is not None and s.rounds > rounds_max:
            continue
        if temp_min is not None and (s.temperature is None or s.temperature < temp_min):
            continue
        if temp_max is not None and (s.temperature is None or s.temperature > temp_max):
            continue
        if max_tokens_min is not None and (s.max_tokens is None or s.max_tokens < max_tokens_min):
            continue
        if max_tokens_max is not None and (s.max_tokens is None or s.max_tokens > max_tokens_max):
            continue
        if window_size is not None and s.window_size != window_size:
            continue
        if since is not None and (s.started_at is None or s.started_at < since):
            continue
        if until is not None and (s.started_at is None or s.started_at > until):
            continue
        result.append(s)
    return result


def create_app(base_dir: Path) -> FastAPI:
    index = Index(base_dir)
    app = FastAPI(title="renewal viewer")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/facets")
    def facets() -> dict[str, Any]:
        summaries = index.summaries()
        models = sorted({m.get("model", "") for s in summaries for m in s.models})
        rounds = [s.rounds for s in summaries]
        temps = [s.temperature for s in summaries if s.temperature is not None]
        tokens = [s.max_tokens for s in summaries if s.max_tokens is not None]
        windows = sorted({s.window_size for s in summaries})
        started = [s.started_at for s in summaries if s.started_at]
        return {
            "models": models,
            "rounds": {"min": min(rounds) if rounds else None, "max": max(rounds) if rounds else None},
            "temperature": {"min": min(temps) if temps else None, "max": max(temps) if temps else None},
            "max_tokens": {"min": min(tokens) if tokens else None, "max": max(tokens) if tokens else None},
            "window_size": windows,
            "time": {"min": min(started) if started else None, "max": max(started) if started else None},
            "count": len(summaries),
        }

    @app.get("/api/runs")
    def runs(
        model: list[str] | None = Query(default=None),
        rounds_min: int | None = None,
        rounds_max: int | None = None,
        temp_min: float | None = None,
        temp_max: float | None = None,
        max_tokens_min: int | None = None,
        max_tokens_max: int | None = None,
        window_size: int | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[dict[str, Any]]:
        summaries = _filter_summaries(
            index.summaries(),
            model=model or [],
            rounds_min=rounds_min,
            rounds_max=rounds_max,
            temp_min=temp_min,
            temp_max=temp_max,
            max_tokens_min=max_tokens_min,
            max_tokens_max=max_tokens_max,
            window_size=window_size,
            since=since,
            until=until,
        )
        return [s.to_dict() for s in summaries]

    @app.get("/api/runs/{run_id}/transcript")
    def transcript(run_id: str) -> list[dict[str, Any]]:
        return read_transcript(index.run_dir(run_id))

    @app.get("/api/runs/{run_id}/metrics")
    def metrics(run_id: str) -> list[dict[str, Any]]:
        return read_metrics(index.run_dir(run_id))

    @app.get("/api/runs/{run_id}/meta")
    def meta(run_id: str) -> dict[str, Any]:
        return read_meta(index.run_dir(run_id))

    # Serve the built frontend when it exists (production, single process).
    dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="static")

    return app
