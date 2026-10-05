"""Scan a run tree into filterable summaries for the viewer.

The viewer reads the run artifacts directly (rather than importing ``renewal``)
so it stays a separate process. The artifact filenames below are the stable
contract defined in ``renewal/artifacts.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

META_FILENAME = "meta.json"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
TRANSCRIPT_FILENAME = "transcript.jsonl"
METRICS_FILENAME = "metrics.parquet"
EMBEDDINGS_FILENAME = "embeddings.parquet"


@dataclass
class RunSummary:
    run_id: str
    run_dir: str
    experiment: str
    seed: int
    status: str
    started_at: str | None
    finished_at: str | None
    rounds: int
    n_agents: int
    models: list[dict[str, str]]
    temperature: float | None
    max_tokens: int | None
    top_p: float | None
    window_size: int
    metrics_status: str | None
    metrics_enabled: list[str]
    has_metrics: bool
    labels: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "experiment": self.experiment,
            "seed": self.seed,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "rounds": self.rounds,
            "n_agents": self.n_agents,
            "models": self.models,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "window_size": self.window_size,
            "metrics_status": self.metrics_status,
            "metrics_enabled": self.metrics_enabled,
            "has_metrics": self.has_metrics,
            "labels": self.labels,
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    return value if isinstance(value, dict) else {}


def load_summary(run_dir: Path) -> RunSummary | None:
    """Build a RunSummary from one run directory, or None when not a run."""
    meta_path = run_dir / META_FILENAME
    if not meta_path.exists():
        return None
    meta = _load_json(meta_path)
    cfg_path = run_dir / RESOLVED_CONFIG_FILENAME
    cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
    agents = cfg.get("agents") or {}
    params = agents.get("params") or {}
    run = cfg.get("run") or {}
    metrics = cfg.get("metrics") or {}
    models = meta.get("models") or []
    return RunSummary(
        run_id=meta.get("run_id") or run_dir.name,
        run_dir=str(run_dir),
        experiment=meta.get("experiment_name") or run_dir.parent.name,
        seed=int(meta.get("seed") or 0),
        status=meta.get("status") or "unknown",
        started_at=meta.get("started_at"),
        finished_at=meta.get("finished_at"),
        rounds=int(meta.get("rounds") or run.get("rounds") or 0),
        n_agents=int(meta.get("n_agents") or agents.get("n") or 0),
        models=[dict(m) for m in models],
        temperature=params.get("temperature"),
        max_tokens=params.get("max_tokens"),
        top_p=params.get("top_p"),
        window_size=int(meta.get("window_size") or run.get("window_size") or metrics.get("window_size") or 0),
        metrics_status=meta.get("metrics_status"),
        metrics_enabled=list(meta.get("metrics_enabled") or []),
        has_metrics=(run_dir / METRICS_FILENAME).exists(),
        labels=dict(meta.get("labels") or {}),
    )


def find_run_dirs(base: Path) -> list[Path]:
    """Recursively find directories that contain a meta.json."""
    return sorted((p.parent for p in base.rglob(META_FILENAME)), key=str)


def build_index(base: Path) -> list[RunSummary]:
    """Scan the run tree and return summaries, newest first."""
    summaries = [s for s in (load_summary(d) for d in find_run_dirs(base)) if s is not None]
    summaries.sort(key=lambda s: (s.started_at or ""), reverse=True)
    return summaries
