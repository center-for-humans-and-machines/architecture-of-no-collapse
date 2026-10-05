"""Run folders and the per-run recorder.

One run = one directory holding the source config, a resolved config, the
transcript, the event stream, and metadata. The recorder writes transcript and
event JSONL incrementally (append + close per line, so a crash preserves
partial data) and updates ``meta.json`` status on start/finish/fail.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from renewal.config import RunConfig, dump_run_config
from renewal.core.message import Message

LOGGER = logging.getLogger(__name__)

TRANSCRIPT_FILENAME = "transcript.jsonl"
EVENTS_FILENAME = "events.jsonl"
META_FILENAME = "meta.json"
SOURCE_CONFIG_FILENAME = "config.yaml"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
METRICS_FILENAME = "metrics.parquet"
EMBEDDINGS_FILENAME = "embeddings.parquet"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    return {"git_commit": commit, "git_dirty": bool(dirty)}


def _required_env(config: RunConfig) -> list[str]:
    providers = {p.provider for p in config.agents.pool}
    names: list[str] = []
    if "vllm" in providers:
        names += ["MPCDF_VLLM_ENDPOINT_URL", "MPCDF_VLLM_MODEL"]
    return names


def _base_meta(config: RunConfig, seed: int, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": seed,
        "rounds": config.run.rounds,
        "window_size": config.run.window_size,
        "n_agents": config.agents.n,
        "models": [p.model_dump(mode="json") for p in config.agents.pool],
        "experiment_name": config.logging.experiment_name,
        "labels": dict(config.logging.labels),
        "metrics_enabled": list(config.metrics.enabled),
        "embedding_model": config.metrics.embedding.model,
        "metrics_status": "skipped",
        "status": "running",
        "started_at": _now(),
        "finished_at": None,
        "python": platform.python_version(),
        "requires_env": _required_env(config),
        **_git_metadata(),
    }


class RunRecorder:
    """Write transcript, events, and metadata for one run."""

    def __init__(self, run_dir: Path, run_id: str | None = None) -> None:
        self.run_dir = run_dir
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._transcript_path = run_dir / TRANSCRIPT_FILENAME
        self._events_path = run_dir / EVENTS_FILENAME
        self._meta_path = run_dir / META_FILENAME
        self._meta: dict[str, Any] = {}

    def start(self, *, seed: int, config: RunConfig) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._meta = _base_meta(config, seed, self.run_id)
        self._write_meta()
        self._event("run_started", seed=seed, rounds=config.run.rounds)

    def turn(self, message: Message) -> None:
        self._transcript(message)
        self._event(
            "turn",
            turn_index=message.turn_index,
            round=message.meta.get("round"),
            position=message.meta.get("position"),
            speaker=message.speaker,
        )

    def intervention(self, message: Message) -> None:
        self._transcript(message)
        self._event(
            "intervention",
            turn_index=message.turn_index,
            speaker=message.speaker,
        )

    def set_meta(self, **fields: Any) -> None:
        """Update one or more meta fields and persist immediately."""
        self._meta.update(fields)
        self._write_meta()

    def finish(self) -> None:
        self._meta["status"] = "completed"
        self._meta["finished_at"] = _now()
        self._write_meta()
        self._event("run_completed")

    def fail(self, error: BaseException) -> None:
        self._meta["status"] = "failed"
        self._meta["finished_at"] = _now()
        self._meta["error"] = f"{type(error).__name__}: {error}"
        self._write_meta()
        self._event("run_failed", error=f"{type(error).__name__}: {error}")

    def _transcript(self, message: Message) -> None:
        record = {"run_id": self.run_id, **message.to_dict()}
        with self._transcript_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _event(self, name: str, **fields: Any) -> None:
        record = {"run_id": self.run_id, "event": name, "ts": _now(), **fields}
        with self._events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _write_meta(self) -> None:
        self._meta_path.write_text(
            json.dumps(self._meta, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def allocate_run_dir(
    base: Path,
    experiment_name: str,
    seed: int,
    labels: Mapping[str, str],
    now: datetime | None = None,
) -> Path:
    """Create and return a unique run directory under ``base``."""
    stamp = (now or datetime.now()).strftime(TIMESTAMP_FORMAT)
    parts = [experiment_name]
    condition = labels.get("condition")
    if condition:
        parts.append(condition)
    parts.append(f"{stamp}_seed{seed}")
    return _unique_dir(Path(base).joinpath(*parts))


def snapshot_run_inputs(
    run_dir: Path,
    config: RunConfig,
    source_path: Path | None = None,
) -> None:
    """Write the source and resolved configs into ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved = dump_run_config(config)
    _write_yaml(run_dir / RESOLVED_CONFIG_FILENAME, resolved)
    if source_path is not None:
        (run_dir / SOURCE_CONFIG_FILENAME).write_bytes(
            Path(source_path).read_bytes()
        )
    else:
        _write_yaml(run_dir / SOURCE_CONFIG_FILENAME, resolved)
    LOGGER.info("snapshotted run inputs into %s", run_dir)


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{index}")
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
    raise RuntimeError(f"could not allocate unique directory from {path}")


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(dict(data), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _write_parquet_atomic(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to parquet atomically (temp file + rename).

    A reader polling the file (e.g. the viewer) never sees a half-written
    parquet, because the bytes land under a ``.tmp`` name and are swapped into
    place with an atomic ``os.replace``.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_metrics_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write long-format metric rows to parquet (no-op when empty)."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["value"] = df["value"].astype("float64")
    df["index"] = df["index"].astype("int64")
    df["window_start_round"] = df["window_start_round"].astype("Int64")
    df["window_end_round"] = df["window_end_round"].astype("Int64")
    _write_parquet_atomic(path, df)


def write_embeddings_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-window embedding vectors to parquet (no-op when empty)."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df["window_index"] = df["window_index"].astype("int64")
    df["start_round"] = df["start_round"].astype("int64")
    df["end_round"] = df["end_round"].astype("int64")
    _write_parquet_atomic(path, df)
