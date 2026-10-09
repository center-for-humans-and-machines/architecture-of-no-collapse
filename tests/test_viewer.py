"""Viewer backend tests."""

import json
from pathlib import Path

import pandas as pd
import yaml
from fastapi.testclient import TestClient

from renewal.artifacts import write_metrics_parquet
from viewer.backend.app import create_app
from viewer.backend.index import build_index, load_summary


def _write_run(
    run_dir: Path,
    *,
    run_id: str,
    experiment: str = "e",
    seed: int = 0,
    rounds: int = 30,
    n_agents: int = 3,
    models: list[dict] | None = None,
    temperature: float = 0.9,
    max_tokens: int = 200,
    window_size: int = 10,
    status: str = "completed",
    started_at: str = "2026-10-05T10:00:00+00:00",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "seed": seed,
        "rounds": rounds,
        "n_agents": n_agents,
        "models": models or [{"provider": "fake", "model": "fake"}],
        "experiment_name": experiment,
        "status": status,
        "started_at": started_at,
        "finished_at": None,
        "window_size": window_size,
        "metrics_status": "completed",
        "metrics_enabled": ["lexical_diversity", "semantic_anchor_sim"],
        "labels": {},
    }
    (run_dir / "meta.json").write_text(json.dumps(meta))
    resolved = {
        "run": {"rounds": rounds, "window_size": window_size, "seed": seed},
        "agents": {
            "n": n_agents,
            "params": {"temperature": temperature, "max_tokens": max_tokens},
        },
        "metrics": {"window_size": window_size},
    }
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(resolved))


def test_load_summary(tmp_path):
    run_dir = tmp_path / "outputs" / "e" / "stamp_seed0"
    _write_run(run_dir, run_id="abc123")
    summary = load_summary(run_dir)
    assert summary.run_id == "abc123"
    assert summary.temperature == 0.9
    assert summary.max_tokens == 200
    assert summary.window_size == 10
    assert summary.models == [{"provider": "fake", "model": "fake"}]


def test_load_summary_returns_none_for_non_run(tmp_path):
    assert load_summary(tmp_path) is None


def test_build_index_sorts_newest_first(tmp_path):
    base = tmp_path / "outputs"
    _write_run(base / "a" / "run1", run_id="r1", started_at="2026-10-05T10:00:00+00:00")
    _write_run(base / "b" / "run2", run_id="r2", started_at="2026-10-05T11:00:00+00:00")
    assert [s.run_id for s in build_index(base)] == ["r2", "r1"]


def test_runs_filter_by_model(tmp_path):
    base = tmp_path / "outputs"
    _write_run(base / "a", run_id="r1", models=[{"provider": "vllm", "model": "Qwen/X"}])
    _write_run(base / "b", run_id="r2", models=[{"provider": "openai", "model": "gpt-4o-mini"}])
    client = TestClient(create_app(base))
    resp = client.get("/api/runs", params={"model": "Qwen/X"})
    assert resp.status_code == 200
    assert [r["run_id"] for r in resp.json()] == ["r1"]


def test_runs_filter_by_rounds_and_temperature(tmp_path):
    base = tmp_path / "outputs"
    _write_run(base / "a", run_id="r1", rounds=30, temperature=0.9)
    _write_run(base / "b", run_id="r2", rounds=200, temperature=0.5)
    client = TestClient(create_app(base))
    resp = client.get("/api/runs", params={"rounds_min": 100, "temp_max": 0.8})
    assert [r["run_id"] for r in resp.json()] == ["r2"]


def test_facets(tmp_path):
    base = tmp_path / "outputs"
    _write_run(base / "a", run_id="r1", models=[{"provider": "vllm", "model": "Qwen/X"}])
    _write_run(base / "b", run_id="r2", models=[{"provider": "openai", "model": "gpt-4o-mini"}])
    client = TestClient(create_app(base))
    facets = client.get("/api/facets").json()
    assert set(facets["models"]) == {"Qwen/X", "gpt-4o-mini"}
    assert facets["rounds"] == {"min": 30, "max": 30}


def test_metrics_endpoint(tmp_path):
    base = tmp_path / "outputs"
    run_dir = base / "e" / "run"
    _write_run(run_dir, run_id="r1")
    write_metrics_parquet(
        run_dir / "metrics.parquet",
        [
            {
                "run_id": "r1", "metric": "lexical_diversity", "cadence": "per_window",
                "index": 0, "value": 0.5, "window_start_round": 0, "window_end_round": 10,
                "seed": 0, "experiment": "e", "condition": None,
            },
            {
                "run_id": "r1", "metric": "semantic_adjacent_sim", "cadence": "per_window",
                "index": 0, "value": float("nan"), "window_start_round": 0,
                "window_end_round": 10, "seed": 0, "experiment": "e", "condition": None,
            },
        ],
    )
    client = TestClient(create_app(base))
    resp = client.get("/api/runs/r1/metrics")
    assert resp.status_code == 200
    rows = resp.json()
    assert rows[0]["metric"] == "lexical_diversity"
    assert rows[0]["value"] == 0.5
    # NaN value must serialize as JSON null, not invalid JSON
    assert rows[1]["value"] is None


def test_transcript_endpoint(tmp_path):
    base = tmp_path / "outputs"
    run_dir = base / "e" / "run"
    _write_run(run_dir, run_id="r1")
    (run_dir / "transcript.jsonl").write_text(
        json.dumps(
            {
                "run_id": "r1", "role": "assistant", "speaker": "agent_0",
                "content": "hi", "turn_index": 0, "meta": {"round": 0},
            }
        )
        + "\n"
    )
    client = TestClient(create_app(base))
    resp = client.get("/api/runs/r1/transcript")
    assert resp.status_code == 200
    assert resp.json()[0]["content"] == "hi"


def test_unknown_run_is_404(tmp_path):
    client = TestClient(create_app(tmp_path / "outputs"))
    resp = client.get("/api/runs/nope/metrics")
    assert resp.status_code == 404


def test_memories_endpoint_reconstructs_from_events(tmp_path):
    base = tmp_path / "outputs"
    run_dir = base / "e" / "run"
    _write_run(run_dir, run_id="r1")
    lines = [
        {
            "run_id": "r1", "event": "memory_created", "id": 1,
            "text": "first memory", "created_turn": 0, "since_turn": 0,
            "surfaced_count": 0, "last_surfaced_turn": None,
            "provenance": {"words": ["a", "b"]}, "embedding": [0.1, 0.2, 0.3],
            "selector": "random_unsurfaced",
        },
        {
            "run_id": "r1", "event": "memory_created", "id": 2,
            "text": "second memory", "created_turn": 3, "since_turn": 1,
            "surfaced_count": 0, "last_surfaced_turn": None,
            "provenance": {}, "embedding": None,
            "selector": "random_unsurfaced",
        },
        {
            "run_id": "r1", "event": "memory_surfaced", "id": 1,
            "turn_index": 5, "surfaced_count": 1, "last_surfaced_turn": 5,
            "selector": "random_unsurfaced",
        },
        {
            "run_id": "r1", "event": "memory_surfaced", "id": 1,
            "turn_index": 9, "surfaced_count": 2, "last_surfaced_turn": 9,
            "selector": "random_unsurfaced",
        },
    ]
    (run_dir / "memories.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )
    client = TestClient(create_app(base))
    resp = client.get("/api/runs/r1/memories")
    assert resp.status_code == 200
    memories = resp.json()
    assert [m["id"] for m in memories] == [1, 2]
    first = memories[0]
    assert first["text"] == "first memory"
    assert first["surfaced_count"] == 2
    assert first["last_surfaced_turn"] == 9
    assert first["surfaced_turns"] == [5, 9]
    assert first["embedding_dim"] == 3
    assert first["provenance"] == {"words": ["a", "b"]}
    assert memories[1]["embedding_dim"] is None


def test_memories_endpoint_empty_without_file(tmp_path):
    base = tmp_path / "outputs"
    _write_run(base / "e" / "run", run_id="r1")
    client = TestClient(create_app(base))
    resp = client.get("/api/runs/r1/memories")
    assert resp.status_code == 200
    assert resp.json() == []
