"""Cross-run analysis tests."""

import json

import numpy as np
import pytest

from renewal.artifacts import write_embeddings_parquet
from renewal.metrics.analyze import analyze_experiment


def _write_run(run_dir, run_id, vectors, seed=0, condition=None):
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "seed": seed,
        "experiment_name": "e",
        "labels": {"condition": condition} if condition else {},
        "status": "completed",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta))
    rows = [
        {
            "run_id": run_id,
            "window_index": i,
            "start_round": i,
            "end_round": i + 1,
            "embedding": list(vector),
        }
        for i, vector in enumerate(vectors)
    ]
    write_embeddings_parquet(run_dir / "embeddings.parquet", rows)


def _group_row(rows):
    return [r for r in rows if r["run_id"].startswith("group")][0]


def test_cross_run_sim_two_runs(tmp_path):
    _write_run(tmp_path / "a", "a", [[1.0, 0.0], [0.0, 1.0]])
    _write_run(tmp_path / "b", "b", [[1.0, 0.0], [0.0, 1.0]])
    rows = analyze_experiment(tmp_path)
    assert _group_row(rows)["value"] == pytest.approx(1.0)


def test_cross_run_sim_orthogonal(tmp_path):
    _write_run(tmp_path / "a", "a", [[1.0, 0.0]])
    _write_run(tmp_path / "b", "b", [[0.0, 1.0]])
    rows = analyze_experiment(tmp_path)
    assert _group_row(rows)["value"] == pytest.approx(0.0)


def test_cross_run_sim_single_run_is_nan(tmp_path):
    _write_run(tmp_path / "a", "a", [[1.0, 0.0]])
    rows = analyze_experiment(tmp_path)
    assert np.isnan(_group_row(rows)["value"])


def test_cross_run_sim_aligns_shared_prefix(tmp_path):
    _write_run(tmp_path / "a", "a", [[1.0, 0.0], [0.0, 1.0]])
    _write_run(tmp_path / "b", "b", [[1.0, 0.0]])
    rows = analyze_experiment(tmp_path)
    # w = min(2, 1) = 1 -> cos([1,0],[1,0]) = 1
    assert _group_row(rows)["value"] == pytest.approx(1.0)


def test_analyze_skips_runs_without_embeddings(tmp_path):
    _write_run(tmp_path / "a", "a", [[1.0, 0.0]])
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "meta.json").write_text(json.dumps({"run_id": "b", "labels": {}}))
    rows = analyze_experiment(tmp_path)
    # only run "a" has embeddings -> single run -> NaN
    assert np.isnan(_group_row(rows)["value"])
