"""Recorder and run-directory artifact tests."""

import json

from renewal.artifacts import RunRecorder, allocate_run_dir, snapshot_run_inputs
from renewal.config import RunConfig
from renewal.core.message import Message


def _config() -> RunConfig:
    return RunConfig.model_validate(
        {
            "agents": {
                "n": 2,
                "pool": [
                    {"provider": "fake", "model": "fake"},
                    {"provider": "fake", "model": "fake"},
                ],
            },
            "logging": {"experiment_name": "t", "out_dir": "outputs"},
        }
    )


def test_recorder_lifecycle(tmp_path):
    run_dir = tmp_path / "run"
    recorder = RunRecorder(run_dir)
    recorder.start(seed=0, config=_config())
    recorder.turn(
        Message("assistant", "agent_0", "hi", 0, {"round": 0, "position": 0})
    )
    recorder.finish()

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["status"] == "completed"
    assert meta["seed"] == 0
    assert meta["run_id"] == recorder.run_id

    lines = (run_dir / "transcript.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["speaker"] == "agent_0"
    assert record["role"] == "assistant"

    events = (run_dir / "events.jsonl").read_text().strip().splitlines()
    event_names = [json.loads(e)["event"] for e in events]
    assert "run_started" in event_names
    assert "run_completed" in event_names


def test_recorder_fail_marks_status(tmp_path):
    run_dir = tmp_path / "run"
    recorder = RunRecorder(run_dir)
    recorder.start(seed=0, config=_config())
    recorder.fail(RuntimeError("boom"))
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["status"] == "failed"
    assert "boom" in meta["error"]


def test_allocate_run_dir_is_unique(tmp_path):
    first = allocate_run_dir(tmp_path, "exp", 0, {})
    second = allocate_run_dir(tmp_path, "exp", 0, {})
    assert first != second
    assert first.exists() and second.exists()


def test_snapshot_writes_configs(tmp_path):
    config = _config()
    run_dir = tmp_path / "run"
    snapshot_run_inputs(run_dir, config, source_path=None)
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "resolved_config.yaml").exists()
