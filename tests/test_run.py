"""Replicate orchestration: concurrent execution and the ``parallel`` cap."""

from __future__ import annotations

import asyncio
from pathlib import Path

from renewal import run as run_module
from renewal.config import RunConfig


def _config(
    tmp_path: Path,
    *,
    replicates: int,
    parallel: int | None = None,
) -> RunConfig:
    run: dict = {"rounds": 1, "replicates": replicates, "seed": 0}
    if parallel is not None:
        run["parallel"] = parallel
    return RunConfig.model_validate(
        {
            "run": run,
            "agents": {"n": 1, "pool": [{"provider": "fake", "model": "fake"}]},
            "logging": {"out_dir": str(tmp_path), "experiment_name": "exp"},
        }
    )


def _tracking_replicate(tracker: dict):
    async def fake(config, *, run_dir, seed, debug=False, force_metrics=False):
        tracker["active"] += 1
        tracker["peak"] = max(tracker["peak"], tracker["active"])
        try:
            await asyncio.sleep(0.02)
        finally:
            tracker["active"] -= 1
        return {"run_id": "fake", "run_dir": str(run_dir), "seed": seed, "turns": 1}

    return fake


async def test_replicates_run_concurrently(monkeypatch, tmp_path):
    config = _config(tmp_path, replicates=4)
    tracker = {"active": 0, "peak": 0}
    monkeypatch.setattr(run_module, "run_replicate", _tracking_replicate(tracker))

    plans = run_module._plan_replicates(config, tmp_path, config.run.seed, None)
    failures = await run_module._run_replicates(
        config, plans, debug=False, force_metrics=False
    )

    assert failures == []
    assert tracker["peak"] == 4
    assert len(plans) == 4


async def test_parallel_caps_concurrency(monkeypatch, tmp_path):
    config = _config(tmp_path, replicates=4, parallel=2)
    tracker = {"active": 0, "peak": 0}
    monkeypatch.setattr(run_module, "run_replicate", _tracking_replicate(tracker))

    plans = run_module._plan_replicates(config, tmp_path, config.run.seed, None)
    await run_module._run_replicates(config, plans, debug=False, force_metrics=False)

    assert tracker["peak"] == 2


async def test_failing_replicate_does_not_cancel_siblings(monkeypatch, tmp_path):
    config = _config(tmp_path, replicates=3)
    completed: list[int] = []

    async def fake(config, *, run_dir, seed, debug=False, force_metrics=False):
        if seed == 1:
            raise RuntimeError("boom")
        completed.append(seed)
        await asyncio.sleep(0)
        return {"run_id": "fake", "run_dir": str(run_dir), "seed": seed, "turns": 1}

    monkeypatch.setattr(run_module, "run_replicate", fake)

    plans = run_module._plan_replicates(config, tmp_path, 0, None)
    failures = await run_module._run_replicates(
        config, plans, debug=False, force_metrics=False
    )

    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
    assert completed == [0, 2]
