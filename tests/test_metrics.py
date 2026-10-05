"""Metric tool, windowing, loop-hook, and config validation tests."""

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from renewal.config import RunConfig, load_run_config
from renewal.core.agent import Agent
from renewal.core.loop import Loop
from renewal.core.message import Message
from renewal.core.scheduler import Scheduler
from renewal.llm.fake import FakeLLM
from renewal.metrics.base import RunView, build_windows
from renewal.metrics.lexical import LexicalDiversity
from renewal.metrics.semantic import SemanticAdjacentSim, SemanticAnchorSim
from renewal.run import run_replicate

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "smoke_metrics.yaml"


def _msg(speaker, content, round_index):
    return Message("assistant", speaker, content, 0, {"round": round_index})


def _view(turns, *, window_size=2, embedder=None):
    return RunView(
        turns,
        window_size=window_size,
        run_id="r",
        seed=0,
        experiment="e",
        condition=None,
        embedder=embedder,
    )


# --- windowing ---


def test_build_windows_groups_by_round():
    turns = [_msg("a", "x", 0), _msg("b", "y", 0), _msg("a", "z", 1)]
    windows = build_windows(turns, 2)
    assert len(windows) == 1
    assert windows[0].text == "x\ny\nz"


def test_build_windows_partial_last():
    turns = [
        _msg("a", "x", 0),
        _msg("b", "y", 0),
        _msg("a", "z", 1),
        _msg("b", "w", 2),
    ]
    windows = build_windows(turns, 2)
    assert len(windows) == 2
    assert windows[1].start_round == 2
    assert windows[1].end_round == 3
    assert windows[1].text == "w"


# --- loop window hook ---


class _Spy:
    def __init__(self):
        self.calls = []

    async def __call__(self, index, turns):
        self.calls.append((index, turns))


class _NullRecorder:
    def turn(self, message):
        pass

    def intervention(self, message):
        pass


class AppendIntervention:
    name = "append"

    async def act(self, messages):
        return Message("intervention", "iv", "note", 0)


async def test_loop_hook_receives_assistant_turns_only():
    agents = [Agent(f"a{i}", FakeLLM(), "p", {}) for i in range(2)]
    spy = _Spy()
    loop = Loop(
        agents,
        Scheduler(2, 0),
        [AppendIntervention()],
        _NullRecorder(),
        window_size=2,
        on_window=spy,
    )
    await loop.run(4)

    assert [index for index, _ in spy.calls] == [0, 1]
    for _, turns in spy.calls:
        assert all(m.role == "assistant" for m in turns)
        assert len(turns) == 4  # 2 agents * 2 rounds


# --- lexical diversity ---


async def test_lexical_diversity_ratio():
    tool = LexicalDiversity()
    view = _view([_msg("s", "a b a", 0), _msg("s", "b c", 1)], window_size=1)
    assert await tool.compute(view, 0) == {"value": 2 / 3}
    assert await tool.compute(view, 1) == {"value": 3 / 5}


# --- semantic metrics with a stub embedder ---


class _StubEmbedder:
    def __init__(self, mapping):
        self._mapping = mapping

    async def embed(self, texts):
        return [self._mapping[t] for t in texts]


async def test_anchor_sim_uses_first_window():
    embedder = _StubEmbedder({"zero": [1.0, 0.0], "one": [0.0, 1.0]})
    view = _view(
        [_msg("s", "zero", 0), _msg("s", "one", 1)],
        window_size=1,
        embedder=embedder,
    )
    tool = SemanticAnchorSim()
    assert await tool.compute(view, 0) == {"value": 1.0}
    assert await tool.compute(view, 1) == {"value": 0.0}


async def test_adjacent_sim_uses_previous_window():
    embedder = _StubEmbedder(
        {"zero": [1.0, 0.0], "one": [0.0, 1.0], "two": [1.0, 0.0]}
    )
    view = _view(
        [_msg("s", "zero", 0), _msg("s", "one", 1), _msg("s", "two", 2)],
        window_size=1,
        embedder=embedder,
    )
    tool = SemanticAdjacentSim()
    r0 = await tool.compute(view, 0)
    r1 = await tool.compute(view, 1)
    r2 = await tool.compute(view, 2)
    assert math.isnan(r0["value"])
    assert r1["value"] == 0.0
    assert r2["value"] == 0.0


# --- config validation ---


def _base() -> dict:
    return {"agents": {"n": 1, "pool": [{"provider": "fake", "model": "fake"}]}}


def test_unknown_metric_name_rejected():
    data = _base()
    data["metrics"] = {"enabled": ["bogus"]}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_known_metrics_accepted():
    data = _base()
    data["metrics"] = {"enabled": ["lexical_diversity", "cross_run_sim"]}
    config = RunConfig.model_validate(data)
    assert config.metrics.enabled == ["lexical_diversity", "cross_run_sim"]


def test_unknown_embedding_key_rejected():
    data = _base()
    data["metrics"] = {"embedding": {"bogus": 1}}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_embedding_defaults():
    config = RunConfig.model_validate(_base())
    assert config.metrics.embedding.provider == "vllm"
    assert config.metrics.embedding.model == "Qwen/Qwen3-Embedding-8B"


# --- offline smoke ---


async def test_smoke_metrics_run(tmp_path):
    config = load_run_config(SMOKE)
    run_dir = tmp_path / "run"
    await run_replicate(config, run_dir=run_dir, seed=0, debug=False)

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["metrics_status"] == "completed"

    metrics = pd.read_parquet(run_dir / "metrics.parquet")
    assert set(metrics["metric"]) == {
        "lexical_diversity",
        "semantic_anchor_sim",
        "semantic_adjacent_sim",
    }
    assert len(metrics) == 9  # 3 windows * 3 per_window metrics

    embeddings = pd.read_parquet(run_dir / "embeddings.parquet")
    assert len(embeddings) == 3
