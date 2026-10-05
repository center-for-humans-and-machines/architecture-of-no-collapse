"""Scaffolder intervention tests (offline: fake embedder, sampler, search)."""

from __future__ import annotations

import pytest

from renewal.core.message import Message
from renewal.interventions.scaffolder import (
    DEEPEN_PROMPTS,
    INNOVATE_PROMPTS,
    Action,
    EscalationPolicy,
    GloVeWordSampler,
    ScaffolderIntervention,
    SearchResult,
    TopicInfuser,
    TopicUnavailable,
)
from renewal.interventions.scaffolder.signals import AdjacentSimilaritySignal
from renewal.llm.embed import FakeEmbedding


# --- policy ---------------------------------------------------------------


def test_policy_deepen_below_threshold():
    policy = EscalationPolicy(threshold=0.85)
    assert policy.decide(0.3) == Action.DEEPEN
    assert policy.decide(0.3) == Action.DEEPEN


def test_policy_escalation_and_reset():
    policy = EscalationPolicy(threshold=0.85)
    assert policy.decide(0.9) == Action.INNOVATE  # first crossing
    assert policy.decide(0.9) == Action.INJECT     # crossed again
    assert policy.decide(0.9) == Action.INJECT     # stays stuck -> keep injecting
    assert policy.decide(0.1) == Action.DEEPEN     # recovered -> reset
    assert policy.decide(0.9) == Action.INNOVATE   # streak restarts


def test_policy_none_signal_is_healthy():
    policy = EscalationPolicy(threshold=0.85)
    assert policy.decide(None) == Action.DEEPEN


def test_policy_rotates_prompts():
    policy = EscalationPolicy(threshold=0.85)
    deepen = [policy.prompt(Action.DEEPEN) for _ in range(4)]
    innovate = [policy.prompt(Action.INNOVATE) for _ in range(4)]
    assert deepen == list(DEEPEN_PROMPTS)
    assert innovate == list(INNOVATE_PROMPTS)
    # rotation wraps around
    assert policy.prompt(Action.DEEPEN) == DEEPEN_PROMPTS[0]


def test_policy_inject_has_no_fixed_prompt():
    policy = EscalationPolicy(threshold=0.85)
    with pytest.raises(ValueError):
        policy.prompt(Action.INJECT)


# --- signal ---------------------------------------------------------------


async def test_signal_first_turn_is_none_then_cosine():
    signal = AdjacentSimilaritySignal(FakeEmbedding(dim=64))
    assert await signal.update("alpha beta gamma") is None
    # identical text -> cosine ~1.0
    value = await signal.update("alpha beta gamma")
    assert value is not None and value > 0.99


# --- word sampler ----------------------------------------------------------


def _write_vectors(tmp_path, *, header=True):
    lines = []
    if header:
        lines.append("3 2")
    lines += [
        "apple 1.0 0.0",
        "banana 0.0 1.0",
        "cherry 0.6 0.8",
    ]
    path = tmp_path / "tiny_glove.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_glove_sampler_returns_known_words(tmp_path):
    sampler = GloVeWordSampler(model_path=str(_write_vectors(tmp_path)), seed=0)
    words = sampler.sample(3)
    assert len(words) == 3
    assert set(words) <= {"apple", "banana", "cherry"}


def test_glove_sampler_headerless(tmp_path):
    path = _write_vectors(tmp_path, header=False)
    sampler = GloVeWordSampler(model_path=str(path), seed=0)
    assert len(sampler.sample(2)) == 2


def test_glove_sampler_deterministic(tmp_path):
    path = _write_vectors(tmp_path)
    a = GloVeWordSampler(model_path=str(path), seed=1).sample(5)
    b = GloVeWordSampler(model_path=str(path), seed=1).sample(5)
    assert a == b


# --- topic infuser ---------------------------------------------------------


class _FakeSampler:
    def sample(self, count):
        return ["mars", "terraforming", "colony"][:count]


class _FakeSearch:
    def __init__(self, results=None):
        self.results = results or {}
        self.calls = []

    async def search(self, query, top_k):
        self.calls.append(query)
        return self.results.get(query, [])


async def test_infuser_builds_topic():
    client = _FakeSearch(
        {
            "mars terraforming colony": [
                SearchResult("Title", "https://example.com/x", "Some content.", 0.9)
            ]
        }
    )
    infuser = TopicInfuser(_FakeSampler(), client, num_words=3)
    topic = await infuser.inject()
    assert topic.words == ("mars", "terraforming", "colony")
    assert topic.query == "mars terraforming colony"
    assert topic.title == "Title"
    assert topic.url == "https://example.com/x"


async def test_infuser_narrows_query_until_hit():
    client = _FakeSearch(
        {
            "mars terraforming": [
                SearchResult("T", "https://example.com/y", "Body", 0.5)
            ]
        }
    )
    infuser = TopicInfuser(_FakeSampler(), client, num_words=3)
    topic = await infuser.inject()
    assert topic.query == "mars terraforming"
    assert client.calls == ["mars terraforming colony", "mars terraforming"]


async def test_infuser_raises_when_no_results():
    infuser = TopicInfuser(_FakeSampler(), _FakeSearch({}), num_words=3)
    with pytest.raises(TopicUnavailable):
        await infuser.inject()


# --- intervention ----------------------------------------------------------


def _intervention(*, visibility="all", threshold=0.85, search=None):
    return ScaffolderIntervention(
        threshold=threshold,
        visibility=visibility,
        embedder=FakeEmbedding(dim=64),
        sampler=_FakeSampler(),
        search_client=search or _FakeSearch(
            {
                "mars terraforming colony": [
                    SearchResult("Mars", "https://example.com/m", "content", 0.9)
                ]
            }
        ),
    )


def _assistant(text, turn):
    return Message("assistant", f"agent_{turn}", text, turn)


async def test_intervention_deepen_then_innovate_then_inject():
    iv = _intervention(visibility="all")
    # turn 1: no previous -> deepen
    m1 = await iv.act([_assistant("topic A", 0)])
    assert m1.meta["level"] == "deepen"
    assert m1.transient is False
    # turn 2: identical text -> high similarity -> innovate
    m2 = await iv.act([_assistant("topic A", 0), _assistant("topic A", 1)])
    assert m2.meta["level"] == "innovate"
    # turn 3: still identical -> inject (grounded in search)
    m3 = await iv.act(
        [_assistant("topic A", 0), _assistant("topic A", 1), _assistant("topic A", 2)]
    )
    assert m3.meta["level"] == "inject"
    assert "source_url" in m3.meta and m3.meta["source_url"] == "https://example.com/m"
    assert "Mars" in m3.content


async def test_intervention_visibility_injections():
    iv = _intervention(visibility="injections")
    m1 = await iv.act([_assistant("topic A", 0)])
    assert m1.meta["level"] == "deepen" and m1.transient is True
    m2 = await iv.act([_assistant("topic A", 0), _assistant("topic A", 1)])
    assert m2.meta["level"] == "innovate" and m2.transient is True
    m3 = await iv.act(
        [_assistant("topic A", 0), _assistant("topic A", 1), _assistant("topic A", 2)]
    )
    assert m3.meta["level"] == "inject" and m3.transient is False


async def test_intervention_visibility_transient():
    iv = _intervention(visibility="transient")
    m1 = await iv.act([_assistant("topic A", 0)])
    assert m1.transient is True


async def test_intervention_no_assistant_is_noop():
    iv = _intervention()
    assert await iv.act([]) is None


async def test_intervention_inject_falls_back_on_search_failure():
    class _BrokenSearch:
        async def search(self, query, top_k):
            raise RuntimeError("boom")

    iv = _intervention(search=_BrokenSearch())
    await iv.act([_assistant("topic A", 0)])
    await iv.act([_assistant("topic A", 0), _assistant("topic A", 1)])
    result = await iv.act(
        [_assistant("topic A", 0), _assistant("topic A", 1), _assistant("topic A", 2)]
    )
    assert result.meta["level"] == "inject"
    assert "fallback_reason" in result.meta
    assert "topic switch" in result.content
