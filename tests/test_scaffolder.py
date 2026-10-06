"""Scaffolder intervention tests (offline: fake embedder, sampler, search)."""

from __future__ import annotations

import pytest

from renewal.core.message import Message
from renewal.interventions.scaffolder import (
    DEEPEN_PROMPTS,
    INNOVATE_PROMPTS,
    Action,
    CommonWordSampler,
    EscalationPolicy,
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


def test_common_word_sampler_returns_distinct_words():
    sampler = CommonWordSampler(seed=0)
    words = sampler.sample(3)
    assert len(words) == 3
    assert len(set(words)) == 3
    assert all(word.isalpha() for word in words)


def test_common_word_sampler_is_deterministic():
    a = CommonWordSampler(seed=1).sample(5)
    b = CommonWordSampler(seed=1).sample(5)
    assert a == b


def test_common_word_sampler_uses_injected_words():
    sampler = CommonWordSampler(seed=0, words=["apple", "banana", "cherry"])
    assert set(sampler.sample(3)) == {"apple", "banana", "cherry"}


def test_common_word_sampler_rejects_bad_count():
    with pytest.raises(ValueError):
        CommonWordSampler(seed=0).sample(0)


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


def _long_search(content):
    return _FakeSearch(
        {
            "mars terraforming colony": [
                SearchResult("Title", "https://example.com/x", content, 0.9)
            ]
        }
    )


async def test_infuser_returns_full_excerpt_by_default():
    content = "y" * 1000
    infuser = TopicInfuser(_FakeSampler(), _long_search(content), num_words=3)
    topic = await infuser.inject()
    assert topic.excerpt == content


async def test_infuser_truncates_excerpt_when_configured():
    content = "y" * 1000
    infuser = TopicInfuser(
        _FakeSampler(), _long_search(content), num_words=3, max_excerpt_chars=10
    )
    topic = await infuser.inject()
    assert topic.excerpt == "y" * 10


async def test_infuser_prefers_raw_content_over_snippet():
    client = _FakeSearch(
        {
            "mars terraforming colony": [
                SearchResult(
                    "Title",
                    "https://example.com/x",
                    "short snippet",
                    0.9,
                    raw_content="A much longer full page body.",
                )
            ]
        }
    )
    infuser = TopicInfuser(_FakeSampler(), client, num_words=3)
    topic = await infuser.inject()
    assert topic.excerpt == "A much longer full page body."


async def test_infuser_truncates_at_sentence_boundary():
    text = "First sentence here. Second sentence here. Third sentence here."
    infuser = TopicInfuser(
        _FakeSampler(), _long_search(text), num_words=3, max_excerpt_chars=30
    )
    topic = await infuser.inject()
    assert topic.excerpt == "First sentence here."


async def test_infuser_truncates_at_word_boundary_without_sentence():
    text = "one two three four five six seven"
    infuser = TopicInfuser(
        _FakeSampler(), _long_search(text), num_words=3, max_excerpt_chars=10
    )
    topic = await infuser.inject()
    assert topic.excerpt == "one two"


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
    # The excerpt is surfaced; the title and source URL are not.
    assert "content" in m3.content
    assert "Mars" not in m3.content
    assert "example.com" not in m3.content


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


async def test_intervention_prime_injects_opening_topic():
    iv = _intervention()
    message = await iv.prime()
    assert message is not None
    assert message.role == "intervention"
    assert message.meta["level"] == "inject"
    assert message.meta["opening"] is True
    assert message.meta["source_url"] == "https://example.com/m"
    assert message.turn_index == -1
    assert message.transient is False
    assert message.content.startswith("Opening topic:")
    assert "Start the conversation" in message.content


async def test_intervention_prime_disabled_is_noop():
    iv = ScaffolderIntervention(
        open_with_topic=False,
        embedder=FakeEmbedding(dim=64),
        sampler=_FakeSampler(),
        search_client=_FakeSearch(),
    )
    assert await iv.prime() is None


async def test_intervention_prime_falls_back_on_search_failure():
    class _BrokenSearch:
        async def search(self, query, top_k):
            raise RuntimeError("boom")

    iv = _intervention(search=_BrokenSearch())
    message = await iv.prime()
    assert message.meta["opening"] is True
    assert "fallback_reason" in message.meta
    assert "Start the conversation" in message.content


async def test_intervention_prime_honors_transient_visibility():
    iv = _intervention(visibility="transient")
    message = await iv.prime()
    assert message.transient is True


async def test_intervention_prime_persists_under_injections_visibility():
    iv = _intervention(visibility="injections")
    message = await iv.prime()
    assert message.transient is False


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
