"""Random scaffolder intervention tests (offline: fake sampler, fake search)."""

from __future__ import annotations

import pytest

from renewal.core.message import Message
from renewal.interventions.random_scaffolder import (
    DEEPEN_PROMPTS,
    INNOVATE_PROMPTS,
    Action,
    CommonWordSampler,
    ConstantSchedule,
    Distribution,
    RandomPolicy,
    RandomScaffolderIntervention,
    SearchResult,
    TopicInfuser,
    TopicUnavailable,
)


# --- distribution ----------------------------------------------------------


def test_distribution_accepts_valid_triple():
    distribution = Distribution(deepen=0.8, innovate=0.15, inject=0.05)
    assert distribution.as_dict() == {
        "deepen": 0.8,
        "innovate": 0.15,
        "inject": 0.05,
    }


@pytest.mark.parametrize(
    "triple",
    [
        (0.5, 0.4, 0.4),   # sums to 1.3
        (0.2, 0.2, 0.2),   # sums to 0.6
        (-0.1, 0.6, 0.5),  # negative
    ],
)
def test_distribution_rejects_invalid(triple):
    with pytest.raises(ValueError):
        Distribution(*triple)


# --- policy ----------------------------------------------------------------


def _schedule(deepen, innovate, inject):
    return ConstantSchedule(Distribution(deepen, innovate, inject))


def test_policy_is_seeded_and_reproducible():
    a = RandomPolicy(_schedule(0.6, 0.3, 0.1), seed=7)
    b = RandomPolicy(_schedule(0.6, 0.3, 0.1), seed=7)
    assert [a.decide(t) for t in range(30)] == [b.decide(t) for t in range(30)]


def test_policy_deterministic_actions():
    assert all(
        RandomPolicy(_schedule(1.0, 0.0, 0.0), seed=0).decide(t) == Action.DEEPEN
        for t in range(10)
    )
    assert all(
        RandomPolicy(_schedule(0.0, 1.0, 0.0), seed=0).decide(t) == Action.INNOVATE
        for t in range(10)
    )
    assert all(
        RandomPolicy(_schedule(0.0, 0.0, 1.0), seed=0).decide(t) == Action.INJECT
        for t in range(10)
    )


def test_policy_draw_is_independent_each_turn():
    policy = RandomPolicy(_schedule(0.4, 0.3, 0.3), seed=1)
    seen = {policy.decide(t) for t in range(200)}
    assert seen == {Action.DEEPEN, Action.INNOVATE, Action.INJECT}


def test_policy_draw_reports_choice():
    policy = RandomPolicy(_schedule(1.0, 0.0, 0.0), seed=0)
    action, distribution, draw = policy.draw(5)
    assert action == Action.DEEPEN
    assert distribution == Distribution(1.0, 0.0, 0.0)
    assert 0.0 <= draw < 1.0


def test_policy_rotates_prompts():
    policy = RandomPolicy(_schedule(1.0, 0.0, 0.0), seed=0)
    deepen = [policy.prompt(Action.DEEPEN) for _ in range(4)]
    innovate = [policy.prompt(Action.INNOVATE) for _ in range(4)]
    assert deepen == list(DEEPEN_PROMPTS)
    assert innovate == list(INNOVATE_PROMPTS)
    assert policy.prompt(Action.DEEPEN) == DEEPEN_PROMPTS[0]


def test_policy_inject_has_no_fixed_prompt():
    policy = RandomPolicy(_schedule(1.0, 0.0, 0.0), seed=0)
    with pytest.raises(ValueError):
        policy.prompt(Action.INJECT)


def test_policy_accepts_custom_prompt_families():
    policy = RandomPolicy(
        _schedule(1.0, 0.0, 0.0),
        seed=0,
        deepen_prompts=["custom one", "custom two"],
        innovate_prompts=["custom innovation"],
    )
    assert [policy.prompt(Action.DEEPEN) for _ in range(3)] == [
        "custom one",
        "custom two",
        "custom one",
    ]
    assert policy.prompt(Action.INNOVATE) == "custom innovation"


def test_policy_rejects_empty_custom_prompts():
    with pytest.raises(ValueError):
        RandomPolicy(_schedule(1.0, 0.0, 0.0), seed=0, deepen_prompts=[])


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
    client = _FakeSearch(
        {
            "mars terraforming colony": [
                SearchResult("Title", "https://example.com/x", text, 0.9)
            ]
        }
    )
    infuser = TopicInfuser(
        _FakeSampler(), client, num_words=3, max_excerpt_chars=30
    )
    topic = await infuser.inject()
    assert topic.excerpt == "First sentence here."


# --- intervention ----------------------------------------------------------


def _intervention(*, visibility="all", probabilities=None, seed=0, search=None):
    return RandomScaffolderIntervention(
        probabilities=probabilities
        or {"deepen": 0.0, "innovate": 0.0, "inject": 1.0},
        visibility=visibility,
        seed=seed,
        sampler=_FakeSampler(),
        search_client=search
        or _FakeSearch(
            {
                "mars terraforming colony": [
                    SearchResult("Mars", "https://example.com/m", "content", 0.9)
                ]
            }
        ),
    )


def _assistant(text, turn):
    return Message("assistant", f"agent_{turn}", text, turn)


async def test_intervention_inject_path():
    iv = _intervention()
    message = await iv.act([_assistant("topic A", 0)])
    assert message.meta["level"] == "inject"
    assert message.meta["source_url"] == "https://example.com/m"
    # The excerpt is surfaced; the title and source URL are not.
    assert "content" in message.content
    assert "Mars" not in message.content
    assert "example.com" not in message.content
    assert message.meta["probabilities"] == {
        "deepen": 0.0,
        "innovate": 0.0,
        "inject": 1.0,
    }
    assert 0.0 <= message.meta["draw"] < 1.0


async def test_intervention_deepen_records_probabilities():
    iv = _intervention(
        probabilities={"deepen": 1.0, "innovate": 0.0, "inject": 0.0}
    )
    message = await iv.act([_assistant("topic A", 0)])
    assert message.meta["level"] == "deepen"
    assert message.meta["probabilities"] == {
        "deepen": 1.0,
        "innovate": 0.0,
        "inject": 0.0,
    }


async def test_intervention_visibility_injections():
    iv = _intervention(
        visibility="injections",
        probabilities={"deepen": 1.0, "innovate": 0.0, "inject": 0.0},
    )
    message = await iv.act([_assistant("topic A", 0)])
    assert message.meta["level"] == "deepen" and message.transient is True


async def test_intervention_visibility_transient():
    iv = _intervention(visibility="transient")
    message = await iv.act([_assistant("topic A", 0)])
    assert message.transient is True


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
    iv = RandomScaffolderIntervention(
        open_with_topic=False,
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
    result = await iv.act([_assistant("topic A", 0)])
    assert result.meta["level"] == "inject"
    assert "fallback_reason" in result.meta
    assert "topic switch" in result.content


def test_intervention_accepts_distribution_object():
    iv = RandomScaffolderIntervention(
        probabilities=Distribution(1.0, 0.0, 0.0),
        sampler=_FakeSampler(),
        search_client=_FakeSearch(),
    )
    assert iv.distribution == Distribution(1.0, 0.0, 0.0)


def test_intervention_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ValueError):
        RandomScaffolderIntervention(
            probabilities={"deepen": 0.5, "innovate": 0.5, "inject": 0.5},
            sampler=_FakeSampler(),
            search_client=_FakeSearch(),
        )


async def test_intervention_loads_deepen_prompts_from_yaml(tmp_path):
    path = tmp_path / "deepen.yaml"
    path.write_text(
        "- id: a\n"
        "  text: What could we assume instead?\n"
        "- id: b\n"
        "  text: What evidence supports this?\n",
        encoding="utf-8",
    )
    iv = RandomScaffolderIntervention(
        probabilities={"deepen": 1.0, "innovate": 0.0, "inject": 0.0},
        deepen_prompts=path,
        sampler=_FakeSampler(),
        search_client=_FakeSearch(),
    )
    first = await iv.act([_assistant("topic A", 0)])
    second = await iv.act([_assistant("topic B", 1)])
    assert first.meta["level"] == "deepen"
    assert first.content == "What could we assume instead?"
    assert second.content == "What evidence supports this?"


async def test_intervention_accepts_inline_deepen_prompts():
    iv = RandomScaffolderIntervention(
        probabilities={"deepen": 1.0, "innovate": 0.0, "inject": 0.0},
        deepen_prompts=["Only prompt."],
        sampler=_FakeSampler(),
        search_client=_FakeSearch(),
    )
    message = await iv.act([_assistant("topic A", 0)])
    assert message.content == "Only prompt."


def test_intervention_rejects_empty_deepen_source():
    with pytest.raises(ValueError):
        RandomScaffolderIntervention(
            deepen_prompts=[],
            sampler=_FakeSampler(),
            search_client=_FakeSearch(),
        )


# --- run-seed wiring -------------------------------------------------------


def test_intervention_opts_into_run_seed():
    assert RandomScaffolderIntervention.uses_run_seed is True


def test_build_interventions_injects_run_seed():
    from types import SimpleNamespace

    from renewal.config import ensure_builtins
    from renewal.run import build_interventions

    ensure_builtins()
    config = SimpleNamespace(
        interventions=[
            SimpleNamespace(
                type="random_scaffolder",
                options={
                    "probabilities": {"deepen": 0.5, "innovate": 0.3, "inject": 0.2}
                },
            ),
            SimpleNamespace(type="noop", options={}),
        ]
    )
    interventions = build_interventions(config, seed=123)
    seed_set = RandomScaffolderIntervention(
        probabilities={"deepen": 0.5, "innovate": 0.3, "inject": 0.2},
        seed=123,
    )
    assert [interventions[0]._policy.decide(t) for t in range(10)] == [
        seed_set._policy.decide(t) for t in range(10)
    ]


# --- word sampler ----------------------------------------------------------


def test_common_word_sampler_returns_distinct_words():
    sampler = CommonWordSampler(seed=0)
    words = sampler.sample(3)
    assert len(words) == 3
    assert len(set(words)) == 3


def test_common_word_sampler_uses_injected_words():
    sampler = CommonWordSampler(seed=0, words=["apple", "banana", "cherry"])
    assert set(sampler.sample(3)) == {"apple", "banana", "cherry"}


def test_common_word_sampler_rejects_bad_count():
    with pytest.raises(ValueError):
        CommonWordSampler(seed=0).sample(0)

