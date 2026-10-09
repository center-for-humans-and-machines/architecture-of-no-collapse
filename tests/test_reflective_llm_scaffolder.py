"""Reflective LLM scaffolder tests (offline: fake LLM, sampler, search)."""

from __future__ import annotations

import pytest

from renewal.core.message import Message
from renewal.interventions.base import HistoryReplacement
from renewal.interventions.reflective_llm_scaffolder import (
    DEFAULT_CONDENSE_PROMPT,
    DEFAULT_DEEPEN_PROMPT,
    DEFAULT_INNOVATE_PROMPT,
    DEFAULT_SUMMARIZE_PROMPT,
    Action,
    ConstantSchedule,
    Distribution,
    RandomPolicy,
    ReflectiveLLMScaffolderIntervention,
    SearchResult,
    TopicInfuser,
    TopicUnavailable,
)
from renewal.llm.fake import FakeLLM


# --- helpers ---------------------------------------------------------------


class ScriptedLLM(FakeLLM):
    """Return queued outputs and record every prompt it is given."""

    def __init__(self, outputs=None):
        super().__init__()
        self.outputs = list(outputs or [])
        self.calls = []

    async def generate(self, messages, params):
        self.calls.append(([m.to_dict() for m in messages], dict(params)))
        if self.outputs:
            return self.outputs.pop(0)
        return "scripted"


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


_SOURCE = "A long source body about terraforming Mars."


def _search(raw=True, content=_SOURCE):
    return _FakeSearch(
        {
            "mars terraforming colony": [
                SearchResult(
                    "Mars",
                    "https://example.com/m",
                    content,
                    0.9,
                    raw_content=content if raw else None,
                )
            ]
        }
    )


def _assistant(text, turn):
    return Message("assistant", f"agent_{turn}", text, turn)


def _intervention(
    *, llm=None, probabilities=None, visibility="all", search=None, **overrides
):
    options = dict(
        probabilities=probabilities
        or {"deepen": 0.0, "innovate": 0.0, "inject": 1.0},
        visibility=visibility,
        seed=0,
        llm_client=llm or ScriptedLLM(["SCRIPT"]),
        sampler=_FakeSampler(),
        search_client=search or _search(),
    )
    options.update(overrides)
    return ReflectiveLLMScaffolderIntervention(**options)


_ALL_DEEPEN = {"deepen": 1.0, "innovate": 0.0, "inject": 0.0}
_ALL_INNOVATE = {"deepen": 0.0, "innovate": 1.0, "inject": 0.0}


# --- distribution / policy -------------------------------------------------


def test_distribution_roundtrips():
    distribution = Distribution(deepen=0.8, innovate=0.15, inject=0.05)
    assert distribution.as_dict() == {
        "deepen": 0.8,
        "innovate": 0.15,
        "inject": 0.05,
    }


@pytest.mark.parametrize("triple", [(0.5, 0.4, 0.4), (-0.1, 0.6, 0.5)])
def test_distribution_rejects_invalid(triple):
    with pytest.raises(ValueError):
        Distribution(*triple)


def test_policy_is_seeded_and_reproducible():
    def schedule():
        return ConstantSchedule(Distribution(0.6, 0.3, 0.1))

    a = RandomPolicy(schedule(), seed=7)
    b = RandomPolicy(schedule(), seed=7)
    assert [a.decide(t) for t in range(30)] == [b.decide(t) for t in range(30)]


# --- deep / innovate -------------------------------------------------------


async def test_deepen_uses_llm_output_as_steer():
    llm = ScriptedLLM(["DEEPEN STEER"])
    iv = _intervention(llm=llm, probabilities=_ALL_DEEPEN)
    message = await iv.act([_assistant("topic A", 0)])

    assert message.content == "DEEPEN STEER"
    assert message.role == "intervention"
    assert message.meta["level"] == "deepen"
    # The default deepen system prompt is used and the last message is the input.
    system, user = llm.calls[0][0]
    assert system["content"] == DEFAULT_DEEPEN_PROMPT
    assert user["role"] == "user" and "topic A" in user["content"]


async def test_innovate_uses_its_own_prompt():
    llm = ScriptedLLM(["INNOVATE STEER"])
    iv = _intervention(llm=llm, probabilities=_ALL_INNOVATE)
    message = await iv.act([_assistant("topic A", 0)])

    assert message.content == "INNOVATE STEER"
    assert message.meta["level"] == "innovate"
    assert llm.calls[0][0][0]["content"] == DEFAULT_INNOVATE_PROMPT


async def test_deepen_history_turns_controls_visible_messages():
    llm = ScriptedLLM(["STEER"])
    iv = _intervention(
        llm=llm,
        probabilities=_ALL_DEEPEN,
        deepen={"enabled": True, "history_turns": 2},
    )
    await iv.act([_assistant("first", 0), _assistant("second", 1)])

    user = llm.calls[0][0][1]["content"]
    assert "first" in user and "second" in user


async def test_custom_system_prompt_overrides_default():
    llm = ScriptedLLM(["STEER"])
    iv = _intervention(
        llm=llm,
        probabilities=_ALL_DEEPEN,
        deepen={"system_prompt": "BE CURIOUS"},
    )
    await iv.act([_assistant("topic A", 0)])
    assert llm.calls[0][0][0]["content"] == "BE CURIOUS"


async def test_disabled_method_is_a_noop():
    iv = _intervention(probabilities=_ALL_DEEPEN, deepen={"enabled": False})
    assert await iv.act([_assistant("topic A", 0)]) is None


async def test_unknown_method_option_is_rejected():
    with pytest.raises(ValueError):
        _intervention(deepen={"bogus": True})


async def test_visibility_injections_makes_steers_transient():
    iv = _intervention(
        probabilities=_ALL_DEEPEN,
        visibility="injections",
    )
    message = await iv.act([_assistant("topic A", 0)])
    assert message.transient is True


async def test_no_assistant_is_noop():
    assert await _intervention().act([]) is None


# --- summarize -------------------------------------------------------------


async def test_summarize_replaces_raw_source_in_message():
    llm = ScriptedLLM(["CONDENSED SOURCE", "RECAP"])
    iv = _intervention(llm=llm)
    result = await iv.act([_assistant("topic A", 0)])

    assert isinstance(result, HistoryReplacement)
    topic = result.messages[-1]
    assert "CONDENSED SOURCE" in topic.content
    assert _SOURCE not in topic.content
    assert topic.meta["summarized"] is True
    # The summarize call ran first (before the condense call).
    assert llm.calls[0][0][0]["content"].startswith("Summarize the following text")


async def test_summarize_targets_loop_max_tokens():
    llm = ScriptedLLM(["S", "R"])
    iv = _intervention(llm=llm, loop_max_tokens=333)
    await iv.act([_assistant("topic A", 0)])

    system, _ = llm.calls[0][0]
    assert "333" in system["content"]
    assert "{max_tokens}" not in system["content"]
    assert llm.calls[0][1]["max_tokens"] == 333


async def test_summarize_disabled_keeps_raw_source_and_skips_llm():
    llm = ScriptedLLM()
    iv = _intervention(
        llm=llm,
        summarize={"enabled": False},
        condense={"enabled": False},
    )
    message = await iv.act([_assistant("topic A", 0)])

    assert _SOURCE in message.content
    assert message.meta["summarized"] is False
    assert llm.calls == []


async def test_summarize_failure_falls_back_to_raw_source():
    class _BrokenLLM(ScriptedLLM):
        async def generate(self, messages, params):
            raise RuntimeError("boom")

    iv = _intervention(
        llm=_BrokenLLM(),
        condense={"enabled": False},
    )
    message = await iv.act([_assistant("topic A", 0)])

    assert _SOURCE in message.content
    assert "summarize_fallback_reason" in message.meta


# --- condense (history replacement) ----------------------------------------


async def test_inject_condenses_and_replaces_history():
    llm = ScriptedLLM(["SUMMARY", "RECAP"])
    iv = _intervention(llm=llm)
    result = await iv.act([_assistant("first", 0), _assistant("second", 1)])

    assert isinstance(result, HistoryReplacement)
    assert result.since == 0
    assert len(result.messages) == 2
    recap, topic = result.messages
    assert recap.content == "RECAP"
    assert recap.meta["condense"] is True
    assert recap.meta["since_turn"] == 0
    assert topic.meta["level"] == "inject"


async def test_condense_uses_custom_prompt_and_window():
    llm = ScriptedLLM(["SUMMARY", "RECAP"])
    iv = _intervention(
        llm=llm,
        condense={"system_prompt": "RECAP THIS", "history_turns": 1},
    )
    await iv.act([_assistant("first", 0), _assistant("second", 1)])

    # calls[1] is the condense call; only the last message is visible.
    system, user = llm.calls[1][0]
    assert system["content"] == "RECAP THIS"
    assert "second" in user["content"] and "first" not in user["content"]


async def test_condense_disabled_returns_plain_topic_message():
    llm = ScriptedLLM(["SUMMARY"])
    iv = _intervention(llm=llm, condense={"enabled": False})
    result = await iv.act([_assistant("first", 0)])

    assert isinstance(result, Message)
    assert "SUMMARY" in result.content
    assert len(llm.calls) == 1  # summarize only


async def test_second_topic_only_condenses_since_last_topic():
    llm = ScriptedLLM(["S1", "R1", "S2", "R2"])
    iv = _intervention(llm=llm)

    first = await iv.act([_assistant("a", 0)])
    assert first.since == 0

    # Simulate the loop applying the replacement: recap + topic at turn 0.
    history = list(first.messages)
    history.append(_assistant("later", 1))
    second = await iv.act(history)

    assert isinstance(second, HistoryReplacement)
    assert second.since == 1
    condense_input = llm.calls[3][0][1]["content"]
    assert "later" in condense_input
    assert "R1" not in condense_input  # the previous recap is before the boundary


async def test_search_failure_falls_back_without_condensing():
    class _BrokenSearch:
        async def search(self, query, top_k):
            raise RuntimeError("boom")

    iv = _intervention(search=_BrokenSearch(), condense={"enabled": False})
    message = await iv.act([_assistant("topic A", 0)])

    assert "fallback_reason" in message.meta
    assert "topic switch" in message.content


# --- prime -----------------------------------------------------------------


async def test_prime_infuses_summarized_opening():
    llm = ScriptedLLM(["OPENING SOURCE"])
    iv = _intervention(llm=llm)
    message = await iv.prime()

    assert message is not None
    assert message.role == "intervention"
    assert message.meta["level"] == "inject"
    assert message.meta["opening"] is True
    assert message.turn_index == -1
    assert "OPENING SOURCE" in message.content
    assert message.content.startswith("Opening topic:")
    assert len(llm.calls) == 1  # no condense at prime time


async def test_prime_disabled_is_noop():
    iv = _intervention(open_with_topic=False)
    assert await iv.prime() is None


# --- infuser ---------------------------------------------------------------


async def test_infuser_returns_full_raw_content():
    long_text = "sentence. " * 2000
    client = _FakeSearch(
        {
            "mars terraforming colony": [
                SearchResult(
                    "T",
                    "https://example.com/y",
                    "snippet",
                    0.5,
                    raw_content=long_text,
                )
            ]
        }
    )
    infuser = TopicInfuser(_FakeSampler(), client, num_words=3)
    topic = await infuser.inject()
    assert topic.excerpt == long_text


async def test_infuser_raises_when_no_results():
    infuser = TopicInfuser(_FakeSampler(), _FakeSearch({}), num_words=3)
    with pytest.raises(TopicUnavailable):
        await infuser.inject()


# --- harness wiring --------------------------------------------------------


def test_intervention_opts_into_run_hooks():
    assert ReflectiveLLMScaffolderIntervention.uses_run_seed is True
    assert ReflectiveLLMScaffolderIntervention.uses_loop_max_tokens is True


def test_build_interventions_injects_seed_and_loop_max_tokens():
    from types import SimpleNamespace

    from renewal.config import ensure_builtins
    from renewal.run import build_interventions

    ensure_builtins()
    config = SimpleNamespace(
        interventions=[
            SimpleNamespace(
                type="reflective_llm_scaffolder",
                options={
                    "probabilities": {"deepen": 0.5, "innovate": 0.3, "inject": 0.2}
                },
            ),
            SimpleNamespace(type="noop", options={}),
        ],
        agents=SimpleNamespace(params=SimpleNamespace(max_tokens=321)),
    )
    interventions = build_interventions(config, seed=7)
    assert interventions[0].loop_max_tokens == 321
    assert isinstance(interventions[0]._policy, RandomPolicy)


def test_default_prompts_are_exported():
    assert "deepen the topic" in DEFAULT_DEEPEN_PROMPT
    assert "novel angle" in DEFAULT_INNOVATE_PROMPT
    assert "{max_tokens}" in DEFAULT_SUMMARIZE_PROMPT
    assert "previous conversation" in DEFAULT_CONDENSE_PROMPT


async def test_loop_applies_condense_replacement():
    from renewal.core.agent import Agent
    from renewal.core.loop import Loop
    from renewal.core.scheduler import Scheduler
    from renewal.llm.fake import FakeLLM

    class _Recorder:
        def __init__(self):
            self.condensed = []

        def turn(self, message):
            pass

        def intervention(self, message):
            pass

        def condense(self, *, since_turn, removed_turns):
            self.condensed.append((since_turn, list(removed_turns)))

    recorder = _Recorder()
    llm = ScriptedLLM(["S0", "R0", "S1", "R1", "S2", "R2"])
    iv = _intervention(llm=llm, open_with_topic=False)  # inject on every turn
    agent = Agent(name="agent_0", llm=FakeLLM(), system_prompt="SYS", params={})
    loop = Loop([agent], Scheduler(1, 0), [iv], recorder)
    await loop.run(3)

    # Every turn compressed the history-since-last-topic down to recap + topic.
    assert [since for since, _ in recorder.condensed] == [0, 1, 2]
    assert len(loop.history) == 6
    # The full assistant turns are still available to the metric windows.
    assert len(loop.turns) == 3
