"""Memory scaffolder tests (offline: fake LLM, embedder, sampler, search)."""

from __future__ import annotations

import random

import pytest

from renewal.core.message import Message
from renewal.interventions.memory_scaffolder import (
    DEFAULT_REMEMBER_PROMPT,
    DEFAULT_SUMMARIZE_PROMPT,
    RESURFACE_PROMPTS,
    Action,
    ConstantSchedule,
    Distribution,
    Memory,
    MemoryScaffolderIntervention,
    MemoryStore,
    RandomPolicy,
    RandomUnsurfacedSelector,
    SearchResult,
    TopicInfuser,
    TopicUnavailable,
)
from renewal.llm.embed import FakeEmbedding
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


class _RecordingSink:
    def __init__(self):
        self.events = []

    def record_memory(self, event, **fields):
        self.events.append((event, fields))


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
    *,
    llm=None,
    probabilities=None,
    visibility="all",
    search=None,
    embedder=None,
    sink=None,
    **overrides,
):
    options = dict(
        probabilities=probabilities
        or {"deepen": 0.0, "innovate": 0.0, "inject": 1.0, "resurface": 0.0},
        visibility=visibility,
        seed=0,
        llm_client=llm or ScriptedLLM(["CONDENSED", "MEMORY"]),
        sampler=_FakeSampler(),
        search_client=search or _search(),
        embedder=embedder or FakeEmbedding(dim=8),
        memory_sink=sink,
    )
    options.update(overrides)
    return MemoryScaffolderIntervention(**options)


_ALL_DEEPEN = {"deepen": 1.0, "innovate": 0.0, "inject": 0.0, "resurface": 0.0}
_ALL_INNOVATE = {"deepen": 0.0, "innovate": 1.0, "inject": 0.0, "resurface": 0.0}
_ALL_INJECT = {"deepen": 0.0, "innovate": 0.0, "inject": 1.0, "resurface": 0.0}
_ALL_RESURFACE = {"deepen": 0.0, "innovate": 0.0, "inject": 0.0, "resurface": 1.0}


# --- distribution / policy -------------------------------------------------


def test_distribution_roundtrips():
    distribution = Distribution(deepen=0.5, innovate=0.3, inject=0.1, resurface=0.1)
    assert distribution.as_dict() == {
        "deepen": 0.5,
        "innovate": 0.3,
        "inject": 0.1,
        "resurface": 0.1,
    }


@pytest.mark.parametrize(
    "quadruple",
    [
        (0.5, 0.3, 0.3, 0.3),  # sums to 1.4
        (0.2, 0.2, 0.2, 0.2),  # sums to 0.8
        (-0.1, 0.5, 0.3, 0.3),  # negative
    ],
)
def test_distribution_rejects_invalid(quadruple):
    with pytest.raises(ValueError):
        Distribution(*quadruple)


def test_policy_is_seeded_and_reproducible():
    def schedule():
        return ConstantSchedule(Distribution(0.4, 0.3, 0.2, 0.1))

    a = RandomPolicy(schedule(), seed=7)
    b = RandomPolicy(schedule(), seed=7)
    assert [a.decide(t) for t in range(40)] == [b.decide(t) for t in range(40)]


def test_policy_draw_can_select_resurface():
    assert all(
        RandomPolicy(
            ConstantSchedule(Distribution(0.0, 0.0, 0.0, 1.0)), seed=0
        ).decide(t)
        == Action.RESURFACE
        for t in range(10)
    )


def test_policy_rotates_resurface_prompts():
    policy = RandomPolicy(
        ConstantSchedule(Distribution(0.0, 0.0, 0.0, 1.0)), seed=0
    )
    prompts = [policy.prompt(Action.RESURFACE) for _ in range(len(RESURFACE_PROMPTS))]
    assert prompts == list(RESURFACE_PROMPTS)
    assert policy.prompt(Action.RESURFACE) == RESURFACE_PROMPTS[0]


def test_resurface_prompts_carry_the_memory_slot():
    assert 4 <= len(RESURFACE_PROMPTS) <= 5
    assert all("{memory}" in prompt for prompt in RESURFACE_PROMPTS)


@pytest.mark.parametrize("action", [Action.DEEPEN, Action.INNOVATE, Action.INJECT])
def test_policy_has_no_fixed_prompt_for_other_actions(action):
    policy = RandomPolicy(ConstantSchedule(Distribution(1.0, 0.0, 0.0, 0.0)), seed=0)
    with pytest.raises(ValueError):
        policy.prompt(action)


# --- memory store ----------------------------------------------------------


async def test_store_assigns_sequential_ids_and_embeds():
    store = MemoryStore(FakeEmbedding(dim=8))
    first = await store.add("first memory", created_turn=0)
    second = await store.add("second memory", created_turn=3)

    assert (first.id, second.id) == (1, 2)
    assert first.embedding is not None and len(first.embedding) == 8
    assert first.surfaced_count == 0
    assert [m.text for m in store.memories] == ["first memory", "second memory"]
    assert len(store) == 2


async def test_store_reports_created_and_surfaced_events():
    sink = _RecordingSink()
    store = MemoryStore(FakeEmbedding(dim=8), sink=sink)
    memory = await store.add("a memory", created_turn=2, since_turn=0)
    store.mark_surfaced(memory, turn_index=5)

    names = [name for name, _ in sink.events]
    assert names == ["memory_created", "memory_surfaced"]
    created = sink.events[0][1]
    assert created["id"] == 1 and created["text"] == "a memory"
    assert created["embedding"] is not None
    surfaced = sink.events[1][1]
    assert surfaced["id"] == 1
    assert surfaced["surfaced_count"] == 1
    assert surfaced["turn_index"] == 5


def _memory(memory_id, surfaced_count):
    return Memory(
        id=memory_id,
        text=f"memory {memory_id}",
        embedding=None,
        created_turn=memory_id,
        surfaced_count=surfaced_count,
    )


def test_selector_prefers_never_surfaced():
    selector = RandomUnsurfacedSelector()
    memories = [_memory(1, 2), _memory(2, 0), _memory(3, 0)]
    picked = selector.select(
        memories, rng=random.Random(0), turn_index=10
    )
    assert picked.surfaced_count == 0


def test_selector_falls_back_to_least_surfaced():
    selector = RandomUnsurfacedSelector()
    memories = [_memory(1, 4), _memory(2, 1), _memory(3, 1)]
    picked = selector.select(
        memories, rng=random.Random(0), turn_index=10
    )
    assert picked.surfaced_count == 1


def test_selector_returns_none_without_memories():
    selector = RandomUnsurfacedSelector()
    assert selector.select([], rng=random.Random(0), turn_index=0) is None


# --- inject + remember -----------------------------------------------------


async def test_inject_stores_a_memory_and_does_not_replace_history():
    llm = ScriptedLLM(["CONDENSED SOURCE", "MEMORY TEXT"])
    sink = _RecordingSink()
    iv = _intervention(llm=llm, probabilities=_ALL_INJECT, sink=sink)

    result = await iv.act([_assistant("first", 0), _assistant("second", 1)])

    assert isinstance(result, Message)  # never a HistoryReplacement
    assert "CONDENSED SOURCE" in result.content
    assert result.meta["level"] == "inject"
    memories = iv._store.memories
    assert len(memories) == 1
    assert memories[0].text == "MEMORY TEXT"
    assert memories[0].created_turn == 1
    assert [name for name, _ in sink.events] == ["memory_created"]


async def test_remember_targets_loop_max_tokens():
    llm = ScriptedLLM(["S", "M"])
    iv = _intervention(
        llm=llm, probabilities=_ALL_INJECT, loop_max_tokens=333
    )
    await iv.act([_assistant("topic A", 0)])

    # calls[0] is the source summarize; calls[1] is the memory remember.
    system, _ = llm.calls[1][0]
    assert DEFAULT_REMEMBER_PROMPT.replace("{max_tokens}", "333") in system["content"]
    assert llm.calls[1][1]["max_tokens"] == 333


async def test_remember_disabled_skips_memory_creation():
    llm = ScriptedLLM(["CONDENSED SOURCE"])
    iv = _intervention(
        llm=llm,
        probabilities=_ALL_INJECT,
        remember={"enabled": False},
        summarize={"enabled": True},
    )
    await iv.act([_assistant("topic A", 0)])
    assert iv._store.memories == []
    assert len(llm.calls) == 1  # summarize only


async def test_remember_uses_custom_prompt_and_window():
    llm = ScriptedLLM(["CONDENSED", "MEMORY"])
    iv = _intervention(
        llm=llm,
        probabilities=_ALL_INJECT,
        remember={"system_prompt": "REMEMBER THIS", "history_turns": 1},
    )
    await iv.act([_assistant("first", 0), _assistant("second", 1)])

    system, user = llm.calls[1][0]
    assert system["content"].startswith("REMEMBER THIS")
    assert "second" in user["content"] and "first" not in user["content"]


async def test_second_topic_remembers_only_since_last_topic():
    llm = ScriptedLLM(["S1", "M1", "S2", "M2"])
    iv = _intervention(llm=llm, probabilities=_ALL_INJECT)

    first = await iv.act([_assistant("a", 0)])
    # Simulate the loop appending the topic message and a later turn.
    history = [first, _assistant("later", 1)]
    await iv.act(history)

    # The second memory summarizes only the turn(s) after the first topic.
    remember_input = llm.calls[3][0][1]["content"]
    assert "later" in remember_input
    assert "agent_0: a" not in remember_input
    assert [m.id for m in iv._store.memories] == [1, 2]


async def test_prime_creates_no_memory():
    llm = ScriptedLLM(["OPENING SOURCE"])
    iv = _intervention(llm=llm)
    message = await iv.prime()
    assert message is not None
    assert message.meta["opening"] is True
    assert iv._store.memories == []


# --- resurface -------------------------------------------------------------


async def test_resurface_renders_a_memory():
    iv = _intervention(probabilities=_ALL_RESURFACE)
    await iv._store.add("we discussed memory and forgetting", created_turn=4)

    message = await iv.act([_assistant("current topic", 7)])

    assert message.meta["level"] == "resurface"
    assert message.meta["memory_id"] == 1
    assert "we discussed memory and forgetting" in message.content
    assert "{memory}" not in message.content
    assert iv._store.memories[0].surfaced_count == 1


async def test_resurface_no_memories_is_noop():
    iv = _intervention(probabilities=_ALL_RESURFACE)
    assert await iv.act([_assistant("current topic", 0)]) is None


async def test_resurface_marks_surfaced_for_the_sink():
    sink = _RecordingSink()
    iv = _intervention(probabilities=_ALL_RESURFACE, sink=sink)
    await iv._store.add("a memory", created_turn=0)

    await iv.act([_assistant("current", 1)])

    assert [name for name, _ in sink.events] == ["memory_created", "memory_surfaced"]


async def test_resurface_is_transient_under_injections_visibility():
    iv = _intervention(probabilities=_ALL_RESURFACE, visibility="injections")
    await iv._store.add("a memory", created_turn=0)
    message = await iv.act([_assistant("current", 1)])
    assert message.transient is True


async def test_resurface_accepts_custom_prompts():
    iv = _intervention(
        probabilities=_ALL_RESURFACE,
        resurface_prompts=["CUSTOM: {memory}"],
    )
    await iv._store.add("hello memory", created_turn=0)
    message = await iv.act([_assistant("current", 1)])
    assert message.content == "CUSTOM: hello memory"


# --- deep / innovate -------------------------------------------------------


async def test_deepen_uses_llm_output_as_steer():
    llm = ScriptedLLM(["DEEPEN STEER"])
    iv = _intervention(llm=llm, probabilities=_ALL_DEEPEN)
    message = await iv.act([_assistant("topic A", 0)])
    assert message.content == "DEEPEN STEER"
    assert message.meta["level"] == "deepen"


async def test_innovate_uses_llm_output_as_steer():
    llm = ScriptedLLM(["INNOVATE STEER"])
    iv = _intervention(llm=llm, probabilities=_ALL_INNOVATE)
    message = await iv.act([_assistant("topic A", 0)])
    assert message.content == "INNOVATE STEER"
    assert message.meta["level"] == "innovate"


async def test_no_assistant_is_noop():
    assert await _intervention().act([]) is None


def test_intervention_accepts_distribution_object():
    iv = _intervention(probabilities=Distribution(1.0, 0.0, 0.0, 0.0))
    assert iv.distribution == Distribution(1.0, 0.0, 0.0, 0.0)


def test_intervention_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ValueError):
        _intervention(
            probabilities={
                "deepen": 0.5,
                "innovate": 0.5,
                "inject": 0.5,
                "resurface": 0.5,
            }
        )


def test_intervention_rejects_missing_resurface_key():
    with pytest.raises(ValueError):
        _intervention(probabilities={"deepen": 0.5, "innovate": 0.3, "inject": 0.2})


# --- infuser (copied path still works) -------------------------------------


async def test_infuser_raises_when_no_results():
    infuser = TopicInfuser(_FakeSampler(), _FakeSearch({}), num_words=3)
    with pytest.raises(TopicUnavailable):
        await infuser.inject()


# --- harness wiring --------------------------------------------------------


def test_intervention_opts_into_run_hooks():
    assert MemoryScaffolderIntervention.uses_run_seed is True
    assert MemoryScaffolderIntervention.uses_loop_max_tokens is True
    assert MemoryScaffolderIntervention.uses_memory_sink is True


def test_default_prompts_are_exported():
    assert "max_tokens" in DEFAULT_SUMMARIZE_PROMPT
    assert "max_tokens" in DEFAULT_REMEMBER_PROMPT
    assert "memory" in DEFAULT_REMEMBER_PROMPT


def test_build_interventions_injects_seed_tokens_and_sink():
    from types import SimpleNamespace

    from renewal.config import ensure_builtins
    from renewal.run import build_interventions

    ensure_builtins()
    config = SimpleNamespace(
        interventions=[
            SimpleNamespace(
                type="memory_scaffolder",
                options={
                    "probabilities": {
                        "deepen": 0.5,
                        "innovate": 0.3,
                        "inject": 0.1,
                        "resurface": 0.1,
                    }
                },
            ),
            SimpleNamespace(type="noop", options={}),
        ],
        agents=SimpleNamespace(params=SimpleNamespace(max_tokens=321)),
    )
    sink = _RecordingSink()
    interventions = build_interventions(config, seed=7, memory_sink=sink)
    assert interventions[0].loop_max_tokens == 321
    assert interventions[0]._store._sink is sink
    assert isinstance(interventions[0]._policy, RandomPolicy)
