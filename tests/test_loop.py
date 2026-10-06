"""Loop behavior tests using the deterministic fake model."""

from collections import Counter

from renewal.core.agent import Agent
from renewal.core.loop import Loop
from renewal.core.message import Message
from renewal.core.scheduler import Scheduler
from renewal.llm.fake import FakeLLM


class NullRecorder:
    def __init__(self):
        self.turns = []
        self.interventions = []

    def turn(self, message):
        self.turns.append(message)

    def intervention(self, message):
        self.interventions.append(message)


def _agents(n=3):
    return [
        Agent(name=f"agent_{i}", llm=FakeLLM(), system_prompt="p", params={})
        for i in range(n)
    ]


async def test_loop_rounds_and_alternation():
    agents = _agents(3)
    recorder = NullRecorder()
    loop = Loop(agents, Scheduler(3, 0), [], recorder)
    await loop.run(4)

    assert len(loop.history) == 12
    assert len(recorder.turns) == 12
    assert Counter(m.speaker for m in loop.history) == {
        "agent_0": 4,
        "agent_1": 4,
        "agent_2": 4,
    }
    assert all(m.role == "assistant" for m in loop.history)
    assert [m.turn_index for m in loop.history] == list(range(12))


class AppendIntervention:
    name = "append"

    async def act(self, messages):
        return Message("intervention", "append", f"note {len(messages)}", 0)


class TransientIntervention:
    name = "transient"

    async def act(self, messages):
        return Message(
            "intervention", "transient", "nudge", len(messages), transient=True
        )


async def test_intervention_appends_to_history():
    agents = _agents(2)
    recorder = NullRecorder()
    loop = Loop(agents, Scheduler(2, 0), [AppendIntervention()], recorder)
    await loop.run(2)

    # 2 rounds * 2 agents = 4 turns + 4 interventions
    assert len(loop.history) == 8
    assert len(recorder.interventions) == 4
    assert sum(1 for m in loop.history if m.role == "intervention") == 4


async def test_transient_intervention_prompted_but_not_stored():
    captured = {}

    class CapturingLLM(FakeLLM):
        async def generate(self, messages, params):
            captured["contents"] = [m.content for m in messages]
            return "ok"

    agent = Agent(name="agent_0", llm=CapturingLLM(), system_prompt="SYS", params={})
    recorder = NullRecorder()
    loop = Loop([agent], Scheduler(1, 0), [TransientIntervention()], recorder)
    await loop.run(2)

    # The transient nudge never enters canonical history...
    assert len(loop.history) == 2
    assert all(m.role == "assistant" for m in loop.history)
    # ...but it did reach the next agent's prompt and the recorder.
    assert "nudge" in captured["contents"]
    assert len(recorder.interventions) == 2


async def test_prompt_includes_system_and_history():
    captured = {}

    class CapturingLLM(FakeLLM):
        async def generate(self, messages, params):
            captured["roles"] = [m.role for m in messages]
            captured["speakers"] = [m.speaker for m in messages]
            return "ok"

    agent = Agent(name="agent_0", llm=CapturingLLM(), system_prompt="SYS", params={})
    recorder = NullRecorder()
    loop = Loop([agent], Scheduler(1, 0), [], recorder)
    await loop.run(1)

    assert captured["roles"][0] == "system"
    assert captured["speakers"][0] == "system"


class PrimeIntervention:
    name = "prime"

    async def prime(self):
        return Message("intervention", "prime", "opening", -1)

    async def act(self, messages):
        return None


class TransientPrimeIntervention:
    name = "prime"

    async def prime(self):
        return Message("intervention", "prime", "opening", -1, transient=True)

    async def act(self, messages):
        return None


async def test_prime_hook_seeds_before_first_turn():
    captured = {}

    class CapturingLLM(FakeLLM):
        async def generate(self, messages, params):
            captured.setdefault("first", [m.content for m in messages])
            return "ok"

    agent = Agent(name="agent_0", llm=CapturingLLM(), system_prompt="SYS", params={})
    recorder = NullRecorder()
    loop = Loop([agent], Scheduler(1, 0), [PrimeIntervention()], recorder)
    await loop.run(3)

    # The opening is in canonical history before the first agent turn...
    assert loop.history[0].role == "intervention"
    assert loop.history[0].content == "opening"
    assert loop.history[0].turn_index == -1
    # ...reached the very first prompt...
    assert "opening" in captured["first"]
    # ...and was primed exactly once, regardless of the number of rounds.
    assert len(recorder.interventions) == 1


async def test_transient_prime_reaches_first_prompt_then_drops():
    captured = {}

    class CapturingLLM(FakeLLM):
        async def generate(self, messages, params):
            captured.setdefault("first", [m.content for m in messages])
            return "ok"

    agent = Agent(name="agent_0", llm=CapturingLLM(), system_prompt="SYS", params={})
    recorder = NullRecorder()
    loop = Loop([agent], Scheduler(1, 0), [TransientPrimeIntervention()], recorder)
    await loop.run(2)

    assert "opening" in captured["first"]
    # A transient opening is prompt-scoped: it never enters canonical history.
    assert all(m.role == "assistant" for m in loop.history)
    assert len(recorder.interventions) == 1


async def test_interventions_without_prime_are_skipped():
    agents = _agents(2)
    recorder = NullRecorder()
    loop = Loop(agents, Scheduler(2, 0), [AppendIntervention()], recorder)
    await loop.run(1)

    # No prime hook: the loop runs normally (2 turns + 2 post-turn notes).
    assert len(loop.history) == 4
    assert len(recorder.turns) == 2


class CountingLLM(FakeLLM):
    """Return a distinguishable body per turn and record each prompt."""

    def __init__(self):
        super().__init__()
        self.prompts = []

    async def generate(self, messages, params):
        self.prompts.append([m.content for m in messages])
        return f"turn{len(self.prompts) - 1}"


async def test_memory_truncates_prompt_context():
    llm = CountingLLM()
    agent = Agent(name="agent_0", llm=llm, system_prompt="SYS", params={})
    loop = Loop([agent], Scheduler(1, 0), [], NullRecorder(), memory_turns=2)
    await loop.run(4)

    # Turn 3's prompt remembers only the last two turns (turn1, turn2)...
    assert llm.prompts[3] == ["SYS", "turn1", "turn2"]
    # ...while the canonical history still holds every turn.
    assert [m.content for m in loop.history] == [
        "turn0",
        "turn1",
        "turn2",
        "turn3",
    ]


async def test_memory_none_keeps_full_history_in_prompt():
    llm = CountingLLM()
    agent = Agent(name="agent_0", llm=llm, system_prompt="SYS", params={})
    loop = Loop([agent], Scheduler(1, 0), [], NullRecorder(), memory_turns=None)
    await loop.run(3)

    assert llm.prompts[2] == ["SYS", "turn0", "turn1"]


class TagIntervention:
    name = "tag"

    async def prime(self):
        return None

    async def act(self, messages):
        latest = [m for m in messages if m.role == "assistant"][-1]
        return Message(
            "intervention",
            "tag",
            f"tag{latest.turn_index}",
            latest.turn_index,
        )


async def test_memory_keeps_interventions_of_retained_turns():
    llm = CountingLLM()
    agent = Agent(name="agent_0", llm=llm, system_prompt="SYS", params={})
    loop = Loop(
        [agent],
        Scheduler(1, 0),
        [TagIntervention()],
        NullRecorder(),
        memory_turns=1,
    )
    await loop.run(3)

    # Turn 2 keeps only the latest agent turn and its attached intervention;
    # the earlier turn's tag (turn_index 0) is forgotten.
    assert llm.prompts[2] == ["SYS", "turn1", "tag1"]
