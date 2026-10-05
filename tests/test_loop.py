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
