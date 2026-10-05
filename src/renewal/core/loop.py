"""The run loop: agents take turns; interventions append to shared history.

The loop holds one canonical ``history`` list. Prompt construction is the
agent's job (system prompt + full shared history). After every agent turn,
each active intervention runs in config order and any returned message is
appended to history. The recorder receives every turn and intervention so a
run leaves a complete transcript and event stream.
"""

from __future__ import annotations

import logging

from renewal.core.agent import Agent
from renewal.core.message import Message
from renewal.core.scheduler import Scheduler
from renewal.interventions.base import Intervention

LOGGER = logging.getLogger(__name__)


class Loop:
    def __init__(
        self,
        agents: list[Agent],
        scheduler: Scheduler,
        interventions: list[Intervention],
        recorder: object,
    ) -> None:
        self.agents = agents
        self.scheduler = scheduler
        self.interventions = interventions
        self.recorder = recorder
        self.history: list[Message] = []

    async def run(self, rounds: int) -> None:
        for round_index in range(rounds):
            for position, agent_index in enumerate(self.scheduler.order()):
                turn_index = self.scheduler.next_turn()
                agent = self.agents[agent_index]
                message = await agent.respond(
                    self.history,
                    turn_index=turn_index,
                    round_index=round_index,
                    position=position,
                )
                self.history.append(message)
                self.recorder.turn(message)

                for intervention in self.interventions:
                    extra = await intervention.act(self.history)
                    if extra is not None:
                        self.history.append(extra)
                        self.recorder.intervention(extra)
