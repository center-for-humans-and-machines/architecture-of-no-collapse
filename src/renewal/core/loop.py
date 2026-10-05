"""The run loop: agents take turns; interventions append to shared history.

The loop holds one canonical ``history`` list. Prompt construction is the
agent's job (system prompt + full shared history). After every agent turn,
each active intervention runs in config order and any returned message is
appended. A message marked ``transient`` is instead kept in a separate
prompt-scoped buffer: it is fed to the next agent's prompt and then dropped,
never entering canonical history. The recorder receives every turn and
intervention so a run leaves a complete transcript and event stream.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

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
        *,
        window_size: int | None = None,
        on_window: Callable[[int, list[Message]], Awaitable[None]] | None = None,
    ) -> None:
        self.agents = agents
        self.scheduler = scheduler
        self.interventions = interventions
        self.recorder = recorder
        self.window_size = window_size
        self.on_window = on_window
        self.history: list[Message] = []
        self.transient: list[Message] = []

    async def run(self, rounds: int) -> None:
        for round_index in range(rounds):
            for position, agent_index in enumerate(self.scheduler.order()):
                turn_index = self.scheduler.next_turn()
                agent = self.agents[agent_index]
                context = [*self.history, *self.transient]
                message = await agent.respond(
                    context,
                    turn_index=turn_index,
                    round_index=round_index,
                    position=position,
                )
                self.history.append(message)
                self.recorder.turn(message)
                self.transient.clear()

                for intervention in self.interventions:
                    extra = await intervention.act(self.history)
                    if extra is None:
                        continue
                    self.recorder.intervention(extra)
                    if extra.transient:
                        self.transient.append(extra)
                    else:
                        self.history.append(extra)

            if self.window_size is not None and self.on_window is not None:
                if (round_index + 1) % self.window_size == 0:
                    await self._emit_window(round_index)

        # Emit a final partial window when the round count is not a multiple of
        # the window size, so short runs (rounds < window_size) still yield one
        # window of metrics.
        if (
            self.window_size is not None
            and self.on_window is not None
            and rounds > 0
            and rounds % self.window_size != 0
        ):
            await self._emit_window(rounds - 1)

    async def _emit_window(self, round_index: int) -> None:
        assert self.window_size is not None and self.on_window is not None
        window_index = round_index // self.window_size
        start = window_index * self.window_size
        end = start + self.window_size
        turns = [
            m
            for m in self.history
            if m.role == "assistant" and start <= m.meta.get("round", -1) < end
        ]
        await self.on_window(window_index, turns)
