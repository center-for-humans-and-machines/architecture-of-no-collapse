"""The run loop: agents take turns; interventions append to shared history.

The loop holds one canonical ``history`` list. Prompt construction is the
agent's job (system prompt + full shared history). Before the first agent turn
each active intervention may prime the conversation via its optional ``prime``
hook (see ``_prime``); after every agent turn, each active intervention runs in
config order and any returned message is appended. A message marked
``transient`` is instead kept in a separate
prompt-scoped buffer: it is fed to the next agent's prompt and then dropped,
never entering canonical history. The recorder receives every turn and
intervention so a run leaves a complete transcript and event stream.

Agents need not see the whole history: when ``memory_turns`` is set, each
prompt is built from only the most recent turns (see ``_recent``). The
canonical history and recorder still keep everything.
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
        memory_turns: int | None = None,
    ) -> None:
        self.agents = agents
        self.scheduler = scheduler
        self.interventions = interventions
        self.recorder = recorder
        self.window_size = window_size
        self.on_window = on_window
        self.memory_turns = memory_turns
        self.history: list[Message] = []
        self.transient: list[Message] = []

    async def run(self, rounds: int) -> None:
        await self._prime()
        for round_index in range(rounds):
            for position, agent_index in enumerate(self.scheduler.order()):
                turn_index = self.scheduler.next_turn()
                agent = self.agents[agent_index]
                context = [*self._recent(self.history), *self.transient]
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

    def _recent(self, messages: list[Message]) -> list[Message]:
        """Return the slice of ``messages`` an agent is allowed to remember.

        Forgetting keeps the last ``memory_turns`` agent (assistant) turns plus
        any intervention messages attached to them — interventions share the
        ``turn_index`` of the turn they follow, so a single cutoff preserves
        them together. ``None`` disables forgetting. Canonical history is never
        truncated; only the prompt context is.
        """
        if self.memory_turns is None:
            return messages
        assistants = [m for m in messages if m.role == "assistant"]
        if len(assistants) <= self.memory_turns:
            return messages
        cutoff = assistants[-self.memory_turns].turn_index
        return [m for m in messages if m.turn_index >= cutoff]

    async def _prime(self) -> None:
        """Run each intervention's optional pre-turn-0 opening hook.

        An intervention may define ``prime`` to seed the conversation before any
        agent speaks (e.g. the scaffolders inject a starting topic). The returned
        message is recorded and appended exactly like a post-turn intervention:
        ``transient`` openings are prompt-scoped, persistent ones enter canonical
        history. Interventions without a ``prime`` hook are skipped, so the loop
        stays compatible with older and custom interventions.
        """
        for intervention in self.interventions:
            prime = getattr(intervention, "prime", None)
            if prime is None:
                continue
            extra = await prime()
            if extra is None:
                continue
            self.recorder.intervention(extra)
            if extra.transient:
                self.transient.append(extra)
            else:
                self.history.append(extra)

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
