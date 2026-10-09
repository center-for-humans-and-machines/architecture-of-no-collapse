"""Memory-scaffolder policy: a four-way random draw (no I/O, seeded).

Each turn independently samples one of four actions from a fixed probability
distribution:

* ``deepen``    -> ask the scaffolding LLM for a deepening steer
* ``innovate``  -> ask the scaffolding LLM for an innovation steer
* ``inject``    -> infuse a brand-new topic via search
* ``resurface`` -> bring back one earlier memory (see :mod:`.memory`)

Unlike the similarity-driven scaffolder there is no signal and no escalation
state: ``p_t(x)`` is the same distribution on every turn and each draw is
independent. The policy only *picks* the action and, for RESURFACE, *which*
wording from a fixed family; DEEPEN and INNOVATE wording is authored by the
scaffolding LLM (see the intervention layer), and INJECT content is built by the
infuser.

This package is a deliberate copy of
:mod:`renewal.interventions.reflective_llm_scaffolder` so the two can evolve
independently; only the random-draw mechanics are shared in spirit, not in code.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

RESURFACE_PROMPTS = (
    "Here is something from earlier in our conversation:\n\n"
    "{memory}\n\n"
    "Think about how it is relevant to our current topic and bring it back in.",
    "Earlier we explored this:\n\n"
    "{memory}\n\n"
    "Consider how it connects to what we are discussing now.",
    "A memory from before:\n\n"
    "{memory}\n\n"
    "Reflect on how this memory bears on the present topic.",
    "Recall this from earlier:\n\n"
    "{memory}\n\n"
    "Draw a connection between it and where the conversation has gone.",
    "You once summarized this:\n\n"
    "{memory}\n\n"
    "Revisit it and see what it adds to the current discussion.",
)


def _as_prompts(
    level: str, prompts: Sequence[str] | None, fallback: tuple[str, ...]
) -> tuple[str, ...]:
    """Return a non-empty prompt family, falling back to the built-in one."""
    chosen = fallback if prompts is None else tuple(prompts)
    if not chosen:
        raise ValueError(f"{level} prompts must not be empty")
    return chosen


class Action(str, Enum):
    """Observable level selected by the policy."""

    DEEPEN = "deepen"
    INNOVATE = "innovate"
    INJECT = "inject"
    RESURFACE = "resurface"


@dataclass(frozen=True)
class Distribution:
    """A categorical distribution over the four actions.

    The four probabilities must be non-negative and sum to one. The policy
    draws one action from this distribution on every turn.
    """

    deepen: float
    innovate: float
    inject: float
    resurface: float

    def __post_init__(self) -> None:
        values = (self.deepen, self.innovate, self.inject, self.resurface)
        if any(value < 0.0 for value in values):
            raise ValueError("probabilities must be non-negative")
        total = sum(values)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities must sum to 1, got {total!r}")

    def as_dict(self) -> dict[str, float]:
        """Return the quadruple keyed by action value, for provenance."""
        return {
            "deepen": self.deepen,
            "innovate": self.innovate,
            "inject": self.inject,
            "resurface": self.resurface,
        }


class ProbabilitySchedule(Protocol):
    """Return the action distribution that applies to a turn."""

    def probabilities(self, turn_index: int) -> Distribution:
        ...


class ConstantSchedule:
    """The same distribution on every turn."""

    def __init__(self, distribution: Distribution) -> None:
        self._distribution = distribution

    def probabilities(self, turn_index: int) -> Distribution:
        return self._distribution


class RandomPolicy:
    """Sample one action per turn from a probability schedule."""

    def __init__(
        self,
        schedule: ProbabilitySchedule,
        *,
        seed: int = 0,
        resurface_prompts: Sequence[str] | None = None,
    ) -> None:
        self._schedule = schedule
        self._rng = random.Random(seed)
        self._resurface_index = 0
        self._resurface_prompts = _as_prompts(
            "resurface", resurface_prompts, RESURFACE_PROMPTS
        )

    def draw(self, turn_index: int) -> tuple[Action, Distribution, float]:
        """Draw an action for ``turn_index`` and report how it was chosen.

        Returns the action, the distribution it was drawn from, and the uniform
        draw ``u`` so the decision can be audited. Each call is independent of
        every other turn.
        """
        distribution = self._schedule.probabilities(turn_index)
        draw = self._rng.random()
        cumulative = distribution.deepen
        if draw < cumulative:
            return Action.DEEPEN, distribution, draw
        cumulative += distribution.innovate
        if draw < cumulative:
            return Action.INNOVATE, distribution, draw
        cumulative += distribution.inject
        if draw < cumulative:
            return Action.INJECT, distribution, draw
        return Action.RESURFACE, distribution, draw

    def decide(self, turn_index: int) -> Action:
        """Advance the RNG and return the action for this turn."""
        return self.draw(turn_index)[0]

    def prompt(self, action: Action) -> str:
        """Return the next rotating prompt for a RESURFACE action.

        DEEPEN and INNOVATE wording is authored by the scaffolding LLM and
        INJECT content is built by the topic infuser, so neither has a fixed
        prompt here.
        """
        if action == Action.RESURFACE:
            text = self._resurface_prompts[
                self._resurface_index % len(self._resurface_prompts)
            ]
            self._resurface_index += 1
            return text
        raise ValueError(
            f"no fixed prompt for {action.value!r}; DEEPEN/INNOVATE wording is "
            "authored by the scaffolding LLM and INJECT content by the infuser"
        )
