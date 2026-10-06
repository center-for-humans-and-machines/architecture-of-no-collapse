"""Random three-level policy (no I/O, deterministic given a seed).

Each turn independently samples one of three actions from a fixed probability
distribution:

* ``deepen``   -> nudge the model to go deeper (rotate the deepen prompts)
* ``innovate`` -> nudge it toward a new connected idea (rotate innovate prompts)
* ``inject``   -> infuse a brand-new topic via search

Unlike the similarity-driven scaffolder there is no signal and no escalation
state: ``p_t(x)`` is the same distribution on every turn and each draw is
independent. The policy only picks *what* to do and, for DEEPEN/INNOVATE,
*which* wording; INJECT has no fixed wording, the topic content is built by the
infuser and the intervention layer composes the final message.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

DEEPEN_PROMPTS = (
    "Add one concrete detail.",
    "Give one specific example.",
    "Develop one consequence.",
    "Name one edge case.",
)

INNOVATE_PROMPTS = (
    "Introduce a genuinely new idea that still connects to this topic.",
    "Take this topic in an unexpected but clearly connected direction.",
    "Add a new concept that changes how this topic can be viewed.",
    "Find a non-obvious neighboring idea and explain the connection.",
)


class Action(str, Enum):
    """Observable level selected by the policy."""

    DEEPEN = "deepen"
    INNOVATE = "innovate"
    INJECT = "inject"


@dataclass(frozen=True)
class Distribution:
    """A categorical distribution over the three actions.

    The three probabilities must be non-negative and sum to one. The policy
    draws one action from this distribution on every turn.
    """

    deepen: float
    innovate: float
    inject: float

    def __post_init__(self) -> None:
        values = (self.deepen, self.innovate, self.inject)
        if any(value < 0.0 for value in values):
            raise ValueError("probabilities must be non-negative")
        total = sum(values)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"probabilities must sum to 1, got {total!r}")

    def as_dict(self) -> dict[str, float]:
        """Return the triple keyed by action value, for provenance."""
        return {
            "deepen": self.deepen,
            "innovate": self.innovate,
            "inject": self.inject,
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

    def __init__(self, schedule: ProbabilitySchedule, *, seed: int = 0) -> None:
        self._schedule = schedule
        self._rng = random.Random(seed)
        self._deepen_index = 0
        self._innovate_index = 0

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
        return Action.INJECT, distribution, draw

    def decide(self, turn_index: int) -> Action:
        """Advance the RNG and return the action for this turn."""
        return self.draw(turn_index)[0]

    def prompt(self, action: Action) -> str:
        """Return the next rotating prompt for a DEEPEN/INNOVATE action."""
        if action == Action.DEEPEN:
            text = DEEPEN_PROMPTS[self._deepen_index % len(DEEPEN_PROMPTS)]
            self._deepen_index += 1
            return text
        if action == Action.INNOVATE:
            text = INNOVATE_PROMPTS[self._innovate_index % len(INNOVATE_PROMPTS)]
            self._innovate_index += 1
            return text
        raise ValueError(
            f"no fixed prompt for {action.value!r}; INJECT content is built "
            "by the topic infuser"
        )
