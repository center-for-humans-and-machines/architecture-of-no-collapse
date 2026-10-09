"""Reflective three-level policy (no I/O, deterministic given a seed).

Each turn independently samples one of three actions from a fixed probability
distribution:

* ``deepen``   -> ask the scaffolding LLM for a deepening steer
* ``innovate`` -> ask the scaffolding LLM for an innovation steer
* ``inject``   -> infuse a brand-new topic via search

Unlike the similarity-driven scaffolder there is no signal and no escalation
state: ``p_t(x)`` is the same distribution on every turn and each draw is
independent. The policy only *picks* the action; the wording for DEEPEN and
INNOVATE is authored at runtime by the scaffolding LLM (see the intervention
layer), not rotated from a fixed list.

This package is a deliberate copy of
:mod:`renewal.interventions.random_scaffolder` so the two can evolve
independently; only the random-draw mechanics are shared in spirit, not in code.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


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
