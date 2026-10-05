"""Pure three-level escalation policy (no I/O, deterministic).

Drives the scaffolder each turn from a single scalar: the adjacent similarity
``s`` (cosine of the latest model turn against the previous one). Higher ``s``
means more repetition, i.e. more collapse. A single ``crossed_streak`` counter
maps the signal to one of three actions:

* ``s < threshold``          -> DEEPEN   (rotate the four deepening prompts)
* ``s >= threshold`` (first) -> INNOVATE (rotate the four innovation prompts)
* ``s >= threshold`` (again) -> INJECT   (infuse a brand-new topic via search)

The policy only picks *what* to do and, for DEEPEN/INNOVATE, *which* wording.
INJECT has no fixed wording: the topic content is built by the infuser and the
intervention layer composes the final message.
"""

from __future__ import annotations

from enum import Enum

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


class EscalationPolicy:
    """Map an adjacent-similarity signal to one of the three actions."""

    def __init__(self, *, threshold: float = 0.85) -> None:
        self.threshold = threshold
        self._crossed_streak = 0
        self._deepen_index = 0
        self._innovate_index = 0

    def decide(self, signal: float | None) -> Action:
        """Advance the escalation state and return the action for this turn.

        A ``None`` signal (no previous turn to compare against) is treated as
        healthy and resets any streak.
        """
        if signal is None or signal < self.threshold:
            self._crossed_streak = 0
            return Action.DEEPEN
        self._crossed_streak += 1
        if self._crossed_streak >= 2:
            return Action.INJECT
        return Action.INNOVATE

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
