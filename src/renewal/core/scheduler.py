"""Randomized round-robin scheduler.

Each round, the N agents are shuffled (seeded) and each acts exactly once.
The same seed produces the same order, making a run reproducible.
"""

from __future__ import annotations

import random


class Scheduler:
    def __init__(self, n_agents: int, seed: int) -> None:
        if n_agents < 1:
            raise ValueError("need at least one agent")
        self.n_agents = n_agents
        self._rng = random.Random(seed)
        self.turn_index = -1

    def order(self) -> list[int]:
        """Return a fresh shuffled agent order for one round."""
        order = list(range(self.n_agents))
        self._rng.shuffle(order)
        return order

    def next_turn(self) -> int:
        """Advance and return the global turn index."""
        self.turn_index += 1
        return self.turn_index
