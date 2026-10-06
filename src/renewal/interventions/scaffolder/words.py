"""Random-word sampling from a bundled list of common English words.

The sampler draws ``count`` distinct words from the bundled frequency list
(:data:`renewal.interventions.common_words.COMMON_WORDS`). Sampling common words
instead of arbitrary points in a word-vector space keeps the resulting 3-word
sequences usable as web-search queries and removes the GloVe download entirely.

The per-package class mirrors its sibling in
:mod:`renewal.interventions.random_scaffolder`; the shared word data lives once
in :mod:`renewal.interventions.common_words`.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Protocol

from renewal.interventions.common_words import COMMON_WORDS


class WordSampler(Protocol):
    """Return random common words."""

    def sample(self, count: int) -> list[str]:
        ...


class CommonWordSampler:
    """Draw distinct words from a bundled list of common English words."""

    def __init__(
        self,
        *,
        seed: int = 0,
        words: Sequence[str] | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._words = tuple(words) if words is not None else COMMON_WORDS
        if not self._words:
            raise ValueError("common word list is empty")

    def sample(self, count: int) -> list[str]:
        """Return ``count`` distinct common words (drawn without replacement)."""
        if count < 1:
            raise ValueError("count must be >= 1")
        return self._rng.sample(self._words, min(count, len(self._words)))
