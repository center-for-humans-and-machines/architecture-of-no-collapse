"""Level-3 topic injection: sample words -> search -> surface a novel topic."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from renewal.interventions.random_scaffolder.search import (
    SearchClient,
    SearchResult,
)
from renewal.interventions.random_scaffolder.words import WordSampler

LOGGER = logging.getLogger(__name__)


class TopicUnavailable(RuntimeError):
    """Raised when the search pipeline cannot produce a grounded topic."""


@dataclass(frozen=True)
class Topic:
    """A novel topic plus the provenance that produced it."""

    words: tuple[str, ...]
    query: str
    title: str
    url: str
    excerpt: str
    score: float | None


class TopicInfuser:
    """Sample words, search the web, and return the top hit as a topic."""

    def __init__(
        self,
        sampler: WordSampler,
        search_client: SearchClient,
        *,
        num_words: int = 3,
        top_k: int = 3,
        max_excerpt_chars: int = 400,
    ) -> None:
        if num_words < 1:
            raise ValueError("num_words must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._sampler = sampler
        self._search_client = search_client
        self._num_words = num_words
        self._top_k = top_k
        self._max_excerpt_chars = max_excerpt_chars

    async def inject(self) -> Topic:
        """Run the full pipeline or raise :class:`TopicUnavailable`."""
        try:
            words = await asyncio.to_thread(self._sampler.sample, self._num_words)
        except Exception as exc:  # loading/network failures must not break
            raise TopicUnavailable(f"word sampling failed: {exc}") from exc
        query, results = await self._search_words(words)
        result = results[0]
        return Topic(
            words=tuple(words),
            query=query,
            title=result.title,
            url=result.url,
            excerpt=result.content[: self._max_excerpt_chars],
            score=result.score,
        )

    async def _search_words(
        self, words: list[str]
    ) -> tuple[str, list[SearchResult]]:
        """Search joined words, narrowing the query until it hits."""
        last_error: Exception | None = None
        for query in _candidate_queries(words):
            try:
                results = await self._search_client.search(query, self._top_k)
            except Exception as exc:
                last_error = exc
                continue
            if results:
                return query, results
        if last_error is not None:
            raise TopicUnavailable(f"search failed: {last_error}") from last_error
        raise TopicUnavailable("search returned no results")


def _candidate_queries(words: list[str]) -> list[str]:
    """Query candidates from most to least specific.

    Arbitrary sampled words rarely match together, so the infuser falls back
    from the full join to the first two words, then the first word alone.
    """
    cleaned = [word for word in (word.strip() for word in words) if word]
    if not cleaned:
        return []
    candidates = [" ".join(cleaned)]
    if len(cleaned) >= 2:
        candidates.append(" ".join(cleaned[:2]))
    candidates.append(cleaned[0])
    seen: set[str] = set()
    unique: list[str] = []
    for query in candidates:
        if query and query not in seen:
            seen.add(query)
            unique.append(query)
    return unique
