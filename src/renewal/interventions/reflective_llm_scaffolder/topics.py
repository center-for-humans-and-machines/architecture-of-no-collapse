"""Topic injection: sample words -> search -> surface the raw source text.

Unlike the random scaffolder's infuser, this infuser never truncates the source
text itself: it hands the *full* parsed page text to the intervention layer,
which is responsible for turning it into a right-sized excerpt (via the
scaffolding LLM's summarizer, or a fixed character cap when summarization is
disabled).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from renewal.interventions.reflective_llm_scaffolder.search import (
    SearchClient,
    SearchResult,
)
from renewal.interventions.reflective_llm_scaffolder.words import WordSampler

LOGGER = logging.getLogger(__name__)


class TopicUnavailable(RuntimeError):
    """Raised when the search pipeline cannot produce a grounded topic."""


@dataclass(frozen=True)
class Topic:
    """A novel topic plus the provenance that produced it.

    ``excerpt`` carries the full source text (typically ``raw_content``); the
    caller decides how to condense it before it enters the message stream.
    """

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
    ) -> None:
        if num_words < 1:
            raise ValueError("num_words must be >= 1")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._sampler = sampler
        self._search_client = search_client
        self._num_words = num_words
        self._top_k = top_k

    async def inject(self) -> Topic:
        """Run the full pipeline or raise :class:`TopicUnavailable`."""
        try:
            words = await asyncio.to_thread(self._sampler.sample, self._num_words)
        except Exception as exc:  # loading/network failures must not break
            raise TopicUnavailable(f"word sampling failed: {exc}") from exc
        query, results = await self._search_words(words)
        result = results[0]
        # Prefer the full parsed page text; fall back to Tavily's short snippet.
        source_text = result.raw_content or result.content
        return Topic(
            words=tuple(words),
            query=query,
            title=result.title,
            url=result.url,
            excerpt=source_text,
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
