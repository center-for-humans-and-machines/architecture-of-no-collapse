"""Web search via the Tavily API.

Tavily is a search API designed for LLM grounding: one POST returns cleaned
``title`` / ``url`` / ``content`` for each hit, so there is no HTML scraping and
no anti-bot fragility (the problem that made the DuckDuckGo scraper fail). We
also request ``raw_content`` (the full parsed page text) so topic injections can
use verbose, substantive source text instead of Tavily's short snippet.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

LOGGER = logging.getLogger(__name__)

_TAVILY_URL = "https://api.tavily.com/search"


@dataclass(frozen=True)
class SearchResult:
    """One search hit.

    ``content`` is Tavily's short AI-extracted snippet; ``raw_content`` is the
    full parsed page text (present only when the request asks for it).
    """

    title: str
    url: str
    content: str
    score: float | None = None
    raw_content: str | None = None


class SearchClient(Protocol):
    """Return search hits for a query."""

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        ...


class TavilySearchClient:
    """Query the Tavily search API for grounded results."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = 8.0,
        search_depth: str = "basic",
    ) -> None:
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.timeout = timeout
        self.search_depth = search_depth

    async def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return up to ``top_k`` Tavily results for ``query``."""
        if not query.strip() or top_k < 1:
            return []
        if not self.api_key:
            # Deferred so an offline run can build the intervention without a
            # key; the caller falls back gracefully if injection actually fires.
            raise RuntimeError("TAVILY_API_KEY is not set")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                _TAVILY_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": self.search_depth,
                    "max_results": top_k,
                    "include_answer": False,
                    # Full parsed page text as plain text (no markdown markup).
                    "include_raw_content": "text",
                },
            )
        response.raise_for_status()
        return _parse_results(response.json(), top_k)


def _parse_results(payload: dict[str, Any], top_k: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    for item in payload.get("results", [])[:top_k]:
        url = str(item.get("url", "")).strip()
        if not url:
            continue
        raw = item.get("raw_content")
        results.append(
            SearchResult(
                title=str(item.get("title", "")).strip(),
                url=url,
                content=str(item.get("content", "")).strip(),
                score=item.get("score"),
                raw_content=str(raw).strip() if raw else None,
            )
        )
    return results
