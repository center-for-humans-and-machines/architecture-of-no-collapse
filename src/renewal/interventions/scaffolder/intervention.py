"""The scaffolder intervention: measure, decide, nudge, and (rarely) inject.

Each turn it (1) measures the adjacent similarity of the latest model turn,
(2) asks the policy which level applies, (3) produces the corresponding
steering message, and (4) tags it with provenance so every decision is
auditable. Level-3 injection uses the topic infuser and falls back to a fixed
prompt if search fails, so the conversation never breaks.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

from renewal.core.message import Message
from renewal.interventions.scaffolder.policy import Action, EscalationPolicy
from renewal.interventions.scaffolder.search import SearchClient, TavilySearchClient
from renewal.interventions.scaffolder.signals import AdjacentSimilaritySignal
from renewal.interventions.scaffolder.topics import (
    Topic,
    TopicInfuser,
    TopicUnavailable,
)
from renewal.interventions.scaffolder.words import CommonWordSampler, WordSampler
from renewal.llm.embed import EmbeddingClient, FakeEmbedding, VllmEmbedding
from renewal.registry import Registry

LOGGER = logging.getLogger(__name__)

Visibility = Literal["all", "injections", "transient"]

_INJECT_FALLBACK = (
    "It is time for a topic switch. Introduce a new topic with one clear "
    "connection to the current discussion."
)

_OPENING_FALLBACK = (
    "Start the conversation by introducing one interesting, concrete topic "
    "and giving one informative observation about it."
)


def _topic_content(topic: Topic, *, opening: bool) -> str:
    """Compose the steering text for a grounded topic.

    Only the excerpt is surfaced; the source title and URL are kept as
    provenance in ``meta`` but never shown to the model. An ``opening`` topic
    seeds a conversation that has not started yet, so it asks the model to
    begin; a mid-conversation injection asks it to connect the new topic to what
    came before.
    """
    if opening:
        return (
            f"Opening topic: {topic.excerpt}\n\n"
            "Start the conversation from this topic and open with one "
            "informative observation about it."
        )
    return (
        f"New topic: {topic.excerpt}\n\n"
        "Connect it to the current conversation and give one informative "
        "observation."
    )


def _build_embedder(embedding: dict | None) -> EmbeddingClient:
    """Build the signal embedder from an ``embedding`` option (offline default)."""
    if not embedding:
        return FakeEmbedding(dim=64)
    provider = embedding.get("provider", "fake")
    model = embedding.get("model", "fake")
    if provider == "vllm":
        return VllmEmbedding(model=model)
    return FakeEmbedding(model=model, dim=embedding.get("dim"))


def _json_signal(signal: float | None) -> float | None:
    if signal is None or (isinstance(signal, float) and math.isnan(signal)):
        return None
    return signal


@Registry.register("intervention", "scaffolder")
class ScaffolderIntervention:
    """Three-level novelty scaffolder driven by adjacent similarity."""

    name = "scaffolder"

    def __init__(
        self,
        *,
        threshold: float = 0.85,
        visibility: Visibility = "all",
        open_with_topic: bool = True,
        seed: int = 0,
        embedding: dict | None = None,
        num_words: int = 3,
        search_top_k: int = 3,
        max_excerpt_chars: int | None = None,
        search_depth: str = "basic",
        search_timeout: float = 8.0,
        embedder: EmbeddingClient | None = None,
        sampler: WordSampler | None = None,
        search_client: SearchClient | None = None,
    ) -> None:
        self.threshold = threshold
        self.visibility = visibility
        if visibility not in ("all", "injections", "transient"):
            raise ValueError(f"unknown visibility {visibility!r}")
        self.open_with_topic = open_with_topic

        self._policy = EscalationPolicy(threshold=threshold)
        self._signal = AdjacentSimilaritySignal(embedder or _build_embedder(embedding))
        self._infuser = TopicInfuser(
            sampler or CommonWordSampler(seed=seed),
            search_client or TavilySearchClient(
                timeout=search_timeout,
                search_depth=search_depth,
            ),
            num_words=num_words,
            top_k=search_top_k,
            max_excerpt_chars=max_excerpt_chars,
        )

    async def act(self, messages: list[Message]) -> Message | None:
        """Measure the latest turn and return the matching steering message."""
        assistant = [m for m in messages if m.role == "assistant"]
        if not assistant:
            return None
        latest = assistant[-1]

        signal = await self._signal.update(latest.content)
        action = self._policy.decide(signal)

        if action == Action.INJECT:
            content, provenance = await self._injection_content()
        else:
            content = self._policy.prompt(action)
            provenance = {}

        return Message(
            role="intervention",
            speaker=self.name,
            content=content,
            turn_index=latest.turn_index,
            meta={
                "level": action.value,
                "signal": _json_signal(signal),
                "threshold": self.threshold,
                **provenance,
            },
            transient=self._is_transient(action),
        )

    async def prime(self) -> Message | None:
        """Inject a grounded starting topic before turn 0, if enabled."""
        if not self.open_with_topic:
            return None
        content, provenance = await self._injection_content(opening=True)
        return Message(
            role="intervention",
            speaker=self.name,
            content=content,
            turn_index=-1,
            meta={
                "level": Action.INJECT.value,
                "threshold": self.threshold,
                "opening": True,
                **provenance,
            },
            transient=self._is_transient(Action.INJECT),
        )

    def _is_transient(self, action: Action) -> bool:
        if self.visibility == "transient":
            return True
        if self.visibility == "injections":
            return action != Action.INJECT
        return False

    async def _injection_content(self, *, opening: bool = False) -> tuple[str, dict]:
        fallback = _OPENING_FALLBACK if opening else _INJECT_FALLBACK
        try:
            topic = await self._infuser.inject()
        except TopicUnavailable as exc:
            LOGGER.warning("topic injection failed, falling back: %s", exc)
            return fallback, {"fallback_reason": str(exc)}
        provenance = {
            "words": list(topic.words),
            "query": topic.query,
            "source_url": topic.url,
            "source_title": topic.title,
            "source_score": topic.score,
        }
        return _topic_content(topic, opening=opening), provenance
