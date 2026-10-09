"""The memory scaffolder: sample, steer, inject, and (rarely) resurface.

Derived from — but deliberately independent of — the reflective LLM scaffolder.
Each turn the policy independently samples one of four actions from a fixed
distribution. The deepen / innovate wording is authored at runtime by a shared
*scaffolding LLM*, the inject path summarizes the searched source before it
enters the message stream, and the novel fourth action, resurface, brings back
an earlier memory.

A *memory* is created whenever a new topic is introduced: just before the switch
the scaffolding LLM summarizes the topic that is ending (everything since the
last topic) at the looping model's own ``max_tokens`` budget, and the summary is
stored in a :class:`~renewal.interventions.memory_scaffolder.memory.MemoryStore`
— it does **not** replace history, unlike the reflective scaffolder's condense.
The resurface action later selects a memory (see the store's selector) and asks
the model to connect it to the current discussion.

The store optionally reports ``memory_created`` / ``memory_surfaced`` events to
a sink so every run leaves a ``memories.jsonl`` artifact for offline review.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from renewal.core.message import Message
from renewal.interventions.memory_scaffolder.memory import (
    MemorySink,
    MemorySelector,
    MemoryStore,
)
from renewal.interventions.memory_scaffolder.policy import (
    Action,
    ConstantSchedule,
    Distribution,
    RandomPolicy,
)
from renewal.interventions.memory_scaffolder.search import (
    SearchClient,
    TavilySearchClient,
)
from renewal.interventions.memory_scaffolder.topics import (
    Topic,
    TopicInfuser,
    TopicUnavailable,
)
from renewal.interventions.memory_scaffolder.words import (
    CommonWordSampler,
    WordSampler,
)
from renewal.llm.base import LLM
from renewal.llm.config import LLMConfig
from renewal.llm.embed import EmbeddingClient, FakeEmbedding, VllmEmbedding
from renewal.prompts.loader import load_prompt_file
from renewal.registry import Registry

LOGGER = logging.getLogger(__name__)

Visibility = Literal["all", "injections", "transient"]

# A prompt family may be given inline or as a path to a YAML file of {id, text}
# entries (the same format as prompts/prompts.yaml).
PromptFamily = str | Path | list[str]

DEFAULT_PROBABILITIES = Distribution(
    deepen=0.80, innovate=0.05, inject=0.05, resurface=0.10
)

DEFAULT_DEEPEN_PROMPT = (
    "Given the previous message ask an interesting question or make a comment "
    "that allows your conversation partner to deepen the topic at hand."
)

DEFAULT_INNOVATE_PROMPT = (
    "Given the previous message ask an interesting question or make a comment, "
    "that allows your conversation partner to 30% provide a novel angle, 30% "
    "connect it in an interesting way to another topic, 30%, reconsider if "
    "there might be another way to view this."
)

DEFAULT_SUMMARIZE_PROMPT = (
    "Summarize the following text in at most {max_tokens} tokens, preserving "
    "the key information and dropping boilerplate."
)

DEFAULT_REMEMBER_PROMPT = (
    "Summarize the following conversation into a single memory of what was "
    "discussed, in at most {max_tokens} tokens."
)

# Decoding defaults for the scaffolding LLM; the summarize and remember calls
# override ``max_tokens`` with the looping model's own budget.
DEFAULT_GENERATION: dict[str, Any] = {"temperature": 0.9, "max_tokens": 200}

_FALLBACK_EXCERPT_CHARS = 800

_INJECT_FALLBACK = (
    "It is time for a topic switch. Introduce a new topic with one clear "
    "connection to the current discussion."
)

_OPENING_FALLBACK = (
    "Start the conversation by introducing one interesting, concrete topic "
    "and giving one informative observation about it."
)


def _as_distribution(probabilities: Distribution | dict | None) -> Distribution:
    """Coerce a config value (mapping or Distribution) into a Distribution."""
    if probabilities is None:
        return DEFAULT_PROBABILITIES
    if isinstance(probabilities, Distribution):
        return probabilities
    if isinstance(probabilities, dict):
        missing = [
            key
            for key in ("deepen", "innovate", "inject", "resurface")
            if key not in probabilities
        ]
        if missing:
            raise ValueError(
                "probabilities must define deepen/innovate/inject/resurface; "
                f"missing {', '.join(missing)}"
            )
        return Distribution(
            deepen=float(probabilities["deepen"]),
            innovate=float(probabilities["innovate"]),
            inject=float(probabilities["inject"]),
            resurface=float(probabilities["resurface"]),
        )
    raise ValueError(f"unsupported probabilities: {probabilities!r}")


@dataclass(frozen=True)
class MethodConfig:
    """One configurable scaffolding method (prompt + window + on/off)."""

    enabled: bool
    system_prompt: str
    history_turns: int | None


def _method(
    options: dict | None,
    default_prompt: str,
    *,
    default_history: int | None,
) -> MethodConfig:
    """Build a :class:`MethodConfig` from a raw options mapping."""
    options = options or {}
    unknown = set(options) - {"enabled", "system_prompt", "history_turns"}
    if unknown:
        raise ValueError(f"unknown method option(s): {sorted(unknown)}")
    history = options.get("history_turns", default_history)
    if history is not None and int(history) < 0:
        raise ValueError("history_turns must be >= 0 or null")
    return MethodConfig(
        enabled=bool(options.get("enabled", True)),
        system_prompt=str(options.get("system_prompt", default_prompt)),
        history_turns=None if history is None else int(history),
    )


def _build_llm(option: dict | None) -> LLM:
    """Materialize the shared scaffolding LLM from a ``{provider, model}`` map.

    Defaults to the deterministic fake adapter so an offline config never
    reaches for the network.
    """
    if not option:
        return LLMConfig(provider="fake", model="fake").materialize()
    return LLMConfig.model_validate(option).materialize()


def _build_embedder(embedding: dict | None) -> EmbeddingClient:
    """Build the memory embedder from an ``embedding`` option (offline default)."""
    if not embedding:
        return FakeEmbedding(dim=64)
    provider = embedding.get("provider", "fake")
    model = embedding.get("model", "fake")
    if provider == "vllm":
        return VllmEmbedding(model=model)
    return FakeEmbedding(model=model, dim=embedding.get("dim"))


def _resolve_prompts(source: PromptFamily | None, *, level: str) -> list[str] | None:
    """Resolve a prompt family from a YAML path or an inline list."""
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        texts = [prompt.text for prompt in load_prompt_file(Path(source))]
    else:
        texts = [str(text) for text in source]
    if not texts:
        raise ValueError(f"{level} prompt source is empty: {source!r}")
    return texts


def _truncate(text: str, limit: int | None) -> str:
    """Trim ``text`` to ``limit`` characters without cutting mid-word."""
    if limit is None or len(text) <= limit:
        return text
    window = text[:limit]
    space = window.rfind(" ")
    return window[:space] if space > 0 else window


def _recent(messages: list[Message], turns: int | None) -> list[Message]:
    """Keep the last ``turns`` messages (``None`` = all, ``0`` = none)."""
    if turns is None:
        return list(messages)
    if turns <= 0:
        return []
    return list(messages[-turns:])


def _render(messages: list[Message]) -> str:
    """Render messages as a labeled transcript for the scaffolding LLM."""
    return "\n\n".join(f"{m.speaker}: {m.content}" for m in messages)


def _compose_topic(excerpt: str, *, opening: bool) -> str:
    """Compose the steering text for a (summarized) topic excerpt."""
    if opening:
        return (
            f"Opening topic: {excerpt}\n\n"
            "Start the conversation from this topic and open with one "
            "informative observation about it."
        )
    return (
        f"New topic: {excerpt}\n\n"
        "Connect it to the current conversation and give one informative "
        "observation."
    )


@Registry.register("intervention", "memory_scaffolder")
class MemoryScaffolderIntervention:
    """Four-level scaffolder that also stores and resurfaces memories."""

    name = "memory_scaffolder"
    # The run harness injects the application seed (per replicate) so word
    # sampling, action draws, and memory selection are reproducible; the loop's
    # token budget so summaries are sized to the model being steered; and the
    # run recorder so the memory set is written to ``memories.jsonl``.
    uses_run_seed = True
    uses_loop_max_tokens = True
    uses_memory_sink = True

    def __init__(
        self,
        *,
        probabilities: Distribution | dict | None = None,
        visibility: Visibility = "all",
        open_with_topic: bool = True,
        llm: dict | None = None,
        deepen: dict | None = None,
        innovate: dict | None = None,
        summarize: dict | None = None,
        remember: dict | None = None,
        generation: dict | None = None,
        seed: int = 0,
        resurface_prompts: PromptFamily | None = None,
        loop_max_tokens: int = 200,
        embedding: dict | None = None,
        num_words: int = 3,
        search_top_k: int = 3,
        search_depth: str = "basic",
        search_timeout: float = 8.0,
        max_excerpt_chars: int | None = _FALLBACK_EXCERPT_CHARS,
        sampler: WordSampler | None = None,
        search_client: SearchClient | None = None,
        llm_client: LLM | None = None,
        embedder: EmbeddingClient | None = None,
        selector: MemorySelector | None = None,
        memory_sink: MemorySink | None = None,
    ) -> None:
        self.visibility = visibility
        if visibility not in ("all", "injections", "transient"):
            raise ValueError(f"unknown visibility {visibility!r}")
        self.open_with_topic = open_with_topic
        self.loop_max_tokens = int(loop_max_tokens)
        self.max_excerpt_chars = max_excerpt_chars

        self.distribution = _as_distribution(probabilities)
        self._policy = RandomPolicy(
            ConstantSchedule(self.distribution),
            seed=seed,
            resurface_prompts=_resolve_prompts(
                resurface_prompts, level="resurface"
            ),
        )
        # A dedicated RNG stream for memory selection, so the action sequence
        # stays reproducible no matter how many memories exist.
        self._selector_rng = random.Random(seed)

        self._deepen = _method(deepen, DEFAULT_DEEPEN_PROMPT, default_history=1)
        self._innovate = _method(innovate, DEFAULT_INNOVATE_PROMPT, default_history=1)
        self._summarize = _method(
            summarize, DEFAULT_SUMMARIZE_PROMPT, default_history=0
        )
        self._remember = _method(
            remember, DEFAULT_REMEMBER_PROMPT, default_history=None
        )

        self._generation: dict[str, Any] = {
            **DEFAULT_GENERATION,
            **(generation or {}),
        }
        self._llm = llm_client or _build_llm(llm)

        self._infuser = TopicInfuser(
            sampler or CommonWordSampler(seed=seed),
            search_client
            or TavilySearchClient(
                timeout=search_timeout,
                search_depth=search_depth,
            ),
            num_words=num_words,
            top_k=search_top_k,
        )
        self._store = MemoryStore(
            embedder or _build_embedder(embedding),
            selector=selector,
            sink=memory_sink,
        )
        # Turn index of the most recently introduced topic; everything with a
        # later turn index is "since the last topic" and eligible for a memory.
        self._last_topic_turn = -1
        # Provenance of the topic introduced most recently, attached to the next
        # memory as the topic it summarizes.
        self._last_topic_provenance: dict[str, Any] = {}

    async def act(self, messages: list[Message]) -> Message | None:
        """Draw an action and return the matching steering message."""
        assistant = [m for m in messages if m.role == "assistant"]
        if not assistant:
            return None
        latest = assistant[-1]

        action, distribution, draw = self._policy.draw(latest.turn_index)
        meta = {
            "level": action.value,
            "probabilities": distribution.as_dict(),
            "draw": draw,
        }

        if action == Action.DEEPEN:
            if not self._deepen.enabled:
                return None
            content = await self._steer(self._deepen, assistant)
            return self._steering(content, latest, meta, action)

        if action == Action.INNOVATE:
            if not self._innovate.enabled:
                return None
            content = await self._steer(self._innovate, assistant)
            return self._steering(content, latest, meta, action)

        if action == Action.RESURFACE:
            return await self._resurface(latest, meta)

        return await self._inject(messages, latest, meta)

    async def prime(self) -> Message | None:
        """Infuse a grounded opening topic before turn 0, if enabled.

        No memory is created at prime time: there is no prior topic to summarize.
        """
        if not self.open_with_topic:
            return None
        content, provenance = await self._topic_content(opening=True)
        self._last_topic_provenance = provenance
        return Message(
            role="intervention",
            speaker=self.name,
            content=content,
            turn_index=-1,
            meta={
                "level": Action.INJECT.value,
                "opening": True,
                **provenance,
            },
            transient=self._is_transient(Action.INJECT),
        )

    def _steering(
        self, content: str, latest: Message, meta: dict, action: Action
    ) -> Message:
        return Message(
            role="intervention",
            speaker=self.name,
            content=content,
            turn_index=latest.turn_index,
            meta=meta,
            transient=self._is_transient(action),
        )

    async def _steer(self, config: MethodConfig, assistant: list[Message]) -> str:
        """Ask the scaffolding LLM for a deepening/innovation steer."""
        recent = _recent(assistant, config.history_turns)
        return await self._call_llm(config.system_prompt, _render(recent))

    async def _inject(
        self, messages: list[Message], latest: Message, meta: dict
    ) -> Message:
        """Infuse a new topic, remembering the topic it replaces first."""
        since = self._last_topic_turn + 1
        content, provenance = await self._topic_content(opening=False)
        if self._remember.enabled:
            await self._remember_history(
                messages,
                since,
                latest.turn_index,
                provenance=self._last_topic_provenance,
            )
        self._last_topic_turn = latest.turn_index
        self._last_topic_provenance = provenance
        return Message(
            role="intervention",
            speaker=self.name,
            content=content,
            turn_index=latest.turn_index,
            meta={**meta, **provenance},
            transient=self._is_transient(Action.INJECT),
        )

    async def _remember_history(
        self,
        messages: list[Message],
        since: int,
        until: int,
        *,
        provenance: dict[str, Any],
    ) -> None:
        """Summarize the ending topic into a stored memory (never replaces history)."""
        conversation = [
            m
            for m in messages
            if m.turn_index >= since and m.role in ("assistant", "intervention", "user")
        ]
        conversation = _recent(conversation, self._remember.history_turns)
        if not conversation:
            return
        prompt = self._remember.system_prompt.replace(
            "{max_tokens}", str(self.loop_max_tokens)
        )
        try:
            summary = await self._call_llm(
                prompt, _render(conversation), max_tokens=self.loop_max_tokens
            )
            memory = await self._store.add(
                summary,
                created_turn=until,
                since_turn=since,
                provenance=provenance,
            )
            LOGGER.info("stored memory #%s at turn %s", memory.id, until)
        except Exception as exc:  # noqa: BLE001 - a failed memory must not break
            LOGGER.warning("memory creation failed, skipping: %s", exc)

    async def _resurface(self, latest: Message, meta: dict) -> Message | None:
        """Bring back one stored memory, or no-op when none exist yet."""
        memory = self._store.select(
            rng=self._selector_rng,
            turn_index=latest.turn_index,
            context=latest.content,
        )
        if memory is None:
            return None
        template = self._policy.prompt(Action.RESURFACE)
        content = template.replace("{memory}", memory.text)
        self._store.mark_surfaced(memory, latest.turn_index)
        meta = {
            **meta,
            "memory_id": memory.id,
            "memory_surfaced_count": memory.surfaced_count,
            "memory_created_turn": memory.created_turn,
        }
        return Message(
            role="intervention",
            speaker=self.name,
            content=content,
            turn_index=latest.turn_index,
            meta=meta,
            transient=self._is_transient(Action.RESURFACE),
        )

    async def _topic_content(self, *, opening: bool) -> tuple[str, dict]:
        """Search a topic and summarize its source text for the loop."""
        fallback = _OPENING_FALLBACK if opening else _INJECT_FALLBACK
        try:
            topic = await self._infuser.inject()
        except TopicUnavailable as exc:
            LOGGER.warning("topic injection failed, falling back: %s", exc)
            return fallback, {"fallback_reason": str(exc)}
        provenance = self._provenance(topic)
        excerpt = await self._excerpt(topic, provenance)
        return _compose_topic(excerpt, opening=opening), provenance

    async def _excerpt(self, topic: Topic, provenance: dict) -> str:
        """Summarize the source via the LLM, or cap it when summarization is off."""
        if self._summarize.enabled:
            try:
                return await self._summarize_text(topic.excerpt)
            except Exception as exc:  # noqa: BLE001 - fall back to the raw text
                LOGGER.warning("topic summarization failed, using raw excerpt: %s", exc)
                provenance["summarize_fallback_reason"] = str(exc)
        return _truncate(topic.excerpt, self.max_excerpt_chars)

    async def _summarize_text(self, text: str) -> str:
        """Ask the scaffolding LLM to shorten ``text`` to the loop's budget."""
        prompt = self._summarize.system_prompt.replace(
            "{max_tokens}", str(self.loop_max_tokens)
        )
        return await self._call_llm(prompt, text, max_tokens=self.loop_max_tokens)

    async def _call_llm(
        self, system_prompt: str, user_content: str, *, max_tokens: int | None = None
    ) -> str:
        """One scaffolding-LLM round trip with the configured decode params."""
        params = dict(self._generation)
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        prompt = [
            Message(
                role="system",
                speaker="system",
                content=system_prompt,
                turn_index=-1,
            ),
            Message(role="user", speaker="user", content=user_content, turn_index=-1),
        ]
        return await self._llm.generate(prompt, params)

    def _provenance(self, topic: Topic) -> dict:
        return {
            "words": list(topic.words),
            "query": topic.query,
            "source_url": topic.url,
            "source_title": topic.title,
            "source_score": topic.score,
            "summarized": self._summarize.enabled,
        }

    def _is_transient(self, action: Action) -> bool:
        if self.visibility == "transient":
            return True
        if self.visibility == "injections":
            return action != Action.INJECT
        return False
