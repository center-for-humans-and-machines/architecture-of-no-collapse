"""The random scaffolder intervention: sample, nudge, and (rarely) inject.

Each turn the policy independently samples one of three actions from a fixed
probability distribution (deepen / innovate / inject), the corresponding
steering message is produced, and the decision is tagged with provenance. The
level-3 injection uses the topic infuser and falls back to a fixed prompt if
search fails, so the conversation never breaks.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from renewal.core.message import Message
from renewal.interventions.random_scaffolder.policy import (
    Action,
    ConstantSchedule,
    Distribution,
    RandomPolicy,
)
from renewal.interventions.random_scaffolder.search import (
    SearchClient,
    TavilySearchClient,
)
from renewal.interventions.random_scaffolder.topics import (
    Topic,
    TopicInfuser,
    TopicUnavailable,
)
from renewal.interventions.random_scaffolder.words import (
    CommonWordSampler,
    WordSampler,
)
from renewal.prompts.loader import load_prompt_file
from renewal.registry import Registry

LOGGER = logging.getLogger(__name__)

Visibility = Literal["all", "injections", "transient"]

# A prompt family may be given inline or as a path to a YAML file of {id, text}
# entries (the same format as prompts/prompts.yaml).
PromptFamily = str | Path | Sequence[str]

_INJECT_FALLBACK = (
    "It is time for a topic switch. Introduce a new topic with one clear "
    "connection to the current discussion."
)

_OPENING_FALLBACK = (
    "Start the conversation by introducing one interesting, concrete topic "
    "and giving one informative observation about it."
)

DEFAULT_PROBABILITIES = Distribution(deepen=0.8, innovate=0.15, inject=0.05)


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


def _as_distribution(probabilities: Distribution | dict | None) -> Distribution:
    """Coerce a config value (mapping or Distribution) into a Distribution."""
    if probabilities is None:
        return DEFAULT_PROBABILITIES
    if isinstance(probabilities, Distribution):
        return probabilities
    if isinstance(probabilities, dict):
        missing = [
            key for key in ("deepen", "innovate", "inject") if key not in probabilities
        ]
        if missing:
            raise ValueError(
                "probabilities must define deepen/innovate/inject; "
                f"missing {', '.join(missing)}"
            )
        return Distribution(
            deepen=float(probabilities["deepen"]),
            innovate=float(probabilities["innovate"]),
            inject=float(probabilities["inject"]),
        )
    raise ValueError(f"unsupported probabilities: {probabilities!r}")


def _resolve_prompts(source: PromptFamily | None, *, level: str) -> list[str] | None:
    """Resolve a deepen/innovate family from a YAML path or an inline list.

    A ``str``/``Path`` names a YAML file of ``{id, text}`` entries (the same
    format as ``prompts/prompts.yaml``); any other sequence is used as-is.
    ``None`` keeps the policy's built-in family. The resolved family must be
    non-empty, so a misconfigured source fails loudly at build time.
    """
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        texts = [prompt.text for prompt in load_prompt_file(Path(source))]
    else:
        texts = [str(text) for text in source]
    if not texts:
        raise ValueError(f"{level} prompt source is empty: {source!r}")
    return texts


@Registry.register("intervention", "random_scaffolder")
class RandomScaffolderIntervention:
    """Three-level novelty scaffolder driven by independent random draws."""

    name = "random_scaffolder"
    # The run harness injects the application seed (per replicate) so the draw
    # sequence is reproducible and differs between replicates; this intervention
    # has no seed of its own to configure.
    uses_run_seed = True

    def __init__(
        self,
        *,
        probabilities: Distribution | dict | None = None,
        visibility: Visibility = "all",
        open_with_topic: bool = True,
        seed: int = 0,
        num_words: int = 3,
        search_top_k: int = 3,
        max_excerpt_chars: int | None = None,
        search_depth: str = "basic",
        search_timeout: float = 8.0,
        deepen_prompts: PromptFamily | None = None,
        innovate_prompts: PromptFamily | None = None,
        sampler: WordSampler | None = None,
        search_client: SearchClient | None = None,
    ) -> None:
        self.visibility = visibility
        if visibility not in ("all", "injections", "transient"):
            raise ValueError(f"unknown visibility {visibility!r}")
        self.open_with_topic = open_with_topic

        self.distribution = _as_distribution(probabilities)
        self._policy = RandomPolicy(
            ConstantSchedule(self.distribution),
            seed=seed,
            deepen_prompts=_resolve_prompts(deepen_prompts, level="deepen"),
            innovate_prompts=_resolve_prompts(innovate_prompts, level="innovate"),
        )
        self._infuser = TopicInfuser(
            sampler or CommonWordSampler(seed=seed),
            search_client
            or TavilySearchClient(
                timeout=search_timeout,
                search_depth=search_depth,
            ),
            num_words=num_words,
            top_k=search_top_k,
            max_excerpt_chars=max_excerpt_chars,
        )

    async def act(self, messages: list[Message]) -> Message | None:
        """Draw an action and return the matching steering message."""
        assistant = [m for m in messages if m.role == "assistant"]
        if not assistant:
            return None
        latest = assistant[-1]

        action, distribution, draw = self._policy.draw(latest.turn_index)

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
                "probabilities": distribution.as_dict(),
                "draw": draw,
                **provenance,
            },
            transient=self._is_transient(action),
        )

    async def prime(self) -> Message | None:
        """Inject a grounded starting topic before turn 0, if enabled.

        The opening is not selected by a random draw, so it records no
        ``probabilities`` or ``draw``; it is always an INJECT-level message.
        """
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
