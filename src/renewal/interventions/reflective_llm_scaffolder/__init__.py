"""A three-level scaffolder whose steering text is authored by an LLM.

Same deepen / innovate / inject skeleton as
:mod:`renewal.interventions.random_scaffolder`, but the action wording is
generated at runtime by a shared scaffolding LLM (guided by configurable system
prompts), searched source text is summarized before it enters the stream, and
the conversation since the last topic is condensed into a single recap that
replaces that slice of history.

The package is a deliberate, self-contained copy of
:mod:`renewal.interventions.random_scaffolder` so the two can be iterated on
independently.
"""

from renewal.interventions.reflective_llm_scaffolder.intervention import (
    DEFAULT_CONDENSE_PROMPT,
    DEFAULT_DEEPEN_PROMPT,
    DEFAULT_GENERATION,
    DEFAULT_INNOVATE_PROMPT,
    DEFAULT_PROBABILITIES,
    DEFAULT_SUMMARIZE_PROMPT,
    MethodConfig,
    ReflectiveLLMScaffolderIntervention,
    Visibility,
)
from renewal.interventions.reflective_llm_scaffolder.policy import (
    Action,
    ConstantSchedule,
    Distribution,
    ProbabilitySchedule,
    RandomPolicy,
)
from renewal.interventions.reflective_llm_scaffolder.search import (
    SearchClient,
    SearchResult,
    TavilySearchClient,
)
from renewal.interventions.reflective_llm_scaffolder.topics import (
    Topic,
    TopicInfuser,
    TopicUnavailable,
)
from renewal.interventions.reflective_llm_scaffolder.words import (
    CommonWordSampler,
    WordSampler,
)

__all__ = [
    "Action",
    "CommonWordSampler",
    "ConstantSchedule",
    "DEFAULT_CONDENSE_PROMPT",
    "DEFAULT_DEEPEN_PROMPT",
    "DEFAULT_GENERATION",
    "DEFAULT_INNOVATE_PROMPT",
    "DEFAULT_PROBABILITIES",
    "DEFAULT_SUMMARIZE_PROMPT",
    "Distribution",
    "MethodConfig",
    "ProbabilitySchedule",
    "RandomPolicy",
    "ReflectiveLLMScaffolderIntervention",
    "SearchClient",
    "SearchResult",
    "TavilySearchClient",
    "Topic",
    "TopicInfuser",
    "TopicUnavailable",
    "Visibility",
    "WordSampler",
]
