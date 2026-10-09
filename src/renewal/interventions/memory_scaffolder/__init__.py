"""A four-level scaffolder that also stores and resurfaces memories.

Same deepen / innovate / inject skeleton as
:mod:`renewal.interventions.reflective_llm_scaffolder` — its steering text is
authored at runtime by a shared scaffolding LLM, searched source text is
summarized before it enters the stream — plus a fourth action, ``resurface``.

Whenever a new topic is introduced the scaffolder first has the scaffolding LLM
summarize the topic that is ending (at the looping model's ``max_tokens``) and
stores the summary as a :class:`~.memory.Memory`. The ``resurface`` action later
brings one memory back, asking the model to connect it to the current
discussion. Unlike the reflective scaffolder's condense, remembering never
replaces history. Memories can be inspected offline in ``memories.jsonl``.

The package is a deliberate, self-contained copy of
:mod:`renewal.interventions.reflective_llm_scaffolder` so the variants can be
iterated on independently.
"""

from renewal.interventions.memory_scaffolder.intervention import (
    DEFAULT_DEEPEN_PROMPT,
    DEFAULT_GENERATION,
    DEFAULT_INNOVATE_PROMPT,
    DEFAULT_PROBABILITIES,
    DEFAULT_REMEMBER_PROMPT,
    DEFAULT_SUMMARIZE_PROMPT,
    MemoryScaffolderIntervention,
    MethodConfig,
    Visibility,
)
from renewal.interventions.memory_scaffolder.memory import (
    Memory,
    MemorySelector,
    MemorySink,
    MemoryStore,
    RandomUnsurfacedSelector,
)
from renewal.interventions.memory_scaffolder.policy import (
    RESURFACE_PROMPTS,
    Action,
    ConstantSchedule,
    Distribution,
    ProbabilitySchedule,
    RandomPolicy,
)
from renewal.interventions.memory_scaffolder.search import (
    SearchClient,
    SearchResult,
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

__all__ = [
    "Action",
    "CommonWordSampler",
    "ConstantSchedule",
    "DEFAULT_DEEPEN_PROMPT",
    "DEFAULT_GENERATION",
    "DEFAULT_INNOVATE_PROMPT",
    "DEFAULT_PROBABILITIES",
    "DEFAULT_REMEMBER_PROMPT",
    "DEFAULT_SUMMARIZE_PROMPT",
    "Distribution",
    "Memory",
    "MemoryScaffolderIntervention",
    "MemorySelector",
    "MemorySink",
    "MemoryStore",
    "MethodConfig",
    "ProbabilitySchedule",
    "RESURFACE_PROMPTS",
    "RandomPolicy",
    "RandomUnsurfacedSelector",
    "SearchClient",
    "SearchResult",
    "TavilySearchClient",
    "Topic",
    "TopicInfuser",
    "TopicUnavailable",
    "Visibility",
    "WordSampler",
]
