"""A three-level scaffolder whose policy samples actions at random.

Same deepen / innovate / inject behaviour as
:mod:`renewal.interventions.scaffolder`, but each turn independently draws one
action from a fixed probability distribution instead of reacting to an
adjacent-similarity signal.
"""

from renewal.interventions.random_scaffolder.intervention import (
    DEFAULT_PROBABILITIES,
    RandomScaffolderIntervention,
    Visibility,
)
from renewal.interventions.random_scaffolder.policy import (
    DEEPEN_PROMPTS,
    INNOVATE_PROMPTS,
    Action,
    ConstantSchedule,
    Distribution,
    ProbabilitySchedule,
    RandomPolicy,
)
from renewal.interventions.random_scaffolder.search import (
    SearchClient,
    SearchResult,
    TavilySearchClient,
)
from renewal.interventions.random_scaffolder.topics import (
    Topic,
    TopicInfuser,
    TopicUnavailable,
)
from renewal.interventions.random_scaffolder.words import (
    DEFAULT_MODEL,
    GloVeWordSampler,
    WordSampler,
)

__all__ = [
    "DEEPEN_PROMPTS",
    "DEFAULT_MODEL",
    "DEFAULT_PROBABILITIES",
    "INNOVATE_PROMPTS",
    "Action",
    "ConstantSchedule",
    "Distribution",
    "GloVeWordSampler",
    "ProbabilitySchedule",
    "RandomPolicy",
    "RandomScaffolderIntervention",
    "SearchClient",
    "SearchResult",
    "TavilySearchClient",
    "Topic",
    "TopicInfuser",
    "TopicUnavailable",
    "Visibility",
    "WordSampler",
]
