"""Metric plugins. Importing this package registers built-ins."""

from renewal.metrics.lexical import LexicalDiversity  # noqa: F401
from renewal.metrics.semantic import (  # noqa: F401
    CrossRunSim,
    SemanticAdjacentSim,
    SemanticAnchorSim,
)
