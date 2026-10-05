"""Run configuration: schema, loading, and resolved dump.

One YAML file fully describes a run. ``RunConfig`` is its schema; every level
is frozen and ``extra="forbid"``, so unknown or misspelled keys fail loading
instead of being ignored. Paths (``prompt_source``) are resolved relative to
the working directory, matching the repository-root convention.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from renewal.llm.config import LLMConfig
from renewal.llm.embed import FakeEmbedding, VllmEmbedding
from renewal.registry import Registry

LOGGER = logging.getLogger(__name__)


def ensure_builtins() -> None:
    """Import plugin packages so their registry entries resolve."""
    import renewal.interventions  # noqa: F401
    import renewal.metrics  # noqa: F401


class RunSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rounds: int = Field(default=200, ge=1)
    replicates: int = Field(default=1, ge=1)
    seed: int = 0
    window_size: int = Field(default=10, ge=1)


class GenerationParams(BaseModel):
    """Per-agent decoding parameters (Kong defaults for temperature/tokens)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: float = 0.9
    max_tokens: int = 200
    top_p: float | None = None
    seed: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None


class AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    n: int = Field(ge=1)
    params: GenerationParams = Field(default_factory=GenerationParams)
    pool: list[LLMConfig]
    prompt_source: Path | None = None
    system_prompt: str = "You are an agent in an open-ended multi-agent conversation."
    prompts_override: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _pool_matches_n(self) -> "AgentsConfig":
        if len(self.pool) != self.n:
            raise ValueError(f"agents.pool must have exactly {self.n} entries")
        return self


class InterventionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_registered(self) -> "InterventionConfig":
        try:
            Registry.get("intervention", self.type)
        except KeyError as error:
            raise ValueError(str(error)) from error
        return self


class EmbeddingConfig(BaseModel):
    """Select the embedding model used by the semantic metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["vllm", "fake"] = "vllm"
    model: str = "Qwen/Qwen3-Embedding-8B"
    dim: int | None = None

    def materialize(self):
        """Build the selected embedding client."""
        match self.provider:
            case "vllm":
                return VllmEmbedding(model=self.model)
            case "fake":
                return FakeEmbedding(model=self.model, dim=self.dim)
        raise ValueError(f"unknown embedding provider: {self.provider}")


class MetricsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    window_size: int = Field(default=10, ge=1)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    enabled: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_enabled(self) -> "MetricsConfig":
        for name in self.enabled:
            try:
                Registry.get("metric", name)
            except KeyError as error:
                raise ValueError(str(error)) from error
        return self


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    out_dir: Path = Path("outputs")
    experiment_name: str = "run"
    labels: dict[str, str] = Field(default_factory=dict)
    console_level: str = "INFO"
    file_level: str = "DEBUG"


class RunConfig(BaseModel):
    """One validated run configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: RunSettings = Field(default_factory=RunSettings)
    agents: AgentsConfig
    interventions: list[InterventionConfig] = Field(default_factory=list)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="before")
    @classmethod
    def _ensure_builtins(cls, data: Any) -> Any:
        ensure_builtins()
        return data


def load_run_config(path: str | Path) -> RunConfig:
    """Read and validate one YAML run configuration."""
    ensure_builtins()
    with Path(path).open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    config = RunConfig.model_validate(values)
    LOGGER.debug(
        "loaded config: path=%s rounds=%s seed=%s n_agents=%s",
        path,
        config.run.rounds,
        config.run.seed,
        config.agents.n,
    )
    return config


def dump_run_config(config: RunConfig) -> dict[str, Any]:
    """Return a YAML-loadable mapping of ``config`` with defaults filled."""
    return config.model_dump(mode="json")
