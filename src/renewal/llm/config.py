"""Environment-authenticated LLM adapter configuration.

``LLMConfig`` is the YAML-facing surface: a config names a provider and a
model, and ``materialize`` returns the matching adapter. It carries no
credentials of its own, so experiment files stay safe to commit while keys,
endpoints, and deployment details come from the environment.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict

from renewal.llm.base import BaseLLM
from renewal.llm.fake import FakeLLM
from renewal.llm.vllm import VllmAPI

LOGGER = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Select an adapter by provider name.

    Frozen and extra-forbidding, so a typo in a YAML key is an error rather
    than a silently ignored field. Validation checks only the provider name;
    whether credentials exist is decided by the adapter when ``materialize``
    runs.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: Literal["vllm", "fake"] = "vllm"
    model: str

    def materialize(self) -> BaseLLM:
        """Build the selected adapter.

        Returns:
            A ready adapter with default retry delays. Constructing it opens
            the provider client but sends no request.

        Raises:
            ValueError: The adapter found no usable endpoint or model.
        """
        LOGGER.debug("materializing api: provider=%s model=%s", self.provider, self.model)
        match self.provider:
            case "vllm":
                return VllmAPI(model=self.model)
            case "fake":
                return FakeLLM(model=self.model)
        raise ValueError(f"unknown provider: {self.provider}")
