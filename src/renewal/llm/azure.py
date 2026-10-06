"""Azure OpenAI adapter.

Wraps the ``openai`` SDK's ``AsyncAzureOpenAI`` client and calls chat
completions for plain text, mirroring ``VllmAPI``. It is selected by the
provider name ``azure`` in ``LLMConfig``.

Configuration comes from ``AZURE_OPENAI_DEPLOYMENT``,
``AZURE_OPENAI_ENDPOINT``, ``AZURE_OPENAI_API_KEY``, and the optional
``AZURE_OPENAI_API_VERSION`` unless the corresponding constructor argument is
given.

Azure deviates from OpenAI proper in two ways that matter here. Requests
address a *deployment* created in the Azure resource, not a model id, and the
deployment is passed in the ``model`` field of the request. And the API
surface is versioned per resource, so a pinned ``api_version`` is required.
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncAzureOpenAI

from renewal.core.message import Message
from renewal.llm.base import BaseLLM
from renewal.llm.env import safe_url, setting
from renewal.llm.openai_compat import clean_params, first_text, to_openai

LOGGER = logging.getLogger(__name__)

# Recent enough for strict schema output on the current service; also the
# version Azure's own examples pin.
DEFAULT_API_VERSION = "2024-12-01-preview"


class AzureAPI(BaseLLM):
    """Call one Azure OpenAI deployment for plain text.

    Attributes:
        deployment: Azure deployment name sent as the request's model.
        model: Alias of ``deployment`` so the shared LLM-call log reports a
            model name exactly as the other adapters do.
        client: ``AsyncAzureOpenAI`` instance, or an injected substitute.
    """

    def __init__(
        self,
        deployment: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        api_version: str | None = None,
        client: Any | None = None,
        retry_delays: tuple[float, ...] = (0.5, 1.0),
    ) -> None:
        """Resolve Azure settings and open a client.

        Args:
            deployment: Deployment name, falling back to
                ``AZURE_OPENAI_DEPLOYMENT``. Required even when a client is
                injected, because it is part of every request rather than of
                the connection.
            endpoint: Resource URL such as
                ``https://example-resource.openai.azure.com/``, falling back
                to ``AZURE_OPENAI_ENDPOINT``.
            api_key: Credential, falling back to ``AZURE_OPENAI_API_KEY``.
            api_version: API version to pin, falling back to
                ``AZURE_OPENAI_API_VERSION`` and then to
                ``DEFAULT_API_VERSION``.
            client: Ready-made async client used verbatim. Intended for tests,
                and it bypasses the endpoint, key, and version checks.
            retry_delays: Retry policy handed to ``BaseLLM``.

        Raises:
            ValueError: No deployment resolved, or, when building a client, a
                missing endpoint, API key, or API version.
        """
        self.deployment = setting(deployment, "AZURE_OPENAI_DEPLOYMENT")
        if not self.deployment:
            raise ValueError("Azure deployment is required")
        self.model = self.deployment
        # Resolved even when a client is injected, so the debug record reports
        # the resolved settings rather than the arguments.
        endpoint = setting(endpoint, "AZURE_OPENAI_ENDPOINT")
        api_key = setting(api_key, "AZURE_OPENAI_API_KEY")
        api_version = setting(
            api_version,
            "AZURE_OPENAI_API_VERSION",
            DEFAULT_API_VERSION,
        )
        LOGGER.debug(
            "Initializing AzureAPI: deployment=%s endpoint=%s api_key=%s "
            "api_version=%s client=%s retry_delays=%s",
            self.deployment,
            safe_url(endpoint),
            "set" if api_key else "unset",
            api_version,
            "injected" if client is not None else "unset",
            retry_delays,
        )
        # An injected client already carries endpoint, key, and version, so
        # none of them are required.
        if client is not None:
            self.client = client
            super().__init__(retry_delays=retry_delays)
            return
        if not endpoint or not api_key:
            raise ValueError("Azure endpoint and API key are required")
        # A blank variable defeats the default, so the version is rechecked
        # rather than assumed present.
        if not api_version:
            raise ValueError("Azure API version is required")
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
            # SDK retries are off because BaseLLM owns the retry budget.
            max_retries=0,
        )
        super().__init__(retry_delays=retry_delays)

    async def generate(self, messages: list[Message], params: dict[str, Any]) -> str:
        kwargs: dict[str, Any] = {
            # Azure routes by deployment, so the deployment name goes where an
            # OpenAI model id would.
            "model": self.deployment,
            "messages": [to_openai(m) for m in messages],
        }
        kwargs.update(clean_params(params))
        response = await self.client.chat.completions.create(**kwargs)
        return first_text(response, "Azure")
