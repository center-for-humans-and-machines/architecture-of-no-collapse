"""OpenAI-compatible vLLM adapter for MPCDF-cluster models.

Talks to a self-hosted vLLM server through its OpenAI-compatible chat
completions route, using the ``openai`` SDK's ``AsyncOpenAI`` client pointed at
that server's base URL.

MPCDF serves one model per endpoint, so endpoint selection is keyed by model
(plan §5.6). Resolution order: explicit constructor arg →
``MPCDF_VLLM_ENDPOINTS[model]`` → single-endpoint fallback
``MPCDF_VLLM_ENDPOINT_URL``. The API key is resolved as
``MPCDF_VLLM_API_KEYS[endpoint]`` → ``MPCDF_VLLM_API_KEY`` → a local no-auth
placeholder.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI

from renewal.core.message import Message
from renewal.llm.base import BaseLLM
from renewal.llm.env import safe_url, setting

LOGGER = logging.getLogger(__name__)

# Placeholder credential for an unauthenticated local server; the OpenAI
# client refuses to start without one. Never a real secret.
LOCAL_API_KEY = "local-no-auth"

# Generation params we forward to the server, dropped when None.
_ALLOWED_PARAMS = frozenset(
    {
        "temperature",
        "max_tokens",
        "top_p",
        "seed",
        "frequency_penalty",
        "presence_penalty",
    }
)


def _json_env(name: str) -> dict[str, str] | None:
    raw = os.getenv(name)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must be valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _resolve_endpoint(model: str, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    endpoints = _json_env("MPCDF_VLLM_ENDPOINTS")
    if endpoints and model in endpoints:
        return endpoints[model]
    return os.getenv("MPCDF_VLLM_ENDPOINT_URL")


def _resolve_api_key(endpoint: str | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    keys = _json_env("MPCDF_VLLM_API_KEYS")
    if keys and endpoint and endpoint in keys:
        return keys[endpoint]
    return os.getenv("MPCDF_VLLM_API_KEY") or LOCAL_API_KEY


def _to_openai(message: Message) -> dict[str, str]:
    # OpenAI-compatible APIs have no "intervention" role; interventions are
    # surfaced as plain user turns (the speaker is kept on the Message for the
    # transcript, but is not injected into the prompt).
    if message.role == "intervention":
        return {"role": "user", "content": message.content}
    return {"role": message.role, "content": message.content}


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if k in _ALLOWED_PARAMS and v is not None}


class VllmAPI(BaseLLM):
    """Call one vLLM OpenAI-compatible server for plain text."""

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        retry_delays: tuple[float, ...] = (0.5, 1.0),
    ) -> None:
        self.model = setting(model, "MPCDF_VLLM_MODEL")
        if not self.model:
            raise ValueError("vLLM model is required")
        self.endpoint = _resolve_endpoint(self.model, endpoint)
        api_key = _resolve_api_key(self.endpoint, api_key)
        LOGGER.debug(
            "Initializing VllmAPI: model=%s endpoint=%s api_key=%s client=%s",
            self.model,
            safe_url(self.endpoint),
            "set" if api_key else "unset",
            "injected" if client is not None else "unset",
        )
        if client is not None:
            self.client = client
            super().__init__(retry_delays=retry_delays)
            return
        if not self.endpoint:
            raise ValueError("vLLM endpoint is required")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.endpoint,
            # SDK retries are off because BaseLLM owns the retry budget.
            max_retries=0,
        )
        super().__init__(retry_delays=retry_delays)

    async def generate(self, messages: list[Message], params: dict[str, Any]) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [_to_openai(m) for m in messages],
        }
        kwargs.update(_clean_params(params))
        response = await self.client.chat.completions.create(**kwargs)
        if not response.choices:
            raise ValueError("vLLM response had no choices")
        content = response.choices[0].message.content
        if not content:
            raise ValueError("vLLM response content was empty")
        return content
