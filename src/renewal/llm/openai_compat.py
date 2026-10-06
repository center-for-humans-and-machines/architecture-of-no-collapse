"""Shared helpers for OpenAI-compatible chat adapters.

Both ``VllmAPI`` and ``AzureAPI`` speak the OpenAI chat-completions
envelope, so the request/response glue lives here instead of being
duplicated: provider-agnostic message mapping, decoding-parameter
filtering, and the "one usable text completion" check. Only the parts
that differ between providers — client construction and endpoint
resolution — stay in each adapter.
"""

from __future__ import annotations

from typing import Any

from renewal.core.message import Message

# Placeholder credential for an unauthenticated local server; the OpenAI
# client refuses to start without one. Never a real secret.
LOCAL_API_KEY = "local-no-auth"

# Generation params we forward to an OpenAI-compatible server, dropped when
# None or unknown.
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


def to_openai(message: Message) -> dict[str, str]:
    """Map a ``Message`` to an OpenAI chat turn.

    OpenAI-compatible APIs have no "intervention" role; interventions are
    surfaced as plain user turns (the speaker is kept on the ``Message`` for
    the transcript, but is not injected into the prompt).
    """
    if message.role == "intervention":
        return {"role": "user", "content": message.content}
    return {"role": message.role, "content": message.content}


def clean_params(params: dict[str, Any]) -> dict[str, Any]:
    """Keep only the known, non-None generation parameters."""
    return {k: v for k, v in params.items() if k in _ALLOWED_PARAMS and v is not None}


def first_text(response: Any, provider: str) -> str:
    """Return the first choice's message content, or raise ``ValueError``.

    Attribute access is defensive because the same envelope is produced by
    the OpenAI SDK, Azure OpenAI, and OpenAI-compatible servers such as vLLM,
    which do not all populate every optional field.

    Args:
        response: Chat completion object exposing ``choices``.
        provider: Display name used in error messages, for example ``"Azure"``.

    Raises:
        ValueError: The envelope carried no choice, or the message content
            was missing or empty. Adapters let this propagate so the base
            class treats it as a retryable attempt.
    """
    if not getattr(response, "choices", None):
        raise ValueError(f"{provider} response had no choices")
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"{provider} response content was empty")
    return content
