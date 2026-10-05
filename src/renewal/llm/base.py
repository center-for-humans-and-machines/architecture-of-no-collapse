"""Uniform model interface.

``LLM`` is the protocol every adapter implements: one async ``generate`` that
turns a message list into a string. ``BaseLLM`` is the concrete base that adds
the retry policy; adapters subclass it and implement a single unretried
``generate``, and ``__init__`` installs the retry wrapper around it.
"""

from __future__ import annotations

import asyncio
import logging
from functools import wraps
from typing import Any

from renewal.core.message import Message

LOGGER = logging.getLogger(__name__)

# Dedicated channel for full model calls, written to llm_calls.jsonl under
# ``--debug``; kept separate from the operational log. propagate=False keeps
# the (potentially large) call payloads off the console/file logs; only the
# explicit ``--debug`` handler ever receives them.
LLM_CALL_LOGGER = logging.getLogger("renewal.llm.calls")
LLM_CALL_LOGGER.propagate = False


class GenerationExhaustedError(RuntimeError):
    """Raised when every provider attempt failed.

    Carries the attempt count and the final exception (chained as __cause__)
    so the provider-level detail survives.
    """

    def __init__(self, attempts: int, last_error: Exception) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"exhausted {attempts} generation attempt(s)")


class LLM:
    """Protocol: one async ``generate(messages, params) -> str``."""

    async def generate(self, messages: list[Message], params: dict[str, Any]) -> str:
        ...


class BaseLLM:
    """Adapter base that owns the retry policy around ``generate``.

    Subclasses implement ``generate`` as exactly one provider round trip and
    must call ``super().__init__`` after setting up their client; that call
    wraps ``generate`` with the retry logic. Only the provider round trip is
    retried: every exception is treated as a retryable failure, and final
    failure raises ``GenerationExhaustedError``.
    """

    def __init__(self, retry_delays: tuple[float, ...] = (0.5, 1.0)) -> None:
        self.retry_delays = tuple(retry_delays)
        self._wrap_generate()

    def _wrap_generate(self) -> None:
        method = self._inherited_generate()
        while method is not None and hasattr(method, "__wrapped__"):
            method = method.__wrapped__
        if method is None:
            return
        bound = method.__get__(self, type(self))
        wrapped = self._retry_wrapper(bound)
        object.__setattr__(self, "generate", wrapped)

    def _inherited_generate(self):
        for cls in type(self).__mro__:
            if cls is BaseLLM:
                return None
            if "generate" in cls.__dict__:
                return cls.__dict__["generate"]
        return None

    def _retry_wrapper(self, func):
        @wraps(func)
        async def wrapper(messages: list[Message], params: dict[str, Any]) -> str:
            LLM_CALL_LOGGER.info(
                "model=%s params=%s messages=%s",
                getattr(self, "model", "unknown"),
                params,
                [m.to_dict() for m in messages],
            )
            return await self._retry(func, messages, params)

        return wrapper

    async def _retry(self, func, messages: list[Message], params: dict[str, Any]) -> str:
        attempts = 1 + len(self.retry_delays)
        last_error: Exception | None = None
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(self.retry_delays[attempt - 1])
            LOGGER.debug(
                "%s attempt %s/%s",
                type(self).__name__,
                attempt + 1,
                attempts,
            )
            try:
                return await func(messages, params)
            except Exception as error:  # noqa: BLE001 - any failure retries
                last_error = error
                LOGGER.warning(
                    "%s attempt %s/%s failed: %s",
                    type(self).__name__,
                    attempt + 1,
                    attempts,
                    error,
                )
        assert last_error is not None
        LOGGER.warning("%s exhausted %s attempts", type(self).__name__, attempts)
        raise GenerationExhaustedError(attempts, last_error) from last_error
