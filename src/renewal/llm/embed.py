"""Embedding client: one uniform interface over vLLM, plus a deterministic fake.

Embeddings power the semantic metrics (Step 2). The interface mirrors ``LLM``:
a single ``embed`` that turns a list of texts into a list of unit-norm vectors.
``VllmEmbedding`` reuses the same per-model endpoint resolution as ``VllmAPI``
(Step 1 §6); ``FakeEmbedding`` is a deterministic feature-hashing fallback for
offline tests and smoke runs.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Protocol

import numpy as np
from openai import AsyncOpenAI

from renewal.core.text import tokenize
from renewal.llm.env import safe_url
from renewal.llm.vllm import _resolve_api_key, _resolve_endpoint

LOGGER = logging.getLogger(__name__)


class EmbeddingClient(Protocol):
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one unit-norm vector per input text, in input order."""
        ...


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


class VllmEmbedding:
    """Call one vLLM OpenAI-compatible server for embeddings."""

    def __init__(
        self,
        model: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or "Qwen/Qwen3-Embedding-8B"
        if not self.model:
            raise ValueError("embedding model is required")
        self.endpoint = _resolve_endpoint(self.model, endpoint)
        api_key = _resolve_api_key(self.endpoint, api_key)
        LOGGER.debug(
            "Initializing VllmEmbedding: model=%s endpoint=%s api_key=%s client=%s",
            self.model,
            safe_url(self.endpoint),
            "set" if api_key else "unset",
            "injected" if client is not None else "unset",
        )
        if client is not None:
            self.client = client
            return
        if not self.endpoint:
            raise ValueError("vLLM endpoint is required for the embedding model")
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.endpoint,
            # SDK retries are off; embeddings are issued once per window.
            max_retries=0,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(model=self.model, input=texts)
        data = sorted(response.data, key=lambda item: item.index)
        return [
            _unit(np.asarray(item.embedding, dtype=np.float64)).tolist()
            for item in data
        ]


class FakeEmbedding:
    """Deterministic feature-hashing embedder for offline tests.

    Each unigram is hashed to seed a per-token RNG that draws ``dim`` standard
    normals; the token vectors are summed and L2-normalized. Result: cosine is
    approximately token overlap, so the same code path runs offline and
    deterministically.
    """

    def __init__(self, model: str = "fake", dim: int | None = None, seed: int = 0) -> None:
        self.model = model
        self.dim = dim or 128
        self.seed = seed

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=np.float64)
        for token in tokenize(text):
            vector += self._token_vector(token)
        return _unit(vector).tolist()

    def _token_vector(self, token: str) -> np.ndarray:
        digest = hashlib.sha256(f"{self.seed}:{token}".encode("utf-8")).digest()[:8]
        seed = int.from_bytes(digest, "big")
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim)
