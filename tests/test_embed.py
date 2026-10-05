"""Embedding client tests."""

import math
import types

from renewal.llm.embed import FakeEmbedding, VllmEmbedding


async def test_fake_embedding_is_deterministic():
    embedder = FakeEmbedding(dim=32)
    a = await embedder.embed(["hello world"])
    b = await embedder.embed(["hello world"])
    assert a == b


async def test_fake_embedding_returns_unit_vectors():
    embedder = FakeEmbedding(dim=32)
    (vector,) = await embedder.embed(["hello world"])
    norm = math.sqrt(sum(x * x for x in vector))
    assert abs(norm - 1.0) < 1e-9


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


async def test_fake_embedding_cosine_reflects_overlap():
    embedder = FakeEmbedding(dim=128)
    va, vb, vc = await embedder.embed(["the cat sat", "the cat sat", "quantum quarks"])
    assert _dot(va, vb) > _dot(va, vc)


class _FakeEmbeddings:
    def __init__(self, vectors):
        self._vectors = vectors

    async def create(self, model, input):
        items = [
            types.SimpleNamespace(index=i, embedding=v) for i, v in enumerate(self._vectors)
        ]
        items.reverse()  # return out of order to exercise index sorting
        return types.SimpleNamespace(data=items)


class _FakeClient:
    def __init__(self, vectors):
        self.embeddings = _FakeEmbeddings(vectors)


async def test_vllm_embedding_orders_and_normalizes():
    client = _FakeClient([[2.0, 0.0], [0.0, 3.0]])
    embedder = VllmEmbedding(model="m", client=client)
    out = await embedder.embed(["a", "b"])
    assert out == [[1.0, 0.0], [0.0, 1.0]]


def test_vllm_embedding_endpoint_resolution(monkeypatch):
    monkeypatch.setenv(
        "MPCDF_VLLM_ENDPOINTS",
        '{"Qwen/Qwen3-Embedding-8B": "https://host-b/v1"}',
    )
    embedder = VllmEmbedding(model="Qwen/Qwen3-Embedding-8B", client=_FakeClient([]))
    assert embedder.endpoint == "https://host-b/v1"
