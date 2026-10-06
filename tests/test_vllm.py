"""vLLM endpoint/key resolution and message mapping tests."""

import pytest

from renewal.core.message import Message
from renewal.llm.env import safe_url
from renewal.llm.openai_compat import clean_params, to_openai
from renewal.llm.vllm import (
    VllmAPI,
    _resolve_api_key,
    _resolve_endpoint,
)


def test_endpoint_explicit_wins(monkeypatch):
    monkeypatch.delenv("MPCDF_VLLM_ENDPOINTS", raising=False)
    monkeypatch.setenv("MPCDF_VLLM_ENDPOINT_URL", "http://fallback/v1")
    assert _resolve_endpoint("m", "http://explicit/v1") == "http://explicit/v1"


def test_endpoint_from_map(monkeypatch):
    monkeypatch.setenv(
        "MPCDF_VLLM_ENDPOINTS", '{"a":"http://a/v1","b":"http://b/v1"}'
    )
    monkeypatch.setenv("MPCDF_VLLM_ENDPOINT_URL", "http://fallback/v1")
    assert _resolve_endpoint("a", None) == "http://a/v1"
    assert _resolve_endpoint("zzz", None) == "http://fallback/v1"


def test_endpoint_fallback(monkeypatch):
    monkeypatch.delenv("MPCDF_VLLM_ENDPOINTS", raising=False)
    monkeypatch.setenv("MPCDF_VLLM_ENDPOINT_URL", "http://fallback/v1")
    assert _resolve_endpoint("m", None) == "http://fallback/v1"


def test_api_key_defaults_to_local(monkeypatch):
    monkeypatch.delenv("MPCDF_VLLM_API_KEY", raising=False)
    monkeypatch.delenv("MPCDF_VLLM_API_KEYS", raising=False)
    assert _resolve_api_key("http://x/v1", None) == "local-no-auth"


def test_api_key_from_map(monkeypatch):
    monkeypatch.delenv("MPCDF_VLLM_API_KEY", raising=False)
    monkeypatch.setenv('MPCDF_VLLM_API_KEYS', '{"http://x/v1":"secret"}')
    assert _resolve_api_key("http://x/v1", None) == "secret"


def test_clean_params_drops_none_and_unknown():
    assert clean_params(
        {"temperature": 0.9, "max_tokens": 200, "top_p": None, "bogus": 1}
    ) == {"temperature": 0.9, "max_tokens": 200}


def test_role_mapping():
    intervention = Message("intervention", "noise", "hello", 0)
    assert to_openai(intervention) == {
        "role": "user",
        "content": "hello",
    }
    assistant = Message("assistant", "agent_0", "hi", 0)
    assert to_openai(assistant) == {"role": "assistant", "content": "hi"}


def test_safe_url():
    assert safe_url("https://user:pass@host:1234/v1?x=1") == "https://host:1234"
    assert safe_url(None) == "unset"
    assert safe_url("not-a-url") == "set"


def test_vllm_requires_model(monkeypatch):
    monkeypatch.delenv("MPCDF_VLLM_MODEL", raising=False)
    with pytest.raises(ValueError):
        VllmAPI(model=None)


def test_vllm_requires_endpoint(monkeypatch):
    monkeypatch.setenv("MPCDF_VLLM_MODEL", "m")
    monkeypatch.delenv("MPCDF_VLLM_ENDPOINT_URL", raising=False)
    monkeypatch.delenv("MPCDF_VLLM_ENDPOINTS", raising=False)
    with pytest.raises(ValueError):
        VllmAPI(model="m")
