"""Azure configuration, request shape, and response decoding tests.

Nothing leaves the process: either ``renewal.llm.azure.AsyncAzureOpenAI`` is
monkeypatched with a constructor that records its keyword arguments, or a
``SimpleNamespace`` client with an ``AsyncMock`` ``chat.completions.create`` is
injected. Endpoints and keys are placeholders.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from renewal.core.message import Message
from renewal.llm.azure import DEFAULT_API_VERSION, AzureAPI
from renewal.llm.base import GenerationExhaustedError


def make_client(response: object) -> SimpleNamespace:
    """Return an Azure client-shaped async fake.

    Exposes only ``chat.completions.create``, the single endpoint the adapter
    uses. Streaming, embeddings, and the sync client are deliberately absent.
    """
    create = AsyncMock(return_value=response)
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        )
    )


def make_response(content: str | None = "hello") -> SimpleNamespace:
    """Return a minimal Azure-compatible completion envelope."""
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def clear_azure_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove Azure configuration variables for one test.

    A repository ``.env`` is loaded at import, so a leftover value could make a
    validation test pass for the wrong reason.
    """
    for name in (
        "AZURE_OPENAI_DEPLOYMENT",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_API_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)


def test_azure_builds_client_from_environment(monkeypatch):
    """Every connection setting may be supplied by the environment."""
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "deployment")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.test")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "env-version")
    constructed = SimpleNamespace()
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return constructed

    monkeypatch.setattr("renewal.llm.azure.AsyncAzureOpenAI", constructor)

    api = AzureAPI(retry_delays=())

    assert api.deployment == "deployment"
    assert api.model == "deployment"
    assert api.client is constructed
    assert captured["api_version"] == "env-version"


def test_azure_passes_settings_and_disables_sdk_retries(monkeypatch):
    """Explicit settings reach the SDK with its own retries turned off."""
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("renewal.llm.azure.AsyncAzureOpenAI", constructor)

    AzureAPI(
        deployment="deployment",
        endpoint="https://example.test",
        api_key="secret",
        api_version="version",
        retry_delays=(),
    )

    assert captured == {
        "api_key": "secret",
        "azure_endpoint": "https://example.test",
        "api_version": "version",
        "max_retries": 0,
    }


def test_azure_uses_default_version(monkeypatch):
    """An unconfigured API version falls back to ``DEFAULT_API_VERSION``."""
    captured = {}

    def constructor(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    clear_azure_environment(monkeypatch)
    monkeypatch.setattr("renewal.llm.azure.AsyncAzureOpenAI", constructor)

    AzureAPI(
        deployment="deployment",
        endpoint="https://example.test",
        api_key="secret",
        retry_delays=(),
    )

    assert captured["api_version"] == DEFAULT_API_VERSION


def test_azure_validates_configuration(monkeypatch):
    """A blank deployment counts as absent, and endpoint plus key follow."""
    clear_azure_environment(monkeypatch)

    with pytest.raises(ValueError, match="deployment"):
        AzureAPI()
    with pytest.raises(ValueError, match="deployment"):
        AzureAPI(deployment=" ")
    with pytest.raises(ValueError, match="endpoint and API key"):
        AzureAPI(deployment="deployment")


def test_azure_validates_blank_api_version(monkeypatch):
    """A whitespace-only API version is refused, not quietly defaulted."""
    clear_azure_environment(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", " ")

    with pytest.raises(ValueError, match="API version"):
        AzureAPI(
            deployment="deployment",
            endpoint="https://example.test",
            api_key="secret",
        )


def test_azure_allows_injected_client_without_credentials(monkeypatch):
    """An injected client is used verbatim and skips endpoint and key."""
    clear_azure_environment(monkeypatch)
    client = SimpleNamespace()

    api = AzureAPI(deployment="deployment", client=client, retry_delays=())

    assert api.client is client


async def test_azure_generate_addresses_deployment_and_forwards_params():
    """The deployment fills ``model`` and known params are forwarded."""
    client = make_client(make_response("hello"))
    api = AzureAPI(
        deployment="deployment",
        client=client,
        retry_delays=(),
    )
    messages = [
        Message("system", "system", "be nice", 0),
        Message("intervention", "noise", "steer", 1),
    ]
    params = {"temperature": 0.9, "max_tokens": 200, "top_p": None, "bogus": 1}

    result = await api.generate(messages, params)

    assert result == "hello"
    client.chat.completions.create.assert_awaited_once_with(
        model="deployment",
        messages=[
            {"role": "system", "content": "be nice"},
            {"role": "user", "content": "steer"},
        ],
        temperature=0.9,
        max_tokens=200,
    )


@pytest.mark.parametrize(
    "response, message",
    [
        # No choice at all is what a server-side filter or cancellation
        # looks like.
        (SimpleNamespace(choices=[]), "no choices"),
        (make_response(content=None), "content was empty"),
        (make_response(content=""), "content was empty"),
    ],
)
async def test_azure_generate_rejects_unusable_responses(response, message):
    """Every unusable response reaches the caller as spent attempts."""
    api = AzureAPI(
        deployment="deployment",
        client=make_client(response),
        retry_delays=(),
    )

    with pytest.raises(GenerationExhaustedError) as raised:
        await api.generate([], {})
    assert isinstance(raised.value.last_error, ValueError)
    assert message in str(raised.value.last_error)


def test_azure_does_not_log_api_key(monkeypatch, caplog):
    """Constructor DEBUG records omit the credential itself."""
    clear_azure_environment(monkeypatch)
    monkeypatch.setattr(
        "renewal.llm.azure.AsyncAzureOpenAI",
        lambda **kwargs: SimpleNamespace(),
    )

    with caplog.at_level(logging.DEBUG, logger="renewal.llm.azure"):
        AzureAPI(
            deployment="deployment",
            endpoint="https://example.test",
            api_key="super-secret-key",
            retry_delays=(),
        )

    assert "super-secret-key" not in caplog.text
    assert "Initializing AzureAPI" in caplog.text
