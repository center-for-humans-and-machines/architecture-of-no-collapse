"""Schema validation tests."""

import pytest
from pydantic import ValidationError

from renewal.config import RunConfig


def _base() -> dict:
    return {
        "agents": {
            "n": 3,
            "pool": [
                {"provider": "fake", "model": "fake"},
                {"provider": "fake", "model": "fake"},
                {"provider": "fake", "model": "fake"},
            ],
        }
    }


def test_defaults():
    config = RunConfig.model_validate(_base())
    assert config.run.rounds == 200
    assert config.run.seed == 0
    assert config.run.window_size == 10
    assert config.run.replicates == 1
    assert config.agents.params.temperature == 0.9
    assert config.agents.params.max_tokens == 200
    assert config.metrics.enabled == []
    assert config.interventions == []


def test_unknown_top_level_key_rejected():
    data = _base()
    data["bogus"] = 1
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_unknown_agent_key_rejected():
    data = _base()
    data["agents"]["bogus"] = 1
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_pool_length_must_match_n():
    data = _base()
    data["agents"]["pool"] = data["agents"]["pool"][:2]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_unknown_intervention_type_rejected():
    data = _base()
    data["interventions"] = [{"type": "noise"}]
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)


def test_known_intervention_accepted():
    data = _base()
    data["interventions"] = [{"type": "noop"}]
    config = RunConfig.model_validate(data)
    assert config.interventions[0].type == "noop"


def test_unknown_provider_rejected():
    data = _base()
    data["agents"]["pool"][0] = {"provider": "openai", "model": "gpt-4o-mini"}
    with pytest.raises(ValidationError):
        RunConfig.model_validate(data)
