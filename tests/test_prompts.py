"""Prompt loading and per-agent assignment tests."""

import pytest

from renewal.prompts.loader import Prompt, PromptSource


def test_assign_cycles_and_overrides():
    source = PromptSource([Prompt("a", "alpha"), Prompt("b", "beta")], "DEFAULT")
    result = source.assign(3, 0, overrides={"0": "b", "agent_2": "a"})
    assert len(result) == 3
    assert result[0] == "beta"
    assert result[2] == "alpha"
    assert result[1] in {"alpha", "beta"}


def test_assign_without_file_uses_default():
    source = PromptSource.load(None, "DEFAULT")
    assert source.assign(3, 0) == ["DEFAULT", "DEFAULT", "DEFAULT"]


def test_unknown_id_raises():
    source = PromptSource([Prompt("a", "alpha")], "DEFAULT")
    with pytest.raises(ValueError):
        source.assign(1, 0, overrides={"0": "nope"})


def test_override_bad_index_raises():
    source = PromptSource([Prompt("a", "alpha")], "DEFAULT")
    with pytest.raises(ValueError):
        source.assign(1, 0, overrides={"5": "a"})


def test_load_prompt_file(tmp_path):
    path = tmp_path / "prompts.yaml"
    path.write_text("- id: a\n  text: alpha\n- id: b\n  text: beta\n")
    source = PromptSource.load(path, "DEFAULT")
    assert set(source._ids) == {"a", "b"}


def test_load_prompt_file_rejects_duplicates(tmp_path):
    path = tmp_path / "prompts.yaml"
    path.write_text("- id: a\n  text: alpha\n- id: a\n  text: beta\n")
    with pytest.raises(ValueError):
        PromptSource.load(path, "DEFAULT")


def test_missing_prompt_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PromptSource.load(tmp_path / "nope.yaml", "DEFAULT")
