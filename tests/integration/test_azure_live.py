"""Live Azure integration test — runs a short real run against .env.

Skipped automatically when no Azure OpenAI endpoint and key are configured.
"""

import json
import os
from pathlib import Path

import pytest

from renewal.config import load_run_config
from renewal.run import run_replicate

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "smoke_azure.yaml"


def _azure_configured() -> bool:
    return bool(
        os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY")
    )


@pytest.mark.skipif(
    not _azure_configured(),
    reason="no Azure OpenAI endpoint and key configured in .env",
)
async def test_azure_live_run(tmp_path):
    config = load_run_config(CONFIG)
    run_dir = tmp_path / "run"
    await run_replicate(config, run_dir=run_dir, seed=0, debug=False)

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["status"] == "completed"

    lines = (run_dir / "transcript.jsonl").read_text().strip().splitlines()
    expected_turns = config.agents.n * config.run.rounds
    assert len(lines) == expected_turns
    for line in lines:
        record = json.loads(line)
        assert record["role"] == "assistant"
        assert record["content"].strip()
