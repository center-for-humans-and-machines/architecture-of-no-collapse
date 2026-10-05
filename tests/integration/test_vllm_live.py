"""Live vLLM integration test — runs a short real run against .env endpoint.

Skipped automatically when no MPCDF vLLM endpoint is configured.
"""

import json
import os
from pathlib import Path

import pytest

from renewal.config import load_run_config
from renewal.run import run_replicate

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "smoke_vllm.yaml"


def _endpoint_configured() -> bool:
    return bool(
        os.getenv("MPCDF_VLLM_ENDPOINT_URL") or os.getenv("MPCDF_VLLM_ENDPOINTS")
    )


@pytest.mark.skipif(
    not _endpoint_configured(),
    reason="no MPCDF vLLM endpoint configured in .env",
)
async def test_vllm_live_run(tmp_path):
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
