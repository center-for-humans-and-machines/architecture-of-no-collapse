"""Live metrics integration test — runs a short real run with metrics on.

Skipped automatically when the embedding model has no MPCDF vLLM endpoint.
"""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from renewal.config import load_run_config
from renewal.run import run_replicate

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "smoke_vllm.yaml"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


def _embedding_endpoint_configured() -> bool:
    """True when the embedding model has its own MPCDF vLLM endpoint."""
    raw = os.getenv("MPCDF_VLLM_ENDPOINTS")
    if not raw:
        return False
    try:
        endpoints = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return isinstance(endpoints, dict) and EMBEDDING_MODEL in endpoints


@pytest.mark.skipif(
    not _embedding_endpoint_configured(),
    reason="no MPCDF vLLM endpoint for the embedding model in .env",
)
async def test_metrics_live_run(tmp_path):
    config = load_run_config(CONFIG)
    run_dir = tmp_path / "run"
    await run_replicate(config, run_dir=run_dir, seed=0, debug=False, force_metrics=True)

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["metrics_status"] == "completed"

    embeddings = pd.read_parquet(run_dir / "embeddings.parquet")
    assert len(embeddings) > 0
