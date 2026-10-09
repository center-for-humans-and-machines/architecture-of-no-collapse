# Testing

Tests live in `tests/`. They are split between fast, offline unit/smoke tests
and live integration tests that require a configured endpoint.

## Unit and offline smoke tests

These need no network and no credentials:

```bash
poetry run python -m pytest
```

Relevant files:

| File | Covers |
| --- | --- |
| `test_config.py` | Schema validation, defaults, unknown-key rejection, prompt layering |
| `test_scheduler.py` | Seeded determinism and shuffled order |
| `test_loop.py` | Transcript shape, transient handling, window hook |
| `test_artifacts.py` | Run-dir allocation and artifact round-trips |
| `test_metrics.py` | Windowing, lexical/semantic math, row shape |
| `test_embed.py` | Fake embedding determinism and vLLM request shaping |
| `test_analyze.py` | Cross-run aggregation edge cases |
| `test_scaffolder.py` / `test_random_scaffolder.py` | Policy levels, visibility, provenance |
| `test_vllm.py` / `test_azure.py` | Endpoint/key resolution |
| `test_viewer.py` | Viewer index and API behavior |

## Live integration tests

These hit real providers and are skipped automatically when the relevant
endpoint is not configured in `.env`:

```bash
poetry run python -m pytest tests/integration
```

| File | Requires |
| --- | --- |
| `tests/integration/test_vllm_live.py` | A reachable MPCDF vLLM chat endpoint |
| `tests/integration/test_metrics_live.py` | A reachable vLLM embedding endpoint |
| `tests/integration/test_azure_live.py` | Azure OpenAI credentials |

A live metrics test asserts `meta.json` `metrics_status == "completed"` and that
`embeddings.parquet` is non-empty.

## Writing tests

Pytest is configured in `pyproject.toml` with `testpaths = ["tests"]`,
`pythonpath = ["src", "."]`, and `asyncio_mode = "auto"`, so async tests need no
marker. Prefer the fake LLM and fake embedding for anything that does not
specifically need a live provider.
