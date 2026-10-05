# Architecture of Renewal — harness

Closed-loop multi-agent harness for reproducing Kong-style semantic collapse
and testing interventions. Steps 1–2: skeleton + config loader + one vLLM model
(MPCDF) + loop + logging, and the four Kong-default collapse metrics. See
`plan/step-01-skeleton-config-vllm-loop-logging.md` and
`plan/step-02-metric-tools.md`.

## Install

```bash
poetry install
cp .env.example .env   # fill in the MPCDF vLLM endpoint
```

Requires Python 3.13.

## Run

Offline (no network, deterministic fake model):

```bash
poetry run renewal run configs/smoke_fake.yaml
```

Live (one MPCDF vLLM model, from `.env`):

```bash
poetry run renewal run configs/smoke_vllm.yaml
```

Validate a config without calling any model:

```bash
poetry run renewal run configs/smoke_vllm.yaml --dry-run
```

Outputs land in `outputs/<experiment>/<timestamp>_seed<N>/` with
`config.yaml`, `resolved_config.yaml`, `transcript.jsonl`, `events.jsonl`,
`meta.json`, and `run.log`. With metrics enabled, each run also writes
`metrics.parquet` (long-format measures) and `embeddings.parquet` (per-window
vectors).

## Metrics

Offline smoke (fake LLM + fake embedding, all four metrics):

```bash
poetry run renewal run configs/smoke_metrics.yaml
```

Compute the cross-run similarity over completed runs of an experiment:

```bash
poetry run renewal analyze outputs/smoke_metrics/
```

Live metrics use the MPCDF vLLM embedding model (`Qwen/Qwen3-Embedding-8B`,
served with `--task embed`); force them on for a config with empty
`metrics.enabled`:

```bash
poetry run renewal run configs/smoke_vllm.yaml --metrics
```

## Viewer

A standalone web app (FastAPI + Svelte + Plotly) for browsing runs, reading
conversations, and comparing metric timelines. See `viewer/README.md`.

```bash
poetry run uvicorn viewer.backend.main:app --reload --port 8000   # API
cd viewer/frontend && npm install && npm run dev                  # UI
```

## Tests

```bash
poetry run python -m pytest            # unit + offline smoke (no network)
poetry run python -m pytest tests/integration   # live vLLM run (needs .env endpoint)
```

The live integration test is skipped automatically when no vLLM endpoint is
configured in `.env`.
