# Architecture of Renewal — harness

Closed-loop multi-agent harness for reproducing Kong-style semantic collapse
and testing interventions. This is Step 1: skeleton + config loader + one
vLLM model (MPCDF) + loop + logging. See `plan/step-01-skeleton-config-vllm-loop-logging.md`.

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
`meta.json`, and `run.log`.

## Tests

```bash
poetry run pytest            # unit + offline smoke (no network)
poetry run pytest tests/integration   # live vLLM run (needs .env endpoint)
```

The live integration test is skipped automatically when no vLLM endpoint is
configured in `.env`.
