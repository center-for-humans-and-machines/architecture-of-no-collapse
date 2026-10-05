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

## Interventions

The `scaffolder` intervention is a three-level novelty controller that reacts
each turn to the **adjacent similarity** of the latest model turn (cosine vs.
the previous turn; higher = more repetition = more collapse):

| Level | Trigger | Action |
| --- | --- | --- |
| `deepen` | adjacent sim `< threshold` | rotate: "Add one concrete detail.", "Give one specific example.", "Develop one consequence.", "Name one edge case." |
| `innovate` | adjacent sim `>= threshold` (first time) | rotate: "Introduce a genuinely new idea…", "Take this topic in an unexpected…", "Add a new concept…", "Find a non-obvious neighboring idea…" |
| `inject` | adjacent sim `>= threshold` again | sample 3 GloVe words → Tavily search → surface the top hit as a new topic |

Every message records its `level`, `signal`, `threshold`, and (for `inject`)
the sampled `words`, `query`, `source_url`, and `source_title` in `meta`.

`visibility` controls whether steering messages persist in history:

- `all` (default) — every message is canonical history.
- `injections` — only level-3 injections persist; deepen/innovate nudges are
  prompt-scoped and dropped after the next turn.
- `transient` — all steering messages are prompt-scoped.

Offline smoke (fake LLM + fake embedding, no Tavily, no word-vector download):

```bash
poetry run renewal run configs/scaffolder_smoke.yaml
```

A live run sets `embedding: { provider: vllm, model: Qwen/Qwen3-Embedding-8B }`
and requires `TAVILY_API_KEY` (used only when a level-3 injection fires) plus
the GloVe word vectors, downloaded and cached on first use.

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
