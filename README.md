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

### LLM providers

Each agent names a `provider` and `model` in the config `pool`. Supported
providers are `vllm` (default, MPCDF cluster), `azure`, and `fake`. Credentials
and endpoints never live in the YAML — they come from `.env`:

- `vllm` — `MPCDF_VLLM_ENDPOINT_URL` (or per-model `MPCDF_VLLM_ENDPOINTS`) and
  `MPCDF_VLLM_MODEL`; `model` is the served model id.
- `azure` — `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and optionally
  `AZURE_OPENAI_API_VERSION` (default `2024-12-01-preview`); `model` is the
  Azure **deployment name**, not an OpenAI model id. See
  `configs/smoke_azure.yaml`.
- `fake` — deterministic offline model, no credentials.

```yaml
agents:
  pool:
    - { provider: azure, model: gpt-4o-mini }
```

### Memory (forgetting)

Agents need not see the whole conversation. `run.memory_turns` (default `20`)
limits each agent's prompt to the most recent N agent turns, together with any
intervention/steering messages attached to those turns; older turns scroll out
of view. The canonical `transcript.jsonl`, `events.jsonl`, and metrics always
keep the full conversation — forgetting affects only the live prompt. Set
`memory_turns: null` to disable forgetting and show agents the entire history.

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
| `inject` | adjacent sim `>= threshold` again | sample 3 common words → Tavily search → surface the top hit's text as a new topic |

Every message records its `level`, `signal`, `threshold`, and (for `inject`)
the sampled `words`, `query`, `source_url`, and `source_title` in `meta`. The
visible injection text contains only the search excerpt and the instruction —
the source title and URL are kept in `meta` but not shown to the model. The
excerpt is Tavily's full parsed page text (`raw_content`), preferred over its
short snippet; set `max_excerpt_chars` to cap it (the default `null` uses the
full text). When a cap applies, the text is trimmed back to the last sentence
boundary so it never cuts mid-sentence.

Before turn 0 the scaffolder injects a grounded **opening topic** (the same
common-word → Tavily pipeline) so the first agent responds to a concrete
starting point instead of generating from an empty history. The opening is
tagged `meta.opening: true` with `turn_index: -1`, honors `visibility` like any
other message, and falls back to a generic opening prompt if the topic pipeline
fails. Disable it with `open_with_topic: false` (the offline smoke configs do,
to stay network-free).

`visibility` controls whether steering messages persist in history:

- `all` (default) — every message is canonical history.
- `injections` — only level-3 injections persist; deepen/innovate nudges are
  prompt-scoped and dropped after the next turn.
- `transient` — all steering messages are prompt-scoped.

Offline smoke (fake LLM + fake embedding, no Tavily):

```bash
poetry run renewal run configs/scaffolder_smoke.yaml
```

A live run sets `embedding: { provider: vllm, model: Qwen/Qwen3-Embedding-8B }`
and requires `TAVILY_API_KEY` (used only when a level-3 injection fires). Topic
seed words are sampled from a bundled list of the 20,000 most common English
words — no vector download.

### Random scaffolder

`random_scaffolder` is a self-contained sibling of the scaffolder with the same
three levels and prompts, but the policy ignores similarity. Instead it defines
a fixed probability distribution `p(x)` and, **on every turn independently**,
draws one action from it:

```yaml
interventions:
  - type: random_scaffolder
    options:
      probabilities:
        deepen: 0.80    # nudge to deepen
        innovate: 0.15  # nudge to innovate
        inject: 0.05    # infuse a new topic via search
      visibility: all
```

The three probabilities must be non-negative and sum to one. The random draw
uses the run's application seed (`run.seed` / `--seed`, offset by replicate), so
there is no per-intervention seed and the sequence is reproducible. `visibility`
keeps the same meaning as above (`all` / `injections` / `transient`). Each
message records its `level`, the `probabilities` used, and the uniform `draw`
that selected the action (plus the sampled `words`, `query`, `source_url`, and
`source_title` for `inject`).

Like the scaffolder above, the random scaffolder injects a grounded **opening
topic** before turn 0 (`open_with_topic`, default true; `meta.opening: true`,
`turn_index: -1`). This opening is not produced by a random draw, so it records
no `probabilities` or `draw`.

Offline smoke (fake LLM, inject disabled so no Tavily call):

```bash
poetry run renewal run configs/random_scaffolder_smoke.yaml
```

A live run is `configs/run_qwen_random_scaffolder.yaml`, which sets
`inject: 0.05` and uses the same Tavily injection path as the scaffolder.

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
