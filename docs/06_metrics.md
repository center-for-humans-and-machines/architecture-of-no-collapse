# Metrics

Metrics are pluggable tools that measure how the conversation collapses. The
four Kong defaults are registered out of the box:

| Metric | Cadence | What it measures |
| --- | --- | --- |
| `lexical_diversity` | per window | Cumulative unique unigrams (surface redundancy) |
| `semantic_anchor_sim` | per window | Cosine to the **first** window (drift from the start) |
| `semantic_adjacent_sim` | per window | Cosine to the **previous** window (convergence) |
| `cross_run_sim` | per run | Pairwise cosine across independent runs |

All four are expressed as similarities (cosine over unit-norm embeddings, or a
token ratio), so **higher means more alike / less collapsed** and the curves are
directly comparable to Kong's reported numbers.

## Enabling metrics

List the names under `metrics.enabled`; unknown names fail at load.

```yaml
metrics:
  window_size: 10
  embedding: { provider: vllm, model: Qwen/Qwen3-Embedding-8B }
  enabled: [lexical_diversity, semantic_anchor_sim, semantic_adjacent_sim, cross_run_sim]
```

`cross_run_sim` needs multiple runs, so the in-run runner skips it; it is
computed by `renewal analyze` (below).

An empty `metrics.enabled` disables metrics for that config. `renewal run
<config> --metrics` forces metrics on and, when `enabled` is empty, uses the
Kong default set.

## Windowing

The per-window tools share one windowing rule:

1. Keep only `assistant` turns (system and intervention messages are excluded).
2. Group them by `meta["round"]`.
3. Window `i` spans rounds `[i·W, (i+1)·W)` where `W = window_size`. The last
   window may be partial.
4. Window text is the turn contents joined by newlines.
5. A window with no turns has no embedding; its metrics are `NaN`.

Because `window_size` is in **rounds**, a 10-round window with 3 agents holds up
to 30 utterances — the Kong-comparable granularity.

## Embeddings

The semantic metrics embed each window with the client in `metrics.embedding`:

- `provider: vllm`, `model: Qwen/Qwen3-Embedding-8B` — the live default,
  served by vLLM with `--task embed`. Its endpoint is resolved like any other
  vLLM model (see [LLM providers](05_llm-providers.md)).
- `provider: fake` — a deterministic offline embedder used by the smoke configs
  and tests. Vectors are always L2-normalized so cosine is a plain dot product.

Metrics are computed incrementally: each window is measured the moment it
closes, and `metrics.parquet` / `embeddings.parquet` are rewritten atomically
(a reader never sees a half-written file). A failed embedding or metric records
`meta.json` `metrics_status: partial` and leaves `NaN` for that window; it does
not abort the run. Otherwise `metrics_status` is `completed`, or `skipped` when
metrics are disabled.

## Offline and live runs

Offline (fake LLM + fake embedding, no network):

```bash
poetry run renewal run configs/smoke_metrics.yaml
```

Live (MPCDF vLLM embedding model), for a config whose `metrics.enabled` is
empty:

```bash
poetry run renewal run configs/smoke_vllm.yaml --metrics
```

## Cross-run similarity

`cross_run_sim` answers whether collapse is deterministic: if independent runs
of the same experiment land on the same attractor, their window embeddings align
and the similarity is high. It is computed over the run directories **directly
under** the given directory:

```bash
poetry run renewal analyze outputs/smoke_metrics/
```

The command writes an experiment-level `metrics.parquet` containing one
`cross_run_sim` row per run (the run's contribution to the group mean) plus one
group-level row. Runs are grouped by their `condition` label; `--condition
LABEL` restricts the analysis to one label. Fewer than two runs in a group
yields `NaN` (and a warning). Runs without `embeddings.parquet` are skipped and
reported.

Because `condition` labels nest runs one directory deeper (see
[Run artifacts](08_artifacts.md)), point `analyze` at the directory whose direct
children are the run folders.
