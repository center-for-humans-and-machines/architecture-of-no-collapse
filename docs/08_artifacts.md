# Run artifacts

One run = one self-contained directory. The recorder writes the transcript and
event stream incrementally (append + flush per line) and updates `meta.json` on
start, finish, and failure, so a crashed run keeps its partial data.

## Directory layout

```
outputs/<experiment_name>/[<condition>/]<timestamp>_seed<N>[_<n>]/
```

- `<condition>` is present only when `logging.labels.condition` is set.
- `_<n>` is appended only if a directory with the same name already exists.
- With `run.replicates > 1`, each replicate is a sibling with `seed = run.seed
  + replicate`.

## Files

| File | Contents | Written when |
| --- | --- | --- |
| `config.yaml` | The source config, verbatim | Run start |
| `resolved_config.yaml` | Fully resolved config with all defaults filled in | Run start |
| `transcript.jsonl` | Canonical turns and intervention messages, one JSON object per line | Appended per turn |
| `events.jsonl` | Lifecycle and per-turn events (`run_started`, `turn`, `intervention`, `condense`, `run_completed` / `run_failed`) | Appended per event |
| `meta.json` | Run metadata and status (see below) | Updated on start/finish/fail |
| `run.log` | Per-run log at `logging.file_level` | Throughout the run |
| `llm_calls.jsonl` | One record per model call | Only with `--debug` |
| `metrics.parquet` | Long-format metric rows | Per window, when metrics are enabled |
| `embeddings.parquet` | Per-window embedding vectors | Per window, when the enabled metrics need embeddings |
| `metrics.parquet` (experiment dir) | `cross_run_sim` rows | By `renewal analyze`, not the run |

`metrics.parquet` and `embeddings.parquet` are written atomically (temp file +
`os.replace`), so a polling reader such as the [viewer](03_viewer.md) never sees a
half-written file.

## `transcript.jsonl`

Each line is a `Message` plus the `run_id`:

| Field | Meaning |
| --- | --- |
| `run_id` | The run identifier. |
| `role` | `system` / `user` / `assistant` / `intervention`. |
| `speaker` | `agent_<i>`, `system`, or the intervention name. |
| `content` | The message text. |
| `turn_index` | Global turn index; the opening topic uses `-1`. |
| `meta` | Provenance (see below). |
| `transient` | `true` for prompt-scoped messages that never enter canonical history. |

Assistant turns carry `round`, `position`, `provider`, and `model` in `meta`.
Intervention messages carry intervention-specific provenance, such as `level`,
`signal`, `threshold`, `probabilities`, `draw`, `words`, `query`,
`source_url`, `source_title`, `summarized`, `condense`, and `opening`.

A `condense` event marks a history compression: its `since_turn` is the first
turn dropped and `removed_turns` lists every dropped index. The dropped messages
remain in `transcript.jsonl`; only the loop's in-memory history is replaced by
the recap (and, for the reflective scaffolder, the new topic).

## `meta.json`

| Field | Meaning |
| --- | --- |
| `run_id` | Run identifier. |
| `seed` | Seed used for this replicate. |
| `rounds`, `window_size`, `n_agents` | Run shape. |
| `models` | The `{ provider, model }` entries of the pool. |
| `experiment_name`, `labels` | From `logging`. |
| `metrics_enabled`, `embedding_model` | Metric configuration. |
| `metrics_status` | `completed` / `partial` / `skipped`. |
| `status` | `running` → `completed` or `failed`. |
| `started_at`, `finished_at` | UTC timestamps. |
| `error` | Present when the run failed. |
| `python` | Python version. |
| `requires_env` | Env variable names the config's providers need. |
| `git_commit`, `git_dirty` | Git provenance when available. |

The [viewer](03_viewer.md) reads `meta.json` (plus `resolved_config.yaml`) to build
its filterable run index.

## `metrics.parquet`

Long-format, one row per named measure:

| Column | Meaning |
| --- | --- |
| `run_id` | Run identifier. |
| `metric` | Metric name (e.g. `semantic_anchor_sim`); sub-measures are namespaced as `metric.key`. |
| `cadence` | `per_turn` / `per_window` / `per_run`. |
| `index` | Window index (or turn index for `per_turn`). |
| `value` | The measure. |
| `window_start_round`, `window_end_round` | Round bounds (null for `per_run`). |
| `seed`, `experiment`, `condition` | Provenance. |

## `embeddings.parquet`

| Column | Meaning |
| --- | --- |
| `run_id` | Run identifier. |
| `window_index` | Window index. |
| `start_round`, `end_round` | Round bounds. |
| `embedding` | The unit-norm vector (`list<float64>`). |

`renewal analyze` reads these vectors to compute `cross_run_sim`.
