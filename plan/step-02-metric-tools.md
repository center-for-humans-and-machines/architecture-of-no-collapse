# Step 2 — MetricTools: the four Kong-default collapse measures

Status: plan (locked)
Parent: [[Re-implementation Architecture Plan]] §9 build order, item 2
Repo: `/Users/mienhardt/Programming/architecture_of_renewal`

---

## 0. Definition of done

A completed run leaves a `metrics.parquet` (long-format scalar measures) and an
`embeddings.parquet` (per-window embedding vectors), computed **incrementally
during the loop**, plus a `renewal analyze <experiment_dir>` command that fills
in the one cross-run measure. The four Kong defaults are implemented and
registered:

| Metric | Cadence | Registered name |
| --- | --- | --- |
| Lexical diversity | `per_window` | `lexical_diversity` |
| Window-anchored semantic similarity | `per_window` | `semantic_anchor_sim` |
| Adjacent-window similarity | `per_window` | `semantic_adjacent_sim` |
| Cross-run semantic similarity | `per_run` | `cross_run_sim` |

End-to-end this must work offline (no network, no credentials) and against the
real embedding endpoint:

```bash
poetry run renewal run configs/smoke_metrics.yaml   # fake LLM + fake embedding
poetry run renewal analyze outputs/smoke_metrics/    # cross_run_sim
```

and live (MPCDF vLLM embedding model), gated by the endpoint env var:

```bash
poetry run renewal run configs/smoke_vllm.yaml --metrics
```

---

## 1. Module layout (Step-2 subset of plan §3)

```
src/renewal/
  llm/
    embed.py          # EmbeddingClient protocol + VllmEmbedding + FakeEmbedding
  metrics/
    __init__.py       # registers the four built-in MetricTools
    base.py           # MetricTool protocol, RunView, windowing, MetricRunner
    lexical.py        # LexicalDiversity
    semantic.py       # SemanticAnchorSim, SemanticAdjacentSim, CrossRunSim
    analyze.py        # cross-run aggregation + `renewal analyze`
  config.py           # MetricsConfig gains `embedding`; validation
  artifacts.py        # metrics.parquet / embeddings.parquet writers
  core/loop.py        # optional window-boundary hook (structural, metric-agnostic)
  run.py              # wire MetricRunner into the loop; `analyze` subcommand
tests/
  test_metrics.py  test_embed.py  test_analyze.py
configs/
  smoke_metrics.yaml
```

---

## 2. Interfaces

### 2.1 `EmbeddingClient`

```python
class EmbeddingClient(Protocol):
    model: str
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one unit-norm vector per input text, in input order."""
```

- **`VllmEmbedding`** — `AsyncOpenAI` pointed at the embedding model's endpoint;
  calls `client.embeddings.create(model=..., input=texts)`, returns
  `[d.embedding for d in sorted(response.data, key=lambda d: d.index)]`. Endpoint
  resolution is the **same** as `VllmAPI` (Step 1 §6): explicit arg →
  `MPCDF_VLLM_ENDPOINTS[model]` → `MPCDF_VLLM_ENDPOINT_URL`; key
  `MPCDF_VLLM_API_KEYS[endpoint]` → `MPCDF_VLLM_API_KEY` → `local-no-auth`.
- **`FakeEmbedding`** — deterministic feature-hashing: each unigram is hashed
  (SHA-256, first 8 bytes) to seed a per-token RNG that draws `dim` standard
  normals; those token vectors are summed and L2-normalized. Result: cosine ≈
  token overlap, so it exercises the same code path offline and deterministically.
- Vectors are L2-normalized by the client so every cosine below is a plain dot
  product of unit vectors, regardless of whether the server normalizes.

### 2.2 `MetricTool`

```python
class MetricTool(Protocol):
    name: str
    cadence: Literal["per_turn", "per_window", "per_run"]
    async def compute(self, view: RunView, index: int) -> dict[str, float]: ...
```

`compute` receives the read-only `view` plus the scope index (window index for
`per_window`, turn index for `per_turn`, unused/0 for `per_run`) and returns a
flat `dict[str, float]` of named sub-measures. The runner records each entry as
one long-format row `{metric, cadence, index, value}`.

### 2.3 `RunView` (read-only)

```python
class RunView:
    run_id: str
    seed: int
    experiment: str
    condition: str | None
    window_size: int
    def window_text(self, index: int) -> str          # concatenated assistant turns
    def window_turns(self, index: int) -> list[Message]
    def embedding(self, text: str) -> Awaitable[...]  # cached per text
    @property
    def n_windows(self) -> int
```

---

## 3. Config schema

`MetricsConfig` gains an embedding sub-block; `enabled` names validate against
the existing `Registry` (kind `metric`; unknown names fail at load, mirroring
interventions):

```yaml
metrics:
  window_size: 10                                  # in rounds
  embedding:
    provider: vllm                                  # vllm | fake
    model: Qwen/Qwen3-Embedding-8B
    dim: 4096                                       # fake-embedding width; vllm returns native
  enabled: [lexical_diversity, semantic_anchor_sim,
            semantic_adjacent_sim, cross_run_sim]
```

`cross_run_sim` validates as a registered name but is **skipped** by the in-run
runner (it needs multiple runs); `renewal analyze` owns it.

---

## 4. Windowing (shared by every per_window tool)

1. Filter the canonical transcript to `role == "assistant"` messages (agent
   turns only; system and intervention messages are excluded).
2. Group those turns by `meta["round"]`.
3. Window `i` spans rounds `[i*W, (i+1)*W)` where `W = window_size`.
   `n_windows = ceil(rounds / W)`; the last window may be partial.
4. Window text = `"\n".join(turn.content for turn in turns in turn order)`.
   (No speaker prefixes; content only. Revisit if speaker identity turns out to
   matter for the embedding.)
5. A window with no assistant turns has no embedding; its metrics are `NaN`.

Because `window_size` is in **rounds**, a window of 10 rounds with N=3 agents
holds up to 30 utterances — the Kong-comparable granularity.

---

## 5. The metrics in detail

All measures are **similarities** (cosine over unit-norm embeddings, or a token
ratio), so a higher value means "more alike / less collapsed" throughout; this
keeps every curve directly comparable to Kong's reported numbers, which are
similarities.

### 5.1 `lexical_diversity` — cumulative unique unigrams

**Purpose.** The surface-level redundancy signal. As a population collapses
semantically, its vocabulary contracts too: agents recycle the same phrases and
stop introducing new words. Lexical diversity is the cheap, model-free
counterpart to the embedding measures.

**Tokenization.** A "unigram" is a word token produced by: lowercasing, splitting
on whitespace, stripping leading/trailing non-alphanumeric characters, and
dropping tokens that become empty. No stemming or stop-word removal. This is
deliberately simple and deterministic so the number is reproducible.

**State (accumulated across the whole run).** Two counters:
- `unique: set[str]` — every unigram seen so far.
- `total: int` — every unigram occurrence so far.

**Formula.** At the end of window `i`, after ingesting that window's tokens:

```
lexical_diversity[i] = |unique up to and including window i|
                       --------------------------------------
                       total   up to and including window i
```

**Interpretation.** Starts high and is monotonically non-increasing (cumulative
type–token ratio always decreases as the corpus grows — a well-known artifact).
The collapse signal is therefore *relative*: under collapse it falls faster and
further than a healthy baseline, and its trajectory across windows (especially
the drop at perturbed windows and the rate of recovery) is what we compare to
Kong. We report the ratio as-is, exactly Kong's "cumulative unique unigrams"
measure, and treat it as a contrast curve rather than an absolute collapse
indicator.

**Edge cases.** Window with no tokens contributes nothing to `unique`/`total` and
is reported as the current cumulative value (not `NaN`), because the measure is
defined cumulatively. A run with zero total tokens yields `NaN`.

### 5.2 `semantic_anchor_sim` — window-anchored similarity to the first window

**Purpose.** The **primary collapse measure**. It tracks how far the
conversation drifts from its *initial* semantic state. Kong's headline numbers
(similarity-to-first-window `0.556 ± 0.111` at perturbed windows, returning to
`0.805 ± 0.011` over windows 16–25) are this measure.

**Formula.** For each window `i`, with embedding `v_i`:

```
semantic_anchor_sim[i] = cosine(v_i, v_0)
```

where `v_0` is the embedding of window **0** (the first window of the run).

**Interpretation.**
- `anchor_sim[0] = 1.0` by definition.
- Values near `1.0` = the conversation still lives in the same semantic
  neighborhood it started in (no collapse).
- A sustained decline = the population has moved onto a new, self-reinforcing
  topic and is drifting away from the original state. Kong's transient-noise
  result — a local dip at the perturbed windows followed by a return to
  baseline — is read directly off this curve.

**Incremental state.** Cache `v_0` (computed once at window 0). Every later
window embeds only its own text and dots it against the cached `v_0`. No
re-embedding of earlier windows.

**Edge cases.** Window 0 has no assistant turns → every `anchor_sim` is `NaN`
(there is no anchor). A window with no turns → `NaN` for that window.

### 5.3 `semantic_adjacent_sim` — similarity to the previous window

**Purpose.** The **convergence-degree** measure, and the discriminator between
two kinds of drift. It complements `anchor_sim`: anchor similarity tells you
*how far from the start* you are; adjacent similarity tells you *how smoothly*
you got there.

**Formula.** For each window `i >= 1`, with embedding `v_i`:

```
semantic_adjacent_sim[i] = cosine(v_i, v_{i-1})
```

`adjacent_sim[0]` is undefined and reported as `NaN`.

**Interpretation.**
- **High adjacent + dropping anchor** = smooth drift: the population is
  converging onto a new attractor, step by step (Kong's "convergence degree"
  rising as the system settles).
- **Low adjacent** = abrupt jumps between windows — either an intervention
  knocked the system off course, or the population is not yet stable.
- Kong uses this as both a collapse measure and a comparison target; a rising
  adjacent-similarity curve is the signature of convergence.

**Incremental state.** Cache `v_{i-1}`; on window `i`, embed `v_i` and dot
against the cached previous vector, then overwrite the cache.

**Edge cases.** `i = 0` → `NaN`. A window with no turns → `NaN` for that window
(and, since it is not embedded, the *next* window's adjacent similarity is also
`NaN` — the "previous" is missing).

### 5.4 `cross_run_sim` — cross-run semantic similarity

**Purpose.** Whether collapse is **deterministic**. If `R` independent runs of
the same experiment (same config, different seeds) all converge to the *same*
attractor, their transcripts are semantically aligned and cross-run similarity
is high. If collapse is chaotic, runs land in different places. Kong reports
"cross-run alignment stays high (`0.754 ± 0.049`)".

**Cadence.** `per_run`, but computed **across** runs, so it is the only measure
that cannot be produced inside a single run. `renewal analyze` owns it.

**Inputs.** The `embeddings.parquet` of `R` sibling run dirs under one
experiment directory (grouped by condition label, default: all runs in the
directory).

**Formula.** Let `v_i^r` be the embedding of window `i` of run `r`, and let
`W` be the number of window indices shared by every run in the group (the
minimum over runs; for replicates of one config they match exactly):

```
cross_run_sim = mean over all run pairs (r_a, r_b), r_a < r_b,
                     over all windows i in 0..W-1,
                     of cosine(v_i^r_a, v_i^r_b)
```

**Output.** A single similarity value, reported in the same direction as the
other three metrics, so it maps directly onto Kong's `0.754 ± 0.049`.
`analyze` emits one per-run row (the run's contribution to the group mean) plus
one group-level row.

**Interpretation.**
- `cross_run_sim` near `1.0` = deterministic collapse; the system reliably lands
  on the same attractor. This is Kong's finding.
- Low `cross_run_sim` = run-to-run variance dominates; the collapse is sensitive
  to seed / turn order, which matters for how we later evaluate interventions.

**Edge cases.** `R < 2` runs in a group → no pairs, metric is `NaN` (and
`analyze` warns). Runs with differing window counts → align on the shared prefix
`W = min(...)`, and note the truncation in the group record. A missing
`embeddings.parquet` (e.g. metrics were disabled) → the run is skipped and
reported.

---

## 6. Embedding layer detail

- **Model.** `Qwen/Qwen3-Embedding-8B`. Served by vLLM with `--task embed`
  (vLLM ≥ 0.8.5); exposes the OpenAI-compatible `POST /v1/embeddings` route and
  returns 4096-dim vectors by default (MRL truncation is *not* used; we keep the
  native 4096). The model id is sent verbatim, so the config id must match the
  id the server was launched with — exactly the Step-1 vLLM rule. Default in
  config, swapped via config/env (never code); fallbacks: `Qwen/Qwen3-Embedding-4B`
  (2560-dim), `nvidia/NV-Embed-v2` (4096-dim, English).
- **Endpoint.** Same per-model resolution as `VllmAPI` (Step 1 §6). The embedding
  model id becomes another key in `MPCDF_VLLM_ENDPOINTS` (or falls back to
  `MPCDF_VLLM_ENDPOINT_URL`), so no new env-var family is introduced.
- **Chunk-and-pool (Kong).** If a window's text would exceed the model's context
  (32k tokens for Qwen3-Embedding-8B), split into overlapping chunks, embed each,
  and average-pool the vectors. With `max_tokens=200` per utterance and 30
  utterances per window (≈6k tokens) this path is rarely hit, but it is
  implemented as the safety net so long runs don't silently truncate.
- **Normalization.** All returned vectors are L2-normalized before storage so
  cosine is a dot product and the `embeddings.parquet` values are unit vectors.
- **`fake` provider.** Deterministic feature-hashing (§2.1), used by
  `configs/smoke_metrics.yaml` and all offline tests. It never appears in a live
  run.

---

## 7. Incremental computation pipeline

`Loop` stays metric-agnostic; it gains one *structural* hook:

```python
class Loop:
    def __init__(self, agents, scheduler, interventions, recorder,
                 *, window_size: int | None = None,
                    on_window: Callable[[int, list[Message]], Awaitable[None]] | None = None):
        ...
```

After finishing round `r`, if `window_size` is set and `(r + 1) % window_size ==
0`, the loop calls `await on_window(window_index, turns)` with the assistant
turns of the just-completed window. The loop computes nothing metric-related.

`MetricRunner` (in `metrics/base.py`) is the subscriber:

1. **`on_window(i, turns)`**
   - Build window text from assistant turns (§4).
   - If text is empty, record `NaN` rows and return.
   - Embed the text once (`await view.embedding(text)`).
   - `semantic_anchor_sim`: dot against cached `v_0` (compute and cache `v_0`
     at `i == 0`; `anchor_sim[0] = 1.0`).
   - `semantic_adjacent_sim`: dot against cached `v_{i-1}` (`NaN` at `i == 0`),
     then cache `v_i`.
   - `lexical_diversity`: update `unique`/`total` counters with window `i`
     tokens, compute ratio.
   - Append the window's rows to `metrics.parquet` and its vector to
     `embeddings.parquet`; flush both.
2. **`finalize()`** (called once after `loop.run` returns) — handles the partial
   last window (when `rounds % window_size != 0`) by running `on_window` for the
   final index, and records `meta["metrics_status"] = "completed"`.

**Failure handling.** Each window's embedding/metric computation is wrapped; a
failure logs a warning, records `meta["metrics_status"] = "partial"` with the
error, and does **not** propagate to the run (the transcript is already
committed). A failed embedding leaves `NaN` rows for that window.

**Why incremental.** Each window's measures are available the moment its window
closes rather than in a single post-run batch, and memory stays bounded (only
`v_0`, `v_{i-1}`, and the lexical counters are held — not the full embedding
history). It also keeps the window-boundary hook in place for the live
visualization server we add later. The one tradeoff: embeddings are issued
one-per-window during the run rather than in a single post-run batch; they are
async and do not block turns.

---

## 8. Artifacts

Each run dir gains, alongside the Step-1 files:

| file | content |
| --- | --- |
| `metrics.parquet` | long-format rows `run_id, metric, cadence, index, value, window_start_round, window_end_round, seed, experiment, condition` (window cols null for `per_run`). Appended + flushed per window. |
| `embeddings.parquet` | `run_id, window_index, start_round, end_round, embedding (list<float64>, unit norm)`. Appended + flushed per window. |

`meta.json` gains `metrics_status` (`"completed"` | `"partial"` | `"skipped"`),
`embedding_model`, and the set of enabled metric names.

`renewal analyze <experiment_dir>` writes an experiment-level `metrics.parquet`
in the experiment directory containing the `cross_run_sim` rows (one per run
plus one group-level), keyed by `run_id` and `condition`.

---

## 9. CLI

```
renewal run <config> [--out DIR] [--seed N] [--debug] [--dry-run] [--metrics]
  --metrics        force metric computation on (default: on when metrics.enabled
                   is non-empty)

renewal analyze <experiment_dir> [--condition LABEL]
  compute cross_run_sim over sibling run dirs
```

---

## 10. Verification

- **Unit (`test_embed.py`)** — `FakeEmbedding` determinism (same text ⇒ same
  vector) and monotonic cosine (overlapping tokens ⇒ higher cosine than
  disjoint); `VllmEmbedding` request shaping + index-ordered return + endpoint/key
  resolution via monkeypatched env.
- **Unit (`test_metrics.py`)** — windowing (round grouping, partial last window,
  assistant-only filtering, empty-window `NaN`); `lexical_diversity` math on
  hand-built token sequences (including the cumulative-ratio property);
  `semantic_anchor_sim` and `semantic_adjacent_sim` cosines against `FakeEmbedding`
  (known vectors); long-format row shape; config validation (unknown metric name
  and unknown embedding key rejected).
- **Unit (`test_analyze.py`)** — `cross_run_sim` on synthetic
  `embeddings.parquet` for `R=2..4` runs; `R<2` → `NaN`; window-count mismatch →
  shared-prefix alignment.
- **Smoke (offline)** — `configs/smoke_metrics.yaml` (fake LLM + fake embedding,
  `rounds=30` ⇒ 3 windows, all four metrics enabled) runs in-process and asserts
  `metrics.parquet`/`embeddings.parquet` contents and row counts; `analyze` over
  two such runs (different seeds) produces finite `cross_run_sim`.
- **Integration (live)** — `tests/integration/test_metrics_live.py` loads
  `smoke_vllm.yaml` with metrics on, gated with `pytest.mark.skipif` on the
  embedding endpoint env var, asserting `meta["metrics_status"] == "completed"`
  and non-empty `embeddings.parquet`.

---

## 11. Build order

1. deps (`numpy`, `pandas`, `pyarrow`) in `pyproject.toml`.
2. `llm/embed.py` (`EmbeddingClient`, `VllmEmbedding`, `FakeEmbedding`).
3. `metrics/base.py` (`MetricTool`, `RunView`, windowing, `MetricRunner`).
4. `metrics/lexical.py` + `metrics/semantic.py` + `metrics/__init__.py`.
5. `config.py` schema + `artifacts.py` writers + `core/loop.py` hook + `run.py`
   wiring.
6. `metrics/analyze.py` + `renewal analyze`.
7. unit tests + `configs/smoke_metrics.yaml` offline smoke.
8. `tests/integration/test_metrics_live.py` (runs once the embedding endpoint is up).

---

## 12. Open items / assumptions

- **Tokenization** of unigrams (lowercase, strip punctuation, drop empties) is a
  reasonable default but not verified against Kong's exact tokenizer; revisit if
  strict replication is required.
- **Window text** concatenates raw `content` without speaker prefixes; if speaker
  identity measurably changes the embedding, prepend `speaker:`.
- **Cross-run window alignment** defaults to the shared-prefix `W = min(...)`;
  alternative (compare only the final window, or a whole-run pooled embedding) is
  deferred and can be a `--mode` flag on `analyze`.
- **Chunk-and-pool** chunk size / overlap defaults are placeholders until a run
  actually exceeds the 32k context.
- **`Qwen/Qwen3-Embedding-8B` endpoint** must be served on MPCDF before the live
  integration test can run; offline coverage is complete via `fake`.

---

## 13. Explicitly deferred to later steps

- Live visualization server (`renewal serve`, FastAPI + Plotly.js) → later,
  together.
- `vendi`, `distinct_n`, `self_bleu` contrast tools → Step 6.
- `perplexity` (reference-model control) and `criticality` diagnostics → later.
- `openai`/`anthropic`/`kimi`/`deepseek` LLM adapters → Step 4 (the embedding
  layer is vLLM-only by design).
- Kong `noise` intervention → Step 3 (its transient-vs-history divergence is
  already noted in Step 1 §1).
- Sweep expansion / `submit` / `--workers` → Step 5.
