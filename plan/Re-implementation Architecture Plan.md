# Architecture of Renewal — Re-implementation Architecture Plan

Status: draft for discussion (Plan mode)
Scope: the **harness** for replicating Kong-style closed-loop collapse and testing
our interventions. Grounding and the controller metric are deferred.

---

## 0. One-paragraph summary

A small, readable closed-loop multi-agent harness. A fixed set of `N` agents
(mixed models, prompts, params) take turns in a **randomized round-robin** with
**full shared history**. Between turns, pluggable **Interventions** may append
messages (Kong's noise perturbation is the first; our novelty-scaffolding and
grounding come later). Pluggable **MetricTools** compute collapse measures with
Kong-compatible defaults. Runs are described by a base YAML config; a **sweep
file** expands a grid of overrides into many runs, so one config yields many
experiments. Everything is logged per run.

---

## 1. Design principles

1. **Readable over clever.** One obvious way to do each thing; a new person can
   follow a run end-to-end in one sitting.
2. **One interface per concept.** `LLM.generate`, `Intervention.act`,
   `MetricTool.compute`. Nothing else is load-bearing.
3. **Interventions and metrics are plug-ins**, never hard-coded branches.
4. **Uniform model interface** — MPCDF-cluster **vLLM** (open-source:
   GPT-2 XL, Qwen) and hosted APIs (OpenAI, Anthropic, Kimi, DeepSeek) are
   interchangeable. **No local model servers.**
5. **Kong-comparable by default** for the main measures; other measures opt-in.
6. **Deterministic and reproducible** — seeds, run manifests, one run = one
   self-contained artifact directory.
7. **One config → many experiments** via a sweep grid (see §4).

---

## 2. Core concepts and interfaces

### 2.1 `Message`
The only data that flows through the loop.

```python
@dataclass
class Message:
    role: str          # "system" | "user" | "assistant" | "intervention"
    speaker: str       # agent id, or "system" / intervention id
    content: str
    turn_index: int
    transient: bool = False   # True => prompt-scoped only, not part of the canonical transcript
    meta: dict = field(default_factory=dict)  # model, params, provenance
```

- `transient=True` is how Kong-style noise works: the passage is added to the
  next prompt only and then dropped, rather than becoming stored history.

### 2.2 `LLM` (uniform model interface)

```python
class LLM(Protocol):
    def generate(self, messages: list[Message], params: dict) -> str: ...
```

- Adapters: `openai` (OpenAI and any OpenAI-compatible endpoint: Kimi,
  DeepSeek), `anthropic`, `vllm` (MPCDF-cluster open-source models).
- Chosen by config; no other module knows which provider is used.

### 2.3 `Agent`
A thin bundle; does not loop.

```python
@dataclass
class Agent:
    name: str
    llm: LLM
    system_prompt: str
    params: dict            # temperature, max_tokens, top_p, seed, ...
    def respond(self, messages: list[Message]) -> Message: ...
```

### 2.4 `Scheduler` — randomized round-robin

- Each **cycle**, shuffle the `N` agents (seeded); each agent acts once.
- Full history available to every turn.
- `rounds` = cycles (config; 200–500, default 200).

### 2.5 `Intervention` — the single plug-in interface

> "Takes a message stack and appends a new message."

```python
class Intervention(Protocol):
    def act(self, messages: list[Message]) -> Message | None: ...
```

- Return `None` = no-op; return a `Message` = it is appended to the stack.
- Internal logic is irrelevant to the loop.
- The loop applies all active interventions after each agent turn, in config order.
- This one interface covers: Kong noise, our novelty nudge, and grounding/RAG.

### 2.6 `MetricTool` — the analysis plug-in interface (improved babel-ai design)

babel-ai's original: one `Analyzer` ABC + a big `SimilarityAnalyzer` + a fixed
`AnalysisResult` schema — so adding a metric means editing the schema.

Improved:

```python
class MetricTool(Protocol):
    name: str
    cadence: Literal["per_turn", "per_window", "per_run"]
    def compute(self, view: RunView) -> dict[str, float]: ...
```

- `RunView` is a **read-only** view (turns, windows, run meta, cross-run handle).
  Tools are pure: no mutation.
- Returns a flat `dict[str, float]` (namespaced by `name`).
- A `MetricRegistry` + config decides what is enabled; **no central schema edits**.
- Windowing is centralized (`window_size`, default 10) so all tools share it.
- Output is long-format: `(run_id, metric, scope, index, value)` → trivial to plot.

Example tools (initial set):
| Tool | Cadence | Notes |
| --- | --- | --- |
| `lexical_diversity` | per_window | cumulative unique unigrams (Kong) |
| `semantic_anchor_sim` | per_window | cosine(window emb, initial window emb) — Kong primary |
| `semantic_adjacent_sim` | per_window | cosine(prev window, this window) — "convergence degree" + comparison |
| `cross_run_dissim` | per_run | pairwise cosine across independent runs |
| `vendi` | per_window | Vendi score / entropy / effective support |
| `distinct_n`, `self_bleu` | per_window | surface-level contrast |
| `perplexity` | per_turn | reference-model control |
| `criticality` | per_window | branching ratio, autocorr, variance (later) |

### 2.7 `PromptSource`
Loads a file-based prompt list; assigns one system prompt per agent (seeded or
explicit), with per-run overrides layered on top (see §4.2, §6).

### 2.8 `Run` / `Recorder`
- One run = one directory:
  `config.yaml`, `transcript.jsonl`, `events.jsonl`, `metrics.parquet`, `meta.json`.
- `meta.json`: run id, seed, model versions, git commit, timestamps, labels.

---

## 3. Module layout

```
renewal/
  core/
    message.py        # Message
    agent.py          # Agent
    scheduler.py      # randomized round-robin
    loop.py           # the loop; applies agents + interventions; logs
  llm/
    base.py           # LLM protocol + retry + registry
    env.py            # setting() env resolution + load_dotenv
    config.py         # LLMConfig(provider, model) -> materialize()
    openai.py         # OpenAI + OpenAI-compatible (Kimi, DeepSeek via base_url)
    anthropic.py
    vllm.py           # MPCDF-cluster vLLM (open-source: Qwen, GPT-2 XL)
  interventions/
    base.py           # Intervention protocol
    noop.py
    noise.py          # Kong random-noise perturbation
    novelty.py        # (later) diversity-driven nudge
    grounding.py      # (later) online RAG
  metrics/
    base.py           # MetricTool protocol + registry + RunView
    lexical.py semantic.py diversity.py criticality.py
  prompts/
    loader.py
    prompts.yaml      # large, editable prompt list (global defaults)
  config.py           # base-run schema + sweep expansion
  run.py              # entry point: python -m renewal.run <config|sweep> [options]
```

---

## 4. Config & multi-experiment sweeps

Adapted from the **machine-cultural-evolution** config system, kept simpler.

### 4.1 One YAML = one run
A single base config fully describes a run. Nested, validated (Pydantic):
`run`, `agents`, `interventions`, `metrics`, `logging`. Unknown keys are errors.

### 4.2 Layered config / prompts
Precedence (MCE pattern): **experiment YAML → global defaults file → code
default**. Prompts live in a global `prompts/prompts.yaml`; a run's
`agents.prompts_override` (keyed by agent index or name) overrides entries. One
edit to the global file changes all runs.

### 4.3 Sweep file — one config, many experiments
A sweep file has a `meta`, a base `config`, and a `grid` of dotted-path
overrides. `grid` is a list of dimensions; each dimension maps a dotted path to a
list of choices (or an explicit mapping of combinations). The Cartesian product
is expanded into one run per combination.

```yaml
# sweeps/kong_models.yaml
meta:
  project_id: architecture_of_renewal
  sweep_name: kong_models

config:                                   # base run config (same schema as §4.1)
  run: { rounds: 200, replicates: 3, seed: 0, window_size: 10 }
  agents:
    n: 3
    params: { temperature: 0.9, max_tokens: 200 }   # Kong defaults
    pool:
      - { provider: openai,   model: gpt-4o-mini }
      - { provider: openai,   model: gpt-4o-mini }
      - { provider: openai,   model: gpt-4o-mini }
    prompt_source: prompts/prompts.yaml
  interventions:
    - { type: noise, rounds: [3, 6, 9, 12, 15], passages: 5 }
  metrics:
    window_size: 10
    enabled: [lexical_diversity, semantic_anchor_sim,
              semantic_adjacent_sim, cross_run_dissim, vendi]
  logging:
    out_dir: outputs/
    labels: { experiment: kong_models }

grid:
  # explicit combos swap provider+model together; simple lists vary scalars
  - agents.pool.0:
      - { provider: openai,   model: gpt-4o-mini }
      - { provider: openai,   model: gpt-5.6-sol }
      - { provider: kimi,     model: kimi-k2.6 }
  - agents.pool.1:
      - { provider: openai,   model: gpt-4o-mini }
      - { provider: deepseek, model: deepseek-flash }
      - { provider: anthropic, model: claude-opus-5-5 }
  - agents.pool.2:
      - { provider: openai,   model: gpt-4o-mini }
      - { provider: anthropic, model: claude-haiku-4-5-20251001 }
      - { provider: vllm,     model: Qwen/Qwen3-30B-A3B }
  - run.seed: [0, 1, 2]
```

### 4.4 Expansion and execution
- `renewal submit sweeps/kong_models.yaml` expands the grid and writes one run
  directory per combination, injecting `labels` (e.g. `model_0`, `seed`) for
  analysis.
- `renewal run outputs/kong_models/` runs all expanded runs; `--workers N` runs
  them in parallel locally; a single `--run <id>` runs one.
- Each run stays fully described by its own resolved `config.yaml`, so any run is
  reproducible in isolation.
- **Optional (later):** a Mongo-backed queue with `submit` + `worker` processes,
  exactly as MCE does, if we outgrow local parallelism.

### 4.5 Worked conditions
- **Kong baseline:** one model × bare loop.
- **Perturbation:** baseline + `noise` intervention.
- **Mixed models:** sweep `agents.pool.*`.
- **Our intervention (later):** swap `noise` for `novelty`/`grounding` and sweep
  its thresholds.

---

## 5. LLM API setup (adapted from machine-cultural-evolution)

MCE keeps credentials out of config, resolves them from `.env`, and gives every
provider one adapter behind a single call. We reuse that pattern.

### 5.1 Base adapter + retry
- One base class: `generate(messages, params) -> str`, plus a per-instance
  **retry wrapper** (`retry_delays`, e.g. `(0.5, 1.0)`); on exhaustion raise a
  typed error (`ReasoningExhaustedError`).
- Provider SDK retries are **disabled** so attempts are not multiplied.
- Every adapter does exactly one provider round trip.

### 5.2 Environment resolution
- `env.py`: `setting(explicit, ENV_VAR, default)` with precedence
  **explicit arg → environment → default**; `load_dotenv()` once at import.
- Secrets are never logged — log only presence (`set`/`unset`) and a
  credential-free URL (`scheme://host:port`).
- `.env` is git-ignored; `.env.example` documents every variable.

### 5.3 Provider registry
- `LLMConfig(provider, model)` is a frozen, extra-forbidding Pydantic model;
  `materialize()` maps the provider literal to an adapter constructor.
- Unknown provider fails at config validation, not at run time.
- Adding a provider = extend the literal + the mapping + one adapter.

### 5.4 Providers and env vars for our models

| `provider` | Adapter | Env vars | Serves |
| --- | --- | --- | --- |
| `openai` | `OpenAIAPI` | `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` (optional) | GPT-5.6 Sol, GPT-4o-mini |
| `anthropic` | `AnthropicAPI` | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `ANTHROPIC_MAX_TOKENS` (optional) | Claude Opus 5.5, Claude Haiku 4.5 |
| `kimi` | `OpenAIAPI` w/ default base_url | `KIMI_API_KEY`, `KIMI_MODEL`, `KIMI_BASE_URL` | Kimi K2.6 (`https://api.moonshot.ai/v1`) |
| `deepseek` | `OpenAIAPI` w/ default base_url | `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL` | DeepSeek 4.1 Flash (`https://api.deepseek.com`) |
| `vllm` | `VllmAPI` (OpenAI SDK → vLLM base URL) | `MPCDF_VLLM_ENDPOINTS` (JSON model→endpoint map), `MPCDF_VLLM_API_KEYS` (JSON, optional), `MPCDF_VLLM_ENDPOINT_URL`/`MPCDF_VLLM_MODEL` (single-endpoint fallback) | All open-source: GPT-2 XL, Qwen3-1.7B, Qwen3-30B-A3B, Qwen3-4B-Base — **one endpoint per model** |

Kimi and DeepSeek are OpenAI-compatible, so they reuse the OpenAI adapter with a
different `base_url` and key — no separate SDK. All open-source models are served
by **vLLM on the MPCDF cluster**; the config's `model` is the Hugging Face repo
id the server was launched with, and each model has its **own endpoint** (§5.6).
There is no local model server.

### 5.5 Reasoning-model handling
Reasoning models expose a separate reasoning channel (Kimi `reasoning_content`,
Anthropic extended thinking, OpenAI/DeepSeek reasoning). The adapter returns only
the **final** message text and records the reasoning in `Message.meta` so
transcripts and metrics stay clean.

### 5.6 Per-model vLLM endpoints (MPCDF)

MPCDF serves **one model per vLLM endpoint**, so endpoint selection is keyed by
model rather than being a single provider-wide URL.

- `VllmAPI` resolves its endpoint in order:
  **explicit constructor arg → `MPCDF_VLLM_ENDPOINTS[model]` → single-endpoint
  fallback `MPCDF_VLLM_ENDPOINT_URL`**.
- `MPCDF_VLLM_ENDPOINTS` is a JSON object in `.env` (deployment-specific, never
  committed):

  ```dotenv
  MPCDF_VLLM_ENDPOINTS={"Qwen/Qwen3-1.7B":"https://host-a.invalid/v1","Qwen/Qwen3-30B-A3B":"https://host-b.invalid/v1","Qwen/Qwen3-4B-Base":"https://host-c.invalid/v1","openai-community/gpt2-xl":"https://host-d.invalid/v1"}
  # optional per-endpoint keys; falls back to MPCDF_VLLM_API_KEY
  MPCDF_VLLM_API_KEYS={"https://host-b.invalid/v1":"secret"}
  ```

- Every open-source agent in a mixed run binds its own `LLM` instance to its own
  endpoint, so multiple cluster endpoints coexist in one run.
- `materialize()` for `provider: vllm` looks up the endpoint by the config's
  `model`, so the experiment YAML stays provider-free and portable.

---

## 6. Prompt list — where to get one

1. **Kong's own prompts** (Supplementary Note 1 / Extended Data Table 3): best
   for replication — environment instruction + identity profiles.
2. **Public persona/system-prompt collections**: e.g. `f/awesome-chatgpt-prompts`,
   LMSYS-Chat-1M system prompts. Convert to one `prompts.yaml`.
3. **Kong's identity profiles** (Herbert Simon, Judea Pearl, Deborah Mayo, …) are
   already in the extract and make good, citable identities.
4. Keep all of these **in one editable file** so the list grows without code
   changes; layer run-specific overrides on top (§4.2).

---

## 7. Kong replication mapping (what we copy)

### 7.1 Environment
- Minimal MAS, no task/roles/reward.
- At each round, agents act sequentially in **randomized order**.
- Agents are conditioned on: a static environment instruction, identity
  profiles, and a short-term conversation buffer of the most recent rounds.

### 7.2 Defaults
- Temperature `0.9`, max tokens `200`.
- Standard runs: **200 rounds**, 3 replicates; extended: 1000 rounds.

### 7.3 Random-noise perturbation (the method we replicate)
> **Kong's exact procedure:** append **five semantically unrelated text passages**
> to the agents' prompts at **Rounds 3, 6, 9, 12, and 15** (i.e. every 3 rounds
> early on), then let the simulation proceed with no further intervention. The
> passages are added **only during prompt construction** and are **not written
> into stored history**, so this is a **transient external disturbance**.

- In our architecture: `noise.py` is an `Intervention` that, at the configured
  rounds, returns one `Message(role="intervention", transient=True, ...)`
  containing the unrelated passages (prompt-scoped, then dropped).
- Kong's reported effect (our comparison target): local deviation at the
  perturbed windows (similarity-to-first-window `0.556 ± 0.111`), then a rapid
  return to baseline (`0.805 ± 0.011` over windows 16–25); cross-run alignment
  stays high (`0.754 ± 0.049`); sensitivity to the shock **decreases over time**.
- Config replicates the "per 3-window" cadence and the 5-passage dose.

### 7.4 Kong-compatible metrics (defaults)
- Embeddings: `text-embedding-3-large`, chunk-and-pool for long text.
- **Window-anchored semantic similarity:** cosine(window embedding, **initial** window embedding).
- **Adjacent-window similarity:** cosine(prev window, this window) → *convergence degree*.
- **Cross-run semantic dissimilarity:** average pairwise cosine across independent runs.
- **Lexical diversity:** cumulative unique unigrams.
- **Vendi** score/entropy/effective support (windowed, utterance-level).

---

## 8. Model pool — the kept set

Exactly the models we build around; all resolved through the provider registry
(§5). `type` tags (`base` / `instruct` / `reasoning`) let us analyze by category.
**Verify IDs at build time** — vendors deprecate fast.

| # | Slot | Exact model ID | Provider | Type | Access |
| --- | --- | --- | --- | --- | --- |
| 1 | Kimi K2.6 | `kimi-k2.6` | `kimi` | instruct + reasoning | API |
| 2 | Claude Opus 5.5 | `claude-opus-5-5` | `anthropic` | reasoning | API |
| 3 | Claude Haiku 4.5 | `claude-haiku-4-5-20251001` | `anthropic` | cheap reasoning | API |
| 4 | GPT 5.6 Sol | `gpt-5.6-sol` | `openai` | reasoning (SOTA) | API |
| 5 | GPT 4 Mini | `gpt-4o-mini` | `openai` | instruct | API |
| 6 | GPT 2 XL | `openai-community/gpt2-xl` | `vllm` | **base** | MPCDF vLLM |
| 7 | Qwen 3 1.7 | `Qwen/Qwen3-1.7B` | `vllm` | instruct + thinking | MPCDF vLLM |
| 8 | Qwen 30B A3B | `Qwen/Qwen3-30B-A3B` | `vllm` | MoE instruct | MPCDF vLLM |
| 9 | Qwen base | `Qwen/Qwen3-4B-Base` | `vllm` | **base** | MPCDF vLLM |
| 10 | DeepSeek 4.1 Flash | `deepseek-flash` | `deepseek` | reasoning | API |

Notes:
- Qwen base = the base (pretrained, non-instruct) slot; swap to any
  `Qwen3-*-Base` if we prefer a different size.
- Embeddings for metrics: `text-embedding-3-large` (fixed, Kong-matching).

---

## 9. Build order (small steps)

1. **Skeleton + config loader + one vLLM model (MPCDF) + loop + logging.**
2. **MetricTools**: Kong defaults (`semantic_anchor_sim`, `semantic_adjacent_sim`,
   `lexical_diversity`, `cross_run_dissim`).
3. **Kong noise `Intervention`** + baseline collapse run + first collapse curves.
4. **Provider adapters** (`openai`, `anthropic`, `kimi`, `deepseek`) + `.env` setup.
5. **Sweep expansion** (§4) + multi-model runs.
6. **Vendi / distinct-n / Self-BLEU** contrast tools.
7. (later) **novelty scaffolding** + **grounding**, criticality diagnostics.

---

## 10. Open items

- **Grounding**: deferred; online RAG scope TBD.
- **Controller metric**: deferred; will be a `MetricTool` read by the novelty
  `Intervention`.
- **Qwen base size**: default `Qwen3-4B-Base`; confirm or change.
