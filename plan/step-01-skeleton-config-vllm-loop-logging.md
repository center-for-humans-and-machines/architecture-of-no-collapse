# Step 1 — Skeleton + config loader + one vLLM model (MPCDF) + loop + logging

Status: plan (locked)
Parent: [[Re-implementation Architecture Plan]] §9 build order, item 1
Repo: `/Users/mienhardt/architecture_of_renewal`

---

## 0. Definition of done

A single command runs a bare multi-agent loop on **one MPCDF vLLM model** from a
validated YAML file and leaves a **self-contained run directory**. No metrics, no
interventions, no sweeps yet — but the interfaces, config envelope, and artifact
layout are in place so Steps 2–3 slot in without rework.

End-to-end this must work against a real endpoint:

```bash
poetry run renewal run configs/smoke_vllm.yaml
# → outputs/<experiment>/<timestamp>/ {config.yaml, resolved_config.yaml,
#    transcript.jsonl, events.jsonl, meta.json, run.log}
```

and fully offline (no network, no credentials):

```bash
poetry run renewal run configs/smoke_fake.yaml
```

A pytest **integration test** (see §9) runs a short real run against the vLLM
model configured in `.env`, so the whole path is proven on MPCDF, not only
against the fake.

---

## 1. Locked decisions

| Area | Decision |
| --- | --- |
| Repo | `git init`; **Poetry**; Python `>=3.13,<4.0`; **src-layout** `src/renewal/`; console script `renewal = renewal.run:main` |
| Deps (now) | `openai`, `pydantic>=2`, `pyyaml`, `python-dotenv`; dev `pytest`, `pytest-asyncio`, `pytest-mock` (pandas/pyarrow deferred to Step 2) |
| Interface | **async** `LLM.generate`; async loop/CLI; scheduler sync |
| Config | Full envelope `run / agents / interventions / metrics / logging`, `extra="forbid"`; only `noop` registered; `metrics.enabled: []`; unknown keys/names fail at load |
| Prompts | Layered `PromptSource`: code default → `prompts/prompts.yaml` → `agents.prompts_override` (by index/name) |
| JSONL | Incremental append + flush per turn; `meta.json.status` set started→completed/failed so a crash preserves partial data |
| Round unit | **round = one randomized pass over all agents** (Kong 2026); `window_size` is in rounds (10) for Kong comparability |
| Transient | **removed** — no `transient` flag, single shared history (§3) |

### Explicit simplification vs. the architecture plan

`Message.transient` (plan §2.1) is dropped. The loop keeps a single canonical
`history`; interventions append to it. Consequence: when Step 3 implements Kong
`noise`, the five unrelated passages will be appended as **ordinary history**
rather than prompt-scoped-then-dropped (Kong's "not written into stored history"
procedure). That is a fidelity divergence to record in the methods, accepted in
favor of a simpler transcript model.

---

## 2. Repo bootstrap

Currently `/Users/mienhardt/architecture_of_renewal` is empty and not a git repo.

- `git init`; `.gitignore` covers `.env`, `outputs/`, `.venv/`, `__pycache__/`,
  `*.parquet`, `*.tmp`.
- Poetry with `python = ">=3.13,<4.0"`, package `renewal` from `src`, and the
  console script `renewal = "renewal.run:main"`.
- `.env.example` (documents every variable; `.env` itself is gitignored).
- `README.md` (quick start, offline + live run, integration test instructions).
- `configs/` and `prompts/` at repo root.

---

## 3. Module layout (Step-1 subset of plan §3)

```
src/renewal/
  core/
    message.py        # Message dataclass (no transient)
    agent.py          # Agent dataclass (async respond)
    scheduler.py      # randomized round-robin, seeded
    loop.py           # run loop: agents + interventions + recorder
  llm/
    base.py           # LLM protocol + async retry wrapper + typed error
    env.py            # setting(), safe_url(), load_dotenv()
    config.py         # LLMConfig(provider, model).materialize()
    vllm.py           # VllmAPI (text output, MPCDF endpoints)
    fake.py           # deterministic offline adapter (tests/smoke)
  interventions/
    base.py           # Intervention protocol (async)
    noop.py           # NoopIntervention
  prompts/
    loader.py         # PromptSource
    prompts.yaml      # global defaults
  config.py           # RunConfig schema + load/dump
  artifacts.py        # run dir allocation + writers + meta.json
  logging_setup.py    # console/file logging, PROGRESS level
  run.py              # CLI entry point
tests/
  test_config.py test_scheduler.py test_loop.py test_vllm.py test_artifacts.py
  integration/test_vllm_live.py
configs/
  smoke_vllm.yaml  smoke_fake.yaml
```

---

## 4. Interfaces

- **`Message`** — `role` (`system` | `user` | `assistant` | `intervention`),
  `speaker`, `content`, `turn_index`, `meta` (dict). No `transient`.
- **`LLM`** protocol — `async generate(messages: list[Message], params: dict)
  -> str`.
- **`Agent`** — `name`, `llm`, `system_prompt`, `params`; `async
  respond(history) -> Message`.
- **`Scheduler`** — seeded RNG; each round shuffles the N agents and each acts
  once; stores a global `turn_index` and `(round, position)` in message meta.
  Same seed ⇒ same order.
- **`Intervention`** — `async act(messages: list[Message]) -> Message | None`;
  applied after each agent turn in config order; a returned message is appended
  to history. Only `noop` available now.
- **`Loop`** — one canonical `history` list; prompt = `system + history` (full
  shared history). Every turn and intervention emits an event.
- **`MetricTool`** — define `RunView` / `MetricTool` stubs only (no
  implementations) so Step 2 fills them without touching the loop.

---

## 5. Config schema (plan §4.1–4.2)

`RunConfig` — Pydantic `extra="forbid"`, frozen, nested; unknown keys are errors.

- `run`: `rounds` (=cycles, default 200), `replicates` (1), `seed` (0),
  `window_size` (10).
- `agents`: `n`, `params` (default `temperature: 0.9, max_tokens: 200`),
  `pool: list[LLMConfig]`, `prompt_source`, `system_prompt` default,
  `prompts_override`.
- `interventions`: list resolved via a registry (`noop`); empty allowed.
- `metrics`: `window_size`, `enabled: []` (registry added in Step 2; empty
  allowed).
- `logging`: `out_dir`, `labels`, `console_level`, `file_level`.

Prompt layering: **code default → `prompts/prompts.yaml` →
`agents.prompts_override` (by index or name)**; unknown prompt/override keys
fail at load.

`smoke_vllm.yaml` (3 agents, 2 rounds) matches plan §4.3's base config with one
vLLM model repeated; `smoke_fake.yaml` swaps `provider: fake`.

---

## 6. LLM layer

- `env.py`: copy MCE's `setting(explicit, ENV, default)` (explicit → env →
  default, trimmed, secret-free logging) and `safe_url`; `load_dotenv()` once at
  import.
- `LLMConfig(provider: Literal["vllm", "fake"], model: str)`, frozen,
  `extra="forbid"`; `.materialize()` maps provider → adapter; unknown provider
  fails at config validation.
- `vllm.py` — `AsyncOpenAI` client, `max_retries=0`:
  - endpoint resolution order (plan §5.6): **constructor arg →
    `MPCDF_VLLM_ENDPOINTS[model]` → `MPCDF_VLLM_ENDPOINT_URL`** (JSON map parsed
    from env).
  - key: `MPCDF_VLLM_API_KEYS[endpoint]` → `MPCDF_VLLM_API_KEY` →
    `local-no-auth` placeholder.
  - `chat.completions.create(model, messages, **params)`; drop `None` params;
    return `choices[0].message.content`. Log only `safe_url(endpoint)` +
    credential presence.
  - role mapping: `system→system`, `user→user`, `assistant→assistant`,
    `intervention→user` with a speaker tag (OpenAI-compatible APIs have no
    intervention role).
- `base.py`: async retry wrapper adapted from MCE's `SchematicAPI`
  (per-instance wrapping, `retry_delays=(0.5, 1.0)`, `await asyncio.sleep`
  between retries, every exception retryable) ending in a typed
  `GenerationExhaustedError`.
- `fake.py`: deterministic async adapter (e.g. `${speaker} r${round} p${position}`),
  registered as provider `fake` so smoke tests need no network/credentials.

---

## 7. Loop semantics (Kong-aligned)

- Round = one randomized sequential pass over all agents (Kong Methods: "at each
  round, agents act sequentially in randomized order").
- Full shared history for every turn (per plan §2.4). Kong actually uses a
  short-term buffer + RAG memory; that is a noted divergence, surfaced later as
  a `history: full | last_rounds` knob (default `full`).
- `window_size` is in **rounds** (10 = 10 cycles = 30 utterances for N=3) for
  Kong comparability; Step 2 consumes it.
- Record `round`, `position`, `turn_index` per message so metric windowing can
  choose granularity later.

---

## 8. Logging & artifacts (plan §2.8)

Run dir `out_dir/<experiment>/<timestamp>[_<condition>]/` containing:

| file | content |
| --- | --- |
| `config.yaml` | verbatim source config |
| `resolved_config.yaml` | fully-resolved dump for isolated rerun |
| `transcript.jsonl` | canonical turns (message dicts), appended + flushed per turn |
| `events.jsonl` | per-turn + intervention + run lifecycle events, appended + flushed |
| `meta.json` | `run_id`, seed, rounds, models, config hash, git commit/dirty, python, `status`, labels, required-env names |
| `run.log` | console/file log; `llm_calls.jsonl` beside it under `--debug` |

Console logging adapts MCE's `PROGRESS` level + color formatter; secrets never
logged (presence only). `replicates` run as sibling dirs with `seed =
base_seed + i`. Incremental writing means a crashed run keeps its partial
transcript and `meta.json.status` is set to `failed`.

---

## 9. Verification

- **Unit:** unknown-key rejection, defaults, prompt-layering precedence;
  scheduler determinism (same seed ⇒ same order) and shuffled-order property;
  loop transcript shape with `FakeLLM`; async retry exhaustion → typed error;
  vLLM endpoint/key resolution and URL safety via monkeypatched env; artifact
  writer round-trip.
- **Smoke (offline):** `smoke_fake.yaml` runs in-process, asserts run-dir
  contents and transcript shape.
- **Integration (live vLLM):** `tests/integration/test_vllm_live.py` loads
  `configs/smoke_vllm.yaml` and runs a real short run against the `.env`
  endpoint, asserting `meta.json.status == "completed"`, the expected
  transcript length, non-empty turn content, and config round-trip. Gated with
  `pytest.mark.skipif` on the endpoint env var, so it skips cleanly where no
  `.env` exists and runs for real on the machine that has it.

`.env` (one exposed model, either shape):

```dotenv
# single-endpoint fallback (simplest for one exposed model)
MPCDF_VLLM_ENDPOINT_URL=https://<host>/v1
MPCDF_VLLM_MODEL=Qwen/Qwen3-30B-A3B
# optional
MPCDF_VLLM_API_KEY=

# or the per-model map
# MPCDF_VLLM_ENDPOINTS={"Qwen/Qwen3-30B-A3B":"https://<host>/v1"}
```

The `smoke_vllm.yaml` model id must match the id the server advertises (the id
it was launched with), since it is sent verbatim in the request.

---

## 10. CLI (plan §4.4 subset)

`renewal run <config> [--out DIR] [--seed N] [--debug] [--dry-run]`
- `--dry-run`: validate + print resolved plan (agents, models, rounds, run dir)
  without calling any LLM.
- `--out`: override `logging.out_dir`. `--seed`: override for ad-hoc replicates.
- Sweep/`submit` and parallel `--workers` are Step 5.

---

## 11. Build order

1. repo + Poetry + `.gitignore` + `.env.example`
2. `core/message.py`, `interventions/base.py` + `noop.py`
3. `llm/env.py` → `base.py` → `config.py` → `fake.py` → `vllm.py`
4. `prompts/loader.py` + `prompts.yaml`
5. `config.py` + `configs/smoke_fake.yaml`
6. `core/agent.py` → `scheduler.py` → `loop.py`
7. `artifacts.py`, `logging_setup.py`, `run.py`
8. unit tests + offline smoke (`smoke_fake.yaml`)
9. `tests/integration/test_vllm_live.py` + `configs/smoke_vllm.yaml`, run with
   the provided `.env`

---

## 12. Explicitly deferred to later steps

- Metrics (`metrics.parquet`, `MetricTool` implementations) → Step 2.
- Kong `noise` intervention → Step 3 (appends to history; see §1).
- `openai`/`anthropic`/`kimi`/`deepseek` adapters + `.env` keys → Step 4.
- Sweep expansion / `submit` / `--workers` → Step 5.
- Vector memory, referee routing, RAG, prompt-confidence — out of scope.
