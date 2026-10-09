# Architecture

The harness is a small, readable closed-loop multi-agent system. A fixed set of
`N` agents take turns in a seeded randomized order over one shared history;
interventions append steering messages between turns; metrics subscribe to
window boundaries. The full design rationale and build order live in
[`plan/`](../plan/).

## Module layout

```
src/renewal/
  core/
    message.py     # Message — the only data that flows through the loop
    agent.py       # Agent — model + prompt + params; does not loop
    scheduler.py   # seeded randomized round-robin
    loop.py        # the loop: agents + interventions + window hook
    text.py        # tokenizer shared by metrics and the fake embedder
  llm/
    base.py        # LLM protocol + retry wrapper
    env.py         # setting() resolution + load_dotenv()
    config.py      # LLMConfig(provider, model).materialize()
    vllm.py        # MPCDF vLLM adapter
    azure.py       # Azure OpenAI adapter
    fake.py        # deterministic offline adapter
    embed.py       # EmbeddingClient + VllmEmbedding + FakeEmbedding
  interventions/
    base.py        # Intervention protocol + HistoryReplacement
    noop.py        # no-op plugin
    scaffolder/    # similarity-driven three-level scaffolder
    random_scaffolder/  # probability-driven sibling
    reflective_llm_scaffolder/  # LLM-authored sibling (summarize + condense)
  metrics/
    base.py        # MetricTool, RunView, windowing, MetricRunner
    lexical.py     # lexical_diversity
    semantic.py    # semantic_anchor_sim, semantic_adjacent_sim, cross_run_sim
    analyze.py     # cross-run aggregation (`renewal analyze`)
  prompts/         # PromptSource loader + prompts.yaml
  config.py        # RunConfig schema + load/dump
  artifacts.py     # run dir allocation, writers, meta.json
  registry.py      # named plug-in catalog
  logging_setup.py # console/file logging
  run.py           # CLI entry point
```

## Core concepts

**`Message`** is the only record of a turn: `role` (`system`, `user`,
`assistant`, `intervention`), `speaker`, `content`, `turn_index`, `meta`, and
`transient`. A `transient` message is prompt-scoped — shown to the next agent
and then dropped, never entering canonical history.

**`Agent`** bundles a model (`LLM`), a system prompt, and decoding params. Its
`respond` builds `system + history` and returns one `assistant` message.

**`Scheduler`** shuffles the agent order once per round using a seeded RNG, so
the same seed reproduces the same order.

**`Loop`** holds one canonical `history`. Scoped to each round, it calls each
agent once (in the shuffled order), then runs every intervention in config
order. A returned message is appended to `history`, or to a transient buffer if
flagged. Before turn 0 it runs each intervention's optional `prime` hook. When a
window boundary is reached it calls the metric runner's `on_window` hook.

**`Intervention`** is the single plug-in interface: `act(messages) ->
Message | None`. The loop (and metrics) know nothing about an intervention's
internals.

**`MetricTool`** declares a `name`, a `cadence`, whether it needs embeddings,
and `compute(view, index) -> dict[str, float]`. A `RunView` is a read-only view
of one run's turns, windows, and embedder. The `MetricRunner` computes enabled
per-window tools incrementally and writes the long-format artifacts.

## Plug-in registry

Interventions and metrics are registered by name at import time
(`@Registry.register("intervention", "scaffolder")`) and selected by name in the
config. Unknown names fail at config load, not at run time. Adding a plug-in is
a matter of implementing the interface and registering it.

## Reproducibility

A run's seed drives the scheduler order, prompt assignment, and any
seed-using intervention. `resolved_config.yaml` plus `meta.json` (including the
git commit and dirty flag) fully describe a run, so any run can be reproduced in
isolation. See [Run artifacts](08_artifacts.md).

## See also

- [Re-implementation Architecture Plan](../plan/Re-implementation%20Architecture%20Plan.md)
- [Step 1 — skeleton, config, vLLM, loop, logging](../plan/step-01-skeleton-config-vllm-loop-logging.md)
- [Step 2 — metric tools](../plan/step-02-metric-tools.md)
