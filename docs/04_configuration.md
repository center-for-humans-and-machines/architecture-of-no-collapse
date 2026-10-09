# Configuration

One YAML file fully describes a run. Every level is validated with Pydantic and
forbids unknown keys, so a typo or misspelled field fails at load time rather
than being silently ignored. A config has five top-level blocks: `run`,
`agents`, `interventions`, `metrics`, and `logging`.

```yaml
run:
  rounds: 100          # number of rounds (each round = one pass over all agents)
  replicates: 1        # independent runs, seeds seed, seed+1, ...
  seed: 67             # base RNG seed
  window_size: 10      # metric window width, in rounds
  memory_turns: 20     # prompt-only history window, or null for no forgetting

agents:
  n: 1                 # must equal len(pool)
  params:
    temperature: 0.9
    max_tokens: 200
  pool:
    - { provider: vllm, model: Qwen/Qwen3-30B-A3B-Instruct-2507 }
  prompt_source: prompts/prompts.yaml
  system_prompt: "You are an agent in an open-ended multi-agent conversation."
  prompts_override: {}

interventions: []      # see Interventions

metrics:
  window_size: 10
  embedding:
    provider: vllm
    model: Qwen/Qwen3-Embedding-8B
  enabled: [lexical_diversity, semantic_anchor_sim, semantic_adjacent_sim]

logging:
  out_dir: outputs
  experiment_name: run
  labels: {}
  console_level: INFO
  file_level: DEBUG
```

## `run`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `rounds` | int ≥ 1 | `200` | Number of rounds. A round is one randomized pass over all agents, so `N` agents produce `N × rounds` turns. |
| `replicates` | int ≥ 1 | `1` | Number of independent runs. Replicate `i` uses seed `seed + i` and its own output directory. |
| `seed` | int | `0` | Base seed for the scheduler, prompt assignment, and seed-using interventions. |
| `window_size` | int ≥ 1 | `10` | Metric window width, **in rounds**, for Kong comparability. |
| `memory_turns` | int ≥ 1 or `null` | `20` | How many recent agent turns each agent sees in its prompt. `null` disables forgetting. See [Memory](#memory). |

## `agents`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `n` | int ≥ 1 | required | Number of agents. Must equal `len(pool)`. |
| `params` | mapping | Kong defaults | Per-agent decoding parameters (see below). |
| `pool` | list | required | One `{ provider, model }` entry per agent. |
| `prompt_source` | path | `null` | YAML prompt list to draw system prompts from. Resolved relative to the working directory. |
| `system_prompt` | string | see example | Code-level fallback used when no prompt id resolves. |
| `prompts_override` | mapping | `{}` | Per-agent system-prompt overrides, keyed by agent index or name. |

`params` fields (all optional): `temperature` (default `0.9`), `max_tokens`
(default `200`), `top_p`, `seed`, `frequency_penalty`, `presence_penalty`.
`None` values are dropped and not sent to the provider.

### Providers

Each entry names a `provider` (`vllm`, `azure`, or `fake`) and a `model`. A run
may mix providers and models across the pool. Credentials for the chosen
providers come from `.env` — never from the YAML. See
[LLM providers](05_llm-providers.md).

```yaml
agents:
  n: 3
  pool:
    - { provider: vllm,  model: Qwen/Qwen3-30B-A3B-Instruct-2507 }
    - { provider: azure, model: gpt-4o-mini }
    - { provider: fake,  model: fake }
```

### Prompt layering

System prompts for the `n` agents are resolved in this order (later wins):

1. A seeded default selection over the `prompt_source` list.
2. `agents.prompts_override` entries, keyed by agent index (`"0"`) or name.
3. `agents.system_prompt`, used as the code-level fallback.

The global list lives in [`prompts/prompts.yaml`](../prompts/prompts.yaml), so
one edit changes the defaults for every run.

## `interventions`

A list of `{ type, options }` blocks. Each `type` must be a registered
intervention name, and the loop applies them in config order after every agent
turn. An empty list (or omitting the block) means a bare conversation. See
[Interventions](07_interventions.md).

```yaml
interventions:
  - type: scaffolder
    options:
      threshold: 0.7
      visibility: all
      open_with_topic: false
```

## `metrics`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `window_size` | int ≥ 1 | `10` | Window width for the metric runner. Usually matches `run.window_size`. |
| `embedding` | mapping | vLLM Qwen3-Embedding-8B | Embedding client for the semantic metrics. |
| `enabled` | list of names | `[]` | Metric names to compute. Empty means no metrics unless `--metrics` is passed. |

`embedding.provider` is `vllm` or `fake`; `embedding.model` defaults to
`Qwen/Qwen3-Embedding-8B`; `embedding.dim` sets the fake embedding width (used
by the offline smoke configs). Unknown metric names fail at load. See
[Metrics](06_metrics.md).

## `logging`

| Key | Type | Default | Meaning |
| --- | --- | --- | --- |
| `out_dir` | path | `outputs` | Root directory for run folders. Overridable with `--out`. |
| `experiment_name` | string | `run` | First path component of the run directory. |
| `labels` | mapping | `{}` | Free-form labels recorded in `meta.json`. A `condition` label nests runs one level deeper. |
| `console_level` | string | `INFO` | Console log level. |
| `file_level` | string | `DEBUG` | Level for the per-run `run.log`. |

## Memory

Agents do not have to see the entire conversation. `run.memory_turns` (default
`20`) limits each prompt to the most recent N agent turns, together with any
intervention messages attached to those turns; older turns scroll out of view.
The canonical `transcript.jsonl`, `events.jsonl`, and metrics always keep the
full conversation — forgetting affects only the live prompt. Set
`memory_turns: null` to show agents the entire history.
