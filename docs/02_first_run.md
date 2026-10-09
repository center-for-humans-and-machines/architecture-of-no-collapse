# First run

The CLI has two commands. `renewal run` executes a run config; `renewal
analyze` computes the cross-run metric over completed runs.

```bash
poetry run renewal run <config.yaml>      [--out DIR] [--seed N] [--debug] [--dry-run] [--metrics]
poetry run renewal analyze <experiment_dir> [--condition LABEL]
```

## 1. Run offline (no network)

The fake provider is deterministic and needs no credentials. This is the
fastest way to confirm everything works:

```bash
poetry run renewal run configs/smoke_fake.yaml
```

It runs 3 agents for 3 rounds and writes a run directory under `outputs/`.

## 2. Validate a config without calling a model

`--dry-run` loads and validates the config, then prints the resolved plan
(rounds, replicates, memory, agents and models, interventions, metrics, output
directory) without making any network call:

```bash
poetry run renewal run configs/smoke_vllm.yaml --dry-run
```

## 3. Run live

With your `.env` pointing at a reachable endpoint
(see [LLM providers](05_llm-providers.md)):

```bash
poetry run renewal run configs/smoke_vllm.yaml
```

The model id in the config must match the id the vLLM server advertises,
because it is sent verbatim in the request.

The other run configs cover more of the harness:

| Config | What it exercises |
| --- | --- |
| `configs/smoke_metrics.yaml` | Offline fake LLM + fake embedding, all four metrics |
| `configs/scaffolder_smoke.yaml` | Offline scaffolder (open topic and inject disabled) |
| `configs/random_scaffolder_smoke.yaml` | Offline random scaffolder, `inject: 0.0` |
| `configs/reflective_llm_scaffolder_smoke.yaml` | Offline reflective LLM scaffolder, `inject: 0.0` |
| `configs/smoke_azure.yaml` | One Azure deployment, credentials from `.env` |
| `configs/run_qwen.yaml` | A real 100-round Qwen run with metrics |
| `configs/run_qwen_scaffolder.yaml` | Real run + similarity-driven scaffolder |
| `configs/run_qwen_random_scaffolder.yaml` | Real run + random scaffolder |
| `configs/run_qwen_random_scaffolder_socratic.yaml` | Same, but the `deepen` level draws from the Socratic question bank |
| `configs/run_qwen_reflective_llm_scaffolder.yaml` | Real run + LLM-driven reflective scaffolder |
| `configs/run_qwen_no_scaffolding.yaml` | Control run with no intervention |
| `configs/run_azure_random_scaffolder.yaml` | Azure agent + random scaffolder |
| `configs/run_deepseek_random_scaffolder.yaml` | DeepSeek agent + random scaffolder |

## Where the output goes

One run = one self-contained directory:

```
outputs/<experiment_name>/<timestamp>_seed<N>/
```

If the config sets `logging.labels.condition`, runs are nested one level deeper
under that condition:

```
outputs/<experiment_name>/<condition>/<timestamp>_seed<N>/
```

When `run.replicates > 1`, each replicate gets its own sibling directory with
`seed = run.seed + replicate`. Every artifact is documented in
[Run artifacts](08_artifacts.md). Replicates run concurrently by default; set
`run.parallel` to cap how many run at once.

`--out DIR` overrides `logging.out_dir`; `--seed N` overrides `run.seed`.

## Metrics

Metrics run automatically when `metrics.enabled` is non-empty. `--metrics`
forces them on and, for a config with an empty `metrics.enabled`, uses the Kong
default set. See [Metrics](06_metrics.md).

## Debugging

`--debug` raises the console log level and writes `llm_calls.jsonl` next to the
other run files, recording each model call.

## Memory

`run.memory_turns` (default `20`) limits each agent's prompt to the most recent
N agent turns, plus any intervention messages attached to those turns. Older
turns scroll out of view of the prompt but are never dropped from the canonical
transcript. Set `memory_turns: null` to give agents the full history. See
[Configuration](04_configuration.md).
