# Architecture of Renewal

Closed-loop multi-agent harness for reproducing Kong-style semantic collapse
and testing interventions. A fixed set of `N` agents take turns over a shared
conversation; pluggable interventions steer it, and pluggable metrics measure
how it collapses. Every run is deterministic and leaves a self-contained
artifact directory, and a bundled web viewer browses and compares runs.

Design notes and the step-by-step build plan live in [`plan/`](plan/).

## Quick start

Requires Python 3.13 and [Poetry](https://python-poetry.org/). Node.js ≥ 20 is
only needed for the viewer.

```bash
poetry install
cp .env.example .env   # endpoints and credentials for live runs

# Offline: deterministic fake model, no network or credentials.
poetry run renewal run configs/smoke_fake.yaml

# Live: one MPCDF vLLM model, endpoint from .env.
poetry run renewal run configs/smoke_vllm.yaml
```

Outputs land in `outputs/<experiment>/<timestamp>_seed<N>/`. See the
[first-run guide](docs/02_first_run.md) for details.

## Documentation

Full documentation is in [`docs/`](docs/00_index.md):

| Guide | Contents |
| --- | --- |
| [Installation](docs/01_installation.md) | Prerequisites, dependencies, `.env` setup |
| [First run](docs/02_first_run.md) | Offline and live runs, dry runs, where output goes |
| [Viewer](docs/03_viewer.md) | Browse and compare runs in the web app |
| [Configuration](docs/04_configuration.md) | The run YAML reference and defaults |
| [LLM providers](docs/05_llm-providers.md) | vLLM (MPCDF), Azure, and fake; credentials and env vars |
| [Metrics](docs/06_metrics.md) | The four Kong-default collapse measures and `analyze` |
| [Interventions](docs/07_interventions.md) | The `scaffolder` and `random_scaffolder` policies |
| [Run artifacts](docs/08_artifacts.md) | Every file and column a run writes |
| [Architecture](docs/09_architecture.md) | Module layout and core interfaces |
| [Testing](docs/10_testing.md) | Unit, offline smoke, and live integration tests |

## Project layout

```
src/renewal/   # harness package: loop, config, llm, metrics, interventions
configs/       # example run configurations
prompts/       # global system-prompt defaults
viewer/        # FastAPI + Svelte run viewer (separate process)
plan/          # design and implementation plans
docs/          # user and developer documentation
tests/         # unit, offline smoke, and live integration tests
```

## License

MIT (declared in `pyproject.toml`).
