# Architecture of Renewal — Documentation

This is the entry point for the harness documentation. See the
[main README](../README.md) for a one-paragraph overview and a quick start, or
the [design plans](../plan/) for the architecture and build order.

## Getting started

1. [Installation](01_installation.md) — prerequisites, `poetry install`, `.env`.
2. [First run](02_first_run.md) — run offline and live, and find your output.
3. [Running the viewer](03_viewer.md) — browse runs in the web app.

## Reference

- [Configuration](04_configuration.md) — the run YAML schema and every default.
- [LLM providers](05_llm-providers.md) — vLLM (MPCDF), Azure, and fake.
- [Metrics](06_metrics.md) — the four Kong-default measures and `renewal analyze`.
- [Interventions](07_interventions.md) — `scaffolder`, `random_scaffolder`, and `reflective_llm_scaffolder`.
- [Run artifacts](08_artifacts.md) — files and columns written per run.

## For developers

- [Architecture](09_architecture.md) — module layout and core interfaces.
- [Testing](10_testing.md) — unit, offline smoke, and live integration tests.
- Design plans — [`plan/Re-implementation Architecture Plan.md`](../plan/Re-implementation%20Architecture%20Plan.md).
