# Installation

## Prerequisites

| Tool | Version | Needed for |
| --- | --- | --- |
| [Python](https://www.python.org/) | 3.13 (the project pins `>=3.13,<4.0`) | the harness |
| [Poetry](https://python-poetry.org/) | 2.x | dependency and virtualenv management |
| [Node.js](https://nodejs.org/) | ≥ 20 | the [viewer](03_viewer.md) frontend only |

Only Python and Poetry are required to run experiments. The viewer frontend is
optional.

## Install the harness

From the repository root:

```bash
poetry install
```

This installs the main dependencies plus the default `dev` and `viewer`
dependency groups (pytest, FastAPI, uvicorn, …), so a single `poetry install`
covers tests and the viewer backend too.

Run commands through Poetry so they use the project virtualenv:

```bash
poetry run renewal --help
```

## Configure credentials

Copy the template and fill in only the providers you will use:

```bash
cp .env.example .env
```

`.env` is git-ignored; `.env.example` documents every variable. A run only
needs credentials for the providers named in its config:

- **vLLM (MPCDF)** — `MPCDF_VLLM_ENDPOINTS` (a JSON map of model → endpoint) or
  the single-endpoint fallback `MPCDF_VLLM_ENDPOINT_URL` / `MPCDF_VLLM_MODEL`.
- **Azure OpenAI** — `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and
  `AZURE_OPENAI_DEPLOYMENT` (plus optional `AZURE_OPENAI_API_VERSION`).
- **Tavily** — `TAVILY_API_KEY`, used only by the scaffolder interventions when
  a topic injection fires.
- The **fake** provider needs no credentials and no network.

See [LLM providers](05_llm-providers.md) for the full variable reference and
resolution order.

## Verify the install

Run the offline smoke test (no network, no credentials needed):

```bash
poetry run renewal run configs/smoke_fake.yaml
poetry run python -m pytest tests/test_config.py
```

If that succeeds, continue to the [first-run guide](02_first_run.md).
