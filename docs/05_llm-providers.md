# LLM providers

Each agent names a `provider` and a `model` in the config `pool`. Supported
providers are `vllm` (default, MPCDF cluster), `azure`, and `fake`. Credentials
and endpoints never live in the YAML — they are resolved from `.env` (loaded
once at import via `python-dotenv`). [`.env.example`](../.env.example)
documents every variable.

```yaml
agents:
  pool:
    - { provider: azure, model: gpt-4o-mini }   # model = Azure deployment name
```

Resolution precedence for every adapter is: **explicit constructor argument →
environment variable → caller default**. Secrets are never logged; only
presence (`set` / `unset`) and a credential-free URL are.

## `vllm` — MPCDF cluster

Open-source models (Qwen, GPT-2 XL, …) served by vLLM on the MPCDF cluster.
MPCDF serves **one model per endpoint**, so the endpoint is selected by model
id.

- Endpoint resolution: `MPCDF_VLLM_ENDPOINTS[model]` (a JSON map), falling back
  to the single-endpoint `MPCDF_VLLM_ENDPOINT_URL`.
- API key resolution: `MPCDF_VLLM_API_KEYS[endpoint]` (a JSON map), falling back
  to `MPCDF_VLLM_API_KEY`, falling back to the `local-no-auth` placeholder.
- The YAML `model` is the model id the server was launched with — it is sent
  verbatim, so it must match the id the server advertises.

```dotenv
# Simplest: one exposed model.
MPCDF_VLLM_ENDPOINT_URL=https://<host>/v1
MPCDF_VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
MPCDF_VLLM_API_KEY=           # optional

# Per-model endpoint map (deployment-specific, never committed).
MPCDF_VLLM_ENDPOINTS={"Qwen/Qwen3-30B-A3B-Instruct-2507":"https://host-b/v1","Qwen/Qwen3-Embedding-8B":"https://host-e/v1"}
MPCDF_VLLM_API_KEYS={"https://host-b/v1":"secret"}
```

`meta.json` records `MPCDF_VLLM_ENDPOINT_URL` / `MPCDF_VLLM_MODEL` as required
env names whenever the pool uses `vllm`.

## `azure` — Azure OpenAI

Wraps the OpenAI SDK's `AsyncAzureOpenAI` client. Two Azure specifics matter:

- Requests address a **deployment**, not a model id. The YAML `model` is the
  deployment name, and it is sent in the request's `model` field.
- The API surface is versioned, so a version must be pinned.

- `model` (deployment) falls back to `AZURE_OPENAI_DEPLOYMENT`.
- `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` are both required.
- `AZURE_OPENAI_API_VERSION` defaults to `2024-12-01-preview`.

```yaml
agents:
  pool:
    - { provider: azure, model: gpt-4o-mini }
```

`meta.json` records `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and
`AZURE_OPENAI_DEPLOYMENT` as required env names whenever the pool uses `azure`.

## `fake` — offline deterministic model

A deterministic, network-free adapter used by the offline smoke configs and all
unit tests. It needs no credentials and never appears in a live run.

```yaml
agents:
  pool:
    - { provider: fake, model: fake }
```

## Embeddings

The semantic metrics embed each window with a separate embedding model. It uses
the same per-model vLLM endpoint resolution as the chat models, so its id is
just another key in `MPCDF_VLLM_ENDPOINTS`:

```dotenv
MPCDF_VLLM_ENDPOINTS={"Qwen/Qwen3-30B-A3B-Instruct-2507":"https://host-b/v1","Qwen/Qwen3-Embedding-8B":"https://host-e/v1"}
```

`Qwen/Qwen3-Embedding-8B` is served by vLLM with `--task embed`. Configure it
under `metrics.embedding`:

```yaml
metrics:
  embedding:
    provider: vllm
    model: Qwen/Qwen3-Embedding-8B
```

For fully offline runs set `provider: fake` (optionally with `dim`). See
[Metrics](06_metrics.md).

## Tavily

The scaffolder interventions sample common words and run one search per topic
injection. `TAVILY_API_KEY` is used only when such an injection fires, so it is
optional for runs that never inject a topic. See
[Interventions](07_interventions.md).
