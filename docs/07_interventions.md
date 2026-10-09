# Interventions

An intervention is a plug-in that appends steering messages to the conversation.
The loop applies the configured interventions in order after every agent turn,
and each intervention may also inject an opening message before turn 0. Several
interventions ship with the harness (`noop`, `scaffolder`, `random_scaffolder`,
`reflective_llm_scaffolder`, `memory_scaffolder`); an empty `interventions` list
runs a bare conversation.

The scaffolders share the same three levels. `scaffolder` and
`random_scaffolder` steer with fixed prompt families; `reflective_llm_scaffolder`
has an LLM author them at runtime; `memory_scaffolder` adds a fourth level that
resurfaces a stored memory.

| Level | Prompt family (fixed scaffolders) |
| --- | --- |
| `deepen` | "Add one concrete detail.", "Give one specific example.", "Develop one consequence.", "Name one edge case." |
| `innovate` | "Introduce a genuinely new idea…", "Take this topic in an unexpected…", "Add a new concept…", "Find a non-obvious neighboring idea…" |
| `inject` | Sample common words → Tavily search → surface the top hit as a new topic |

## `visibility`

`visibility` controls whether steering messages persist in canonical history:

| Value | Behavior |
| --- | --- |
| `all` (default) | Every message is canonical history. |
| `injections` | Only level-3 injections persist; deepen/innovate nudges are prompt-scoped and dropped after the next turn. |
| `transient` | All steering messages are prompt-scoped. |

Prompt-scoped messages are shown to the next agent's prompt and then dropped;
they are still recorded in the transcript and event stream.

## Opening topic

Both interventions can inject a grounded **opening topic** before turn 0 (the
same common-word → Tavily pipeline), so the first agent responds to a concrete
starting point instead of generating from an empty history. The opening is
tagged `meta.opening: true` with `turn_index: -1`, honors `visibility`, and
falls back to a generic opening prompt if the topic pipeline fails. Disable it
with `open_with_topic: false` (the offline smoke configs do, to stay
network-free).

The visible injection text contains only the search excerpt and the instruction
— the source title and URL are kept in `meta` but never shown to the model. For
`scaffolder` and `random_scaffolder` the excerpt is Tavily's full parsed page
text (`raw_content`), preferred over its short snippet; `max_excerpt_chars` caps
it (default `null` uses the full text), trimmed back to the last sentence
boundary. `reflective_llm_scaffolder` instead has its scaffolding LLM summarize
the raw text (see below).

## `scaffolder`

A three-level novelty controller that reacts each turn to the **adjacent
similarity** of the latest model turn (cosine vs. the previous turn; higher =
more repetition = more collapse):

- `deepen` when adjacent similarity `< threshold`
- `innovate` on the first turn where similarity `>= threshold`
- `inject` on the next turn where similarity `>= threshold` again

```yaml
interventions:
  - type: scaffolder
    options:
      threshold: 0.85
      visibility: all
      open_with_topic: false
      seed: 0
      embedding: { provider: fake, model: fake, dim: 64 }
```

| Option | Default | Meaning |
| --- | --- | --- |
| `threshold` | `0.85` | Adjacent-similarity cutoff between `deepen` and `innovate`/`inject`. |
| `visibility` | `all` | `all` / `injections` / `transient`. |
| `open_with_topic` | `true` | Inject a grounded opening topic before turn 0. |
| `seed` | `0` | Seeds the common-word sampler. |
| `embedding` | fake, dim 64 | Embedder for the similarity signal: `{ provider: vllm|fake, model, dim }`. |
| `num_words` | `3` | Common words sampled to build the search query. |
| `search_top_k` | `3` | Number of Tavily hits considered. |
| `max_excerpt_chars` | `null` | Cap on the injected excerpt (full text by default). |
| `search_depth` | `basic` | Tavily search depth. |
| `search_timeout` | `8.0` | Tavily request timeout, seconds. |

Every message records `level`, `signal`, `threshold`, and (for `inject`) the
sampled `words`, `query`, `source_url`, `source_title`, and `source_score` in
`meta`.

Offline smoke (fake LLM + fake embedding, no Tavily):

```bash
poetry run renewal run configs/scaffolder_smoke.yaml
```

A live run sets `embedding: { provider: vllm, model: Qwen/Qwen3-Embedding-8B }`
and requires `TAVILY_API_KEY` (only when a level-3 injection fires). Topic seed
words come from a bundled list of the 20,000 most common English words — no
vector download.

## `random_scaffolder`

A sibling of the scaffolder with the same three levels and prompts, but the
policy ignores similarity. Instead it defines a fixed probability distribution
and, **on every turn independently**, draws one action from it:

```yaml
interventions:
  - type: random_scaffolder
    options:
      probabilities:
        deepen: 0.80    # nudge to deepen
        innovate: 0.15  # nudge to innovate
        inject: 0.05    # infuse a new topic via search
      visibility: all
      open_with_topic: false
```

The three probabilities must be non-negative and sum to one. The draw uses the
run's application seed (`run.seed` / `--seed`, offset by replicate), so there is
no per-intervention seed and the sequence is reproducible. Each message records
its `level`, the `probabilities` used, and the uniform `draw` that selected the
action (plus the sampled `words`, `query`, `source_url`, `source_title`, and
`source_score` for `inject`). Options that affect injection are the same as for
the scaffolder (`num_words`, `search_top_k`, `max_excerpt_chars`,
`search_depth`, `search_timeout`); there is no `threshold` or `embedding`. The
opening topic records no `probabilities` or `draw`, because it is not produced
by a random draw.

`deepen_prompts` and `innovate_prompts` optionally replace the built-in deepening
and innovation wordings. Each accepts either a path to a YAML file of
`{id, text}` entries (the same format as `prompts/prompts.yaml`) or an inline
list of strings; omit them to keep the built-in families. For example,
`configs/run_qwen_random_scaffolder_socratic.yaml` sets `deepen_prompts:
prompts/socratic_question.yaml` to draw from a curated, non-LLM Socratic
question bank while leaving `innovate` unchanged, and
`configs/run_qwen_random_scaffolder_innovate.yaml` sets `innovate_prompts:
prompts/innovate_prompts.yaml` (a merge of several published creativity decks)
while leaving `deepen` unchanged.

Offline smoke (fake LLM, inject disabled so no Tavily call):

```bash
poetry run renewal run configs/random_scaffolder_smoke.yaml
```

A live run is `configs/run_qwen_random_scaffolder.yaml`, which sets `inject:
0.05` and uses the same Tavily injection path as the scaffolder.

## `reflective_llm_scaffolder`

A sibling of `random_scaffolder` with the same three-level skeleton and the same
independent probability draw, but the steering text is authored at runtime by a
separate **scaffolding LLM** (its own provider/model, distinct from the looping
model). Its package is a self-contained copy of `random_scaffolder` so the two
can evolve independently.

- **deepen / innovate** — the scaffold LLM receives the method's system prompt
  plus the last `history_turns` model messages and returns the steering prompt
  that is fed to the looping model.
- **summarize** — a searched source's raw text goes only to the scaffold LLM,
  which summarizes it to the looping model's `max_tokens` budget before it
  enters the message stream.
- **condense** — when a novel topic is introduced, the scaffold LLM summarizes
  everything since the last topic into a single recap that **replaces** that
  slice of canonical history (the transcript keeps the originals; the model's
  context is compressed to the recap plus the new topic).

```yaml
interventions:
  - type: reflective_llm_scaffolder
    options:
      probabilities: { deepen: 0.50, innovate: 0.40, inject: 0.10 }
      visibility: injections
      open_with_topic: true
      llm: { provider: vllm, model: Qwen/Qwen3-30B-A3B-Instruct-2507 }
      deepen:    { enabled: true, history_turns: 1 }
      innovate:  { enabled: true, history_turns: 1 }
      summarize: { enabled: true }
      condense:  { enabled: true, history_turns: null }
```

| Option | Default | Meaning |
| --- | --- | --- |
| `probabilities` | deepen 0.8 / innovate 0.15 / inject 0.05 | Same action distribution as `random_scaffolder`. |
| `visibility` | `all` | `all` / `injections` / `transient`. |
| `open_with_topic` | `true` | Infuse a summarized opening topic before turn 0. |
| `llm` | fake | The shared scaffolding LLM: `{ provider: vllm|azure|fake, model }`. |
| `deepen` / `innovate` | enabled, `history_turns: 1` | `{ enabled, system_prompt, history_turns }`. |
| `summarize` | enabled | `{ enabled, system_prompt }`; the prompt may use `{max_tokens}`. |
| `condense` | enabled, `history_turns: null` | `{ enabled, system_prompt, history_turns }`; `null` = all since the last topic. |
| `generation` | temp 0.9, max 200 | Scaffold-LLM decode params (summarize overrides `max_tokens`). |
| `max_excerpt_chars` | `800` | Cap on the raw excerpt when summarization is off or fails. |
| `num_words`, `search_top_k`, `search_depth`, `search_timeout` | as `random_scaffolder` | Tavily injection knobs. |

Each of the four methods can be disabled with `enabled: false` (a disabled
deepen/innovate draw is a no-op; a disabled `summarize` uses the capped raw
text; a disabled `condense` appends the topic without compressing history). The
deepen/innovate default prompts are the built-in reflective wording; summarize
and condense have sensible default prompts that the config can override. The
scaffolder sizes its summaries to the looping model because the harness injects
the looping `max_tokens` automatically (via the intervention's
`uses_loop_max_tokens` hook), just as it injects `seed` (via `uses_run_seed`).

Offline smoke (fake LLM on both sides, inject disabled so no Tavily call):

```bash
poetry run renewal run configs/reflective_llm_scaffolder_smoke.yaml
```

A live run is `configs/run_qwen_reflective_llm_scaffolder.yaml`, which needs
`TAVILY_API_KEY` (only when inject fires).

## `memory_scaffolder`

A sibling of `reflective_llm_scaffolder` (its package is a self-contained copy)
that adds a fourth action, **`resurface`**, and remembers the topics that end.
It is meant for runs where the loop forgets (`run.memory_turns`) and earlier
context is otherwise lost.

Each turn the policy independently draws one of **four** actions:

| Level | Behavior |
| --- | --- |
| `deepen` | Scaffolding LLM authors a deepening steer (as reflective). |
| `innovate` | Scaffolding LLM authors an innovation steer (as reflective). |
| `inject` | Summarize the searched source, introduce a new topic, and store a memory first. |
| `resurface` | Bring back one stored memory and ask the model to connect it to the current topic. |

**Memories.** Just before a new topic is introduced, the scaffolding LLM
summarizes the topic that is ending — everything since the last topic, at the
looping model's `max_tokens` — and stores the summary as a `Memory`. Unlike the
reflective scaffolder's `condense`, remembering never replaces history; it only
saves a copy for later. A `Memory` keeps a sequential `id`, its `text`, a
unit-norm `embedding`, `surfaced_count`, `last_surfaced_turn`, the turn range it
was distilled from, and provenance. The resurface action selects a memory, renders
one of the `RESURFACE_PROMPTS` (five fixed templates that insert the memory and
ask how it is relevant), marks it surfaced, and emits it as a steering message.
With the first `~` inject-only start there are no memories, so an early resurface
draw is a no-op.

The first selection algorithm, `random_unsurfaced`, surfaces a memory that has
never been surfaced, uniformly at random; once every memory has been seen at
least once it falls back to the least-surfaced ones. The store is built around a
`MemorySelector` protocol so future algorithms (e.g. embedding similarity to the
current turn) can be dropped in.

```yaml
interventions:
  - type: memory_scaffolder
    options:
      probabilities: { deepen: 0.50, innovate: 0.30, inject: 0.10, resurface: 0.10 }
      visibility: injections
      open_with_topic: true
      llm: { provider: vllm, model: Qwen/Qwen3-30B-A3B-Instruct-2507 }
      embedding: { provider: vllm, model: Qwen/Qwen3-Embedding-8B }  # memory vectors
      summarize: { enabled: true }
      remember:  { enabled: true }
```

| Option | Default | Meaning |
| --- | --- | --- |
| `probabilities` | deepen 0.80 / innovate 0.05 / inject 0.05 / resurface 0.10 | Four-way action distribution; must sum to 1. |
| `visibility` | `all` | `all` / `injections` / `transient`. |
| `open_with_topic` | `true` | Infuse a summarized opening topic before turn 0 (no memory is created). |
| `llm` | fake | The shared scaffolding LLM: `{ provider: vllm|azure|fake, model }`. |
| `embedding` | fake, dim 64 | Embedder for stored memories: `{ provider: vllm|fake, model, dim }`. |
| `deepen` / `innovate` | enabled, `history_turns: 1` | `{ enabled, system_prompt, history_turns }`. |
| `summarize` | enabled | `{ enabled, system_prompt }`; the prompt may use `{max_tokens}`. |
| `remember` | enabled, `history_turns: null` | `{ enabled, system_prompt, history_turns }`; the memory summary prompt. |
| `resurface_prompts` | 5 built-ins | Path to a YAML `{id, text}` file or an inline list; each template uses `{memory}`. |
| `generation` | temp 0.9, max 200 | Scaffold-LLM decode params (summarize/remember override `max_tokens`). |
| `max_excerpt_chars` | `800` | Cap on the raw excerpt when summarization is off or fails. |
| `num_words`, `search_top_k`, `search_depth`, `search_timeout` | as `reflective_llm_scaffolder` | Tavily injection knobs. |

The run harness injects the seed, the looping `max_tokens`, and the recorder
(via `uses_run_seed` / `uses_loop_max_tokens` / `uses_memory_sink`). Every
`memory_created` / `memory_surfaced` event is appended to `memories.jsonl` in the
run directory (see [Run artifacts](08_artifacts.md)) and can be browsed in the
[viewer](03_viewer.md) **Memories** tab.

Offline smoke (fake LLM + fake embedder, inject disabled so no Tavily call):

```bash
poetry run renewal run configs/memory_scaffolder_smoke.yaml
```

Live runs: `configs/run_qwen_memory_scaffolder.yaml` and
`configs/run_4o-mini_memory_scaffolder.yaml`, which need `TAVILY_API_KEY` (only
when inject fires).
