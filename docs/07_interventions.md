# Interventions

An intervention is a plug-in that appends steering messages to the conversation.
The loop applies the configured interventions in order after every agent turn,
and each intervention may also inject an opening message before turn 0. Two
interventions ship with the harness; an empty `interventions` list runs a bare
conversation.

Both interventions share the same three levels and steering prompts:

| Level | Prompt family |
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
— the source title and URL are kept in `meta` but never shown to the model. The
excerpt is Tavily's full parsed page text (`raw_content`), preferred over its
short snippet; `max_excerpt_chars` caps it (default `null` uses the full text),
trimmed back to the last sentence boundary.

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

Offline smoke (fake LLM, inject disabled so no Tavily call):

```bash
poetry run renewal run configs/random_scaffolder_smoke.yaml
```

A live run is `configs/run_qwen_random_scaffolder.yaml`, which sets `inject:
0.05` and uses the same Tavily injection path as the scaffolder.
