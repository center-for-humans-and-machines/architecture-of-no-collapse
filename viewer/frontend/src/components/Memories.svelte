<script>
  let { memories = [] } = $props();
</script>

<div class="memories">
  <p class="muted">
    {memories.length} memory(ies) — each summarizes a topic that ended and can be
    resurfaced later.
  </p>

  {#each memories as memory (memory.id)}
    <div class="memory" class:unsurfaced={memory.surfaced_count === 0}>
      <div class="meta">
        <strong>#{memory.id}</strong>
        {#if memory.created_turn != null}
          <span class="muted">created turn {memory.created_turn}</span>
        {/if}
        {#if memory.since_turn != null}
          <span class="muted">since turn {memory.since_turn}</span>
        {/if}
        <span class="badge" class:seen={memory.surfaced_count > 0}>
          surfaced {memory.surfaced_count}×
        </span>
        {#if memory.last_surfaced_turn != null}
          <span class="muted">last turn {memory.last_surfaced_turn}</span>
        {/if}
        {#if memory.embedding_dim}
          <span class="muted">emb {memory.embedding_dim}d</span>
        {/if}
        {#if memory.selector}
          <span class="muted">{memory.selector}</span>
        {/if}
      </div>

      <pre class="content">{memory.text}</pre>

      {#if memory.provenance && Object.keys(memory.provenance).length}
        <div class="provenance">
          {#if memory.provenance.words}
            <span>words: {memory.provenance.words.join(', ')}</span>
          {/if}
          {#if memory.provenance.source_title}
            <span>source: {memory.provenance.source_title}</span>
          {/if}
          {#if memory.provenance.source_url}
            <a href={memory.provenance.source_url} target="_blank" rel="noreferrer">
              {memory.provenance.source_url}
            </a>
          {/if}
        </div>
      {/if}
    </div>
  {/each}

  {#if memories.length === 0}
    <p class="muted">
      No memories yet. They are created when a new topic is introduced.
    </p>
  {/if}
</div>

<style>
  .memories {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .memory {
    border: 1px solid var(--border);
    border-left: 3px solid var(--intervention);
    border-radius: 6px;
    padding: 8px 12px;
    background: var(--panel);
  }

  .memory.unsurfaced {
    border-left-color: var(--accent);
  }

  .meta {
    display: flex;
    gap: 10px;
    align-items: baseline;
    margin-bottom: 4px;
    flex-wrap: wrap;
  }

  .badge {
    color: var(--muted);
    font-size: 12px;
  }

  .badge.seen {
    color: var(--accent);
  }

  .muted {
    color: var(--muted);
    font-size: 12px;
  }

  .content {
    margin: 0;
    white-space: pre-wrap;
    font-family: inherit;
  }

  .provenance {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 6px;
    font-size: 12px;
    color: var(--muted);
  }
</style>
