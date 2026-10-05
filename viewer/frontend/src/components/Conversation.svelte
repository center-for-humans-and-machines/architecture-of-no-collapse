<script>
  let { messages = [] } = $props();
</script>

<div class="conversation">
  {#each messages as m, i (i)}
    <div class="msg {m.role}">
      <div class="meta">
        <strong>{m.speaker}</strong>
        <span class="badge">{m.role}</span>
        {#if m.meta?.round != null}<span class="muted">round {m.meta.round}</span>{/if}
        {#if m.turn_index != null}<span class="muted">turn {m.turn_index}</span>{/if}
        {#if m.meta?.model}<span class="muted">{m.meta.model}</span>{/if}
      </div>
      <pre class="content">{m.content}</pre>
    </div>
  {/each}
</div>

<style>
  .conversation {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .msg {
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 12px;
    background: var(--panel);
  }

  .msg.intervention {
    border-left: 3px solid var(--intervention);
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

  .muted {
    color: var(--muted);
    font-size: 12px;
  }

  .content {
    margin: 0;
    white-space: pre-wrap;
    font-family: inherit;
  }
</style>
