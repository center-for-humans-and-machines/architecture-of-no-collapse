<script>
  import { api } from '../lib/api.js';
  import Conversation from './Conversation.svelte';
  import Memories from './Memories.svelte';
  import MetricsTable from './MetricsTable.svelte';

  let { runId } = $props();

  let tab = $state('conversation');
  let transcript = $state([]);
  let metrics = $state([]);
  let memories = $state([]);
  let meta = $state(null);

  $effect(() => {
    api.transcript(runId).then((t) => (transcript = t)).catch(console.error);
    api.meta(runId).then((m) => (meta = m)).catch(console.error);
  });

  // Poll metrics and memories while the detail is open so running runs extend live.
  $effect(() => {
    const load = () => api.metrics(runId).then((m) => (metrics = m)).catch(console.error);
    load();
    const id = setInterval(load, 1500);
    return () => clearInterval(id);
  });

  $effect(() => {
    const load = () => api.memories(runId).then((m) => (memories = m)).catch(console.error);
    load();
    const id = setInterval(load, 1500);
    return () => clearInterval(id);
  });

  let tabs = $derived(
    memories.length > 0
      ? ['conversation', 'metrics', 'memories']
      : ['conversation', 'metrics']
  );
</script>

<div class="detail">
  <div class="run-header">
    <h2>{runId}</h2>
    {#if meta}
      <span class="muted">{meta.experiment_name} · seed {meta.seed} · {meta.status}</span>
    {/if}
  </div>

  <nav class="tabs">
    {#each tabs as t}
      <button class:active={tab === t} onclick={() => (tab = t)}>{t}</button>
    {/each}
  </nav>

  {#if tab === 'conversation'}
    <Conversation messages={transcript} />
  {:else if tab === 'metrics'}
    <MetricsTable {metrics} />
  {:else if tab === 'memories'}
    <Memories {memories} />
  {/if}
</div>

<style>
  .run-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
  }

  .tabs {
    display: flex;
    gap: 6px;
    margin: 12px 0;
    border-bottom: 1px solid var(--border);
  }

  .tabs button {
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    background: transparent;
    padding: 6px 14px;
  }

  .tabs button.active {
    border-bottom-color: var(--accent);
    color: var(--accent);
  }

  .muted {
    color: var(--muted);
  }
</style>
