<script>
  import { api } from './lib/api.js';
  import Explorer from './components/Explorer.svelte';
  import RunDetail from './components/RunDetail.svelte';
  import Timeline from './components/Timeline.svelte';

  let runs = $state([]);
  let facets = $state(null);
  let filters = $state({});
  let selected = $state([]); // run ids chosen for timeline comparison
  let activeRunId = $state(null); // null -> main view
  let mainTab = $state('runs'); // 'runs' | 'timeline'

  // Load facets once.
  $effect(() => {
    api.facets().then((f) => (facets = f)).catch(console.error);
  });

  // Poll the run list (re-fetches immediately when filters change).
  $effect(() => {
    const load = () => api.runs(filters).then((r) => (runs = r)).catch(console.error);
    load();
    const id = setInterval(load, 1500);
    return () => clearInterval(id);
  });

  function setFilters(next) {
    filters = next;
  }

  function openRun(id) {
    activeRunId = id;
  }

  function closeDetail() {
    activeRunId = null;
  }

  function toggleSelect(id) {
    selected = selected.includes(id)
      ? selected.filter((x) => x !== id)
      : [...selected, id];
  }

  function toggleAll(ids) {
    const all = ids.length > 0 && ids.every((id) => selected.includes(id));
    selected = all
      ? selected.filter((id) => !ids.includes(id))
      : [...new Set([...selected, ...ids])];
  }
</script>

<main>
  <header>
    <h1>Renewal Viewer</h1>
    {#if activeRunId}
      <button onclick={closeDetail}>← back to runs</button>
    {:else}
      <nav class="tabs">
        <button class:active={mainTab === 'runs'} onclick={() => (mainTab = 'runs')}>Runs</button>
        <button class:active={mainTab === 'timeline'} onclick={() => (mainTab = 'timeline')}>
          Timeline
        </button>
      </nav>
      <span class="muted">{runs.length} run(s)</span>
    {/if}
  </header>

  {#if activeRunId}
    <RunDetail runId={activeRunId} />
  {:else if mainTab === 'timeline'}
    <Timeline runIds={selected} />
  {:else}
    <Explorer
      {runs}
      {facets}
      {filters}
      {selected}
      onfilters={setFilters}
      ontoggle={toggleSelect}
      ontoggleall={toggleAll}
      onopen={openRun}
    />
  {/if}
</main>

<style>
  main {
    padding: 16px 24px;
    max-width: 1200px;
    margin: 0 auto;
  }

  header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 16px;
  }

  h1 {
    font-size: 20px;
    margin: 0;
  }

  .muted {
    color: var(--muted);
  }

  .tabs {
    display: flex;
    gap: 6px;
  }

  .tabs button {
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    background: transparent;
    padding: 2px 14px;
  }

  .tabs button.active {
    border-bottom-color: var(--accent);
    color: var(--accent);
  }
</style>
