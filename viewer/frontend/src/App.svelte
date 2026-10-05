<script>
  import { api } from './lib/api.js';
  import Explorer from './components/Explorer.svelte';
  import RunDetail from './components/RunDetail.svelte';

  let runs = $state([]);
  let facets = $state(null);
  let filters = $state({});
  let selected = $state([]); // run ids chosen for timeline comparison
  let activeRunId = $state(null); // null -> explorer view

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
</script>

<main>
  <header>
    <h1>Renewal Viewer</h1>
    {#if activeRunId}
      <button onclick={closeDetail}>← back to runs</button>
    {:else}
      <span class="muted">{runs.length} run(s)</span>
    {/if}
  </header>

  {#if activeRunId}
    <RunDetail runId={activeRunId} comparisonIds={selected.length ? selected : [activeRunId]} />
  {:else}
    <Explorer
      {runs}
      {facets}
      {filters}
      {selected}
      onfilters={setFilters}
      ontoggle={toggleSelect}
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
</style>
