<script>
  import { api } from '../lib/api.js';
  import MetricChart from './MetricChart.svelte';

  let { runIds = [] } = $props();

  let metrics = $state([]); // [{ metric, traces: [{ name, x, y }] }]

  $effect(() => {
    let cancelled = false;

    async function load() {
      const all = await Promise.all(
        runIds.map(async (id) => ({ id, rows: await api.metrics(id) }))
      );
      const perWindow = new Map();
      const experiments = new Map(); // run id -> experiment name
      for (const { id, rows } of all) {
        const experiment = rows.find((row) => row.experiment)?.experiment;
        if (experiment) experiments.set(id, experiment);
        for (const row of rows) {
          if (row.cadence !== 'per_window') continue;
          if (!perWindow.has(row.metric)) perWindow.set(row.metric, []);
          perWindow.get(row.metric).push({ id, index: row.index, value: row.value });
        }
      }
      const next = [];
      for (const [metric, points] of perWindow) {
        const traces = runIds.map((id) => {
          const pts = points.filter((p) => p.id === id).sort((a, b) => a.index - b.index);
          const experiment = experiments.get(id);
          return {
            name: experiment ? `${experiment} / ${id.slice(0, 8)}` : id.slice(0, 8),
            x: pts.map((p) => p.index),
            y: pts.map((p) => p.value),
          };
        });
        next.push({ metric, traces });
      }
      next.sort((a, b) => a.metric.localeCompare(b.metric));
      if (!cancelled) metrics = next;
    }

    load();
    const id = setInterval(load, 1500);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  });
</script>

<div class="timeline">
  {#if runIds.length === 0}
    <p class="muted">Select runs in the Runs tab to compare their timelines.</p>
  {:else if metrics.length === 0}
    <p class="muted">No metrics for the selected run(s).</p>
  {:else}
    {#each metrics as m (m.metric)}
      <MetricChart metric={m.metric} traces={m.traces} />
    {/each}
  {/if}
</div>

<style>
  .timeline {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .muted {
    color: var(--muted);
  }
</style>
