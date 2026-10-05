<script>
  import { shortId } from '../lib/api.js';

  let { runs = [], selected = [], onopen, ontoggle } = $props();

  function models(run) {
    return (run.models || []).map((m) => m.model).join(', ');
  }

  function when(run) {
    return run.started_at ? run.started_at.slice(0, 19).replace('T', ' ') : '—';
  }
</script>

<table>
  <thead>
    <tr>
      <th></th>
      <th>run</th>
      <th>experiment</th>
      <th>seed</th>
      <th>status</th>
      <th>started</th>
      <th>rounds</th>
      <th>models</th>
      <th>metrics</th>
    </tr>
  </thead>
  <tbody>
    {#each runs as run (run.run_id)}
      <tr>
        <td>
          <input
            type="checkbox"
            checked={selected.includes(run.run_id)}
            onchange={() => ontoggle(run.run_id)}
          />
        </td>
        <td>
          <a href="#" onclick={(e) => { e.preventDefault(); onopen(run.run_id); }}>
            {shortId(run.run_id)}
          </a>
        </td>
        <td>{run.experiment}</td>
        <td>{run.seed}</td>
        <td class:running={run.status === 'running'}>{run.status}</td>
        <td>{when(run)}</td>
        <td>{run.rounds}</td>
        <td>{models(run)}</td>
        <td>{run.has_metrics ? (run.metrics_status || 'yes') : '—'}</td>
      </tr>
    {/each}
  </tbody>
</table>

<style>
  td {
    white-space: nowrap;
  }

  .running {
    color: var(--accent);
  }
</style>
