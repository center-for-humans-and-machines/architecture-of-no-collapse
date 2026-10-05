<script>
  import { fmt } from '../lib/api.js';

  let { metrics = [] } = $props();

  let columns = $derived(
    [...new Set(metrics.filter((m) => m.cadence === 'per_window').map((m) => m.metric))]
  );

  let rows = $derived.by(() => {
    const byIndex = new Map();
    for (const m of metrics) {
      if (m.cadence !== 'per_window') continue;
      if (!byIndex.has(m.index)) {
        byIndex.set(m.index, {
          index: m.index,
          start: m.window_start_round,
          end: m.window_end_round,
        });
      }
      byIndex.get(m.index)[m.metric] = m.value;
    }
    return [...byIndex.values()].sort((a, b) => a.index - b.index);
  });
</script>

<table>
  <thead>
    <tr>
      <th>window</th>
      <th>rounds</th>
      {#each columns as c}<th>{c}</th>{/each}
    </tr>
  </thead>
  <tbody>
    {#each rows as row (row.index)}
      <tr>
        <td>{row.index}</td>
        <td>{row.start}–{row.end}</td>
        {#each columns as c}<td>{fmt(row[c])}</td>{/each}
      </tr>
    {/each}
  </tbody>
</table>
