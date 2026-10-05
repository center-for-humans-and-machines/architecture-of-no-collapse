<script>
  import Plotly from 'plotly.js-dist-min';

  let { metric, traces = [] } = $props();
  let el = $state(null);

  $effect(() => {
    if (!el) return;
    Plotly.react(
      el,
      traces.map((t) => ({
        x: t.x,
        y: t.y,
        type: 'scatter',
        mode: 'lines+markers',
        name: t.name,
      })),
      {
        title: metric,
        yaxis: { range: [0, 1] },
        margin: { t: 36, b: 32, l: 44, r: 16 },
        height: 240,
        paper_bgcolor: '#0f141a',
        plot_bgcolor: '#0f141a',
        font: { color: '#dbe4ee' },
        showlegend: true,
      },
      { displayModeBar: false }
    );
  });
</script>

<div bind:this={el} class="chart"></div>

<style>
  .chart {
    width: 100%;
    height: 240px;
  }
</style>
