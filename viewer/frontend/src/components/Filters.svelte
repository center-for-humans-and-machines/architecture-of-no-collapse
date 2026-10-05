<script>
  let { facets = null, filters = {}, onchange } = $props();

  let model = $state([]);
  let roundsMin = $state('');
  let roundsMax = $state('');
  let tempMin = $state('');
  let tempMax = $state('');
  let tokensMin = $state('');
  let tokensMax = $state('');
  let windowSize = $state([]);
  let since = $state('');
  let until = $state('');

  function emit() {
    onchange({
      model: model.length ? model : undefined,
      rounds_min: roundsMin === '' ? undefined : Number(roundsMin),
      rounds_max: roundsMax === '' ? undefined : Number(roundsMax),
      temp_min: tempMin === '' ? undefined : Number(tempMin),
      temp_max: tempMax === '' ? undefined : Number(tempMax),
      max_tokens_min: tokensMin === '' ? undefined : Number(tokensMin),
      max_tokens_max: tokensMax === '' ? undefined : Number(tokensMax),
      window_size: windowSize.length ? windowSize.map(Number) : undefined,
      since: since || undefined,
      until: until || undefined,
    });
  }

  function toggle(arr, value) {
    return arr.includes(value) ? arr.filter((v) => v !== value) : [...arr, value];
  }

  function reset() {
    model = [];
    roundsMin = '';
    roundsMax = '';
    tempMin = '';
    tempMax = '';
    tokensMin = '';
    tokensMax = '';
    windowSize = [];
    since = '';
    until = '';
    emit();
  }
</script>

<div class="filters">
  <div class="group">
    <label>Models</label>
    {#if facets && facets.models.length}
      {#each facets.models as m}
        <label class="check">
          <input
            type="checkbox"
            checked={model.includes(m)}
            onchange={() => { model = toggle(model, m); emit(); }}
          />
          <span>{m}</span>
        </label>
      {/each}
    {:else}
      <span class="muted">no models</span>
    {/if}
  </div>

  <div class="group">
    <label>Rounds</label>
    <div class="range">
      <input type="number" placeholder="min" bind:value={roundsMin} oninput={emit} />
      <input type="number" placeholder="max" bind:value={roundsMax} oninput={emit} />
    </div>
  </div>

  <div class="group">
    <label>Temperature</label>
    <div class="range">
      <input type="number" step="0.1" placeholder="min" bind:value={tempMin} oninput={emit} />
      <input type="number" step="0.1" placeholder="max" bind:value={tempMax} oninput={emit} />
    </div>
  </div>

  <div class="group">
    <label>Max tokens</label>
    <div class="range">
      <input type="number" placeholder="min" bind:value={tokensMin} oninput={emit} />
      <input type="number" placeholder="max" bind:value={tokensMax} oninput={emit} />
    </div>
  </div>

  <div class="group">
    <label>Window size</label>
    {#if facets && facets.window_size.length}
      {#each facets.window_size as w}
        <label class="check">
          <input
            type="checkbox"
            checked={windowSize.includes(String(w))}
            onchange={() => { windowSize = toggle(windowSize, String(w)); emit(); }}
          />
          <span>{w}</span>
        </label>
      {/each}
    {/if}
  </div>

  <div class="group">
    <label>Started after</label>
    <input type="datetime-local" bind:value={since} oninput={emit} />
    <label>Started before</label>
    <input type="datetime-local" bind:value={until} oninput={emit} />
  </div>

  <button onclick={reset}>Reset</button>
</div>

<style>
  .filters {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 12px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 6px;
  }

  .group {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .group > label {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  .range {
    display: flex;
    gap: 6px;
  }

  .range input {
    width: 100%;
  }

  .check {
    display: flex;
    align-items: center;
    gap: 6px;
    cursor: pointer;
  }

  .check input {
    margin: 0;
  }

  .muted {
    color: var(--muted);
  }
</style>
