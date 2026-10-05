<script>
  import Filters from './Filters.svelte';
  import RunTable from './RunTable.svelte';

  let { runs, facets, filters, selected, onfilters, ontoggle, onopen } = $props();
</script>

<div class="layout">
  <aside>
    <Filters {facets} {filters} onchange={onfilters} />
  </aside>
  <section>
    <RunTable {runs} {selected} onopen={onopen} ontoggle={ontoggle} />
  </section>
</div>

<style>
  .layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 16px;
    align-items: start;
  }

  @media (max-width: 800px) {
    .layout {
      grid-template-columns: 1fr;
    }
  }
</style>
