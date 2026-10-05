function toQuery(params) {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params || {})) {
    if (value === null || value === undefined || value === '') continue;
    if (Array.isArray(value)) {
      for (const v of value) qs.append(key, v);
    } else {
      qs.set(key, value);
    }
  }
  const s = qs.toString();
  return s ? '?' + s : '';
}

async function getJSON(path, params) {
  const res = await fetch(path + toQuery(params));
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${path}`);
  return res.json();
}

export const api = {
  facets: () => getJSON('/api/facets'),
  runs: (filters) => getJSON('/api/runs', filters),
  transcript: (id) => getJSON(`/api/runs/${id}/transcript`),
  metrics: (id) => getJSON(`/api/runs/${id}/metrics`),
  meta: (id) => getJSON(`/api/runs/${id}/meta`),
};

export function fmt(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toFixed(3);
  return String(v);
}

export function shortId(id) {
  return id ? id.slice(0, 8) : '';
}
