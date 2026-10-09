# Viewer

A standalone web app (FastAPI + Svelte + Plotly.js) for browsing renewal run
artifacts: filter runs, read a conversation end-to-end (with interventions),
inspect the metrics table, and overlay several runs' metric timelines live.

```
viewer/
  backend/     # FastAPI JSON API (its own process)
  frontend/    # Svelte + Vite + Plotly.js (its own process)
```

## Run it (development)

The frontend needs Node.js ≥ 20. The backend reads run artifacts directly
rather than importing the `renewal` package, so it only needs the Poetry env
for FastAPI/uvicorn.

Terminal 1 — the API:

```bash
cd <repo root>
poetry run uvicorn viewer.backend.main:app --reload --port 8000
```

Terminal 2 — the frontend:

```bash
cd viewer/frontend
npm install
npm run dev          # http://localhost:5173, proxies /api -> :8000
```

Open <http://localhost:5173>.

The backend scans `outputs/` by default; point it elsewhere with the
`RENEWAL_OUTPUTS` environment variable:

```bash
RENEWAL_OUTPUTS=/path/to/outputs poetry run uvicorn viewer.backend.main:app --port 8000
```

## Pages

The main view has two tabs: **Runs** and **Timeline**.

1. **Runs** — filter by models involved, rounds, temperature, max tokens,
   window size, and start time; the run table updates live (status flips from
   `running` to `completed`). Check boxes to select runs for comparison, use the
   header checkbox to select or clear all visible runs at once, or click a run
   id to open it.
2. **Timeline** — every selected run overlaid, one chart per metric (x = window
   index); extends live as a running run writes more windows.

Opening a run shows the single-run view, which has its own tabs:

1. **Conversation** — the full transcript in order, one message per block, with
   `intervention` blocks inset in orange.
2. **Metrics table** — one row per window, one column per metric.

## Production

Build the frontend and have the backend serve it:

```bash
cd viewer/frontend && npm run build
```

then serve `viewer/frontend/dist` with any static file server (or add a
`StaticFiles` mount in `viewer/backend/app.py`), keeping `/api` pointed at the
FastAPI app.
