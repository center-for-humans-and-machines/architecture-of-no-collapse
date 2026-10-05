# Renewal Viewer

A standalone web app for browsing renewal run artifacts: filter runs, read a
conversation end-to-end (with interventions), inspect the metrics table, and
overlay several runs' metric timelines live.

## Layout

```
viewer/
  backend/     # FastAPI JSON API (its own process)
  frontend/    # Svelte + Vite + Plotly.js (its own process)
```

## Run it (development)

Terminal 1 — the API (Python; needs the Poetry env):

```bash
cd <repo root>
poetry run uvicorn viewer.backend.main:app --reload --port 8000
```

Terminal 2 — the frontend (needs Node.js ≥ 20):

```bash
cd viewer/frontend
npm install
npm run dev          # http://localhost:5173, proxies /api -> :8000
```

Open http://localhost:5173.

The backend scans `outputs/` by default; point it elsewhere with the
`RENEWAL_OUTPUTS` env var:

```bash
RENEWAL_OUTPUTS=/path/to/outputs poetry run uvicorn viewer.backend.main:app --port 8000
```

## Pages

1. **Explorer** — filter by models involved, rounds, temperature, max tokens,
   window size, and start time; the run table updates live (status flips from
   `running` to `completed`). Check boxes to select runs for comparison; click a
   run id to open it.
2. **Conversation** — the full transcript in order, one message per block, with
   `intervention` blocks inset in orange.
3. **Metrics table** — one row per window, one column per metric.
4. **Timeline** — the selected runs overlaid, one chart per metric (x = window
   index); extends live as a running run writes more windows.

## Production

Build the frontend and have the backend serve it:

```bash
cd viewer/frontend && npm run build
```

then serve `viewer/frontend/dist` with any static file server (or add a
`StaticFiles` mount in `viewer/backend/app.py`), keeping `/api` pointed at the
FastAPI app.
