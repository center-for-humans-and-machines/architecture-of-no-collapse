"""Cross-run aggregation: compute cross_run_sim over sibling run dirs."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from renewal.metrics.base import cosine

LOGGER = logging.getLogger(__name__)


def _find_run_dirs(experiment_dir: Path) -> list[Path]:
    run_dirs: list[Path] = []
    for child in sorted(experiment_dir.iterdir()):
        if child.is_dir() and (child / "meta.json").exists():
            run_dirs.append(child)
    return run_dirs


def _load_run(run_dir: Path) -> dict[str, Any] | None:
    """Load one run's metadata and window embeddings, or None when unavailable."""
    emb_path = run_dir / "embeddings.parquet"
    if not emb_path.exists():
        return None
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    df = pd.read_parquet(emb_path)
    embeddings: dict[int, np.ndarray] = {}
    for row in df.itertuples(index=False):
        embeddings[int(row.window_index)] = np.asarray(row.embedding, dtype=np.float64)
    return {
        "run_id": meta.get("run_id", run_dir.name),
        "seed": meta.get("seed"),
        "experiment": meta.get("experiment_name"),
        "condition": (meta.get("labels") or {}).get("condition"),
        "embeddings": embeddings,
    }


def _compute_group(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the group cross_run_sim plus per-run contributions."""
    if len(runs) < 2:
        return {"cross_run_sim": float("nan"), "per_run": {}, "n_runs": len(runs), "windows": 0}
    w = min(len(run["embeddings"]) for run in runs)

    per_run: dict[str, float] = {}
    for run in runs:
        others = [r for r in runs if r["run_id"] != run["run_id"]]
        sims = [
            cosine(run["embeddings"][i], other["embeddings"][i])
            for other in others
            for i in range(w)
        ]
        per_run[run["run_id"]] = float(np.mean(sims)) if sims else float("nan")

    pairwise = [
        cosine(runs[a]["embeddings"][i], runs[b]["embeddings"][i])
        for a in range(len(runs))
        for b in range(a + 1, len(runs))
        for i in range(w)
    ]
    group_sim = float(np.mean(pairwise)) if pairwise else float("nan")
    return {"cross_run_sim": group_sim, "per_run": per_run, "n_runs": len(runs), "windows": w}


def analyze_experiment(experiment_dir: Path, condition: str | None = None) -> list[dict[str, Any]]:
    """Compute cross_run_sim over run dirs; return long-format rows."""
    experiment_dir = Path(experiment_dir)
    runs: list[dict[str, Any]] = []
    for run_dir in _find_run_dirs(experiment_dir):
        run = _load_run(run_dir)
        if run is None:
            LOGGER.warning("skipping %s: missing embeddings.parquet", run_dir)
            continue
        runs.append(run)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if condition is not None and run["condition"] != condition:
            continue
        groups[run["condition"] or "all"].append(run)

    rows: list[dict[str, Any]] = []
    for key, group in groups.items():
        result = _compute_group(group)
        for run in group:
            rows.append(
                {
                    "run_id": run["run_id"],
                    "metric": "cross_run_sim",
                    "cadence": "per_run",
                    "index": 0,
                    "value": result["per_run"].get(run["run_id"], float("nan")),
                    "window_start_round": None,
                    "window_end_round": None,
                    "seed": run["seed"],
                    "experiment": run["experiment"],
                    "condition": run["condition"],
                }
            )
        rows.append(
            {
                "run_id": f"group:{key}",
                "metric": "cross_run_sim",
                "cadence": "per_run",
                "index": 0,
                "value": result["cross_run_sim"],
                "window_start_round": None,
                "window_end_round": None,
                "seed": None,
                "experiment": experiment_dir.name,
                "condition": key if key != "all" else None,
            }
        )
    return rows


def run_analyze(experiment_dir: Path, condition: str | None = None) -> Path:
    """Compute cross_run_sim and write the experiment-level metrics.parquet."""
    rows = analyze_experiment(experiment_dir, condition)
    out_path = Path(experiment_dir) / "metrics.parquet"
    if rows:
        df = pd.DataFrame(rows)
        df["value"] = df["value"].astype("float64")
        df["index"] = df["index"].astype("int64")
        df["window_start_round"] = df["window_start_round"].astype("Int64")
        df["window_end_round"] = df["window_end_round"].astype("Int64")
        df.to_parquet(out_path, index=False)
    LOGGER.info("analyze wrote %s rows to %s", len(rows), out_path)
    return out_path
