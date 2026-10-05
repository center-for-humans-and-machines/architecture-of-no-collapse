"""uvicorn entry point for the viewer backend.

Run from the repo root:

    uvicorn viewer.backend.main:app --reload --port 8000

The run tree is ``outputs/`` by default; override with ``RENEWAL_OUTPUTS``.
"""

from __future__ import annotations

import os
from pathlib import Path

from viewer.backend.app import create_app

base_dir = Path(os.environ.get("RENEWAL_OUTPUTS", "outputs"))
app = create_app(base_dir)
