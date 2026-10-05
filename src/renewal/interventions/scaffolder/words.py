"""Random-word sampling from a GloVe word-vector space.

Draws random unit vectors in the embedding space and returns the nearest
vocabulary words. This deliberately surfaces arbitrary, loosely related terms
that seed the level-3 topic search.

Vectors are loaded lazily from a local ``model_path`` or downloaded (and
cached) from the gensim-data release. Parsing uses numpy directly, so there is
no gensim dependency. The heavy load happens once; ``sample`` stays cheap
afterward.
"""

from __future__ import annotations

import gzip
import logging
import random
from pathlib import Path
from typing import Protocol

import numpy as np

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "glove-wiki-gigaword-100"
_DOWNLOAD_BASE = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download"
)


class WordSampler(Protocol):
    """Return random vocabulary words."""

    def sample(self, count: int) -> list[str]:
        ...


class GloVeWordSampler:
    """Draw words nearest to random unit vectors in a GloVe space."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        seed: int = 0,
        model_path: str | None = None,
        cache_dir: str | None = None,
        download_timeout: float = 120.0,
    ) -> None:
        self.model_name = model_name
        self._rng = random.Random(seed)
        self._model_path = Path(model_path).expanduser() if model_path else None
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._download_timeout = download_timeout
        self._words: list[str] | None = None
        self._unit: np.ndarray | None = None

    def sample(self, count: int) -> list[str]:
        """Return ``count`` vocabulary words nearest to random unit vectors."""
        if count < 1:
            raise ValueError("count must be >= 1")
        self._ensure_loaded()
        assert self._unit is not None and self._words is not None
        dim = int(self._unit.shape[1])
        words: list[str] = []
        for _ in range(count):
            vector = self._random_unit_vector(dim)
            index = int((self._unit @ vector).argmax())
            words.append(self._words[index])
        return words

    def _ensure_loaded(self) -> None:
        if self._unit is not None:
            return
        path = self._resolve_path()
        words, matrix = _load_vectors(path)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._unit = matrix / np.clip(norms, 1e-12, None)
        self._words = words

    def _resolve_path(self) -> Path:
        if self._model_path is not None:
            if not self._model_path.is_file():
                raise FileNotFoundError(
                    f"word vector file not found: {self._model_path}"
                )
            return self._model_path
        cache_dir = self._cache_dir or Path.home() / ".cache" / "renewal"
        return download_vectors(
            self.model_name,
            cache_dir=cache_dir,
            timeout=self._download_timeout,
        )

    def _random_unit_vector(self, dim: int) -> np.ndarray:
        vector = np.fromiter(
            (self._rng.gauss(0.0, 1.0) for _ in range(dim)),
            dtype=np.float32,
            count=dim,
        )
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:  # pragma: no cover - vanishingly unlikely
            vector = np.ones(dim, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
        return vector / norm


def _parse_line(line: str) -> tuple[str, np.ndarray]:
    parts = line.split()
    word = parts[0]
    values = np.fromiter(
        (float(part) for part in parts[1:]),
        dtype=np.float32,
        count=len(parts) - 1,
    )
    return word, values


def _load_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse a word2vec/GloVe text file into words + a float matrix.

    Handles the optional ``<vocab> <dim>`` header line emitted by gensim-data
    files as well as headerless files.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError(f"empty vector file: {path}")
        parts = first_line.split()
        has_header = len(parts) == 2 and all(part.isdigit() for part in parts)
        words: list[str] = []
        rows: list[np.ndarray] = []
        if not has_header:
            word, values = _parse_line(first_line)
            words.append(word)
            rows.append(values)
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            word, values = _parse_line(line)
            words.append(word)
            rows.append(values)
    matrix = np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)
    return words, matrix


def download_vectors(
    model_name: str = DEFAULT_MODEL,
    *,
    cache_dir: Path | str | None = None,
    timeout: float = 120.0,
) -> Path:
    """Ensure a gensim-data vector file is cached locally; return its path."""
    import httpx

    root = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "renewal"
    target = root / model_name / f"{model_name}.gz"
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_DOWNLOAD_BASE}/{model_name}/{model_name}.gz"
    LOGGER.info("Downloading word vectors from %s", url)
    temporary = target.with_suffix(".gz.part")
    try:
        with httpx.stream(
            "GET", url, timeout=timeout, follow_redirects=True
        ) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1 << 20):
                    handle.write(chunk)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return target
