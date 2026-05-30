"""Text embedding with a pluggable backend.

Design choice: the test suite must run offline with no model download, so embedding
is an interface with two backends:

* ``SentenceTransformerEmbedder`` — real semantic embeddings (all-MiniLM-L6-v2,
  384-dim). Powers real retrieval.
* ``HashingEmbedder`` — deterministic, dependency-free feature-hashing. Used in
  tests and as an offline fallback. Similarity is lexical, sufficient for controlled
  unit tests.

Both produce L2-normalised vectors of the same dimension (384), so cosine similarity
is a plain dot product and the backend is interchangeable with no other code changes.
``get_embedder`` caches the chosen backend per (backend, model, dim) triple.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray:  # pragma: no cover
        ...


def _normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


class HashingEmbedder:
    """Deterministic feature-hashing embedder — no network, no model download.

    Tokens are hashed into ``dim`` buckets with a signed hash; result is
    L2-normalised. Identical text always yields an identical vector.
    Cosine similarity reflects shared tokens.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            digest = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = digest % self.dim
            sign = 1.0 if (digest // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        return _normalise(vec)


class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers (lazy-loaded on first use)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        try:
            self.dim = int(self._model.get_embedding_dimension())
        except AttributeError:
            self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)


_CACHE: dict = {}


def get_embedder(
    backend: str = "auto",
    model_name: str = "all-MiniLM-L6-v2",
    dim: int = 384,
) -> Embedder:
    """Return a cached embedder for the requested backend.

    ``auto`` prefers sentence-transformers and silently falls back to hashing
    when unavailable (e.g. offline test runs or CI without model cache).
    """
    cache_key = (backend, model_name, dim)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    embedder: Embedder
    if backend == "hash":
        embedder = HashingEmbedder(dim=dim)
    elif backend == "sentence-transformers":
        embedder = SentenceTransformerEmbedder(model_name=model_name)
    else:  # auto
        try:
            embedder = SentenceTransformerEmbedder(model_name=model_name)
        except Exception:
            embedder = HashingEmbedder(dim=dim)

    _CACHE[cache_key] = embedder
    return embedder


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for already-normalised vectors (plain dot product)."""
    return float(np.dot(a, b))


def cosine_batch(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query against an (N, dim) matrix of rows."""
    if matrix.size == 0:
        return np.empty(0, dtype=np.float32)
    return matrix @ query
