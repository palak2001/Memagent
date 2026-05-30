"""Relevance-based retrieval over the long-term store.

Design choice: at ~1,000 memories a brute-force cosine scan in numpy is sub-
millisecond, so there is no ANN index. The ranking blends three signals so that
semantic relevance dominates but ties break toward fresh, important memories:

    score = cosine + w_recency * recency_decay + w_importance * (importance - 1)

Two filters enforce the "forget the right things" policy without deletion:
PII-flagged memories are never surfaced into prompts, and memories older than
their category's TTL fall out of retrieval (they remain in the table for history).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Tuple

import numpy as np

from ..config import Settings
from .embeddings import Embedder, cosine_batch
from ..storage.models import Memory, parse_iso
from ..storage.store import MemoryStore

_RECENCY_HORIZON_DAYS = 30.0
_W_RECENCY = 0.10
_W_IMPORTANCE = 0.05


def _age_days(ts: str) -> float:
    delta = datetime.now(timezone.utc) - parse_iso(ts)
    return max(delta.total_seconds() / 86400.0, 0.0)


def _is_expired(mem: Memory, settings: Settings) -> bool:
    ttl = settings.category_ttl_days.get(mem.category)
    if ttl is None:
        return False
    return _age_days(mem.created_at) > ttl


def retrieve_scored(
    store: MemoryStore,
    embedder: Embedder,
    query: str,
    settings: Settings,
) -> List[Tuple[Memory, float]]:
    """Return active, non-expired, non-PII memories ranked by blended score."""
    candidates = [
        m
        for m in store.list_active()
        if m.embedding is not None and not m.is_pii and not _is_expired(m, settings)
    ]
    if not candidates:
        return []

    query_vec = embedder.embed(query)
    matrix = np.vstack([m.embedding for m in candidates])
    sims = cosine_batch(query_vec, matrix)

    scored = []
    for mem, sim in zip(candidates, sims):
        recency = math.exp(-_age_days(mem.created_at) / _RECENCY_HORIZON_DAYS)
        score = (
            float(sim)
            + _W_RECENCY * recency
            + _W_IMPORTANCE * (mem.importance - 1.0)
        )
        scored.append((mem, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[: settings.retrieval_top_k]


def retrieve(
    store: MemoryStore,
    embedder: Embedder,
    query: str,
    settings: Settings,
) -> List[Memory]:
    """Top-k relevant memories for a query (convenience wrapper)."""
    return [mem for mem, _ in retrieve_scored(store, embedder, query, settings)]
