"""Latency benchmark: p50 first-token at turn 1 vs turn 1,000.

Seeds 1,000 facts into the store, then measures the median first-token latency
across enough samples for a stable estimate. Uses the hash embedder and a mocked
LLM (only the read path matters for latency).

Usage: python benchmark.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from neander.agent.agent import Agent
from neander.config import Settings
from neander.memory.embeddings import HashingEmbedder
from neander.storage.models import Memory
from neander.storage.store import MemoryStore

SAMPLES = 30
SEED_COUNT = 1000
_DB_EMPTY = "bench_empty.db"
_DB_FULL = "bench_full.db"


def _mock_provider():
    provider = MagicMock()
    provider.chat_stream.side_effect = lambda msgs: iter(["token"])
    return provider


def _make_store(path: str, n_facts: int) -> MemoryStore:
    if os.path.exists(path):
        os.remove(path)
    store = MemoryStore(path)
    rng = np.random.default_rng(42)
    for i in range(n_facts):
        vec = rng.standard_normal(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        store.add(Memory(content=f"Fact {i}", category="fact", embedding=vec))
    return store


def _measure(agent: Agent, queries: list[str]) -> list[float]:
    times = []
    for q in queries:
        start = time.perf_counter()
        gen = agent.chat(q)
        next(gen, None)
        times.append((time.perf_counter() - start) * 1000)
    return times


def main() -> None:
    settings = Settings(embed_backend="hash")
    embedder = HashingEmbedder(dim=384)

    print(f"Building stores ({SEED_COUNT} facts for full, 0 for empty)...")
    store_empty = _make_store(_DB_EMPTY, 0)
    store_full = _make_store(_DB_FULL, SEED_COUNT)

    queries = [f"benchmark query number {i}" for i in range(SAMPLES)]

    agent_empty = Agent(store_empty, embedder, settings, _mock_provider())
    agent_full = Agent(store_full, embedder, settings, _mock_provider())

    print(f"Measuring p50 over {SAMPLES} samples each...")
    times_empty = _measure(agent_empty, queries)
    times_full = _measure(agent_full, queries)

    p50_empty = statistics.median(times_empty)
    p50_full = statistics.median(times_full)
    delta = p50_full - p50_empty

    store_empty.close()
    store_full.close()

    print()
    print(f"  p50 first-token (0 memories):       {p50_empty:.2f} ms")
    print(f"  p50 first-token ({SEED_COUNT} memories): {p50_full:.2f} ms")
    print(f"  delta:                              {delta:.2f} ms")
    print()

    if delta < 200:
        print(f"  PASS — delta {delta:.1f}ms is within the 200ms budget.")
    else:
        print(f"  FAIL — delta {delta:.1f}ms exceeds the 200ms budget.")

    # Cleanup
    for path in (_DB_EMPTY, _DB_FULL):
        if os.path.exists(path):
            os.remove(path)


if __name__ == "__main__":
    main()
