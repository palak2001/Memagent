"""M7 gate: p50 first-token latency at 1,000 memories within 200ms of turn 1.

This test is marked slow — skip with: pytest -m 'not slow'
"""

import time
import statistics
from unittest.mock import MagicMock

import numpy as np
import pytest

from neander.agent.agent import Agent
from neander.config import Settings
from neander.memory.embeddings import HashingEmbedder
from neander.storage.models import Memory
from neander.storage.store import MemoryStore


@pytest.mark.slow
def test_latency_delta_under_200ms(tmp_path):
    """p50(turn_1000) - p50(turn_1) must be < 200ms."""
    db_path = str(tmp_path / "latency_test.db")
    store = MemoryStore(db_path)
    embedder = HashingEmbedder(dim=384)
    settings = Settings(embed_backend="hash")

    # Seed 1,000 facts
    rng = np.random.default_rng(0)
    for i in range(1000):
        vec = rng.standard_normal(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        store.add(Memory(content=f"Seeded fact {i}", category="fact", embedding=vec))

    mock_provider = MagicMock()

    def mock_stream(messages):
        # Simulate first-token: just yield one token immediately
        yield "token"

    mock_provider.chat_stream.side_effect = mock_stream

    agent_empty = Agent(
        MemoryStore(str(tmp_path / "empty.db")),
        embedder, settings, mock_provider,
    )
    agent_full = Agent(store, embedder, settings, mock_provider)

    SAMPLES = 20

    def measure_first_token(agent, query):
        start = time.perf_counter()
        gen = agent.chat(query)
        next(gen, None)
        return (time.perf_counter() - start) * 1000

    times_empty = [measure_first_token(agent_empty, f"query {i}") for i in range(SAMPLES)]
    times_full = [measure_first_token(agent_full, f"query {i}") for i in range(SAMPLES)]

    p50_empty = statistics.median(times_empty)
    p50_full = statistics.median(times_full)
    delta = p50_full - p50_empty

    store.close()

    print(f"\np50 (turn 1):    {p50_empty:.2f}ms")
    print(f"p50 (turn 1000): {p50_full:.2f}ms")
    print(f"delta:           {delta:.2f}ms")

    assert delta < 200, (
        f"Latency delta {delta:.1f}ms exceeds 200ms budget. "
        f"p50_empty={p50_empty:.1f}ms, p50_full={p50_full:.1f}ms"
    )
