"""M2 gate: relevance ranking, invalidated-fact exclusion, and scale timing."""

import time

import numpy as np
import pytest

from neander.config import Settings
from neander.memory.embeddings import HashingEmbedder, get_embedder
from neander.storage.models import Memory
from neander.memory.retrieval import retrieve, retrieve_scored
from neander.storage.store import MemoryStore


@pytest.fixture
def setup(tmp_path):
    db_path = str(tmp_path / "retrieval_test.db")
    store = MemoryStore(db_path)
    embedder = HashingEmbedder(dim=384)
    settings = Settings(embed_backend="hash")
    yield store, embedder, settings
    store.close()


def _stored(store: MemoryStore, embedder, content: str, category: str = "fact") -> Memory:
    vec = embedder.embed(content)
    mem = Memory(content=content, category=category, embedding=vec)
    store.add(mem)
    return mem


def test_relevant_fact_ranks_first(setup):
    store, embedder, settings = setup
    m_python = _stored(store, embedder, "The user knows Python programming language")
    m_coffee = _stored(store, embedder, "The user drinks coffee every morning")
    m_guitar = _stored(store, embedder, "The user plays guitar on weekends")

    results = retrieve(store, embedder, "python programming", settings)
    assert results, "Expected at least one result"
    assert results[0].id == m_python.id, f"Python fact should rank first, got: {results[0].content}"


def test_invalidated_fact_not_returned(setup):
    store, embedder, settings = setup
    m = _stored(store, embedder, "The user prefers tabs for indentation")
    store.invalidate(m.id)

    results = retrieve(store, embedder, "tabs indentation preference", settings)
    ids = [r.id for r in results]
    assert m.id not in ids, "Invalidated memory should not appear in results"


def test_pii_fact_not_returned(setup):
    store, embedder, settings = setup
    vec = embedder.embed("email contact")
    mem = Memory(content="user@example.com", category="fact", embedding=vec, is_pii=True)
    store.add(mem)

    results = retrieve(store, embedder, "email contact", settings)
    ids = [r.id for r in results]
    assert mem.id not in ids, "PII memory should not appear in retrieval results"


def test_scale_retrieval_under_100ms(setup):
    """1,000 random facts retrieved in well under 100ms (brute-force cosine)."""
    store, embedder, settings = setup
    rng = np.random.default_rng(42)
    for i in range(1000):
        vec = rng.standard_normal(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        mem = Memory(content=f"Random fact number {i}", category="fact", embedding=vec)
        store.add(mem)

    start = time.perf_counter()
    retrieve(store, embedder, "random query about something", settings)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 100, f"Retrieval took {elapsed_ms:.1f}ms, expected <100ms"
