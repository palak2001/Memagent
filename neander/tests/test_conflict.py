"""M5 gate: conflict resolution (invalidate-on-contradiction) and category TTL."""

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from neander.config import Settings
from neander.memory.embeddings import HashingEmbedder
from neander.storage.models import Memory, now_iso
from neander.storage.store import MemoryStore
from neander.pipeline.worker import ExtractionWorker


@pytest.fixture
def env(tmp_path):
    db_path = str(tmp_path / "conflict_test.db")
    store = MemoryStore(db_path)
    embedder = HashingEmbedder(dim=384)
    settings = Settings(
        embed_backend="hash",
        conflict_similarity=0.3,   # low threshold so hash embedder triggers it
        dedupe_similarity=0.98,
        category_ttl_days={
            "preference": 365,
            "decision": 90,
            "fact": 180,
            "context": 7,
        },
    )

    # Mock LLM provider: extraction returns a given fact; conflict judge says "contradicts"
    provider = MagicMock()
    provider.extract_json.return_value = {"verdict": "contradicts"}

    worker = ExtractionWorker(store, embedder, settings, provider)
    yield store, embedder, settings, provider, worker
    store.close()


def _seed(store, embedder, content, category="preference"):
    vec = embedder.embed(content)
    mem = Memory(content=content, category=category, embedding=vec)
    store.add(mem)
    return mem


def test_new_contradicting_fact_invalidates_old(env):
    """Storing 'prefers tabs' then 'prefers spaces' should invalidate tabs."""
    store, embedder, settings, provider, worker = env

    old_mem = _seed(store, embedder, "The user prefers tabs for indentation", "preference")

    # Directly call _write_fact (bypasses queue, synchronous)
    worker._write_fact("The user prefers spaces for indentation", "preference")

    # Old fact should now be invalidated
    old_fetched = store.get(old_mem.id)
    assert old_fetched.valid_until is not None, "Old preference should be invalidated"

    # New fact should be active
    active = store.list_active()
    active_contents = [m.content for m in active]
    assert any("spaces" in c for c in active_contents), "New preference should be active"


def test_old_fact_still_queryable_after_invalidation(env):
    """History is preserved — invalidated facts remain in the table."""
    store, embedder, settings, provider, worker = env

    old_mem = _seed(store, embedder, "The user prefers tabs for indentation", "preference")
    worker._write_fact("The user prefers spaces for indentation", "preference")

    # Should be findable via get() even though invalid
    fetched = store.get(old_mem.id)
    assert fetched is not None
    assert "tabs" in fetched.content


def test_context_fact_excluded_by_ttl(env):
    """A context fact older than TTL days should not appear in retrieval."""
    store, embedder, settings, provider, worker = env
    from neander.memory.retrieval import retrieve

    # Create a context fact with a past created_at timestamp
    old_date = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    vec = embedder.embed("Working on feature X today")
    mem = Memory(
        content="Working on feature X today",
        category="context",
        embedding=vec,
        created_at=old_date,
        valid_from=old_date,
    )
    store.add(mem)

    # TTL for context is 7 days; this fact is 10 days old → should be excluded
    results = retrieve(store, embedder, "feature X", settings)
    ids = [m.id for m in results]
    assert mem.id not in ids, "Context fact older than TTL should be excluded from retrieval"


def test_duplicate_fact_not_inserted(env):
    """Near-identical facts should not create duplicates."""
    store, embedder, settings, provider, worker = env

    # Set high dedupe threshold by using same content (cosine=1.0)
    settings_high_dedupe = Settings(
        embed_backend="hash",
        dedupe_similarity=0.95,
        conflict_similarity=0.99,
    )
    worker2 = ExtractionWorker(store, embedder, settings_high_dedupe, provider)

    _seed(store, embedder, "The user prefers Python", "preference")
    initial_count = store.count_active()

    worker2._write_fact("The user prefers Python", "preference")
    assert store.count_active() == initial_count, "Duplicate fact should not be inserted"
