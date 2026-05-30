"""M1 gate: storage CRUD, active filtering, persistence, invalidation."""

import os
import tempfile

import numpy as np
import pytest

from neander.storage.models import Memory
from neander.storage.store import MemoryStore


@pytest.fixture
def tmp_store(tmp_path):
    path = str(tmp_path / "test.db")
    store = MemoryStore(path)
    yield store
    store.close()


def _make_mem(content: str, category: str = "fact") -> Memory:
    vec = np.random.rand(384).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return Memory(content=content, category=category, embedding=vec)


def test_add_get_roundtrip(tmp_store):
    mem = _make_mem("The user likes dark mode.")
    tmp_store.add(mem)
    fetched = tmp_store.get(mem.id)
    assert fetched is not None
    assert fetched.content == mem.content
    assert fetched.category == mem.category
    assert fetched.valid_until is None


def test_list_active_excludes_invalidated(tmp_store):
    m1 = _make_mem("Fact one")
    m2 = _make_mem("Fact two")
    tmp_store.add(m1)
    tmp_store.add(m2)
    tmp_store.invalidate(m1.id)

    active_ids = {m.id for m in tmp_store.list_active()}
    assert m1.id not in active_ids
    assert m2.id in active_ids


def test_persistence_across_reopen(tmp_path):
    """The cross-session guarantee: data survives closing and reopening the file."""
    db_path = str(tmp_path / "persist.db")
    mem = _make_mem("The user prefers tabs.")

    store1 = MemoryStore(db_path)
    store1.add(mem)
    store1.close()

    store2 = MemoryStore(db_path)
    fetched = store2.get(mem.id)
    store2.close()

    assert fetched is not None
    assert fetched.content == "The user prefers tabs."


def test_invalidate_sets_valid_until_not_deletes(tmp_store):
    mem = _make_mem("Stale fact")
    tmp_store.add(mem)
    tmp_store.invalidate(mem.id)

    # Should NOT appear in active list
    active_ids = {m.id for m in tmp_store.list_active()}
    assert mem.id not in active_ids

    # Should still be queryable via get (history preserved)
    fetched = tmp_store.get(mem.id)
    assert fetched is not None
    assert fetched.valid_until is not None


def test_delete_removes_row(tmp_store):
    mem = _make_mem("Fact to delete")
    tmp_store.add(mem)
    tmp_store.delete(mem.id)
    assert tmp_store.get(mem.id) is None


def test_embedding_roundtrip(tmp_store):
    """Embeddings survive the BLOB roundtrip."""
    vec = np.array([0.1, 0.2, 0.3] * 128, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    mem = Memory(content="embed test", embedding=vec)
    tmp_store.add(mem)
    fetched = tmp_store.get(mem.id)
    assert fetched.embedding is not None
    np.testing.assert_allclose(fetched.embedding, vec, atol=1e-6)
