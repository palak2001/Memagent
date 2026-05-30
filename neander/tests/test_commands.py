"""M6 gate: /memory list, forget, delete commands."""

import pytest

from neander.agent.commands import cmd_memory_delete, cmd_memory_forget, cmd_memory_list, handle_command
from neander.config import Settings
from neander.memory.embeddings import HashingEmbedder
from neander.storage.models import Memory
from neander.storage.store import MemoryStore


@pytest.fixture
def env(tmp_path):
    db_path = str(tmp_path / "cmd_test.db")
    store = MemoryStore(db_path)
    embedder = HashingEmbedder(dim=384)
    settings = Settings(embed_backend="hash", retrieval_top_k=6)
    yield store, embedder, settings
    store.close()


def _seed(store, embedder, content, category="preference"):
    vec = embedder.embed(content)
    mem = Memory(content=content, category=category, embedding=vec)
    store.add(mem)
    return mem


def test_memory_list_shows_active(env):
    store, embedder, settings = env
    m = _seed(store, embedder, "The user likes dark mode")
    output = cmd_memory_list(store)
    assert "dark mode" in output
    assert m.id[:8] in output


def test_memory_list_empty(env):
    store, embedder, settings = env
    output = cmd_memory_list(store)
    assert "No active memories" in output


def test_memory_forget_invalidates_match(env):
    store, embedder, settings = env
    m = _seed(store, embedder, "The user prefers tabs for indentation")
    output = cmd_memory_forget(store, embedder, settings, "tabs indentation")
    assert m.id[:8] in output
    assert store.get(m.id).valid_until is not None


def test_memory_list_excludes_invalidated(env):
    store, embedder, settings = env
    m = _seed(store, embedder, "The user prefers tabs")
    store.invalidate(m.id)
    output = cmd_memory_list(store)
    assert "tabs" not in output


def test_memory_delete_removes_row(env):
    store, embedder, settings = env
    m = _seed(store, embedder, "Fact to hard delete")
    cmd_memory_delete(store, m.id[:8])
    assert store.get(m.id) is None


def test_handle_command_dispatches_list(env):
    store, embedder, settings = env
    result = handle_command("/memory list", store, embedder, settings)
    assert result is not None


def test_handle_command_unknown_subcommand(env):
    store, embedder, settings = env
    result = handle_command("/memory blah", store, embedder, settings)
    assert result is not None
    assert "Unknown" in result


def test_handle_command_non_memory(env):
    store, embedder, settings = env
    result = handle_command("/help", store, embedder, settings)
    assert result is None
