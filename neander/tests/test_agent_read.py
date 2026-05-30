"""M3 gate: prompt assembly, memory injection, turn buffer (LLM mocked)."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from neander.agent.agent import Agent
from neander.config import Settings
from neander.memory.embeddings import HashingEmbedder
from neander.storage.models import Memory
from neander.storage.store import MemoryStore


@pytest.fixture
def env(tmp_path):
    db_path = str(tmp_path / "agent_test.db")
    store = MemoryStore(db_path)
    embedder = HashingEmbedder(dim=384)
    settings = Settings(
        embed_backend="hash",
        turn_buffer_size=3,
        retrieval_top_k=6,
    )

    mock_provider = MagicMock()
    mock_provider.chat_stream.return_value = iter(["Hello", " there!"])

    agent = Agent(store, embedder, settings, mock_provider)
    yield store, embedder, settings, mock_provider, agent
    store.close()


def _seed(store, embedder, content, category="preference"):
    vec = embedder.embed(content)
    mem = Memory(content=content, category=category, embedding=vec)
    store.add(mem)
    return mem


def test_memory_injected_into_prompt(env):
    store, embedder, settings, mock_provider, agent = env
    _seed(store, embedder, "The user prefers spaces over tabs for indentation")

    messages = agent.build_prompt_for_query("How should I indent code?")

    system_msg = messages[0]["content"]
    assert "spaces" in system_msg.lower(), (
        f"Expected 'spaces' in system prompt, got:\n{system_msg}"
    )


def test_turn_buffer_included_in_prompt(env):
    store, embedder, settings, mock_provider, agent = env

    # Simulate two prior turns
    agent._turn_buffer = [
        {"role": "user", "content": "First message"},
        {"role": "assistant", "content": "First reply"},
    ]

    messages = agent.build_prompt_for_query("Second message")
    roles = [m["role"] for m in messages]
    assert "user" in roles
    # The buffer messages should appear between system and the new user message
    contents = [m["content"] for m in messages]
    assert "First message" in contents


def test_turn_buffer_capped(env):
    store, embedder, settings, mock_provider, agent = env

    # Fill buffer with more turns than the cap (turn_buffer_size=3 → max 6 messages)
    for i in range(10):
        agent._turn_buffer.append({"role": "user", "content": f"msg {i}"})
        agent._turn_buffer.append({"role": "assistant", "content": f"reply {i}"})

    messages = agent.build_prompt_for_query("new question")
    # system + up to 6 buffer + 1 new user = at most 8
    assert len(messages) <= 8, f"Expected at most 8 messages, got {len(messages)}"


def test_chat_updates_turn_buffer(env):
    store, embedder, settings, mock_provider, agent = env
    mock_provider.chat_stream.return_value = iter(["Test response"])

    list(agent.chat("Hello there"))  # consume generator

    assert len(agent.turn_buffer) == 2
    assert agent.turn_buffer[0]["role"] == "user"
    assert agent.turn_buffer[1]["role"] == "assistant"
    assert "Test response" in agent.turn_buffer[1]["content"]
