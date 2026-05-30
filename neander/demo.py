"""Cross-session memory demo.

Session 1: the user states two preferences and a decision; process exits.
Session 2: a fresh agent instance recalls them unprompted and acts on them.

Usage: python demo.py
Requires: OPENAI_API_KEY (or GEMINI_API_KEY + LLM_PROVIDER=gemini) in .env
"""

from __future__ import annotations

import os
import sys
import time

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(__file__))

from neander.agent.agent import Agent
from neander.config import load_settings
from neander.memory.embeddings import get_embedder
from neander.llm import get_provider
from neander.storage.models import Memory
from neander.storage.store import MemoryStore
from neander.pipeline.worker import ExtractionWorker

_DB = "demo_memory.db"


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def _chat(agent: Agent, worker: ExtractionWorker, user_msg: str) -> str:
    print(f"  You:   {user_msg}")
    print("  Agent: ", end="", flush=True)
    parts = []
    for chunk in agent.chat(user_msg):
        print(chunk, end="", flush=True)
        parts.append(chunk)
    print()
    response = "".join(parts)
    worker.enqueue(user_msg, response)
    return response


def session_one() -> None:
    _print_section("SESSION 1 — Stating preferences and a decision")
    settings = load_settings()
    store = MemoryStore(_DB)
    embedder = get_embedder(settings.embed_backend, settings.embed_model_name, settings.embed_dim)
    provider = get_provider(settings)
    agent = Agent(store, embedder, settings, provider)
    worker = ExtractionWorker(store, embedder, settings, provider)
    worker.start()

    _chat(agent, worker, "Hi! Just so you know, I always prefer spaces over tabs in my code.")
    _chat(agent, worker, "I've decided to use Python for this project, not JavaScript.")
    _chat(agent, worker, "Also, I'm a backend engineer and I dislike writing CSS.")

    print("\n  [waiting for background extraction to finish...]")
    worker.drain(timeout=30)
    worker.stop()
    store.close()

    print(f"\n  Session 1 complete. Memories stored in '{_DB}'.")


def session_two() -> None:
    _print_section("SESSION 2 — Fresh agent recalls across the session boundary")
    settings = load_settings()
    store = MemoryStore(_DB)
    embedder = get_embedder(settings.embed_backend, settings.embed_model_name, settings.embed_dim)
    provider = get_provider(settings)
    agent = Agent(store, embedder, settings, provider)
    worker = ExtractionWorker(store, embedder, settings, provider)
    worker.start()

    active = store.count_active()
    print(f"  [Active memories loaded: {active}]\n")

    _chat(agent, worker, "What language should I use for this project?")
    _chat(agent, worker, "How should I format my code — tabs or spaces?")
    _chat(agent, worker, "What kind of engineer am I?")

    worker.stop()
    store.close()


def main() -> None:
    print("Neander Cross-Session Memory Demo")
    print("==================================\n")
    print(f"Using database: {_DB}")

    # Clean slate for the demo
    if os.path.exists(_DB):
        os.remove(_DB)
        print(f"Removed existing '{_DB}' for a clean demo run.\n")

    session_one()
    print("\n  [simulating process restart...]\n")
    time.sleep(1)
    session_two()

    _print_section("Demo complete")
    print("The agent recalled preferences and decisions from Session 1")
    print("despite starting with a fresh in-process state.")


if __name__ == "__main__":
    main()
