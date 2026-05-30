"""REPL entrypoint for the Neander memory agent.

Run with: python -m neander.cli
"""

from __future__ import annotations

import os
import sys

# Load .env as early as possible — the hf_xet native extension checks HF_TOKEN
# before Python's lazy imports get a chance to set it via load_dotenv() inside Settings.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass

# Suppress the HuggingFace Hub "unauthenticated requests" notice.
# hf_xet emits it from Rust native code; setting verbosity + token early suppresses it.
os.environ.setdefault("HUGGINGFACE_HUB_VERBOSITY", "error")


def main() -> None:
    from .agent.agent import Agent
    from .agent.commands import handle_command
    from .config import load_settings
    from .memory.embeddings import get_embedder
    from .llm import get_provider
    from .storage.store import MemoryStore
    from .pipeline.worker import ExtractionWorker

    settings = load_settings()
    store = MemoryStore(settings.db_path)
    embedder = get_embedder(
        settings.embed_backend, settings.embed_model_name, settings.embed_dim
    )
    provider = get_provider(settings)
    agent = Agent(store, embedder, settings, provider)
    worker = ExtractionWorker(store, embedder, settings, provider)
    worker.start()

    print("Neander  —  /memory list | forget <desc> | delete <id>  |  /quit to exit")
    print(f"Provider: {settings.provider}   Model: {settings.chat_model}")
    print(f"Active memories: {store.count_active()}\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not user_input:
                continue

            if user_input in ("/quit", "/exit", "exit", "quit"):
                print("Bye.")
                break

            if user_input.startswith("/memory"):
                # Drain in-flight extractions so the list/forget commands see
                # up-to-date data; typical wait is one LLM extraction call.
                worker.drain(timeout=30)
                result = handle_command(user_input, store, embedder, settings)
                print(result or "")
                continue

            print("Agent: ", end="", flush=True)
            try:
                for chunk in agent.chat(user_input):
                    print(chunk, end="", flush=True)
                print()
                # Reconstruct the full response from the turn buffer (agent
                # appended it after streaming completed).
                last_response = (
                    agent.turn_buffer[-1]["content"] if agent.turn_buffer else ""
                )
                worker.enqueue(user_input, last_response)
            except Exception as exc:
                print(f"\n[error: {exc}]")

    finally:
        # Drain ensures all queued extractions finish before the DB closes.
        print("\n[saving memories...]", end="", flush=True)
        worker.stop(timeout=30)
        print(" done.")
        store.close()


if __name__ == "__main__":
    main()
