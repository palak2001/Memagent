"""Read path: retrieve relevant memories, assemble system prompt, stream response.

Design choice: two-tier memory — a recent-turns buffer always in the prompt handles
short-term recall for free, and the long-term SQLite store covers cross-session facts.
Writing happens on a background worker (see worker.py) and never touches this path,
so first-token latency grows only with the (sub-ms) cosine scan, not with extraction.

Prompt assembly order: system instructions → memories grouped by category → recent turns.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator, List

from ..config import Settings
from ..memory.embeddings import Embedder
from ..llm import LLMProvider
from ..storage.models import Memory
from ..memory.retrieval import retrieve, retrieve_scored
from ..storage.store import MemoryStore

_SYSTEM_BASE = """\
You are a knowledgeable, thoughtful assistant that knows the user personally \
over time. A separate memory system automatically records what the user shares \
— you do NOT need to say "I'll remember that" or announce that you're storing \
anything. Just use what you know naturally, the way a trusted colleague would.

Guidelines:
- When the user shares personal info (name, role, preferences, tools, decisions), \
  acknowledge it and engage with it — do NOT interpret it as a request to take action.
- Use stored memories naturally in your answers without quoting them verbatim or \
  saying "according to my memory."
- Be concise. End when you've answered. NEVER append filler phrases like "feel free \
  to ask", "let me know if you need anything", or "I'm here to help."
- If a memory seems to conflict with what the user just said, defer to the new info.\
"""


def _format_memories(memories: List[Memory]) -> str:
    """Group memories by category and format them for the system prompt."""
    if not memories:
        return ""
    by_cat: dict = defaultdict(list)
    for m in memories:
        by_cat[m.category].append(m.content)

    lines = ["\n## What I remember about you\n"]
    order = ["preference", "decision", "fact", "context"]
    for cat in order:
        if cat not in by_cat:
            continue
        lines.append(f"**{cat.capitalize()}s:**")
        for content in by_cat[cat]:
            lines.append(f"- {content}")
    return "\n".join(lines)


class Agent:
    """Conversational agent with long-term memory on the read path."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        settings: Settings,
        provider: LLMProvider,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.provider = provider
        # Recent-turns buffer: list of {"role": ..., "content": ...}
        self._turn_buffer: List[dict] = []

    def chat(self, user_message: str) -> Iterator[str]:
        """Retrieve memories, assemble prompt, stream response.

        Yields response text chunks as they arrive from the LLM.
        The turn buffer is updated after the full response is collected.
        """
        memories = retrieve_scored(
            self.store, self.embedder, user_message, self.settings
        )
        # Bump importance for accessed memories
        for mem, _ in memories:
            self.store.bump_importance(mem.id)

        memory_list = [m for m, _ in memories]
        system_content = _SYSTEM_BASE + _format_memories(memory_list)

        messages: List[dict] = [{"role": "system", "content": system_content}]
        # Append the capped recent-turns buffer
        buffer_window = self._turn_buffer[-(self.settings.turn_buffer_size * 2):]
        messages.extend(buffer_window)
        messages.append({"role": "user", "content": user_message})

        response_parts: List[str] = []
        for chunk in self.provider.chat_stream(messages):
            response_parts.append(chunk)
            yield chunk

        response = "".join(response_parts)

        # Update turn buffer
        self._turn_buffer.append({"role": "user", "content": user_message})
        self._turn_buffer.append({"role": "assistant", "content": response})
        # Trim to configured size (pairs of user+assistant)
        max_messages = self.settings.turn_buffer_size * 2
        if len(self._turn_buffer) > max_messages:
            self._turn_buffer = self._turn_buffer[-max_messages:]

    @property
    def turn_buffer(self) -> List[dict]:
        """Read-only view of the current turn buffer (for testing)."""
        return list(self._turn_buffer)

    def build_prompt_for_query(self, user_message: str) -> List[dict]:
        """Return the assembled messages list without streaming — used in tests."""
        memories = retrieve(self.store, self.embedder, user_message, self.settings)
        system_content = _SYSTEM_BASE + _format_memories(memories)
        messages: List[dict] = [{"role": "system", "content": system_content}]
        buffer_window = self._turn_buffer[-(self.settings.turn_buffer_size * 2):]
        messages.extend(buffer_window)
        messages.append({"role": "user", "content": user_message})
        return messages
