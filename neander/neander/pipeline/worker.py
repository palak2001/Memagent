"""Async background write path — extraction, safety, conflict resolution.

Design choice: writing never blocks the response. A background thread drains a queue
of (user_message, assistant_response) pairs. For each extracted fact the worker:
  1. Runs the safety filter (drop credentials, tag PII).
  2. Checks for near-duplicate active facts in the same category (cosine > dedupe_similarity).
  3. Checks for contradictions among high-similarity facts (cosine > conflict_similarity)
     using a cheap LLM judge; if contradicted, invalidates the old fact first.
  4. Inserts the new fact with its embedding.

The conflict judge only fires when cosine similarity is high (rare) and category
matches — so it costs almost nothing and runs async anyway.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import List, Optional

import numpy as np

from ..config import Settings
from ..memory.embeddings import Embedder, cosine
from .extractor import extract_facts
from ..llm import LLMProvider
from ..storage.models import Memory, now_iso
from .safety import filter_fact
from ..storage.store import MemoryStore

_CONFLICT_PROMPT = """\
Fact A: {fact_a}
Fact B: {fact_b}

Does Fact B **contradict or update** Fact A about the same topic?
Reply with JSON: {{"verdict": "contradicts"}} or {{"verdict": "compatible"}}.
No other keys, no markdown.
"""


def _judge_conflict(provider: LLMProvider, old: str, new: str) -> bool:
    """Return True if ``new`` contradicts or updates ``old``."""
    try:
        result = provider.extract_json(
            _CONFLICT_PROMPT.format(fact_a=old, fact_b=new)
        )
        return result.get("verdict", "compatible") == "contradicts"
    except Exception:
        return False


class ExtractionWorker:
    """Background thread that drains an exchange queue and writes memories."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder,
        settings: Settings,
        provider: LLMProvider,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._settings = settings
        self._provider = provider
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._running = False
        self._queue.put(None)  # sentinel
        if self._thread:
            self._thread.join(timeout=timeout)

    def enqueue(self, user_message: str, assistant_message: str) -> None:
        """Add an exchange to the write queue."""
        self._queue.put((user_message, assistant_message))

    def drain(self, timeout: float = 10.0) -> None:
        """Block until the queue is empty (used in tests to wait for writes)."""
        self._queue.join()

    def _run(self) -> None:
        # Loop until the sentinel (None) arrives. The sentinel is always placed
        # AFTER all previously-queued items (FIFO), so every enqueued exchange
        # is guaranteed to be processed before the thread exits.
        while True:
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if item is None:  # shutdown sentinel
                self._queue.task_done()
                break
            try:
                self._process(*item)
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def _process(self, user_message: str, assistant_message: str) -> None:
        facts = extract_facts(self._provider, user_message, assistant_message)
        for fact in facts:
            self._write_fact(fact["content"], fact["category"])

    def _write_fact(self, content: str, category: str) -> None:
        drop, tag_pii = filter_fact(content)
        if drop:
            return

        embedding = self._embedder.embed(content)

        # PII facts are stored but never surfaced in prompts
        if tag_pii:
            mem = Memory(content=content, category=category, embedding=embedding, is_pii=True)
            self._store.add(mem)
            return

        # Build a similarity-sorted list of same-category active facts once.
        actives = [m for m in self._store.list_active() if m.category == category and m.embedding is not None]
        scored = sorted(
            [(m, cosine(embedding, m.embedding)) for m in actives],
            key=lambda x: x[1],
            reverse=True,
        )

        # Unified dedupe + conflict pass over the top-5 most-similar candidates.
        #
        # For high-similarity facts (≥ dedupe threshold) we MUST ask the judge before
        # skipping — "backend engineer" and "frontend engineer" score 0.94 similarity
        # yet contradict each other, so blind deduplication would hide the update.
        #
        # Logic:
        #   - judge says "contradicts" → mark for invalidation (always)
        #   - judge says "compatible" AND sim ≥ dedupe threshold → it's a true
        #     duplicate; don't store the new fact (is_dup = True)
        #   - judge says "compatible" AND sim < dedupe threshold → related but
        #     distinct; store both (the new fact is genuinely new)
        is_dup = False
        to_invalidate: list = []

        for existing, sim in scored[:5]:
            if sim < self._settings.conflict_similarity:
                break  # sorted desc; remaining are all below threshold
            if _judge_conflict(self._provider, existing.content, content):
                to_invalidate.append(existing.id)
            elif sim >= self._settings.dedupe_similarity:
                is_dup = True  # a near-identical, compatible fact already exists

        for mem_id in to_invalidate:
            self._store.invalidate(mem_id)

        if not is_dup:
            self._store.add(Memory(content=content, category=category, embedding=embedding))
