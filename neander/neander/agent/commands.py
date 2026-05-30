"""User-facing memory editing commands: /memory list|forget|delete.

Design choice: editing is near-free because storage is plain SQLite. ``forget``
uses semantic search so users can describe what to forget in natural language
rather than knowing exact memory IDs.
"""

from __future__ import annotations

from typing import Optional

from ..config import Settings
from ..memory.embeddings import Embedder, cosine
from ..memory.retrieval import retrieve_scored
from ..storage.store import MemoryStore

# Minimum cosine similarity for /memory forget — prevents low-relevance collateral.
# Facts must be semantically close to the forget query to be affected.
_FORGET_THRESHOLD = 0.35


def cmd_memory_list(store: MemoryStore) -> str:
    """Return a formatted string of all active memories."""
    memories = store.list_active()
    if not memories:
        return "No active memories."
    lines = [f"Active memories ({len(memories)}):"]
    for m in memories:
        pii_tag = " [PII]" if m.is_pii else ""
        lines.append(f"  [{m.id[:8]}] ({m.category}){pii_tag}  {m.content}")
    return "\n".join(lines)


def cmd_memory_forget(
    store: MemoryStore,
    embedder: Embedder,
    settings: Settings,
    query: str,
) -> str:
    """Semantically search for memories matching ``query`` and invalidate them.

    Only invalidates memories whose raw cosine similarity to the query exceeds
    _FORGET_THRESHOLD (0.35), preventing unrelated facts from being swept up.
    """
    query_vec = embedder.embed(query)
    actives = [m for m in store.list_active() if m.embedding is not None]
    matches = sorted(
        [(m, cosine(query_vec, m.embedding)) for m in actives if cosine(query_vec, m.embedding) >= _FORGET_THRESHOLD],
        key=lambda x: x[1],
        reverse=True,
    )
    if not matches:
        return "No matching memories found."
    forgotten = []
    for mem, _ in matches:
        store.invalidate(mem.id)
        forgotten.append(f"  [{mem.id[:8]}] {mem.content}")
    return "Forgotten:\n" + "\n".join(forgotten)


def cmd_memory_delete(store: MemoryStore, memory_id: str) -> str:
    """Hard-delete a memory by its ID (or id prefix)."""
    all_memories = store.list_all()
    matches = [m for m in all_memories if m.id.startswith(memory_id)]
    if not matches:
        return f"No memory found with id starting '{memory_id}'."
    for m in matches:
        store.delete(m.id)
    return f"Deleted {len(matches)} memory(ies)."


def handle_command(
    raw: str,
    store: MemoryStore,
    embedder: Embedder,
    settings: Settings,
) -> Optional[str]:
    """Parse and dispatch a /memory command. Returns the output string or None."""
    parts = raw.strip().split(None, 2)
    if len(parts) < 2 or parts[0] != "/memory":
        return None

    sub = parts[1].lower()

    if sub == "list":
        return cmd_memory_list(store)

    if sub == "forget":
        query = parts[2] if len(parts) > 2 else ""
        if not query:
            return "Usage: /memory forget <description>"
        return cmd_memory_forget(store, embedder, settings, query)

    if sub == "delete":
        mem_id = parts[2] if len(parts) > 2 else ""
        if not mem_id:
            return "Usage: /memory delete <id>"
        return cmd_memory_delete(store, mem_id)

    return f"Unknown /memory sub-command '{sub}'. Use list | forget | delete."
