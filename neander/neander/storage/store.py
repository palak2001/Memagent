"""SQLite-backed persistent memory store.

Design choice: a single SQLite file gives cross-session persistence with zero
infrastructure. Embeddings are stored inline as float32 BLOBs alongside their
dimension. The connection is guarded by a lock because the async write worker
and the foreground read path share the same store.

Reads always filter ``valid_until IS NULL`` so retired memories never surface,
yet remain in the table for history and audit.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import List, Optional

import numpy as np

from .models import Memory, now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    content       TEXT NOT NULL,
    category      TEXT NOT NULL,
    valid_from    TEXT NOT NULL,
    valid_until   TEXT,                 -- NULL = currently active
    created_at    TEXT NOT NULL,
    importance    REAL NOT NULL DEFAULT 1.0,
    is_pii        INTEGER NOT NULL DEFAULT 0,
    embedding     BLOB,                 -- float32 bytes
    embedding_dim INTEGER
);
CREATE INDEX IF NOT EXISTS idx_category    ON memories(category);
CREATE INDEX IF NOT EXISTS idx_valid_until ON memories(valid_until);
"""


def _to_blob(vec: Optional[np.ndarray]):
    if vec is None:
        return None, None
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes(), int(arr.shape[0])


def _from_blob(blob, dim) -> Optional[np.ndarray]:
    if blob is None or dim is None:
        return None
    return np.frombuffer(blob, dtype=np.float32, count=int(dim)).copy()


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=row["id"],
        content=row["content"],
        category=row["category"],
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        created_at=row["created_at"],
        importance=row["importance"],
        is_pii=bool(row["is_pii"]),
        embedding=_from_blob(row["embedding"], row["embedding_dim"]),
    )


class MemoryStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # --- writes ---

    def add(self, memory: Memory) -> str:
        """Insert or replace a memory record; returns the memory id."""
        blob, dim = _to_blob(memory.embedding)
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO memories
                   (id, content, category, valid_from, valid_until, created_at,
                    importance, is_pii, embedding, embedding_dim)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory.id, memory.content, memory.category,
                    memory.valid_from, memory.valid_until, memory.created_at,
                    memory.importance, int(memory.is_pii), blob, dim,
                ),
            )
            self._conn.commit()
        return memory.id

    def invalidate(self, memory_id: str, when: Optional[str] = None) -> None:
        """Retire a memory by closing its validity window (no data loss)."""
        when = when or now_iso()
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET valid_until=? WHERE id=? AND valid_until IS NULL",
                (when, memory_id),
            )
            self._conn.commit()

    def bump_importance(self, memory_id: str, delta: float = 0.1) -> None:
        """Increase importance score when a memory is accessed."""
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET importance = importance + ? WHERE id=?",
                (delta, memory_id),
            )
            self._conn.commit()

    def delete(self, memory_id: str) -> None:
        """Hard delete — used only by explicit user ``/memory delete <id>``."""
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
            self._conn.commit()

    # --- reads ---

    def get(self, memory_id: str) -> Optional[Memory]:
        """Fetch a single memory by id (active or retired)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        return _row_to_memory(row) if row else None

    def list_active(self) -> List[Memory]:
        """All currently active memories (``valid_until IS NULL``)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE valid_until IS NULL"
            ).fetchall()
        return [_row_to_memory(r) for r in rows]

    def list_all(self) -> List[Memory]:
        """All memories including retired ones (for history queries)."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM memories").fetchall()
        return [_row_to_memory(r) for r in rows]

    def count_active(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE valid_until IS NULL"
            ).fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
