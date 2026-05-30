"""The ``Memory`` record — the single unit of long-term memory.

Design choice: every memory is an *atomic fact* (one preference/decision/fact, not
a raw conversation turn) carrying a validity window. ``valid_until is None`` means
the fact is currently true; setting it (rather than deleting the row) is how we
retire stale or contradicted memories while preserving history.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

VALID_CATEGORIES = ("preference", "decision", "fact", "context")


def now_iso() -> str:
    """Current UTC time as an ISO-8601 string (used for all timestamps)."""
    return datetime.now(timezone.utc).isoformat()


def parse_iso(ts: str) -> datetime:
    """Parse an ISO timestamp produced by :func:`now_iso`."""
    return datetime.fromisoformat(ts)


@dataclass
class Memory:
    """A single atomic memory.

    Attributes:
        id: stable unique id.
        content: the atomic fact, phrased as a standalone sentence.
        category: one of VALID_CATEGORIES; drives retention and surfacing.
        valid_from: when the fact started being true (ISO).
        valid_until: when it stopped being true; None while active (ISO).
        created_at: when we recorded it (ISO).
        importance: ranking weight; bumped on retrieval, decays with age.
        embedding: L2-normalised vector for similarity search.
        is_pii: flagged personal data — stored but never injected into prompts.
    """

    content: str
    category: str = "fact"
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    valid_from: str = field(default_factory=now_iso)
    valid_until: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    importance: float = 1.0
    embedding: Optional[np.ndarray] = None
    is_pii: bool = False

    @property
    def is_active(self) -> bool:
        return self.valid_until is None

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            self.category = "fact"
