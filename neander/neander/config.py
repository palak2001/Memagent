"""Central configuration for Neander.

Design choice: all tunables live here as a single frozen ``Settings`` object loaded
from environment variables with safe defaults. Nothing else in the codebase reads
``os.environ`` directly — this keeps the system testable and makes every knob
auditable in one place.

The ``provider`` field drives which LLM adapter is used. Swapping OpenAI → Gemini
is a one-line env-var change (``LLM_PROVIDER=gemini``) with no code changes elsewhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """All runtime configuration for the agent.

    Provider selection and model names are read from environment variables so
    the same codebase works with OpenAI or Gemini without code changes.
    """

    # --- LLM provider ("openai" | "gemini") ---
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai"))
    chat_model: str = field(default_factory=lambda: _env("CHAT_MODEL", "gpt-4o"))
    extract_model: str = field(
        default_factory=lambda: _env("EXTRACT_MODEL", "gpt-4o-mini")
    )

    # --- Embeddings ---
    # "auto" tries sentence-transformers; tests force "hash" for offline/hermetic runs.
    embed_backend: str = field(default_factory=lambda: _env("EMBED_BACKEND", "auto"))
    embed_model_name: str = field(
        default_factory=lambda: _env("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")
    )
    embed_dim: int = field(default_factory=lambda: _env_int("EMBED_DIM", 384))

    # --- Storage ---
    db_path: str = field(default_factory=lambda: _env("DB_PATH", "neander_memory.db"))

    # --- Read path ---
    turn_buffer_size: int = field(
        default_factory=lambda: _env_int("TURN_BUFFER_SIZE", 6)
    )
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 6))
    candidate_pool: int = field(default_factory=lambda: _env_int("CANDIDATE_POOL", 15))

    # --- Write path / conflict resolution ---
    conflict_similarity: float = field(
        default_factory=lambda: _env_float("CONFLICT_SIMILARITY", 0.50)
    )
    dedupe_similarity: float = field(
        default_factory=lambda: _env_float("DEDUPE_SIMILARITY", 0.92)
    )

    # --- Categories & retention (days) ---
    category_ttl_days: dict = field(
        default_factory=lambda: {
            "preference": 365,
            "decision": 90,
            "fact": 180,
            "context": 7,
        }
    )

    @property
    def api_key(self) -> str:
        """Read the provider API key lazily; only the LLM client needs it."""
        if self.provider == "gemini":
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError(
                    "No Gemini API key found. Set GEMINI_API_KEY in your .env file."
                )
        else:
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "No OpenAI API key found. Set OPENAI_API_KEY in your .env file."
                )
        return key


def load_settings() -> Settings:
    """Build a Settings object from the current environment."""
    return Settings()
