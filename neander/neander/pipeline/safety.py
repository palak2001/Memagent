"""PII and credential filter for the write path.

Design choice: the filter runs synchronously before any write. Credentials (API keys,
SSNs, card numbers, labelled secrets) are dropped entirely — they are never stored.
Email addresses and phone numbers are tagged ``is_pii=True`` and stored, but are
never injected into prompts by the retrieval layer.

Running this synchronously (not in the background worker) ensures nothing slips
through during a race between enqueueing and processing.
"""

from __future__ import annotations

import re

# Patterns whose presence causes the entire fact to be dropped.
_DROP_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9\-_]{20,}", re.I),          # OpenAI-style keys
    re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}", re.I),       # Anthropic keys
    re.compile(r"AKIA[0-9A-Z]{16}", re.I),                  # AWS access key IDs
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                   # US SSN
    re.compile(r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b"),  # Credit card
    re.compile(r"(?:password|passwd|secret|api[_-]?key)\s*[:=]\s*\S+", re.I),
    re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]{20,}", re.I), # Bearer tokens
    re.compile(r"[A-Za-z0-9+/]{40,}={0,2}"),               # Base64 blobs (API keys)
]

# Patterns whose presence tags the fact as PII (stored but never prompted).
_PII_PATTERNS = [
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(r"\b(\+1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"),  # phone
]


def is_credential(text: str) -> bool:
    """Return True if ``text`` contains a credential or secret that must be dropped."""
    return any(pat.search(text) for pat in _DROP_PATTERNS)


def is_pii(text: str) -> bool:
    """Return True if ``text`` contains PII (email, phone) that must never be prompted."""
    return any(pat.search(text) for pat in _PII_PATTERNS)


def filter_fact(content: str) -> tuple[bool, bool]:
    """Check a candidate fact.

    Returns:
        (drop, tag_pii) — if ``drop`` is True the fact must not be stored at all.
        If ``tag_pii`` is True the fact should be stored with ``is_pii=True``.
    """
    if is_credential(content):
        return True, False
    return False, is_pii(content)
