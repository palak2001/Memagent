"""Atomic fact extraction from conversation exchanges.

Design choice: only extracted atomic facts are stored — raw conversation turns are
never persisted as memories. A cheap LLM call with a strict JSON schema decomposes
each exchange into at most a handful of typed facts (preference / decision / fact /
context). Running this off the critical path (in the worker) keeps latency flat.
"""

from __future__ import annotations

import json
from typing import List

from ..llm import LLMProvider

_EXTRACT_PROMPT = """\
Extract durable, memory-worthy atomic facts from the conversation below.
These facts will be retrieved in future sessions to personalize the assistant's responses.

Return JSON matching this schema **exactly** — no markdown, no extra keys:
{{
  "facts": [
    {{"content": "<one-sentence fact>", "category": "preference|decision|fact|context"}}
  ]
}}

Category definitions:
- preference  : a lasting like/dislike, tool choice, coding style, or working habit
- decision    : a concrete commitment the user has made (project choice, migration plan)
- fact        : an objective fact about the user (job title, employer, skill, location)
- context     : short-lived situational info worth remembering for days (current project name, active task)

Rules:
1. One atomic fact per entry — no compound sentences. NEVER merge separate facts into one.
   "I'm Jordan, a backend engineer at Stripe" → three separate entries:
   "The user's name is Jordan.", "The user is a backend engineer.", "The user works at Stripe."
2. Phrase each fact as a standalone sentence starting with "The user …".
3. Extract ONLY from what the **User** explicitly stated. NEVER extract from what the
   Assistant said, restated, or confirmed — even if it accurately reflects the user.
4. Ask: "Would knowing this in a future session actually help?" If no, skip it.
5. SKIP: pleasantries, filler, compliments, opinions about third parties, casual remarks,
   statements of obvious common knowledge, anything that would be strange to recall later
   (e.g. "The user loves emojis"), and anything only relevant to this exact exchange.
6. SPECIFICITY: If one fact is a strict subset of another (e.g. "backend engineer" is
   implied by "backend engineer at Stripe"), extract ONLY the more specific one.
7. Do NOT extract the same fact in multiple phrasings — pick the most precise wording once.
8. Return {{"facts": []}} if nothing passes the bar.

---
User: {user_message}
Assistant: {assistant_message}
---
"""


def extract_facts(
    provider: LLMProvider,
    user_message: str,
    assistant_message: str,
) -> List[dict]:
    """Extract a list of {{content, category}} dicts from one exchange.

    Returns an empty list on any parsing failure — caller decides what to do.
    """
    prompt = _EXTRACT_PROMPT.format(
        user_message=user_message,
        assistant_message=assistant_message,
    )
    try:
        result = provider.extract_json(prompt)
        facts = result.get("facts", [])
        if not isinstance(facts, list):
            return []
        valid = []
        for f in facts:
            if isinstance(f, dict) and "content" in f and "category" in f:
                valid.append({"content": str(f["content"]), "category": str(f["category"])})
        return valid
    except Exception:
        return []
