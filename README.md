# Neander

## About

This is a Memory persistent conversational agent that remember across sessions.

Built directly on top of Openai api.

Focus areas

* What to remember
* What to forget
* How to handle stale or conflicting memories
* Stable latency as memory grows

## Architecture

Two tier memory architecture


```
┌─────────────────────────────────────────────────────────────────┐
│                        USER / CLI                               │
└───────────────────────────────┬─────────────────────────────────┘
                                │ user message
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  READ PATH  (sync — runs before every response)                 │
│                                                                 │
│  embed query  →  cosine over ACTIVE memories (valid_until NULL) │
│             →  rank: similarity × recency × category TTL        │
│             →  top-k                                            │
│             →  assemble prompt:                                 │
│                  [ system instructions         ]                │
│                  [ memories, grouped by category]               │
│                  [ recent-turns buffer, last ~6 ]               │
│             →  stream LLM response  →  user sees first token    │
└─────────────────────────────────────────────────────────────────┘
                                │ response done
                                ▼ enqueue(exchange) — user already unblocked
┌─────────────────────────────────────────────────────────────────┐
│  WRITE PATH  (async background worker)                          │
│                                                                 │
│  extract atomic facts   (cheap model, strict JSON schema)       │
│       ↓                                                         │
│  PII / credential filter  (sync regex — before any write)       │
│       ↓                                                         │
│  conflict check  (same-category, cosine > 0.85)                 │
│       ↓ contradiction found?                                    │
│       ├─ yes →  invalidate old (valid_until = now)              │
│       │         insert new as active                            │
│       └─ no  →  insert / skip duplicate                         │
│       ↓                                                         │
│  embed + commit  →  SQLite                                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  STORAGE  (SQLite, single file)                                 │
│                                                                 │
│  id · content · category · embedding (384-dim)                  │
│  importance · valid_from · valid_until · created_at             │
│                                                                 │
│  ACTIVE  →  valid_until IS NULL                                 │
│  HISTORY →  valid_until IS NOT NULL  (never hard-deleted)       │
│                                                                 │
│  Categories & TTL weighting:                                    │
│    preference  — long,   surfaced first                         │
│    decision    — medium, surfaced on relevance                  │
│    fact        — medium, surfaced on relevance                  │
│    context     — short,  deprioritised after ~session end       │
└─────────────────────────────────────────────────────────────────┘
```

Recent Turn Buffer

* Last few conversation turns
* Handles short term recall

Long Term Memory Store

* SQLite based persistence
* Stores atomic facts, not raw conversations
* Embeddings for retrieval
* Categories, importance and validity tracking

There are two flows integrated here,

Read Path

* Embed user query
* Retrieve relevant active memories (active memories are the one's whose validity is non null)
* Rank by relevance and recency
* Build prompt with memories and recent context
* Generate response

Write Path

* Runs asynchronously
* Extract atomic facts
* Filter sensitive information
* Detect conflicts
* Store embeddings and metadata

## Setup

Refer to SETUP.md

For demo, refer to ConversationSample.txt

## Design choices, tradeoffs and reasoning

Atomic facts over raw conversation

* Better retrieval quality
* Easier conflict handling
* Less noise

Async memory writes

* Keeps response latency stable
* Memory may appear one turn later

SQLite over external infrastructure

* Simple
* Persistent
* Easy to inspect and edit

Brute force cosine search

* Fast enough for current scale
* Avoids vector database complexity

Validity windows instead of deletion

* Preserves history
* Supports memory updates and corrections

Two tier memory instead of multi tier memory

* Simpler architecture
* Sufficient for assignment scale

----------
## Benchmark

Measuring p50 over 30 samples each...

p50 first-token (0 memories):       0.01 ms
p50 first-token (1000 memories): 6.17 ms
delta:                              6.16 ms

PASS — delta 6.2ms is within the 200ms budget.

-----

## Tests

The core memory logic is covered end to end, all hermetic:

- **Capture** — an exchange produces typed, stored facts.
- **Persistence** — write, close the connection, reopen on the same file, facts survive. This is the cross-session guarantee in miniature.
- **Retrieval** — a query ranks the relevant fact first and never returns an invalidated one.
- **Conflict** — "tabs" then "spaces" invalidates the old and activates the new, and the old is still reachable as history.
- **Safety** — a message containing a fake API key and an SSN stores neither.
- **Latency** — delta between turn 1 and turn 1,000 stays under budget.

---

## What should be done next

- Add an importance score so memories that are frequently useful are surfaced more often, instead of relying on fixed ranking weights.
- Introduce an episodic memory layer where related conversations are grouped together. This would help retrieve surrounding context instead of isolated facts.
- Move from a flat memory store to a temporal graph of entities and relationships. This would make cross-session reasoning much stronger as the memory base grows.
- Add a periodic reflection job to clean up stale memories, merge duplicates and detect conflicts that were missed during normal processing.


## Time spent

Roughly 4 hours, broken down honestly:

**~1 hr 15 min — problem understanding and research.**
I didn't start writing code until I'd read the spec carefully and spent real time on the research side: what MemGPT, Mem0, Zep/Graphiti, and the GraphRAG line of work actually do, where they're strong, and where their complexity is premature for this scale. The goal wasn't a literature review — it was deciding what to steal and what to leave out. By the end of this I had a list of the one idea to take from each (bi-temporal invalidation from Zep; two-tier rather than three from MemGPT; atomic extraction rather than raw-turn storage as the right primitive) and a clear sense of what the architecture would look like.

**~30 min — finalizing the architecture and design.**
Committing the memory model to paper: the two-tier split, the async write path as the core latency decision, the validity-window schema, the category taxonomy, and the conflict-check gating strategy. This is also where the milestone ordering got locked — making sure M3 produced a working, talking agent before any of the smarter machinery was layered on.

**~1 hr — implementation with Claude, milestones and unit tests.**
Built milestone by milestone, running the gate tests at each step before moving on. Used Claude Opus 4.8 as a sub-agent to help draft and refine the extraction prompt schema, the conflict-judge prompt, and the safety filter patterns — not to design the system, but to speed up the parts where prompt quality is the variable. Minor refinements to the build plan also happened here as the implementation surfaced small gaps (the turn-buffer bridging the async write gap being the clearest example).

**~1 hr — deliberate testing and threshold tuning.**
This took longer than expected and I think it was the right place to spend the time. The conflict-check similarity threshold (settled at 0.85), the category TTL weights, and the top-k retrieval count all needed real testing with realistic inputs, not just the unit test fixtures. Ran the demo script repeatedly with varied phrasings, checked whether the conflict judge fired correctly on near-misses, and used ChatGPT as an external sub-agent reviewer to pressure-test the design decisions and flag anything the system got wrong. Found and fixed two threshold issues that the unit tests weren't catching.

**~30 min — README and final checks.**
Since I documented decisions as I made them throughout the build, this was assembly rather than writing — pulling the design-decision log, the future-work section, and the build log into the final shape, then a clean run of `pytest` and `demo.py` to confirm everything was still green.

---
