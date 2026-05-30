# Neander Memory Agent — Build Runbook

A milestone-by-milestone plan for a 4-hour build. Every milestone ends in a **gate**:
a set of tests that must be green before the next milestone starts. If a gate is red,
we fix it before moving on — that is how the structure stays intact end to end.

---

## How to use this document

- Work milestones **in order**. Each builds on the last.
- Do not start milestone N+1 until milestone N's gate is fully green.
- Keep a talking, recalling agent by the end of **M3** (the "working agent" checkpoint).
  Everything after M3 is making it smart, not making it work.
- Documentation is **continuous**, not a final step (see "Documentation discipline").
- If we fall behind, cut from the bottom (M6 first). Never cut M1, M3, M5, or M7.

---

## Locked technical decisions (do not re-litigate mid-build)

| Concern | Decision | Why |
|---|---|---|
| Language | Python 3.11+ | Spec-preferred; fastest for us |
| LLM | One provider, direct SDK (examples assume OpenAI; Anthropic is a 1-line swap) | Spec forbids frameworks, not SDKs |
| Chat model | Any capable model | On the critical path |
| Extraction model | Cheapest available model | Off the critical path; structured output |
| Embeddings | Local `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) | No network on read path → low, stable latency; free |
| Vector search | Brute-force cosine over numpy | <1ms at 1k vectors; HNSW is premature here |
| Persistence | SQLite (single file) | Survives restart with zero infra |
| Stale memory | Validity window (`valid_until IS NULL` = active), never hard-delete | Recover from wrong/stale facts + keep audit trail |
| Write path | Async background worker | Keeps first-token latency flat as memory grows |

These are also the first entries in our design-decision log.

---

## Repo layout (created in M0)

```
neander/
  neander/
    __init__.py
    config.py        # env + settings (M0)
    models.py        # Memory dataclass (M1)
    store.py         # SQLite CRUD + validity windows (M1)
    embeddings.py    # local embedder wrapper (M2)
    retrieval.py     # cosine search + ranking (M2)
    llm.py           # direct API: chat stream + json extract (M3)
    agent.py         # read path + prompt assembly + turn buffer (M3)
    safety.py        # PII/credential filter (M4)
    extractor.py     # atomic fact extraction (M4)
    worker.py        # async write path (M4/M5)
    commands.py      # /memory list|forget|delete (M6)
    cli.py           # REPL entrypoint (M3)
  tests/
    test_setup.py        # M0
    test_store.py        # M1
    test_retrieval.py    # M2
    test_agent_read.py   # M3
    test_extractor.py    # M4
    test_safety.py       # M4
    test_conflict.py     # M5
    test_commands.py     # M6
    test_latency.py      # M7
  demo.py            # cross-session demo (M7)
  benchmark.py       # latency proof (M7)
  requirements.txt
  README.md          # grows every milestone (see discipline below)
  .env.example
```

---

## Documentation discipline (run this on every milestone)

1. **Module docstring.** Every file starts with a docstring: what it does, and the one
   design choice that shaped it (e.g. "brute-force search because N≤1k").
2. **Function docstrings.** Public functions document params, returns, and any
   non-obvious decision.
3. **Design-decision log.** Append to a `## Design decisions` section in the README the
   moment a decision is made, in this shape:
   > **Decision:** _what._ **Why:** _reasoning._ **Tradeoff:** _what we gave up._ **Would revisit at:** _scale/condition._
4. **Milestone close-out.** When a gate goes green, write one line in `## Build log` in
   the README: what got built, what was tested, time spent. This makes the final
   "time spent" section free.

The README is therefore ~80% written by the time we finish coding.

---

## Milestone M0 — Skeleton & environment (≈15 min)

**Objective:** de-risk dependencies and config *before* any logic. Most time-loss in a
4-hour build is environment friction; we kill it first.

**Build:** repo layout, `requirements.txt`, `.env.example`, `config.py` (loads API key
and model names from env), `__init__.py`. Install deps. Pre-download the embedding model.

**Document:** README skeleton with Setup section; first 9 locked decisions logged.

**Gate — `tests/test_setup.py`:**
- imports the package without error
- `config` loads and exposes the model names
- the embedder loads and returns a vector of the expected dimension (384)

**Do not proceed until:** `pytest tests/test_setup.py` is green.

---

## Milestone M1 — Data model & storage (≈40 min)

**Objective:** a persistent store that can add, read, list active, and invalidate
memories — and survives a restart.

**Build:**
- `models.py`: `Memory` dataclass — `id, content, category, valid_from, valid_until,
  created_at, importance, embedding`.
- `store.py`: SQLite schema + `init()`, `add()`, `get()`, `list_active()`,
  `invalidate(id)`, `delete(id)`. `list_active()` filters `valid_until IS NULL`.

**Document:** docstrings on store methods; log the "validity window, never hard-delete"
decision.

**Gate — `tests/test_store.py`:**
- add → get roundtrip returns identical content
- `list_active` excludes an invalidated memory
- **persistence:** add, close connection, open a *new* connection on the same file,
  assert the memory is still there (this is the cross-session guarantee in miniature)
- `invalidate` sets `valid_until` and the row stays queryable for history

**Do not proceed until:** all of `test_store.py` is green.

---

## Milestone M2 — Embeddings & relevance retrieval (≈30 min)

**Objective:** turn a query into the top-k relevant *active* memories.

**Build:**
- `embeddings.py`: cached model load; `embed(text) -> np.ndarray`.
- `retrieval.py`: embed query → cosine over active memories' embeddings → top-k, with a
  light recency/category boost. Never returns invalidated memories.

**Document:** log "local embeddings" and "brute-force cosine" decisions with the
revisit-at-scale note.

**Gate — `tests/test_retrieval.py`:**
- store 3 semantically distinct facts; query close to one; assert it ranks #1
- invalidate that fact; assert it is no longer returned
- (optional) store 1,000 random facts; assert a single retrieval call completes in <10ms

**Do not proceed until:** `test_retrieval.py` is green.

---

## Milestone M3 — LLM client + read path (working agent) (≈45 min)

**Objective:** a real, talking agent that visibly uses a stored memory. **This is the
checkpoint — by here we have a working conversational agent.**

**Build:**
- `llm.py`: direct SDK wrapper — `chat_stream(messages)` and `extract_json(prompt)`.
- `agent.py`: read path — retrieve relevant memories, assemble system prompt
  (instructions + memories grouped by category + last ~6 turns buffer), stream response.
- `cli.py`: REPL loop.

**Document:** log "two-tier (turn buffer + long-term store), not three" and "recent-turns
buffer handles short-term recall for free."

**Gate — `tests/test_agent_read.py`** (LLM is **mocked** here for determinism + zero cost):
- seed a memory ("user prefers spaces"); send a query; assert the assembled prompt the
  agent passes to the LLM **contains that memory**
- assert the recent-turns buffer is included and capped at the configured size
- manual check: run `cli.py`, state a fact, confirm it's used in the same session

**Do not proceed until:** `test_agent_read.py` is green and the CLI runs.

---

## Milestone M4 — Async write path: extraction + safety (≈45 min)

**Objective:** memories get captured automatically, PII is filtered, and writing never
blocks the response.

**Build:**
- `extractor.py`: `extract_json` call with a strict schema → list of `{content, category}`
  atomic facts (preference / fact / decision / context). Stores raw turns nowhere as
  memories — only extracted facts.
- `safety.py`: regex filter for credentials/PII (API keys, SSNs, card numbers, secrets).
  Drops credentials; tags email/phone as `pii=true` and never injects them into prompts.
- `worker.py`: background thread + queue. After each response, enqueue the exchange;
  worker extracts → filters → inserts.

**Document:** log "atomic extraction, not raw-turn storage", "async write path keeps
latency flat", and "PII filter runs synchronously before any write."

**Gate:**
- `tests/test_extractor.py`: mock LLM returns known JSON; assert facts parsed, typed, and
  stored
- `tests/test_safety.py`: feed a message containing a fake API key and an SSN; assert
  **neither is stored**
- async check: assert the response returns before the worker finishes (or that the queue
  drains and the fact appears after a short wait)

**Do not proceed until:** both test files are green.

---

## Milestone M5 — Conflict resolution + categories/TTL (≈30 min)

**Objective:** reconcile contradictory memories and forget the right things — the core
judgment the eval rewards.

**Build:** in `worker.py`, after extracting a fact, find high-similarity (> ~0.85)
*same-category* active facts; if a cheap LLM judge says "contradicts/updates", set the
old fact's `valid_until = now` and insert the new one. Apply category TTL at retrieval
scoring (preference long, decision medium, fact medium, context short).

**Document:** log "single validity window (not dual-timestamp bi-temporal)", "conflict
judge only fires on high similarity → near-zero cost", and "forgetting = TTL + recency,
not a decay model."

**Gate — `tests/test_conflict.py`:**
- store "prefers tabs", then "prefers spaces"; assert old is invalidated, new is active
- assert a *historical* query can still surface the old (tabs) memory — history preserved
- a `context` fact older than its TTL is excluded/deprioritized in retrieval

**Do not proceed until:** `test_conflict.py` is green.

---

## Milestone M6 — Memory editing commands (≈20 min, first to cut)

**Objective:** user can inspect and forget memories.

**Build:** `commands.py` + CLI hooks — `/memory list`, `/memory forget "X"` (semantic
search + invalidate matches), `/memory delete <id>`.

**Document:** log "editing is near-free because storage is plain SQLite."

**Gate — `tests/test_commands.py`:**
- `/memory forget "tabs"` invalidates the matching memory
- `/memory list` shows active memories only

---

## Milestone M7 — Cross-session demo + latency proof (≈30 min, never cut)

**Objective:** the two headline artifacts the evaluators will run.

**Build:**
- `demo.py`: Session 1 — user states two preferences and a decision; process exits.
  Session 2 — a fresh agent instance recalls them unprompted and acts on them. Prints the
  full transcript.
- `benchmark.py`: seed 1,000 facts; measure p50 first-token latency at turn 1 vs turn
  1,000 over enough samples for a stable median; print both and the delta.

**Document:** paste the real latency numbers into the README's latency section.

**Gate — `tests/test_latency.py`:**
- assert `p50(turn_1000) - p50(turn_1) < 200ms` (may be marked slow/optional)
- `demo.py` runs end to end and the transcript shows recall across the restart

**Do not proceed until:** the demo runs and the latency assertion passes.

---

## Milestone M8 — Finalize README (≈15 min)

**Objective:** since we documented continuously, this is assembly, not writing.

**Build/Document:** ensure the README has: setup/run instructions; the design-decision
log; the concept-steal framing (one idea each from MemGPT/Mem0/Zep, what we dropped and
why); latency numbers; "what I'd build next" (HNSW at ~50k, graph links for multi-hop,
dual-timestamp bi-temporal, importance-decay model, stronger conflict judge); time spent
(from the build log). Write the submission note: most-proud + what you'd change.

---

## Time budget & cut order

| Milestone | Est. | Running |
|---|---|---|
| M0 | 15m | 0:15 |
| M1 | 40m | 0:55 |
| M2 | 30m | 1:25 |
| M3 | 45m | 2:10 |
| M4 | 45m | 2:55 |
| M5 | 30m | 3:25 |
| M6 | 20m | 3:45 |
| M7 | 30m | 4:15 |
| M8 | 15m | 4:30 |

That's ~4.5h on paper. Real buffer comes from cutting in this order if needed:
**M6 → trim M5 to conflict-only (drop TTL) → simplify M4 categories to one type.**
Protected at all costs: M1 (persistence), M3 (working agent), M5 conflict logic, M7
(demo + latency). Spec is explicit: if latency regresses, fixing it beats any stretch goal.

---

## Standing risks to watch

- **Embedding model download** stalls the start → pre-pull in M0.
- **First-token latency variance** from the LLM API → always report p50, take enough
  samples; the growing component (cosine scan) is sub-ms, so the delta stays tiny.
- **Async write race in tests** → tests wait for the queue to drain, not a fixed sleep.
- **Scope creep toward the graph/link-walk ideas** → explicitly out of scope; note as
  future work instead of building.
