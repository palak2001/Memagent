# CLAUDE.md — How to work in this repo

Companion to `BUILD_PLAN.md`. The plan says *what to build and when*. This says *what must
stay true while you build it*. Read both first. If a change breaks a rule here, stop and ask.

---

## What we're building
- A chat agent with memory that survives across sessions.
- Calls a real LLM directly. Memory layer is hand-built.
- Two tiers: recent-turns buffer (always in the prompt) + long-term SQLite store of atomic facts.
- The whole point: store only what matters, forget the noise, fix wrong/stale memories — and stay fast as memory grows.

## Hard rules (from the spec — don't cross)
- No frameworks: no LangChain, LlamaIndex, LangGraph, CrewAI, AutoGen, Semantic Kernel, Haystack.
- No memory libs: no mem0, Letta/MemGPT, Zep, MotorHead. We steal their *ideas*, never their code.
- OK to use: LLM SDK, HTTP clients, test framework, embedding models, DB drivers, numpy.
- Litmus test: *does this library make a memory decision for us?* If yes → out.

## Invariants (breaking one is a bug, even if tests pass)
- **Writes are async.** Extraction runs in the background, after the response is sent. Never block a reply to write memory.
- **Read path stays cheap.** embed query → cosine over active memories → rank → build prompt → stream. No LLM call on the read path.
- **Never hard-delete.** Retire a memory by setting `valid_until = now`. Active set = `valid_until IS NULL`. History stays queryable.
- **Store atomic facts, not raw turns.** Raw transcript lives in the episode log only — never embedded, never injected as "memory."
- **PII filter runs before every write.** Drop credentials, tag emails/phones, keep them out of prompts. It's a gate, not an add-on.
- **Embeddings go through the pluggable interface.** Real model in prod, offline backend in tests. Never hardcode a network call on the read path.

## Locked decisions (don't quietly change — raise it if you disagree)
- Python 3.11+ (currently running 3.14), SQLite single file.
- Local sentence-transformers embeddings (384-dim, all-MiniLM-L6-v2).
- Brute-force cosine over numpy — no HNSW / vector DB at this scale.
- Two tiers, not three.
- Single validity window (not full event-time vs ingestion-time).
- Cheap model for extraction (`gpt-4o-mini` default), capable model for chat (`gpt-4o` default).
- LLM provider is adapter-based (OpenAI / Gemini switchable via `LLM_PROVIDER` env var).
- Conflict judge fires on same-category facts with cosine ≥ 0.50. Top-5 candidates checked max.
- Dedupe: skip if most-similar same-category fact scores ≥ 0.92 AND judge says "compatible".
- `/memory forget` threshold: 0.35 cosine minimum to prevent collateral invalidation.

## Out of scope (note as future work, don't build)
- Knowledge / entity-relationship graph, multi-hop traversal.
- Zettelkasten link-walking, spreading activation.
- Episode-to-episode correlation links.
- Summarization/consolidation of old memories.
- LLM router on the read path.
- These belong in the README's "next steps" — add to it, don't implement it.

## Repo layout

```
neander/                    ← repo root
  neander/                  ← Python package
    __init__.py
    config.py               ← all env/settings (no neander deps)
    llm.py                  ← LLM provider adapter (OpenAI + Gemini); shared by pipeline + agent
    cli.py                  ← REPL entrypoint  python -m neander.cli
    storage/                ← persistence layer
      models.py             ← Memory dataclass + validity-window helpers
      store.py              ← SQLite CRUD, thread-safe, BLOB embeddings
    memory/                 ← read path
      embeddings.py         ← pluggable embedder (sentence-transformers / hash offline fallback)
      retrieval.py          ← cosine ranking + recency/importance blend + TTL/PII filter
    pipeline/               ← async write path
      safety.py             ← credential drop + PII tag (runs before every write)
      extractor.py          ← atomic fact extraction via cheap LLM + strict JSON schema
      worker.py             ← background thread queue; dedup + conflict resolution
    agent/                  ← conversation layer
      agent.py              ← read path: retrieve → assemble prompt → stream response
      commands.py           ← /memory list | forget | delete
  tests/
    test_setup.py           ← M0: imports, config, embedder dim
    test_store.py           ← M1: CRUD, active filter, persistence across reopen
    test_retrieval.py       ← M2: relevance ranking, TTL, PII exclusion, scale timing
    test_agent_read.py      ← M3: prompt assembly, memory injection, turn buffer
    test_extractor.py       ← M4: extraction parsing (LLM mocked)
    test_safety.py          ← M4: credential drop + PII tag
    test_conflict.py        ← M5: contradiction invalidation + TTL expiry
    test_commands.py        ← M6: /memory list/forget/delete
    test_latency.py         ← M7: p50 delta < 200ms at 1,000 memories
  demo.py                   ← cross-session memory demo (requires API key)
  benchmark.py              ← latency proof (mocked LLM, hash embedder)
  requirements.txt
  pytest.ini
  .env.example
  README.md
  SETUP.md
```

**Dependency layers (each layer may only import from layers below it):**
```
cli.py  ──►  agent/  ──►  memory/  ──►  storage/
              │             │
              ▼             ▼
           pipeline/   config.py
              │
              ▼
           llm.py  (shared — also used by agent/)
```

- Keep modules small and single-purpose. Don't add new top-level modules without a reason.
- `config.py` and `llm.py` stay at the package root — they are shared by all layers.

## Code style
- Type hints on public functions. Dataclasses for records, not dicts.
- Every module: a docstring with what it does + the one design choice behind it.
- No premature abstraction. The pluggable embedder is the only seam we keep upfront.
- All SQL goes through `store.py` — no raw SQLite scattered around.
- Secrets from env via `config.py`. `.env` gitignored, `.env.example` documents the vars.
- Readable over clever. This gets read more than it gets run.

## Testing (non-negotiable)
- Work the milestone gates in order. Don't start the next one until the current is green.
- Write the test in the same change as the code. The gate defines "done."
- Tests are hermetic: no network, no key. Mock the LLM, use the offline embedder.
- Test behavior, not internals (e.g. "invalidated memory isn't returned", not the call order).
- Persistence test: actually close and reopen the DB file. No shortcuts.
- Async tests wait for the queue to drain — never a fixed `sleep`.

## Docs (continuous, not a final step)
- Made a decision? Append to the README log: **Decision — Why — Tradeoff — Revisit at**.
- Gate went green? One line in the README build log: built what, tested what, time spent.
- Deferring something? It goes in "next steps" with a reason — not a rotting TODO.

## Commands
```bash
# setup (see SETUP.md for the full walkthrough)
cd neander
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY (or GEMINI_API_KEY + LLM_PROVIDER=gemini)

# run
python -m neander.cli          # interactive REPL
python demo.py                 # cross-session recall demo  (requires API key)
python benchmark.py            # latency proof              (mocked, no key needed)

# test (all hermetic — no network, no API key)
pytest                         # full suite (55 tests)
pytest tests/test_store.py     # single gate
pytest -m slow                 # include latency test
pytest -m "not slow"           # skip latency test
```

## If you're short on time
- Cut in this order: editing commands → trim conflict to contradiction-only (drop TTL) → collapse to one category.
- Never cut: persistence, the working-agent checkpoint (M3), conflict/invalidation, demo + latency benchmark.
- If latency regresses, fixing it beats any stretch goal. Fast-but-slightly-forgetful is usable; thorough-but-slow isn't.