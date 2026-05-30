# Setup Guide — Neander Memory Agent

Step-by-step instructions for getting Neander running locally from scratch.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.14 tested |
| pip | any | comes with Python |
| An LLM API key | — | OpenAI (default) or Google Gemini |

---

## Step 1 — Clone / enter the project directory

```bash
cd /path/to/neander
```

The repo root contains `neander/` (the Python package), `tests/`, `demo.py`, and `benchmark.py`.

---

## Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows
```

You should see `(.venv)` in your shell prompt.

---

## Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Key packages installed:

| Package | Purpose |
|---|---|
| `openai` | Default LLM provider SDK |
| `google-generativeai` | Gemini provider SDK (optional) |
| `sentence-transformers` | Local semantic embeddings (all-MiniLM-L6-v2, 384-dim) |
| `numpy` | Brute-force cosine similarity |
| `python-dotenv` | Load `.env` file |
| `pytest` | Test runner |

The first run will download the embedding model (~90 MB) and cache it locally.
Subsequent runs load from cache.

---

## Step 4 — Configure your API key

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```bash
# Default: OpenAI
OPENAI_API_KEY=sk-...

# --- OR switch to Gemini ---
# LLM_PROVIDER=gemini
# GEMINI_API_KEY=...
# CHAT_MODEL=gemini-1.5-flash
# EXTRACT_MODEL=gemini-1.5-flash
```

All other variables have safe defaults. The full list is documented in `.env.example`.

### HuggingFace token (optional — suppresses a rate-limit notice)

If you have an HF token, add it to suppress the startup warning:

```bash
HF_TOKEN=hf_...
HUGGINGFACE_HUB_VERBOSITY=error
```

---

## Step 5 — Verify the setup (no API key needed)

```bash
pytest
```

Expected output: **55 passed** in ~10 seconds.
All tests are hermetic — they use a deterministic offline embedder and a mocked LLM.

---

## Step 6 — Run the agent

```bash
python -m neander.cli
```

Example session:

```
Neander  —  /memory list | forget <desc> | delete <id>  |  /quit to exit
Provider: openai   Model: gpt-4o
Active memories: 0

You: My name is Alex and I'm a backend engineer at Stripe
Agent: Hi Alex! What's keeping you busy at Stripe these days?

You: I always use Python and prefer spaces over tabs
Agent: Consistent with PEP 8 — good call.

You: /memory list
Active memories (3):
  [a1b2c3d4] (fact)        The user's name is Alex.
  [e5f6a7b8] (fact)        The user is a backend engineer at Stripe.
  [c9d0e1f2] (preference)  The user always uses Python and prefers spaces over tabs.

You: /quit
[saving memories...] done.
```

Next time you start the agent, it will remember Alex, Stripe, and the Python preference automatically.

---

## Step 7 — Run the cross-session demo

Demonstrates cross-session recall end-to-end (requires an API key):

```bash
python demo.py
```

The script runs two back-to-back "sessions" (process restarts between them) and prints
the full transcript. Session 2's agent recalls what Session 1 stated.

---

## Step 8 — Run the latency benchmark

Proves that p50 first-token latency at 1,000 memories is within 200ms of latency at 0 memories.
Uses a mocked LLM (no API key needed):

```bash
python benchmark.py
```

Expected output (values vary by machine):

```
  p50 first-token (0 memories):       0.01 ms
  p50 first-token (1000 memories): 5.87 ms
  delta:                              5.86 ms

  PASS — delta 5.9ms is within the 200ms budget.
```

---

## CLI commands reference

| Command | Effect |
|---|---|
| `/memory list` | Show all currently active memories |
| `/memory forget <description>` | Semantically find and invalidate matching memories |
| `/memory delete <id>` | Hard-delete a memory by its id prefix |
| `/quit` or `/exit` | Exit (waits for background extraction to finish) |

---

## Switching LLM providers

Change one line in `.env` — no code changes required:

```bash
# Use OpenAI (default)
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
CHAT_MODEL=gpt-4o
EXTRACT_MODEL=gpt-4o-mini

# Use Google Gemini
LLM_PROVIDER=gemini
GEMINI_API_KEY=...
CHAT_MODEL=gemini-1.5-flash
EXTRACT_MODEL=gemini-1.5-flash
```

---

## Project structure (quick reference)

```
neander/
  neander/
    config.py           all env/settings
    llm.py              LLM provider adapter (OpenAI + Gemini)
    cli.py              REPL entrypoint
    storage/
      models.py         Memory dataclass
      store.py          SQLite CRUD
    memory/
      embeddings.py     embedder (sentence-transformers / offline hash)
      retrieval.py      cosine ranking + TTL/PII filter
    pipeline/
      safety.py         credential drop + PII tag
      extractor.py      atomic fact extraction
      worker.py         async background write queue
    agent/
      agent.py          read path + prompt assembly
      commands.py       /memory commands
  tests/                55 hermetic tests (M0–M7 gates)
  demo.py               cross-session demo
  benchmark.py          latency proof
```

See `README.md` for design decisions, `CLAUDE.md` for contributor invariants.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `RuntimeError: No OpenAI API key found` | Add `OPENAI_API_KEY` to `.env` |
| `RuntimeError: No Gemini API key found` | Add `GEMINI_API_KEY` and `LLM_PROVIDER=gemini` |
| `ModuleNotFoundError: sentence_transformers` | `pip install -r requirements.txt` |
| Slow first startup | Embedding model downloading (~90MB) — only once |
| `/memory list` shows old data | Expected during a session — run `/memory list` after a brief pause |
| Tests fail after restructure | `pytest --tb=short` to see which import broke |
