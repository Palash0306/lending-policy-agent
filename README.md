# Lending Policy Data Assurance Agent

Self-hosted, chat-based agent that reads bank lending policy documents and
extracts boundary conditions (min/max numeric fields) and date cutoff
conditions, with grounding verification and source citations. Runs entirely
on local infrastructure via **Ollama** — no policy text leaves your machine.

This repo is being built phase by phase (see the roadmap in the project
brief). This drop implements:
- **Phase 1** — Ollama setup + model pull, confirm local inference works
- **Phase 2** — Lending policy document ingestion (PDF/Word parsing, chunking)
- **Phase 3** — Embedding + Chroma indexing
- **Phase 4** — Field synonym expansion (plus evidence retrieval across phrasings)
- **Phase 5** — Boundary + date-cutoff extraction with a structured schema and classification tag
- **Phase 6** — Grounding verification (plain-code check of every quote, number and date) and a conflict flag for human review

---

## What's here

```
lending-policy-agent/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
├── src/
│   └── lending_agent/
│       ├── __init__.py
│       ├── config.py                # env-driven settings: Ollama, embeddings, Chroma
│       ├── llm/
│       │   ├── __init__.py
│       │   └── ollama_client.py     # wrapper: connectivity, model pull, generate/chat, healthcheck
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── models.py            # PageText / SourceDocument / Chunk data classes
│       │   ├── loaders.py           # load_pdf, load_docx, load_document (dispatches on extension)
│       │   ├── chunker.py           # chunk_text: sentence-aware, overlapping chunks
│       │   └── pipeline.py          # ingest_document(s): file(s)/folder -> list[Chunk]
│       ├── timing.py                # RunTimer: start/end times, per-step durations, still-working heartbeat
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── field_expansion.py   # expand_field_terms: field name/question -> phrasings to search for
│       │   ├── retrieval.py         # retrieve_evidence: search per phrasing, merge into ranked chunks
│       │   ├── extraction.py        # extract_conditions: paragraphs -> strictly checked, tagged findings
│       │   ├── focus.py             # focus_text: trim paragraphs to the sentences that hold a number or date
│       │   ├── text_values.py       # numbers_in_text / dates_in_text, shared by focus and grounding
│       │   ├── grounding.py         # check_finding: re-check quote, number/date and wording against the source text
│       │   ├── assurance.py         # verify_extraction: grounding + review flags + conflict detection
│       │   └── llm_json.py          # shared helper: pull a JSON object out of an LLM reply
│       └── indexing/
│           ├── __init__.py
│           ├── embeddings.py        # EmbeddingModel: local sentence-transformers wrapper
│           ├── vector_store.py      # VectorStore: Chroma add/search/reset
│           └── indexer.py           # index_documents: ingest + embed + store in one call
├── scripts/
│   ├── install_ollama.py            # CLI: detects your OS, offers to install Ollama itself
│   ├── verify_ollama.py             # CLI: confirms Phase 1 works end to end
│   ├── index_documents.py           # CLI: chunk + embed + store into Chroma
│   ├── search_index.py              # CLI: semantic search against the indexed collection
│   ├── expand_field.py              # CLI: see how a field gets expanded (and what it finds)
│   └── extract_field.py             # CLI: expand + retrieve + extract + verify, the full flow so far
└── tests/
    ├── conftest.py                  # sample .pdf/.docx files + a deterministic fake embedder
    ├── test_ollama_client.py        # Phase 1: mocked, no live server needed
    ├── test_chunker.py              # Phase 2: chunking logic
    ├── test_loaders.py              # Phase 2: PDF/DOCX text + table extraction
    ├── test_pipeline.py             # Phase 2: end-to-end file(s) -> chunks
    ├── test_embeddings.py           # Phase 3: embedding wrapper, offline (fake sentence-transformers)
    ├── test_vector_store.py         # Phase 3: real Chroma + fake embedder
    ├── test_indexer.py              # Phase 3: end-to-end file(s) -> indexed & searchable
    ├── test_field_expansion.py      # Phase 4: expansion with a scripted fake LLM
    ├── test_retrieval.py            # Phase 4: merged multi-phrasing retrieval
    ├── test_extraction.py           # Phase 5: schema, tags, honest status (scripted fake LLM)
    ├── test_timing.py               # run timer: durations, heartbeat
    ├── test_focus.py                # focus trimming
    ├── test_text_values.py          # number words, day-word dates, month-level dates
    ├── test_grounding.py            # Phase 6: number/date/quote/direction checks (plain text, no LLM)
    └── test_assurance.py            # Phase 6: verified/needs-review flags and conflict detection
```

`OllamaClient` is deliberately the *only* thing later phases (retrieval,
extraction, grounding, LangGraph, FastAPI) will call for local inference —
centralizing connection handling, auto-pull, and error types here now means
Phase 5's extraction node and Phase 8's chat endpoint don't re-implement any
of this.

## Setup

1. **Install Ollama** (once, on the machine/server that will run inference).
   Ollama is a separate application, not a Python package — it's the local
   server that actually loads and runs the AI model. `pip install ollama`
   (step 4 below) only installs a small Python client that *talks to* that
   server; it doesn't install the server itself.

   Easiest path — run the helper script, which detects your OS and asks for
   confirmation before doing anything (it never installs silently):
   ```bash
   python3 scripts/install_ollama.py
   ```
   Or do it by hand:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh   # Linux
   # or: brew install ollama                        # macOS
   # Windows: download & run the installer from https://ollama.com/download/windows
   ```

2. **Start the Ollama server** (if it isn't already running as a service):
   ```bash
   ollama serve
   ```

3. **Pull a model.** Any Ollama-compatible open-weight chat model works; the
   default here is `qwen2.5:7b` (good instruction-following for extraction
   tasks, runs on modest hardware). You can also let `verify_ollama.py`
   auto-pull it for you (step 6).
   ```bash
   ollama pull qwen2.5:7b
   ```

4. **Python environment:**
   ```bash
   cd lending-policy-agent
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

5. **Configure:**
   ```bash
   cp .env.example .env
   # edit .env if your Ollama server isn't on localhost:11434,
   # or if you want a different model
   ```

6. **Verify Phase 1 works end to end:**
   ```bash
   python scripts/verify_ollama.py
   ```
   This checks: the Ollama server is reachable → the configured model is
   pulled (auto-pulling it if missing) → a real local `generate()` call
   returns a response. Example output:
   ```
   → Ollama host:   http://localhost:11434
   → Target model:  qwen2.5:7b
   → Checking connection and model availability...

     Server reachable:   yes
     Model available:    yes
     Installed models:   qwen2.5:7b
     Sample inference:   'OK'

   ✓ Phase 1 check passed — local inference is working end to end.
   ```
   Non-zero exit code and a `✗ FAILED` line (with a hint) if anything's wrong
   — useful as a pre-flight check in CI or before demoing.

   Useful flags:
   ```bash
   python scripts/verify_ollama.py --model llama3.1:8b   # try a different model
   python scripts/verify_ollama.py --host http://gpu-box:11434
   python scripts/verify_ollama.py --no-pull              # fail instead of auto-pulling
   ```

7. **Run the unit tests** (fully mocked/self-contained — no live Ollama
   server and no external sample files required, safe for CI):
   ```bash
   pytest -v
   ```

## Phase 2: reading and chunking policy documents

Phase 2 is library code, not a script you run on its own: `scripts/index_documents.py`
(Phase 3) calls it for you. It accepts one PDF/Word file, several at once, or a
whole folder:
```bash
python scripts/index_documents.py path/to/policy.pdf
python scripts/index_documents.py policy1.pdf policy2.docx policy3.pdf   # several files in one call
python scripts/index_documents.py path/to/policy_folder/                 # every .pdf/.docx in a folder
```
Every chunk carries a `chunk_id` (collision-proof even if two different files
share a filename), `source_file`, `page_number` (PDF only — `.docx` has no
native pagination), and `chunk_index` — this is what lets later phases cite
exactly which document and page a boundary or date-cutoff value came from,
even when several policies are loaded together and might disagree.

To check how a document is being chunked, index it and then run
`scripts/search_index.py` — each result shows the chunk text, source file and
page. You can also tune `--chunk-size` and `--chunk-overlap` on
`index_documents.py` (use `--reset` when re-indexing with new values).

Two design choices worth knowing about:
- **Chunking is sentence-aware, not a blind character cut.** `chunk_text()`
  packs whole sentences up to `chunk_size` characters, and carries the
  trailing `chunk_overlap` characters' worth of sentences into the start of
  the next chunk — so a boundary value and its qualifying clause in the
  next sentence are less likely to end up split across two unrelated
  chunks.
- **DOCX tables are extracted, not skipped.** Lending policies often put
  tiered limits (e.g. loan-to-value ratio by tier) in a table, which
  python-docx's plain paragraph iteration ignores entirely. `load_docx()`
  pulls table rows separately and appends them as `cell | cell | cell` text
  so those values aren't silently lost.

## Phase 3: embedding + Chroma indexing

This turns chunks from something you can only eyeball in a terminal (or a
JSON file) into a real, persistent, searchable local database. Two new
pieces of one-time setup:

- **Embedding model**: uses `sentence-transformers` (default model
  `all-MiniLM-L6-v2`), which runs entirely on your own machine. The very
  first time you run it, it downloads the model weights (~80MB) from
  Hugging Face once; every run after that is fully offline, matching the
  project's "policy documents never leave your network" requirement — the
  model itself is a one-time download, but no document text is ever sent
  anywhere.
- **Chroma**: a local, file-based vector database. By default it's stored
  in `./chroma_db` inside the project folder (configurable via
  `CHROMA_PERSIST_DIR` in `.env`). Nothing to install or run separately —
  it's just files on disk.

**Index a document (or several, or a folder):**
```bash
python scripts/index_documents.py policy.pdf
python scripts/index_documents.py policy1.pdf policy2.docx
python scripts/index_documents.py path/to/policy_folder/
python scripts/index_documents.py policy.pdf --reset   # wipe the collection and start over
```
Re-running this on the same file is safe — chunks are upserted by their
`chunk_id`, so you won't get duplicates if you index the same policy twice.

**Search what you've indexed:**
```bash
python scripts/search_index.py "maximum loan amount"
python scripts/search_index.py "when does the new eligibility rule start" --n 3
python scripts/search_index.py "repayment period" --source-file policy.pdf
```
This is a plain semantic search — it finds chunks with *similar meaning* to
your query, not just exact keyword matches, which is what lets a query
about `loan_amount` find a chunk that actually says "sanctioned amount" or
"mortgage amount." Each result shows the source file, page number, and a
rough similarity score, so you can confirm retrieval is finding the right
paragraphs before Phase 4/5 build the actual boundary-value extraction
logic on top of this.

## Phase 4: field synonym expansion

Your system calls a field `loan_amount`, but the policy may say "sanctioned
amount" or "mortgage amount". Searching only for the field name would miss
those paragraphs, so this step first builds a list of likely phrasings:

```bash
python scripts/expand_field.py loan_amount
python scripts/expand_field.py "What is the maximum loan tenure allowed?"
python scripts/expand_field.py loan_tenure --no-llm      # built-in word list only; Ollama not needed
python scripts/expand_field.py loan_tenure --retrieve    # also search your index with every phrasing
```

The phrasings come from two places, merged together:
1. **A built-in word list** of common lending terms (loan amount, tenure, age,
   interest rate, LTV, debt-to-income, credit score, down payment, income).
   Always available, works offline, same answer every time.
2. **The local LLM (Ollama)**, which suggests extra wording, including for
   fields the built-in list has never heard of.

If Ollama isn't running, or answers with something that isn't usable, the
agent falls back to the built-in list and tells you why (`Note:` line) instead
of crashing. The LLM's output is treated as untrusted: it must be valid JSON,
phrases must be short, and any phrase containing a number is thrown away so an
invented value can't leak into the search.

`--retrieve` (needs a built index) runs one search per phrasing and merges the
results: a chunk found by several phrasings is listed once, with a count of how
many phrasings found it.

## Phase 5: boundary and date-cutoff extraction

This is where the local LLM reads the paragraphs found in Phase 4 and pulls out
the actual limits:

```bash
python scripts/index_documents.py pub-ch-retail-lending.pdf    # once, if not already done
python scripts/extract_field.py loan_tenure
python scripts/extract_field.py "What is the maximum loan tenure allowed?"
python scripts/extract_field.py loan_amount --kinds boundary   # numeric limits only
python scripts/extract_field.py loan_tenure --kinds date       # date cutoffs only
python scripts/extract_field.py loan_tenure --json             # machine-readable
```

**Fixed output shape.** Every finding has the same fields: `kind`
(numeric_boundary or date_cutoff), `bound_type` (minimum/maximum), `value`,
`unit`, `date`, `date_precision` (day or month), `date_direction` (on_or_after, after,
on_or_before, before), `applies_to`, the exact `quote`, and the `source_file` / `page_number`. Anything
the model returns that doesn't fit is thrown away and listed under "rejected"
with the reason, never silently kept.

**Classification tag.** Each finding is tagged:
- `firm_rule`: the document itself requires or limits this ("shall", "must")
- `example`: only an illustration ("for example", "some banks")
- `guidance`: recommended or typical practice ("should", "generally")

A number tagged `example` is real text from the document but is not a rule, so
the output warns you not to use it as a hard limit.

**Three different answers.** `found`, `not_found` and `error` are kept apart.
"Ollama is switched off" is an `error`, never reported as "not found". And
`not_found` is a legitimate answer: a document may simply not state a limit.

**Source details come from us, not the model.** The paragraphs are numbered in
the prompt and the model names the number it used; the file and page are then
looked up from our own records, so the model cannot invent a citation.

**Not trusted on its own.** Phase 5 only records what the model claimed; the
check against the source text is Phase 6 (next section), which now runs
automatically in `extract_field.py`.

**Context window.** Ollama only reads a small window of text by default and silently
cuts off whatever does not fit. The tool asks for 4096 tokens (`--num-ctx` to change
it) and raises the window automatically if a prompt would ever be bigger. A bigger
window also costs more memory, so it never asks for more than needed.

**Focus (less for the model to read).** Reading is the slow step, and its cost grows
with the amount of text sent. A boundary value or date can only sit in a sentence
that has a number or a date, so before Ollama sees anything, plain code drops whole
paragraphs with no number or date and trims the rest to their number-bearing
sentences (plus the one sentence before each, for context). It tests for *numbers*,
never for words like "maximum", so implicit wording is still found. The output says
how many paragraphs were sent. Limits: a value with no digit or number word ("a year",
"half of income") is not seen, and a stray number (a footnote, "Version 2.0") keeps a
paragraph in. `--no-focus` sends whole paragraphs. The quote is still checked against
the full original paragraph in Phase 6, so trimming cannot make a wrong value pass.

## Phase 6: grounding verification and conflict flags

Phase 5 records what the model *claimed*. Phase 6 re-checks every claim with
plain code (no AI, so nothing to hallucinate) against the paragraph the finding
cites. `extract_field.py` now runs it automatically as step 4.

**Hard checks. All three must pass for a finding to be `grounded`:**
1. the quote really appears in the cited paragraph (ignoring case, line breaks, curly quotes and hyphens left by PDF extraction)
2. the number or date really appears in the quote
3. the number or date really appears in the paragraph

Numbers are recognised in digits (`500,000`, `6.5%`, `1.5 million`) and in words
(`thirty`, `seventy two`, `twenty-five`, `five hundred thousand`). Dates are recognised
however they are written (`January 1, 2026`, `1 Jan 2026`, `2026-01-01`, `01.02.2026`)
and with the day as a word (`the first of January 2026`, `January first, 2026`,
`the twenty-first of March 2026`).

**Number words in the model's answer.** If the model writes `"seventy two"` or
`"thirty (30)"` instead of `72` / `30`, it is converted. The converted number is still
checked against the quote like any other. Words that are not a number (`"a dozen"`,
`"half"`, `"thirty apples"`) are rejected, and listed with the reason.

**Month-level dates.** A rule dated only "January 2026" has no day. The model records it
as `2026-01-01` with `date_precision: "month"` (never an invented day). It is checked by
confirming that month and year really appear in the quote, and it is **always flagged for
review** with "the day is not given, so it is treated as the 1st: confirm the exact day",
because day-before / day-of / day-after test dates depend on the real day. A model that
claims a *full* date the text does not contain (`2026-01-15`) is still caught as NOT grounded.

**Soft checks.** They never decide `grounded`, but they ask a person to look:
- the quote has wording that fits the claimed direction (minimum/maximum, or on/after/before). Swapping min and max is a classic model mistake.
- a finding tagged `firm_rule` must not contain wording like "for example" or "some banks".

These are simple word patterns. They can miss unusual phrasing, so treat them as a
second pair of eyes rather than proof.

**Nothing is dropped.** A finding that fails is kept and flagged with the reason,
so the reviewer sees exactly what the model claimed and why it was doubted.

**Conflict flag (human in the loop).** Two grounded findings conflict when they
describe the same limit (same kind, minimum/maximum or date direction, unit,
classification and scope) but give different values, in two documents or in
two places in one document. Different scopes (such as "LTV below 90 percent"
versus "above 90 percent") are tiers, not conflicts. Both values are listed
with their file and page. Scope wording comes from the model and can vary, so a
person should still glance at several values listed for the same field.

**Reading the result:**
- `verified: true` means every finding passed the three hard checks.
- `needs_review: true` means a person should look: a check failed, or values conflict.
- `not_found` is neither verified nor flagged. Absence cannot be proved by code, which is why the tool says so plainly instead of guessing.

**Slow or freezing laptop?** Extraction can take minutes, and a silent terminal looks
the same as a frozen one. So `extract_field.py` now shows a timer:
- the start time, then a line as each of the 4 steps starts and finishes (with the clock time)
- a "still working" line every 15 seconds during the slow Ollama steps
- a **Run time** summary at the end: start, end, total, and how long each step took
- `--json` output includes the same timings under `"timing"`
- pressing Ctrl+C stops cleanly and tells you how long it ran

If the laptop itself is struggling (fans loud, everything lagging, an editor crashing),
the cause is almost always memory and CPU: the 7B model, the embedding model and your
editor all compete for the same RAM, and CPU-only inference uses every core. Lightest fixes first:
1. **Run it from a plain terminal, not the one inside your editor,** and close the editor
   (and heavy browser tabs) while it runs. If the editor crashes, a run started inside it
   is killed too.
2. `python scripts/extract_field.py loan_tenure --fast` searches for 3 paragraphs instead of 6.
3. Run `ollama ps` while it works: the PROCESSOR column shows GPU or CPU. CPU-only is much slower.
4. Use a smaller model: `ollama pull qwen2.5:3b`, then set `OLLAMA_MODEL=qwen2.5:3b` in `.env`.
   Roughly half the memory and noticeably faster, but check its answers more carefully.
5. `--timeout 900` (or `OLLAMA_TIMEOUT` in `.env`) just waits longer. The default is 300 seconds.

**How the time scales with more documents.** Extraction time depends on how many
paragraphs are *read* (6 by default), not on how many PDFs are indexed. Indexing is
a one-time cost per document and grows with its size. Searching grows only slightly.
One catch: the best 6 paragraphs may all come from one document, so a rule in another
could be missed. Reading the best paragraphs of each document separately would catch it
but costs more time per document; that is a design choice for later phases.

## What's *not* in this drop

LangGraph orchestration, the FastAPI `/chat` endpoint with a web page, the run
report file, and final CI (Phases 7–9). `requirements.txt` reflects only what Phases 1–6 need; later phases will add `langgraph`, `fastapi`, and `uvicorn` as those
pieces land.
