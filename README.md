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
- **Phase 7** — LangGraph agent: retry with feedback, widen the search once, and route each answer to verified / needs_review / not_found / error
- **Phase 8** — FastAPI chat server (`/chat` with conversation history), multi-file upload, and a simple web page
- **Phase 9** — batch runs for many fields, one Excel workbook of test values, test coverage and CI, and the guides in `docs/`

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
│       ├── testcases.py             # Phase 9: findings -> test values (boundary maths, dates, inclusive/exclusive wording)
│       ├── export_xlsx.py           # Phase 9: the Excel workbook and csv (formula-injection safe)
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── field_expansion.py   # expand_field_terms: field name/question -> phrasings to search for
│       │   ├── retrieval.py         # retrieve_evidence: search per phrasing, merge into ranked chunks
│       │   ├── extraction.py        # extract_conditions: paragraphs -> strictly checked, tagged findings
│       │   ├── focus.py             # focus_text: trim paragraphs to the sentences that hold a number or date
│       │   ├── text_values.py       # numbers_in_text / dates_in_text, shared by focus and grounding
│       │   ├── grounding.py         # check_finding: re-check quote, number/date and wording against the source text
│       │   ├── assurance.py         # verify_extraction: grounding + review flags + conflict detection
│       │   ├── graph.py             # Phase 7: the LangGraph agent (build_graph / ask)
│       │   └── llm_json.py          # shared helper: pull a JSON object out of an LLM reply
│       ├── api/                     # Phase 8: the chat server
│       │   ├── __init__.py
│       │   ├── app.py               # FastAPI routes (thin: connects URLs to the services below)
│       │   ├── chat.py              # ChatService: sessions, follow-up rewriting, runs the agent
│       │   ├── summary.py           # agent answer -> chat JSON (headline values, review routing; no LLM)
│       │   ├── sessions.py          # SessionStore: server-side conversation history
│       │   ├── jobs.py              # JobRunner: one worker, so questions never run at the same time
│       │   ├── documents.py         # upload checks, indexing, listing, removal
│       │   ├── batches.py           # Phase 9: BatchManager (parse fields, run one by one, save each result, cancel, restart-safe)
│       │   ├── errors.py            # ApiError (becomes an HTTP status)
│       │   └── static/index.html    # the web page (one file, no internet needed)
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
│   ├── extract_field.py             # CLI: expand + retrieve + extract + verify, the full flow so far
│   ├── run_agent.py                 # CLI: the Phase 7 agent for one or many fields
│   ├── serve.py                     # CLI: start the chat server + web page (Phase 8)
│   ├── ci.sh                        # Phase 9: the one script any CI system runs (lint, tests, coverage, audit)
│   └── check_no_documents.sh        # Phase 9: fails if a .pdf/.docx is staged or tracked in git
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
    ├── test_assurance.py            # Phase 6: verified/needs-review flags and conflict detection
    ├── test_graph.py                # Phase 7: agent routes and loops (scripted model)
    ├── api_helpers.py               # Phase 8: shared fakes (scripted model, in-memory store)
    ├── test_api_summary.py          # Phase 8: headline values, review routing
    ├── test_api_sessions.py         # Phase 8: history, expiry, id checks
    ├── test_api_jobs.py             # Phase 8: one-at-a-time queue
    ├── test_api_documents.py        # Phase 8: upload safety, indexing, replace, delete
    ├── test_api_chat.py             # Phase 8: follow-ups, error handling
    ├── test_api_app.py              # Phase 8: the real HTTP routes (skipped if fastapi is not installed)
    ├── test_testcases.py            # Phase 9: boundary values, wording cues, six-point sets, dates, conflicts
    ├── test_export_xlsx.py          # Phase 9: workbook sheets, cell types, formula safety, csv
    ├── test_batches.py              # Phase 9: batch run, save-as-you-go, cancel, restart
    ├── test_api_batches.py          # Phase 9: batch and export routes over HTTP
    ├── test_golden_phrasing.py      # Phase 9: ~60 realistic sentences for number/date reading and grounding
    ├── test_error_branches.py       # Phase 9: error paths (Ollama client, summary, graph)
    ├── test_end_to_end.py           # Phase 9: PDF + Word -> index -> agent -> API -> Excel
    ├── test_live_ollama.py          # Phase 9: needs real Ollama; run with: pytest -m live
    ├── test_ui_page.py              # Phase 9: runs tests/ui/ (the web page in a simulated browser, needs node)
    └── ui/                          # Phase 9: web page scenarios (node, no packages)
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

## Phase 7: the agent (LangGraph)

`scripts/run_agent.py` runs the same four steps as `extract_field.py`, but as a graph that can
loop (`src/lending_agent/agent/graph.py`):
- **retry**: if the model's reply was not valid JSON, or a finding failed the grounding check,
  extraction runs again (at most `--max-attempts` in total) with the problems written into the prompt;
- **widen**: if nothing was found, the search is repeated once with twice as many paragraphs (`--no-widen` turns this off);
- **route**: every answer ends as `verified`, `needs_review`, `not_found` or `error`. Ollama being
  unreachable is always `error`, never "not found".

```bash
python scripts/run_agent.py loan_tenure
python scripts/run_agent.py loan_tenure loan_amount --out-dir results --resume
```

## Phase 8: chat server, upload and web page

```bash
pip install -r requirements.txt         # adds fastapi, uvicorn, python-multipart
python scripts/serve.py                 # then open http://127.0.0.1:8000
```
The page lets you upload one or more `.pdf`/`.docx` files (they are chunked, embedded and indexed
into Chroma exactly like `index_documents.py`), list or remove indexed documents, and chat.
Uploaded files are kept in `./uploads` (`UPLOAD_DIR` in `.env`).

**The API** (interactive docs at `http://127.0.0.1:8000/docs`):

| Call | What it does |
|---|---|
| `POST /chat` | Ask a question. Body: `{"message": "...", "session_id": null, "source_file": null, "kinds": "both", "fast": false}` |
| `GET /jobs/{job_id}` | Progress or result of a question or upload started with `?wait=false` |
| `GET /sessions/{id}`, `DELETE /sessions/{id}` | The conversation so far / forget it |
| `GET /documents`, `POST /documents`, `DELETE /documents/{name}` | What is indexed / upload files / remove one |
| `GET /health` | Server, Ollama (reachable? model installed?) and index status |

```bash
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
     -d '{"message": "What is the maximum loan tenure allowed?"}'
```
returns, among other fields:
```json
{"answer": "For loan_tenure: maximum 30 years, per policy.pdf, page 12.", "field": "loan_tenure",
 "route": "verified", "max_value": 30, "unit": "years", "source_quote": "...", "source_file": "policy.pdf",
 "page_number": 12, "grounded": true, "needs_review": false, "session_id": "...", "turn": 1,
 "query_used": "What is the maximum loan tenure allowed?", "findings": [...], "review_reasons": []}
```

**Reading an answer.**
- `max_value`, `min_value` and `cutoff_date` are filled only from findings that are **grounded and tagged
  `firm_rule`**, and only when they agree. Tiered limits (72 months below 90 percent LTV, 60 above) leave
  them empty; every value is in `answer` and `findings`.
- `route` is the agent's route with one extra rule: if values were found but none is a grounded firm rule
  (only examples or guidance), it becomes `needs_review`. The agent's own route is in `agent_route`.
- `not_found` is a normal answer, and `error` means no answer could be produced (Ollama down, no index...).
  Both come back as HTTP 200 with the reason in the JSON.

**Follow-up questions.** Send the returned `session_id` with your next message. History is kept on the
server. A message that names a field (`loan_tenure`, "loan amount"...) is used as written. Otherwise the
local model rewrites it using the previous question ("what about for commercial loans?" becomes "What is
the maximum loan tenure for commercial loans?"). The rewrite is untrusted: it must be one short line and
may not contain a number that is not already in the conversation; if it fails, plain code prepends the
previous field's name. `query_used` always shows what was really searched.

**Why `?wait=false` and a job queue.** One question can take minutes on a laptop, and browsers give up on a
request that stays silent that long. So the web page starts a job and checks on it every 2 seconds.
`POST /chat` and `POST /documents` also work as one simple call (they wait and return the result), which is
what scripts and curl want. There is exactly **one worker**: questions and uploads run one at a time, in
order. On a laptop where two runs at once would freeze the machine, that is deliberate.

**Conversations and the web page.**
- The *list* of your conversations is kept in the browser (localStorage); the *answers* live on the server, in
  memory, for about 8 hours (`SESSION_TTL_MINUTES`).
- A question that is still running keeps running if you reload the page, and the page picks it up again.
- "Report (.md)" builds a Markdown report in the browser from the answer you are looking at.

**Safety.**
- Uploaded file names are reduced to a harmless name (no folders). Only `.pdf`/`.docx` files whose first bytes
  really match are accepted, and there is a size limit (`MAX_UPLOAD_MB`, default 50).
- Re-uploading a file with the name of an indexed one replaces it **safely**: the new file is read and split
  first, and the old version is removed only if that worked. A broken or text-less file never removes a good one.
- Basic security headers are sent on every response (no framing, no content sniffing, no caching, a
  same-origin-only content policy).
- **There is no login.** The server listens on `127.0.0.1` only. Do not use `--host 0.0.0.0` on a network you do
  not trust.

**Known limitations** (not bugs to wait for a fix on):
- Conversations are in memory: a server restart clears them. Documents and the index stay.
- Documents indexed from the command line (`index_documents.py`) do not show on the page until the server restarts.
- A follow-up question costs one extra Ollama call (slow on a CPU). If that call fails, plain code is used instead.
- There is no login; the server listens on `127.0.0.1` by default.
- Re-uploading overwrites the saved copy of the file even if the new one is broken. The index stays safe.

## Phase 9: batches, Excel test values, tests and CI

### Run many fields and get one Excel file
On the web page open the **Batch** tab, paste your fields (one per line, or comma-separated) and press **Run batch**.
The whole batch is one job on the single worker, so it never runs at the same time as a question. Fields run one after
another through the same agent as `/chat`; one failing field is recorded as an error and the batch carries on. **Every
finished field is saved to disk at once** (`BATCH_DIR`, default `./data/batches/<batch_id>/`), so a crash or restart loses
nothing; a batch that was running when the server stopped comes back as *interrupted* and can still be exported.

| Call | What it does |
|---|---|
| `POST /batches` | `{"fields": [...], "source_file": null, "kinds": "both", "fast": false}` -> 202 `{batch_id, job_id, count}`. 1 to `MAX_BATCH_FIELDS` (default 50) fields, each at most 500 characters |
| `GET /batches`, `GET /batches/{id}` | Recent batches / progress: status (`queued`, `running`, `done`, `cancelled`, `interrupted`), done, current field, per-field result |
| `GET /batches/{id}/results` | The full answer of every finished field (the page uses it to open each field) |
| `POST /batches/{id}/cancel` | Stop **between** fields: the current one finishes, the rest are skipped |
| `DELETE /batches/{id}` | Remove a finished batch and its saved results |
| `GET /batches/{id}/export?format=xlsx\|csv&include_examples=false&step=` | One file with **all** fields |
| `GET /sessions/{id}/export`, `GET /sessions/{id}/turns/{n}/export` | The same, for a whole conversation or one answer |

Fields are split on new lines, commas and semicolons; bullets and numbering are stripped; repeats are removed ignoring case.
Time: about 4 to 5 minutes per field on a laptop CPU, so 20 fields take roughly 1.5 hours. A batch makes this "start it and
walk away"; it does not make each field faster.

### The workbook
Five sheets, `.xlsx` only (no macros): **Summary** (one row per field, counts per outcome, models and documents used),
**Test values**, **Findings** (everything, including failed and example findings), **Search log** (phrasings, paragraphs read,
attempts, timings) and **Read me**. Numbers and dates are real number and date cells. The csv export is the *Test values* sheet
as UTF-8 with a byte-order mark. File name: `lending-policy-test-values_YYYY-MM-DD_HHMM.xlsx`. Every column is explained in
`docs/USER_GUIDE.md` and inside the file.

### How test values are made (`testcases.py`)
- Only findings that **passed the grounding checks** are used. By default only `firm_rule`; with `include_examples=true`,
  example/guidance rows are added and labelled **Not a firm rule**.
- Exact arithmetic (`Decimal`, built from the stated text). The step is one unit of the last decimal place stated (6.5 gives 0.1,
  90 gives 1), or `step=`. Units are copied, never converted.
- Minimum V: V-step Reject; V by wording; V+step Accept. Maximum V: V-step Accept; V by wording; V+step Reject. A field with both a
  minimum and a maximum (same unit and scope) gets **one** six-point set.
- **Whether the limit value itself is allowed** comes from the wording around that value: "up to", "not exceed", "at least",
  "not less than" allow it; "less than", "below", "greater than", "above" do not. The part of the sentence that only says *who*
  the limit applies to is ignored, and a sentence with two limits is read one limit at a time. If the wording is missing or
  contradictory the expected result is **Confirm with policy owner**. **Nothing is ever guessed.**
- Dates use calendar arithmetic (month ends, leap days). A rule dated only to a month uses the 1st and every expected result says
  **Confirm (document gives only the month)**.
- Conflicting grounded values for one limit each get cases, flagged `CONFLICT` with the other value and source in Notes.
- Text that starts with `=`, `+`, `-`, `@`, a tab or a carriage return is stored as plain text, in xlsx and csv, so a quote from a
  document can never run as a formula.

## Operations

```bash
pytest                               # everything except the live tests
pytest --cov=src/lending_agent       # with coverage (needs pytest-cov)
pytest -m live -v                    # LIVE: real Ollama + real embedding model + the sample handbook; slow; skips with a reason if Ollama is missing
node tests/ui/run_all.js             # the web page scenarios on their own (needs node, no packages)
bash scripts/ci.sh                   # exactly what CI runs: install, no-documents check, ruff, tests + coverage, web page tests, pip-audit
```
- **CI.** `scripts/ci.sh` does everything, so any CI system can call it; `.github/workflows/ci.yml` runs it on Python 3.10 and 3.12.
  `requirements-ci.txt` leaves out `sentence-transformers`/torch because the tests use a fake embedder. Coverage floor: `COV_MIN`
  (default 85). The dependency audit (`pip-audit`) never blocks the build.
- **Bank policy documents must never be committed.** `.gitignore` excludes `*.pdf`, `*.docx`, `.env`, `chroma_db/`, `uploads/`,
  `results/`, `data/` and `chunks.json`; tests build their files on the fly. `scripts/check_no_documents.sh` fails CI if a document is
  tracked. Turn on the local guard once per clone: `git config core.hooksPath .githooks`. If a document was committed earlier, run
  `git rm --cached <file>` (the sample handbook `pub-ch-retail-lending.pdf` is only used by the live test and stays on your disk).
- **Known weakness, recorded as strict expected failures** in `tests/test_golden_phrasing.py`: the number reader also returns the
  *parts* of a number written in words ("seventy two" gives 2, 70 and 72), so a model that claims the value 2 for "seventy two months"
  passes the hard grounding checks. The test will start failing the day this is fixed, and then the `xfail` marker should be removed.

## Roadmap / Not built yet

Not built, by decision: Streamlit (the HTML page replaces it); login or single sign-on; OCR for scanned PDFs; hybrid search, the
reranker and section-aware chunking; speed optimisations; freeze protection (memory guard, `--threads`, `--gentle`); a glossary file
and document-aware synonym expansion.

**Known limitations:** conversations are in memory (a restart clears them); batches run one field at a time (about 4-5 minutes each on a
laptop CPU); expected results are only as reliable as the extracted rule, and unclear cases are marked "Confirm"; documents indexed from the
command line appear only after a server restart; there is no login (the server listens on `127.0.0.1`); re-uploading overwrites the saved copy
of a file even if the new one is broken (the index stays safe).

Guides: `docs/USER_GUIDE.md` (for testers) and `docs/DEPLOYMENT.md` (install, offline set-up, shared Ollama, security, troubleshooting).
