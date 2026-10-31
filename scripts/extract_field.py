#!/usr/bin/env python3
"""
Phases 4-6 in one command: ask about a field and get structured boundary /
date-cutoff findings, each with its exact quote, source file and page, and each
re-checked against the source text by plain code.

Steps:
    1. expand the field into phrasings      (Phase 4, uses Ollama)
    2. search the index with every phrasing (needs index_documents.py first)
    3. read the best paragraphs and extract (Phase 5, uses Ollama)
    4. verify every finding against the text and flag conflicts (Phase 6, no AI)

Usage:
    python scripts/extract_field.py loan_tenure
    python scripts/extract_field.py "What is the maximum loan tenure allowed?"
    python scripts/extract_field.py loan_amount --kinds boundary      # numeric limits only
    python scripts/extract_field.py loan_tenure --kinds date          # date cutoffs only
    python scripts/extract_field.py loan_tenure --timeout 900         # allow Ollama 15 minutes
    python scripts/extract_field.py loan_tenure --fast                # lighter on memory and time
    python scripts/extract_field.py loan_tenure --no-focus            # send whole paragraphs (slower)
    python scripts/extract_field.py loan_tenure --json                # machine-readable output
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lending_agent.agent.assurance import VerifiedFinding, verify_extraction  # noqa: E402
from lending_agent.agent.extraction import (  # noqa: E402
    ALL_KINDS,
    DEFAULT_NUM_CTX,
    KIND_DATE,
    KIND_NUMERIC,
    STATUS_ERROR,
    STATUS_FOUND,
    extract_conditions,
)
from lending_agent.agent.field_expansion import expand_field_terms  # noqa: E402
from lending_agent.agent.retrieval import retrieve_evidence  # noqa: E402
from lending_agent.config import get_settings  # noqa: E402
from lending_agent.timing import RunTimer, format_duration  # noqa: E402

KIND_CHOICES = {"both": ALL_KINDS, "boundary": (KIND_NUMERIC,), "date": (KIND_DATE,)}


def _print_finding(i: int, vf: VerifiedFinding) -> None:
    f = vf.finding
    if f.kind == KIND_NUMERIC:
        headline = f"{f.bound_type.upper()}  {f.value} {f.unit or ''}".rstrip()
    else:
        headline = f"DATE CUTOFF  {f.date_direction}  {f.date}"
        if f.date_precision == "month":
            headline += "  (month only: day not stated)"
    mark = "✓ grounded" if vf.grounded else "✗ NOT grounded"
    print(f"[{i}] {headline}   ({f.classification})   {mark}")
    if f.classification != "firm_rule":
        print(f"    ! tagged '{f.classification}': not a firm rule, do not use as a hard limit without checking")
    if f.applies_to:
        print(f"    applies to: {f.applies_to}")
    print(f'    quote: "{f.quote}"')
    page = f", page {f.page_number}" if f.page_number else " (no pagination)"
    print(f"    source: {f.source_file}{page}")
    for reason in vf.review_reasons:
        print(f"    ⚠ {reason}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="A field name like loan_tenure, or a short question")
    parser.add_argument("--kinds", choices=sorted(KIND_CHOICES), default="both",
                        help="What to look for: numeric limits, date cutoffs, or both (default)")
    parser.add_argument("--top-k", type=int, default=6, help="How many paragraphs to read (default: 6)")
    parser.add_argument("--source-file", default=None, help="Restrict the search to one indexed file")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX,
                        help=f"Ollama context window in tokens (default: {DEFAULT_NUM_CTX})")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Seconds to wait for each Ollama answer (default: OLLAMA_TIMEOUT from .env, else 300)")
    parser.add_argument("--fast", action="store_true",
                        help="Lighter run for slow or freezing laptops: searches for 3 paragraphs instead of 6 "
                             "(overrides --top-k)")
    parser.add_argument("--no-focus", action="store_true",
                        help="Send whole paragraphs to Ollama instead of only the sentences with numbers or dates (slower)")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON")
    args = parser.parse_args()
    if args.fast:
        args.top_k = 3

    timer = RunTimer()
    timer.start()
    try:
        return run(args, timer)
    except KeyboardInterrupt:
        timer.finish()
        print(f"\n✗ Stopped by you after {format_duration(timer.total_seconds)}. Nothing was saved.", file=sys.stderr)
        return 130


def run(args: argparse.Namespace, timer: RunTimer) -> int:

    from lending_agent.indexing.embeddings import EmbeddingModelError
    from lending_agent.indexing.vector_store import VectorStore
    from lending_agent.llm.ollama_client import OllamaClient

    ollama_settings = get_settings().ollama
    if args.timeout:
        ollama_settings = replace(ollama_settings, request_timeout=args.timeout)
    llm = OllamaClient(ollama_settings)

    with timer.step("Step 1/4: finding likely phrasings for the field (Ollama)", note="Ollama is thinking; not frozen."):
        expansion = expand_field_terms(args.query, llm=llm)

    with timer.step("Step 2/4: searching your indexed documents (loads the embedding model)", note="Loading the embedding model; not frozen."):
        store = VectorStore()
        try:
            if store.count() == 0:
                print("✗ The index is empty. Run scripts/index_documents.py on a policy file first.")
                return 1
            evidence = retrieve_evidence(expansion.terms, store, top_k=args.top_k, source_file=args.source_file)
        except EmbeddingModelError as exc:
            print(f"✗ {exc}")
            return 1

    with timer.step("Step 3/4: Ollama reads the paragraphs that hold numbers or dates. This is the slow step",
                    note="Ollama is thinking; not frozen."):
        result = extract_conditions(
            expansion.field, evidence, llm=llm, kinds=KIND_CHOICES[args.kinds], num_ctx=args.num_ctx,
            focus=not args.no_focus,
        )

    with timer.step("Step 4/4: checking every finding against the source text", heartbeat_every=None):
        verified = verify_extraction(result, evidence)

    timer.finish()

    if args.json:
        print(json.dumps({**verified.to_dict(), "timing": timer.to_dict()}, indent=2))
        return 1 if verified.status == STATUS_ERROR else 0

    print(f"\n→ Field: {verified.field}   (searched {len(expansion.terms)} phrasings, found {len(evidence)} paragraph(s), "
          f"sent {result.paragraphs_sent} to Ollama)")
    if result.paragraphs_sent < len(evidence):
        print("→ Paragraphs without any number or date were not sent: they cannot hold a boundary value. "
              "Use --no-focus to send everything.")
    if expansion.llm_error:
        print(f"→ Note:  {expansion.llm_error}")
    print(f"→ Status: {verified.status.upper()}\n")

    if verified.status == STATUS_ERROR:
        print(f"✗ {verified.error}")
        if "timed out" in (verified.error or ""):
            print("  Ollama needs more time. Try: --timeout 900   (or set OLLAMA_TIMEOUT in .env)")
        return 1

    if verified.status == STATUS_FOUND:
        for i, vf in enumerate(verified.findings, 1):
            _print_finding(i, vf)

        for conflict in verified.conflicts:
            print(f"⚠ CONFLICT: different values for the same {conflict.limit} ({conflict.classification})"
                  + (f", scope: {conflict.scope}" if conflict.scope else ""))
            for v in conflict.values:
                page = f", page {v['page_number']}" if v["page_number"] else ""
                print(f"    finding #{v['finding']}: {v['value']}   ({v['source_file']}{page})")
            print()

        if verified.verified and not verified.needs_review:
            print("✓ VERIFIED: every quote, number and date was found in the source text.")
        elif verified.verified:
            print("✓ Grounded in the source text, but ⚠ a person should review the flagged items above.")
        else:
            print("✗ NEEDS REVIEW: at least one finding could not be confirmed in the source text (see ⚠ above).")
        print("  Spot-check the quoted page before using any value in a test plan.")
    else:
        print("No boundary or date cutoff for this field was found in the paragraphs read.")
        print("(That is a real answer, not a failure: the document may not state one.)")

    if verified.rejected:
        print(f"\n{len(verified.rejected)} answer(s) from the model were thrown away for not fitting the format:")
        for reason in verified.rejected:
            print(f"  - {reason}")

    print("\n⏱  Run time")
    for line in timer.summary_lines():
        print(f"   {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
