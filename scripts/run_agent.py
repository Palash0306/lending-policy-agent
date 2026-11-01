#!/usr/bin/env python3
"""
Phase 7: run the agent (a LangGraph graph) for one or many fields.

For each field it expands the phrasings, searches, has Ollama extract, verifies by plain
code, and loops back when it should (retry with feedback, widen the search once). The
final answer is routed to: verified / needs_review / not_found / error.

Results are saved to --out-dir as each field finishes, and --resume skips fields already
saved, so a stopped or crashed run loses nothing.

Usage:
    python scripts/run_agent.py loan_tenure
    python scripts/run_agent.py loan_tenure loan_amount applicant_age --out-dir results
    python scripts/run_agent.py loan_tenure loan_amount --out-dir results --resume
    python scripts/run_agent.py "What is the maximum loan tenure allowed?" --timeout 900
Options for slow machines: --top-k 3, --no-llm-expansion, --max-attempts 1, --no-widen
Optional precision extras (need the precision-drop files): --hybrid, --rerank
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lending_agent.agent.extraction import ALL_KINDS, KIND_DATE, KIND_NUMERIC  # noqa: E402
from lending_agent.agent.graph import (  # noqa: E402
    ROUTE_ERROR, ROUTE_NEEDS_REVIEW, ROUTE_NOT_FOUND, ROUTE_VERIFIED, AgentAnswer, AgentConfig, ask, build_graph,
)
from lending_agent.config import get_settings  # noqa: E402
from lending_agent.timing import RunTimer, format_duration  # noqa: E402

KIND_CHOICES = {"both": ALL_KINDS, "boundary": (KIND_NUMERIC,), "date": (KIND_DATE,)}
LABELS = {ROUTE_VERIFIED: "✓ VERIFIED", ROUTE_NEEDS_REVIEW: "⚠ NEEDS REVIEW", ROUTE_NOT_FOUND: "– NOT FOUND", ROUTE_ERROR: "✗ ERROR"}


def _filename(field: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", field).strip("_")[:80] or "field"


def _print_answer(answer: AgentAnswer) -> None:
    extras = []
    if answer.attempts > 1:
        extras.append(f"{answer.attempts} extraction attempts")
    if answer.widened:
        extras.append("search widened")
    print(f"\n━━ {answer.field}  {LABELS[answer.route]}" + (f"   ({', '.join(extras)})" if extras else ""))
    print("   " + answer.summary.replace("\n", "\n   "))
    for number, vf in enumerate(answer.result.findings, 1):
        for reason in vf.review_reasons:
            print(f"   ⚠ #{number}: {reason}")
    for conflict in answer.result.conflicts:
        print(f"   ⚠ CONFLICT in {conflict.limit}: " + ", ".join(f"{v['value']} ({v['source_file']})" for v in conflict.values))
    if answer.route in (ROUTE_VERIFIED, ROUTE_NEEDS_REVIEW):
        print("   Spot-check the quoted page before using any value in a test plan.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fields", nargs="+", help="Field names (loan_tenure) or short questions")
    parser.add_argument("--kinds", choices=sorted(KIND_CHOICES), default="both")
    parser.add_argument("--top-k", type=int, default=6, help="Paragraphs to read (default: 6)")
    parser.add_argument("--max-attempts", type=int, default=2, help="Total extraction attempts per field (default: 2)")
    parser.add_argument("--no-widen", action="store_true", help="Do not widen the search when nothing is found")
    parser.add_argument("--no-focus", action="store_true", help="Send whole paragraphs to Ollama (slower)")
    parser.add_argument("--no-llm-expansion", action="store_true", help="Use only the built-in phrasing list (saves one Ollama call per field)")
    parser.add_argument("--source-file", default=None, help="Restrict the search to one indexed file")
    parser.add_argument("--timeout", type=float, default=None, help="Seconds to wait for each Ollama answer")
    parser.add_argument("--out-dir", default=None, help="Save each field's result here as JSON, as it finishes")
    parser.add_argument("--resume", action="store_true", help="Skip fields already saved in --out-dir")
    parser.add_argument("--hybrid", action="store_true", help="Use keyword + meaning search (precision-drop files needed)")
    parser.add_argument("--rerank", action="store_true", help="Re-sort results with the reranker (precision-drop files needed)")
    parser.add_argument("--json", action="store_true", help="Print all results as one JSON list instead of text")
    args = parser.parse_args()

    from lending_agent.indexing.embeddings import EmbeddingModelError
    from lending_agent.indexing.vector_store import VectorStore
    from lending_agent.llm.ollama_client import OllamaClient

    store = VectorStore()
    if store.count() == 0:
        print("✗ The index is empty. Run scripts/index_documents.py on a policy file first.")
        return 1

    searcher, reranker = store, None
    try:
        if args.hybrid:
            from lending_agent.indexing.hybrid import HybridSearcher

            searcher = HybridSearcher(store)
        if args.rerank:
            from lending_agent.indexing.reranker import Reranker

            reranker = Reranker()
    except ImportError:
        print("✗ --hybrid / --rerank need the precision-drop files (indexing/hybrid.py, indexing/reranker.py).")
        return 1

    settings = get_settings().ollama
    if args.timeout:
        settings = replace(settings, request_timeout=args.timeout)
    config = AgentConfig(
        kinds=KIND_CHOICES[args.kinds], top_k=args.top_k, widen_factor=1 if args.no_widen else 2,
        max_attempts=max(1, args.max_attempts), focus=not args.no_focus,
        use_llm_expansion=not args.no_llm_expansion, source_file=args.source_file,
    )
    graph = build_graph(searcher, llm=OllamaClient(settings), config=config, reranker=reranker)

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    timer = RunTimer()
    timer.start()
    results: list[tuple[str, str]] = []
    collected: list[dict] = []
    try:
        for field in args.fields:
            target = out_dir / f"{_filename(field)}.json" if out_dir else None
            if args.resume and target and target.exists():
                saved = json.loads(target.read_text(encoding="utf-8"))
                results.append((field, saved["route"] + " (from earlier run)"))
                collected.append(saved)
                continue
            try:
                with timer.step(f"Field: {field}", note="Ollama is thinking; not frozen."):
                    answer = ask(graph, field)
            except EmbeddingModelError as exc:
                print(f"✗ {exc}")
                return 1
            except Exception as exc:
                if type(exc).__name__ == "RerankerError":
                    print(f"✗ {exc}\n  Run again without --rerank.")
                    return 1
                raise
            if not args.json:
                _print_answer(answer)
            data = answer.to_dict()
            collected.append(data)
            results.append((field, answer.route))
            if target:
                target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except KeyboardInterrupt:
        print(f"\n✗ Stopped by you after {len(collected)} of {len(args.fields)} field(s)."
              + (" Finished fields are saved; add --resume to continue." if out_dir else ""), file=sys.stderr)
        return 130

    timer.finish()
    if args.json:
        print(json.dumps(collected, indent=2))
    else:
        print("\n⏱  Summary")
        for field, route in results:
            print(f"   {field:<28} {route}")
        print(f"   Total time: {format_duration(timer.total_seconds or 0)}" + (f"   (results in {out_dir}/)" if out_dir else ""))
    return 1 if any(route == ROUTE_ERROR for _, route in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
