#!/usr/bin/env python3
"""
Phase 4 entry point: see how a field name (or a question) gets expanded into
the phrasings a policy document might use -- and, optionally, which chunks of
your indexed documents those phrasings find.

Usage:
    python scripts/expand_field.py loan_amount
    python scripts/expand_field.py "What is the maximum loan tenure allowed?"
    python scripts/expand_field.py loan_tenure --no-llm        # built-in list only, Ollama not needed
    python scripts/expand_field.py loan_tenure --retrieve      # also search the index (run index_documents.py first)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lending_agent.agent.field_expansion import expand_field_terms  # noqa: E402
from lending_agent.agent.retrieval import retrieve_evidence  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="A field name like loan_amount, or a short question")
    parser.add_argument("--no-llm", action="store_true", help="Skip Ollama; use only the built-in word list")
    parser.add_argument("--retrieve", action="store_true", help="Also search the Chroma index with each phrasing")
    parser.add_argument("--top-k", type=int, default=5, help="How many evidence chunks to show (default: 5)")
    parser.add_argument("--source-file", default=None, help="Restrict retrieval to one indexed file")
    args = parser.parse_args()

    expansion = expand_field_terms(args.query, use_llm=not args.no_llm)

    print(f"→ Field:   {expansion.field}")
    print(f"→ Source:  {expansion.source}")
    if expansion.llm_error:
        print(f"→ Note:    {expansion.llm_error}")
    print("→ Phrasings to search for:")
    for term in expansion.terms:
        print(f"    - {term}")

    if not args.retrieve:
        return 0

    from lending_agent.indexing.embeddings import EmbeddingModelError
    from lending_agent.indexing.vector_store import VectorStore

    store = VectorStore()
    try:
        if store.count() == 0:
            print("\n✗ The index is empty. Run scripts/index_documents.py on a policy file first.")
            return 1
        evidence = retrieve_evidence(
            expansion.terms, store, top_k=args.top_k, source_file=args.source_file
        )
    except EmbeddingModelError as exc:
        print(f"\n✗ {exc}")
        return 1

    print(f"\n→ Top {len(evidence)} chunk(s) found:\n")
    for i, e in enumerate(evidence, 1):
        page = f"page {e.page_number}" if e.page_number else "(no pagination)"
        print(f"[{i}] {e.source_file} | {page} | similarity ~{e.similarity:.2f} | found by {len(e.matched_terms)} phrasing(s)")
        preview = e.text[:280].replace("\n", " ")
        print(f"    {preview}{'...' if len(e.text) > 280 else ''}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
