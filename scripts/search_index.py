#!/usr/bin/env python3
"""
Phase 3 sanity check: run a plain-text search against the indexed
collection and see what comes back. This confirms embeddings + Chroma are
working end to end, before Phase 4/5 build field-aware retrieval on top.

Usage:
    python scripts/search_index.py "maximum loan amount"
    python scripts/search_index.py "when does the new eligibility rule start" --n 3
    python scripts/search_index.py "repayment period" --source-file policy.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lending_agent.indexing.embeddings import EmbeddingModelError  # noqa: E402
from lending_agent.indexing.vector_store import VectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", help="Free-text search query")
    parser.add_argument("--n", type=int, default=5, help="Number of results to return (default: 5)")
    parser.add_argument("--source-file", default=None, help="Restrict the search to one indexed source file")
    args = parser.parse_args()

    store = VectorStore()

    try:
        total = store.count()
    except Exception as exc:
        print(f"✗ Could not open the Chroma collection: {exc}")
        return 1

    if total == 0:
        print("✗ The index is empty. Run scripts/index_documents.py on a policy file first.")
        return 1

    try:
        results = store.search(args.query, n_results=args.n, source_file=args.source_file)
    except EmbeddingModelError as exc:
        print(f"✗ {exc}")
        return 1

    print(f"→ Query: '{args.query}'  ({total} chunk(s) indexed)\n")
    if not results:
        print("(no matches)")
        return 0

    for i, r in enumerate(results, 1):
        page = f"page {r.page_number}" if r.page_number else "(no pagination)"
        print(f"[{i}] {r.source_file} | {page} | similarity ~{r.similarity:.2f}")
        preview = r.text[:280].replace("\n", " ")
        print(f"    {preview}{'...' if len(r.text) > 280 else ''}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
