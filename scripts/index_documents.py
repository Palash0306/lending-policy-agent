#!/usr/bin/env python3
"""
Phase 3 entry point: chunk one or more policy documents (as in Phase 2)
and store them, embedded, in a local persistent Chroma collection -- so
retrieval (Phase 4/5) has something real to search against.

Usage:
    python scripts/index_documents.py policy.pdf
    python scripts/index_documents.py policy1.pdf policy2.docx
    python scripts/index_documents.py path/to/policy_folder/
    python scripts/index_documents.py policy.pdf --reset   # wipe the collection first
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lending_agent.indexing.embeddings import EmbeddingModelError  # noqa: E402
from lending_agent.indexing.indexer import index_documents  # noqa: E402
from lending_agent.indexing.vector_store import VectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="One or more .pdf/.docx files and/or folders containing them")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max characters per chunk (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=150, help="Overlap characters between chunks (default: 150)")
    parser.add_argument("--reset", action="store_true", help="Delete the existing collection before indexing")
    args = parser.parse_args()

    store = VectorStore()
    print(f"→ Chroma collection: '{store.settings.collection_name}' at {store.settings.persist_directory}")
    print(f"→ Embedding model:   {store.embedding_model.settings.model_name}\n")

    if args.reset:
        print("→ Resetting existing collection...")
        store.reset()

    try:
        summary = index_documents(args.paths, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, store=store)
    except FileNotFoundError as exc:
        print(f"✗ {exc}")
        return 1
    except EmbeddingModelError as exc:
        print(f"✗ {exc}")
        return 1

    total = sum(summary.values())
    print(f"→ Indexed {len(summary)} file(s), {total} chunk(s) total:")
    for path, n in summary.items():
        print(f"  {path}: {n} chunk(s)")

    print(f"\n✓ Collection '{store.settings.collection_name}' now has {store.count()} chunk(s) total.")
    print("  Try: python scripts/search_index.py \"your question about the policy\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
