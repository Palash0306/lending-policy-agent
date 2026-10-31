"""
Phase 3 entry point: file(s) on disk -> loaded -> chunked -> embedded ->
stored in Chroma. This is what scripts/index_documents.py calls, and later
what the FastAPI upload endpoint (Phase 8) will call for each uploaded file.
"""
from __future__ import annotations

from typing import Iterable

from lending_agent.indexing.vector_store import VectorStore
from lending_agent.ingestion.pipeline import ingest_documents


def index_documents(
    paths: Iterable[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    store: VectorStore | None = None,
) -> dict[str, int]:
    """
    Ingest and index one or more PDF/DOCX files (or folders containing
    them). Returns a dict of {file_path: chunks_added}.
    """
    store = store or VectorStore()
    results = ingest_documents(paths, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return {path: store.add_chunks(chunks) for path, chunks in results.items()}
