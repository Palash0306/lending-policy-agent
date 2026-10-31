"""
Ties loaders + chunker together: file(s) on disk -> a list of cited Chunk
objects, ready for Chroma indexing in Phase 3.

Includes both a single-file entry point (`ingest_document`) and a
multi-file one (`ingest_documents`) that accepts several PDFs/DOCX files
(or a mix of files and folders) in one call -- this is what the FastAPI
upload endpoint in Phase 8 will call under the hood when several files
are uploaded at once, and what `indexing/indexer.py` uses today.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

from lending_agent.ingestion.chunker import chunk_text
from lending_agent.ingestion.loaders import SUPPORTED_EXTENSIONS, load_document
from lending_agent.ingestion.models import Chunk, SourceDocument


def _short_hash(text: str, length: int = 8) -> str:
    """Short, stable hash of a full file path, used to keep chunk_ids unique
    even if two different documents happen to share the same filename."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def ingest_document(path: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Chunk]:
    """Load a single PDF or DOCX policy file and split it into overlapping, cited chunks."""
    doc = load_document(path)
    return chunk_source_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_source_document(doc: SourceDocument, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[Chunk]:
    """Chunk an already-loaded SourceDocument (useful if a caller loaded it themselves)."""
    chunks: list[Chunk] = []
    global_index = 0
    doc_hash = _short_hash(os.path.abspath(doc.path))
    for page in doc.pages:
        if not page.text.strip():
            continue
        for piece in chunk_text(page.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            if not piece.strip():
                continue
            chunk_id = f"{doc.filename}::{doc_hash}::p{page.page_number or 0}::c{global_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_file=doc.filename,
                    doc_type=doc.doc_type,
                    page_number=page.page_number,
                    chunk_index=global_index,
                    text=piece.strip(),
                )
            )
            global_index += 1
    return chunks


def resolve_input_paths(paths: Iterable[str]) -> list[str]:
    """
    Expand a mix of file paths and folder paths into a flat, sorted list of
    supported document files. A folder is scanned (non-recursively) for any
    .pdf/.docx files -- this is what lets the CLI accept a whole policy
    folder in one call instead of listing every file by hand.
    """
    resolved: list[str] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(f"No such file or directory: '{raw}'")
        if p.is_dir():
            found = sorted(
                str(f) for f in p.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not found:
                raise FileNotFoundError(
                    f"'{raw}' is a directory but contains no supported files {sorted(SUPPORTED_EXTENSIONS)}."
                )
            resolved.extend(found)
        else:
            resolved.append(str(p))
    # De-duplicate while preserving order, in case a file was named both
    # directly and via a folder that contains it.
    seen: set[str] = set()
    deduped = []
    for path in resolved:
        key = os.path.abspath(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def ingest_documents(
    paths: Iterable[str], chunk_size: int = 1000, chunk_overlap: int = 150
) -> dict[str, list[Chunk]]:
    """
    Ingest several PDF/DOCX files (or folders containing them) in one call.

    Returns a dict keyed by the original file path, each value the list of
    Chunks for that file -- so a caller (the CLI, or later the FastAPI
    upload endpoint) can report per-file results, while still being able to
    flatten everything into one list for indexing:

        results = ingest_documents(paths)
        all_chunks = [c for chunks in results.values() for c in chunks]
    """
    file_paths = resolve_input_paths(paths)
    results: dict[str, list[Chunk]] = {}
    for path in file_paths:
        results[path] = ingest_document(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return results
