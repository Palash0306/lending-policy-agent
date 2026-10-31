"""
Node 2 of the pipeline: retrieve_evidence.

Takes the phrasings from field_expansion.py, searches the index once per
phrasing, and merges everything into one ranked list of unique chunks. A
chunk found by several different phrasings is only listed once, with all
the phrasings that found it -- a useful signal that it really is about
the field.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

from lending_agent.indexing.vector_store import SearchResult


class _Searchable(Protocol):
    def search(self, query: str, n_results: int = 5, source_file: str | None = None) -> list[SearchResult]: ...


@dataclass
class Evidence:
    chunk_id: str
    text: str
    source_file: str
    page_number: int | None
    doc_type: str
    best_distance: float                       # lowest (best) distance across all phrasings
    matched_terms: list[str] = field(default_factory=list)

    @property
    def similarity(self) -> float:
        return max(0.0, 1.0 - self.best_distance)


def retrieve_evidence(
    terms: Iterable[str],
    store: _Searchable,
    n_per_term: int = 5,
    top_k: int = 8,
    source_file: str | None = None,
) -> list[Evidence]:
    """Search once per phrasing, merge by chunk_id, return the best `top_k` chunks."""
    merged: dict[str, Evidence] = {}

    for term in terms:
        if not term.strip():
            continue
        for hit in store.search(term, n_results=n_per_term, source_file=source_file):
            existing = merged.get(hit.chunk_id)
            if existing is None:
                merged[hit.chunk_id] = Evidence(
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                    source_file=hit.source_file,
                    page_number=hit.page_number,
                    doc_type=hit.doc_type,
                    best_distance=hit.distance,
                    matched_terms=[term],
                )
            else:
                existing.best_distance = min(existing.best_distance, hit.distance)
                existing.matched_terms.append(term)

    ranked = sorted(merged.values(), key=lambda e: (e.best_distance, -len(e.matched_terms)))
    return ranked[:top_k]
