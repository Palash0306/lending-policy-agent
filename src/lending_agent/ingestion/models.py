"""
Data structures shared by the ingestion pipeline (Phase 2) and everything
downstream that consumes chunks -- vector indexing (Phase 3), extraction
(Phase 5). Kept dependency-free (no pdfplumber/docx imports here) so these
types can be imported anywhere without pulling in parsing libraries.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageText:
    """Raw extracted text from one page (PDF) or the whole document (DOCX)."""

    page_number: int | None  # 1-indexed for PDFs; None for DOCX (no native pagination)
    text: str


@dataclass(frozen=True)
class SourceDocument:
    """A loaded policy document, before chunking."""

    path: str
    filename: str
    doc_type: str  # "pdf" or "docx"
    pages: list[PageText] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text)


@dataclass(frozen=True)
class Chunk:
    """
    A retrieval-sized piece of a document, with enough metadata to cite it
    back to the analyst (source file + page) and to feed Chroma (Phase 3).
    """

    chunk_id: str
    source_file: str
    doc_type: str
    page_number: int | None
    chunk_index: int
    text: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "doc_type": self.doc_type,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }
