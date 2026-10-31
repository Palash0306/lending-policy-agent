"""
Loaders that turn a lending policy file on disk into a SourceDocument.
Two formats are supported, matching the brief: PDF (pdfplumber) and
Word/.docx (python-docx). Policy documents are internal bank files, so
these operate purely on local files -- nothing here makes a network call.
"""
from __future__ import annotations

import os

import pdfplumber
from docx import Document as DocxDocument

from lending_agent.ingestion.models import PageText, SourceDocument

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class UnsupportedDocumentError(ValueError):
    """Raised when a file extension isn't one of SUPPORTED_EXTENSIONS."""


def load_pdf(path: str) -> SourceDocument:
    """Extract text page by page, preserving page numbers for citations."""
    pages: list[PageText] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append(PageText(page_number=i + 1, text=text.strip()))
    return SourceDocument(path=path, filename=os.path.basename(path), doc_type="pdf", pages=pages)


def load_docx(path: str) -> SourceDocument:
    """
    .docx has no native page boundaries (pagination is a rendering-time
    concept in Word, not stored in the file), so the whole document is
    returned as a single PageText with page_number=None.

    Paragraphs and tables are extracted separately and concatenated: plain
    `doc.paragraphs` in python-docx skips table content entirely, and
    lending policies frequently put tiered limits (e.g. LTV ratio by tier)
    in tables -- silently dropping those would mean missing real boundary
    values. Each table row is rendered as "cell | cell | cell" so it stays
    human-readable in citations and search results.
    """
    doc = DocxDocument(path)
    parts: list[str] = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())

    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))

    full_text = "\n\n".join(parts)
    return SourceDocument(
        path=path,
        filename=os.path.basename(path),
        doc_type="docx",
        pages=[PageText(page_number=None, text=full_text)],
    )


def load_document(path: str) -> SourceDocument:
    """Dispatch to the right loader based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    if ext == ".docx":
        return load_docx(path)
    raise UnsupportedDocumentError(
        f"Unsupported file type '{ext}' for '{path}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )
