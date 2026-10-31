"""
Splits page/section text into overlapping, retrieval-sized chunks.

Chunking (rather than retrieving whole pages) matters here because a single
policy page routinely covers several unrelated fields -- e.g. loan amount
limits AND applicant age eligibility on the same page. Smaller, overlapping
chunks let Phase 3's vector search return just the paragraph that actually
contains a given boundary, and the overlap keeps a boundary condition that
straddles two sentences (a stated limit whose date-cutoff qualifier is in
the *next* sentence) from being cut apart at a chunk edge.
"""
from __future__ import annotations

import re

# Split after sentence-ending punctuation, but only when followed by
# whitespace and then something that looks like the start of a new
# sentence (capital letter, digit, opening quote/paren) -- this avoids
# splitting on things like "Rs." or "e.g." followed by a lowercase word.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'])")


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    sentences: list[str] = []
    for para in paragraphs:
        para = para.replace("\n", " ").strip()
        for piece in _SENTENCE_SPLIT_RE.split(para):
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    return sentences


def _hard_split(sentence: str, chunk_size: int) -> list[str]:
    """Fallback for a single 'sentence' longer than chunk_size (no punctuation, OCR noise, a giant table row)."""
    return [sentence[i : i + chunk_size] for i in range(0, len(sentence), chunk_size)]


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """
    Greedily pack sentences into chunks up to `chunk_size` characters each,
    carrying the trailing `chunk_overlap` characters' worth of sentences
    forward into the start of the next chunk for context continuity.

    Returns an empty list for empty/whitespace-only input.
    """
    if not text or not text.strip():
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        if current:
            chunks.append(" ".join(current))

    for sentence in sentences:
        if len(sentence) > chunk_size:
            flush()
            current.clear()
            current_len = 0
            chunks.extend(_hard_split(sentence, chunk_size))
            continue

        added_len = len(sentence) + (1 if current else 0)
        if current and current_len + added_len > chunk_size:
            flush()
            # Carry the trailing sentences of the chunk just flushed
            # forward as the start of the next one, up to chunk_overlap chars.
            overlap: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                extra = len(s) + (1 if overlap else 0)
                if overlap_len + extra > chunk_overlap:
                    break
                overlap.insert(0, s)
                overlap_len += extra
            current = overlap
            current_len = overlap_len

        current.append(sentence)
        current_len += len(sentence) + (1 if len(current) > 1 else 0)

    flush()
    return chunks


# Public name, for code outside this file that needs the same sentence splitting.
split_sentences = _split_sentences
