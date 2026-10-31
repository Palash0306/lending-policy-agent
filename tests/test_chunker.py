from lending_agent.ingestion.chunker import chunk_text


class TestChunkText:
    def test_short_text_is_a_single_chunk(self):
        text = "Loan amount shall not exceed $500,000."
        chunks = chunk_text(text, chunk_size=1000, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_is_split_into_multiple_chunks(self):
        sentence = "The sanctioned amount shall not exceed the applicable tier limit. "
        text = sentence * 40  # ~2,800 chars
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=80)
        assert len(chunks) > 1
        # allow some slack: chunks are built from whole sentences, not hard character cuts
        assert all(len(c) <= 700 for c in chunks)

    def test_consecutive_chunks_overlap(self):
        sentences = [f"Clause number {i} sets a boundary condition for testing purposes." for i in range(30)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=400, chunk_overlap=100)
        assert len(chunks) > 1
        for a, b in zip(chunks, chunks[1:]):
            tail_words = set(a[-100:].split())
            head_words = set(b[:100].split())
            assert tail_words & head_words, "consecutive chunks should share some overlapping text"

    def test_empty_text_returns_no_chunks(self):
        assert chunk_text("", chunk_size=500, chunk_overlap=50) == []
        assert chunk_text("   \n\n  ", chunk_size=500, chunk_overlap=50) == []

    def test_oversized_single_sentence_is_hard_split(self):
        text = "x" * 3000  # no punctuation at all -- simulates OCR noise / a giant table row
        chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 6
        assert all(len(c) <= 500 for c in chunks)

    def test_rejects_overlap_not_smaller_than_chunk_size(self):
        import pytest

        with pytest.raises(ValueError):
            chunk_text("some text", chunk_size=100, chunk_overlap=100)
