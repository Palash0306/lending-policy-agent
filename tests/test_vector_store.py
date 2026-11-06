"""
Phase 3 tests for VectorStore. These use the *real* chromadb library
(pure local storage, no network needed) pointed at a temp directory, with
a deterministic FakeEmbeddingModel injected instead of the real
sentence-transformers model -- so tests are fast, fully offline, and still
exercise real Chroma read/write/query behavior.
"""
from __future__ import annotations

from lending_agent.config import ChromaSettings
from lending_agent.indexing.vector_store import VectorStore
from lending_agent.ingestion.models import Chunk


def _make_chunk(chunk_id, text, source_file="policy.pdf", page_number=1, chunk_index=0):
    return Chunk(
        chunk_id=chunk_id,
        source_file=source_file,
        doc_type="pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        text=text,
    )


def _store(tmp_path, fake_embedding_model, collection_name="test"):
    settings = ChromaSettings(persist_directory=str(tmp_path / "chroma"), collection_name=collection_name)
    return VectorStore(settings=settings, embedding_model=fake_embedding_model)


class TestAddChunks:
    def test_add_chunks_returns_count_and_updates_store(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        chunks = [
            _make_chunk("c1", "loan amount shall not exceed 500000"),
            _make_chunk("c2", "applicant age must be at least 21"),
        ]
        added = store.add_chunks(chunks)
        assert added == 2
        assert store.count() == 2

    def test_add_empty_list_is_a_noop(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        assert store.add_chunks([]) == 0
        assert store.count() == 0

    def test_upserting_the_same_chunk_id_does_not_duplicate(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        chunk = _make_chunk("c1", "loan amount shall not exceed 500000")
        store.add_chunks([chunk])
        store.add_chunks([chunk])  # re-indexing the same file/chunk should be safe
        assert store.count() == 1


class TestSearch:
    def test_returns_the_more_relevant_chunk_first(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        store.add_chunks(
            [
                _make_chunk(
                    "loan",
                    "The sanctioned loan amount shall not exceed five hundred thousand dollars",
                    chunk_index=0,
                ),
                _make_chunk(
                    "age",
                    "Applicant age at time of application must be at least twenty one years",
                    chunk_index=1,
                ),
            ]
        )
        results = store.search("maximum loan amount dollars", n_results=2)
        assert results[0].chunk_id == "loan"

    def test_respects_source_file_filter(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        store.add_chunks(
            [
                _make_chunk("a", "loan amount limit text here", source_file="policy_a.pdf"),
                _make_chunk("b", "loan amount limit text here", source_file="policy_b.pdf"),
            ]
        )
        results = store.search("loan amount limit", n_results=5, source_file="policy_a.pdf")
        assert len(results) == 1
        assert results[0].source_file == "policy_a.pdf"

    def test_docx_style_null_page_number_round_trips_as_none(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        store.add_chunks([_make_chunk("c1", "docx text with no pagination", page_number=None)])
        results = store.search("docx text", n_results=1)
        assert results[0].page_number is None

    def test_similarity_is_bounded_between_zero_and_one(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        store.add_chunks([_make_chunk("c1", "loan amount shall not exceed 500000")])
        results = store.search("loan amount", n_results=1)
        assert 0.0 <= results[0].similarity <= 1.0 + 1e-9


class TestReset:
    def test_reset_clears_the_collection(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        store.add_chunks([_make_chunk("c1", "some policy text")])
        assert store.count() == 1
        store.reset()
        assert store.count() == 0

    def test_reset_on_an_empty_store_does_not_raise(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        store.reset()  # nothing exists yet -- should be a harmless no-op


class TestSourcesAndDelete:
    def test_sources_counts_chunks_per_file(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        store.add_chunks([
            _make_chunk("a1", "loan amount shall not exceed 500000", source_file="a.pdf"),
            _make_chunk("a2", "applicant age must be at least 21", source_file="a.pdf", chunk_index=1),
            _make_chunk("b1", "interest rate is capped at 36 percent", source_file="b.pdf"),
        ])
        assert store.sources() == {"a.pdf": 2, "b.pdf": 1}

    def test_sources_of_an_empty_store(self, tmp_path, fake_embedding_model):
        assert _store(tmp_path, fake_embedding_model).sources() == {}

    def test_delete_source_removes_only_that_file(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        store.add_chunks([
            _make_chunk("a1", "loan amount shall not exceed 500000", source_file="a.pdf"),
            _make_chunk("b1", "interest rate is capped at 36 percent", source_file="b.pdf"),
        ])
        assert store.delete_source("a.pdf") == 1
        assert store.sources() == {"b.pdf": 1} and store.count() == 1

    def test_deleting_an_unknown_file_removes_nothing(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        store.add_chunks([_make_chunk("a1", "loan amount shall not exceed 500000", source_file="a.pdf")])
        assert store.delete_source("ghost.pdf") == 0 and store.count() == 1
