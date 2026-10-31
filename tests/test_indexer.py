"""
Phase 3 tests for index_documents(): ties real PDF/DOCX ingestion
(Phase 2) to a real local Chroma store, with a fake embedder standing in
for sentence-transformers so no network/model download is needed.
"""
from __future__ import annotations

from lending_agent.config import ChromaSettings
from lending_agent.indexing.indexer import index_documents
from lending_agent.indexing.vector_store import VectorStore


def _store(tmp_path, fake_embedding_model):
    settings = ChromaSettings(persist_directory=str(tmp_path / "chroma"), collection_name="test")
    return VectorStore(settings=settings, embedding_model=fake_embedding_model)


class TestIndexDocuments:
    def test_indexes_a_single_pdf(self, tmp_path, sample_pdf_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        summary = index_documents([sample_pdf_path], store=store)
        assert sum(summary.values()) > 0
        assert store.count() == sum(summary.values())

    def test_indexes_multiple_files_in_one_call(self, tmp_path, sample_pdf_path, sample_docx_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        summary = index_documents([sample_pdf_path, sample_docx_path], store=store)
        assert len(summary) == 2
        assert store.count() == sum(summary.values())

    def test_indexed_content_is_actually_searchable(self, tmp_path, sample_pdf_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        index_documents([sample_pdf_path], store=store)
        results = store.search("maximum loan amount five hundred thousand dollars", n_results=3)
        assert any("500,000" in r.text for r in results)

    def test_reindexing_the_same_file_does_not_duplicate_chunks(self, tmp_path, sample_pdf_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        index_documents([sample_pdf_path], store=store)
        first_count = store.count()
        index_documents([sample_pdf_path], store=store)  # re-run on the same file
        assert store.count() == first_count
