"""
Phase 4 tests for retrieve_evidence: real Chroma in a temp folder, with the
deterministic fake embedder from conftest.py.
"""
from __future__ import annotations

from lending_agent.agent.retrieval import retrieve_evidence
from lending_agent.config import ChromaSettings
from lending_agent.indexing.vector_store import VectorStore
from lending_agent.ingestion.models import Chunk


def _chunk(chunk_id, text, source_file="policy.pdf", page=1, index=0):
    return Chunk(chunk_id=chunk_id, source_file=source_file, doc_type="pdf",
                 page_number=page, chunk_index=index, text=text)


def _store(tmp_path, embedder):
    settings = ChromaSettings(persist_directory=str(tmp_path / "chroma"), collection_name="test")
    return VectorStore(settings=settings, embedding_model=embedder)


def _loaded_store(tmp_path, embedder):
    store = _store(tmp_path, embedder)
    store.add_chunks([
        _chunk("tenure", "The repayment period shall not extend beyond thirty years", index=0),
        _chunk("amount", "The sanctioned amount shall not exceed five hundred thousand", index=1),
        _chunk("age", "Applicant age must be at least twenty one years", index=2),
    ])
    return store


class TestRetrieveEvidence:
    def test_a_chunk_found_by_two_phrasings_appears_once_with_both(self, tmp_path, fake_embedding_model):
        store = _loaded_store(tmp_path, fake_embedding_model)
        evidence = retrieve_evidence(["repayment period", "extend beyond thirty years"], store, n_per_term=1)
        tenure = [e for e in evidence if e.chunk_id == "tenure"]
        assert len(tenure) == 1
        assert set(tenure[0].matched_terms) == {"repayment period", "extend beyond thirty years"}

    def test_phrasing_that_differs_from_the_field_name_still_finds_the_chunk(self, tmp_path, fake_embedding_model):
        store = _loaded_store(tmp_path, fake_embedding_model)
        # The analyst's field is "loan amount" but the document says "sanctioned amount".
        evidence = retrieve_evidence(["loan amount", "sanctioned amount"], store, n_per_term=1)
        assert any(e.chunk_id == "amount" for e in evidence)

    def test_results_are_ranked_best_first(self, tmp_path, fake_embedding_model):
        store = _loaded_store(tmp_path, fake_embedding_model)
        evidence = retrieve_evidence(["sanctioned amount shall not exceed"], store, n_per_term=3)
        assert evidence[0].chunk_id == "amount"
        distances = [e.best_distance for e in evidence]
        assert distances == sorted(distances)

    def test_top_k_limits_the_number_of_results(self, tmp_path, fake_embedding_model):
        store = _loaded_store(tmp_path, fake_embedding_model)
        evidence = retrieve_evidence(["amount", "period", "age"], store, n_per_term=3, top_k=2)
        assert len(evidence) == 2

    def test_blank_phrasings_are_skipped(self, tmp_path, fake_embedding_model):
        store = _loaded_store(tmp_path, fake_embedding_model)
        assert retrieve_evidence(["", "   "], store) == []

    def test_source_file_filter_is_passed_through(self, tmp_path, fake_embedding_model):
        store = _store(tmp_path, fake_embedding_model)
        store.add_chunks([
            _chunk("a", "loan amount limit", source_file="a.pdf"),
            _chunk("b", "loan amount limit", source_file="b.pdf"),
        ])
        evidence = retrieve_evidence(["loan amount limit"], store, source_file="a.pdf")
        assert [e.source_file for e in evidence] == ["a.pdf"]
