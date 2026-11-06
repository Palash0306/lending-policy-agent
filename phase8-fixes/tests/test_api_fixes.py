"""
Phase 8 fixes: a broken re-upload must never erase a good document, odd delete names must be refused,
security headers must be sent, and .env.example must hold the settings the code reads.

Uses a real Chroma store (with the fake embedder) and a real DocumentService (no injected indexer).
"""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from lending_agent.api.documents import DocumentService
from lending_agent.api.errors import ApiError
from lending_agent.config import ChromaSettings
from lending_agent.indexing.vector_store import VectorStore

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_TEXT = "alpha rule: the repayment period shall not extend beyond thirty (30) years."
NEW_TEXT = "bravo rule: the repayment period shall not extend beyond twenty-five (25) years."


def _store(tmp_path, fake_embedding_model):
    settings = ChromaSettings(persist_directory=str(tmp_path / "chroma"), collection_name="fixes_test")
    return VectorStore(settings=settings, embedding_model=fake_embedding_model)


def _service(tmp_path, fake_embedding_model):
    store = _store(tmp_path, fake_embedding_model)
    return DocumentService(store, tmp_path / "up", 10_000_000), store


def _pdf_bytes(text: str | None) -> bytes:
    """A real one-page PDF. With text=None the page is blank (like a scan with no text layer)."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    if text:
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, text)
    return bytes(pdf.output())


def _upload(svc, name, data):
    return svc.index_files([svc.save_upload(name, io.BytesIO(data))])


class TestReUploadNeverDestroysAGoodDocument:
    def test_a_broken_file_with_the_same_name_leaves_the_good_one_in_place(self, tmp_path, fake_embedding_model):
        svc, store = _service(tmp_path, fake_embedding_model)
        assert _upload(svc, "policy.pdf", _pdf_bytes(OLD_TEXT))["indexed"] == [{"file": "policy.pdf", "chunks": 1}]
        result = _upload(svc, "policy.pdf", b"%PDF-1.4 this is not really a pdf \x00\x01\x02 garbage")
        assert result["failed"] and not result["indexed"]
        assert result["documents"] == [{"name": "policy.pdf", "chunks": 1}]
        assert store.sources() == {"policy.pdf": 1}

    def test_a_good_re_upload_replaces_the_old_text(self, tmp_path, fake_embedding_model):
        svc, store = _service(tmp_path, fake_embedding_model)
        _upload(svc, "policy.pdf", _pdf_bytes(OLD_TEXT))
        result = _upload(svc, "policy.pdf", _pdf_bytes(NEW_TEXT))
        assert result["indexed"] and not result["failed"]
        texts = " ".join(hit.text for hit in store.search("repayment period", n_results=3))
        assert "bravo" in texts and "alpha" not in texts

    def test_a_pdf_with_no_text_is_reported_and_the_existing_document_is_untouched(self, tmp_path, fake_embedding_model):
        svc, store = _service(tmp_path, fake_embedding_model)
        _upload(svc, "policy.pdf", _pdf_bytes(OLD_TEXT))
        result = _upload(svc, "policy.pdf", _pdf_bytes(None))
        assert not result["indexed"] and "no readable text" in result["failed"][0]["error"]
        assert store.sources() == {"policy.pdf": 1}
        assert "alpha" in " ".join(hit.text for hit in store.search("repayment period", n_results=3))


class TestDeleteNames:
    @pytest.mark.parametrize("name", ["", ".", ".."])
    def test_names_that_are_not_documents_are_refused_with_400(self, tmp_path, fake_embedding_model, name):
        svc, _ = _service(tmp_path, fake_embedding_model)
        with pytest.raises(ApiError) as exc:
            svc.delete(name)
        assert exc.value.status == 400


class TestHttp:
    @staticmethod
    def _client(tmp_path, fake_embedding_model):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from lending_agent.api.app import create_app
        from lending_agent.api.chat import ChatService
        from lending_agent.api.jobs import JobRunner
        from lending_agent.api.sessions import SessionStore

        svc, store = _service(tmp_path, fake_embedding_model)
        sessions, runner = SessionStore(), JobRunner()
        return TestClient(create_app(ChatService(store, None, sessions), svc, runner, sessions)), runner

    def test_deleting_dot_dot_over_http_is_a_400_not_a_500(self, tmp_path, fake_embedding_model):
        client, runner = self._client(tmp_path, fake_embedding_model)
        try:
            assert client.delete("/documents/%2e%2e").status_code == 400
        finally:
            runner.shutdown()

    def test_the_home_page_carries_security_headers(self, tmp_path, fake_embedding_model):
        client, runner = self._client(tmp_path, fake_embedding_model)
        try:
            headers = client.get("/").headers
        finally:
            runner.shutdown()
        assert headers["x-frame-options"] == "DENY" and headers["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in headers["content-security-policy"]


class TestEnvExample:
    def test_it_holds_the_settings_the_code_reads(self):
        text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        for expected in ("EMBEDDING_MODEL=", "CHROMA_PERSIST_DIR=", "OLLAMA_TIMEOUT=300", "API_HOST=", "MAX_UPLOAD_MB="):
            assert expected in text
        assert "OLLAMA_EMBEDDING_MODEL" not in text
