"""Phase 8: uploading, indexing, listing and removing documents (no web framework, fake store)."""
from __future__ import annotations

import io
from pathlib import Path

import pytest
from docx import Document

from lending_agent.api.documents import DocumentService, looks_like, safe_filename
from lending_agent.api.errors import ApiError
from tests.api_helpers import FakeStore

PDF = b"%PDF-1.4\n" + b"x" * 100
ZIP = b"PK\x03\x04" + b"x" * 100


def _service(tmp_path, store=None, index_fn=None, max_bytes=10_000):
    return DocumentService(store or FakeStore(), tmp_path / "up", max_bytes, index_fn=index_fn)


def _docx_bytes(text="The repayment period shall not extend beyond thirty (30) years from disbursement."):
    buffer = io.BytesIO()
    doc = Document()
    doc.add_paragraph(text)
    doc.save(buffer)
    return buffer.getvalue()


class TestSafeFilename:
    @pytest.mark.parametrize("raw,expected", [
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\me\\Policy.PDF", "Policy.pdf"),
        ("my policy (v2).docx", "my policy (v2).docx"),
        ("weird*name?.docx", "weird_name_.docx"),
    ])
    def test_only_a_harmless_file_name_is_kept(self, raw, expected):
        assert safe_filename(raw) == expected

    @pytest.mark.parametrize("raw", ["notes.txt", "", ".pdf", "noextension", "evil.exe", "a.pdf.exe"])
    def test_other_types_and_empty_names_are_refused(self, raw):
        with pytest.raises(ApiError) as exc:
            safe_filename(raw)
        assert exc.value.status == 400

    def test_looks_like(self):
        assert looks_like("a.pdf", PDF) and not looks_like("a.pdf", ZIP)
        assert looks_like("a.docx", ZIP) and not looks_like("a.docx", PDF)


class TestSaveUpload:
    def test_a_good_file_is_saved_under_its_clean_name(self, tmp_path):
        svc = _service(tmp_path)
        path = svc.save_upload("../../sneaky.pdf", io.BytesIO(PDF))
        assert path == tmp_path / "up" / "sneaky.pdf" and path.read_bytes() == PDF
        assert not (tmp_path / "sneaky.pdf").exists()

    @pytest.mark.parametrize("data,status", [(b"", 400), (b"just text, not a pdf", 400), (b"%PDF" + b"x" * 20_000, 413)])
    def test_bad_files_are_refused_and_leave_nothing_behind(self, tmp_path, data, status):
        svc = _service(tmp_path, max_bytes=10_000)
        with pytest.raises(ApiError) as exc:
            svc.save_upload("a.pdf", io.BytesIO(data))
        assert exc.value.status == status
        assert list((tmp_path / "up").iterdir()) == []             # no half-written file either

    def test_a_docx_must_be_a_zip(self, tmp_path):
        with pytest.raises(ApiError):
            _service(tmp_path).save_upload("a.docx", io.BytesIO(PDF))


class TestIndexing:
    def test_a_real_docx_is_chunked_and_indexed(self, tmp_path):
        svc, store = _service(tmp_path, max_bytes=1_000_000), FakeStore()
        svc.store = store
        path = svc.save_upload("policy.docx", io.BytesIO(_docx_bytes()))
        result = svc.index_files([path])
        assert result["indexed"] == [{"file": "policy.docx", "chunks": len(store.chunks)}] and result["failed"] == []
        assert result["documents"] == [{"name": "policy.docx", "chunks": len(store.chunks)}]
        assert "thirty (30) years" in store.chunks[0].text and store.chunks[0].source_file == "policy.docx"

    def test_one_bad_file_does_not_stop_the_others_and_is_removed(self, tmp_path):
        def index(paths, store):
            if Path(paths[0]).name == "bad.pdf":
                raise ValueError("cannot parse")
            return {paths[0]: 3}

        svc = _service(tmp_path, index_fn=index)
        good, bad = svc.save_upload("good.pdf", io.BytesIO(PDF)), svc.save_upload("bad.pdf", io.BytesIO(PDF))
        result = svc.index_files([bad, good])
        assert result["indexed"] == [{"file": "good.pdf", "chunks": 3}]
        assert result["failed"] == [{"file": "bad.pdf", "error": "cannot parse"}]
        assert good.exists() and not bad.exists()

    def test_a_file_with_no_readable_text_is_reported_not_silently_accepted(self, tmp_path):
        svc = _service(tmp_path, index_fn=lambda paths, store: {paths[0]: 0})
        result = svc.index_files([svc.save_upload("scan.pdf", io.BytesIO(PDF))])
        assert result["indexed"] == [] and "OCR" in result["failed"][0]["error"]

    def test_uploading_the_same_name_again_replaces_the_old_chunks_first(self, tmp_path):
        store = FakeStore()
        svc = _service(tmp_path, store=store, max_bytes=1_000_000)
        svc.index_files([svc.save_upload("p.docx", io.BytesIO(_docx_bytes("Old text. The limit is 10 years.")))])
        first = len(store.chunks)
        svc.index_files([svc.save_upload("p.docx", io.BytesIO(_docx_bytes("New text. The limit is 20 years.")))])
        assert store.calls == [("delete", "p.docx"), ("delete", "p.docx")] and len(store.chunks) == first
        assert "20 years" in store.chunks[0].text and not any("10 years" in c.text for c in store.chunks)


class TestListAndDelete:
    def test_the_listing_is_empty_until_refreshed(self, tmp_path):
        assert _service(tmp_path).listing() == {"documents": [], "total_chunks": 0}

    def test_refresh_reads_the_store(self, tmp_path):
        svc, store = _service(tmp_path), FakeStore()
        svc.store = store
        store.chunks = [type("C", (), {"source_file": "a.pdf"})(), type("C", (), {"source_file": "a.pdf"})()]
        svc.refresh()
        assert svc.listing() == {"documents": [{"name": "a.pdf", "chunks": 2}], "total_chunks": 2}

    def test_delete_removes_chunks_and_the_saved_file(self, tmp_path):
        store = FakeStore()
        svc = _service(tmp_path, store=store, max_bytes=1_000_000)
        path = svc.save_upload("p.docx", io.BytesIO(_docx_bytes()))
        svc.index_files([path])
        result = svc.delete("p.docx")
        assert result["chunks_removed"] > 0 and result["documents"] == [] and not Path(path).exists()

    def test_deleting_an_unknown_document_is_a_404(self, tmp_path):
        with pytest.raises(ApiError) as exc:
            _service(tmp_path).delete("ghost.pdf")
        assert exc.value.status == 404

    def test_delete_ignores_any_path_in_the_name(self, tmp_path):
        svc = _service(tmp_path)
        outside = tmp_path / "keep.pdf"
        outside.write_bytes(PDF)
        with pytest.raises(ApiError):
            svc.delete("../keep.pdf")
        assert outside.exists()
