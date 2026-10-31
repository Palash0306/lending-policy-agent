import pytest

from lending_agent.ingestion.loaders import (
    UnsupportedDocumentError,
    load_document,
    load_docx,
    load_pdf,
)


class TestLoadPdf:
    def test_extracts_text_across_pages(self, sample_pdf_path):
        doc = load_pdf(sample_pdf_path)
        assert doc.doc_type == "pdf"
        assert len(doc.pages) == 2
        assert "500,000" in doc.full_text
        assert "January 1, 2026" in doc.full_text

    def test_pages_are_numbered_from_one(self, sample_pdf_path):
        doc = load_pdf(sample_pdf_path)
        assert [p.page_number for p in doc.pages] == [1, 2]


class TestLoadDocx:
    def test_extracts_paragraph_text(self, sample_docx_path):
        doc = load_docx(sample_docx_path)
        assert doc.doc_type == "docx"
        assert len(doc.pages) == 1
        assert doc.pages[0].page_number is None
        assert "shall not exceed $500,000" in doc.full_text

    def test_extracts_table_text(self, sample_docx_path):
        doc = load_docx(sample_docx_path)
        assert "Maximum Loan-to-Value Ratio" in doc.full_text
        assert "80%" in doc.full_text
        assert "90%" in doc.full_text


class TestLoadDocumentDispatch:
    def test_dispatches_pdf(self, sample_pdf_path):
        assert load_document(sample_pdf_path).doc_type == "pdf"

    def test_dispatches_docx(self, sample_docx_path):
        assert load_document(sample_docx_path).doc_type == "docx"

    def test_raises_on_unsupported_extension(self, tmp_path):
        bogus = tmp_path / "policy.txt"
        bogus.write_text("hello")
        with pytest.raises(UnsupportedDocumentError):
            load_document(str(bogus))
