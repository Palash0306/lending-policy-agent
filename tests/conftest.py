"""
Shared pytest fixtures for ingestion tests: build small sample .pdf and
.docx files on the fly, so tests don't depend on any external sample file
and can run anywhere (including CI). The sample text deliberately uses the
same *implicit* boundary/date-cutoff phrasing the real agent needs to
handle later (Phase 5) -- "shall not exceed", "on or after" -- rather than
literal words like "maximum" or "cutoff".
"""
from __future__ import annotations

import hashlib

import pytest

SAMPLE_POLICY_TEXT = [
    "Section 4: Loan Amount Limits",
    "The sanctioned loan amount shall not be less than $5,000 and shall not "
    "exceed $500,000 for any individual retail borrower.",
    "Section 7: Eligibility Effective Dates",
    "This eligibility criterion applies to applications submitted on or after "
    "January 1, 2026. Applications submitted prior to this date are "
    "grandfathered under the previous policy.",
    "Section 9: Repayment Period",
    "The repayment period shall not extend beyond thirty (30) years from the "
    "date of disbursement.",
]


@pytest.fixture
def sample_pdf_path(tmp_path):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)

    # new_x="LMARGIN": multi_cell's default (new_x="RIGHT") leaves the cursor
    # at the right edge after a call, which starves the *next* multi_cell of
    # horizontal space. Explicitly returning to the left margin between
    # lines avoids that.
    pdf.add_page()
    for line in SAMPLE_POLICY_TEXT[:3]:
        pdf.multi_cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")

    pdf.add_page()
    for line in SAMPLE_POLICY_TEXT[3:]:
        pdf.multi_cell(0, 8, line, new_x="LMARGIN", new_y="NEXT")

    path = tmp_path / "sample_policy.pdf"
    pdf.output(str(path))
    return str(path)


@pytest.fixture
def sample_docx_path(tmp_path):
    from docx import Document

    doc = Document()
    for line in SAMPLE_POLICY_TEXT:
        doc.add_paragraph(line)

    table = doc.add_table(rows=3, cols=2)
    table.rows[0].cells[0].text = "Tier"
    table.rows[0].cells[1].text = "Maximum Loan-to-Value Ratio"
    table.rows[1].cells[0].text = "Standard"
    table.rows[1].cells[1].text = "80%"
    table.rows[2].cells[0].text = "Preferred"
    table.rows[2].cells[1].text = "90%"

    path = tmp_path / "sample_policy.docx"
    doc.save(str(path))
    return str(path)


class FakeEmbeddingModel:
    """
    Deterministic, fully offline stand-in for the real sentence-transformers
    model, used by vector-store/indexer tests (Phase 3): no network, no
    model download, but texts that share words still end up with closer
    vectors than unrelated texts -- enough to test that search returns
    genuinely relevant results, not just "some" results.
    """

    DIMENSION = 32

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.DIMENSION
        for word in text.lower().split():
            idx = int(hashlib.sha1(word.encode()).hexdigest(), 16) % self.DIMENSION
            vec[idx] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed(self, texts):
        return [self._vector(t) for t in texts]

    def embed_one(self, text):
        return self._vector(text)


@pytest.fixture
def fake_embedding_model():
    return FakeEmbeddingModel()

