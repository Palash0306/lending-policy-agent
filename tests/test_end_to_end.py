"""
One end-to-end test of the whole product (no Ollama): a real PDF and a real Word file are uploaded through the API,
split, embedded (fake embedder) and indexed in a real Chroma store; a batch of three fields is run through the REAL
agent graph with a scripted model; the result is exported as ONE Excel file, whose test values are checked.

Skipped (not failed) when langgraph, fastapi, fpdf2 or chromadb are not installed.
"""
from __future__ import annotations

import io
import json
import re
import time

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("fpdf")
pytest.importorskip("chromadb")
try:
    import python_multipart  # noqa: F401
except ImportError:
    pytest.importorskip("multipart")

import openpyxl  # noqa: E402
from docx import Document  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from fpdf import FPDF  # noqa: E402

from lending_agent.api.app import create_app  # noqa: E402
from lending_agent.api.batches import BatchManager  # noqa: E402
from lending_agent.api.chat import ChatService  # noqa: E402
from lending_agent.api.documents import DocumentService  # noqa: E402
from lending_agent.api.jobs import JobRunner  # noqa: E402
from lending_agent.api.sessions import SessionStore  # noqa: E402
from lending_agent.config import ChromaSettings  # noqa: E402
from lending_agent.indexing.vector_store import VectorStore  # noqa: E402

TENURE = "The repayment period shall not extend beyond thirty (30) years from disbursement."
AMOUNT = "The sanctioned loan amount shall not be less than $5,000 and shall not exceed $500,000."
EFFECTIVE = "This criterion applies to applications submitted on or after January 1, 2026."
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _finding(**kw):
    base = {"kind": "numeric_boundary", "bound_type": None, "value": None, "unit": None, "date": None, "date_precision": None,
            "date_direction": None, "applies_to": None, "classification": "firm_rule"}
    return {**base, **kw}


class ScriptedModel:
    """Plays the model: suggests phrasings, and 'reads' the numbered extracts it is shown."""

    FINDINGS = {
        "loan_tenure": [(TENURE, _finding(bound_type="maximum", value=30, unit="years"))],
        "loan_amount": [(AMOUNT, _finding(bound_type="minimum", value=5000, unit="USD")),
                        (AMOUNT, _finding(bound_type="maximum", value=500000, unit="USD"))],
        "effective_date": [(EFFECTIVE, _finding(kind="date_cutoff", date="2026-01-01", date_precision="day", date_direction="on_or_after"))],
    }

    def generate(self, prompt, **kwargs):
        if "EXTRACTS:" not in prompt:
            return json.dumps({"terms": ["repayment period", "sanctioned amount"]})
        name = re.search(r'the field "([^"]+)"', prompt).group(1)
        extracts = re.findall(r"^\[(\d+)\] (.+?)(?=\n\n\[\d+\] |\n\nWHAT TO LOOK FOR)", prompt, re.S | re.M)
        out = []
        for sentence, template in self.FINDINGS.get(name, []):
            for number, text in extracts:
                if sentence in text:
                    out.append({**template, "evidence_id": int(number), "quote": sentence})
                    break
        return json.dumps({"findings": out})


def _pdf() -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    pdf.add_page()
    pdf.multi_cell(0, 8, "Section 4. Terms of credit.", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 8, TENURE, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def _docx() -> bytes:
    document, buffer = Document(), io.BytesIO()
    document.add_paragraph(AMOUNT)
    document.add_paragraph(EFFECTIVE)
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def client(tmp_path, fake_embedding_model):
    store = VectorStore(settings=ChromaSettings(persist_directory=str(tmp_path / "chroma"), collection_name="e2e"), embedding_model=fake_embedding_model)
    sessions, runner, model = SessionStore(), JobRunner(), ScriptedModel()
    documents = DocumentService(store, tmp_path / "uploads", 10_000_000)                 # the real indexing path
    meta = lambda: {"llm_model": "scripted", "embedding_model": "fake", "documents": [d["name"] for d in documents.listing()["documents"]]}  # noqa: E731
    app = create_app(ChatService(store, model, sessions), documents, runner, sessions, BatchManager(tmp_path / "batches"), meta)
    with TestClient(app) as test_client:
        yield test_client
    runner.shutdown()


def test_documents_to_one_excel_file(client):
    # 1. upload a real PDF and a real Word file
    upload = client.post("/documents", files=[("files", ("policy.pdf", _pdf(), "application/pdf")),
                                              ("files", ("rules.docx", _docx(), "application/octet-stream"))])
    assert upload.status_code == 200 and not upload.json()["failed"]
    assert {d["name"] for d in upload.json()["documents"]} == {"policy.pdf", "rules.docx"}

    # 2. run a batch of three fields through the real agent graph
    accepted = client.post("/batches", json={"fields": ["loan_tenure", "loan_amount", "effective_date"]})
    assert accepted.status_code == 202
    batch_id, end = accepted.json()["batch_id"], time.monotonic() + 60
    while time.monotonic() < end:
        status = client.get(f"/batches/{batch_id}").json()
        if status["status"] in ("done", "cancelled", "interrupted"):
            break
        time.sleep(0.05)
    assert status["status"] == "done" and [i["route"] for i in status["items"]] == ["verified"] * 3
    assert [i["headline"] for i in status["items"]][0] == "maximum 30 years"

    # 3. ONE workbook with all three fields
    response = client.get(f"/batches/{batch_id}/export")
    assert response.status_code == 200 and response.headers["content-type"].startswith(XLSX)
    wb = openpyxl.load_workbook(io.BytesIO(response.content))
    assert wb.sheetnames == ["Summary", "Test values", "Findings", "Search log", "Read me"]
    summary = {r[0]: r for r in wb["Summary"].iter_rows(values_only=True) if r[0] in ("loan_tenure", "loan_amount", "effective_date")}
    assert [summary[f][1] for f in ("loan_tenure", "loan_amount", "effective_date")] == ["Verified"] * 3
    assert summary["loan_tenure"][3:5] == ("policy.pdf", 1)

    rows = [r for r in wb["Test values"].iter_rows(min_row=2, values_only=True) if r[0]]
    by_field = {f: [(r[2], r[3], r[5]) for r in rows if r[0] == f] for f in ("loan_tenure", "loan_amount", "effective_date")}
    assert by_field["loan_tenure"] == [("Max - 1", 29, "Accept"), ("Max", 30, "Accept"), ("Max + 1", 31, "Reject")]
    assert by_field["loan_amount"] == [("Min - 1", 4999, "Reject"), ("Min", 5000, "Accept"), ("Min + 1", 5001, "Accept"),
                                       ("Max - 1", 499999, "Accept"), ("Max", 500000, "Accept"), ("Max + 1", 500001, "Reject")]
    assert [(c, v.date().isoformat(), e) for c, v, e in by_field["effective_date"]] == [
        ("Day before cutoff", "2025-12-31", "Rule does not apply"), ("Cutoff day", "2026-01-01", "Rule applies"),
        ("Day after cutoff", "2026-01-02", "Rule applies")]
    quotes = {r[0]: r[9] for r in rows}
    assert quotes["loan_tenure"] == TENURE and quotes["effective_date"] == EFFECTIVE
    assert {r[7] for r in rows if r[0] == "loan_amount"} == {"rules.docx"}

    # 4. the search log shows what was searched and read
    log = [c for r in wb["Search log"].iter_rows(values_only=True) for c in r if c is not None]
    assert any("repayment period" in str(c) for c in log) and "policy.pdf" in log
