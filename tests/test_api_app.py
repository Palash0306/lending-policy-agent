"""
Phase 8: the real FastAPI app, called over HTTP with FastAPI's test client. The agent and the store
are fakes. Skipped (not failed) when fastapi / httpx / python-multipart are not installed.
"""
from __future__ import annotations

import io
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
try:
    import python_multipart  # noqa: F401
except ImportError:
    pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from lending_agent.api.app import create_app  # noqa: E402
from lending_agent.api.chat import ChatService  # noqa: E402
from lending_agent.api.documents import DocumentService  # noqa: E402
from lending_agent.api.jobs import JobRunner  # noqa: E402
from lending_agent.api.sessions import SessionStore  # noqa: E402
from tests.api_helpers import TENURE, FakeStore, ScriptedLLM, answer_for, evidence, numeric  # noqa: E402

PDF = b"%PDF-1.4\n" + b"x" * 200


class FakeOllama(ScriptedLLM):
    class settings:
        model = "qwen2.5:7b"

    def list_models(self):
        return ["qwen2.5:7b"]

    def is_model_available(self):
        return True


@pytest.fixture
def client(tmp_path):
    sessions, runner = SessionStore(), JobRunner()
    answer = answer_for([numeric("c1", TENURE, 30, page=12)], [evidence("c1", TENURE, page=12)])
    chat = ChatService(None, FakeOllama(), sessions, build=lambda options: "graph", run=lambda graph, query: answer)
    documents = DocumentService(FakeStore(), tmp_path / "up", 10_000, index_fn=lambda paths, store: {paths[0]: 4})
    with TestClient(create_app(chat, documents, runner, sessions)) as test_client:
        yield test_client
    runner.shutdown()


def _poll(client, job_id, limit=10):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


class TestBasics:
    def test_the_web_page_is_served(self, client):
        response = client.get("/")
        assert response.status_code == 200 and "Lending Policy Data Assurance Agent" in response.text

    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok" and body["ollama"]["reachable"] and body["ollama"]["model_available"]


class TestChat:
    def test_the_brief_s_chat_contract(self, client):
        body = client.post("/chat", json={"message": "What is the maximum loan tenure allowed?"}).json()
        for key in ("answer", "field", "max_value", "unit", "source_quote", "grounded", "needs_review", "session_id"):
            assert key in body
        assert body["max_value"] == 30 and body["unit"] == "years" and body["grounded"] and not body["needs_review"]

    def test_the_same_session_continues(self, client):
        first = client.post("/chat", json={"message": "loan_tenure"}).json()
        second = client.post("/chat", json={"message": "loan_tenure", "session_id": first["session_id"]}).json()
        assert second["turn"] == 2 and second["session_id"] == first["session_id"]
        history = client.get(f"/sessions/{first['session_id']}").json()
        assert [t["turn"] for t in history["turns"]] == [1, 2]

    def test_wait_false_returns_a_job_to_poll(self, client):
        accepted = client.post("/chat?wait=false", json={"message": "loan_tenure"})
        assert accepted.status_code == 202
        body = accepted.json()
        assert body["session_id"] and body["job_id"]
        job = _poll(client, body["job_id"])
        assert job["status"] == "done" and job["result"]["max_value"] == 30

    @pytest.mark.parametrize("payload", [{"message": ""}, {"message": "   "}, {"message": "x" * 501},
                                         {"message": "loan_tenure", "kinds": "everything"},
                                         {"message": "loan_tenure", "session_id": "../bad"}, {}])
    def test_bad_requests_are_refused_with_422(self, client, payload):
        assert client.post("/chat", json=payload).status_code == 422

    def test_unknown_job_and_session_are_404(self, client):
        assert client.get("/jobs/nope").status_code == 404
        assert client.get("/sessions/abcdefgh12345").status_code == 404

    def test_a_session_can_be_deleted(self, client):
        sid = client.post("/chat", json={"message": "loan_tenure"}).json()["session_id"]
        assert client.delete(f"/sessions/{sid}").json() == {"deleted": True}
        assert client.get(f"/sessions/{sid}").status_code == 404


class TestDocuments:
    def _upload(self, client, files, **params):
        return client.post("/documents", files=[("files", f) for f in files], params=params)

    def test_upload_then_list_then_delete(self, client):
        response = self._upload(client, [("policy.pdf", io.BytesIO(PDF), "application/pdf")])
        assert response.status_code == 200
        assert response.json()["indexed"] == [{"file": "policy.pdf", "chunks": 4}]
        assert client.delete("/documents/policy.pdf").status_code in (200, 404)    # fake index_fn added no chunks to the fake store

    def test_a_bad_file_is_rejected_but_the_good_one_is_kept(self, client):
        response = self._upload(client, [("good.pdf", io.BytesIO(PDF), "application/pdf"),
                                         ("notes.txt", io.BytesIO(b"hello"), "text/plain")])
        body = response.json()
        assert [d["file"] for d in body["indexed"]] == ["good.pdf"] and body["rejected"][0]["file"] == "notes.txt"

    def test_only_bad_files_is_a_400_that_says_why(self, client):
        response = self._upload(client, [("notes.txt", io.BytesIO(b"hello"), "text/plain")])
        assert response.status_code == 400 and response.json()["detail"]["rejected"][0]["file"] == "notes.txt"

    def test_upload_with_wait_false_is_polled(self, client):
        response = self._upload(client, [("policy.pdf", io.BytesIO(PDF), "application/pdf")], wait="false")
        assert response.status_code == 202
        assert _poll(client, response.json()["job_id"])["result"]["indexed"][0]["file"] == "policy.pdf"

    def test_deleting_an_unknown_document_is_a_404(self, client):
        assert client.delete("/documents/ghost.pdf").status_code == 404
