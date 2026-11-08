"""
Phase 9: the batch and export routes over HTTP (FastAPI's test client). Skipped when fastapi is not installed.
The agent is a fake; the exporter and the batch manager are the real ones.
"""
from __future__ import annotations

import io
import time

import openpyxl
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
try:
    import python_multipart  # noqa: F401
except ImportError:
    pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from lending_agent.api.app import create_app  # noqa: E402
from lending_agent.api.batches import BatchManager  # noqa: E402
from lending_agent.api.documents import DocumentService  # noqa: E402
from lending_agent.api.jobs import JobRunner  # noqa: E402
from lending_agent.api.sessions import SessionStore  # noqa: E402
from tests.api_helpers import FakeStore  # noqa: E402
from tests.test_batches import make_chat  # noqa: E402

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@pytest.fixture
def env(tmp_path):
    sessions, runner = SessionStore(), JobRunner()
    chat, calls = make_chat(fail_on={"loan_amount"})
    chat.sessions = sessions
    batches = BatchManager(tmp_path / "batches", max_fields=5)
    docs = DocumentService(FakeStore(), tmp_path / "up", 10_000, index_fn=lambda paths, store: {paths[0]: 1})
    meta = lambda: {"llm_model": "qwen2.5:7b", "embedding_model": "all-MiniLM-L6-v2", "documents": ["policy.pdf"]}  # noqa: E731
    with TestClient(create_app(chat, docs, runner, sessions, batches, meta)) as client:
        client.batches = batches                      # lets a test create a batch without running it
        yield client, calls, sessions
    runner.shutdown()


def wait_done(client, batch_id, limit=10):
    end = time.monotonic() + limit
    while time.monotonic() < end:
        body = client.get(f"/batches/{batch_id}").json()
        if body["status"] in ("done", "cancelled", "interrupted"):
            return body
        time.sleep(0.02)
    raise AssertionError("batch did not finish")


def workbook(response):
    return openpyxl.load_workbook(io.BytesIO(response.content))


class TestBatchRoutes:
    def test_run_three_fields_get_one_workbook(self, env):
        client, calls, _ = env
        accepted = client.post("/batches", json={"fields": ["loan_tenure\nloan_amount, applicant_age"]})
        assert accepted.status_code == 202 and accepted.json()["count"] == 3
        body = wait_done(client, accepted.json()["batch_id"])
        assert body["status"] == "done" and [i["route"] for i in body["items"]] == ["verified", "error", "not_found"]
        response = client.get(f"/batches/{body['batch_id']}/export")
        assert response.status_code == 200 and response.headers["content-type"].startswith(XLSX)
        assert "attachment" in response.headers["content-disposition"] and ".xlsx" in response.headers["content-disposition"]
        names = [r[0] for r in workbook(response)["Summary"].iter_rows(values_only=True) if r[0] in ("loan_tenure", "loan_amount", "applicant_age")]
        assert names == ["loan_tenure", "loan_amount", "applicant_age"] and calls == names

    def test_csv_export_and_options(self, env):
        client, _, _ = env
        batch_id = client.post("/batches", json={"fields": ["loan_tenure"]}).json()["batch_id"]
        wait_done(client, batch_id)
        response = client.get(f"/batches/{batch_id}/export", params={"format": "csv", "step": "5"})
        assert response.status_code == 200 and response.content.startswith(b"\xef\xbb\xbf") and "csv" in response.headers["content-type"]
        assert b"loan_tenure,maximum 30 years,Max - 1,25" in response.content

    @pytest.mark.parametrize("payload", [{"fields": []}, {"fields": ["", " , "]}, {"fields": [f"f{i}" for i in range(6)]},
                                         {"fields": ["x" * 501]}, {"fields": ["a"], "kinds": "everything"}, {}])
    def test_bad_requests_are_422(self, env, payload):
        assert env[0].post("/batches", json=payload).status_code == 422

    def test_list_results_cancel_delete(self, env):
        client, _, _ = env
        batch_id = client.post("/batches", json={"fields": ["loan_tenure", "applicant_age"]}).json()["batch_id"]
        wait_done(client, batch_id)
        assert [b["batch_id"] for b in client.get("/batches").json()["batches"]] == [batch_id]
        assert len(client.get(f"/batches/{batch_id}/results").json()["results"]) == 2
        assert client.post(f"/batches/{batch_id}/cancel").status_code == 409
        assert client.delete(f"/batches/{batch_id}").json() == {"deleted": batch_id}
        assert client.get(f"/batches/{batch_id}").status_code == 404

    def test_unknown_batches_are_404(self, env):
        client, _, _ = env
        for call in (client.get, client.delete, client.post):
            url = "/batches/nope" if call is not client.post else "/batches/nope/cancel"
            assert call(url).status_code == 404
        assert client.get("/batches/" + "0" * 32 + "/export").status_code == 404

    def test_export_before_anything_finished_is_404_with_a_reason(self, env):
        from lending_agent.api.chat import ChatOptions

        client = env[0]
        batch_id = client.batches.create("loan_tenure", ChatOptions())["batch_id"]          # created, never run
        response = client.get(f"/batches/{batch_id}/export")
        assert response.status_code == 404 and "nothing to export" in response.json()["detail"]


class TestSessionExports:
    def _chat(self, client, message="loan_tenure", session_id=None):
        return client.post("/chat", json={"message": message, "session_id": session_id}).json()

    def test_export_all_answers_in_a_conversation(self, env):
        client, _, _ = env
        first = self._chat(client)
        self._chat(client, "applicant_age", first["session_id"])
        response = client.get(f"/sessions/{first['session_id']}/export")
        assert response.status_code == 200 and response.headers["content-type"].startswith(XLSX)
        assert [r[0] for r in workbook(response)["Summary"].iter_rows(values_only=True) if r[0] in ("loan_tenure", "applicant_age")] == ["loan_tenure", "applicant_age"]

    def test_export_one_answer(self, env):
        client, _, _ = env
        first = self._chat(client)
        response = client.get(f"/sessions/{first['session_id']}/turns/{first['turn']}/export", params={"format": "csv"})
        assert response.status_code == 200 and b"Max - 1" in response.content

    def test_error_paths(self, env):
        client, _, _ = env
        first = self._chat(client)
        sid = first["session_id"]
        assert client.get("/sessions/abcdefgh12345/export").status_code == 404
        assert client.get(f"/sessions/{sid}/turns/99/export").status_code == 404
        assert client.get(f"/sessions/{sid}/export", params={"format": "pdf"}).status_code == 422
        for bad in ("abc", "0", "-1"):
            assert client.get(f"/sessions/{sid}/export", params={"step": bad}).status_code == 422
        assert client.get(f"/sessions/{sid}/export", params={"step": "0.5"}).status_code == 200
