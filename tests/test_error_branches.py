"""
Phase 9 coverage work: the error and edge branches that had no test. Each test checks a behaviour a person would
notice (a clear error, a fallback, a label), not just that a line ran.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from lending_agent.agent.extraction import Finding
from lending_agent.api.summary import _number, build_payload
from lending_agent.config import ApiSettings, OllamaSettings, get_settings
from lending_agent.llm.ollama_client import OllamaClient, OllamaConnectionError, OllamaModelNotAvailableError
from tests.api_helpers import answer_for, evidence


@pytest.fixture
def settings():
    return OllamaSettings(host="http://localhost:11434", model="qwen2.5:7b")


def client_and_mock(settings):
    with patch("lending_agent.llm.ollama_client.ollama.Client") as mock_client:
        return OllamaClient(settings), mock_client.return_value


class TestOllamaClientErrors:
    def test_a_refused_pull_says_the_model_could_not_be_pulled(self, settings):
        from ollama import ResponseError

        client, mock = client_and_mock(settings)
        mock.list.return_value = {"models": []}
        mock.pull.side_effect = ResponseError("pull model manifest: file does not exist")
        with pytest.raises(OllamaModelNotAvailableError, match="Failed to pull model"):
            client.ensure_model_pulled("qwen2.5:7b")

    def test_a_dropped_connection_during_a_pull_is_a_connection_error(self, settings):
        client, mock = client_and_mock(settings)
        mock.list.return_value = {"models": []}
        mock.pull.side_effect = RuntimeError("connection reset")
        with pytest.raises(OllamaConnectionError, match="while pulling"):
            client.ensure_model_pulled("qwen2.5:7b")

    def test_a_failed_chat_is_a_connection_error(self, settings):
        client, mock = client_and_mock(settings)
        mock.chat.side_effect = RuntimeError("boom")
        with pytest.raises(OllamaConnectionError, match="Chat failed"):
            client.chat([{"role": "user", "content": "hi"}])

    def test_the_healthcheck_can_pull_a_missing_model_and_then_succeed(self, settings):
        client, mock = client_and_mock(settings)
        state = {"pulled": False}
        mock.list.side_effect = lambda: {"models": [{"model": "qwen2.5:7b"}] if state["pulled"] else []}
        mock.pull.side_effect = lambda **kw: (state.update(pulled=True), iter([{"status": "success"}]))[1]
        mock.generate.return_value = {"response": "OK"}
        result = client.healthcheck(auto_pull=True)
        assert result.reachable and result.model_available and result.error is None

    def test_the_healthcheck_reports_a_pull_that_fails(self, settings):
        client, mock = client_and_mock(settings)
        mock.list.return_value = {"models": []}
        mock.pull.side_effect = RuntimeError("no network")
        result = client.healthcheck(auto_pull=True)
        assert result.reachable and not result.model_available and "no network" in result.error

    def test_the_healthcheck_reports_a_model_that_cannot_answer(self, settings):
        client, mock = client_and_mock(settings)
        mock.list.return_value = {"models": [{"model": "qwen2.5:7b"}]}
        mock.generate.side_effect = RuntimeError("out of memory")
        result = client.healthcheck()
        assert result.reachable and result.model_available and "out of memory" in result.error


class TestSummaryEdges:
    @pytest.mark.parametrize("value,text", [(None, "?"), (30.0, "30"), (6.5, "6.5"), (500000, "500,000"), (1234.5, "1234.5")])
    def test_number_formatting(self, value, text):
        assert _number(value) == text

    def test_a_month_only_cutoff_is_labelled_in_the_answer_and_the_headline(self):
        quote = "This rule applies on or after January 2026."
        finding = Finding(kind="date_cutoff", classification="firm_rule", quote=quote, source_file="p.pdf", page_number=2, chunk_id="c1",
                          date="2026-01-01", date_direction="on_or_after", date_precision="month")
        payload = build_payload(answer_for([finding], [evidence("c1", quote, page=2)], "effective_date"))
        assert (payload["cutoff_date"], payload["date_direction"], payload["date_precision"]) == ("2026-01-01", "on_or_after", "month")
        assert "month only, day not stated" in payload["answer"] and payload["needs_review"]

    def test_a_date_that_applies_to_a_scope_says_so(self):
        quote = "For auto loans, this applies on or after January 1, 2026."
        finding = Finding(kind="date_cutoff", classification="firm_rule", quote=quote, source_file="p.pdf", page_number=2, chunk_id="c1",
                          date="2026-01-01", date_direction="on_or_after", date_precision="day", applies_to="auto loans")
        payload = build_payload(answer_for([finding], [evidence("c1", quote, page=2)], "effective_date"))
        assert "applies to: auto loans" in payload["answer"]


class TestConfig:
    def test_settings_include_the_api_block_with_safe_defaults(self):
        api = get_settings().api
        assert isinstance(api, ApiSettings) and api.host == "127.0.0.1" and api.max_batch_fields == 50
        assert api.batch_dir.endswith("batches") and api.max_upload_mb == 50


# ---- graph branches (these use the real graph, so they need langgraph) ---------------------------------------------
class TestGraphBranches:
    @pytest.fixture(autouse=True)
    def _needs_langgraph(self):
        pytest.importorskip("langgraph")

    def test_search_details_travel_with_the_answer(self):
        from tests.test_graph import ScriptedLLM, finding, hit, reply, run

        answer, _ = run(ScriptedLLM(reply(finding())), results=[hit()])
        search = answer.search
        assert search["terms"] and search["terms_source"] in ("seed", "input", "seed+llm", "llm")
        assert search["paragraphs"][0]["source_file"] == "a.pdf" and search["paragraphs"][0]["page_number"] == 12
        assert "thirty (30) years" in search["paragraphs"][0]["text"] and search["paragraphs_sent"] >= 1
        assert json.loads(json.dumps(answer.to_dict()))["search"]["terms"] == search["terms"]
        assert build_payload(answer)["search"]["paragraphs"][0]["chunk_id"] == "c1"

    def test_a_source_without_pages_is_described_as_such_after_a_retry(self):
        from tests.test_graph import ScriptedLLM, finding, hit, reply, run

        llm = ScriptedLLM(reply(finding(value=25)), reply(finding(value=30)))      # first claim is wrong, so the agent retries
        answer, _ = run(llm, results=[hit(page=None, source="rules.docx")], max_attempts=2)
        assert answer.attempts == 2 and answer.route == "verified"
        assert "rules.docx, no pagination" in answer.summary and "number 25 was not found" in llm.prompts[1]

    def test_the_default_model_client_is_created_when_none_is_given(self, monkeypatch):
        from lending_agent.agent.graph import AgentConfig, ask, build_graph
        from tests.test_graph import FakeSearcher, ScriptedLLM, finding, hit, reply

        scripted = ScriptedLLM(reply(finding()))
        monkeypatch.setattr("lending_agent.llm.ollama_client.OllamaClient", lambda: scripted)
        graph = build_graph(FakeSearcher([hit()]), config=AgentConfig(use_llm_expansion=False))
        assert ask(graph, "loan_tenure").route == "verified" and scripted.prompts
