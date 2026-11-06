"""Phase 8: the chat logic: sessions, follow-ups, error handling. The agent and the model are fakes."""
from __future__ import annotations

import json

import pytest

from lending_agent.api.chat import ChatOptions, ChatService, mentions_known_field, ollama_status, rewrite_follow_up
from lending_agent.api.errors import ApiError
from lending_agent.api.sessions import SessionStore
from tests.api_helpers import TENURE, ScriptedLLM, answer_for, evidence, not_found_answer, numeric


def _good_answer():
    return answer_for([numeric("c1", TENURE, 30, page=12)], [evidence("c1", TENURE, page=12)])


def _service(llm=None, answer=None, run=None):
    seen = {"queries": [], "options": []}

    def build(options):
        seen["options"].append(options)
        return "graph"

    def default_run(graph, query):
        seen["queries"].append(query)
        return answer or _good_answer()

    service = ChatService(searcher=None, llm=llm or ScriptedLLM(), sessions=SessionStore(), build=build,
                          run=run or default_run)
    return service, seen


def _rewrite(question):
    return json.dumps({"question": question})


class TestMentionsKnownField:
    @pytest.mark.parametrize("text", ["loan_tenure", "What is the maximum loan tenure allowed?", "what about the repayment period",
                                      "and the interest rate?", "show collateral_value please"])
    def test_stand_alone_messages(self, text):
        assert mentions_known_field(text)

    @pytest.mark.parametrize("text", ["what about for commercial loans specifically?", "and for new customers?", "why?"])
    def test_follow_ups_that_name_no_field(self, text):
        assert not mentions_known_field(text)


class TestFirstQuestion:
    def test_a_first_message_is_used_as_written_and_gets_a_session(self):
        service, seen = _service()
        out = service.chat("loan_tenure")
        assert seen["queries"] == ["loan_tenure"] and out["query_used"] == "loan_tenure" and not out["rewritten"]
        assert out["max_value"] == 30 and out["route"] == "verified" and out["turn"] == 1
        assert len(out["session_id"]) >= 8 and out["elapsed_seconds"] >= 0

    def test_the_turn_is_saved_in_the_session(self):
        service, _ = _service()
        out = service.chat("loan_tenure")
        session = service.sessions.get(out["session_id"])
        assert session.turns[0]["message"] == "loan_tenure" and session.turns[0]["payload"]["max_value"] == 30

    def test_the_first_message_never_calls_the_model_for_rewriting(self):
        llm = ScriptedLLM()                                   # would raise if called (no replies queued)
        _service(llm=llm)[0].chat("what about for commercial loans?")
        assert llm.prompts == []


class TestFollowUps:
    def test_a_follow_up_is_rewritten_using_the_earlier_question(self):
        llm = ScriptedLLM(_rewrite("What is the maximum loan tenure for commercial loans?"))
        service, seen = _service(llm=llm)
        first = service.chat("What is the maximum loan tenure allowed?")
        second = service.chat("what about for commercial loans specifically?", first["session_id"])
        assert seen["queries"][1] == "What is the maximum loan tenure for commercial loans?"
        assert second["rewritten"] and second["turn"] == 2 and second["session_id"] == first["session_id"]
        assert "What is the maximum loan tenure allowed?" in llm.prompts[0] and "loan_tenure" in llm.prompts[0]

    def test_a_message_that_names_a_field_is_not_rewritten(self):
        llm = ScriptedLLM()
        service, seen = _service(llm=llm)
        first = service.chat("loan_tenure")
        service.chat("and the loan amount?", first["session_id"])
        assert llm.prompts == [] and seen["queries"][1] == "and the loan amount?"

    @pytest.mark.parametrize("reply", [
        _rewrite("What is the maximum loan tenure when the loan is 500000 dollars?"),   # invented number
        "I think you mean loan tenure.",                                                 # not JSON
        json.dumps({"question": "x"}),                                                   # too short
        json.dumps({"question": "y" * 400}),                                             # too long
        json.dumps({"other": "key"}),
        RuntimeError("ollama down"),
    ])
    def test_an_unusable_rewrite_falls_back_to_plain_code(self, reply):
        service, seen = _service(llm=ScriptedLLM(reply))
        first = service.chat("loan_tenure")
        out = service.chat("what about for commercial loans?", first["session_id"])
        assert seen["queries"][1] == "loan tenure what about for commercial loans?" and out["rewritten"]

    def test_a_number_from_the_conversation_is_allowed_in_the_rewrite(self):
        previous = {"message": "max loan tenure when LTV is 90 percent", "payload": {"field": "loan_tenure", "query_used": "max loan tenure when LTV is 90 percent"}}
        question, rewritten = rewrite_follow_up("and for commercial?", previous,
                                                ScriptedLLM(_rewrite("What is the maximum loan tenure for commercial loans when LTV is 90 percent?")))
        assert "90 percent" in question and rewritten

    def test_a_rewrite_that_changes_nothing_is_reported_as_not_rewritten(self):
        previous = {"message": "q", "payload": {"field": "loan_tenure", "query_used": "q"}}
        question, rewritten = rewrite_follow_up("What is the cap for commercial loans?", previous,
                                                ScriptedLLM(_rewrite("What is the cap for commercial loans?")))
        assert question == "What is the cap for commercial loans?" and not rewritten


class TestOptionsAndErrors:
    def test_options_reach_the_agent(self):
        service, seen = _service()
        service.chat("loan_tenure", options=ChatOptions(source_file="a.pdf", kinds="date", fast=True))
        assert seen["options"] == [ChatOptions(source_file="a.pdf", kinds="date", fast=True)]

    @pytest.mark.parametrize("message", ["", "   ", "x" * 501])
    def test_bad_messages_are_refused_before_any_work(self, message):
        service, seen = _service()
        with pytest.raises(ApiError) as exc:
            service.chat(message)
        assert exc.value.status == 422 and seen["queries"] == [] and len(service.sessions) == 0

    def test_an_unknown_kind_is_refused(self):
        with pytest.raises(ApiError):
            _service()[0].chat("loan_tenure", options=ChatOptions(kinds="everything"))

    def test_a_crash_inside_the_agent_becomes_an_error_answer_and_the_session_survives(self):
        def boom(graph, query):
            raise RuntimeError("Could not load embedding model")

        service, _ = _service(run=boom)
        out = service.chat("loan_tenure")
        assert out["route"] == "error" and "Could not load embedding model" in out["answer"]
        assert out["query_used"] == "loan_tenure" and service.sessions.get(out["session_id"]).turns

    def test_not_found_is_a_normal_answer(self):
        out = _service(answer=not_found_answer())[0].chat("loan_tenure")
        assert out["route"] == "not_found" and out["max_value"] is None


class TestOllamaStatus:
    class Up:
        class settings:
            model = "qwen2.5:7b"

        def __init__(self, available=True):
            self.available = available

        def list_models(self):
            return ["qwen2.5:7b"]

        def is_model_available(self):
            return self.available

    class Down(Up):
        def list_models(self):
            raise ConnectionError("refused")

    def test_ready(self):
        assert ollama_status(self.Up()) == {"reachable": True, "model": "qwen2.5:7b", "model_available": True, "error": None}

    def test_model_not_installed(self):
        status = ollama_status(self.Up(available=False))
        assert status["reachable"] and not status["model_available"]

    def test_unreachable_never_raises(self):
        status = ollama_status(self.Down())
        assert not status["reachable"] and "refused" in status["error"]
        assert ollama_status(None)["error"]
