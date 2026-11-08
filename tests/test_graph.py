"""Phase 7 tests: the agent graph's routes and loops. The model is scripted; no Ollama needed."""
from __future__ import annotations

import json

from lending_agent.agent.graph import (
    ROUTE_ERROR, ROUTE_NEEDS_REVIEW, ROUTE_NOT_FOUND, ROUTE_VERIFIED, AgentConfig, ask, build_graph,
)
from lending_agent.indexing.vector_store import SearchResult

QUOTE = "The repayment period shall not extend beyond thirty (30) years from disbursement."


class FakeSearcher:
    def __init__(self, results):
        self.results, self.sizes = results, []

    def search(self, query, n_results=5, source_file=None):
        self.sizes.append(n_results)
        return list(self.results)


class ScriptedLLM:
    """Hands out one reply per call, in order. An Exception instance is raised instead of returned."""

    def __init__(self, *replies):
        self.replies, self.prompts = list(replies), []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def hit(chunk_id="c1", text=QUOTE, source="a.pdf", page=12):
    return SearchResult(chunk_id=chunk_id, text=text, source_file=source, page_number=page, doc_type="pdf", distance=0.2)


def finding(value=30, quote=QUOTE, evidence_id=1, **overrides):
    item = {"kind": "numeric_boundary", "bound_type": "maximum", "value": value, "unit": "years", "date": None,
            "date_direction": None, "applies_to": None, "classification": "firm_rule", "evidence_id": evidence_id, "quote": quote}
    item.update(overrides)
    return item


def reply(*items):
    return json.dumps({"findings": list(items)})


def run(llm, results=None, **config):
    searcher = FakeSearcher(results if results is not None else [hit()])
    graph = build_graph(searcher, llm=llm, config=AgentConfig(use_llm_expansion=False, **config))
    return ask(graph, "loan_tenure"), searcher


class TestHappyPath:
    def test_a_grounded_finding_is_verified_after_one_attempt(self):
        answer, _ = run(ScriptedLLM(reply(finding())))
        assert answer.route == ROUTE_VERIFIED
        assert answer.attempts == 1 and not answer.widened
        assert "maximum 30 years" in answer.summary and "a.pdf" in answer.summary

    def test_the_trace_lists_each_node_in_order_with_timings(self):
        answer, _ = run(ScriptedLLM(reply(finding())))
        assert [t["node"] for t in answer.trace] == ["expand", "retrieve", "extract", "verify", "respond"]
        assert all(t["seconds"] >= 0 and t["at"] for t in answer.trace)

    def test_the_answer_is_json_serialisable(self):
        answer, _ = run(ScriptedLLM(reply(finding())))
        data = json.loads(json.dumps(answer.to_dict()))
        assert data["route"] == "verified" and data["result"]["verified"] is True


class TestWidening:
    def test_nothing_found_widens_the_search_once_and_can_then_succeed(self):
        llm = ScriptedLLM(reply(), reply(finding()))
        answer, searcher = run(llm, top_k=3)
        assert answer.route == ROUTE_VERIFIED and answer.widened and answer.attempts == 2
        assert searcher.sizes[0] == 3 and searcher.sizes[-1] == 6

    def test_still_nothing_after_widening_is_not_found(self):
        llm = ScriptedLLM(reply(), reply())
        answer, _ = run(llm)
        assert answer.route == ROUTE_NOT_FOUND and answer.widened
        assert len(llm.prompts) == 2

    def test_widening_can_be_turned_off(self):
        llm = ScriptedLLM(reply())
        answer, _ = run(llm, widen_factor=1)
        assert answer.route == ROUTE_NOT_FOUND and not answer.widened
        assert len(llm.prompts) == 1


class TestRetryWithFeedback:
    def test_an_ungrounded_finding_is_retried_with_the_problem_in_the_prompt(self):
        llm = ScriptedLLM(reply(finding(value=99)), reply(finding()))
        answer, _ = run(llm)
        assert answer.route == ROUTE_VERIFIED and answer.attempts == 2
        assert "previous answer had these problems" not in llm.prompts[0]
        assert "previous answer had these problems" in llm.prompts[1]
        assert "number 99 was not found" in llm.prompts[1]

    def test_a_finding_that_stays_ungrounded_goes_to_human_review(self):
        llm = ScriptedLLM(reply(finding(value=99)), reply(finding(value=99)))
        answer, _ = run(llm)
        assert answer.route == ROUTE_NEEDS_REVIEW and answer.attempts == 2
        assert "NOT grounded" in answer.summary

    def test_attempts_are_capped(self):
        llm = ScriptedLLM(reply(finding(value=99)))
        answer, _ = run(llm, max_attempts=1)
        assert answer.route == ROUTE_NEEDS_REVIEW and len(llm.prompts) == 1

    def test_a_reply_that_is_not_json_is_retried_once(self):
        llm = ScriptedLLM("sorry, no idea", reply(finding()))
        answer, _ = run(llm)
        assert answer.route == ROUTE_VERIFIED and answer.attempts == 2
        assert "not valid JSON" in llm.prompts[1]

    def test_two_bad_replies_in_a_row_end_as_an_error(self):
        answer, _ = run(ScriptedLLM("nope", "still nope"))
        assert answer.route == ROUTE_ERROR and "not valid JSON" in answer.summary


class TestErrorsAndReview:
    def test_ollama_being_down_is_an_error_and_is_not_retried(self):
        llm = ScriptedLLM(ConnectionError("ollama down"))
        answer, _ = run(llm)
        assert answer.route == ROUTE_ERROR and "ollama down" in answer.summary
        assert len(llm.prompts) == 1

    def test_conflicting_values_go_to_human_review(self):
        other = "The repayment period shall not extend beyond twenty five (25) years from disbursement."
        results = [hit("c1"), hit("c2", text=other, source="b.pdf", page=3)]
        llm = ScriptedLLM(reply(finding(), finding(value=25, quote=other, evidence_id=2)))
        answer, _ = run(llm, results=results)
        assert answer.route == ROUTE_NEEDS_REVIEW
        assert len(answer.result.conflicts) == 1

    def test_no_paragraph_with_a_number_means_not_found_without_asking_the_model(self):
        llm = ScriptedLLM(reply(finding()))
        answer, _ = run(llm, results=[hit(text="Banks should keep sound underwriting standards.")], widen_factor=1)
        assert answer.route == ROUTE_NOT_FOUND and llm.prompts == []
