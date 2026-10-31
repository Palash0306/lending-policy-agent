"""
Phase 4 tests for expand_field_terms. The LLM is a scripted fake, so these
run with no Ollama server and are fully deterministic.
"""
from __future__ import annotations

import json

import pytest

from lending_agent.agent.field_expansion import expand_field_terms


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = 0
        self.last_kwargs = {}

    def generate(self, prompt, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self.error:
            raise self.error
        return self.reply


def _reply(field, terms):
    return json.dumps({"field": field, "terms": terms})


class TestKnownField:
    def test_merges_original_seed_and_llm_terms_without_duplicates(self):
        llm = FakeLLM(_reply("loan_amount", ["Loan Amount", "advance", "credit line"]))
        result = expand_field_terms("loan_amount", llm=llm)
        assert result.field == "loan_amount"
        assert result.source == "seed+llm"
        assert result.terms[0] == "loan amount"          # the original phrasing leads
        assert "sanctioned amount" in result.terms       # from the built-in list
        assert "credit line" in result.terms             # from the LLM
        lowered = [t.lower() for t in result.terms]
        assert len(lowered) == len(set(lowered))         # "Loan Amount" was not repeated

    def test_asks_the_llm_for_json_output(self):
        llm = FakeLLM(_reply("loan_amount", ["advance"]))
        expand_field_terms("loan_amount", llm=llm)
        assert llm.last_kwargs["format"] == "json"

    def test_respects_max_terms(self):
        llm = FakeLLM(_reply("loan_amount", [f"phrase {chr(97 + i)}" for i in range(10)]))
        result = expand_field_terms("loan_amount", llm=llm, max_terms=5)
        assert len(result.terms) == 5


class TestMessyLLMOutput:
    def test_accepts_json_wrapped_in_code_fences(self):
        fenced = "```json\n" + _reply("loan_amount", ["advance"]) + "\n```"
        result = expand_field_terms("loan_amount", llm=FakeLLM(fenced))
        assert "advance" in result.terms

    def test_accepts_json_surrounded_by_chatter(self):
        chatty = "Sure! Here you go: " + _reply("loan_amount", ["advance"]) + " Hope that helps."
        result = expand_field_terms("loan_amount", llm=FakeLLM(chatty))
        assert "advance" in result.terms

    def test_drops_phrases_containing_numbers(self):
        llm = FakeLLM(_reply("loan_amount", ["advance", "limit of 500000", "tier 2 amount"]))
        result = expand_field_terms("loan_amount", llm=llm)
        assert "advance" in result.terms
        assert not any(ch.isdigit() for t in result.terms for ch in t)

    def test_ignores_non_text_and_overlong_items(self):
        llm = FakeLLM(_reply("loan_amount", ["advance", 42, None, "x" * 80, "one two three four five six seven"]))
        result = expand_field_terms("loan_amount", llm=llm)
        assert "advance" in result.terms
        assert all(isinstance(t, str) and len(t) <= 60 for t in result.terms)


class TestFallbacks:
    def test_garbage_reply_falls_back_to_builtin_list_and_reports_it(self):
        result = expand_field_terms("loan_amount", llm=FakeLLM("I cannot help with that."))
        assert result.source == "seed"
        assert "sanctioned amount" in result.terms
        assert result.llm_error is not None

    def test_llm_failure_falls_back_to_builtin_list_and_reports_it(self):
        result = expand_field_terms("loan_amount", llm=FakeLLM(error=ConnectionError("ollama down")))
        assert result.source == "seed"
        assert "ollama down" in result.llm_error

    def test_unknown_field_and_no_llm_returns_just_the_input(self):
        result = expand_field_terms("collateral_haircut", use_llm=False)
        assert result.field == "collateral_haircut"
        assert result.terms == ["collateral haircut"]
        assert result.source == "input"

    def test_use_llm_false_never_calls_the_llm(self):
        llm = FakeLLM(_reply("loan_amount", ["advance"]))
        result = expand_field_terms("loan_amount", llm=llm, use_llm=False)
        assert llm.calls == 0
        assert result.source == "seed"


class TestNaturalLanguageQuestions:
    def test_recognises_a_known_field_inside_a_question(self):
        result = expand_field_terms("What is the maximum loan tenure allowed?", use_llm=False)
        assert result.field == "loan_tenure"
        assert "repayment period" in result.terms
        assert result.terms[0] == "What is the maximum loan tenure allowed?"

    def test_unknown_field_in_a_question_uses_the_llms_field_name(self):
        llm = FakeLLM(_reply("collateral_haircut", ["valuation discount", "haircut"]))
        result = expand_field_terms("What haircut applies to collateral?", llm=llm)
        assert result.field == "collateral_haircut"
        assert "valuation discount" in result.terms
        assert result.source == "llm"

    def test_unknown_field_with_no_llm_help_gets_a_made_up_slug_from_the_question(self):
        result = expand_field_terms("What haircut applies to collateral", use_llm=False)
        assert result.field == "what_haircut_applies_to"


class TestBadInput:
    @pytest.mark.parametrize("bad", ["", "   "])
    def test_empty_query_is_rejected(self, bad):
        with pytest.raises(ValueError):
            expand_field_terms(bad, use_llm=False)
