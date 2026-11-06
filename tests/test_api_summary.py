"""Phase 8: turning the agent's answer into the chat JSON. Plain code; findings go through the real Phase 6 checks."""
from __future__ import annotations

from tests.api_helpers import (EXAMPLE, LTV_HIGH, LTV_LOW, TENURE, answer_for, error_answer, evidence, not_found_answer,
                               numeric)

from lending_agent.api.summary import build_payload


class TestFound:
    def test_a_verified_firm_rule_fills_the_headline(self):
        p = build_payload(answer_for([numeric("c1", TENURE, 30, page=12)], [evidence("c1", TENURE, page=12)]))
        assert (p["max_value"], p["min_value"], p["unit"]) == (30, None, "years")
        assert p["source_quote"] == TENURE and p["source_file"] == "policy.pdf" and p["page_number"] == 12
        assert p["route"] == "verified" and p["grounded"] and not p["needs_review"]
        assert "maximum 30 years" in p["answer"] and "policy.pdf, page 12" in p["answer"]

    def test_min_and_max_share_one_headline_and_big_numbers_get_commas(self):
        text = "The loan amount shall not be less than $5,000 and shall not exceed $500,000."
        p = build_payload(answer_for([numeric("c1", "shall not be less than $5,000", 5000, "minimum", "USD"),
                                      numeric("c1", "shall not exceed $500,000", 500000, "maximum", "USD")],
                                     [evidence("c1", text)], "loan_amount"))
        assert (p["min_value"], p["max_value"], p["unit"]) == (5000, 500000, "USD")
        assert "500,000" in p["answer"]

    def test_tiers_have_no_single_headline_but_every_value_is_listed(self):
        p = build_payload(answer_for(
            [numeric("c1", LTV_HIGH, 72, unit="months", applies_to="LTV below 90 percent"),
             numeric("c2", LTV_LOW, 60, unit="months", applies_to="LTV above 90 percent")],
            [evidence("c1", LTV_HIGH), evidence("c2", LTV_LOW)]))
        assert p["max_value"] is None and p["source_quote"] is None
        assert "72" in p["answer"] and "60" in p["answer"] and "no single headline value" in p["answer"]
        assert len(p["findings"]) == 2 and not p["needs_review"]


class TestReviewRules:
    def test_a_finding_that_fails_grounding_never_reaches_the_headline(self):
        p = build_payload(answer_for([numeric("c1", TENURE, 25)], [evidence("c1", TENURE)]))   # quote says 30
        assert p["max_value"] is None and not p["grounded"] and p["needs_review"] and p["route"] == "needs_review"
        assert any(r.startswith("finding #1:") for r in p["review_reasons"]) and len(p["findings"]) == 1

    def test_only_example_wording_is_sent_for_review_even_though_the_agent_said_verified(self):
        p = build_payload(answer_for([numeric("c1", EXAMPLE, 72, unit="months", classification="example")],
                                     [evidence("c1", EXAMPLE)]))
        assert p["agent_route"] == "verified"           # the agent's own route is kept...
        assert p["route"] == "needs_review" and p["needs_review"]   # ...but a tester must not treat an example as a rule
        assert p["max_value"] is None and "not a firm rule" in p["answer"]

    def test_a_firm_rule_next_to_an_example_keeps_the_headline(self):
        p = build_payload(answer_for([numeric("c1", TENURE, 30), numeric("c2", EXAMPLE, 72, unit="months", classification="example")],
                                     [evidence("c1", TENURE), evidence("c2", EXAMPLE)]))
        assert p["max_value"] == 30 and "not firm rules" in p["answer"]

    def test_conflicting_values_are_flagged_and_give_no_headline(self):
        other = "The repayment period shall not extend beyond twenty-five (25) years."
        p = build_payload(answer_for([numeric("c1", TENURE, 30), numeric("c2", other, 25, source="b.pdf", page=2)],
                                     [evidence("c1", TENURE), evidence("c2", other, "b.pdf", 2)]))
        assert p["max_value"] is None and p["needs_review"] and len(p["conflicts"]) == 1
        assert "different values" in p["answer"]


class TestNotFoundAndError:
    def test_not_found_is_a_normal_answer(self):
        p = build_payload(not_found_answer())
        assert p["route"] == "not_found" and not p["needs_review"] and p["error"] is None
        assert "does not prove" in p["answer"]

    def test_an_error_is_an_error_never_not_found(self):
        p = build_payload(error_answer("Ollama is down"))
        assert p["route"] == "error" and "Ollama is down" in p["answer"] and p["error"] == "Ollama is down"
        assert "did not find" not in p["answer"]
