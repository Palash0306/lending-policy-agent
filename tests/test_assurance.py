"""
Phase 6 tests for verify_extraction and conflict detection. Plain code only;
the LLM is not involved (Phase 5 results are built by hand).
"""
from __future__ import annotations

import json

from lending_agent.agent.assurance import detect_conflicts, verify_extraction
from lending_agent.agent.extraction import STATUS_ERROR, STATUS_FOUND, STATUS_NOT_FOUND, ExtractionResult, Finding
from lending_agent.agent.retrieval import Evidence

PAGE_23 = (
    "Some banks control this potential exposure using tiered criteria; for example, auto loans with "
    "LTV ratios less than 90 percent may have loan terms up to 72 months, while loans with LTV ratios "
    "greater than 90 percent are limited to 60-month 35 For certain types of retail lending, ..."
)


def _evidence(chunk_id="c1", text="The repayment period shall not exceed thirty (30) years.", source="a.pdf", page=5):
    return Evidence(chunk_id=chunk_id, text=text, source_file=source, page_number=page, doc_type="pdf", best_distance=0.2)


def _finding(value=30, quote="The repayment period shall not exceed thirty (30) years.", chunk_id="c1",
             source="a.pdf", page=5, classification="firm_rule", applies_to=None, unit="years", bound="maximum"):
    return Finding(
        kind="numeric_boundary", classification=classification, quote=quote, source_file=source,
        page_number=page, chunk_id=chunk_id, bound_type=bound, value=value, unit=unit, applies_to=applies_to,
    )


def _result(*findings, status=STATUS_FOUND, **kwargs):
    return ExtractionResult(field="loan_tenure", status=status, findings=list(findings), **kwargs)


class TestVerifyExtraction:
    def test_a_correct_finding_is_verified_and_needs_no_review(self):
        out = verify_extraction(_result(_finding()), [_evidence()])
        assert out.verified and not out.needs_review
        assert out.findings[0].grounded

    def test_a_bad_finding_is_kept_but_flagged(self):
        bad = _finding(value=25)  # the quote says 30
        out = verify_extraction(_result(bad), [_evidence()])
        assert len(out.findings) == 1
        assert not out.verified and out.needs_review
        assert any("25" in r for r in out.findings[0].review_reasons)

    def test_one_bad_finding_stops_the_whole_result_being_verified(self):
        out = verify_extraction(_result(_finding(), _finding(value=99)), [_evidence()])
        assert not out.verified
        assert out.findings[0].grounded and not out.findings[1].grounded

    def test_a_finding_pointing_at_a_missing_paragraph_is_not_grounded(self):
        out = verify_extraction(_result(_finding(chunk_id="ghost")), [_evidence()])
        assert not out.findings[0].grounded

    def test_not_found_is_neither_verified_nor_flagged(self):
        out = verify_extraction(_result(status=STATUS_NOT_FOUND), [_evidence()])
        assert not out.verified and not out.needs_review

    def test_an_error_is_carried_through(self):
        out = verify_extraction(_result(status=STATUS_ERROR, error="ollama down"), [_evidence()])
        assert out.status == STATUS_ERROR and out.error == "ollama down"
        assert not out.verified

    def test_rejected_answers_are_carried_through(self):
        out = verify_extraction(_result(_finding(), rejected=["finding #2: no quote"]), [_evidence()])
        assert out.rejected == ["finding #2: no quote"]

    def test_the_real_page_23_example_passes_cleanly(self):
        ev = _evidence(chunk_id="c98", text=PAGE_23, page=23)
        first = _finding(
            value=72, unit="months", classification="example", chunk_id="c98", page=23,
            applies_to="LTV ratios less than 90 percent",
            quote="auto loans with LTV ratios less than 90 percent may have loan terms up to 72 months",
        )
        second = _finding(
            value=60, unit="months", classification="example", chunk_id="c98", page=23,
            applies_to="LTV ratios greater than 90 percent",
            quote="loans with LTV ratios greater than 90 percent are limited to 60-month",
        )
        out = verify_extraction(_result(first, second), [ev])
        assert out.verified
        assert not out.needs_review
        assert out.conflicts == []


class TestConflicts:
    def _two(self, a_kwargs, b_kwargs, b_text="The repayment period shall not exceed twenty five (25) years."):
        evidence = [
            _evidence("c1", source="a.pdf"),
            _evidence("c2", text=b_text, source="b.pdf", page=9),
        ]
        a = _finding(chunk_id="c1", source="a.pdf", **a_kwargs)
        b = _finding(
            value=25, chunk_id="c2", source="b.pdf", page=9,
            quote="The repayment period shall not exceed twenty five (25) years.", **b_kwargs,
        )
        return verify_extraction(_result(a, b), evidence)

    def test_different_values_for_the_same_limit_are_a_conflict(self):
        out = self._two({}, {})
        assert len(out.conflicts) == 1
        assert out.needs_review
        assert out.findings[0].conflicts_with == [2] and out.findings[1].conflicts_with == [1]
        assert {v["source_file"] for v in out.conflicts[0].values} == {"a.pdf", "b.pdf"}

    def test_a_conflict_is_still_verified_but_needs_review(self):
        out = self._two({}, {})
        assert out.verified and out.needs_review

    def test_different_scopes_are_tiers_not_a_conflict(self):
        out = self._two({"applies_to": "retail borrowers"}, {"applies_to": "commercial borrowers"})
        assert out.conflicts == []
        assert not out.needs_review

    def test_scope_wording_differences_that_mean_the_same_still_conflict(self):
        out = self._two({"applies_to": "The retail borrower"}, {"applies_to": "retail borrower"})
        assert len(out.conflicts) == 1

    def test_the_same_value_in_two_documents_is_agreement_not_conflict(self):
        evidence = [_evidence("c1", source="a.pdf"), _evidence("c2", source="b.pdf", page=9)]
        out = verify_extraction(
            _result(_finding(chunk_id="c1", source="a.pdf"), _finding(chunk_id="c2", source="b.pdf", page=9)),
            evidence,
        )
        assert out.conflicts == [] and not out.needs_review

    def test_an_example_and_a_firm_rule_are_not_compared(self):
        text = "For example, the repayment period may be up to twenty five (25) years."
        ev = [_evidence("c1"), _evidence("c2", text=text)]
        ex = _finding(value=25, chunk_id="c2", classification="example", quote=text[:-1])
        out = verify_extraction(_result(_finding(), ex), ev)
        assert out.conflicts == []

    def test_a_minimum_and_a_maximum_are_not_compared(self):
        text = "The repayment period shall not be less than five (5) years."
        ev = [_evidence("c1"), _evidence("c2", text=text)]
        low = _finding(value=5, chunk_id="c2", bound="minimum", quote=text[:-1])
        out = verify_extraction(_result(_finding(), low), ev)
        assert out.conflicts == []

    def test_an_ungrounded_finding_cannot_create_a_conflict(self):
        out = verify_extraction(_result(_finding(), _finding(value=99)), [_evidence()])
        assert out.conflicts == []

    def test_date_cutoffs_conflict_too(self):
        def date_finding(iso, chunk_id, quote):
            return Finding(
                kind="date_cutoff", classification="firm_rule", quote=quote, source_file="x.pdf",
                page_number=1, chunk_id=chunk_id, date=iso, date_direction="on_or_after",
            )
        q1 = "The rule applies on or after January 1, 2026"
        q2 = "The rule applies on or after March 1, 2026"
        ev = [_evidence("d1", text=q1 + "."), _evidence("d2", text=q2 + ".")]
        out = verify_extraction(_result(date_finding("2026-01-01", "d1", q1), date_finding("2026-03-01", "d2", q2)), ev)
        assert len(out.conflicts) == 1
        assert out.conflicts[0].limit == "on_or_after"

    def test_detect_conflicts_can_be_called_directly(self):
        out = self._two({}, {})
        assert len(detect_conflicts(out.findings)) == 1


class TestSerialisation:
    def test_the_result_is_json_serialisable_with_all_the_flags(self):
        out = verify_extraction(_result(_finding(), _finding(value=99)), [_evidence()])
        data = json.loads(json.dumps(out.to_dict()))
        assert data["verified"] is False and data["needs_review"] is True
        assert data["findings"][0]["grounded"] is True
        assert data["findings"][1]["checks"]["value_in_quote"] is False
        assert "review_reasons" in data["findings"][1]
