"""Phase 9: test-value generation. Table-driven: exact numbers, exact dates, and the 'never guess' rule."""
from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from lending_agent.testcases import (ACCEPT, APPLIES, CONFIRM, CONFIRM_MONTH, EXCLUSIVE, INCLUSIVE, NOT_A_RULE,
                                     NOT_APPLIES, REJECT, UNCLEAR, generate_cases, inclusivity, step_for, strip_scope,
                                     to_decimal)

Q_HIGH = "auto loans with LTV ratios less than 90 percent may have loan terms up to 72 months"
Q_LOW = "loans with LTV ratios greater than 90 percent are limited to 60-month"


def num(bound, value, quote="x", unit="months", **kw):
    return {"kind": "numeric_boundary", "grounded": True, "classification": "firm_rule", "bound_type": bound, "value": value,
            "unit": unit, "quote": quote, "applies_to": None, "source_file": "a.pdf", "page_number": 3, "needs_review": False, **kw}


def day(direction, value="2024-03-01", **kw):
    return {"kind": "date_cutoff", "grounded": True, "classification": "firm_rule", "date": value, "date_direction": direction,
            "date_precision": "day", "quote": "x", "source_file": "a.pdf", "page_number": 1, "needs_review": False, **kw}


def table(rows):
    return [(r["case"], r["value"], r["expected"]) for r in rows]


class TestSteps:
    @pytest.mark.parametrize("value,step", [(90, "1"), (6.5, "0.1"), (0.25, "0.01"), ("1,000", "1"), (30.0, "0.1"), (1e-05, "0.00001")])
    def test_step_is_one_unit_of_the_last_stated_decimal(self, value, step):
        assert step_for(to_decimal(value)) == Decimal(step)

    def test_override_and_its_validation(self):
        assert step_for(Decimal(90), "5") == Decimal(5)
        with pytest.raises(ValueError):
            step_for(Decimal(90), 0)


class TestInclusivity:
    @pytest.mark.parametrize("bound,quote,expected", [
        ("maximum", "The loan amount shall not exceed $500,000", INCLUSIVE),
        ("maximum", "loan terms up to 72 months", INCLUSIVE),
        ("maximum", "at most 40 percent", INCLUSIVE),
        ("maximum", "no more than 30 years", INCLUSIVE),
        ("maximum", "shall not be greater than 10", INCLUSIVE),
        ("maximum", "is limited to 60 months", INCLUSIVE),
        ("maximum", "less than or equal to 40 percent", INCLUSIVE),
        ("maximum", "shall not extend beyond thirty (30) years", INCLUSIVE),
        ("maximum", "the term must be less than 30 years", EXCLUSIVE),
        ("maximum", "fewer than 5 payments", EXCLUSIVE),
        ("maximum", "below 36 percent", EXCLUSIVE),
        ("maximum", "the limit is 5", UNCLEAR),
        ("minimum", "applicants must be at least 18 years old", INCLUSIVE),
        ("minimum", "shall not be less than $5,000", INCLUSIVE),
        ("minimum", "no less than 12 months", INCLUSIVE),
        ("minimum", "greater than or equal to 620", INCLUSIVE),
        ("minimum", "applicants must be greater than 21 years", EXCLUSIVE),
        ("minimum", "must exceed 10 percent", EXCLUSIVE),
        ("minimum", "above 700", EXCLUSIVE),
        ("minimum", "a score of 620", UNCLEAR),
        ("maximum", "up to 5 years but less than 7 years", UNCLEAR),          # both cues -> never guess
        ("sideways", "up to 5", UNCLEAR),
    ])
    def test_cues(self, bound, quote, expected):
        assert inclusivity(bound, quote) == expected

    def test_the_scope_wording_is_not_mistaken_for_the_limit_wording(self):
        # "less than 90 percent" describes WHO the limit applies to; the limit itself says "up to"
        assert inclusivity("maximum", Q_HIGH, "LTV below 90 percent") == INCLUSIVE
        assert inclusivity("maximum", Q_LOW, "LTV above 90 percent") == INCLUSIVE
        assert inclusivity("maximum", Q_HIGH, None) == UNCLEAR                    # without the scope the two cues clash

    def test_a_sentence_with_two_limits_is_read_one_limit_at_a_time(self):
        both = "The sanctioned loan amount shall not be less than $5,000 and shall not exceed $500,000."
        assert inclusivity("minimum", both, None, 5000) == INCLUSIVE and inclusivity("maximum", both, None, 500000) == INCLUSIVE
        assert inclusivity("maximum", both, None) == UNCLEAR                      # without the value the two limits' wording clash
        mixed = "The age must be greater than 18 but not more than 65 years."
        assert inclusivity("minimum", mixed, None, 18) == EXCLUSIVE and inclusivity("maximum", mixed, None, 65) == INCLUSIVE

    def test_strip_scope_removes_only_the_scope_phrase(self):
        assert strip_scope(Q_HIGH, "LTV below 90 percent") == "auto loans with may have loan terms up to 72 months"
        assert strip_scope("anything", None) == "anything"
        assert strip_scope("nothing in common", "zzz yyy") == "nothing in common"


class TestNumbers:
    def test_maximum_inclusive(self):
        rows = generate_cases([num("maximum", 72, "loan terms up to 72 months")])
        assert table(rows) == [("Max - 1", Decimal(71), ACCEPT), ("Max", Decimal(72), ACCEPT), ("Max + 1", Decimal(73), REJECT)]
        assert rows[0]["basis"] == INCLUSIVE and rows[0]["unit"] == "months" and rows[0]["page_number"] == 3

    def test_maximum_exclusive(self):
        rows = generate_cases([num("maximum", 30, "must be less than 30 years", unit="years")])
        assert [r["expected"] for r in rows] == [ACCEPT, REJECT, REJECT]

    def test_minimum_inclusive_and_exclusive(self):
        assert [r["expected"] for r in generate_cases([num("minimum", 18, "at least 18 years")])] == [REJECT, ACCEPT, ACCEPT]
        assert [r["expected"] for r in generate_cases([num("minimum", 21, "greater than 21 years")])] == [REJECT, REJECT, ACCEPT]

    def test_unclear_wording_asks_instead_of_guessing(self):
        rows = generate_cases([num("maximum", 5, "the limit is 5")])
        assert [r["expected"] for r in rows] == [ACCEPT, CONFIRM, REJECT] and rows[1]["basis"] == UNCLEAR

    def test_decimals_use_exact_arithmetic(self):
        rows = generate_cases([num("maximum", 6.5, "not exceed 6.5 percent", unit="percent")])
        assert [r["value"] for r in rows] == [Decimal("6.4"), Decimal("6.5"), Decimal("6.6")]
        rows = generate_cases([num("maximum", 0.3, "not exceed 0.3", unit=None)])
        assert [str(r["value"]) for r in rows] == ["0.2", "0.3", "0.4"]            # never 0.30000000000000004

    def test_step_override(self):
        assert [r["value"] for r in generate_cases([num("maximum", 90, "up to 90")], step="5")] == [Decimal(85), Decimal(90), Decimal(95)]

    def test_a_negative_result_is_noted(self):
        rows = generate_cases([num("minimum", 0, "at least 0")])
        assert rows[0]["value"] == Decimal(-1) and "negative value" in rows[0]["notes"] and "negative" not in rows[1]["notes"]

    def test_the_unit_is_copied_never_converted(self):
        assert {r["unit"] for r in generate_cases([num("maximum", 30, "up to 30", unit="years")])} == {"years"}


class TestPairing:
    def test_min_and_max_with_the_same_unit_and_scope_make_one_six_point_set(self):
        rows = generate_cases([num("minimum", 18, "at least 18 years", unit="years"), num("maximum", 65, "not exceed 65 years", unit="years")])
        assert table(rows) == [("Min - 1", 17, REJECT), ("Min", 18, ACCEPT), ("Min + 1", 19, ACCEPT),
                               ("Max - 1", 64, ACCEPT), ("Max", 65, ACCEPT), ("Max + 1", 66, REJECT)]
        assert len({r["rule"] for r in rows}) == 1 and "minimum 18 and maximum 65 years" in rows[0]["rule"]

    def test_different_units_are_not_paired(self):
        rows = generate_cases([num("minimum", 18, "at least 18", unit="years"), num("maximum", 65, "up to 65", unit="months")])
        assert len(rows) == 6 and rows[0]["rule"] != rows[3]["rule"]

    def test_different_scopes_are_not_paired(self):
        rows = generate_cases([num("minimum", 1, "at least 1", applies_to="new loans"), num("maximum", 9, "up to 9", applies_to="old loans")])
        assert len({r["rule"] for r in rows}) == 2


class TestConflictsAndRepeats:
    def test_conflicting_grounded_values_are_both_generated_and_flagged(self):
        rows = generate_cases([num("maximum", 30, "up to 30", unit="years"), num("maximum", 25, "up to 25", unit="years", source_file="b.pdf", page_number=2)])
        assert len(rows) == 6 and all("CONFLICT" in r["notes"] for r in rows)
        assert "25 (b.pdf, page 2)" in rows[0]["notes"] and "30 (a.pdf, page 3)" in rows[3]["notes"]

    def test_the_same_rule_stated_twice_is_one_set_with_a_note(self):
        rows = generate_cases([num("maximum", 30, "up to 30", unit="years"), num("maximum", 30, "up to 30", unit="years", source_file="b.pdf", page_number=9)])
        assert len(rows) == 3 and "also stated in b.pdf, page 9" in rows[0]["notes"] and "CONFLICT" not in rows[0]["notes"]

    def test_tiers_are_not_conflicts(self):
        rows = generate_cases([num("maximum", 72, "up to 72", applies_to="LTV below 90 percent"), num("maximum", 60, "up to 60", applies_to="LTV above 90 percent")])
        assert len(rows) == 6 and not any("CONFLICT" in r["notes"] for r in rows)


class TestWhichFindingsAreUsed:
    def test_ungrounded_findings_never_produce_cases(self):
        assert generate_cases([num("maximum", 30, "up to 30", grounded=False)], include_examples=True) == []

    def test_examples_are_left_out_by_default_and_labelled_when_included(self):
        example = num("maximum", 72, "may have terms up to 72 months", classification="example")
        assert generate_cases([example]) == []
        rows = generate_cases([example], include_examples=True)
        assert len(rows) == 3 and {r["expected"] for r in rows} == {NOT_A_RULE} and {r["review_status"] for r in rows} == {NOT_A_RULE}

    def test_a_finding_that_needs_review_is_marked(self):
        assert {r["review_status"] for r in generate_cases([num("maximum", 3, "up to 3", needs_review=True)])} == {"Needs review"}

    def test_incomplete_findings_are_skipped_not_crashed_on(self):
        assert generate_cases([num("maximum", None), num("sideways", 4), day("on_or_after", None), day("sometimes")]) == []


class TestDates:
    @pytest.mark.parametrize("direction,expected", [
        ("on_or_after", [NOT_APPLIES, APPLIES, APPLIES]), ("after", [NOT_APPLIES, NOT_APPLIES, APPLIES]),
        ("on_or_before", [APPLIES, APPLIES, NOT_APPLIES]), ("before", [APPLIES, NOT_APPLIES, NOT_APPLIES]),
    ])
    def test_every_direction(self, direction, expected):
        rows = generate_cases([day(direction, "2026-01-01")])
        assert [r["expected"] for r in rows] == expected
        assert [r["value"] for r in rows] == [date(2025, 12, 31), date(2026, 1, 1), date(2026, 1, 2)]
        assert [r["case"] for r in rows] == ["Day before cutoff", "Cutoff day", "Day after cutoff"]

    @pytest.mark.parametrize("cutoff,before,after", [("2024-03-01", date(2024, 2, 29), date(2024, 3, 2)),      # leap day
                                                     ("2023-03-01", date(2023, 2, 28), date(2023, 3, 2)),
                                                     ("2025-12-31", date(2025, 12, 30), date(2026, 1, 1)),      # year end
                                                     ("2025-04-30", date(2025, 4, 29), date(2025, 5, 1))])     # month end
    def test_calendar_arithmetic(self, cutoff, before, after):
        rows = generate_cases([day("on_or_after", cutoff)])
        assert rows[0]["value"] == before and rows[2]["value"] == after

    def test_month_only_dates_use_the_first_and_say_so_everywhere(self):
        rows = generate_cases([day("on_or_after", "2026-01-01", date_precision="month")])
        assert {r["expected"] for r in rows} == {CONFIRM_MONTH} and "only the month" in rows[0]["notes"]

    def test_example_dates_are_labelled_not_a_rule(self):
        rows = generate_cases([day("after", classification="guidance")], include_examples=True)
        assert {r["expected"] for r in rows} == {NOT_A_RULE}
