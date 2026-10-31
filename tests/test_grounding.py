"""
Phase 6 tests for the grounding checks. Everything here is plain text
processing -- no LLM and no Ollama involved.
"""
from __future__ import annotations

import pytest

from lending_agent.agent.extraction import Finding
from lending_agent.agent.grounding import check_finding, dates_in_text, numbers_in_text


def _numeric(quote, value, bound="maximum", unit="years", classification="firm_rule"):
    return Finding(
        kind="numeric_boundary", classification=classification, quote=quote, source_file="p.pdf",
        page_number=1, chunk_id="c1", bound_type=bound, value=value, unit=unit,
    )


def _date(quote, iso, direction="on_or_after", classification="firm_rule"):
    return Finding(
        kind="date_cutoff", classification=classification, quote=quote, source_file="p.pdf",
        page_number=1, chunk_id="c1", date=iso, date_direction=direction,
    )


class TestNumbersInText:
    @pytest.mark.parametrize("text, expected", [
        ("shall not exceed $500,000", 500000),
        ("thirty (30) years", 30),
        ("a term of thirty years", 30),
        ("twenty-five years", 25),
        ("five hundred thousand dollars", 500000),
        ("one hundred and five days", 105),
        ("up to 1.5 million", 1500000),
        ("a rate of 6.5%", 6.5),
        ("amount of 500'000", 500000),
        ("limited to 60-month terms", 60),
    ])
    def test_finds_the_number(self, text, expected):
        assert expected in numbers_in_text(text)

    def test_two_number_words_in_a_row_are_both_kept(self):
        found = numbers_in_text("between thirty and forty years")
        assert 30 in found and 40 in found

    def test_a_number_that_is_not_there_is_not_found(self):
        assert 72 not in numbers_in_text("loan terms up to 60 months")


class TestDatesInText:
    @pytest.mark.parametrize("text", [
        "on or after January 1, 2026",
        "from 1 January 2026",
        "effective 1st of January, 2026",
        "effective Jan. 1, 2026",
        "since 2026-01-01",
    ])
    def test_finds_the_date_however_it_is_written(self, text):
        assert "2026-01-01" in dates_in_text(text)

    def test_september_abbreviations(self):
        assert "2025-09-05" in dates_in_text("as of Sept. 5, 2025")

    def test_numeric_dates_are_ambiguous_so_both_readings_are_kept(self):
        found = dates_in_text("applies from 01.02.2026")
        assert "2026-02-01" in found and "2026-01-02" in found

    def test_month_and_year_alone_is_not_a_full_date(self):
        assert dates_in_text("effective January 2026") == set()

    def test_impossible_dates_are_ignored(self):
        assert dates_in_text("on February 30, 2026") == set()


class TestQuoteAndValueChecks:
    SOURCE = "Intro text. The repayment period shall not extend beyond thirty (30) years from disbursement. More text."

    def test_a_correct_finding_is_grounded(self):
        f = _numeric("The repayment period shall not extend beyond thirty (30) years from disbursement.", 30)
        check = check_finding(f, self.SOURCE)
        assert check.grounded
        assert check.warnings == ()

    def test_quote_missing_from_the_source_is_not_grounded(self):
        f = _numeric("The repayment period must never be longer than 30 years.", 30)
        check = check_finding(f, self.SOURCE)
        assert not check.quote_in_source and not check.grounded
        assert any("quote was not found" in w for w in check.warnings)

    def test_a_number_the_quote_does_not_contain_is_not_grounded(self):
        f = _numeric("The repayment period shall not extend beyond thirty (30) years from disbursement.", 25)
        check = check_finding(f, self.SOURCE)
        assert check.quote_in_source and not check.value_in_quote and not check.grounded

    def test_line_breaks_curly_quotes_and_hyphens_do_not_break_the_match(self):
        source = "The bank\u2019s repay-\nment period shall not exceed 30 years."
        f = _numeric("The bank's repayment period shall not exceed 30 years.", 30)
        assert check_finding(f, source).grounded

    def test_case_differences_are_ignored(self):
        f = _numeric("THE REPAYMENT PERIOD SHALL NOT EXTEND BEYOND THIRTY (30) YEARS", 30)
        assert check_finding(f, self.SOURCE).grounded

    def test_an_empty_quote_is_never_grounded(self):
        assert not check_finding(_numeric("", 30), self.SOURCE).grounded


class TestDateChecks:
    SOURCE = "This criterion applies to applications submitted on or after January 1, 2026."

    def test_a_correct_date_is_grounded(self):
        f = _date(self.SOURCE, "2026-01-01")
        assert check_finding(f, self.SOURCE).grounded

    def test_a_date_the_model_invented_is_not_grounded(self):
        f = _date(self.SOURCE, "2026-03-01")
        check = check_finding(f, self.SOURCE)
        assert not check.value_in_quote and not check.grounded

    def test_a_made_up_day_for_a_month_only_date_is_caught(self):
        source = "The new rule is effective January 2026."
        f = _date(source, "2026-01-01")
        assert not check_finding(f, source).grounded


class TestDirectionCues:
    @pytest.mark.parametrize("quote, bound", [
        ("The loan amount shall not exceed $500,000", "maximum"),
        ("terms up to 72 months", "maximum"),
        ("repayment shall not extend beyond thirty (30) years", "maximum"),
        ("loans are limited to 60-month terms", "maximum"),
        ("no more than 65 years of age", "maximum"),
        ("The amount shall not be less than $5,000", "minimum"),
        ("age must be at least 21 years", "minimum"),
        ("a minimum of 20 percent down", "minimum"),
        ("must be greater than 5,000", "minimum"),
    ])
    def test_matching_direction_is_recognised(self, quote, bound):
        number = 1  # irrelevant here; only the direction warning is inspected
        check = check_finding(_numeric(quote, number, bound=bound), quote)
        assert check.direction_cue_found

    @pytest.mark.parametrize("quote, wrong_bound", [
        ("age must be at least 21 years", "maximum"),
        ("terms up to 72 months", "minimum"),
        ("repayment shall not extend beyond thirty (30) years", "minimum"),
    ])
    def test_a_swapped_minimum_and_maximum_is_flagged(self, quote, wrong_bound):
        check = check_finding(_numeric(quote, 1, bound=wrong_bound), quote)
        assert not check.direction_cue_found
        assert any("clearly means" in w for w in check.warnings)

    def test_a_direction_problem_alone_does_not_make_a_finding_ungrounded(self):
        quote = "age must be at least 21 years"
        check = check_finding(_numeric(quote, 21, bound="maximum"), quote)
        assert check.grounded and not check.direction_cue_found

    @pytest.mark.parametrize("quote, direction", [
        ("applies on or after January 1, 2026", "on_or_after"),
        ("effective from January 1, 2026", "on_or_after"),
        ("applications received after January 1, 2026", "after"),
        ("accounts opened prior to January 1, 2026", "before"),
        ("grandfathered accounts of January 1, 2026", "before"),
        ("valid until January 1, 2026", "on_or_before"),
    ])
    def test_date_directions_are_recognised(self, quote, direction):
        check = check_finding(_date(quote, "2026-01-01", direction=direction), quote)
        assert check.direction_cue_found

    def test_on_or_after_does_not_pass_for_plain_after(self):
        quote = "applies on or after January 1, 2026"
        check = check_finding(_date(quote, "2026-01-01", direction="after"), quote)
        assert not check.direction_cue_found


class TestTagConsistency:
    QUOTE = "For example, some banks allow terms up to 72 months."

    def test_a_firm_rule_with_example_wording_is_flagged(self):
        check = check_finding(_numeric(self.QUOTE, 72, unit="months", classification="firm_rule"), self.QUOTE)
        assert not check.tag_consistent
        assert any("example wording" in w for w in check.warnings)

    def test_the_same_quote_tagged_as_an_example_is_fine(self):
        check = check_finding(_numeric(self.QUOTE, 72, unit="months", classification="example"), self.QUOTE)
        assert check.tag_consistent
