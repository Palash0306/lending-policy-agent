"""
Golden phrasing table: realistic policy sentences and what the plain-code readers (Phase 5/6) must make of them.

Three groups: (A) which numbers are read, (B) which dates are read, (C) whether a claimed finding is grounded, and
whether the wording fits the claimed direction. Group D holds the negative cases: an invented value must NOT ground.
Group E documents one known weakness as a strict expected failure (see the note there).
"""
from __future__ import annotations

import pytest

from lending_agent.agent.extraction import Finding
from lending_agent.agent.grounding import check_finding
from lending_agent.agent.text_values import dates_in_text, months_in_text, numbers_in_text

REAL = ("auto loans with LTV ratios less than 90 percent may have loan terms up to 72 months, while loans with LTV "
        "ratios greater than 90 percent are limited to 60-month")

# ---------------------------------------------------------------------------------------------
# A. numbers: (sentence, numbers that must be read, numbers that must NOT be read)
# ---------------------------------------------------------------------------------------------
NUMBERS = [
    ("The loan amount shall not exceed $500,000.", {500000}, {500, 0}),
    ("Loan terms up to 72 months.", {72}, set()),
    ("The repayment period shall not extend beyond thirty (30) years from disbursement.", {30}, {3}),
    ("The interest rate is capped at 6.5% per annum.", {6.5}, {6, 5}),
    ("Borrowers must be at least eighteen years old.", {18}, set()),
    ("A term of seventy two months.", {72}, {7}),
    ("Loans up to five hundred thousand dollars.", {500000}, set()),
    ("No more than twenty-five percent of income.", {25}, set()),
    ("Limits of 1.5 million per borrower.", {1500000, 1.5}, set()),
    ("DTI must not exceed 43.5 percent.", {43.5}, {43, 435}),
    ("A fee of $1,250.75 applies.", {1250.75}, {1250}),
    ("A minimum credit score of 620.", {620}, set()),
    ("Between 5 and 10 years.", {5, 10}, set()),
    ("Terms of 3 to 5 years.", {3, 5}, set()),
    ("Within thirty (30) days of closing.", {30}, set()),
    (REAL, {90, 72, 60}, {9, 7, 6}),
    ("The bank shall use sound judgment.", set(), {1}),
    ("Some borrowers receive a dozen payments.", set(), {12}),            # spelled-out quantities such as "a dozen" or "half" are not read
]


@pytest.mark.parametrize("sentence,present,absent", NUMBERS)
def test_numbers_read(sentence, present, absent):
    read = numbers_in_text(sentence)
    assert {float(n) for n in present} <= read, f"missing from {sorted(read)}"
    assert not ({float(n) for n in absent} & read), f"unexpected in {sorted(read)}"


# ---------------------------------------------------------------------------------------------
# B. dates: (sentence, full dates read, months read)
# ---------------------------------------------------------------------------------------------
DATES = [
    ("applications submitted on or after January 1, 2026", {"2026-01-01"}, {"2026-01"}),
    ("effective 1 March 2025", {"2025-03-01"}, {"2025-03"}),
    ("effective 2026-01-01", {"2026-01-01"}, {"2026-01"}),
    ("as of the first of January 2026", {"2026-01-01"}, {"2026-01"}),
    ("the twenty-first of March 2026", {"2026-03-21"}, {"2026-03"}),
    ("grandfathered accounts opened before June 30, 2024", {"2024-06-30"}, {"2024-06"}),
    ("loans originated after Dec. 31, 2023", {"2023-12-31"}, {"2023-12"}),
    ("no later than March 15, 2025", {"2025-03-15"}, {"2025-03"}),
    ("Feb 29, 2024", {"2024-02-29"}, {"2024-02"}),                       # a real leap day
    ("01.02.2026", {"2026-01-02", "2026-02-01"}, {"2026-01", "2026-02"}),  # ambiguous: both readings kept, never one guessed
    ("rules start in January 2026", set(), {"2026-01"}),                 # month only: no day is invented
    ("Feb 30, 2024", set(), set()),                                      # not a real date
    ("The policy was updated in 2025.", set(), set()),                   # a year alone is not a date
]


@pytest.mark.parametrize("sentence,full,months", DATES)
def test_dates_read(sentence, full, months):
    assert dates_in_text(sentence) == full and months_in_text(sentence) == months


# ---------------------------------------------------------------------------------------------
# C. grounding of a claimed finding: (quote, claim, grounded?, wording fits the claimed direction?)
# ---------------------------------------------------------------------------------------------
def numeric(quote, bound, value, **kw):
    return Finding(kind="numeric_boundary", classification=kw.pop("classification", "firm_rule"), quote=quote, source_file="a.pdf",
                   page_number=1, chunk_id="c", bound_type=bound, value=value, unit=kw.pop("unit", "months"), **kw)


def dated(quote, direction, date, precision="day", **kw):
    return Finding(kind="date_cutoff", classification="firm_rule", quote=quote, source_file="a.pdf", page_number=1, chunk_id="c",
                   date=date, date_direction=direction, date_precision=precision, **kw)


GROUNDED = [
    # the real sentence from the OCC handbook: both limits, taken from one paragraph
    (numeric("auto loans with LTV ratios less than 90 percent may have loan terms up to 72 months", "maximum", 72, classification="example"), REAL, True, True),
    (numeric("loans with LTV ratios greater than 90 percent are limited to 60-month", "maximum", 60, classification="example"), REAL, True, True),
    (numeric("The loan amount shall not exceed $500,000", "maximum", 500000, unit="USD"), "The loan amount shall not exceed $500,000.", True, True),
    (numeric("shall not extend beyond thirty (30) years", "maximum", 30, unit="years"), "The repayment period shall not extend beyond thirty (30) years from disbursement.", True, True),
    (numeric("capped at 6.5% per annum", "maximum", 6.5, unit="percent"), "The interest rate is capped at 6.5% per annum.", True, True),
    (numeric("at least eighteen years old", "minimum", 18, unit="years"), "Borrowers must be at least eighteen years old.", True, True),
    (numeric("shall not be less than $5,000", "minimum", 5000, unit="USD"), "The loan shall not be less than $5,000.", True, True),
    (numeric("a minimum credit score of 620", "minimum", 620, unit=None), "A minimum credit score of 620 is required.", True, True),
    (dated("on or after January 1, 2026", "on_or_after", "2026-01-01"), "Applies to applications submitted on or after January 1, 2026.", True, True),
    (dated("effective March 1, 2025", "on_or_after", "2025-03-01"), "This rule is effective March 1, 2025.", True, True),
    (dated("before June 30, 2024", "before", "2024-06-30"), "Accounts opened before June 30, 2024 are grandfathered.", True, True),
    (dated("no later than March 15, 2025", "on_or_before", "2025-03-15"), "Submit no later than March 15, 2025.", True, True),
    (dated("after December 31, 2023", "after", "2023-12-31"), "Loans originated after December 31, 2023.", True, True),
    (dated("on or after January 2026", "on_or_after", "2026-01-01", precision="month"), "Applies on or after January 2026.", True, True),
]


@pytest.mark.parametrize("finding,source,grounded,direction", GROUNDED)
def test_claims_that_match_the_text_are_grounded(finding, source, grounded, direction):
    check = check_finding(finding, source)
    assert check.grounded is grounded and check.direction_cue_found is direction


# ---------------------------------------------------------------------------------------------
# D. negatives: nothing invented may ground
# ---------------------------------------------------------------------------------------------
NOT_GROUNDED = [
    ("a wrong number", numeric("loan terms up to 72 months", "maximum", 75), "loan terms up to 72 months"),
    ("a number only in the paragraph, not in the quote", numeric("loan terms up to 72 months", "maximum", 90), REAL),
    ("a number that is nowhere", numeric("loan terms up to 72 months", "maximum", 36), "loan terms up to 72 months"),
    ("a decimal that is really a whole number", numeric("capped at 6 percent", "maximum", 6.5, unit="percent"), "capped at 6 percent"),
    ("a quote that is not in the paragraph", numeric("loan terms up to 84 months", "maximum", 84), "loan terms up to 72 months"),
    ("a quote built from two different places", numeric("up to 72 months and at least 18 years", "maximum", 72), "loan terms up to 72 months. Applicants must be at least 18 years."),
    ("an invented date", dated("on or after January 1, 2026", "on_or_after", "2026-01-02"), "on or after January 1, 2026"),
    ("a full date when only the month is stated", dated("on or after January 2026", "on_or_after", "2026-01-15"), "on or after January 2026"),
    ("a different year", dated("on or after January 1, 2026", "on_or_after", "2025-01-01"), "on or after January 1, 2026"),
    ("an impossible date", dated("on or after Feb 30, 2024", "on_or_after", "2024-02-30"), "on or after Feb 30, 2024"),
    ("an empty quote", numeric("", "maximum", 72), "loan terms up to 72 months"),
]


@pytest.mark.parametrize("label,finding,source", NOT_GROUNDED, ids=[n[0] for n in NOT_GROUNDED])
def test_invented_or_misquoted_values_do_not_ground(label, finding, source):
    check = check_finding(finding, source)
    assert check.grounded is False and check.warnings


def test_min_and_max_swapped_is_flagged_by_the_wording_check():
    swapped = check_finding(numeric("must be at least eighteen years old", "maximum", 18), "must be at least eighteen years old")
    assert swapped.grounded is True and swapped.direction_cue_found is False       # soft check: a person is asked to look


def test_an_after_that_is_really_on_or_after_is_flagged():
    check = check_finding(dated("on or after January 1, 2026", "after", "2026-01-01"), "on or after January 1, 2026")
    assert check.direction_cue_found is False


def test_example_wording_on_a_firm_rule_is_flagged():
    check = check_finding(numeric("for example, up to 72 months", "maximum", 72), "for example, up to 72 months")
    assert check.tag_consistent is False


# ---------------------------------------------------------------------------------------------
# E. KNOWN WEAKNESS (strict expected failures: they will start failing, loudly, the day this is fixed)
# The number reader also returns the PARTS of a number written in words ("seventy two" gives 2, 70 and 72), so a model
# that claims the value 2 for "seventy two months" passes the hard checks. Found while writing this table; not changed here.
# ---------------------------------------------------------------------------------------------
PARTS = [(2, "The term shall not exceed seventy two months."), (70, "The term shall not exceed seventy two months."),
         (5, "The loan shall not exceed five hundred thousand dollars."), (20, "The rate is capped at twenty-five percent.")]


@pytest.mark.xfail(strict=True, reason="word numbers also yield their parts, so an invented part-value grounds (known weakness)")
@pytest.mark.parametrize("value,quote", PARTS)
def test_a_part_of_a_word_number_must_not_ground(value, quote):
    assert check_finding(numeric(quote, "maximum", value), quote).grounded is False
