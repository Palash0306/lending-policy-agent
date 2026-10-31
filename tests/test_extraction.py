"""
Phase 5 tests for extract_conditions. The LLM is a scripted fake, so these run
with no Ollama and are fully deterministic. Phase 5 does not verify quotes
against the source (that is Phase 6), so none of these tests expect it to.
"""
from __future__ import annotations

import json

from lending_agent.agent.extraction import (
    KIND_DATE,
    KIND_NUMERIC,
    STATUS_ERROR,
    STATUS_FOUND,
    STATUS_NOT_FOUND,
    extract_conditions,
)
from lending_agent.agent.retrieval import Evidence


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = 0
        self.prompt = None
        self.kwargs = {}

    def generate(self, prompt, **kwargs):
        self.calls += 1
        self.prompt = prompt
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.reply


def _evidence():
    return [
        Evidence(
            chunk_id="c1", source_file="policy.pdf", page_number=12, doc_type="pdf", best_distance=0.2,
            text="The repayment period shall not extend beyond thirty (30) years from disbursement.",
        ),
        Evidence(
            chunk_id="c2", source_file="policy.pdf", page_number=23, doc_type="pdf", best_distance=0.3,
            text="For example, auto loans with LTV ratios less than 90 percent may have loan terms up to 72 months.",
        ),
        Evidence(
            chunk_id="c3", source_file="addendum.docx", page_number=None, doc_type="docx", best_distance=0.4,
            text="This criterion applies to applications submitted on or after January 1, 2026.",
        ),
    ]


def _numeric(**overrides):
    item = {
        "kind": "numeric_boundary", "bound_type": "maximum", "value": 30, "unit": "years",
        "date": None, "date_direction": None, "applies_to": None, "classification": "firm_rule",
        "evidence_id": 1, "quote": "The repayment period shall not extend beyond thirty (30) years from disbursement.",
    }
    item.update(overrides)
    return item


def _date(**overrides):
    item = {
        "kind": "date_cutoff", "bound_type": None, "value": None, "unit": None,
        "date": "2026-01-01", "date_direction": "on_or_after", "applies_to": None,
        "classification": "firm_rule", "evidence_id": 3,
        "quote": "This criterion applies to applications submitted on or after January 1, 2026.",
    }
    item.update(overrides)
    return item


def _reply(*items):
    return json.dumps({"findings": list(items)})


def _run(*items, **kwargs):
    return extract_conditions("loan_tenure", _evidence(), llm=FakeLLM(_reply(*items)), **kwargs)


class TestGoodAnswers:
    def test_numeric_maximum_is_extracted_with_our_own_source_details(self):
        result = _run(_numeric())
        assert result.status == STATUS_FOUND
        f = result.findings[0]
        assert (f.kind, f.bound_type, f.value, f.unit) == (KIND_NUMERIC, "maximum", 30, "years")
        assert (f.source_file, f.page_number, f.chunk_id) == ("policy.pdf", 12, "c1")

    def test_date_cutoff_is_extracted(self):
        result = _run(_date())
        f = result.findings[0]
        assert (f.kind, f.date, f.date_direction) == (KIND_DATE, "2026-01-01", "on_or_after")
        assert f.source_file == "addendum.docx" and f.page_number is None

    def test_an_example_stays_tagged_as_an_example(self):
        result = _run(_numeric(
            value=72, unit="months", classification="example", evidence_id=2,
            applies_to="LTV below 90 percent",
            quote="For example, auto loans with LTV ratios less than 90 percent may have loan terms up to 72 months.",
        ))
        f = result.findings[0]
        assert f.classification == "example"
        assert f.applies_to == "LTV below 90 percent"
        assert f.page_number == 23

    def test_a_model_that_lies_about_the_page_cannot_change_it(self):
        result = _run(_numeric(page_number=99, source_file="made_up.pdf"))
        assert result.findings[0].page_number == 12
        assert result.findings[0].source_file == "policy.pdf"

    def test_numbers_written_as_text_are_parsed(self):
        result = _run(_numeric(value="500,000"), _date())
        assert result.findings[0].value == 500000

    def test_findings_are_json_serialisable(self):
        result = _run(_numeric(), _date())
        assert json.loads(json.dumps(result.to_dict()))["findings"][1]["date"] == "2026-01-01"

    def test_a_result_starts_out_unverified(self):
        assert _run(_numeric()).verified is False


class TestHonestStatus:
    def test_empty_list_means_not_found_and_is_not_an_error(self):
        result = _run()
        assert result.status == STATUS_NOT_FOUND
        assert result.error is None

    def test_no_evidence_means_not_found_without_calling_the_llm(self):
        llm = FakeLLM(_reply(_numeric()))
        result = extract_conditions("loan_tenure", [], llm=llm)
        assert result.status == STATUS_NOT_FOUND
        assert llm.calls == 0

    def test_ollama_being_down_is_an_error_not_not_found(self):
        result = extract_conditions("loan_tenure", _evidence(), llm=FakeLLM(error=ConnectionError("ollama down")))
        assert result.status == STATUS_ERROR
        assert "ollama down" in result.error

    def test_a_reply_that_is_not_json_is_an_error(self):
        result = extract_conditions("loan_tenure", _evidence(), llm=FakeLLM("I could not find anything."))
        assert result.status == STATUS_ERROR

    def test_json_without_a_findings_list_is_an_error(self):
        result = extract_conditions("loan_tenure", _evidence(), llm=FakeLLM('{"answer": "30 years"}'))
        assert result.status == STATUS_ERROR

    def test_json_in_code_fences_is_accepted(self):
        reply = "```json\n" + _reply(_numeric()) + "\n```"
        result = extract_conditions("loan_tenure", _evidence(), llm=FakeLLM(reply))
        assert result.status == STATUS_FOUND


class TestSchemaIsEnforced:
    def test_unknown_classification_is_rejected_with_a_reason(self):
        result = _run(_numeric(classification="probably a rule"))
        assert result.status == STATUS_NOT_FOUND
        assert "classification" in result.rejected[0]

    def test_bad_bound_type_is_rejected(self):
        assert "bound_type" in _run(_numeric(bound_type="roughly")).rejected[0]

    def test_non_numeric_value_is_rejected(self):
        assert "not a number" in _run(_numeric(value="plenty")).rejected[0]

    def test_boolean_value_is_rejected(self):
        assert "not a number" in _run(_numeric(value=True)).rejected[0]

    def test_invalid_date_is_rejected(self):
        assert "date" in _run(_date(date="January 1st")).rejected[0]

    def test_impossible_date_is_rejected(self):
        assert "date" in _run(_date(date="2026-02-30")).rejected[0]

    def test_bad_date_direction_is_rejected(self):
        assert "date_direction" in _run(_date(date_direction="around")).rejected[0]

    def test_missing_quote_is_rejected(self):
        assert "quote" in _run(_numeric(quote="  ")).rejected[0]

    def test_unknown_kind_is_rejected(self):
        assert "kind" in _run(_numeric(kind="vibe")).rejected[0]

    def test_a_good_finding_survives_next_to_a_bad_one(self):
        result = _run(_numeric(classification="nope"), _date())
        assert result.status == STATUS_FOUND
        assert len(result.findings) == 1
        assert len(result.rejected) == 1

    def test_duplicates_are_collapsed(self):
        assert len(_run(_numeric(), _numeric()).findings) == 1


class TestAttribution:
    def test_bad_evidence_id_falls_back_to_finding_the_quote(self):
        result = _run(_numeric(evidence_id=42))
        assert result.findings[0].chunk_id == "c1"

    def test_bad_evidence_id_and_unknown_quote_is_rejected(self):
        result = _run(_numeric(evidence_id=42, quote="Something nobody wrote."))
        assert result.status == STATUS_NOT_FOUND
        assert "evidence" in result.rejected[0]

    def test_evidence_id_given_as_text_is_accepted(self):
        assert _run(_numeric(evidence_id="1")).findings[0].chunk_id == "c1"


class TestWhatIsSentToTheModel:
    def test_asks_for_json_with_zero_temperature_and_a_big_enough_window(self):
        llm = FakeLLM(_reply())
        extract_conditions("loan_tenure", _evidence(), llm=llm, num_ctx=4096)
        assert llm.kwargs == {"format": "json", "temperature": 0, "num_ctx": 4096}

    def test_prompt_numbers_the_extracts_and_names_the_field(self):
        llm = FakeLLM(_reply())
        extract_conditions("loan_tenure", _evidence(), llm=llm)
        assert "[1] The repayment period" in llm.prompt
        assert "[3] This criterion" in llm.prompt
        assert "loan_tenure" in llm.prompt

    def test_prompt_teaches_implicit_wording(self):
        llm = FakeLLM(_reply())
        extract_conditions("loan_tenure", _evidence(), llm=llm)
        for phrase in ("shall not exceed", "at least", "on or after", "grandfathered", "for example"):
            assert phrase in llm.prompt

    def test_restricting_kinds_leaves_out_the_other_instructions(self):
        llm = FakeLLM(_reply())
        extract_conditions("loan_tenure", _evidence(), llm=llm, kinds=(KIND_DATE,))
        assert "on or after" in llm.prompt
        assert "shall not exceed" not in llm.prompt

    def test_a_kind_that_was_not_requested_is_rejected(self):
        result = _run(_numeric(), kinds=(KIND_DATE,))
        assert result.status == STATUS_NOT_FOUND
        assert "not requested" in result.rejected[0]


class TestFocusAndWindow:
    NO_NUMBERS = "Banks should monitor credit quality closely and keep sound underwriting standards."

    def _mixed_evidence(self):
        ev = _evidence()
        ev.insert(1, Evidence(chunk_id="cx", source_file="policy.pdf", page_number=20, doc_type="pdf",
                              best_distance=0.25, text=self.NO_NUMBERS))
        return ev

    def test_paragraphs_without_numbers_are_not_sent_to_the_model(self):
        llm = FakeLLM(_reply())
        result = extract_conditions("loan_tenure", self._mixed_evidence(), llm=llm)
        assert "credit quality" not in llm.prompt
        assert (result.paragraphs_total, result.paragraphs_sent) == (4, 3)

    def test_the_original_paragraph_numbers_are_kept_so_attribution_still_works(self):
        llm = FakeLLM(_reply(_numeric(evidence_id=1), _date(evidence_id=4)))
        result = extract_conditions("loan_tenure", self._mixed_evidence(), llm=llm)
        assert "[2]" not in llm.prompt            # paragraph 2 was skipped
        assert "[4] This criterion" in llm.prompt  # but the last one keeps its number
        assert result.findings[1].source_file == "addendum.docx"

    def test_a_quote_from_a_trimmed_paragraph_still_grounds_against_the_full_text(self):
        from lending_agent.agent.assurance import verify_extraction
        long_text = ("Filler sentence one is here. Filler sentence two is here. Filler sentence three is here. "
                     "The repayment period shall not extend beyond thirty (30) years from disbursement. "
                     "Filler sentence four is here. Filler sentence five is here.")
        ev = [Evidence(chunk_id="c1", source_file="p.pdf", page_number=1, doc_type="pdf", best_distance=0.1, text=long_text)]
        result = extract_conditions("loan_tenure", ev, llm=FakeLLM(_reply(_numeric(evidence_id=1))))
        assert verify_extraction(result, ev).verified

    def test_if_no_paragraph_has_a_number_the_model_is_never_called(self):
        llm = FakeLLM(_reply(_numeric()))
        ev = [Evidence(chunk_id="cx", source_file="p.pdf", page_number=1, doc_type="pdf", best_distance=0.1, text=self.NO_NUMBERS)]
        result = extract_conditions("loan_tenure", ev, llm=llm)
        assert llm.calls == 0
        assert result.status == STATUS_NOT_FOUND
        assert (result.paragraphs_total, result.paragraphs_sent) == (1, 0)

    def test_focus_can_be_turned_off_to_send_everything(self):
        llm = FakeLLM(_reply())
        result = extract_conditions("loan_tenure", self._mixed_evidence(), llm=llm, focus=False)
        assert "credit quality" in llm.prompt
        assert result.paragraphs_sent == 4

    def test_the_window_is_raised_automatically_when_the_prompt_would_not_fit(self):
        huge = "The limit is 5 units. " + ("Padding words fill this paragraph out. " * 1200)
        ev = [Evidence(chunk_id="c1", source_file="p.pdf", page_number=1, doc_type="pdf", best_distance=0.1, text=huge)]
        llm = FakeLLM(_reply())
        extract_conditions("loan_tenure", ev, llm=llm, num_ctx=4096, focus=False)
        assert llm.kwargs["num_ctx"] > 4096
        assert llm.kwargs["num_ctx"] % 1024 == 0

    def test_the_window_is_not_raised_when_the_prompt_fits(self):
        llm = FakeLLM(_reply())
        extract_conditions("loan_tenure", _evidence(), llm=llm, num_ctx=4096)
        assert llm.kwargs["num_ctx"] == 4096

    def test_paragraph_counts_are_in_the_json_output(self):
        data = _run(_numeric()).to_dict()
        assert data["paragraphs_total"] == 3 and data["paragraphs_sent"] == 3


class TestNumberWordsInTheModelsAnswer:
    def test_a_value_written_in_words_is_converted(self):
        assert _run(_numeric(value="thirty")).findings[0].value == 30

    def test_two_word_numbers_are_converted_with_or_without_a_hyphen(self):
        assert _run(_numeric(value="seventy two", unit="months")).findings[0].value == 72
        assert _run(_numeric(value="seventy-two", unit="months")).findings[0].value == 72

    def test_a_word_and_digit_pair_uses_the_digits(self):
        assert _run(_numeric(value="thirty (30)")).findings[0].value == 30

    def test_large_number_words_are_converted(self):
        assert _run(_numeric(value="five hundred thousand")).findings[0].value == 500000

    def test_words_that_are_not_a_number_are_still_rejected(self):
        for bad in ("a dozen", "many", "thirty apples"):
            assert "not a number" in _run(_numeric(value=bad)).rejected[0]

    def test_a_converted_value_is_still_checked_against_the_quote(self):
        from lending_agent.agent.assurance import verify_extraction
        quote = "The tenure period shall not exceed seventy two months."
        ev = [Evidence(chunk_id="c1", source_file="p.pdf", page_number=1, doc_type="pdf", best_distance=0.1, text=quote)]
        result = extract_conditions("loan_tenure", ev, llm=FakeLLM(_reply(_numeric(value="seventy two", unit="months", quote=quote))))
        assert verify_extraction(result, ev).verified


class TestMonthLevelDates:
    QUOTE = "The new rule is effective January 2026."

    def _ev(self):
        return [Evidence(chunk_id="m1", source_file="p.pdf", page_number=3, doc_type="pdf", best_distance=0.1, text=self.QUOTE)]

    def _month(self, **overrides):
        return _date(date="2026-01-01", date_precision="month", evidence_id=1, quote=self.QUOTE, **overrides)

    def test_a_month_level_date_is_accepted_and_marked(self):
        result = extract_conditions("f", self._ev(), llm=FakeLLM(_reply(self._month())))
        f = result.findings[0]
        assert (f.date, f.date_precision) == ("2026-01-01", "month")

    def test_a_year_month_date_is_accepted_and_treated_as_the_first(self):
        item = _date(date="2026-01", evidence_id=1, quote=self.QUOTE)
        f = extract_conditions("f", self._ev(), llm=FakeLLM(_reply(item))).findings[0]
        assert (f.date, f.date_precision) == ("2026-01-01", "month")

    def test_month_precision_with_a_day_other_than_the_first_is_rejected(self):
        item = _date(date="2026-01-15", date_precision="month", evidence_id=1, quote=self.QUOTE)
        result = extract_conditions("f", self._ev(), llm=FakeLLM(_reply(item)))
        assert result.status == STATUS_NOT_FOUND and "not the 1st" in result.rejected[0]

    def test_saying_day_precision_for_a_month_only_date_is_rejected(self):
        item = _date(date="2026-01", date_precision="day", evidence_id=1, quote=self.QUOTE)
        assert "only a month" in extract_conditions("f", self._ev(), llm=FakeLLM(_reply(item))).rejected[0]

    def test_an_unknown_precision_is_rejected(self):
        item = _date(date="2026-01-01", date_precision="week", evidence_id=1, quote=self.QUOTE)
        assert "date_precision" in extract_conditions("f", self._ev(), llm=FakeLLM(_reply(item))).rejected[0]

    def test_a_normal_date_defaults_to_day_precision(self):
        assert _run(_date()).findings[0].date_precision == "day"

    def test_numeric_findings_have_no_date_precision(self):
        assert _run(_numeric()).findings[0].date_precision is None

    def test_the_month_level_date_passes_grounding_but_asks_for_review(self):
        from lending_agent.agent.assurance import verify_extraction
        ev = self._ev()
        result = extract_conditions("f", ev, llm=FakeLLM(_reply(self._month())))
        out = verify_extraction(result, ev)
        assert out.verified and out.needs_review
        assert any("day is not given" in r for r in out.findings[0].review_reasons)

    def test_a_month_that_is_not_in_the_quote_is_not_grounded(self):
        from lending_agent.agent.assurance import verify_extraction
        ev = self._ev()
        item = _date(date="2026-03-01", date_precision="month", evidence_id=1, quote=self.QUOTE)
        out = verify_extraction(extract_conditions("f", ev, llm=FakeLLM(_reply(item))), ev)
        assert not out.verified

    def test_the_prompt_explains_month_level_dates(self):
        llm = FakeLLM(_reply())
        extract_conditions("loan_tenure", _evidence(), llm=llm)
        assert "date_precision" in llm.prompt and "never invent a day" in llm.prompt

    def test_date_precision_is_in_the_json_output(self):
        assert _run(_date()).to_dict()["findings"][0]["date_precision"] == "day"
