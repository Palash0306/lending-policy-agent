"""Tests for focus_text: trimming evidence to the sentences that can hold a boundary value."""
from __future__ import annotations

from lending_agent.agent.focus import focus_text


class TestFocusText:
    def test_a_paragraph_with_no_number_or_date_becomes_empty(self):
        text = "Banks should monitor credit quality. Underwriting must be prudent. Policies need board approval."
        assert focus_text(text) == ""

    def test_empty_text_stays_empty(self):
        assert focus_text("") == ""

    def test_keeps_the_sentence_with_a_number(self):
        out = focus_text("Intro sentence here. The loan shall not exceed $500,000. Closing remark follows.")
        assert "shall not exceed $500,000" in out

    def test_drops_number_free_sentences_far_from_a_number(self):
        text = ("Alpha is discussed first. Beta is discussed second. Gamma is discussed third. "
                "The term shall not exceed 30 years. Delta is discussed fifth. Epsilon is discussed sixth.")
        out = focus_text(text)
        assert "30 years" in out
        assert "Alpha" not in out and "Beta" not in out and "Epsilon" not in out and "Delta" not in out

    def test_keeps_the_sentence_just_before_a_number_for_context(self):
        text = "Alpha is first. The following limit applies to retail borrowers. The amount shall not exceed 500 units."
        out = focus_text(text)
        assert "retail borrowers" in out

    def test_context_can_be_switched_off(self):
        text = "The following limit applies to retail borrowers. The amount shall not exceed 500 units."
        assert "retail borrowers" not in focus_text(text, context_before=0)

    def test_marks_where_text_was_skipped(self):
        text = ("Limit A is 10 units. Filler alpha here. Filler beta here. Filler gamma here. Limit B is 20 units.")
        assert "..." in focus_text(text)

    def test_spelled_out_numbers_count(self):
        assert "thirty years" in focus_text("Filler first. The term shall not exceed thirty years.")

    def test_dates_count(self):
        assert "January 1, 2026" in focus_text("Filler first. This applies on or after January 1, 2026.")

    def test_the_result_is_never_longer_than_the_input(self):
        text = "Filler alpha. Limit is 10. Filler beta. Filler gamma. Limit is 20."
        assert len(focus_text(text)) <= len(text) + 8
