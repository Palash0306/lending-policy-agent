"""Tests for the shared number/date readers: number phrases, day-words and month-level dates."""
from __future__ import annotations

import pytest

from lending_agent.agent.text_values import dates_in_text, months_in_text, words_to_number


class TestWordsToNumber:
    @pytest.mark.parametrize("text, expected", [
        ("seventy two", 72), ("seventy-two", 72), ("Seventy Two", 72), ("thirty", 30),
        ("one hundred and five", 105), ("five hundred thousand", 500000), ("two million", 2000000),
        ("twelve", 12), ("one hundred twenty", 120),
    ])
    def test_converts_number_phrases(self, text, expected):
        assert words_to_number(text) == expected

    @pytest.mark.parametrize("text", ["", "a dozen", "hundred", "thirty apples", "1.5 million", "72", "thirty (30)", "half"])
    def test_anything_that_is_not_purely_number_words_gives_none(self, text):
        assert words_to_number(text) is None


class TestDayWordDates:
    @pytest.mark.parametrize("text, expected", [
        ("the first of January 2026", "2026-01-01"),
        ("first day of January, 2026", "2026-01-01"),
        ("January first, 2026", "2026-01-01"),
        ("the twenty-first of March 2026", "2026-03-21"),
        ("twenty first day of March, 2026", "2026-03-21"),
        ("the thirty-first of December 2025", "2025-12-31"),
        ("the fifteenth of June 2026", "2026-06-15"),
        ("1st day of January 2026", "2026-01-01"),
    ])
    def test_reads_days_written_as_words(self, text, expected):
        assert expected in dates_in_text(text)

    def test_an_impossible_day_word_date_is_ignored(self):
        assert dates_in_text("the thirty-first of February 2026") == set()

    def test_second_as_a_unit_of_time_is_not_a_date(self):
        assert dates_in_text("wait one second in 2026") == set()


class TestMonthsInText:
    @pytest.mark.parametrize("text", ["effective January 2026", "Jan. 2026", "January of 2026", "01/2026", "2026-01", "1-2026"])
    def test_reads_a_month_and_year(self, text):
        assert "2026-01" in months_in_text(text)

    def test_the_month_of_a_full_date_counts(self):
        assert "2026-01" in months_in_text("on January 15, 2026")

    def test_a_different_month_is_not_matched(self):
        assert "2026-03" not in months_in_text("effective January 2026")

    def test_an_impossible_month_is_ignored(self):
        assert months_in_text("13/2026") == set()

    def test_a_year_alone_is_not_a_month(self):
        assert months_in_text("in 2026 the rule changed") == set()
