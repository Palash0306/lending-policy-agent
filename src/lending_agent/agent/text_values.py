"""
Reading numbers and dates out of plain text.

Shared by two steps that must agree on what counts as a number or a date:
  - focus.py   (before the LLM) skips paragraphs that contain no number at all,
  - grounding.py (after the LLM) checks the model's number or date really is there.

Numbers are recognised in digits (500,000 / 6.5% / 1.5 million) and in words
(thirty / twenty-five / five hundred thousand). Dates are recognised however they
are written (January 1, 2026 / 1 Jan 2026 / 2026-01-01 / 01.02.2026 / the first of
January 2026). A month and year alone ("January 2026") is not a full date;
months_in_text() reads those separately.
"""
from __future__ import annotations

import datetime as dt
import math
import re

# ----------------------------------------------------------------------------
# Numbers
# ----------------------------------------------------------------------------

_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}

_DIGIT_NUMBER_RE = re.compile(
    r"(?<![\w.])(\d{1,3}(?:[,'\u2019]\d{3})+|\d+)(?:\.(\d+))?(?:\s*(thousand|million|billion)\b)?",
    re.IGNORECASE,
)


def _digit_numbers(text: str) -> set[float]:
    found: set[float] = set()
    for whole, decimals, scale in _DIGIT_NUMBER_RE.findall(text):
        number = float(re.sub(r"[,'\u2019]", "", whole) + (f".{decimals}" if decimals else ""))
        found.add(number)
        if scale:
            found.add(number * _SCALES[scale.lower()])
    return found


def _word_numbers(text: str) -> set[float]:
    """Spelled-out numbers: 'thirty', 'twenty-five', 'five hundred thousand'.
    Each number word is also kept on its own, so 'thirty and forty' never loses either."""
    found: set[float] = set()
    total = current = 0
    in_run = False
    previous = ""

    def flush() -> None:
        nonlocal total, current, in_run
        if in_run:
            found.add(float(total + current))
        total = current = 0
        in_run = False

    for token in re.findall(r"[a-z]+", text.lower()):
        if token in _UNITS or token in _TENS:
            value = _UNITS.get(token, _TENS.get(token))
            found.add(float(value))
            current += value
            in_run = True
        elif token == "hundred" and in_run:
            current = (current or 1) * 100
        elif token in _SCALES and in_run:
            total += (current or 1) * _SCALES[token]
            current = 0
        elif token == "and" and previous in ("hundred", *_SCALES):
            pass  # "one hundred and five"
        else:
            flush()
        previous = token
    flush()
    return found


def words_to_number(text: str) -> float | None:
    """'seventy two' / 'seventy-two' / 'five hundred thousand' -> 72 / 72 / 500000.
    Returns None unless the WHOLE text is number words (so '1.5 million' or 'a dozen' give None)."""
    if re.search(r"\d", text):
        return None
    tokens = re.findall(r"[a-z]+", text.lower())
    total = current = 0
    seen = False
    for token in tokens:
        if token in _UNITS:
            current += _UNITS[token]
            seen = True
        elif token in _TENS:
            current += _TENS[token]
            seen = True
        elif token == "hundred" and seen:
            current = (current or 1) * 100
        elif token in _SCALES and seen:
            total += (current or 1) * _SCALES[token]
            current = 0
        elif token == "and":
            continue
        else:
            return None
    return float(total + current) if seen else None


def numbers_in_text(text: str) -> set[float]:
    """Every number the text states, whether written in digits or in words."""
    return _digit_numbers(text) | _word_numbers(text)


def _contains_number(target: float, numbers: set[float]) -> bool:
    return any(math.isclose(target, n, rel_tol=1e-9, abs_tol=1e-9) for n in numbers)


# ----------------------------------------------------------------------------
# Dates
# ----------------------------------------------------------------------------

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ORDINAL_UNITS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9}
_DAY_WORDS = {
    **_ORDINAL_UNITS,
    "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}
for _tens_word, _tens in (("twenty", 20), ("thirty", 30)):
    for _unit_word, _unit in _ORDINAL_UNITS.items():
        if _tens + _unit <= 31:
            _DAY_WORDS[f"{_tens_word}-{_unit_word}"] = _tens + _unit
# "twenty-first" may also be written "twenty first"
_DAY_WORD_RE = "|".join(
    "[- ]".join(map(re.escape, word.split("-"))) for word in sorted(_DAY_WORDS, key=len, reverse=True)
)


def _day_from_word(word: str) -> int:
    return _DAY_WORDS[re.sub(r"\s+", "-", word.lower())]


_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_MONTH_FIRST_RE = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE)
_DAY_FIRST_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:day\s+)?(?:of\s+)?({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.IGNORECASE)
_WORD_DAY_FIRST_RE = re.compile(rf"\b({_DAY_WORD_RE})\s+(?:day\s+)?(?:of\s+)?({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.IGNORECASE)
_MONTH_THEN_WORD_DAY_RE = re.compile(rf"\b({_MONTH_RE})\.?\s+({_DAY_WORD_RE}),?\s+(\d{{4}})\b", re.IGNORECASE)
_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH_RE})\.?,?\s+(?:of\s+)?(\d{{4}})\b", re.IGNORECASE)
_NUMERIC_MONTH_YEAR_RE = re.compile(r"(?<![\d/.-])(\d{1,2})[/-](\d{4})\b")
_ISO_MONTH_RE = re.compile(r"\b(\d{4})-(\d{2})\b")
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[./-](\d{1,2})[./-](\d{4})\b")


def _try_date(year: int, month: int, day: int) -> str | None:
    try:
        return dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def dates_in_text(text: str) -> set[str]:
    """Every full calendar date the text states, as ISO strings. A numeric date
    like 01.02.2026 is ambiguous (1 Feb or 2 Jan), so both readings are kept.
    A month and year alone ('January 2026') is NOT a full date and is ignored."""
    found: set[str] = set()
    for y, m, d in _ISO_RE.findall(text):
        found.add(_try_date(int(y), int(m), int(d)))
    for month, d, y in _MONTH_FIRST_RE.findall(text):
        found.add(_try_date(int(y), _MONTHS[month.lower()], int(d)))
    for d, month, y in _DAY_FIRST_RE.findall(text):
        found.add(_try_date(int(y), _MONTHS[month.lower()], int(d)))
    for word, month, y in _WORD_DAY_FIRST_RE.findall(text):
        found.add(_try_date(int(y), _MONTHS[month.lower()], _day_from_word(word)))
    for month, word, y in _MONTH_THEN_WORD_DAY_RE.findall(text):
        found.add(_try_date(int(y), _MONTHS[month.lower()], _day_from_word(word)))
    for a, b, y in _NUMERIC_DATE_RE.findall(text):
        found.add(_try_date(int(y), int(b), int(a)))  # day first
        found.add(_try_date(int(y), int(a), int(b)))  # month first
    found.discard(None)
    return found


def months_in_text(text: str) -> set[str]:
    """Every month-and-year the text states, as 'YYYY-MM' ('January 2026', '01/2026', '2026-01').
    The month of any full date counts too. Used when a rule is dated only to a month."""
    found: set[str] = set()
    for month, y in _MONTH_YEAR_RE.findall(text):
        found.add(f"{int(y):04d}-{_MONTHS[month.lower()]:02d}")
    for m, y in _NUMERIC_MONTH_YEAR_RE.findall(text):
        if 1 <= int(m) <= 12:
            found.add(f"{int(y):04d}-{int(m):02d}")
    for y, m in _ISO_MONTH_RE.findall(text):
        if 1 <= int(m) <= 12:
            found.add(f"{int(y):04d}-{int(m):02d}")
    for iso in dates_in_text(text):
        found.add(iso[:7])
    return found
