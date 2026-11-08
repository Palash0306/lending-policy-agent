"""
Phase 9 -- turns verified findings into test values (boundary value analysis + date cutoffs).

Pure code: no web framework, no model, no floating point. Numbers are handled with `Decimal`, built from
`str(value)`, so 6.5 stays 6.5 and 0.1 + 0.2 never appears. Dates use `datetime.date`, so month ends and
leap days are right.

What is NEVER done:
  * a case is never generated from a finding that did not pass the grounding checks;
  * units are never converted: the unit is copied exactly as the document states it;
  * an expected result is never guessed. If the quote does not make it clear whether the limit value itself
    is allowed, the expected result is "Confirm with policy owner".

A finding is a dict as produced by the agent (see agent/assurance.py): kind, bound_type, value, unit, date,
date_direction, date_precision, applies_to, classification, quote, source_file, page_number, grounded,
needs_review, review_reasons.
"""
from __future__ import annotations

import math
import re
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Iterable

ACCEPT, REJECT, CONFIRM = "Accept", "Reject", "Confirm with policy owner"
APPLIES, NOT_APPLIES = "Rule applies", "Rule does not apply"
CONFIRM_MONTH = "Confirm (document gives only the month)"
NOT_A_RULE = "Not a firm rule"
INCLUSIVE, EXCLUSIVE, UNCLEAR = "inclusive", "exclusive", "unclear"
FIRM = "firm_rule"
NEGATIVE_NOTE = "negative value: check the system can accept it"

# Wording that says the limit value itself IS allowed ("up to 72" allows 72) or is NOT allowed ("less than 72" does not).
# The "or equal" forms are added on top of the plain cues: they are certain, and without them "less than or equal to"
# would be read as the exclusive "less than".
_CUES: dict[str, dict[str, list[str]]] = {
    "maximum": {
        INCLUSIVE: ["less than or equal to", "equal to or less than", "at or below", "or less", "or fewer", "not exceed", "not to exceed", "not extend beyond", "not go beyond", "no longer than", "not longer than",
                    "up to", "at most", "no more than", "not more than", "maximum", "limited to", "capped at",
                    "no greater than", "not greater than"],
        EXCLUSIVE: ["less than", "fewer than", "below", "under", "shorter than", "lower than"],
    },
    "minimum": {
        INCLUSIVE: ["greater than or equal to", "equal to or greater than", "at or above", "or more", "at least",
                    "not less than", "no less than", "not fewer than", "minimum of", "not lower than", "not below", "no shorter than", "not shorter than"],
        EXCLUSIVE: ["greater than", "more than", "above", "over", "exceed", "longer than"],
    },
}
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


# ---------------------------------------------------------------------------------------------
# reading the wording
# ---------------------------------------------------------------------------------------------
def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def strip_scope(quote: str, applies_to: str | None) -> str:
    """Remove the part of the quote that only describes WHO the limit applies to ("LTV ratios less than 90 percent"),
    so its wording cannot be mistaken for the wording of the limit itself."""
    scope = [w for w in _words(applies_to or "")]
    words = _words(quote)
    if not scope or not words:
        return " ".join(words)
    scope_set, best = set(scope), None
    need = max(1, math.ceil(0.6 * len(scope_set)))
    for length in range(1, min(len(words), 3 * len(scope)) + 1):
        for start in range(len(words) - length + 1):
            window = words[start:start + length]
            hits = len(scope_set & set(window))
            if hits < need or window[0] not in scope_set or window[-1] not in scope_set:
                continue
            key = (-hits, length)
            if best is None or key < best[0]:
                best = (key, start, length)
    if best is None:
        return " ".join(words)
    _, start, length = best
    return " ".join(words[:start] + words[start + length:])


def _find_cues(text: str, cues: Iterable[str]) -> tuple[list[str], str]:
    """Which cue phrases occur (longest first); returns them and the text with those phrases blanked out."""
    found = []
    for cue in sorted(cues, key=len, reverse=True):
        pattern = r"\b" + r"\s+".join(re.escape(w) for w in cue.split()) + r"\b"
        if re.search(pattern, text):
            found.append(cue)
            text = re.sub(pattern, " ", text)
    return found, text


_CLAUSE_SPLIT = re.compile(r"\b(?:and|but|while|whereas)\b|;|,(?!\d)|(?<!\d),", re.IGNORECASE)   # a comma inside 5,000 is not a break


def clause_around(quote: str, value: Any) -> str:
    """When a sentence states several limits ("not less than $5,000 and not exceed $500,000"), keep only the part(s)
    that contain this value, so the other limit's wording cannot be mistaken for this one's."""
    from lending_agent.agent.text_values import numbers_in_text

    try:
        target = float(to_decimal(value))
    except Exception:
        return quote
    keep = [c for c in _CLAUSE_SPLIT.split(quote) if any(math.isclose(target, n, rel_tol=1e-9, abs_tol=1e-9) for n in numbers_in_text(c))]
    return " ".join(keep) if keep else quote


def inclusivity(bound_type: str, quote: str, applies_to: str | None = None, value: Any = None) -> str:
    """"inclusive" (the limit value is allowed), "exclusive" (it is not) or "unclear" (cannot tell from the quote)."""
    cues = _CUES.get(bound_type)
    if not cues:
        return UNCLEAR
    if value is not None:
        quote = clause_around(quote, value)
    text = re.sub(r"\b(not|no)\s+be\s+", r"\1 ", strip_scope(quote, applies_to))   # "shall not be less than" = "not less than"
    inclusive, rest = _find_cues(text, cues[INCLUSIVE])          # inclusive first, and blanked out before the exclusive check
    exclusive, _ = _find_cues(rest, cues[EXCLUSIVE])
    if inclusive and not exclusive:
        return INCLUSIVE
    if exclusive and not inclusive:
        return EXCLUSIVE
    return UNCLEAR


# ---------------------------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------------------------
def to_decimal(value: Any) -> Decimal:
    return Decimal(str(value).strip().replace(",", ""))


def decimals_in(value: Decimal) -> int:
    exponent = value.as_tuple().exponent
    return -exponent if isinstance(exponent, int) and exponent < 0 else 0


def step_for(value: Decimal, override: Any = None) -> Decimal:
    """10^-d, where d is the number of decimals in the stated value (6.5 -> 0.1, 90 -> 1), unless overridden."""
    if override is not None:
        step = to_decimal(override)
        if step <= 0:
            raise ValueError("step must be greater than zero")
        return step
    return Decimal(1).scaleb(-decimals_in(value))


def _norm_scope(text: str | None) -> str:
    return " ".join(_words(text or ""))


def _expected(bound: str, offset: int, basis: str) -> str:
    """offset -1 / 0 / +1 relative to the limit value."""
    if basis == UNCLEAR and offset == 0:
        return CONFIRM
    if bound == "minimum":
        return REJECT if offset < 0 else ACCEPT if offset > 0 else (ACCEPT if basis == INCLUSIVE else REJECT)
    return ACCEPT if offset < 0 else REJECT if offset > 0 else (ACCEPT if basis == INCLUSIVE else REJECT)


def _rule_text(f: dict) -> str:
    unit = f" {f['unit']}" if f.get("unit") else ""
    text = f"{f.get('bound_type')} {_fmt(to_decimal(f['value']))}{unit}"
    return text + (f" (applies to: {f['applies_to']})" if f.get("applies_to") else "")


def _fmt(d: Decimal) -> str:
    return format(d, "f")


def _row(f: dict, rule: str, case: str, value: Decimal | date, expected: str, basis: str, notes: list[str]) -> dict:
    return {
        "kind": "date" if isinstance(value, date) else "number", "rule": rule, "case": case, "value": value,
        "unit": f.get("unit") or "", "expected": expected, "basis": basis,
        "source_file": f.get("source_file"), "page_number": f.get("page_number"), "quote": f.get("quote") or "",
        "classification": f.get("classification") or "", "notes": "; ".join(n for n in notes if n),
        "review_status": _review_status(f),
    }


def _review_status(f: dict) -> str:
    if f.get("classification") != FIRM:
        return NOT_A_RULE
    return "Needs review" if f.get("needs_review") else "Verified"


def _numeric_rows(f: dict, step_override: Any, notes: list[str]) -> list[dict]:
    value = to_decimal(f["value"])
    step = step_for(value, step_override)
    bound = f["bound_type"]
    basis = inclusivity(bound, f.get("quote") or "", f.get("applies_to"), f["value"])
    firm = f.get("classification") == FIRM
    rule = _rule_text(f)
    label = "Min" if bound == "minimum" else "Max"
    rows = []
    for offset, name in ((-1, f"{label} - 1"), (0, label), (1, f"{label} + 1")):
        shown = value + offset * step
        extra = list(notes) + ([NEGATIVE_NOTE] if shown < 0 else [])
        expected = _expected(bound, offset, basis) if firm else NOT_A_RULE
        rows.append(_row(f, rule, name, shown, expected, basis, extra))
    return rows


def _six_point(lo: dict, hi: dict, step_override: Any, notes: list[str]) -> list[dict]:
    """One set for a field that has both a minimum and a maximum: min-1, min, min+1, max-1, max, max+1."""
    rule = f"minimum {_fmt(to_decimal(lo['value']))} and maximum {_fmt(to_decimal(hi['value']))}" + (
        f" {hi['unit']}" if hi.get("unit") else "") + (f" (applies to: {hi['applies_to']})" if hi.get("applies_to") else "")
    rows = []
    for f in (lo, hi):
        for row in _numeric_rows(f, step_override, notes):
            row["rule"] = rule
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------
# dates
# ---------------------------------------------------------------------------------------------
_DATE_EXPECT = {                # direction -> (day before, the day, day after)
    "on_or_after": (False, True, True), "after": (False, False, True),
    "on_or_before": (True, True, False), "before": (True, False, False),
}
_DIRECTION_WORDS = {"on_or_after": "on or after", "after": "after", "on_or_before": "on or before", "before": "before"}


def _date_rows(f: dict, notes: list[str]) -> list[dict]:
    direction = f.get("date_direction")
    if direction not in _DATE_EXPECT:
        return []
    cutoff = date.fromisoformat(str(f["date"])[:10])
    month_only = f.get("date_precision") == "month"
    firm = f.get("classification") == FIRM
    rule = f"applies {_DIRECTION_WORDS[direction]} {cutoff.isoformat()}"
    extra = list(notes) + (["document gives only the month: the 1st is used as a stand-in for the day"] if month_only else [])
    rows = []
    for offset, name, applies in zip((-1, 0, 1), ("Day before cutoff", "Cutoff day", "Day after cutoff"), _DATE_EXPECT[direction]):
        expected = APPLIES if applies else NOT_APPLIES
        if not firm:
            expected = NOT_A_RULE
        elif month_only:
            expected = CONFIRM_MONTH
        rows.append(_row(f, rule, name, cutoff + timedelta(days=offset), expected, _DIRECTION_WORDS[direction], extra))
    return rows


# ---------------------------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------------------------
def _usable(findings: Iterable[dict], include_examples: bool) -> list[dict]:
    keep = []
    for f in findings:
        if not f.get("grounded"):
            continue                                         # never a case from an ungrounded finding
        if f.get("classification") != FIRM and not include_examples:
            continue
        if f.get("kind") == "numeric_boundary" and (f.get("value") is None or f.get("bound_type") not in _CUES):
            continue
        if f.get("kind") == "date_cutoff" and not f.get("date"):
            continue
        keep.append(f)
    return keep


def _limit_key(f: dict) -> tuple:
    which = f.get("bound_type") if f.get("kind") == "numeric_boundary" else f.get("date_direction")
    return (f.get("kind"), which, f.get("unit") or "", f.get("classification"), _norm_scope(f.get("applies_to")))


def _value_of(f: dict) -> str:
    return str(to_decimal(f["value"])) if f.get("kind") == "numeric_boundary" else str(f["date"])[:10]


def _where(f: dict) -> str:
    return f"{f.get('source_file')}" + (f", page {f.get('page_number')}" if f.get("page_number") else "")


def generate_cases(findings: Iterable[dict], step: Any = None, include_examples: bool = False) -> list[dict]:
    """Test-value rows (dicts) for one field's findings. See the module docstring for the rules."""
    usable = _usable(list(findings), include_examples)

    # 1. drop repeats of the same rule (overlapping paragraphs often state it twice); remember where else it was seen
    unique: list[dict] = []
    seen: dict[tuple, int] = {}
    also: dict[int, list[str]] = {}
    for f in usable:
        key = _limit_key(f) + (_value_of(f),)
        if key in seen:
            also.setdefault(seen[key], []).append(_where(f))
            continue
        seen[key] = len(unique)
        unique.append(f)

    # 2. conflicts: the same limit with different grounded values
    groups: dict[tuple, list[int]] = {}
    for i, f in enumerate(unique):
        groups.setdefault(_limit_key(f), []).append(i)
    notes: dict[int, list[str]] = {i: [] for i in range(len(unique))}
    for members in groups.values():
        if len(members) > 1:
            for i in members:
                others = [f"{_value_of(unique[j])} ({_where(unique[j])})" for j in members if j != i]
                notes[i].append("CONFLICT: other value " + "; ".join(others))
    for i, places in also.items():
        notes[i].append("also stated in " + "; ".join(dict.fromkeys(places)))

    # 3. pair a minimum and a maximum of the same field, unit and scope into one six-point set
    consumed: set[int] = set()
    pairs: dict[int, tuple[int, int]] = {}
    scopes: dict[tuple, dict[str, list[int]]] = {}
    for i, f in enumerate(unique):
        if f.get("kind") == "numeric_boundary":
            scopes.setdefault((f.get("unit") or "", f.get("classification"), _norm_scope(f.get("applies_to"))), {}) \
                  .setdefault(f["bound_type"], []).append(i)
    for by_bound in scopes.values():
        if len(by_bound.get("minimum", [])) == 1 and len(by_bound.get("maximum", [])) == 1:
            lo, hi = by_bound["minimum"][0], by_bound["maximum"][0]
            pairs[min(lo, hi)] = (lo, hi)
            consumed.update((lo, hi))

    rows: list[dict] = []
    for i, f in enumerate(unique):
        if i in pairs:
            lo, hi = pairs[i]
            rows += _six_point(unique[lo], unique[hi], step, notes[lo] + [n for n in notes[hi] if n not in notes[lo]])
        elif i in consumed:
            continue
        elif f.get("kind") == "numeric_boundary":
            rows += _numeric_rows(f, step, notes[i])
        else:
            rows += _date_rows(f, notes[i])
    return rows
