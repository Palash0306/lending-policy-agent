"""
Phase 6 -- Node 4 of the pipeline: verify_grounding (the checking part).

The LLM told us a value and quoted a sentence. This file re-checks that claim
with plain code -- no AI, nothing to hallucinate -- against the paragraph the
finding points to:

  HARD checks (all three must pass for a finding to count as "grounded"):
    1. quote_in_source  - the quote really appears in the cited paragraph
    2. value_in_quote   - the number (or date) really appears in the quote
    3. value_in_source  - the number (or date) really appears in the paragraph

  SOFT checks (do not decide "grounded", but ask a human to look):
    4. direction_cue_found - the quote contains wording that fits the claimed
       minimum/maximum (or on/after/before) direction. Swapping min and max is
       a classic model mistake, and boundary testing depends on getting it right.
    5. tag_consistent - a finding tagged "firm_rule" must not contain wording
       like "for example" or "some banks".

The soft checks are deliberately simple word-pattern tests, so they can miss
unusual phrasing and are only a second pair of eyes, not proof.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from lending_agent.agent.extraction import KIND_DATE, KIND_NUMERIC, Finding
from lending_agent.agent.text_values import _contains_number, dates_in_text, months_in_text, numbers_in_text


# ----------------------------------------------------------------------------
# Text comparison
# ----------------------------------------------------------------------------

def _squash(text: str) -> str:
    """Reduce text to lowercase letters and digits only, so line breaks, curly
    quotes, odd hyphens and spacing from PDF extraction cannot cause a miss."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^\w]|_", "", text)


# ----------------------------------------------------------------------------
# Direction and wording cues (soft checks)
# ----------------------------------------------------------------------------

_MAX_RE = re.compile(
    r"\b(?:(?:not|never)\s+(?:to\s+|be\s+)*(?:exceed\w*|(?:more|greater|higher|longer|larger|bigger)\s+than|(?:extend\s+)?beyond|above|over)"
    r"|no\s+(?:more|greater|higher|longer|larger)\s+than|up\s+to|at\s+most|maxim\w+|limited\s+to|limit(?:ed)?\s+of"
    r"|capped?\s+(?:at|to)|ceiling|(?:less|fewer|shorter|lower|smaller)\s+than|below|under|within)\b",
    re.IGNORECASE,
)
_MIN_RE = re.compile(
    r"\b(?:at\s+least|(?:not|no)\s+(?:be\s+)?(?:less|fewer|shorter|lower|smaller)\s+than|not\s+(?:be\s+)?(?:below|under)"
    r"|minim\w+|floor|(?:greater|more|higher|longer|larger)\s+than|exceed\w*|above|over|starting\s+(?:from|at))\b",
    re.IGNORECASE,
)
_DATE_CUES = {
    "on_or_after": re.compile(r"\b(?:on\s+or\s+after|effective|from|since|starting|beginning|commencing|as\s+of)\b", re.IGNORECASE),
    "after": re.compile(r"\b(?:after|following|later\s+than|subsequent\s+to)\b", re.IGNORECASE),
    "on_or_before": re.compile(
        r"\b(?:on\s+or\s+before|up\s+to\s+and\s+including|through|until|by|no\s+later\s+than|not\s+later\s+than)\b", re.IGNORECASE
    ),
    "before": re.compile(r"\b(?:before|prior\s+to|earlier\s+than|preceding|grandfather\w*)\b", re.IGNORECASE),
}
_INCLUSIVE_PHRASES = re.compile(r"\bon\s+or\s+(?:after|before)\b", re.IGNORECASE)
_EXAMPLE_RE = re.compile(
    r"\b(?:for\s+example|for\s+instance|e\.g\.?|such\s+as|some\s+(?:banks|institutions|lenders)|illustrat\w+)\b",
    re.IGNORECASE,
)


def _direction_cue_found(finding: Finding) -> bool:
    quote = finding.quote
    if finding.kind == KIND_NUMERIC:
        pattern = _MAX_RE if finding.bound_type == "maximum" else _MIN_RE
        return bool(pattern.search(quote))
    direction = finding.date_direction or ""
    if direction in ("after", "before"):
        quote = _INCLUSIVE_PHRASES.sub(" ", quote)  # "on or after" must not pass as plain "after"
    pattern = _DATE_CUES.get(direction)
    return bool(pattern and pattern.search(quote))


# ----------------------------------------------------------------------------
# The check
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundingCheck:
    quote_in_source: bool
    value_in_quote: bool
    value_in_source: bool
    direction_cue_found: bool
    tag_consistent: bool
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def grounded(self) -> bool:
        """The hard checks: quote, number/date in quote, number/date in source."""
        return self.quote_in_source and self.value_in_quote and self.value_in_source

    def to_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "quote_in_source": self.quote_in_source,
            "value_in_quote": self.value_in_quote,
            "value_in_source": self.value_in_source,
            "direction_cue_found": self.direction_cue_found,
            "tag_consistent": self.tag_consistent,
            "warnings": list(self.warnings),
        }


def check_finding(finding: Finding, source_text: str) -> GroundingCheck:
    """Check one finding against the text of the paragraph it cites."""
    quote_norm = _squash(finding.quote)
    quote_in_source = bool(quote_norm) and quote_norm in _squash(source_text)

    if finding.kind == KIND_NUMERIC:
        target = float(finding.value)
        value_in_quote = _contains_number(target, numbers_in_text(finding.quote))
        value_in_source = _contains_number(target, numbers_in_text(source_text))
        label = f"number {finding.value}"
    elif finding.date_precision == "month":
        month = (finding.date or "")[:7]
        value_in_quote = month in months_in_text(finding.quote)
        value_in_source = month in months_in_text(source_text)
        label = f"month {month}"
    else:
        value_in_quote = finding.date in dates_in_text(finding.quote)
        value_in_source = finding.date in dates_in_text(source_text)
        label = f"date {finding.date}"

    direction_ok = _direction_cue_found(finding)
    tag_ok = not (finding.classification == "firm_rule" and _EXAMPLE_RE.search(finding.quote))

    warnings: list[str] = []
    if not quote_in_source:
        warnings.append("the quote was not found in the cited paragraph")
    if not value_in_quote:
        warnings.append(f"{label} was not found in the quote")
    if not value_in_source:
        warnings.append(f"{label} was not found in the cited paragraph")
    if not direction_ok:
        wanted = finding.bound_type if finding.kind == KIND_NUMERIC else finding.date_direction
        warnings.append(f"the quote has no wording that clearly means '{wanted}'")
    if not tag_ok:
        warnings.append("tagged 'firm_rule' but the quote contains example wording")
    if finding.kind == KIND_DATE and finding.date_precision == "month":
        warnings.append("only a month and year are stated; the day is not given, so it is treated as the 1st: confirm the exact day")

    return GroundingCheck(
        quote_in_source=quote_in_source,
        value_in_quote=value_in_quote,
        value_in_source=value_in_source,
        direction_cue_found=direction_ok,
        tag_consistent=tag_ok,
        warnings=tuple(warnings),
    )
