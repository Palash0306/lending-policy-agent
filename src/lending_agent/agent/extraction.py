"""
Phase 5 -- Node 3 of the pipeline: extract boundary / date-cutoff conditions.

Input:  a field name plus the evidence paragraphs found in Phase 4.
Output: an ExtractionResult in a fixed, strictly checked shape.

Three ideas drive the design:

1. FIXED SHAPE. Every finding has the same fields (kind, bound, value, unit,
   date, classification, quote, source). Anything the model returns that does
   not fit the shape is rejected and listed in `rejected`, never silently kept.

2. CLASSIFICATION TAG. Each finding is tagged "firm_rule", "example" or
   "guidance". A sentence like "for example, some banks allow 72 months" holds a
   real number, but it is not a rule -- the tag keeps a tester from treating it
   as one.

3. HONEST STATUS. "found", "not_found" and "error" are different answers.
   Ollama being switched off is an *error*, never "not found".

The file name and page of each finding come from our own evidence list (looked
up by evidence number), never from the model's say-so.

NOTE: Phase 5 does NOT verify that the quote or number really appears in the
source text. That is Phase 6. Until then every finding is unverified.
"""
from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from lending_agent.agent.focus import focus_text
from lending_agent.agent.llm_json import parse_llm_json
from lending_agent.agent.text_values import words_to_number
from lending_agent.agent.retrieval import Evidence

KIND_NUMERIC = "numeric_boundary"
KIND_DATE = "date_cutoff"
ALL_KINDS = (KIND_NUMERIC, KIND_DATE)

BOUND_TYPES = ("minimum", "maximum")
DATE_DIRECTIONS = ("on_or_after", "after", "on_or_before", "before")
DATE_PRECISIONS = ("day", "month")
CLASSIFICATIONS = ("firm_rule", "example", "guidance")

STATUS_FOUND = "found"
STATUS_NOT_FOUND = "not_found"
STATUS_ERROR = "error"

# Ollama reads only a small window of text by default (often 2048 tokens) and
# silently cuts off whatever does not fit. 4096 fits six full paragraphs plus our
# instructions; _choose_num_ctx() raises it automatically if a prompt is ever bigger.
# A bigger window also costs more memory, so we do not ask for more than needed.
DEFAULT_NUM_CTX = 4096


class _TextGenerator(Protocol):
    def generate(self, prompt: str, **kwargs) -> str: ...


@dataclass(frozen=True)
class Finding:
    kind: str                       # "numeric_boundary" or "date_cutoff"
    classification: str             # "firm_rule", "example" or "guidance"
    quote: str                      # the sentence the model says the value came from
    source_file: str                # from our evidence list, not from the model
    page_number: int | None
    chunk_id: str
    bound_type: str | None = None   # numeric: "minimum" or "maximum"
    value: int | float | None = None
    unit: str | None = None
    date: str | None = None         # date cutoff: ISO "YYYY-MM-DD"
    date_direction: str | None = None
    date_precision: str | None = None   # date cutoff: "day", or "month" when the text gives only a month and year
    applies_to: str | None = None   # condition the value is limited to, e.g. "LTV below 90 percent"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "classification": self.classification,
            "bound_type": self.bound_type,
            "value": self.value,
            "unit": self.unit,
            "date": self.date,
            "date_direction": self.date_direction,
            "date_precision": self.date_precision,
            "applies_to": self.applies_to,
            "quote": self.quote,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
        }


@dataclass
class ExtractionResult:
    field: str
    status: str                                           # "found", "not_found" or "error"
    findings: list[Finding] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)     # model answers that failed the schema, with reasons
    error: str | None = None
    verified: bool = False                                # becomes True only after Phase 6 grounding
    paragraphs_total: int = 0                             # paragraphs retrieved
    paragraphs_sent: int = 0                              # paragraphs that held a number or date and were sent to the LLM

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "status": self.status,
            "verified": self.verified,
            "paragraphs_total": self.paragraphs_total,
            "paragraphs_sent": self.paragraphs_sent,
            "findings": [f.to_dict() for f in self.findings],
            "rejected": self.rejected,
            "error": self.error,
        }


# ----------------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------------

_NUMERIC_SECTION = (
    '- "numeric_boundary": a minimum or maximum value for the field. The text often '
    'does NOT use the words "minimum" or "maximum". Wording such as "shall not exceed", '
    '"up to", "no more than", "limited to", "capped at" means maximum. Wording such as '
    '"at least", "not less than", "no lower than", "a minimum of" means minimum. '
    "Fill bound_type, value and unit; set date and date_direction to null."
)

_DATE_SECTION = (
    '- "date_cutoff": a date from or until which a rule about this field applies. '
    'Wording such as "on or after", "effective from", "applies to applications submitted since" '
    'means on_or_after. Wording such as "prior to", "before", "grandfathered" for items '
    'dated before a date means before. "up to and including" means on_or_before. '
    'Write the date as YYYY-MM-DD and set date_precision to "day". If the text gives only a month and '
    'year (for example "January 2026"), write YYYY-MM-01 and set date_precision to "month"; never invent '
    "a day. Fill date, date_precision and date_direction; set bound_type and value to null."
)


def _build_prompt(field_name: str, extracts: Sequence[tuple[int, str]], kinds: Sequence[str]) -> str:
    extract_block = "\n\n".join(f"[{number}] {text}" for number, text in extracts)
    sections = []
    if KIND_NUMERIC in kinds:
        sections.append(_NUMERIC_SECTION)
    if KIND_DATE in kinds:
        sections.append(_DATE_SECTION)
    kind_options = " | ".join(f'"{k}"' for k in kinds)

    return (
        "You are a careful analyst reading extracts from a bank's lending policy "
        f'documents. Find limits that apply to the field "{field_name}" '
        f'(in words: "{field_name.replace("_", " ")}").\n\n'
        "EXTRACTS:\n" + extract_block + "\n\n"
        "WHAT TO LOOK FOR:\n" + "\n".join(sections) + "\n\n"
        'HOW TO CLASSIFY EACH FINDING ("classification"):\n'
        '- "firm_rule": the document itself requires or limits this ("shall", "must", "may not", "is limited to").\n'
        '- "example": the text only illustrates ("for example", "e.g.", "some banks", "may have", "such as").\n'
        '- "guidance": recommended or typical practice ("should", "generally", "typically", "expected").\n\n'
        "RULES:\n"
        "- Use ONLY the extracts above. Never use outside knowledge. Never guess.\n"
        '- "quote" must be copied word for word from one extract: the single sentence that contains the value.\n'
        '- "evidence_id" is the number of the extract the quote came from.\n'
        '- Copy numbers exactly as written; write "thirty (30)" as 30. Do not calculate or convert anything.\n'
        '- "applies_to" is a short condition the value is limited to (for example "LTV below 90 percent"), or null.\n'
        "- If no extract states a limit for this field, return an empty list. An empty list is a correct answer.\n\n"
        "Reply with JSON only, in exactly this shape:\n"
        '{"findings": [{"kind": ' + kind_options + ", "
        '"bound_type": "minimum" | "maximum" | null, "value": <number or null>, "unit": "<unit or null>", '
        '"date": "<YYYY-MM-DD or null>", "date_precision": "day" | "month" | null, '
        '"date_direction": "on_or_after" | "after" | "on_or_before" | "before" | null, '
        '"applies_to": "<condition or null>", '
        '"classification": "firm_rule" | "example" | "guidance", '
        '"evidence_id": <number>, "quote": "<exact sentence>"}]}'
    )


# ----------------------------------------------------------------------------
# Checking the model's answer against the schema
# ----------------------------------------------------------------------------

def _number_from_text(raw: str) -> float | None:
    """For text that is not a plain number: 'thirty (30)' -> 30, 'seventy two' -> 72."""
    bracket = re.search(r"\(\s*(\d[\d,]*(?:\.\d+)?)\s*\)", raw)
    if bracket:
        try:
            return float(bracket.group(1).replace(",", ""))
        except ValueError:
            return None
    return words_to_number(raw)


def _to_number(raw: object) -> int | float | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        number = raw
    elif isinstance(raw, str):
        try:
            number = float(re.sub(r"[,$\s]", "", raw))
        except ValueError:
            number = _number_from_text(raw)
            if number is None:
                return None
    else:
        return None
    if isinstance(number, float) and (math.isnan(number) or math.isinf(number)):
        return None
    return int(number) if float(number).is_integer() else number


def _to_index(raw: object) -> int | None:
    if isinstance(raw, bool):
        return None
    try:
        number = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def _clean_text(raw: object, limit: int = 200) -> str | None:
    if not isinstance(raw, str):
        return None
    text = re.sub(r"\s+", " ", raw).strip()
    return text[:limit] if text else None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _attribute(item: dict, quote: str, evidence: Sequence[Evidence]) -> Evidence | None:
    """Work out which evidence chunk a finding belongs to, using only our own data."""
    index = _to_index(item.get("evidence_id"))
    if index is not None and 1 <= index <= len(evidence):
        return evidence[index - 1]
    # Fallback: the model gave a bad number, but the quote may sit in exactly one chunk.
    needle = _normalise(quote)
    matches = [e for e in evidence if needle in _normalise(e.text)]
    return matches[0] if len(matches) == 1 else None


def _validate(
    raw_findings: list, evidence: Sequence[Evidence], kinds: Sequence[str]
) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []
    rejected: list[str] = []
    seen: set[tuple] = set()

    for n, item in enumerate(raw_findings, 1):
        tag = f"finding #{n}"
        if not isinstance(item, dict):
            rejected.append(f"{tag}: not an object")
            continue

        kind = item.get("kind")
        if kind not in ALL_KINDS:
            rejected.append(f"{tag}: unknown kind {kind!r}")
            continue
        if kind not in kinds:
            rejected.append(f"{tag}: kind {kind!r} was not requested")
            continue

        classification = item.get("classification")
        if classification not in CLASSIFICATIONS:
            rejected.append(f"{tag}: classification {classification!r} is not one of {list(CLASSIFICATIONS)}")
            continue

        quote = _clean_text(item.get("quote"), limit=1000)
        if not quote:
            rejected.append(f"{tag}: no quote")
            continue

        source = _attribute(item, quote, evidence)
        if source is None:
            rejected.append(f"{tag}: could not be tied to any evidence extract")
            continue

        bound_type = value = date = date_direction = date_precision = None
        if kind == KIND_NUMERIC:
            bound_type = item.get("bound_type")
            value = _to_number(item.get("value"))
            if bound_type not in BOUND_TYPES:
                rejected.append(f"{tag}: bound_type {bound_type!r} is not 'minimum' or 'maximum'")
                continue
            if value is None:
                rejected.append(f"{tag}: value {item.get('value')!r} is not a number")
                continue
        else:
            date_direction = item.get("date_direction")
            raw_date = str(item.get("date")).strip()
            month_only = re.fullmatch(r"\d{4}-\d{2}", raw_date) is not None
            try:
                date = dt.date.fromisoformat(f"{raw_date}-01" if month_only else raw_date).isoformat()
            except ValueError:
                rejected.append(f"{tag}: date {item.get('date')!r} is not a valid YYYY-MM-DD (or YYYY-MM) date")
                continue
            date_precision = item.get("date_precision")
            if date_precision in (None, ""):
                date_precision = "month" if month_only else "day"
            if date_precision not in DATE_PRECISIONS:
                rejected.append(f"{tag}: date_precision {date_precision!r} is not 'day' or 'month'")
                continue
            if date_precision == "month" and not date.endswith("-01"):
                rejected.append(f"{tag}: date_precision is 'month' but the date {date} is not the 1st")
                continue
            if date_precision == "day" and month_only:
                rejected.append(f"{tag}: only a month was given but date_precision says 'day'")
                continue
            if date_direction not in DATE_DIRECTIONS:
                rejected.append(f"{tag}: date_direction {date_direction!r} is not one of {list(DATE_DIRECTIONS)}")
                continue

        unit = _clean_text(item.get("unit"), limit=40) if kind == KIND_NUMERIC else None
        key = (kind, bound_type, value, unit, date, date_direction, date_precision, source.chunk_id)
        if key in seen:
            continue  # the same finding twice is not two findings
        seen.add(key)

        findings.append(
            Finding(
                kind=kind,
                classification=classification,
                quote=quote,
                source_file=source.source_file,
                page_number=source.page_number,
                chunk_id=source.chunk_id,
                bound_type=bound_type,
                value=value,
                unit=unit,
                date=date,
                date_direction=date_direction,
                date_precision=date_precision,
                applies_to=_clean_text(item.get("applies_to")),
            )
        )
    return findings, rejected


# ----------------------------------------------------------------------------
# The node itself
# ----------------------------------------------------------------------------

def _choose_num_ctx(prompt: str, requested: int) -> int:
    """Ollama silently drops text that does not fit its window, so make sure it fits.
    Rough size: one token per 3 characters (generous), plus room for the answer."""
    needed = len(prompt) // 3 + 700
    if needed <= requested:
        return requested
    return min(32768, ((needed + 1023) // 1024) * 1024)


def extract_conditions(
    field_name: str,
    evidence: Sequence[Evidence],
    llm: _TextGenerator | None = None,
    kinds: Sequence[str] = ALL_KINDS,
    num_ctx: int = DEFAULT_NUM_CTX,
    focus: bool = True,
) -> ExtractionResult:
    """Ask the local LLM to extract limits for `field_name` from `evidence`.

    With focus=True (the default) paragraphs holding no number or date are not sent,
    and the others are trimmed to their number-bearing sentences -- see focus.py.
    Paragraphs keep their original numbers in the prompt, so attribution still works.
    """
    total = len(evidence)
    extracts: list[tuple[int, str]] = []
    for number, item in enumerate(evidence, 1):
        text = focus_text(item.text) if focus else item.text
        if text.strip():
            extracts.append((number, text))

    if not extracts:
        # Nothing to read: no paragraphs, or none that contain a number or a date.
        return ExtractionResult(field=field_name, status=STATUS_NOT_FOUND, paragraphs_total=total, paragraphs_sent=0)

    sent = len(extracts)
    try:
        if llm is None:
            from lending_agent.llm.ollama_client import OllamaClient

            llm = OllamaClient()
        prompt = _build_prompt(field_name, extracts, kinds)
        raw = llm.generate(prompt, format="json", temperature=0, num_ctx=_choose_num_ctx(prompt, num_ctx))
    except Exception as exc:  # Ollama down, model missing, timeout... -> an error, NOT "not found"
        return ExtractionResult(
            field=field_name, status=STATUS_ERROR, error=f"Could not use the LLM: {exc}",
            paragraphs_total=total, paragraphs_sent=sent,
        )

    data = parse_llm_json(raw)
    if data is None:
        return ExtractionResult(
            field=field_name, status=STATUS_ERROR, error="The model's reply was not valid JSON.",
            paragraphs_total=total, paragraphs_sent=sent,
        )
    raw_findings = data.get("findings")
    if not isinstance(raw_findings, list):
        return ExtractionResult(
            field=field_name, status=STATUS_ERROR, error="The model's reply had no 'findings' list.",
            paragraphs_total=total, paragraphs_sent=sent,
        )

    findings, rejected = _validate(raw_findings, evidence, kinds)
    status = STATUS_FOUND if findings else STATUS_NOT_FOUND
    return ExtractionResult(
        field=field_name, status=status, findings=findings, rejected=rejected,
        paragraphs_total=total, paragraphs_sent=sent,
    )
