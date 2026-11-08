"""
Phase 9 -- the Excel (and CSV) export.

`build_workbook(items, meta)` returns the bytes of ONE .xlsx file holding every field. `items` are answer payloads:
the dict stored in a chat turn's payload, a batch result, or the JSON that scripts/run_agent.py writes (load_items
accepts all of them, alone or in a list).

SAFETY. Text in these files comes from bank policy documents and can contain anything. A cell whose text starts
with = + - @ (or a tab / carriage return) could be run as a formula by a spreadsheet program, so such text is
stored as a plain string (forced string type plus the "quote prefix" flag in xlsx; a leading apostrophe in csv).
Control characters that xlsx cannot hold are removed. Nothing is ever written as a formula, macro or link.
"""
from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.properties import PageSetupProperties

from lending_agent.testcases import (ACCEPT, APPLIES, CONFIRM, CONFIRM_MONTH, NOT_A_RULE, NOT_APPLIES, REJECT,
                                     decimals_in, generate_cases)

TOOL_NAME = "Lending Policy Data Assurance Agent"
DISCLAIMER = "Values are candidates for test design; confirm each against the cited page before use."
FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")
_ILLEGAL_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
MAX_CELL = 32000
OUTCOMES = [("verified", "Verified"), ("needs_review", "Needs review"), ("not_found", "Not found"), ("error", "Error")]
TERMS_SOURCE = {"seed+llm": "Built-in list + model", "seed": "Built-in list", "llm": "Model", "input": "As typed"}

TEST_HEADERS = ["Field", "Rule", "Case", "Test value", "Unit", "Expected result", "Basis", "Source file", "Page", "Quote",
                "Classification", "Review status", "Notes"]
FINDING_HEADERS = ["Field", "Kind", "Rule", "Value", "Unit", "Date", "Date direction", "Applies to", "Classification",
                   "Grounded", "Needs review", "Review reasons", "Conflicts with finding #", "Source file", "Page", "Quote"]
SUMMARY_HEADERS = ["Field", "Outcome", "Headline limit(s)", "Source file", "Page", "Classification", "Grounded",
                   "Needs review", "Seconds"]

_HEAD_FILL = PatternFill("solid", fgColor="1F3864")
_AMBER = PatternFill("solid", fgColor="FFE699")
_GREY = PatternFill("solid", fgColor="EDEDED")
_THIN = Side(style="thin", color="BFBFBF")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_BASE = Font(name="Calibri", size=10)
_BOLD = Font(name="Calibri", size=10, bold=True)
_HEAD_FONT = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
_TITLE = Font(name="Calibri", size=14, bold=True)


# ---------------------------------------------------------------------------------------------
# reading the input
# ---------------------------------------------------------------------------------------------
def _headline_from_findings(findings: list[dict]) -> dict:
    """Headline values from the findings alone (used for the run_agent.py JSON shape, which has none)."""
    firm = [f for f in findings if f.get("grounded") and f.get("classification") == "firm_rule"]
    out: dict[str, Any] = {}
    for bound, name in (("minimum", "min_value"), ("maximum", "max_value")):
        picks = [f for f in firm if f.get("kind") == "numeric_boundary" and f.get("bound_type") == bound]
        if picks and len({(f.get("value"), f.get("unit")) for f in picks}) == 1:
            out[name], out["unit"] = picks[0]["value"], picks[0].get("unit")
            out.setdefault("source_file", picks[0].get("source_file"))
            out.setdefault("page_number", picks[0].get("page_number"))
    picks = [f for f in firm if f.get("kind") == "date_cutoff"]
    if picks and len({(f.get("date"), f.get("date_direction")) for f in picks}) == 1:
        out.update(cutoff_date=picks[0]["date"], date_direction=picks[0].get("date_direction"),
                   date_precision=picks[0].get("date_precision"))
        out.setdefault("source_file", picks[0].get("source_file"))
        out.setdefault("page_number", picks[0].get("page_number"))
    return out


def _from_agent_dict(obj: dict) -> dict:
    result = obj["result"]
    findings = list(result.get("findings") or [])
    item = {"field": obj.get("field") or result.get("field"), "route": obj.get("route"), "agent_route": obj.get("route"),
            "answer": obj.get("summary", ""), "findings": findings, "conflicts": list(result.get("conflicts") or []),
            "rejected": list(result.get("rejected") or []), "error": result.get("error"),
            "attempts": obj.get("attempts"), "widened_search": obj.get("widened_search"),
            "trace": list(obj.get("trace") or []), "search": obj.get("search") or {},
            "review_reasons": [], "grounded": bool(result.get("verified")), "needs_review": bool(result.get("needs_review"))}
    item.update(_headline_from_findings(findings))
    return item


def load_items(obj: Any) -> list[dict]:
    """Accept one answer payload, {"result": {...}} (run_agent.py), or a list of either. Returns payload-shaped dicts."""
    if isinstance(obj, (list, tuple)):
        return [item for part in obj for item in load_items(part)]
    if isinstance(obj, dict):
        if isinstance(obj.get("result"), dict) and "findings" in obj["result"]:
            return [_from_agent_dict(obj)]
        if "findings" in obj or "route" in obj:
            return [obj]
    raise ValueError("This is not an answer the exporter understands.")


def field_name(item: dict) -> str:
    """The name to show for a field. A crashed answer has none of its own, so fall back to what was asked."""
    return str(item.get("field") or item.get("batch_field") or item.get("query_used") or item.get("message") or "")


def outcome_label(route: str | None) -> str:
    return dict(OUTCOMES).get(route or "", route or "")


def _seconds(item: dict) -> float | None:
    if item.get("elapsed_seconds") is not None:
        return item["elapsed_seconds"]
    trace = [t.get("seconds") for t in item.get("trace") or [] if t.get("seconds") is not None]
    return round(sum(trace), 1) if trace else None


def headline_text(item: dict) -> str:
    parts, unit = [], f" {item['unit']}" if item.get("unit") else ""
    for key, word in (("min_value", "minimum"), ("max_value", "maximum")):
        if item.get(key) is not None:
            parts.append(f"{word} {item[key]}{unit}")
    if item.get("cutoff_date"):
        words = {"on_or_after": "on or after", "after": "after", "on_or_before": "on or before", "before": "before"}
        parts.append(f"cutoff {words.get(item.get('date_direction') or '', item.get('date_direction') or '')} {item['cutoff_date']}")
    if parts:
        return "; ".join(parts)
    findings = [f for f in item.get("findings") or [] if f.get("grounded")]
    if any(f.get("classification") == "firm_rule" for f in findings):
        return "several values (tiers or a conflict): see Test values"
    return "no firm rule, only examples or guidance: see Findings" if item.get("findings") else ""


def _classes(item: dict) -> str:
    if item.get("min_value") is not None or item.get("max_value") is not None or item.get("cutoff_date"):
        return "firm_rule"
    return ", ".join(dict.fromkeys(f.get("classification") for f in item.get("findings") or []
                                   if f.get("grounded") and f.get("classification")))


# ---------------------------------------------------------------------------------------------
# writing cells safely
# ---------------------------------------------------------------------------------------------
def clean_text(value: Any) -> str:
    text = _ILLEGAL_XML.sub("", str(value))
    return text if len(text) <= MAX_CELL else text[:MAX_CELL] + "…"


def _put(ws, row: int, col: int, value: Any, *, bold=False, wrap=False, fill=None, number_format=None, head=False):
    cell = ws.cell(row=row, column=col)
    if isinstance(value, bool):
        value = "yes" if value else "no"
    if isinstance(value, Decimal):
        digits = decimals_in(value)
        value, number_format = (int(value) if digits == 0 else float(value)), number_format or ("0" if digits == 0 else "0." + "0" * digits)
    elif isinstance(value, float) and value.is_integer() and number_format is None:
        value = int(value)
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        cell.value, cell.number_format = value, number_format or "yyyy-mm-dd"
    elif isinstance(value, (int, float)):
        cell.value = value
        if number_format:
            cell.number_format = number_format
    elif value is None or value == "":
        cell.value = None
    else:
        text = clean_text(value)
        cell.value = text
        if text.startswith(FORMULA_STARTS):          # never a formula: force a plain string and set the quote-prefix flag
            cell.data_type = "s"
            cell.quotePrefix = True
    cell.font = _HEAD_FONT if head else (_BOLD if bold else _BASE)
    cell.alignment = Alignment(vertical="top", wrap_text=wrap or head, horizontal="left" if not isinstance(value, (int, float)) else None)
    if head:
        cell.fill = _HEAD_FILL
    elif fill is not None:
        cell.fill = fill
    return cell


def _table(ws, top: int, headers: list[str], rows: list[list[Any]], wrap_cols: Iterable[int] = (), amber=None) -> int:
    """Write a header row and data rows; returns the next free row."""
    for c, h in enumerate(headers, 1):
        _put(ws, top, c, h, head=True)
        ws.cell(row=top, column=c).border = _BORDER
    wrap_cols = set(wrap_cols)
    for r, values in enumerate(rows, top + 1):
        fill = _AMBER if amber and amber(values) else None
        for c, v in enumerate(values, 1):
            _put(ws, r, c, v, wrap=c in wrap_cols, fill=fill).border = _BORDER
    return top + 1 + len(rows)


def _finish(ws, widths: list[float], header_row: int, last_row: int, filter_ref: bool = True) -> None:
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
    if filter_ref and last_row > header_row:
        ws.auto_filter.ref = f"A{header_row}:{get_column_letter(len(widths))}{last_row}"
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    ws.print_title_rows = f"{header_row}:{header_row}"
    ws.sheet_view.showGridLines = False


# ---------------------------------------------------------------------------------------------
# the sheets
# ---------------------------------------------------------------------------------------------
def _documents_text(meta: dict) -> str:
    docs = meta.get("documents")
    if isinstance(docs, (list, tuple)):
        return f"{len(docs)}: " + ", ".join(str(d) for d in docs) if docs else "0"
    return "" if docs is None else str(docs)


def _summary_sheet(ws, items: list[dict], meta: dict) -> None:
    generated = meta.get("generated_at") or datetime.now()
    _put(ws, 1, 1, "Lending policy test values: summary", bold=True).font = _TITLE
    info = [("Generated", generated.strftime("%Y-%m-%d %H:%M")), ("Tool", TOOL_NAME),
            ("Language model", meta.get("llm_model") or ""), ("Embedding model", meta.get("embedding_model") or ""),
            ("Documents indexed", _documents_text(meta)), ("Fields in this file", len(items))]
    for i, (k, v) in enumerate(info, 2):
        _put(ws, i, 1, k, bold=True)
        _put(ws, i, 2, v)
    row = len(info) + 3
    counts = {key: sum(1 for it in items if it.get("route") == key) for key, _ in OUTCOMES}
    for key, label in OUTCOMES:
        _put(ws, row, 1, label, bold=True)
        _put(ws, row, 2, counts[key])
        row += 1
    _put(ws, row, 1, DISCLAIMER, wrap=False).font = Font(name="Calibri", size=10, italic=True)
    top = row + 2
    rows = [[field_name(it), outcome_label(it.get("route")), headline_text(it), it.get("source_file"),
             it.get("page_number"), _classes(it), bool(it.get("grounded")), bool(it.get("needs_review")), _seconds(it)] for it in items]
    last = _table(ws, top, SUMMARY_HEADERS, rows, wrap_cols=(3,), amber=lambda v: v[1] in ("Needs review", "Error"))
    _finish(ws, [28, 16, 40, 28, 8, 16, 10, 13, 10], top, last - 1)


def _case_rows(items: list[dict], include_examples: bool, step: Any) -> list[list[Any]]:
    rows = []
    for item in items:
        for case in generate_cases(item.get("findings") or [], step=step, include_examples=include_examples):
            rows.append([field_name(item), case["rule"], case["case"], case["value"], case["unit"], case["expected"],
                         case["basis"], case["source_file"], case["page_number"], case["quote"], case["classification"],
                         case["review_status"], case["notes"]])
    return rows


def _needs_attention(row: list[Any]) -> bool:
    return str(row[5]).startswith("Confirm") or row[5] == NOT_A_RULE or row[11] != "Verified" or "CONFLICT" in str(row[12])


def _test_sheet(ws, items: list[dict], include_examples: bool, step: Any) -> None:
    rows = _case_rows(items, include_examples, step)
    last = _table(ws, 1, TEST_HEADERS, rows, wrap_cols=(2, 10, 13), amber=_needs_attention)
    if not rows:
        _put(ws, 2, 1, "No grounded firm rules were found, so there are no test values. See the Summary and Findings sheets.")
    _finish(ws, [24, 34, 18, 13, 12, 22, 11, 26, 7, 60, 14, 14, 40], 1, max(last - 1, 1))


def _finding_rule(f: dict) -> str:
    if f.get("kind") == "numeric_boundary":
        return f"{f.get('bound_type')} {f.get('value')}" + (f" {f['unit']}" if f.get("unit") else "")
    return f"{f.get('date_direction')} {f.get('date')}"


def _as_date(value: Any) -> Any:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return value


def _findings_sheet(ws, items: list[dict]) -> None:
    rows = []
    for item in items:
        for f in item.get("findings") or []:
            rows.append([field_name(item), "limit" if f.get("kind") == "numeric_boundary" else "date", _finding_rule(f),
                         f.get("value"), f.get("unit"), _as_date(f.get("date")), f.get("date_direction"), f.get("applies_to"),
                         f.get("classification"), bool(f.get("grounded")), bool(f.get("needs_review")),
                         "; ".join(f.get("review_reasons") or []), ", ".join(str(n) for n in f.get("conflicts_with") or []),
                         f.get("source_file"), f.get("page_number"), f.get("quote")])
    last = _table(ws, 1, FINDING_HEADERS, rows, wrap_cols=(12, 16),
                  amber=lambda v: v[9] == "no" or v[10] == "yes")
    _finish(ws, [24, 8, 26, 10, 12, 12, 14, 24, 14, 10, 12, 40, 12, 26, 7, 60], 1, max(last - 1, 1))


def _search_sheet(ws, items: list[dict]) -> None:
    row = 1
    for item in items:
        search, trace = item.get("search") or {}, item.get("trace") or []
        _put(ws, row, 1, field_name(item), bold=True, fill=_GREY)
        for c in range(2, 7):
            _put(ws, row, c, None, fill=_GREY)
        row += 1
        facts = [("Phrasings searched", ", ".join(search.get("terms") or [])),
                 ("Where the phrasings came from", TERMS_SOURCE.get(search.get("terms_source"), search.get("terms_source") or "")),
                 ("Note", search.get("note") or ""), ("Extraction attempts", item.get("attempts")),
                 ("Search widened", bool(item.get("widened_search"))), ("Paragraphs sent to the model", search.get("paragraphs_sent")),
                 ("Step timings (seconds)", ", ".join(f"{t.get('node')}: {t.get('seconds')}" for t in trace))]
        for label, value in facts:
            _put(ws, row, 1, label, bold=True)
            _put(ws, row, 2, value, wrap=True)
            ws.merge_cells(start_row=row, start_column=2, end_row=row, end_column=6)
            row += 1
        paragraphs = search.get("paragraphs") or []
        if paragraphs:
            rows = [[p.get("source_file"), p.get("page_number"), p.get("section"), p.get("similarity"), len(p.get("matched_terms") or []),
                     (p.get("text") or "")[:300]] for p in paragraphs]
            row = _table(ws, row, ["Source file", "Page", "Section", "Similarity", "Found by (phrasings)", "Text (first 300 characters)"],
                         rows, wrap_cols=(3, 6))
        else:
            _put(ws, row, 1, "No paragraphs were retrieved." if search else "No search details were recorded for this answer.")
            row += 1
        row += 1
    for i, w in enumerate([30, 8, 18, 11, 18, 80], 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr = PageSetupProperties(fitToPage=True)
    ws.sheet_view.showGridLines = False


_READ_ME = [
    ("How to read this file", None),
    ("Summary", "One row per field: the outcome and the headline limit. Start here."),
    ("Test values", "Boundary values to try for each verified rule, with the expected result, the quote and the page it came from."),
    ("Findings", "Everything the agent extracted, including findings that failed the checks or are only examples."),
    ("Search log", "What was searched and read for each field, so a reviewer can see where an answer came from."),
    ("", None),
    ("Expected result: what each value means", None),
    (ACCEPT, "The system should accept this value."),
    (REJECT, "The system should reject this value."),
    (CONFIRM, "The quote does not say whether the limit value itself is allowed. Ask the policy owner. No guess is made."),
    (APPLIES, "For date rules: the rule is in force on this date."),
    (NOT_APPLIES, "For date rules: the rule is not in force on this date."),
    (CONFIRM_MONTH, "The document gives only a month, so the 1st is used as a stand-in. Confirm the exact day."),
    (NOT_A_RULE, "Only an example or guidance in the document, not a firm requirement. Included only when asked for."),
    ("", None),
    ("Other columns", None),
    ("Basis", "Why the limit value itself is accepted or rejected: inclusive (\"up to\", \"at least\"), exclusive (\"less than\", \"above\") or unclear."),
    ("Review status", "Verified = the quote, number or date was found in the source text. Needs review = a person should check first."),
    ("Amber rows", "Need a person's attention: unclear wording, a failed check, a conflict, or not a firm rule."),
    ("Test value step", "One unit of the last decimal place stated in the policy (6.5 gives 6.4 and 6.6; 90 gives 89 and 91). Units are never converted."),
    ("", None),
    ("Known limitations", None),
    ("1", "A value is only as reliable as the rule that was extracted. Always check the quote on the cited page."),
    ("2", "Rules the agent did not retrieve or could not read are not here. Not found does not mean the policy has no limit."),
    ("3", "Scanned (image-only) PDFs cannot be read."),
    ("", None),
    (DISCLAIMER, None),
]


def _readme_sheet(ws) -> None:
    for i, (a, b) in enumerate(_READ_ME, 1):
        head = b is None and a != "" and a != DISCLAIMER
        _put(ws, i, 1, a, bold=True)
        if head:
            ws.cell(row=i, column=1).font = Font(name="Calibri", size=11, bold=True, color="1F3864")
        if b:
            _put(ws, i, 2, b, wrap=True)
    ws.cell(row=len(_READ_ME), column=1).font = Font(name="Calibri", size=10, italic=True, bold=True)
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 110
    ws.sheet_view.showGridLines = False


# ---------------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------------
def build_workbook(items: Iterable[dict], meta: dict | None = None) -> bytes:
    """One .xlsx (bytes) with every field. `meta` may hold: generated_at, llm_model, embedding_model, documents,
    include_examples, step."""
    items, meta = load_items(items if isinstance(items, (dict, list, tuple)) else list(items)), dict(meta or {})
    include_examples, step = bool(meta.get("include_examples")), meta.get("step")
    wb = Workbook()
    wb.properties.creator = TOOL_NAME
    wb.properties.title = "Lending policy test values"
    wb.properties.lastModifiedBy = TOOL_NAME
    ws = wb.active
    ws.title = "Summary"
    _summary_sheet(ws, items, meta)
    _test_sheet(wb.create_sheet("Test values"), items, include_examples, step)
    _findings_sheet(wb.create_sheet("Findings"), items)
    _search_sheet(wb.create_sheet("Search log"), items)
    _readme_sheet(wb.create_sheet("Read me"))
    buffer = io.BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    text = _ILLEGAL_XML.sub("", str(value))
    return "'" + text if text.startswith(FORMULA_STARTS) else text


def build_csv(items: Iterable[dict], meta: dict | None = None) -> bytes:
    """The "Test values" sheet as UTF-8 csv with a byte-order mark (so Excel reads accents correctly)."""
    items, meta = load_items(items if isinstance(items, (dict, list, tuple)) else list(items)), dict(meta or {})
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow(TEST_HEADERS)
    for row in _case_rows(items, bool(meta.get("include_examples")), meta.get("step")):
        writer.writerow([_csv_value(v) for v in row])
    return out.getvalue().encode("utf-8-sig")


def export_filename(ext: str, now: datetime | None = None) -> str:
    return f"lending-policy-test-values_{(now or datetime.now()):%Y-%m-%d_%H%M}.{ext}"
