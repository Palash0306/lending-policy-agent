"""Phase 9: the Excel/CSV export. Every test reopens the file with openpyxl and checks what a person would see."""
from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime

import openpyxl
import pytest

from lending_agent.api.summary import build_payload
from lending_agent.export_xlsx import (TEST_HEADERS, build_csv, build_workbook, clean_text, export_filename, headline_text,
                                       load_items)
from tests.api_helpers import (EXAMPLE, TENURE, answer_for, error_answer, evidence, not_found_answer, numeric)

META = {"llm_model": "qwen2.5:7b", "embedding_model": "all-MiniLM-L6-v2", "documents": ["policy.pdf"],
        "generated_at": datetime(2026, 3, 4, 9, 30)}


def good(field="loan_tenure"):
    return build_payload(answer_for([numeric("c1", TENURE, 30, page=12)], [evidence("c1", TENURE, page=12)], field))


def sheet(data, name):
    return openpyxl.load_workbook(io.BytesIO(data))[name]


def rows(ws):
    return [list(r) for r in ws.iter_rows(values_only=True)]


class TestStructure:
    def test_five_sheets_with_the_agreed_names_and_properties(self):
        wb = openpyxl.load_workbook(io.BytesIO(build_workbook([good()], META)))
        assert wb.sheetnames == ["Summary", "Test values", "Findings", "Search log", "Read me"]
        assert wb.properties.creator == "Lending Policy Data Assurance Agent"

    def test_the_test_values_sheet_has_the_agreed_columns_and_real_cells(self):
        ws = sheet(build_workbook([good()], META), "Test values")
        assert [c.value for c in ws[1]] == TEST_HEADERS
        first = [c for c in ws[2]]
        assert [c.value for c in first][:3] == ["loan_tenure", "maximum 30 years", "Max - 1"]
        assert first[3].value == 29 and first[3].data_type == "n"                       # a real number, not text
        assert ws.freeze_panes == "A2" and ws.auto_filter.ref.startswith("A1:")
        assert ws.page_setup.orientation == "landscape" and ws.sheet_properties.pageSetUpPr.fitToPage

    def test_dates_are_real_date_cells(self):
        date_finding = {"kind": "date_cutoff", "grounded": True, "classification": "firm_rule", "date": "2026-01-01",
                        "date_direction": "on_or_after", "date_precision": "day", "quote": "on or after January 1, 2026",
                        "source_file": "p.pdf", "page_number": 2, "needs_review": False}
        ws = sheet(build_workbook([{"field": "effective", "route": "verified", "findings": [date_finding]}], META), "Test values")
        cell = ws.cell(row=3, column=4)
        assert cell.value == datetime(2026, 1, 1) and cell.number_format == "yyyy-mm-dd" and ws.cell(row=3, column=6).value == "Rule applies"
        assert ws.cell(row=2, column=4).value == datetime(2025, 12, 31)

    def test_decimal_values_keep_their_decimals(self):
        payload = build_payload(answer_for([numeric("c1", "shall not exceed 6.5 percent", 6.5, unit="percent")], [evidence("c1", "shall not exceed 6.5 percent")], "rate"))
        ws = sheet(build_workbook([payload], META), "Test values")
        assert [ws.cell(row=r, column=4).value for r in (2, 3, 4)] == [6.4, 6.5, 6.6] and ws.cell(row=2, column=4).number_format == "0.0"

    def test_confirm_rows_are_amber(self):
        text = "The limit is 5 months."
        payload = build_payload(answer_for([numeric("c1", text, 5, unit="months")], [evidence("c1", text)]))
        ws = sheet(build_workbook([payload], META), "Test values")
        confirm = [r for r in ws.iter_rows(min_row=2) if r[5].value == "Confirm with policy owner"]
        assert confirm and confirm[0][5].fill.fgColor.rgb.endswith("FFE699")

    def test_a_field_with_no_rules_gets_a_note_not_an_empty_sheet(self):
        ws = sheet(build_workbook([build_payload(not_found_answer())], META), "Test values")
        assert "No grounded firm rules" in ws["A2"].value


class TestSummary:
    def test_one_row_per_field_with_outcome_counts(self):
        items = [good("loan_tenure"), build_payload(not_found_answer("applicant_age")), build_payload(error_answer("Ollama is down", "x")),
                 build_payload(answer_for([numeric("c1", TENURE, 25)], [evidence("c1", TENURE)], "bad"))]
        values = rows(sheet(build_workbook(items, META), "Summary"))
        flat = {r[0]: r[1] for r in values if r[0] and r[1] is not None}
        assert (flat["Verified"], flat["Needs review"], flat["Not found"], flat["Error"]) == (1, 1, 1, 1)
        table = {r[0]: r for r in values if r[0] in ("loan_tenure", "applicant_age", "bad")}
        assert table["loan_tenure"][1:4] == ["Verified", "maximum 30 years", "policy.pdf"] and table["loan_tenure"][4] == 12
        assert table["applicant_age"][1] == "Not found"
        meta_rows = {r[0]: r[1] for r in values if r[0] in ("Language model", "Embedding model", "Documents indexed")}
        assert meta_rows == {"Language model": "qwen2.5:7b", "Embedding model": "all-MiniLM-L6-v2", "Documents indexed": "1: policy.pdf"}

    def test_headline_text(self):
        assert headline_text({"max_value": 30, "unit": "years"}) == "maximum 30 years"
        assert headline_text({"cutoff_date": "2026-01-01", "date_direction": "on_or_after"}) == "cutoff on or after 2026-01-01"
        assert headline_text({"findings": [{"grounded": True, "classification": "example"}]}).startswith("no firm rule")
        assert headline_text({}) == ""


class TestExamplesAndFindings:
    def test_examples_appear_only_when_asked_and_are_labelled(self):
        payload = build_payload(answer_for([numeric("c1", EXAMPLE, 72, unit="months", classification="example")], [evidence("c1", EXAMPLE)]))
        assert "No grounded firm rules" in sheet(build_workbook([payload], META), "Test values")["A2"].value
        ws = sheet(build_workbook([payload], {**META, "include_examples": True}), "Test values")
        assert ws["F2"].value == "Not a firm rule" and ws["L2"].value == "Not a firm rule"

    def test_the_findings_sheet_lists_everything_including_ungrounded_ones(self):
        payload = build_payload(answer_for([numeric("c1", TENURE, 25)], [evidence("c1", TENURE)]))
        ws = sheet(build_workbook([payload], META), "Findings")
        data = rows(ws)
        assert data[0][0] == "Field" and data[1][9] == "no" and "25" in data[1][11]       # grounded=no, reasons say why

    def test_conflicts_are_flagged_in_test_values(self):
        other = "The repayment period shall not extend beyond twenty-five (25) years."
        payload = build_payload(answer_for([numeric("c1", TENURE, 30), numeric("c2", other, 25, source="b.pdf", page=2)], [evidence("c1", TENURE), evidence("c2", other, "b.pdf", 2)]))
        ws = sheet(build_workbook([payload], META), "Test values")
        assert all("CONFLICT" in (ws.cell(row=r, column=13).value or "") for r in range(2, 8))


class TestSearchLog:
    def test_terms_paragraphs_timings_and_attempts_are_shown(self):
        payload = good()
        payload["search"] = {"terms": ["loan tenure", "repayment period"], "terms_source": "seed+llm", "note": None, "paragraphs_sent": 2,
                             "paragraphs": [{"chunk_id": "c1", "source_file": "policy.pdf", "page_number": 12, "section": "Terms", "similarity": 0.61,
                                             "matched_terms": ["a", "b"], "text": "x" * 500}]}
        payload["trace"] = [{"node": "expand", "seconds": 1.5}, {"node": "extract", "seconds": 200.25}]
        payload["attempts"] = 2
        text = [[c for c in r if c is not None] for r in rows(sheet(build_workbook([payload], META), "Search log"))]
        flat = [x for r in text for x in r]
        assert "loan tenure, repayment period" in flat and "Built-in list + model" in flat and 2 in flat and "expand: 1.5, extract: 200.25" in flat
        paragraph = [r for r in text if r[:2] == ["policy.pdf", 12]][0]
        assert paragraph[2:5] == ["Terms", 0.61, 2] and len(paragraph[5]) == 300                 # first 300 characters only

    def test_an_answer_without_search_details_still_exports(self):
        flat = [c for r in rows(sheet(build_workbook([build_payload(not_found_answer())], META), "Search log")) for c in r if c]
        assert any("No search details" in str(c) for c in flat)


class TestSafety:
    EVIL = '=HYPERLINK("http://evil.example","click") shall not exceed 30 years'

    def _payload(self, quote=None):
        quote = quote or self.EVIL
        return build_payload(answer_for([numeric("c1", quote, 30)], [evidence("c1", quote)], "=cmd|' /C calc'!A0"))

    def test_text_that_starts_like_a_formula_is_stored_as_text_not_as_a_formula(self):
        data = build_workbook([self._payload()], META)
        archive = zipfile.ZipFile(io.BytesIO(data))
        for name in archive.namelist():
            if name.startswith("xl/worksheets/"):
                assert b"<f>" not in archive.read(name), name                              # no formula anywhere in the file
        ws = sheet(data, "Test values")
        quote_cell = next(c for c in ws["J"] if isinstance(c.value, str) and c.value.startswith("=HYPERLINK("))
        assert quote_cell.data_type == "s" and quote_cell.quotePrefix
        assert ws["A2"].value.startswith("=cmd") and ws["A2"].data_type == "s"           # the field name too

    @pytest.mark.parametrize("lead", ["=", "+", "-", "@", "\t", "\r"])
    def test_every_dangerous_first_character_is_neutralised_in_xlsx_and_csv(self, lead):
        quote = f"{lead}1+1 shall not exceed 30 years"
        payload = self._payload(quote)
        ws = sheet(build_workbook([payload], META), "Test values")
        cell = next(c for c in ws["J"][1:] if c.value is not None)
        assert cell.data_type == "s"
        body = build_csv([payload], META).decode("utf-8-sig")
        quote_field = list(csv.reader(io.StringIO(body)))[1][9]
        assert quote_field.startswith("'" + lead)

    def test_characters_xlsx_cannot_hold_are_removed_and_long_text_is_cut(self):
        assert clean_text("a\x00b\x07c") == "abc" and len(clean_text("z" * 40000)) < 33000
        payload = self._payload("shall not exceed 30 years\x00\x1f")
        assert sheet(build_workbook([payload], META), "Test values")["J2"].value == "shall not exceed 30 years"


class TestCsv:
    def test_utf8_with_byte_order_mark_and_the_same_columns(self):
        data = build_csv([good()], META)
        assert data.startswith(b"\xef\xbb\xbf")
        table = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
        assert table[0] == TEST_HEADERS and table[1][:4] == ["loan_tenure", "maximum 30 years", "Max - 1", "29"]

    def test_accented_text_survives(self):
        quote = "Le prêt ne doit pas dépasser 30 years, up to 30 years"
        payload = build_payload(answer_for([numeric("c1", quote, 30)], [evidence("c1", quote)]))
        assert "prêt" in build_csv([payload], META).decode("utf-8-sig")


class TestLoader:
    def test_accepts_a_payload_a_run_agent_dict_and_lists(self):
        payload = good()
        agent_dict = {"field": "loan_tenure", "route": "verified", "summary": "ok", "attempts": 1, "widened_search": False,
                      "result": {"field": "loan_tenure", "status": "found", "verified": True, "needs_review": False,
                                 "findings": [{"kind": "numeric_boundary", "bound_type": "maximum", "value": 30, "unit": "years", "classification": "firm_rule",
                                               "grounded": True, "needs_review": False, "quote": TENURE, "source_file": "policy.pdf", "page_number": 12}],
                                 "conflicts": [], "rejected": [], "error": None}, "trace": []}
        assert load_items(payload) == [payload]
        item = load_items(agent_dict)[0]
        assert item["max_value"] == 30 and item["source_file"] == "policy.pdf"
        assert len(load_items([payload, agent_dict, [payload]])) == 3
        assert sheet(build_workbook(agent_dict, META), "Test values")["D2"].value == 29

    def test_rejects_things_that_are_not_answers(self):
        with pytest.raises(ValueError):
            load_items("nope")
        with pytest.raises(ValueError):
            load_items({"hello": 1})


def test_file_name_pattern():
    assert export_filename("xlsx", datetime(2026, 3, 4, 9, 5)) == "lending-policy-test-values_2026-03-04_0905.xlsx"
