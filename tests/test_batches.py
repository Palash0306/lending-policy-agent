"""Phase 9: batch runs. Fields run in order, every result is saved at once, and a restart loses nothing."""
from __future__ import annotations

import io
import json

import openpyxl
import pytest

from lending_agent.api.batches import (CANCELLED, DONE, INTERRUPTED, QUEUED, RUNNING, BatchManager, parse_fields)
from lending_agent.api.chat import ChatOptions, ChatService
from lending_agent.api.errors import ApiError
from lending_agent.api.sessions import SessionStore
from lending_agent.export_xlsx import build_workbook
from tests.api_helpers import (TENURE, ScriptedLLM, answer_for, evidence, not_found_answer, numeric)

OPTIONS = ChatOptions()


def make_chat(fail_on=(), on_field=None):
    """A ChatService whose agent is a fake: loan_tenure verifies, anything in fail_on crashes, others are not found."""
    calls = []

    def run(graph, query):
        calls.append(query)
        if on_field:
            on_field(query)
        if query in fail_on:
            raise RuntimeError(f"model crashed on {query}")
        if query == "loan_tenure":
            return answer_for([numeric("c1", TENURE, 30, page=12)], [evidence("c1", TENURE, page=12)])
        return not_found_answer(query)

    return ChatService(None, ScriptedLLM(), SessionStore(), build=lambda options: "graph", run=run), calls


class TestParseFields:
    @pytest.mark.parametrize("raw,expected", [
        ("loan_tenure\nloan_amount", ["loan_tenure", "loan_amount"]),
        ("loan_tenure, loan_amount; applicant_age", ["loan_tenure", "loan_amount", "applicant_age"]),
        ("- loan_tenure\n* loan_amount\n• applicant_age", ["loan_tenure", "loan_amount", "applicant_age"]),
        ("1. loan_tenure\n2) loan_amount\n(3) applicant_age", ["loan_tenure", "loan_amount", "applicant_age"]),
        ("  Loan_Tenure \nloan_tenure\nLOAN_TENURE\nloan_amount", ["Loan_Tenure", "loan_amount"]),   # repeats dropped, first kept
        ("\n\n a \n\n b ,, ;", ["a", "b"]),
        (["loan_tenure", "loan_amount, loan_tenure"], ["loan_tenure", "loan_amount"]),
        ("What is the maximum loan tenure?", ["What is the maximum loan tenure?"]),
        ("2024-rate", ["2024-rate"]),                                                                # a digit that is not numbering
    ])
    def test_cleaning(self, raw, expected):
        assert parse_fields(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", " , ; \n", [], None, "- \n* "])
    def test_empty_input_is_refused_clearly(self, raw):
        with pytest.raises(ApiError) as exc:
            parse_fields(raw)
        assert exc.value.status == 422 and "at least one field" in exc.value.message

    def test_too_many_fields(self):
        assert len(parse_fields([f"f{i}" for i in range(50)], 50)) == 50
        with pytest.raises(ApiError) as exc:
            parse_fields([f"f{i}" for i in range(51)], 50)
        assert exc.value.status == 422 and "51" in exc.value.message and "50" in exc.value.message

    def test_a_field_that_is_too_long(self):
        with pytest.raises(ApiError):
            parse_fields("x" * 501)


class TestRunning:
    def test_three_fields_one_crash_all_saved_and_one_workbook(self, tmp_path):
        manager, (chat, calls) = BatchManager(tmp_path), make_chat(fail_on={"loan_amount"})
        batch = manager.create("loan_tenure\nloan_amount\napplicant_age", OPTIONS)
        assert batch["status"] == QUEUED and batch["total"] == 3
        done = manager.run(batch["batch_id"], chat)
        assert calls == ["loan_tenure", "loan_amount", "applicant_age"]                        # in order, none skipped
        assert done["status"] == DONE and done["done"] == 3
        assert [(i["field"], i["route"]) for i in done["items"]] == [("loan_tenure", "verified"), ("loan_amount", "error"), ("applicant_age", "not_found")]
        assert done["items"][0]["headline"] == "maximum 30 years"
        wb = openpyxl.load_workbook(io.BytesIO(build_workbook(manager.items_for_export(batch["batch_id"]), {})))
        summary = [r for r in wb["Summary"].iter_rows(values_only=True) if r[0] in ("loan_tenure", "loan_amount", "applicant_age")]
        assert [(r[0], r[1]) for r in summary] == [("loan_tenure", "Verified"), ("loan_amount", "Error"), ("applicant_age", "Not found")]
        assert wb["Test values"]["A2"].value == "loan_tenure"

    def test_every_result_is_on_disk_the_moment_it_finishes(self, tmp_path):
        manager = BatchManager(tmp_path)
        batch_id = manager.create("loan_tenure\nloan_amount", OPTIONS)["batch_id"]
        seen = []

        def look(query):                                    # called while a field is being worked on
            path = tmp_path / batch_id / "results.jsonl"
            seen.append((query, len(path.read_text().splitlines()) if path.exists() else 0))

        chat, _ = make_chat(on_field=look)
        manager.run(batch_id, chat)
        assert seen == [("loan_tenure", 0), ("loan_amount", 1)]                                  # field 1 was already saved before field 2 began

    def test_cancel_stops_between_fields_and_keeps_what_finished(self, tmp_path):
        manager = BatchManager(tmp_path)
        batch_id = manager.create("loan_tenure\nloan_amount\napplicant_age", OPTIONS)["batch_id"]
        chat, calls = make_chat(on_field=lambda q: manager.cancel(batch_id) if q == "loan_amount" else None)
        result = manager.run(batch_id, chat)
        assert calls == ["loan_tenure", "loan_amount"]                                           # the current field finished, the rest skipped
        assert result["status"] == CANCELLED and result["done"] == 2
        assert len(manager.items_for_export(batch_id)) == 2

    def test_cancelling_a_batch_that_has_not_started_skips_it_entirely(self, tmp_path):
        manager = BatchManager(tmp_path)
        batch_id = manager.create("loan_tenure", OPTIONS)["batch_id"]
        assert manager.cancel(batch_id)["status"] == CANCELLED
        chat, calls = make_chat()
        manager.run(batch_id, chat)
        assert calls == [] and manager.describe(batch_id)["status"] == CANCELLED

    def test_cancelling_a_finished_batch_is_a_409(self, tmp_path):
        manager, (chat, _) = BatchManager(tmp_path), make_chat()
        batch_id = manager.create("loan_tenure", OPTIONS)["batch_id"]
        manager.run(batch_id, chat)
        with pytest.raises(ApiError) as exc:
            manager.cancel(batch_id)
        assert exc.value.status == 409

    def test_a_disk_problem_marks_the_batch_interrupted_instead_of_running_forever(self, tmp_path, monkeypatch):
        manager, (chat, _) = BatchManager(tmp_path), make_chat()
        batch_id = manager.create("loan_tenure", OPTIONS)["batch_id"]
        monkeypatch.setattr("lending_agent.api.batches.os.fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full")))
        assert manager.run(batch_id, chat)["status"] == INTERRUPTED


class TestRestart:
    def test_finished_batches_come_back_after_a_restart(self, tmp_path):
        first, (chat, _) = BatchManager(tmp_path), make_chat()
        batch_id = first.create("loan_tenure\napplicant_age", OPTIONS)["batch_id"]
        first.run(batch_id, chat)
        again = BatchManager(tmp_path)                                                           # a new server process
        described = again.describe(batch_id)
        assert described["status"] == DONE and described["done"] == 2 and described["items"][0]["headline"] == "maximum 30 years"
        assert [p["field"] for p in again.results(batch_id)] == ["loan_tenure", None] or len(again.results(batch_id)) == 2
        assert [b["batch_id"] for b in again.list_recent()] == [batch_id]

    @pytest.mark.parametrize("status", [QUEUED, RUNNING])
    def test_an_interrupted_batch_is_marked_but_still_exportable(self, tmp_path, status):
        manager, (chat, _) = BatchManager(tmp_path), make_chat()
        batch_id = manager.create("loan_tenure\nloan_amount\napplicant_age", OPTIONS)["batch_id"]
        manager.run(batch_id, chat)
        folder = tmp_path / batch_id
        meta = json.loads((folder / "batch.json").read_text())
        meta["status"] = status                                                                  # as if the server died mid-batch
        (folder / "batch.json").write_text(json.dumps(meta))
        (folder / "results.jsonl").write_text("".join(folder.joinpath("results.jsonl").read_text().splitlines(True)[:2]) + '{"half written')
        again = BatchManager(tmp_path)
        described = again.describe(batch_id)
        assert described["status"] == INTERRUPTED and described["done"] == 2                     # the torn last line is ignored
        assert len(again.items_for_export(batch_id)) == 2

    def test_junk_folders_are_ignored(self, tmp_path):
        (tmp_path / "not-a-batch").mkdir()
        (tmp_path / ("a" * 32)).mkdir()
        assert BatchManager(tmp_path).list_recent() == []


class TestManaging:
    def test_unknown_or_malformed_ids_are_404_and_cannot_escape_the_folder(self, tmp_path):
        manager = BatchManager(tmp_path)
        for bad in ("../../etc", "nope", "A" * 32, ""):
            with pytest.raises(ApiError) as exc:
                manager.describe(bad)
            assert exc.value.status == 404

    def test_delete_removes_files_but_not_a_running_batch(self, tmp_path):
        manager, (chat, _) = BatchManager(tmp_path), make_chat()
        batch_id = manager.create("loan_tenure", OPTIONS)["batch_id"]
        with pytest.raises(ApiError) as exc:
            manager.delete(batch_id)
        assert exc.value.status == 409
        manager.run(batch_id, chat)
        assert manager.delete(batch_id) == {"deleted": batch_id} and not (tmp_path / batch_id).exists()
        with pytest.raises(ApiError):
            manager.describe(batch_id)

    def test_nothing_finished_means_nothing_to_export(self, tmp_path):
        manager = BatchManager(tmp_path)
        batch_id = manager.create("loan_tenure", OPTIONS)["batch_id"]
        with pytest.raises(ApiError) as exc:
            manager.items_for_export(batch_id)
        assert exc.value.status == 404 and "nothing to export" in exc.value.message

    def test_bad_input_is_refused_before_anything_is_saved(self, tmp_path):
        manager = BatchManager(tmp_path, max_fields=2)
        for fields, options in [("", OPTIONS), ("a\nb\nc", OPTIONS), ("a", ChatOptions(kinds="everything"))]:
            with pytest.raises(ApiError):
                manager.create(fields, options)
        assert list(tmp_path.iterdir()) == []
