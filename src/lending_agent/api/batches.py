"""
Phase 9 -- batch runs: many fields in one go, saved to disk after every field.

A batch is ONE job on the single worker (see jobs.py), so it never runs at the same time as a question or an upload.
Fields run one after another through exactly the same agent as /chat (ChatService.answer_one). One failing field is
recorded as an error and the batch carries on.

Every finished field is written to disk at once (BATCH_DIR/<batch_id>/results.jsonl, flushed), so a crash, a freeze
or a restart loses nothing that was already finished. After a restart, a batch that was queued or running is marked
"interrupted": it cannot continue, but what it finished can still be exported.

No web framework in this file.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from lending_agent.api.chat import KIND_CHOICES, ChatOptions
from lending_agent.api.errors import ApiError

MAX_FIELD_CHARS = 500
QUEUED, RUNNING, DONE, CANCELLED, INTERRUPTED = "queued", "running", "done", "cancelled", "interrupted"
_BATCH_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_BULLET_RE = re.compile(r"^\s*(?:[-*•·–]+|\(?\d{1,3}[.)])\s+")
_SPLIT_RE = re.compile(r"[\n\r;,]+")


def parse_fields(raw: Any, max_fields: int = 50) -> list[str]:
    """Turn what a person typed (or pasted from a file) into a clean list of fields.

    Splits on new lines, commas and semicolons; strips bullets and numbering ("- ", "1. ", "2) "); drops blanks;
    removes repeats (ignoring upper/lower case) keeping the first and the original order."""
    text = "\n".join(str(x) for x in raw) if isinstance(raw, (list, tuple)) else str(raw or "")
    seen, fields = set(), []
    for part in _SPLIT_RE.split(text):
        field = _BULLET_RE.sub("", part).strip()
        if not field or field.lower() in seen:
            continue
        if len(field) > MAX_FIELD_CHARS:
            raise ApiError(422, f"A field is longer than {MAX_FIELD_CHARS} characters: '{field[:40]}…'")
        seen.add(field.lower())
        fields.append(field)
    if not fields:
        raise ApiError(422, "Enter at least one field.")
    if len(fields) > max_fields:
        raise ApiError(422, f"Too many fields ({len(fields)}); the limit is {max_fields} per batch.")
    return fields


def check_batch_id(batch_id: str) -> str:
    if not _BATCH_ID_RE.match(batch_id or ""):
        raise ApiError(404, "Unknown batch.")
    return batch_id


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _headline(payload: dict) -> str:
    from lending_agent.export_xlsx import headline_text

    return headline_text(payload)


class BatchManager:
    def __init__(self, base_dir: str | Path, max_fields: int = 50) -> None:
        self.base_dir, self.max_fields = Path(base_dir), max_fields
        self._lock = threading.RLock()
        self._meta: dict[str, dict] = {}
        self._items: dict[str, list[dict]] = {}       # per batch: [{field, route, headline, seconds}] (full payloads stay on disk)
        self._cancel: set[str] = set()
        self._current: dict[str, str | None] = {}
        self.load_existing()

    # ---- disk -----------------------------------------------------------------------------------
    def _dir(self, batch_id: str) -> Path:
        return self.base_dir / check_batch_id(batch_id)

    def _write_meta(self, batch_id: str) -> None:
        folder = self._dir(batch_id)
        folder.mkdir(parents=True, exist_ok=True)
        temp = folder / "batch.json.tmp"
        temp.write_text(json.dumps(self._meta[batch_id], indent=2), encoding="utf-8")
        os.replace(temp, folder / "batch.json")

    def _read_results(self, batch_id: str) -> list[dict]:
        path = self._dir(batch_id) / "results.jsonl"
        rows = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue                           # a half-written last line after a crash: ignore it
        return rows

    def load_existing(self) -> None:
        """Pick up batches saved by an earlier run of the server. Unfinished ones become 'interrupted'."""
        if not self.base_dir.exists():
            return
        with self._lock:
            for folder in sorted(self.base_dir.iterdir()):
                meta_file = folder / "batch.json"
                if not (folder.is_dir() and _BATCH_ID_RE.match(folder.name) and meta_file.exists()):
                    continue
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except ValueError:
                    continue
                self._meta[folder.name] = meta
                self._items[folder.name] = [self._summarise(r) for r in self._read_results(folder.name)]
                if meta.get("status") in (QUEUED, RUNNING):
                    meta["status"], meta["finished_at"] = INTERRUPTED, meta.get("finished_at") or _now()
                    self._write_meta(folder.name)

    @staticmethod
    def _summarise(row: dict) -> dict:
        payload = row.get("payload") or {}
        return {"field": row.get("field"), "route": payload.get("route"), "headline": _headline(payload),
                "seconds": payload.get("elapsed_seconds")}

    # ---- using it ---------------------------------------------------------------------------------
    def create(self, fields: Any, options: ChatOptions) -> dict:
        names = parse_fields(fields, self.max_fields)
        if options.kinds not in KIND_CHOICES:
            raise ApiError(422, f"kinds must be one of {', '.join(KIND_CHOICES)}.")
        batch_id = uuid.uuid4().hex
        meta = {"batch_id": batch_id, "status": QUEUED, "fields": names, "total": len(names), "created_at": _now(),
                "started_at": None, "finished_at": None,
                "options": {"source_file": options.source_file, "kinds": options.kinds, "fast": options.fast}}
        with self._lock:
            self._meta[batch_id], self._items[batch_id], self._current[batch_id] = meta, [], None
            self._write_meta(batch_id)
        return self.describe(batch_id)

    def run(self, batch_id: str, chat: Any) -> dict:
        """Run every field, one after another (called on the worker thread)."""
        with self._lock:
            meta = self._meta[batch_id]
            if meta["status"] != QUEUED:               # cancelled while it was waiting in the queue
                return self.describe(batch_id)
            meta["status"], meta["started_at"] = RUNNING, _now()
            self._write_meta(batch_id)
        options = ChatOptions(**meta["options"])
        try:
            self._run_fields(batch_id, meta, chat, options)
        except Exception as exc:                       # e.g. the disk is full: say so instead of staying "running" forever
            with self._lock:
                meta["status"], meta["error"] = INTERRUPTED, f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                if meta["status"] == RUNNING:
                    meta["status"] = DONE
                meta["finished_at"], self._current[batch_id] = _now(), None
                self._cancel.discard(batch_id)
                self._write_meta(batch_id)
        return self.describe(batch_id)

    def _run_fields(self, batch_id: str, meta: dict, chat: Any, options: ChatOptions) -> None:
        for index, field in enumerate(meta["fields"]):
            with self._lock:
                if batch_id in self._cancel:
                    meta["status"] = CANCELLED
                    return
                self._current[batch_id] = field
            began = time.monotonic()
            try:
                payload = chat.answer_one(field, options)
            except Exception as exc:                   # answer_one does not raise; this is a safety net
                payload = {"field": field, "route": "error", "error": f"{type(exc).__name__}: {exc}",
                           "answer": f"I could not answer this: {exc}", "findings": [], "conflicts": [], "rejected": []}
            payload.setdefault("elapsed_seconds", round(time.monotonic() - began, 1))
            payload["batch_field"] = field
            payload["field"] = payload.get("field") or field        # a crashed field has no name of its own
            row = {"index": index, "field": field, "payload": payload, "saved_at": _now()}
            with self._lock:
                with (self._dir(batch_id) / "results.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                self._items[batch_id].append(self._summarise(row))

    def describe(self, batch_id: str) -> dict:
        with self._lock:
            meta = self._meta.get(check_batch_id(batch_id))
            if meta is None:
                raise ApiError(404, "Unknown batch.")
            items = list(self._items.get(batch_id, []))
            started = meta.get("started_at")
            end = meta.get("finished_at") or _now()
            elapsed = None
            if started:
                elapsed = round((datetime.fromisoformat(end) - datetime.fromisoformat(started)).total_seconds(), 1) \
                    if meta.get("finished_at") else round((datetime.now().astimezone() - datetime.fromisoformat(started)).total_seconds(), 1)
            return {"batch_id": batch_id, "status": meta["status"], "total": meta["total"], "done": len(items),
                    "current": self._current.get(batch_id), "items": items, "fields": list(meta["fields"]),
                    "created_at": meta["created_at"], "finished_at": meta.get("finished_at"), "elapsed_seconds": elapsed,
                    "options": meta["options"]}

    def list_recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            ids = sorted(self._meta, key=lambda b: self._meta[b]["created_at"], reverse=True)[:limit]
            return [{k: v for k, v in self.describe(b).items() if k not in ("items", "fields")} for b in ids]

    def cancel(self, batch_id: str) -> dict:
        """Stop BETWEEN fields: the field being worked on finishes, the rest are skipped."""
        with self._lock:
            self.describe(batch_id)                    # 404 if unknown
            meta = self._meta[batch_id]
            if meta["status"] == QUEUED:
                meta["status"], meta["finished_at"] = CANCELLED, _now()
                self._write_meta(batch_id)
            elif meta["status"] == RUNNING:
                self._cancel.add(batch_id)
            else:
                raise ApiError(409, f"This batch is already {meta['status']}.")
            return self.describe(batch_id)

    def delete(self, batch_id: str) -> dict:
        with self._lock:
            self.describe(batch_id)
            if self._meta[batch_id]["status"] in (QUEUED, RUNNING):
                raise ApiError(409, "Stop the batch first, then delete it.")
            shutil.rmtree(self._dir(batch_id), ignore_errors=True)
            for store in (self._meta, self._items, self._current):
                store.pop(batch_id, None)
            return {"deleted": batch_id}

    def results(self, batch_id: str) -> list[dict]:
        """Full answer payloads of the fields finished so far (read from disk)."""
        with self._lock:
            self.describe(batch_id)
            return [row["payload"] for row in self._read_results(batch_id)]

    def items_for_export(self, batch_id: str) -> list[dict]:
        payloads = self.results(batch_id)
        if not payloads:
            raise ApiError(404, "This batch has no finished fields yet, so there is nothing to export.")
        return payloads
