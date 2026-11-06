"""
Phase 8 -- uploading, indexing, listing and removing policy documents. No web framework in here.

Safety rules for an uploaded file:
  * only the file NAME is used (never a path), and unusual characters are replaced, so a name like
    "../../etc/passwd.pdf" cannot write outside the upload folder;
  * only .pdf and .docx are accepted, and the first bytes must really look like that type;
  * a size limit is enforced while reading, so a huge file cannot fill the disk;
  * a file that fails to index is deleted again, and one bad file does not stop the others.

Uploading a file with the same name as an indexed one REPLACES it: the old chunks are removed first,
otherwise leftover chunks from the old version would still be searched.
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, BinaryIO, Callable

from lending_agent.api.errors import ApiError
from lending_agent.ingestion.loaders import SUPPORTED_EXTENSIONS

_BAD_CHARS_RE = re.compile(r"[^A-Za-z0-9._ ()\-]+")
_READ_BLOCK = 1024 * 1024


def safe_filename(raw: str) -> str:
    """Keep only the file name, and only harmless characters. Raises ApiError if nothing usable is left."""
    name = (raw or "").replace("\\", "/").split("/")[-1].strip()
    name = _BAD_CHARS_RE.sub("_", name).strip(" .")
    stem, ext = os.path.splitext(name)
    if not stem or ext.lower() not in SUPPORTED_EXTENSIONS:
        raise ApiError(400, f"'{raw}': only {', '.join(sorted(SUPPORTED_EXTENSIONS))} files are accepted.")
    return stem[:150] + ext.lower()


def looks_like(name: str, head: bytes) -> bool:
    """A .pdf must start with %PDF; a .docx is a zip file, so it must start with PK."""
    return head.startswith(b"%PDF") if name.endswith(".pdf") else head.startswith(b"PK")


class DocumentService:
    def __init__(self, store: Any, upload_dir: str | Path, max_bytes: int,
                 index_fn: Callable[..., dict] | None = None) -> None:
        self.store, self.upload_dir, self.max_bytes = store, Path(upload_dir), max_bytes
        self._index_fn = index_fn
        self._cache: dict[str, int] | None = None
        self._lock = threading.Lock()
        self.last_error: str | None = None

    # ---- saving an upload (runs in the web request, quick) -------------------------------
    def save_upload(self, raw_name: str, fileobj: BinaryIO) -> Path:
        name = safe_filename(raw_name)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        target, partial = self.upload_dir / name, self.upload_dir / f".{name}.part"
        size, first = 0, b""
        try:
            with partial.open("wb") as out:
                while True:
                    block = fileobj.read(_READ_BLOCK)
                    if not block:
                        break
                    if not first:
                        first = block[:8]
                    size += len(block)
                    if size > self.max_bytes:
                        raise ApiError(413, f"'{name}' is larger than the {self.max_bytes // (1024 * 1024)} MB limit.")
                    out.write(block)
            if size == 0:
                raise ApiError(400, f"'{name}' is empty.")
            if not looks_like(name, first):
                raise ApiError(400, f"'{name}' does not look like a real {Path(name).suffix} file.")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        return target

    # ---- indexing (runs in the worker, slow) ---------------------------------------------
    def _index_one(self, path: Path) -> int:
        """Index one file, REPLACING any earlier version. Read and chunk first; remove old chunks only once that worked."""
        if self._index_fn is not None:                 # tests inject their own indexer: keep that path as is
            self.store.delete_source(path.name)
            return self._index_fn([str(path)], store=self.store)[str(path)]
        from lending_agent.ingestion.pipeline import ingest_document

        chunks = ingest_document(str(path))            # raises on an unreadable file, before anything is touched
        if chunks:
            self.store.delete_source(path.name)
            self.store.add_chunks(chunks)
        return len(chunks)

    def index_files(self, paths: list[Path]) -> dict:
        indexed, failed = [], []
        for path in paths:
            try:
                added = self._index_one(path)
                if added == 0:
                    raise ValueError("no readable text was found (a scanned, image-only PDF needs OCR, "
                                     "which this tool does not do)")
                indexed.append({"file": path.name, "chunks": added})
            except Exception as exc:
                failed.append({"file": path.name, "error": str(exc)})
                path.unlink(missing_ok=True)
        self.refresh()
        return {"indexed": indexed, "failed": failed, **self.listing()}

    # ---- the list shown on the web page ---------------------------------------------------
    def refresh(self) -> None:
        """Re-read the list of indexed files from the store. Call only from the worker or at startup."""
        counts = self.store.sources()
        with self._lock:
            self._cache, self.last_error = counts, None

    def listing(self) -> dict:
        with self._lock:
            counts = dict(self._cache or {})
        return {"documents": [{"name": n, "chunks": c} for n, c in counts.items()],
                "total_chunks": sum(counts.values())}

    def delete(self, raw_name: str) -> dict:
        """Remove a file's chunks and its saved copy. Raises ApiError 404 if it was neither indexed nor saved."""
        name = Path(raw_name.replace("\\", "/")).name
        if name in ("", ".", ".."):
            raise ApiError(400, "That is not a valid document name.")
        saved = self.upload_dir / name
        removed = self.store.delete_source(name)
        had_file = saved.exists()
        saved.unlink(missing_ok=True)
        if removed == 0 and not had_file:
            raise ApiError(404, f"'{name}' is not an indexed document.")
        self.refresh()
        return {"deleted": name, "chunks_removed": removed, **self.listing()}
