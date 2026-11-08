"""
Phase 8 -- the FastAPI app. This file only connects URLs to the framework-free services
(chat.py, documents.py, jobs.py, sessions.py), so there is little logic here to get wrong.

    GET    /                    the web page
    GET    /health              is the server up, is Ollama up, what is indexed
    POST   /chat                ask a question (see "wait" below)
    GET    /jobs/{job_id}       progress / result of a queued question or upload
    GET    /sessions/{id}       the conversation so far
    DELETE /sessions/{id}       forget a conversation
    GET    /documents           what is indexed
    POST   /documents           upload one or more .pdf/.docx files and index them
    DELETE /documents/{name}    remove a document from the index

"wait": by default POST /chat and POST /documents wait until the work is finished and return the
result (one simple call, ideal for scripts and curl). Pass ?wait=false to get a job_id back at
once and poll GET /jobs/{job_id}: the web page does this, because one question can take minutes
and browsers give up on a request that is silent for too long.

Start it with:  python scripts/serve.py
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from lending_agent.api.batches import BatchManager
from lending_agent.api.chat import MAX_MESSAGE_CHARS, ChatOptions, ChatService, ollama_status
from lending_agent.api.documents import DocumentService
from lending_agent.api.errors import ApiError
from lending_agent.api.jobs import STATUS_FAILED, JobRunner
from lending_agent.api.sessions import SessionStore

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS,
                         description="A field name (loan_tenure) or a question, or a follow-up to the last answer")
    session_id: Optional[str] = Field(None, description="Leave empty to start a conversation; reuse the returned id to continue it")
    source_file: Optional[str] = Field(None, description="Only search this indexed file")
    kinds: Literal["both", "boundary", "date"] = "both"
    fast: bool = Field(False, description="Read fewer paragraphs: quicker, but may miss a rule")


class BatchRequest(BaseModel):
    fields: List[str] = Field(..., description="Field names or questions. Each entry may itself hold several, separated by new lines, commas or semicolons")
    source_file: Optional[str] = None
    kinds: Literal["both", "boundary", "date"] = "both"
    fast: bool = False


XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def create_app(chat: ChatService, documents: DocumentService, runner: JobRunner, sessions: SessionStore,
               batches: Optional[BatchManager] = None, export_meta: Optional[Callable[[], dict]] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            documents.refresh()
        except Exception as exc:                       # an unreadable index must not stop the server starting
            documents.last_error = str(exc)
        yield
        runner.shutdown()

    app = FastAPI(title="Lending Policy Data Assurance Agent", version="0.8.0", lifespan=lifespan)

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = ("default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        return response

    @app.exception_handler(ApiError)
    async def api_error(_, exc: ApiError):
        return JSONResponse(status_code=exc.status, content={"detail": exc.message})

    def dispatch(kind: str, fn: Any, wait: bool, extra: dict | None = None):
        """Run `fn` on the single worker. Wait for it, or hand back a job to poll."""
        job = runner.submit(kind, fn)
        if not wait:
            return JSONResponse(status_code=202, content={**runner.describe(job), **(extra or {})})
        job.wait()
        if job.status == STATUS_FAILED:
            if isinstance(job.exception, ApiError):          # e.g. "no such document": a 404, not a server fault
                raise job.exception
            raise HTTPException(status_code=500, detail=job.error)
        return job.result

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home():
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/health")
    def health():
        listing = documents.listing()
        return {"status": "ok", "ollama": ollama_status(chat.llm), "documents": len(listing["documents"]),
                "chunks": listing["total_chunks"], "jobs_pending": runner.pending(), "index_error": documents.last_error,
                "max_batch_fields": batches.max_fields if batches else None}

    @app.post("/chat")
    def post_chat(request: ChatRequest, wait: bool = True):
        message, options = chat.prepare(request.message, ChatOptions(
            source_file=request.source_file, kinds=request.kinds, fast=request.fast))
        session = sessions.get_or_create(request.session_id)    # fixes the id now, so a job-based client can reuse it
        return dispatch("chat", lambda: chat.chat(message, session.session_id, options), wait,
                        extra={"session_id": session.session_id})

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        job = runner.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown or expired job.")
        return runner.describe(job)

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Unknown or expired session.")
        return session.to_dict()

    @app.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        return {"deleted": sessions.delete(session_id)}

    @app.get("/documents")
    def list_documents():
        return documents.listing()

    @app.post("/documents")
    def upload_documents(files: List[UploadFile] = File(...), wait: bool = True):
        saved, rejected = [], []
        for upload in files:
            try:
                saved.append(documents.save_upload(upload.filename or "", upload.file))
            except ApiError as exc:
                rejected.append({"file": upload.filename, "error": exc.message})
        if not saved:
            raise HTTPException(status_code=400, detail={"message": "No file was accepted.", "rejected": rejected})
        return dispatch("index", lambda: {**documents.index_files(saved), "rejected": rejected}, wait,
                        extra={"rejected": rejected})

    # ---- Phase 9: batches and Excel / CSV export ------------------------------------------------------
    def need_batches() -> BatchManager:
        if batches is None:
            raise ApiError(404, "Batches are not enabled on this server.")
        return batches

    def export_response(items: list, fmt: str, include_examples: bool, step: Optional[str], what: str):
        from datetime import datetime

        from lending_agent.export_xlsx import build_csv, build_workbook, export_filename
        from lending_agent.testcases import to_decimal

        if fmt not in ("xlsx", "csv"):
            raise ApiError(422, "format must be xlsx or csv.")
        if step not in (None, ""):
            try:
                if to_decimal(step) <= 0:
                    raise ValueError
            except Exception:
                raise ApiError(422, "step must be a number greater than zero, such as 1 or 0.1.")
        else:
            step = None
        if not items:
            raise ApiError(404, f"Nothing to export: {what} has no answers yet.")
        meta = {**(export_meta() if export_meta else {}), "include_examples": include_examples, "step": step,
                "generated_at": datetime.now()}
        body = build_workbook(items, meta) if fmt == "xlsx" else build_csv(items, meta)
        name = export_filename(fmt, meta["generated_at"])
        return Response(content=body, media_type=XLSX_TYPE if fmt == "xlsx" else "text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.post("/batches")
    def post_batch(request: BatchRequest):
        manager = need_batches()
        batch = manager.create(request.fields, ChatOptions(source_file=request.source_file, kinds=request.kinds, fast=request.fast))
        job = runner.submit("batch", lambda: manager.run(batch["batch_id"], chat))
        return JSONResponse(status_code=202, content={"batch_id": batch["batch_id"], "job_id": job.job_id, "count": batch["total"]})

    @app.get("/batches")
    def list_batches():
        return {"batches": need_batches().list_recent()}

    @app.get("/batches/{batch_id}")
    def get_batch(batch_id: str):
        return need_batches().describe(batch_id)

    @app.get("/batches/{batch_id}/results")
    def get_batch_results(batch_id: str):
        return {"results": need_batches().results(batch_id)}

    @app.post("/batches/{batch_id}/cancel")
    def cancel_batch(batch_id: str):
        return need_batches().cancel(batch_id)

    @app.delete("/batches/{batch_id}")
    def delete_batch(batch_id: str):
        return need_batches().delete(batch_id)

    @app.get("/batches/{batch_id}/export")
    def export_batch(batch_id: str, fmt: str = Query("xlsx", alias="format"), include_examples: bool = False, step: Optional[str] = None):
        return export_response(need_batches().items_for_export(batch_id), fmt, include_examples, step, "this batch")

    @app.get("/sessions/{session_id}/export")
    def export_session(session_id: str, fmt: str = Query("xlsx", alias="format"), include_examples: bool = False, step: Optional[str] = None):
        session = sessions.get(session_id)
        if session is None:
            raise ApiError(404, "Unknown or expired session.")
        return export_response([t["payload"] for t in session.turns], fmt, include_examples, step, "this conversation")

    @app.get("/sessions/{session_id}/turns/{turn}/export")
    def export_turn(session_id: str, turn: int, fmt: str = Query("xlsx", alias="format"), include_examples: bool = False, step: Optional[str] = None):
        session = sessions.get(session_id)
        found = [t["payload"] for t in session.turns if t["turn"] == turn] if session else []
        if not found:
            raise ApiError(404, "Unknown or expired answer.")
        return export_response(found, fmt, include_examples, step, "this answer")

    @app.delete("/documents/{name}")
    def delete_document(name: str, wait: bool = True):
        return dispatch("delete", lambda: documents.delete(name), wait)

    return app


def create_default_app() -> FastAPI:
    """The real thing: real index, real Ollama. Used by scripts/serve.py (uvicorn --factory)."""
    from lending_agent.config import get_settings
    from lending_agent.indexing.vector_store import VectorStore
    from lending_agent.llm.ollama_client import OllamaClient

    settings = get_settings()
    store, llm = VectorStore(), OllamaClient()
    sessions = SessionStore(max_sessions=settings.api.max_sessions, ttl_seconds=settings.api.session_ttl_minutes * 60,
                            max_turns=settings.api.max_turns_per_session)
    documents = DocumentService(store, settings.api.upload_dir, settings.api.max_upload_mb * 1024 * 1024)
    batches = BatchManager(settings.api.batch_dir, settings.api.max_batch_fields)

    def export_meta() -> dict:
        return {"llm_model": settings.ollama.model, "embedding_model": settings.embedding.model_name,
                "documents": [d["name"] for d in documents.listing()["documents"]]}

    return create_app(ChatService(store, llm, sessions), documents, JobRunner(), sessions, batches, export_meta)
