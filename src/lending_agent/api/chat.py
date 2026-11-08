"""
Phase 8 -- the chat logic, with no web framework in it (so it is easy to test).

One call to `ChatService.chat()` is one question:
  1. find or start the session
  2. if the message depends on the earlier conversation ("what about for commercial loans?"),
     rewrite it into a stand-alone question first
  3. run the Phase 7 agent on it, and turn its answer into the chat JSON (summary.py)
  4. remember the question and answer in the session

FOLLOW-UP QUESTIONS. A message that names a known field (or a snake_case field) is already
stand-alone and is used as written. Otherwise, if the session has an earlier turn, the local model is asked to
rewrite it using the previous question. That rewrite is untrusted input: it must be one short
line, and it may not contain a number that is in neither the new message nor the earlier
conversation (an invented number must never reach the search). If the rewrite fails for any
reason, a plain-code fallback prepends the previous field's name. The question that was really
searched is always returned as `query_used`, so nothing is hidden.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Callable

from lending_agent.agent.field_expansion import SEED_SYNONYMS
from lending_agent.agent.llm_json import parse_llm_json
from lending_agent.api.errors import ApiError
from lending_agent.api.sessions import SessionStore
from lending_agent.api.summary import ROUTE_ERROR, build_payload

MAX_MESSAGE_CHARS = 500
KIND_CHOICES = ("both", "boundary", "date")
_FIELD_TOKEN_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_DIGITS_RE = re.compile(r"\d[\d,.]*")


@dataclass(frozen=True)
class ChatOptions:
    source_file: str | None = None      # only search this indexed file
    kinds: str = "both"                 # "both", "boundary" or "date"
    fast: bool = False                  # read fewer paragraphs: quicker, may miss a rule


def mentions_known_field(message: str) -> bool:
    """True if the message names a field by itself, so it needs nothing from earlier turns."""
    lowered = message.lower()
    if _FIELD_TOKEN_RE.search(lowered):
        return True
    for field, terms in SEED_SYNONYMS.items():
        for phrase in [field.replace("_", " "), *[t.lower() for t in terms]]:
            if re.search(rf"\b{re.escape(phrase)}\b", lowered):
                return True
    return False


def _numbers(text: str) -> set[str]:
    return {m.rstrip(".,").replace(",", "") for m in _DIGITS_RE.findall(text)}


def _rewrite_prompt(message: str, previous_message: str, previous_query: str, field: str) -> str:
    return (
        "You help a data assurance analyst ask questions about a bank's lending policy documents.\n"
        f'Earlier question: "{previous_query}" (field: {field})\n'
        f'New message: "{message}"\n\n'
        "If the new message only makes sense with the earlier question, rewrite it as ONE complete, "
        "stand-alone question about the same field. If it is already complete, return it unchanged. "
        "Do not add numbers or values.\n"
        'Reply with JSON only: {"question": "<the question>"}'
    )


def rewrite_follow_up(message: str, previous: dict, llm: Any) -> tuple[str, bool]:
    """Returns (question to search, whether it was rewritten). Never raises: falls back to plain code."""
    previous_payload = previous.get("payload", {})
    field = previous_payload.get("field") or ""
    previous_query = previous_payload.get("query_used") or previous.get("message", "")
    fallback = f"{field.replace('_', ' ')} {message}".strip() if field else message
    try:
        raw = llm.generate(_rewrite_prompt(message, previous.get("message", ""), previous_query, field),
                           format="json", temperature=0)
        data = parse_llm_json(raw) or {}
        question = re.sub(r"\s+", " ", str(data.get("question", ""))).strip()
    except Exception:
        return fallback, True
    allowed = _numbers(message) | _numbers(previous_query) | _numbers(previous.get("message", ""))
    if not (8 <= len(question) <= 300) or not _numbers(question) <= allowed:
        return fallback, True
    return question, question != message


def ollama_status(llm: Any) -> dict:
    """Is Ollama up and is the configured model installed? Never raises."""
    status = {"reachable": False, "model": getattr(getattr(llm, "settings", None), "model", None),
              "model_available": False, "error": None}
    if llm is None:
        status["error"] = "no model client configured"
        return status
    try:
        llm.list_models()
        status["reachable"] = True
        status["model_available"] = bool(llm.is_model_available())
    except Exception as exc:
        status["error"] = str(exc)
    return status


class ChatService:
    """`build` makes the compiled agent graph for a question; `run` asks it. Both are injectable for tests."""

    def __init__(self, searcher: Any, llm: Any, sessions: SessionStore,
                 build: Callable[[ChatOptions], Any] | None = None,
                 run: Callable[[Any, str], Any] | None = None) -> None:
        self.searcher, self.llm, self.sessions = searcher, llm, sessions
        self._build, self._run = build, run

    # -- the real agent, imported only when first needed (langgraph is heavy) --
    def _default_build(self, options: ChatOptions) -> Any:
        from lending_agent.agent.extraction import ALL_KINDS, KIND_DATE, KIND_NUMERIC
        from lending_agent.agent.graph import AgentConfig, build_graph

        kinds = {"both": ALL_KINDS, "boundary": (KIND_NUMERIC,), "date": (KIND_DATE,)}[options.kinds]
        config = AgentConfig(kinds=kinds, top_k=3 if options.fast else 6, source_file=options.source_file)
        return build_graph(self.searcher, llm=self.llm, config=config)

    def _default_run(self, graph: Any, query: str) -> Any:
        from lending_agent.agent.graph import ask

        return ask(graph, query)

    @staticmethod
    def prepare(message: str, options: ChatOptions | None = None) -> tuple[str, ChatOptions]:
        """Check a request up front (so a bad one is refused at once, not after waiting in the queue)."""
        message = (message or "").strip()
        if not message:
            raise ApiError(422, "message must not be empty.")
        if len(message) > MAX_MESSAGE_CHARS:
            raise ApiError(422, f"message is too long (max {MAX_MESSAGE_CHARS} characters).")
        options = options or ChatOptions()
        if options.kinds not in KIND_CHOICES:
            raise ApiError(422, f"kinds must be one of {', '.join(KIND_CHOICES)}.")
        return message, options

    def answer_one(self, query: str, options: ChatOptions) -> dict:
        """Run the agent on one question and return the answer payload. Never raises: a crash becomes an error answer.
        Used by /chat and by batches, so both go through exactly the same graph."""
        began = time.monotonic()
        try:
            graph = (self._build or self._default_build)(options)
            payload = build_payload((self._run or self._default_run)(graph, query))
        except Exception as exc:                  # embedding model missing, Chroma unreadable...: report it
            payload = {"field": None, "agent_route": ROUTE_ERROR, "route": ROUTE_ERROR, "grounded": False,
                       "needs_review": False, "review_reasons": [], "findings": [], "conflicts": [], "rejected": [],
                       "error": f"{type(exc).__name__}: {exc}", "attempts": 0, "widened_search": False, "trace": [],
                       "search": {}, "answer": f"I could not answer this: {type(exc).__name__}: {exc}"}
        payload.update(query_used=query, rewritten=False,
                       options={"source_file": options.source_file, "kinds": options.kinds, "fast": options.fast},
                       elapsed_seconds=round(time.monotonic() - began, 1))
        return payload

    def chat(self, message: str, session_id: str | None = None, options: ChatOptions | None = None) -> dict:
        message, options = self.prepare(message, options)
        session = self.sessions.get_or_create(session_id)
        began = time.monotonic()
        query, rewritten = message, False
        if session.turns and not mentions_known_field(message):
            query, rewritten = rewrite_follow_up(message, session.turns[-1], self.llm)

        payload = self.answer_one(query, options)
        payload.update(session_id=session.session_id, message=message, query_used=query, rewritten=rewritten,
                       elapsed_seconds=round(time.monotonic() - began, 1))
        turn = self.sessions.add_turn(session.session_id, message, payload)
        payload["turn"] = turn["turn"]
        return payload
