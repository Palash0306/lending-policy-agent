"""
Phase 8 -- conversation history, kept on the server, one session per browser tab / API client.

In memory only: restarting the server forgets every conversation (uploaded documents and the
index are on disk and are NOT affected). Sessions expire after a quiet period and the oldest
are dropped when there are too many, so a long-running server cannot fill memory.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from lending_agent.api.errors import ApiError

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def new_session_id() -> str:
    return uuid.uuid4().hex


def check_session_id(session_id: str) -> str:
    if not _SESSION_ID_RE.match(session_id or ""):
        raise ApiError(422, "session_id must be 8-64 letters, digits, '-' or '_'.")
    return session_id


@dataclass
class Session:
    session_id: str
    turns: list[dict] = field(default_factory=list)
    last_used: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict:
        return {"session_id": self.session_id, "turns": list(self.turns)}


class SessionStore:
    def __init__(self, max_sessions: int = 200, ttl_seconds: float = 8 * 3600, max_turns: int = 30,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.max_sessions, self.ttl_seconds, self.max_turns, self._clock = max_sessions, ttl_seconds, max_turns, clock
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._lock = threading.Lock()

    def _expire(self) -> None:
        now = self._clock()
        for sid in [s for s, sess in self._sessions.items() if now - sess.last_used > self.ttl_seconds]:
            del self._sessions[sid]

    def get_or_create(self, session_id: str | None) -> Session:
        """An unknown id starts a new, empty session with that id (e.g. after a server restart)."""
        with self._lock:
            self._expire()
            sid = check_session_id(session_id) if session_id else new_session_id()
            session = self._sessions.get(sid)
            if session is None:
                session = self._sessions[sid] = Session(sid)
                while len(self._sessions) > self.max_sessions:
                    self._sessions.popitem(last=False)          # drop the least recently used
            session.last_used = self._clock()
            self._sessions.move_to_end(sid)
            return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            self._expire()
            return self._sessions.get(check_session_id(session_id))

    def add_turn(self, session_id: str, message: str, payload: dict) -> dict:
        """Record one finished question and answer. Keeps only the most recent `max_turns`."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:                                  # expired while the answer was being worked out
                session = self._sessions[session_id] = Session(session_id)
            number = (session.turns[-1]["turn"] + 1) if session.turns else 1
            turn = {"turn": number, "at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "message": message, "payload": payload}
            session.turns.append(turn)
            del session.turns[:-self.max_turns]
            session.last_used = self._clock()
            return turn

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(check_session_id(session_id), None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._sessions)
