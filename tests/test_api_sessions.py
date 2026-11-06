"""Phase 8: server-side conversation history."""
from __future__ import annotations

import pytest

from lending_agent.api.errors import ApiError
from lending_agent.api.sessions import SessionStore


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TestSessions:
    def test_no_id_starts_a_new_session_with_a_fresh_id(self):
        store = SessionStore()
        a, b = store.get_or_create(None), store.get_or_create(None)
        assert a.session_id != b.session_id and len(store) == 2

    def test_the_same_id_returns_the_same_session(self):
        store = SessionStore()
        a = store.get_or_create(None)
        assert store.get_or_create(a.session_id) is a

    def test_an_unknown_but_valid_id_starts_an_empty_session_with_that_id(self):
        store = SessionStore()
        assert store.get_or_create("abcdef123456").turns == []

    @pytest.mark.parametrize("bad", ["short", "has spaces in it", "../../etc/passwd", "x" * 100, "semi;colon123"])
    def test_a_malformed_id_is_refused(self, bad):
        with pytest.raises(ApiError) as exc:
            SessionStore().get_or_create(bad)
        assert exc.value.status == 422

    def test_turns_are_numbered_and_kept_in_order(self):
        store = SessionStore()
        sid = store.get_or_create(None).session_id
        store.add_turn(sid, "q1", {"a": 1})
        store.add_turn(sid, "q2", {"a": 2})
        assert [t["turn"] for t in store.get(sid).turns] == [1, 2]
        assert store.get(sid).turns[1]["payload"] == {"a": 2}

    def test_only_the_most_recent_turns_are_kept_but_numbers_keep_counting(self):
        store = SessionStore(max_turns=3)
        sid = store.get_or_create(None).session_id
        for i in range(5):
            store.add_turn(sid, f"q{i}", {})
        assert [t["turn"] for t in store.get(sid).turns] == [3, 4, 5]

    def test_idle_sessions_expire(self):
        clock = Clock()
        store = SessionStore(ttl_seconds=60, clock=clock)
        sid = store.get_or_create(None).session_id
        clock.now = 61
        assert store.get(sid) is None

    def test_the_least_recently_used_session_is_dropped_when_there_are_too_many(self):
        store = SessionStore(max_sessions=2)
        first = store.get_or_create(None).session_id
        second = store.get_or_create(None).session_id
        store.get_or_create(first)                    # touch the first, so the second is now the oldest
        store.get_or_create(None)
        assert store.get(first) is not None and store.get(second) is None

    def test_delete(self):
        store = SessionStore()
        sid = store.get_or_create(None).session_id
        assert store.delete(sid) is True and store.delete(sid) is False
