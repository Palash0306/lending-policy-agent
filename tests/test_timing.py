"""Tests for RunTimer. A fake clock keeps the numbers exact; the heartbeat test uses real (short) sleeps."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from lending_agent.timing import RunTimer, format_duration

T0 = datetime(2026, 10, 17, 10, 42, 0, tzinfo=timezone.utc)


class FakeTime:
    """Hands out clock readings from a list, and a wall clock that moves with them."""

    def __init__(self, readings):
        self.readings = list(readings)
        self.last = 0.0

    def clock(self):
        self.last = self.readings.pop(0)
        return self.last

    def now(self):
        return T0 + timedelta(seconds=self.last)


@pytest.mark.parametrize("seconds, expected", [
    (0.4, "0.4s"), (12.34, "12.3s"), (59.9, "59.9s"), (60, "1m 00s"), (72, "1m 12s"),
    (252, "4m 12s"), (3725, "1h 02m 05s"),
])
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


class TestRunTimer:
    def _timer(self, readings):
        fake = FakeTime(readings)
        lines = []
        return RunTimer(clock=fake.clock, now=fake.now, write=lines.append), lines

    def test_records_start_end_total_and_step_durations(self):
        timer, _ = self._timer([0, 1, 6, 10])           # start, step begins, step ends, finish
        timer.start()
        with timer.step("Step A", heartbeat_every=None):
            pass
        timer.finish()
        assert timer.total_seconds == 10
        assert timer.steps == [("Step A", 5)]
        assert timer.started_at == T0
        assert timer.ended_at == T0 + timedelta(seconds=10)

    def test_prints_start_and_done_lines_with_clock_times(self):
        timer, lines = self._timer([0, 0, 83, 83])
        timer.start()
        with timer.step("Step A", heartbeat_every=None):
            pass
        assert "Started at 2026-10-17 10:42:00" in lines[0]
        assert "[10:42:00] → Step A" in lines[1]
        assert "done in 1m 23s" in lines[2]

    def test_a_step_that_raises_is_still_recorded(self):
        timer, _ = self._timer([0, 1, 4])
        timer.start()
        with pytest.raises(RuntimeError):
            with timer.step("Step A", heartbeat_every=None):
                raise RuntimeError("boom")
        assert timer.steps == [("Step A", 3)]

    def test_to_dict_is_json_friendly(self):
        timer, _ = self._timer([0, 1, 6, 10])
        timer.start()
        with timer.step("Step A", heartbeat_every=None):
            pass
        timer.finish()
        data = timer.to_dict()
        assert data["total_seconds"] == 10
        assert data["steps"] == [{"name": "Step A", "seconds": 5}]
        assert data["started_at"].startswith("2026-10-17T10:42:00")

    def test_summary_lists_start_end_total_and_each_step(self):
        timer, _ = self._timer([0, 1, 6, 10])
        timer.start()
        with timer.step("Step A", heartbeat_every=None):
            pass
        timer.finish()
        text = "\n".join(timer.summary_lines())
        assert "Started:" in text and "Ended:" in text and "Total:   10.0s" in text and "Step A" in text

    def test_finish_twice_keeps_the_first_end_time(self):
        timer, _ = self._timer([0, 10, 99])
        timer.start()
        timer.finish()
        timer.finish()
        assert timer.total_seconds == 10


class TestHeartbeat:
    def test_prints_still_working_lines_while_a_step_runs_and_stops_after(self):
        lines = []
        timer = RunTimer(write=lines.append)
        timer.start()
        with timer.step("Slow step", heartbeat_every=0.05):
            time.sleep(0.3)
        beats = [line for line in lines if "still working" in line]
        assert len(beats) >= 2
        count_after = len(lines)
        time.sleep(0.2)
        assert len(lines) == count_after            # the heartbeat thread has really stopped

    def test_a_fast_step_prints_no_heartbeat(self):
        lines = []
        timer = RunTimer(write=lines.append)
        with timer.step("Quick step", heartbeat_every=5):
            pass
        assert not any("still working" in line for line in lines)

    def test_the_note_is_added_to_each_heartbeat_line(self):
        lines = []
        timer = RunTimer(write=lines.append)
        with timer.step("Slow step", heartbeat_every=0.05, note="Loading the model."):
            time.sleep(0.2)
        beats = [line for line in lines if "still working" in line]
        assert beats and all(line.endswith("Loading the model.") for line in beats)
