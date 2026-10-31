"""
A small run timer: when a run started, how long each step took, when it ended.

Why it exists: extraction can take minutes on a laptop, and a silent terminal
looks exactly like a frozen one. While a step is running, a background
"heartbeat" prints a short still-working line every few seconds, so you can tell
a slow run from a dead one. The same start/end times are returned by to_dict(),
ready to go into the run report planned for later.

All messages go to stderr, so `--json` output on stdout stays clean.
"""
from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Iterator


def format_duration(seconds: float) -> str:
    """0.4 -> '0.4s', 72 -> '1m 12s', 3725 -> '1h 02m 05s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    whole = int(round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _stamp(moment: datetime) -> str:
    return moment.strftime("%H:%M:%S")


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


class RunTimer:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now().astimezone(),
        write: Callable[[str], None] = _stderr,
    ) -> None:
        self._clock = clock
        self._now = now
        self._write = write
        self._t0: float | None = None
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self.total_seconds: float | None = None
        self.steps: list[tuple[str, float]] = []

    def start(self) -> None:
        self._t0 = self._clock()
        self.started_at = self._now()
        self._write(f"⏱  Started at {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}")

    def finish(self) -> None:
        if self.ended_at is not None:
            return
        self.total_seconds = self._clock() - (self._t0 or 0.0)
        self.ended_at = self._now()

    @contextmanager
    def step(self, name: str, heartbeat_every: float | None = 15.0, note: str = "") -> Iterator[None]:
        """Time one step. Prints when it starts and ends, and (if heartbeat_every is
        set) a still-working line every that many seconds while it runs. `note` is
        added to each still-working line, to say what the step is actually doing."""
        began = self._clock()
        self._write(f"[{_stamp(self._now())}] → {name}")
        stop = threading.Event()
        thread = None
        if heartbeat_every:
            def beat() -> None:
                while not stop.wait(heartbeat_every):
                    elapsed = self._clock() - began
                    self._write(
                        f"[{_stamp(self._now())}]   …still working ({format_duration(elapsed)} so far)."
                        + (f" {note}" if note else "")
                    )

            thread = threading.Thread(target=beat, daemon=True)
            thread.start()
        try:
            yield
        finally:
            stop.set()
            if thread:
                thread.join(timeout=1)
            elapsed = self._clock() - began
            self.steps.append((name, elapsed))
            self._write(f"[{_stamp(self._now())}]   done in {format_duration(elapsed)}")

    def summary_lines(self) -> list[str]:
        lines = []
        if self.started_at and self.ended_at:
            lines.append(f"Started: {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Ended:   {self.ended_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.total_seconds is not None:
            lines.append(f"Total:   {format_duration(self.total_seconds)}")
        for name, seconds in self.steps:
            lines.append(f"  {format_duration(seconds):>9}  {name}")
        return lines

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "total_seconds": round(self.total_seconds, 2) if self.total_seconds is not None else None,
            "steps": [{"name": n, "seconds": round(s, 2)} for n, s in self.steps],
        }
