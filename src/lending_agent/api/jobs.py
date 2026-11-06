"""
Phase 8 -- a one-at-a-time job queue.

Why one at a time: the 7B model and the embedding model together are heavy, and two questions
running at once on a laptop would fight over memory (and freeze it). So there is exactly ONE
worker thread; every question or upload waits its turn. It also means a session's questions
are answered in the order they were asked, so a follow-up always sees the answer before it.

Why jobs at all: one question can take minutes. A web page that just waits on one long
request can time out, so the page starts a job, then asks "done yet?" every couple of seconds.
API clients that prefer a single call can still wait (see `wait` in app.py).
"""
from __future__ import annotations

import itertools
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

STATUS_QUEUED, STATUS_RUNNING, STATUS_DONE, STATUS_FAILED = "queued", "running", "done", "failed"


@dataclass
class Job:
    job_id: str
    kind: str
    seq: int
    status: str = STATUS_QUEUED
    result: Any = None
    error: str | None = None
    exception: Exception | None = field(default=None, repr=False)
    created: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None
    _done: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def finished_ok(self) -> bool:
        return self.status == STATUS_DONE

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)


class JobRunner:
    def __init__(self, max_jobs: int = 200) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-worker")
        self._jobs: "OrderedDict[str, Job]" = OrderedDict()
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self.max_jobs = max_jobs

    def submit(self, kind: str, fn: Callable[[], Any]) -> Job:
        with self._lock:
            job = Job(job_id=uuid.uuid4().hex, kind=kind, seq=next(self._counter))
            self._jobs[job.job_id] = job
            self._evict()
        self._executor.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn: Callable[[], Any]) -> None:
        job.started, job.status = time.monotonic(), STATUS_RUNNING
        try:
            job.result = fn()
            job.status = STATUS_DONE
        except Exception as exc:                      # not BaseException: Ctrl+C / shutdown must still work
            job.error, job.exception, job.status = f"{type(exc).__name__}: {exc}", exc, STATUS_FAILED
        finally:
            job.finished = time.monotonic()
            job._done.set()

    def _evict(self) -> None:
        """Forget the oldest finished jobs once there are too many. Unfinished jobs are never dropped."""
        while len(self._jobs) > self.max_jobs:
            victim = next((jid for jid, j in self._jobs.items() if j.status in (STATUS_DONE, STATUS_FAILED)), None)
            if victim is None:
                break
            del self._jobs[victim]

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def pending(self) -> int:
        """Jobs waiting or running right now."""
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status in (STATUS_QUEUED, STATUS_RUNNING))

    def describe(self, job: Job) -> dict:
        """What the web page / API client sees while polling."""
        with self._lock:
            ahead = sum(1 for j in self._jobs.values()
                        if j.seq < job.seq and j.status in (STATUS_QUEUED, STATUS_RUNNING))
        end = job.finished or time.monotonic()
        started = job.started or end
        return {
            "job_id": job.job_id, "kind": job.kind, "status": job.status,
            "jobs_ahead": ahead if job.status == STATUS_QUEUED else 0,
            "waited_seconds": round((job.started or end) - job.created, 1),
            "running_seconds": round(end - started, 1) if job.started else 0.0,
            "result": job.result if job.status == STATUS_DONE else None,
            "error": job.error,
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
