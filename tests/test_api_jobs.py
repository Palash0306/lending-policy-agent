"""Phase 8: the one-at-a-time job queue."""
from __future__ import annotations

import threading
import time

from lending_agent.api.jobs import STATUS_DONE, STATUS_FAILED, STATUS_QUEUED, STATUS_RUNNING, JobRunner


class TestJobRunner:
    def test_a_job_runs_and_keeps_its_result(self):
        runner = JobRunner()
        job = runner.submit("chat", lambda: {"answer": 42})
        assert job.wait(5) and job.status == STATUS_DONE and job.result == {"answer": 42}
        assert runner.describe(job)["result"] == {"answer": 42}
        runner.shutdown()

    def test_an_exception_becomes_a_failed_job_and_the_worker_carries_on(self):
        runner = JobRunner()
        bad = runner.submit("chat", lambda: 1 / 0)
        good = runner.submit("chat", lambda: "fine")
        assert bad.wait(5) and bad.status == STATUS_FAILED and "ZeroDivisionError" in bad.error
        assert good.wait(5) and good.result == "fine"
        assert runner.describe(bad)["result"] is None
        runner.shutdown()

    def test_jobs_run_strictly_one_at_a_time_and_in_order(self):
        runner, order, active, peak, lock = JobRunner(), [], [0], [0], threading.Lock()

        def work(n):
            def run():
                with lock:
                    active[0] += 1
                    peak[0] = max(peak[0], active[0])
                time.sleep(0.02)
                order.append(n)
                with lock:
                    active[0] -= 1
            return run

        jobs = [runner.submit("chat", work(n)) for n in range(5)]
        assert all(j.wait(5) for j in jobs)
        assert order == [0, 1, 2, 3, 4] and peak[0] == 1
        runner.shutdown()

    def test_a_waiting_job_reports_how_many_are_ahead_of_it(self):
        runner, gate = JobRunner(), threading.Event()
        first = runner.submit("chat", lambda: gate.wait(5))
        second = runner.submit("chat", lambda: "x")
        third = runner.submit("chat", lambda: "y")
        deadline = time.monotonic() + 2
        while first.status != STATUS_RUNNING and time.monotonic() < deadline:
            time.sleep(0.005)
        assert runner.describe(first)["status"] == STATUS_RUNNING and runner.describe(first)["jobs_ahead"] == 0
        assert runner.describe(second)["status"] == STATUS_QUEUED and runner.describe(second)["jobs_ahead"] == 1
        assert runner.describe(third)["jobs_ahead"] == 2 and runner.pending() == 3
        gate.set()
        assert third.wait(5) and runner.pending() == 0
        runner.shutdown()

    def test_finished_jobs_are_forgotten_oldest_first_but_unfinished_ones_never(self):
        runner = JobRunner(max_jobs=3)
        done = [runner.submit("chat", lambda: 1) for _ in range(3)]
        assert all(j.wait(5) for j in done)
        gate = threading.Event()
        pending = runner.submit("chat", lambda: gate.wait(5))
        runner.submit("chat", lambda: 2)
        assert runner.get(done[0].job_id) is None and runner.get(pending.job_id) is not None
        gate.set()
        runner.shutdown()

    def test_an_unknown_job_id_is_none(self):
        assert JobRunner().get("nope") is None
