"""Tests for spec 40 D2/D11 (ADR 0021): the dispatcher's own dispatch telemetry --
`_derive_timing` (the pure per-job/per-run metric derivation) and `_aml_submit_and_wait`
writing `<run_root>/_timing.json`.

No test requires Azure: the AML client is a hand-rolled fake (mirrors
`test_scale_runner.py::_FakeMLClient`), and `_import_aml_command` is substituted.
"""

from __future__ import annotations

import json
import types

import pandas as pd
import pytest

from rise_eo.storage import fs
from rise_eo.workflows import runners


def _ts(seconds: float, base: str = "2024-01-01T00:00:00Z") -> str:
    return (pd.Timestamp(base) + pd.Timedelta(seconds=seconds)).isoformat()


# --- _seconds_between: the basic subtraction + missing-value contract --------------

def test_seconds_between_subtracts_and_handles_missing():
    assert runners._seconds_between(_ts(0), _ts(9)) == pytest.approx(9.0)
    assert runners._seconds_between(None, _ts(9)) is None
    assert runners._seconds_between(_ts(0), None) is None


def test_seconds_between_never_floors_a_negative_result():
    """spec 40 D11: clock skew can make a job look like it started before it was
    submitted -- that must be reported as negative, the signal the skew bound was
    exceeded, never floored at 0."""
    assert runners._seconds_between(_ts(10), _ts(2)) == pytest.approx(-8.0)


# --- _derive_timing: per-job metrics + the additive invariant ----------------------

def _two_job_stamps():
    """A hand-worked scenario (see spec 40 D11): 2 jobs, one poll tick, chosen so every
    additive-split leg is non-zero and the whole thing telescopes to t_end - t_start."""
    submitted_at = {0: _ts(1), 1: _ts(2)}
    returned_at = {0: _ts(30), 1: _ts(30)}
    reports = {
        0: {"process_start_at": _ts(10), "work_start_at": _ts(11), "work_end_at": _ts(20),
            "ended_at": _ts(21), "seconds": 9},
        1: {"process_start_at": _ts(15), "work_start_at": _ts(16), "work_end_at": _ts(26),
            "ended_at": _ts(27), "seconds": 10},
    }
    return submitted_at, returned_at, reports


def test_derive_timing_per_job_metrics():
    submitted_at, returned_at, reports = _two_job_stamps()
    timing = runners._derive_timing(
        run_id="r1", t_start=_ts(0), t_last_submit=_ts(2), t_end=_ts(35),
        submitted_at=submitted_at, returned_at=returned_at, reports=reports,
        poll_interval_seconds=30,
    )
    j0, j1 = timing["jobs"][0], timing["jobs"][1]
    assert j0["job_admission_seconds"] == pytest.approx(9.0)   # 10 - 1
    assert j1["job_admission_seconds"] == pytest.approx(13.0)  # 15 - 2
    assert j0["import_seconds"] == pytest.approx(1.0)          # 11 - 10
    assert j1["import_seconds"] == pytest.approx(1.0)          # 16 - 15
    # dispatch_overhead = (returned_at - submitted_at) - work_seconds
    assert j0["dispatch_overhead_seconds"] == pytest.approx((30 - 1) - 9)
    assert j1["dispatch_overhead_seconds"] == pytest.approx((30 - 2) - 10)


def test_derive_timing_additive_invariant_sums_to_the_run_wall():
    """The property the 2026-07-28 forensics depended on (spec 40 D11, run-book 41):
    driver_prep + first_admission + execution_window + teardown_detect + post_collect
    == t_end - t_start, within tolerance."""
    submitted_at, returned_at, reports = _two_job_stamps()
    t_start, t_end = _ts(0), _ts(35)
    timing = runners._derive_timing(
        run_id="r1", t_start=t_start, t_last_submit=_ts(2), t_end=t_end,
        submitted_at=submitted_at, returned_at=returned_at, reports=reports,
        poll_interval_seconds=30,
    )
    wall = timing["wall"]
    total = sum(wall[k] for k in (
        "driver_prep_seconds", "first_admission_seconds", "execution_window_seconds",
        "teardown_detect_seconds", "post_collect_seconds",
    ))
    expected = runners._seconds_between(t_start, t_end)
    assert total == pytest.approx(expected, abs=1e-6)
    # and each individual leg is the hand-worked value
    assert wall["driver_prep_seconds"] == pytest.approx(2.0)
    assert wall["first_admission_seconds"] == pytest.approx(8.0)   # min(10,15) - 2
    assert wall["execution_window_seconds"] == pytest.approx(17.0)  # max(21,27) - min(10,15)
    assert wall["teardown_detect_seconds"] == pytest.approx(3.0)   # 30 - max(21,27)
    assert wall["post_collect_seconds"] == pytest.approx(5.0)      # 35 - 30


def test_derive_timing_one_unit_degenerate_case():
    submitted_at = {0: _ts(0)}
    returned_at = {0: _ts(10)}
    reports = {0: {"process_start_at": _ts(3), "work_start_at": _ts(4), "work_end_at": _ts(8),
                   "ended_at": _ts(9), "seconds": 4}}
    timing = runners._derive_timing(
        run_id="solo", t_start=_ts(0), t_last_submit=_ts(0), t_end=_ts(12),
        submitted_at=submitted_at, returned_at=returned_at, reports=reports,
        poll_interval_seconds=5,
    )
    assert timing["jobs"][0]["job_admission_seconds"] == pytest.approx(3.0)
    wall = timing["wall"]
    total = sum(wall[k] for k in (
        "driver_prep_seconds", "first_admission_seconds", "execution_window_seconds",
        "teardown_detect_seconds", "post_collect_seconds",
    ))
    assert total == pytest.approx(12.0)


def test_derive_timing_skew_makes_admission_negative_and_not_floored():
    """spec 40 D11: ~8s of measured laptop-vs-Azure skew would be a third of a warm
    admission -- a job whose node clock reads *behind* the driver's must show a
    negative job_admission_seconds, not 0."""
    submitted_at = {0: _ts(20)}    # driver submits at t=20
    returned_at = {0: _ts(40)}
    reports = {0: {"process_start_at": _ts(15),  # node's (skewed) clock reads t=15
                   "work_start_at": _ts(16), "work_end_at": _ts(30), "ended_at": _ts(31),
                   "seconds": 14}}
    timing = runners._derive_timing(
        run_id="skew", t_start=_ts(0), t_last_submit=_ts(20), t_end=_ts(45),
        submitted_at=submitted_at, returned_at=returned_at, reports=reports,
        poll_interval_seconds=10,
    )
    assert timing["jobs"][0]["job_admission_seconds"] == pytest.approx(-5.0)


def test_derive_timing_failed_unit_with_no_status_file_has_null_job_metrics():
    """A job that crashed before writing `_status/<k>.json` (spec 40 D3/D15) still
    gets a `submitted_at`/`returned_at` pair, but its in-job stamps -- and everything
    derived from them -- are null rather than raising."""
    submitted_at = {0: _ts(0), 1: _ts(1)}
    returned_at = {0: _ts(20), 1: _ts(20)}
    reports = {
        0: {"process_start_at": _ts(5), "work_start_at": _ts(6), "work_end_at": _ts(15),
            "ended_at": _ts(16), "seconds": 9},
        1: {"unit": 1, "aml_job_status": "Failed"},  # no _status/1.json was ever written
    }
    timing = runners._derive_timing(
        run_id="crash", t_start=_ts(0), t_last_submit=_ts(1), t_end=_ts(25),
        submitted_at=submitted_at, returned_at=returned_at, reports=reports,
        poll_interval_seconds=10,
    )
    j1 = timing["jobs"][1]
    assert j1["job_admission_seconds"] is None
    assert j1["import_seconds"] is None
    assert j1["dispatch_overhead_seconds"] is None
    # the surviving job's derivation is unaffected
    assert timing["jobs"][0]["job_admission_seconds"] == pytest.approx(5.0)


# --- end-to-end: _aml_submit_and_wait writes _timing.json -------------------------

class _NS(types.SimpleNamespace):
    pass


class _FakeMLClient:
    """Mirrors `test_scale_runner.py::_FakeMLClient`: fakes exactly the surface
    `_aml_submit_and_wait` touches."""

    def __init__(self, job_statuses: list[str]):
        self._job_statuses = job_statuses
        self.submitted: list = []
        self.jobs = _NS(create_or_update=self._create_or_update, get=self._get)

    def _create_or_update(self, job):
        idx = len(self.submitted)
        self.submitted.append(job)
        return _NS(name=f"job-{idx}")

    def _get(self, name):
        idx = int(name.rsplit("-", 1)[1])
        return _NS(status=self._job_statuses[idx])


def _write_report(run_root, k, **fields):
    status = {"unit": k, "status": "ok", "seconds": 1.0,
              "process_start_at": _ts(1), "work_start_at": _ts(1.1),
              "work_end_at": _ts(2), "ended_at": _ts(2.1)}
    status.update(fields)
    with fs.open(f"{run_root}/_status/{k}.json", "w") as f:
        json.dump(status, f)


def test_aml_submit_and_wait_writes_timing_json_on_success():
    run_root = "memory://timing_ok/runs/r1"
    _write_report(run_root, 0)
    _write_report(run_root, 1)
    ml_client = _FakeMLClient(["Completed", "Completed"])
    jobs = {0: object(), 1: object()}

    result = runners._aml_submit_and_wait(ml_client, jobs, run_root, "r1",
                                          poll_interval_seconds=1)
    assert result["job_statuses"] == {0: "Completed", 1: "Completed"}

    with fs.open(f"{run_root}/_timing.json", "r") as f:
        timing = json.load(f)
    assert timing["run_id"] == "r1"
    assert set(timing["jobs"].keys()) == {"0", "1"}  # JSON round-trip: keys stringify
    assert timing["jobs"]["0"]["job_admission_seconds"] is not None


def test_aml_submit_and_wait_writes_timing_json_even_when_a_job_fails():
    """spec 40 D3: a crashed dispatch must still leave the earlier telemetry on disk --
    _timing.json is written BEFORE the RuntimeError is raised."""
    run_root = "memory://timing_fail/runs/r1"
    _write_report(run_root, 0, status="ok")
    _write_report(run_root, 1, status="failed", error="boom")
    ml_client = _FakeMLClient(["Completed", "Failed"])
    jobs = {0: object(), 1: object()}

    with pytest.raises(RuntimeError, match=r"\[1\]"):
        runners._aml_submit_and_wait(ml_client, jobs, run_root, "r1",
                                     poll_interval_seconds=1)

    assert fs.exists(f"{run_root}/_timing.json")
    with fs.open(f"{run_root}/_timing.json", "r") as f:
        timing = json.load(f)
    assert timing["jobs"]["0"]["job_admission_seconds"] is not None


# --- the first_admission anchor (revised 2026-07-29 against real data) --------------

def _overlapping_submit_stamps():
    """The shape run 20260729T132222Z actually produced: 32 jobs submitted over ~40 s,
    the earliest node starting BEFORE the last submission went out. Two jobs is enough to
    reproduce it -- submit at t=1 and t=40, first node executing at t=35."""
    submitted_at = {0: _ts(1), 1: _ts(40)}
    returned_at = {0: _ts(200), 1: _ts(200)}
    reports = {
        0: {"process_start_at": _ts(35), "work_start_at": _ts(36), "work_end_at": _ts(110),
            "ended_at": _ts(111), "seconds": 74},
        1: {"process_start_at": _ts(70), "work_start_at": _ts(71), "work_end_at": _ts(150),
            "ended_at": _ts(151), "seconds": 79},
    }
    return submitted_at, returned_at, reports


def test_first_admission_is_not_negative_merely_because_submitting_overlapped_it():
    """Anchored on the LAST submission this leg read -5.0 s on a healthy run -- a node
    started before the final job was submitted. Submission and admission overlap, so they
    cannot be adjacent legs. Anchored on the FIRST submission it is the honest "time until
    a node was executing": 35 - 1 = 34 s."""
    submitted_at, returned_at, reports = _overlapping_submit_stamps()
    timing = runners._derive_timing(
        run_id="overlap", t_start=_ts(0), t_first_submit=_ts(1), t_last_submit=_ts(40),
        t_end=_ts(210), submitted_at=submitted_at, returned_at=returned_at,
        reports=reports, poll_interval_seconds=10,
    )
    wall = timing["wall"]
    assert wall["first_admission_seconds"] == pytest.approx(34.0)
    assert wall["driver_prep_seconds"] == pytest.approx(1.0)     # before ANY submit
    assert wall["submission_span_seconds"] == pytest.approx(39.0)  # reported, not a leg


def test_the_split_still_telescopes_with_the_new_anchor():
    """Additivity is the property the whole forensic story rests on (D11); moving a
    breakpoint must not cost it."""
    submitted_at, returned_at, reports = _overlapping_submit_stamps()
    t_start, t_end = _ts(0), _ts(210)
    wall = runners._derive_timing(
        run_id="overlap", t_start=t_start, t_first_submit=_ts(1), t_last_submit=_ts(40),
        t_end=t_end, submitted_at=submitted_at, returned_at=returned_at,
        reports=reports, poll_interval_seconds=10,
    )["wall"]
    total = sum(wall[k] for k in (
        "driver_prep_seconds", "first_admission_seconds", "execution_window_seconds",
        "teardown_detect_seconds", "post_collect_seconds",
    ))
    assert total == pytest.approx(runners._seconds_between(t_start, t_end), abs=1e-6)
    # and submission_span is deliberately NOT part of it (it overlaps first_admission)
    assert wall["submission_span_seconds"] > 0


def test_first_admission_still_goes_negative_on_real_clock_skew():
    """D11's actual purpose for a negative: the skew bound was exceeded. That signal was
    being drowned out by the overlap artefact -- now a negative means only this."""
    wall = runners._derive_timing(
        run_id="skew", t_start=_ts(0), t_first_submit=_ts(10), t_last_submit=_ts(10),
        t_end=_ts(100), submitted_at={0: _ts(10)}, returned_at={0: _ts(90)},
        reports={0: {"process_start_at": _ts(2),   # node clock reads BEFORE the submit
                     "work_start_at": _ts(3), "work_end_at": _ts(60),
                     "ended_at": _ts(61), "seconds": 57}},
        poll_interval_seconds=10,
    )["wall"]
    assert wall["first_admission_seconds"] == pytest.approx(-8.0)


def test_t_first_submit_defaults_to_t_last_submit_for_an_old_caller():
    submitted_at, returned_at, reports = _two_job_stamps()
    wall = runners._derive_timing(
        run_id="r1", t_start=_ts(0), t_last_submit=_ts(2), t_end=_ts(35),
        submitted_at=submitted_at, returned_at=returned_at, reports=reports,
        poll_interval_seconds=30,
    )["wall"]
    assert wall["driver_prep_seconds"] == pytest.approx(2.0)
    assert wall["submission_span_seconds"] == pytest.approx(0.0)


# --- #80: snakemake is an optional extra, so its absence must name the extra --------
# It is only ever a subprocess (`sys.executable -m snakemake`), so without this guard a
# missing install surfaces as a child-process returncode far from the user's call.


def test_require_snakemake_names_the_local_extra_when_absent(monkeypatch):
    monkeypatch.setattr(runners.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match=r"\[local\]"):
        runners._require_snakemake()


def test_require_snakemake_is_a_no_op_when_installed(monkeypatch):
    monkeypatch.setattr(runners.importlib.util, "find_spec", lambda name: object())
    runners._require_snakemake()


# --- #122: _run echoes the child's stderr and returns a bounded tail ----------------


def test_run_returns_the_stderr_tail_and_echoes_it(capfd):
    import sys

    code = "import sys; sys.stderr.write('x' * 5000 + '\\n'); sys.stderr.write('final line\\n'); sys.exit(3)"
    returncode, tail = runners._run([sys.executable, "-c", code])
    assert returncode == 3
    assert len(tail) <= 4000 and tail.endswith("final line\n")
    assert "final line" in capfd.readouterr().err


def test_run_keeps_the_shutdown_output_after_ctrl_c(monkeypatch):
    import sys

    code = (
        "import signal, sys, time\n"
        "def h(*a):\n"
        "    sys.stderr.write('shutdown msg\\n'); sys.stderr.flush(); sys.exit(130)\n"
        "signal.signal(signal.SIGINT, h)\n"
        "sys.stderr.write('ready\\n'); sys.stderr.flush()\n"
        "while True: time.sleep(0.1)\n"
    )

    class _InterruptOnce:
        def __init__(self):
            self.armed = True

        def write(self, text):
            if self.armed:
                self.armed = False
                raise KeyboardInterrupt

    monkeypatch.setattr(sys, "stderr", _InterruptOnce())
    returncode, tail = runners._run([sys.executable, "-c", code])
    assert returncode == 130
    assert "shutdown msg" in tail
