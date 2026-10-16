"""The cluster sibling of `demos/e2e_austria.py` (spec 40): one script, one command,
unattended, from an empty Azure, emitting a `timings.json` of the same shape so
local-vs-cluster is a diff rather than an essay.

Same **eight steps** as the local demo (D1): `0_preflight` .. `7_report`. `3_training_data`
is ONE call that dispatches both the cube-build fan-out and the flatten reduce
internally (`api.create_training_data(runner="aml")`) -- which is exactly why dispatch
telemetry is a file (`<run_root>/_timing.json`, ADR 0021) rather than a return value: one
step here contains two runs. `1_tiling` duplicates work `run_inference` re-tiles
internally in its own preflight -- symmetric with the local demo, cheap, and load-bearing
for the step-for-step comparison (do not "fix" it).

The download step (`2_download`, D13 as AMENDED 2026-07-28) uses the local demo's window
and bands -- 2018-04-01..09-30, B04/B08/B8A/SCL, max_cloudcover=70 -- but sources them
from **MPC, like every cluster run since P1** (run-book 37 Phase 3), not from CDSE.

The original D13 pinned CDSE to make `2_download` a like-for-like row against the local
laptop run. That traded away more than it bought: **CDSE dispatches exactly ONE job**
(spec 37 D1) so the download leg measured no scale-out at all -- contradicting D11's own
"~49 samples (16 download + ...)" -- while also requiring the CDSE credential dance and
risking the 30-day quota throttle partway through ~80 GB. MPC fans out across the
cluster, is anonymous, and copies inside West Europe. The cost, accepted knowingly: this
step is no longer comparable to the local run's 207 CDSE granules (§7 already said the
demo is not comparable to the *cluster* numbers; now the download row is not comparable
to the *local* ones either -- it is comparable to run-book 37, which is the series that
matters).

**Claude never runs this script** (CLAUDE.md): it is handed to the operator with the
prerequisites in `demos/E2E_AUSTRIA_AML.md` §8 (VM inside the project's compute subnet,
`az login`, `--dry-run` first).

Run as (see E2E_AUSTRIA_AML.md for the full env-var contract):

    export AZ_RG=... AZ_ML_WORKSPACE=... AZ_SUBSCRIPTION_ID=... AZ_CLUSTER=...
    export AZ_UAMI_CLIENT_ID=...
    export AZ_ROOT=abfss://<fs>@<account>.dfs.core.windows.net/<prefix>
    export AZ_ENV_NAME=fsd-aml-env AZ_ENV_VERSION=...
    export AZ_INFER_ENV_NAME=fsd-infer-env AZ_INFER_ENV_VERSION=...
    # no CDSE credentials: MPC is anonymous (D13 as amended), and `run_aml_download`
    # refuses creds for an MPC run rather than staging a secret on blob for nothing.
    python demos/e2e_austria_aml.py --fresh --dry-run
    python demos/e2e_austria_aml.py --fresh --confirm-spend      # after az login, under tmux
    python demos/e2e_austria_aml.py --run-id <id> --confirm-spend   # resume a partial run
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import re
import signal
import sys
import time

import geopandas as gpd
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ["PYTHONPATH"] = _HERE + os.pathsep + os.environ.get("PYTHONPATH", "")

from adapters import DemoRF  # noqa: E402

import rise_eo  # noqa: E402
from rise_eo import config, grid  # noqa: E402
from rise_eo.api import TrainingData  # noqa: E402
from rise_eo.catalog.catalog import TileCatalog  # noqa: E402
from rise_eo.model import bundle  # noqa: E402
from rise_eo.sources import mpc  # noqa: E402
from rise_eo.storage import fs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))  # workspace root

# --- D13 (amended): the local demo's window/bands/cloudcover, sourced from MPC ------
ROI_FP = os.path.join(ROOT, "shapefiles/AT_ROI.geojson")
TRAIN_FP = os.path.join(ROOT, "shapefiles/AT_2018_TRAIN.geojson")
ID_COL = "fid"
LABEL_COL = "crop"
BANDS = ["B04", "B08", "B8A", "SCL"]
COLLECTION = config.SATELLITE_S2L2A   # the download's collection: names the archive subfolder
ARCHIVE_DIRNAME = "imagery"           # `{AZ_ROOT}/imagery`: the one shared archive (spec 59 D12)
# SCL mask classes are declared on the sentinel-2-l2a CollectionDeclaration now (spec 58
# D3), not a verb parameter -- they happen to already be this list, so no behavior change.
# `collection=` (default "sentinel-2-l2a") and `properties_filter=` (spec 58 D9) stay at
# their defaults throughout: S2 L2A declares `mosaic_partition=()`, so nothing needs
# selecting and an empty filter leaves every cube path byte-identical to pre-P2 (AC13).
# `rise_eo.collections.known()` lists what else is registered (`sentinel-1-rtc` since P2).
MOSAIC_DAYS = 20
START = datetime.datetime(2018, 4, 1)
END = datetime.datetime(2018, 9, 30)
# The guardrail (D4: "the ROI x window is bigger than intended"), NOT a prediction.
# 207 is the local demo's CDSE granule count; MPC queries a different catalogue and
# de-duplicates reprocessed acquisitions (spec 33), so its count for the same
# window/cloudcover will not be identical. `--dry-run` reports MPC's real number before
# any spend -- raise this if the dry run is legitimately higher.
MAX_TILES = 250
MAX_CLOUDCOVER = 70

STEP_LABELS = [
    "0_preflight", "1_tiling", "2_download", "3_training_data",
    "4_train_bundle", "5_run_inference", "6_plots", "7_report",
]

_LOCAL_OUTDIR_BASE = os.path.join(_HERE, "..", "tests/outputs/demo_e2e_aml")  # gitignored
FIGDIR = os.path.join(_HERE, "figures")

STEP_RESULTS: dict = {}   # {step: spec-24 result dict}, rewritten to timings.json every step


def log(msg):
    print(f"\n=== {msg} ===", flush=True)


def ok(msg):
    print(f"  ✓ {msg}", flush=True)


def fail(msg):
    raise PreflightFailure(msg)


class PreflightFailure(RuntimeError):
    pass


class DemoInterrupted(RuntimeError):
    pass


def _install_signal_handlers():
    """D7: SIGINT/SIGTERM both exit through the same clean path -- a step's
    `_result.json` is only ever written once its call returns (D3), so there is no
    partial file to corrupt; every step that already finished keeps its numbers, and
    the operator gets the resume line instead of a traceback.

    SIGINT is handled explicitly rather than left to Python's default. The default
    raises `KeyboardInterrupt`, which is a `BaseException`: it slips past `run_step`'s
    `except Exception` *and* past `main`'s `except DemoInterrupted`, so a ^C on an
    unattended run would print a raw traceback and skip the resume hint -- the one
    moment the operator most needs it."""
    def _handler(signum, frame):
        raise DemoInterrupted(f"signal {signum}")

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


# --- driver location (D10) -----------------------------------------------------------

def _driver_location() -> dict:
    """Where THIS demo run's driver executed -- recorded, no causal claim derived from
    it (D10). No network calls: a VM-vs-laptop guess from environment, not a fact."""
    on_azure_vm = os.path.exists("/var/lib/waagent") or bool(os.environ.get("AZ_ON_VM"))
    return {
        "hostname": os.uname().nodename,
        "on_azure_vm": on_azure_vm,
        "note": "best-effort: set AZ_ON_VM=1 on the VM if waagent is absent",
    }


# --- step-level plumbing: resumable, self-contained timings (D3/D5/D9) --------------

class Demo:
    def __init__(self, run_id: str, resume: bool):
        self.run_id = run_id
        self.resume = resume
        self.outdir = os.path.join(_LOCAL_OUTDIR_BASE, run_id)
        os.makedirs(self.outdir, exist_ok=True)
        self.timings_fp = os.path.join(self.outdir, "timings.json")
        self.t0 = time.time()

    def result_fp(self, step: str) -> str:
        return os.path.join(self.outdir, f"{step}_result.json")

    def load_result(self, step: str) -> dict | None:
        fp = self.result_fp(step)
        if self.resume and os.path.exists(fp):
            with open(fp) as f:
                return json.load(f)
        return None

    def run_step(self, step: str, fn, *args, **kwargs):
        """D3/D5: skip instantly if this step's result already exists (resume); else run
        it, write `<step>_result.json`, and rewrite timings.json (self-contained, D9)."""
        cached = self.load_result(step)
        if cached is not None:
            STEP_RESULTS[step] = cached
            print(f"  [{step}] already done ({cached.get('seconds', 0):.1f}s) -- resumed",
                  flush=True)
            self._write_timings()
            return cached

        log(step)
        t = time.time()
        try:
            result = fn(*args, **kwargs)
        except DemoInterrupted:
            raise
        except Exception as exc:  # noqa: BLE001 - a failed step is reported, not swallowed
            result = {"step": step, "status": "failed", "seconds": round(time.time() - t, 3),
                      "error": str(exc)}
            STEP_RESULTS[step] = result
            self._write_timings()
            raise
        result.setdefault("step", step)
        result.setdefault("status", "ok")
        result["seconds"] = round(time.time() - t, 3)
        result.setdefault("error", None)
        with open(self.result_fp(step), "w") as f:
            json.dump(result, f, indent=2, default=str)
        STEP_RESULTS[step] = result
        print(f"  [{step}] took {result['seconds']:.1f}s", flush=True)
        self._write_timings()
        return result

    def _write_timings(self):
        completed = [STEP_RESULTS[s] for s in STEP_LABELS if s in STEP_RESULTS]
        payload = {
            # The SUM OF THE STEPS, not this process's wall. A resumed demo run replays
            # completed steps from disk in milliseconds, so the final process's wall is
            # *smaller than the work it reports*: the 2026-07-29 run wrote
            # total_seconds=640.7 over 1470.0 s of steps, which reads as a demo run twice
            # as fast as it was. D9 hands ONE file to a reader who cannot know how many
            # processes produced it, so the headline number must not depend on that.
            "total_seconds": round(sum(s.get("seconds") or 0 for s in completed), 1),
            "process_wall_seconds": round(time.time() - self.t0, 1),
            "resumed": self.resume,
            "run_id": self.run_id,
            "driver": _driver_location(),
            "steps": completed,
        }
        with open(self.timings_fp, "w") as f:
            json.dump(payload, f, indent=2, default=str)


# --- step 0: preflight (D4) ---------------------------------------------------------

def _make_ml_client():
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential

    return MLClient(
        DefaultAzureCredential(),
        os.environ["AZ_SUBSCRIPTION_ID"], os.environ["AZ_RG"], os.environ["AZ_ML_WORKSPACE"],
    )


def _measure_clock_skew(root: str) -> dict:
    """D11: write a scratch blob, read back the stamp STORAGE put on it, compare to the
    driver's own clock -- this session measured ~8s of laptop-vs-Azure skew, a third of
    a warm admission. Recorded, not fatal (D4).

    The write is bracketed (`before`/`after`) because the storage account stamps the
    blob at some unknown instant *during* the call: comparing against `before` alone
    charges the whole round-trip latency to skew. The midpoint is the estimate and the
    half-width is its own uncertainty, reported alongside -- a skew figure every
    admission number inherits should not quietly include the network.

    Returns `{"seconds": None, ...}` when the backend records no mtime at all, so the
    bound reads as *unmeasured* rather than as a confident zero."""
    probe_url = f"{root.rstrip('/')}/.fsd_clock_skew_probe"
    before = pd.Timestamp.now(tz="UTC")
    with fs.open(probe_url, "w") as f:
        f.write("skew-probe")
    after = pd.Timestamp.now(tz="UTC")
    remote_mtime = fs.modified(probe_url)
    with contextlib.suppress(Exception):
        fs.rm(probe_url)

    if remote_mtime is None:
        return {"seconds": None, "uncertainty_seconds": None,
                "note": "backend records no mtime -- skew is UNMEASURED, not zero."}
    midpoint = before + (after - before) / 2
    return {
        "seconds": (pd.Timestamp(remote_mtime) - midpoint).total_seconds(),
        "uncertainty_seconds": (after - before).total_seconds() / 2,
        "note": "driver clock vs storage-account clock; every job_admission figure "
                "carries this bound (D11).",
    }


# Every third-party module the DRIVER imports after preflight, and the extra that
# supplies it. D4 says preflight is total and fails in seconds; a missing local import is
# the cheapest possible failure and must never surface later. It did on 2026-07-29 --
# `ModuleNotFoundError: joblib` at `4_train_bundle`, three steps and one ~80 GB download
# past the point where a one-line `pip install` would have fixed it.
#
# `adlfs`/`azure.ai.ml` fail earlier still (module import and `_make_ml_client`), so they
# never reach this check; they are listed anyway so the message stays complete if the
# steps are ever reordered, and so this table is the single place that answers "what does
# the driver actually need installed?".
_DRIVER_DEPS = [
    ("adlfs",              "azure",         "blob I/O through the storage seam"),
    ("azure.identity",     "azure",         "DefaultAzureCredential"),
    ("azure.ai.ml",        "aml",           "dispatching every AML job"),
    ("planetary_computer", "mpc",           "MPC discovery (preflight + 2_download)"),
    ("s2",                 "grid",          "1_tiling, and run_inference's ROI re-tiling"),
    ("s2cell",             "grid",          "1_tiling, and run_inference's ROI re-tiling"),
    ("sklearn",            "model-example", "4_train_bundle (RandomForest + LabelEncoder)"),
    ("joblib",             "model-example", "4_train_bundle (model serialisation)"),
    ("matplotlib",         "model-example", "6_plots"),
]


def _missing_driver_deps() -> list[str]:
    """Which of `_DRIVER_DEPS` are absent, as D4-shaped error lines naming the exact fix."""
    import importlib.util

    missing = []
    for module, extra, why in _DRIVER_DEPS:
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append((module, extra, why))
    if not missing:
        return []
    extras = sorted({extra for _, extra, _ in missing})
    return [f"missing driver dependency {module!r} -- needed for {why} "
            f"(comes from the '{extra}' extra)" for module, extra, why in missing] + [
        "fix all of the above at once:  pip install -e \".[" + ",".join(
            sorted({"dev", "azure", "aml", *extras})) + "]\""
    ]


@contextlib.contextmanager
def _timed_check(name: str, into: dict):
    """D7: preflight is a series of opaque network calls, and until now it printed nothing
    until all of them finished -- so "why is preflight slow?" could only be answered by
    re-running with a stopwatch. Prints each check as it starts and again with its
    duration, and records the breakdown in the step's `_result.json` so a slow preflight
    stays diagnosable from `timings.json` alone afterwards (D9).

    The user's standing preference: a stage line is not progress. These are seconds-scale
    checks, so a line each is the right granularity -- no ETA to invent."""
    print(f"  ... {name}", flush=True)
    t = time.time()
    try:
        yield
    finally:
        dt = time.time() - t
        into[name] = round(dt, 2)
        print(f"  {'!' if dt > 10 else '+'} {name}: {dt:.1f}s", flush=True)


def step_preflight(ml_client, root: str) -> dict:
    errs: list[str] = _missing_driver_deps()
    warnings: list[str] = []
    check_seconds: dict = {}

    # D4 wants each failure to name its own exact fix, so "a credential resolves" is
    # checked on its own -- NOT by whether some cluster call happens to work. Asking
    # for an ARM token is the cheapest thing that fails when, and only when, there is
    # no `az login` and no managed identity; a wrong AZ_CLUSTER then reports as a
    # cluster problem below rather than as a bogus credential problem.
    try:
        from azure.identity import DefaultAzureCredential

        # Can be the slow one: DefaultAzureCredential walks a chain (env -> managed
        # identity via IMDS -> az CLI -> ...), and an IMDS probe on a VM whose identity
        # is not wired for this flow burns its retry budget before falling through.
        with _timed_check("credential resolves", check_seconds):
            DefaultAzureCredential().get_token("https://management.azure.com/.default")
    except Exception as exc:  # noqa: BLE001
        errs.append(f"no credential resolves on this host -- run `az login` over SSH, or give "
                    f"the VM a managed identity (E2E_AUSTRIA_AML.md §8.1 step 2): {exc}")

    try:
        probe = f"{root.rstrip('/')}/.fsd_preflight_probe"
        with _timed_check("blob read+write", check_seconds):
            with fs.open(probe, "w") as f:
                f.write("preflight")
            with fs.open(probe, "r") as f:
                f.read()
            fs.rm(probe)
    except Exception as exc:  # noqa: BLE001
        errs.append(f"blob read+write failed at {root!r} -- storage firewall denying this "
                    f"host? (see E2E_AUSTRIA_AML.md §8): {exc}")

    try:
        with _timed_check("cluster resolves", check_seconds):
            compute = ml_client.compute.get(os.environ["AZ_CLUSTER"])
        state = getattr(compute, "provisioning_state", None)
        if state not in (None, "Succeeded"):
            errs.append(f"cluster {os.environ['AZ_CLUSTER']!r} not ready: provisioning_state={state!r}")
    except Exception as exc:  # noqa: BLE001
        errs.append(f"cluster {os.environ['AZ_CLUSTER']!r} not found/unreachable: {exc}")

    # NB the version var is NOT f"{name_var}_VERSION" -- the documented contract
    # (E2E_AUSTRIA_AML.md §8.2, and what `runner_kwargs` reads below) is
    # AZ_ENV_NAME/AZ_ENV_VERSION, not AZ_ENV_NAME_VERSION. Naming them as a pair
    # keeps the two readers of these vars from drifting apart again.
    for name_var, version_var, label in (
        ("AZ_ENV_NAME", "AZ_ENV_VERSION", "build"),
        ("AZ_INFER_ENV_NAME", "AZ_INFER_ENV_VERSION", "inference"),
    ):
        unset = [v for v in (name_var, version_var) if not os.environ.get(v)]
        if unset:
            # An unset env var and an unbuilt Environment need different fixes (D4).
            errs.append(f"{label} Environment: {', '.join(unset)} not set "
                        f"(E2E_AUSTRIA_AML.md §8.2).")
            continue
        try:
            with _timed_check(f"{label} Environment resolves", check_seconds):
                ml_client.environments.get(name=os.environ[name_var],
                                           version=os.environ[version_var])
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{label} Environment "
                        f"{os.environ[name_var]}:{os.environ[version_var]!r} does not resolve -- "
                        f"build it first (D4 is verify-only): {exc}")

    for name, fp in [("ROI", ROI_FP), ("train", TRAIN_FP)]:
        if not os.path.exists(fp):
            errs.append(f"{name} file not found: {fp}")

    n_tiles = None
    if not errs:
        # Usually the dominant term: a paginated MPC STAC search over the whole ROI x
        # 6-month window, whose items are signed as they stream in. Third-party latency,
        # not ours -- but it is the number to look at when preflight feels slow.
        with _timed_check("MPC discovery (paginated STAC search)", check_seconds):
            tiles = mpc.query_catalog(ROI_FP, START, END, max_cloudcover=MAX_CLOUDCOVER)
        n_tiles = len(tiles)
        if n_tiles > MAX_TILES:
            errs.append(f"{n_tiles} discovered tiles exceed max_tiles={MAX_TILES}.")

    clock_skew = None
    if not any("blob read+write" in e for e in errs):
        with _timed_check("clock skew probe", check_seconds):
            clock_skew = _measure_clock_skew(root)
        if clock_skew["seconds"] is None:
            warnings.append(clock_skew["note"])
        elif abs(clock_skew["seconds"]) > 5:
            warnings.append(
                f"clock skew {clock_skew['seconds']:.1f}s "
                f"(±{clock_skew['uncertainty_seconds']:.1f}s) -- job_admission figures "
                "carry this bound (D11).")

    if errs:
        fail("preflight failed:\n  - " + "\n  - ".join(errs))

    return {"n_discovered_tiles": n_tiles,
            "clock_skew_seconds": None if clock_skew is None else clock_skew["seconds"],
            "clock_skew": clock_skew, "check_seconds": check_seconds,
            "warnings": warnings}


# --- step 1: tiling (D1: duplicated on both sides, load-bearing, do not "fix") ------

def step_tiling(outdir: str) -> dict:
    grids = grid.roi_to_s2_grids(ROI_FP, grid_size_km=5, scale_fact=1.1)
    grids_fp = os.path.join(outdir, "inference_s2_grids.geojson")
    grids.to_file(grids_fp, driver="GeoJSON")
    ok(f"{len(grids)} grid cells -> {grids_fp}")
    return {"n_cells": len(grids), "grids_fp": grids_fp}


# --- dispatch telemetry discovery (D2/D9) -------------------------------------------
#
# `_aml_submit_and_wait` writes `<run_root>/_timing.json` under a run_id it generates
# itself (`run_aml`/`run_aml_download`/`run_aml_flatten`/`run_aml_inference` all default
# `run_id=None` -> a fresh timestamp) -- the demo script never learns that id from the
# `rise_eo.download`/`create_training_data`/`run_inference` return values (they don't carry
# it, ADR 0021). So each dispatching step snapshots which `_timing.json` files exist
# under `root/runs/` before its call and diffs after, embedding what's new into its own
# `_result.json` -- which is how `timings.json` stays self-contained (D9) without this
# script hardcoding any run_id plumbing.

def _list_run_ids(root: str) -> set:
    """`fs.glob` returns the filesystem's OWN path form, not `root`'s scheme (adlfs
    gives `container/path/…`, no `abfss://` -- the trap `api._output_key` exists for).
    So this extracts just the `run_id` (the tail component before `_timing.json`) and
    lets the caller rebuild a real url from `root`, rather than ever comparing or
    re-opening a glob hit directly."""
    ids = set()
    for p in fs.glob(f"{root.rstrip('/')}/runs/*/_timing.json"):
        parts = str(p).rstrip("/").replace("\\", "/").split("/")
        if len(parts) >= 3 and parts[-3] == "runs":
            ids.add(parts[-2])
    return ids


def _assert_dispatch_telemetry_complete(dispatch_timings: list, *, step: str) -> None:
    """The four in-job stamps (D2) are written by `fsd` **on the node**, i.e. by whatever
    `fsd` is baked into the AML Environment image -- NOT by the driver's checkout. An
    Environment built before spec 40 therefore produces a `_status/<k>.json` with the old
    shape: `seconds` present, `process_start_at`/`work_start_at`/`work_end_at`/`ended_at`
    absent. Everything still runs and every science output is correct, so nothing fails --
    the run just silently yields `job_admission_seconds: null` for every job, which is
    D11's headline metric and the entire reason this script exists.

    That is exactly what happened on 2026-07-29: a complete, green, ~25-minute demo run
    whose telemetry could not answer the one question it was built to answer, across 97
    jobs and four dispatches.

    Checked after the FIRST dispatch and fatal, because by then the answer is already
    known and the remaining three dispatches would add ~20 minutes and cluster spend to a
    measurement that is already void. Called from `main` *after* the step's `_result.json`
    is on disk, so `--run-id` skips the download rather than repeating it.
    """
    jobs = [j for dt in dispatch_timings for j in dt.get("jobs", {}).values()]
    if not jobs:
        return
    if any(j.get("process_start_at") is not None for j in jobs):
        return
    fail(
        f"{step}: none of the {len(jobs)} dispatched job(s) reported the D2 in-job stamps "
        "(process_start_at/work_start_at/work_end_at/ended_at), so job_admission_seconds "
        "-- D11's headline metric -- would be null for this entire demo run.\n"
        "      The stamps are written by the `fsd` INSIDE the AML Environment image, not "
        "by this checkout, so the image predates spec 40.\n"
        "      Fix: rebuild both Environments from current fsd (see "
        "docs/howto/build-the-images.md), bump AZ_ENV_VERSION / AZ_INFER_ENV_VERSION, then "
        "resume with --run-id <the id printed at the start of this run> -- the download "
        "is already on disk and skips."
    )


def _new_dispatch_timings(root: str, before: set) -> list:
    """The `_timing.json` of every run that appeared during a step's call, **in execution
    order**.

    Ordered by `wall.t_start`, NOT by run_id. Run ids look like sortable timestamps and are
    not: `create_training_data` mints its own right after preflight and hands it to the
    **flatten reduce**, while the **build fan-out** mints one later, when it actually
    dispatches -- so the flatten's id sorts FIRST despite running SECOND. Measured on
    20260729T132222Z: flatten `20260729T132929Z` executed 13:32:58, build
    `20260729T133005Z` executed 13:30:07. Sorted by id, `3_training_data[0]` was the
    reduce and `[1]` the fan-out, which is backwards from how anyone reads an index.
    """
    out = []
    for run_id in sorted(_list_run_ids(root) - before):
        with fs.open(f"{root.rstrip('/')}/runs/{run_id}/_timing.json", "r") as f:
            out.append(json.load(f))
    out.sort(key=lambda d: (d.get("wall") or {}).get("t_start") or "")
    return out


# --- step 2: download (D13 scope, D14 archive-trust assertions) --------------------

_BASELINE_RE = re.compile(r"_N(\d{2})(\d{2})_")


def _asset_key(path: str) -> str:
    """`…/<granule_id>/<filename>` -> `<granule_id>/<filename>`.

    Same trap as `api._output_key`: `fs.glob` returns the *filesystem's* path form
    (adlfs gives `container/path/…`, with no `abfss://` scheme), so a globbed hit never
    string-equals a url built from the catalog's `local_folderpath`. The last two
    components are both scheme-independent and unique.

    Two components, not one: the catalog's `files` column holds bare **basenames**
    (`B04.tif,B08.tif,B8A.tif,MTD_TL.xml,SCL.tif`) which are *identical on every row*,
    so comparing basenames alone collapses 207 granules into 5 names and cannot detect
    a missing granule at all. The granule id is what makes the key unique.
    """
    return "/".join(str(path).rstrip("/").replace("\\", "/").split("/")[-2:])


def _expected_offset(granule_id: str) -> int | None:
    """The additive reflectance offset this granule's own processing baseline implies
    (`…_N0500_…` -> 05.00 -> -1000; -1000 for baseline >= 04.00, else 0 -- ESA, via
    `rise_eo.sources._s2_radiometry`), or `None` if the id carries no baseline token.

    `None` is the common case on MPC, whose item ids drop the `N####` field entirely
    (`S2B_MSIL2A_20220219T100019_R122_T33UWP_20220225T100522` -- verified against
    `tests/outputs/mpc_baseline/catalog.parquet`). That is exactly why this is only ONE
    of D14's two offset checks: on its own it would be a silent no-op for every MPC
    granule. `_assert_cog_tags_match_catalog` is the source-agnostic one.
    """
    m = _BASELINE_RE.search(str(granule_id))
    if not m:
        return None
    return -1000 if (int(m.group(1)), int(m.group(2))) >= (4, 0) else 0


def _assert_cog_tags_match_catalog(row) -> None:
    """D14's "`scale`/`offset`/`nodata` correct on a sample": compare the catalog row
    against the **COG's own stamped GDAL tags** — two independent records of one fact,
    written by different code paths (`catalog.append` vs `raster.cog.stamp_gdal_tags`).

    This is the check that works for BOTH sources, since it reads the granule rather
    than parsing its name. It also pins the precise bug that already shipped once
    (`c2bf1f1`, the black-tile fix): the tag is in **reflectance units** to match
    `scale=1/10000`, because a viewer's `unscale=true` computes `DN*scale + offset`,
    while the catalog column stays in **DN**. Stamping the DN offset (-1000) next to a
    1/10000 scale made titiler render `DN/10000 - 1000` — pure black — and no test
    caught it, because the two records agreed on the same wrong value.

    A header read (no pixels decoded), one band per sampled granule.
    """
    from rise_eo.catalog.declaration import S2_L2A_DECLARATION
    from rise_eo.raster import rio_open

    band_file = next(
        (f.strip() for f in str(row["files"]).split(",")
         if f.strip().endswith(".tif")
         and S2_L2A_DECLARATION.is_radiometry_band(os.path.splitext(f.strip())[0])),
        None,
    )
    if band_file is None:
        return  # no reflectance band in this row (e.g. an SCL-only download)

    fp = f"{str(row['local_folderpath']).rstrip('/')}/{band_file}"
    with rio_open(fp) as src:
        tag_scale, tag_offset, tag_nodata = src.scales[0], src.offsets[0], src.nodata

    want_offset = float(row["offset"]) * config.S2_REFLECTANCE_SCALE
    if abs(tag_offset - want_offset) > 1e-9:
        fail(f"D14: {fp!r} stamps OFFSET={tag_offset!r}, but the catalog declares "
             f"offset={row['offset']!r} DN (= {want_offset} in reflectance units). The COG "
             "and the catalog disagree about radiometry -- one of them is serving/building "
             "wrong pixels (this is the c2bf1f1 black-tile class of bug).")
    if abs(tag_scale - config.S2_REFLECTANCE_SCALE) > 1e-12:
        fail(f"D14: {fp!r} stamps SCALE={tag_scale!r}, expected "
             f"{config.S2_REFLECTANCE_SCALE} -- an unscale=true viewer would misrender it.")
    if tag_nodata != row["nodata"]:
        fail(f"D14: {fp!r} declares nodata={tag_nodata!r} but the catalog says "
             f"{row['nodata']!r}.")


def _assert_archive_trustworthy(catalog_fp: str, dst_folderpath: str) -> dict:
    """D14: fold trust assertions into the download step (not a new step, D1 survives).
    Seconds of listing/tag reads -- the expensive cross-source pixel comparison stays in
    run-book 37-verify-archive (tag `docs-archive-2026`).

    D14 names `scale`/`offset`/`nodata`; the catalog schema carries `offset` and
    `nodata` (spec 34 §1) and no `scale` column -- scale is a fixed per-band constant
    (`config.S2_REFLECTANCE_SCALE`), stamped on the COG rather than declared per row,
    so there is nothing per-granule to disagree with.
    """
    catalog = TileCatalog(catalog_fp)
    rows = catalog.read()
    if catalog.declaration is None:
        fail("D14: catalog carries no stamped CollectionDeclaration.")

    # Every declared asset, keyed `<granule_id>/<filename>`; the extensions come from
    # the catalog itself (not a hardcoded `*.tif`) so a declared sidecar -- CDSE
    # declares `MTD_TL.xml` alongside the bands -- is globbed for rather than reported
    # as missing from a listing that never looked for it.
    # Scoped to this collection's directory: `dst_folderpath` is the ONE shared archive
    # (spec 59 D12), so a bare `**` glob would also list other collections' granules (S1).
    # Other ROIs' granules of THIS collection are in scope: they share this catalog.
    declared: set[str] = set()
    for _, row in rows.iterrows():
        granule = str(row["local_folderpath"]).rstrip("/").replace("\\", "/").split("/")[-1]
        declared |= {f"{granule}/{f.strip()}"
                     for f in str(row["files"]).split(",") if f.strip()}
    exts = {os.path.splitext(k)[1] for k in declared if os.path.splitext(k)[1]}
    present: set[str] = set()
    for ext in sorted(exts):
        present |= {_asset_key(p) for p in fs.glob(f"{dst_folderpath}/{COLLECTION}/**/*{ext}")}

    missing = declared - present
    if missing:
        fail(f"D14: {len(missing)} of {len(declared)} declared asset(s) not found on blob, "
             f"e.g. {sorted(missing)[:3]}.")
    undeclared = present - declared
    # not fatal (a catalog can be a strict subset of what's on blob, e.g. a partial re-run's
    # leftovers) but worth surfacing.

    sample = rows.sample(min(10, len(rows)), random_state=7)
    n_baseline_checked = 0
    for _, row in sample.iterrows():
        if row.get("nodata") != config.NODATA:
            fail(f"D14: row {row['id']!r} declares nodata={row.get('nodata')!r}, not "
                 f"config.NODATA ({config.NODATA}) -- the builder treats {config.NODATA} "
                 "as nodata, so any other declared value silently mixes real pixels "
                 "with fill.")
        if row.get("offset") not in (0, -1000):
            fail(f"D14: row {row['id']!r} declares offset={row.get('offset')!r}; ESA defines "
                 "only 0 (baseline < 04.00) and -1000 (>= 04.00).")
        expected_offset = _expected_offset(row["id"])
        if expected_offset is not None:
            n_baseline_checked += 1
            if row.get("offset") != expected_offset:
                fail(f"D14: row {row['id']!r} declares offset={row.get('offset')!r}, but its "
                     f"processing baseline implies {expected_offset} -- un-harmonized "
                     "radiometry: cubes built from this archive would be ~1000 DN off and "
                     "nothing downstream would notice.")
        # Size first, THEN the header read: a truncated asset must be reported as
        # zero-byte, not as rasterio's "not recognized as being in a supported file
        # format" -- same defect, but only one of those messages names the fix.
        for fn in str(row["files"]).split(","):
            fp = f"{str(row['local_folderpath']).rstrip('/')}/{fn.strip()}"
            if fs.size(fp) <= 0:
                fail(f"D14: zero-byte asset {fp!r}.")
        _assert_cog_tags_match_catalog(row)

    return {"n_catalog_rows": len(rows), "n_declared_assets": len(declared),
            "n_undeclared_objects": len(undeclared), "n_sampled": len(sample),
            # Reported, not silent: 0 here means every sampled id lacked a baseline
            # token (the normal MPC case), so the offset guarantee rests entirely on
            # the COG-tag comparison. A reader can see which check did the work.
            "n_offset_baseline_crosschecked": n_baseline_checked}


def step_download(ml_client, root: str, az_root: str) -> dict:
    # Spec 59 D12: imagery goes to ONE shared archive at `{AZ_ROOT}/imagery`, not under this
    # run's `root`, so a repeat run (or the next demo) downloads nothing it already holds.
    # One catalog per collection directory: the archive layout (D5; see `rise_eo.archive_catalog_filepath`).
    dst_folderpath = f"{az_root.rstrip('/')}/{ARCHIVE_DIRNAME}"
    roi_url = f"{root.rstrip('/')}/_inputs/AT_ROI.geojson"
    with open(ROI_FP, "rb") as src, fs.open(roi_url, "wb") as dst:
        dst.write(src.read())

    runner_kwargs = {"cluster": os.environ["AZ_CLUSTER"],
                     "environment": f"{os.environ['AZ_ENV_NAME']}:{os.environ['AZ_ENV_VERSION']}",
                     "root": root, "identity_client_id": os.environ["AZ_UAMI_CLIENT_ID"],
                     "ml_client": ml_client, "poll_interval_seconds": 10}

    before = _list_run_ids(root)
    # A download is a whole-asset byte copy, so MAX_TILES bounds granules, not bytes: a small
    # ROI still pulls whole 110 km tiles (~122 MB per band per tile). On a collection with
    # bigger scenes that is the dominant cost -- Sentinel-1 RTC measures ~3.7 GB per 250 km
    # scene -- and `properties_filter=` is the lever, since it narrows discovery BEFORE
    # max_tiles is counted. S2 L2A needs none of that, which is why nothing is passed here.
    # `n_shards` is deliberately NOT passed: `run_aml_download` defaults it to the
    # cluster's own `max_instances`, which is what makes this a full-width fan-out and
    # gives D11 its ~16 download admission samples. Pinning a number here would silently
    # under-use a resized cluster.
    catalog_fp = rise_eo.download(roi_url, START, END, BANDS, dst_folderpath,
                              collection=COLLECTION, source="mpc", max_tiles=MAX_TILES,
                              max_cloudcover=MAX_CLOUDCOVER,
                              runner="aml", runner_kwargs=runner_kwargs)
    dispatch_timings = _new_dispatch_timings(root, before)

    granules = len(TileCatalog(catalog_fp).read())
    if granules < 1:
        fail("no granules downloaded.")
    trust = _assert_archive_trustworthy(catalog_fp, dst_folderpath)
    ok(f"{granules} granules in catalog; D14 archive-trust assertions passed")
    return {"catalog_fp": catalog_fp, "n_granules": granules, "dispatch_timings": dispatch_timings,
            **trust}


# --- step 3: training data (D1: ONE call, dispatches build fan-out + flatten reduce) --

def step_training_data(ml_client, root: str, catalog_fp: str, adapter, local_export_fp: str) -> dict:
    train_url = f"{root.rstrip('/')}/_inputs/AT_2018_TRAIN.geojson"
    with open(TRAIN_FP, "rb") as src, fs.open(train_url, "wb") as dst:
        dst.write(src.read())

    runner_kwargs = {"cluster": os.environ["AZ_CLUSTER"],
                     "environment": f"{os.environ['AZ_ENV_NAME']}:{os.environ['AZ_ENV_VERSION']}",
                     "root": root, "identity_client_id": os.environ["AZ_UAMI_CLIENT_ID"],
                     "ml_client": ml_client, "poll_interval_seconds": 10}

    before = _list_run_ids(root)
    td = rise_eo.create_training_data(
        label_polygons=train_url, catalog_filepath=catalog_fp,
        startdate=START, enddate=END, mosaic_days=MOSAIC_DAYS, bands=BANDS,
        id_col=ID_COL, label_col=LABEL_COL,
        export_folderpath=local_export_fp, adapter=adapter,
        # `median_per_id` is the MODELLING UNIT, not a size trick (run-book 40): one
        # `np.nanmedian` row per labelled field instead of every pixel inside it. The
        # labels are field-level, so training per-pixel leaks a field's own pixels
        # across the train/test split -- that is the difference between the discredited
        # 0.696 and the honest field-wise ~0.29. Matches run-books 39/40 AND
        # `e2e_austria.py`, so D1's step-for-step mirror holds here too.
        aggregate="median_per_id",
        runner="aml", runner_kwargs=runner_kwargs,
    )
    # D1: this ONE call dispatches TWO runs internally (the build fan-out + the flatten
    # reduce) -- both appear as new run_ids here (ADR 0021: that is why telemetry is a
    # file, not a return value).
    dispatch_timings = _new_dispatch_timings(root, before)

    d = td.load()
    feats, T = d["features"], rise_eo.compute_n_timestamps(START, END, MOSAIC_DAYS)
    if feats.shape[1] != T:
        fail(f"features T={feats.shape[1]} != expected T={T}")
    ok(f"features {feats.shape}, T={T}, {len(set(d['feature_labels']))} classes")
    return {
        "export_folderpath": td.export_folderpath, "run_folderpath": td.run_folderpath,
        "n_pixels": td.n_pixels, "n_timestamps": td.n_timestamps,
        "bands": td.bands, "feature_bands": td.feature_bands,
        "n_classes": len(set(d["feature_labels"])), "dispatch_timings": dispatch_timings,
    }


# --- step 4: train + bundle (driver-side, identical to local: no runner involved) ---

def step_train(d, adapter, outdir: str) -> dict:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    X = d["features"].reshape(len(d["features"]), -1)
    y_raw = d["feature_labels"]
    import numpy as np
    keep = ~np.isnan(X).any(axis=1)
    X, y_raw = X[keep], np.asarray(y_raw)[keep]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42).fit(X, y)
    model_fp = os.path.join(outdir, "rf.joblib")
    joblib.dump((clf, le), model_fp)

    adapter.artifacts = {"model": model_fp}
    bundle_dir = bundle.save(adapter, {"model": model_fp}, os.path.join(outdir, "bundle"))
    ok(f"trained on {len(X)} samples, {len(le.classes_)} classes -> {bundle_dir}")
    return {"bundle_dir": bundle_dir, "classes": list(le.classes_)}


# --- step 5: run_inference (D8: one call, merge=True) -------------------------------

def step_inference(ml_client, root: str, bundle_dir: str, catalog_fp: str, outdir: str) -> dict:
    runner_kwargs = {"cluster": os.environ["AZ_CLUSTER"],
                     "environment": f"{os.environ['AZ_INFER_ENV_NAME']}:{os.environ['AZ_INFER_ENV_VERSION']}",
                     "root": root, "identity_client_id": os.environ["AZ_UAMI_CLIENT_ID"],
                     "ml_client": ml_client, "poll_interval_seconds": 10}

    roi_url = f"{root.rstrip('/')}/_inputs/AT_ROI.geojson"  # staged already in step 2

    before = _list_run_ids(root)
    result = rise_eo.run_inference(
        model=bundle_dir, output_folderpath=f"{root.rstrip('/')}/model_outputs",
        roi=roi_url, catalog_filepath=catalog_fp,
        startdate=START, enddate=END, mosaic_days=MOSAIC_DAYS, bands=BANDS,
        merge="reproject",
        storage="azure", runner="aml", runner_kwargs=runner_kwargs, overwrite=False,
    )
    dispatch_timings = _new_dispatch_timings(root, before)

    n = len(result.output_filepaths)
    if n < 1:
        fail("no per-cell outputs produced")
    # Land the small display artifacts (merged map + STAC) locally for step 6's plot;
    # the bulk COGs stay on blob (Land-local, CONTEXT.md -- never the raw imagery).
    local_merged = None
    if result.merged_filepath:
        local_merged = os.path.join(outdir, "crop_map_merged.tif")
        fs.transfer(result.merged_filepath, local_merged)
    ok(f"{n} per-cell COGs + STAC + merged map")
    return {"n_outputs": n, "merged_filepath": result.merged_filepath,
            "local_merged_filepath": local_merged,
            "stac_catalog_filepath": result.stac_catalog_filepath,
            "dispatch_timings": dispatch_timings}


# --- step 6: plots (D12: only the data figures; timing figures live in the plotter) --

def step_plots(d, local_merged_fp, classes) -> dict:
    import numpy as np

    n_figs = 0
    try:
        import matplotlib.pyplot as plt

        feats, labels = d["features"], np.asarray(d["feature_labels"])
        ts = d["metadata"]["timestamps"]
        fb = d["metadata"]["feature_bands"]
        ndvi = feats[:, :, fb.index("NDVI")]
        fig, ax = plt.subplots(figsize=(11, 6))
        for lab in sorted(set(labels)):
            med = np.nanmedian(ndvi[labels == lab], axis=0)
            ax.plot(ts, med, marker="o", markersize=3, linewidth=1.2, label=lab)
        ax.set_title("Per-class median NDVI over the season (AML training features)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        fig.autofmt_xdate()
        os.makedirs(FIGDIR, exist_ok=True)
        fig.savefig(os.path.join(FIGDIR, "ndvi_timeseries_aml.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)
        n_figs += 1

        if local_merged_fp and os.path.exists(local_merged_fp):
            import matplotlib.patches as mpatches
            import rasterio
            from matplotlib.colors import BoundaryNorm, ListedColormap

            with rasterio.open(local_merged_fp) as src:
                arr = src.read(1)
            arr = np.ma.masked_equal(arr, 255)
            values = list(range(len(classes)))
            cmap = ListedColormap(plt.cm.tab20(np.linspace(0, 1, max(len(values), 1))))
            norm = BoundaryNorm(np.array(values + [values[-1] + 1]) - 0.5, cmap.N)
            fig, ax = plt.subplots(figsize=(10, 9))
            ax.imshow(arr, cmap=cmap, norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Model output -- crop class map (AML, merged over ROI)")
            handles = [mpatches.Patch(color=cmap(i), label=classes[i]) for i in values]
            ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
            os.makedirs(FIGDIR, exist_ok=True)
            fig.savefig(os.path.join(FIGDIR, "crop_map_aml.png"), dpi=140, bbox_inches="tight")
            plt.close(fig)
            n_figs += 1
    except Exception as exc:  # noqa: BLE001 - plots are nice-to-have, never fail the demo run
        print(f"  ! plotting failed (non-fatal): {exc}", flush=True)
    return {"n_figures": n_figs}


# --- step 7: report -------------------------------------------------------------------

def step_report(demo: Demo) -> dict:
    """D9: the script emits data, not the report -- this step just confirms
    `timings.json` is self-contained and prints where it is."""
    with open(demo.timings_fp) as f:
        payload = json.load(f)
    print(f"  timings.json -> {demo.timings_fp}")
    print(f"  send this ONE file back (D9): {json.dumps({k: v for k, v in payload.items() if k != 'steps'}, indent=2)}")
    # +1: timings.json on disk is one step behind -- it is rewritten *after* each step
    # returns, so this step is not in the copy just read.
    return {"timings_fp": demo.timings_fp, "n_steps_recorded": len(payload["steps"]) + 1}


# --- D6: cost guard ------------------------------------------------------------------

def _dry_run_estimate():
    tiles = mpc.query_catalog(ROI_FP, START, END, max_cloudcover=MAX_CLOUDCOVER)
    grids = grid.roi_to_s2_grids(ROI_FP, grid_size_km=5, scale_fact=1.1)
    fields = gpd.read_file(TRAIN_FP)
    # Two figures, because one would be misleading. `APPROX_GB_PER_TILE` is a whole-tile
    # guard (all bands) -- ~3.4x high for this run's four. The expected value comes from
    # the local demo's own measurement of the SAME window and bands: 44.61 GB / 207
    # granules (E2E_AUSTRIA_AML.md §1). MPC's granule count will differ from CDSE's, so
    # this scales that per-granule rate by whatever MPC actually discovers.
    gb_per_granule_4band = 44.61 / 207
    print(json.dumps({
        "n_tiles": len(tiles), "n_cells": len(grids), "n_fields": len(fields),
        "max_tiles_guardrail": MAX_TILES,
        "estimated_gb": round(len(tiles) * gb_per_granule_4band, 1),
        "estimated_gb_upper_bound": round(len(tiles) * config.APPROX_GB_PER_TILE, 1),
        "note": "estimated_gb scales the local demo's measured 4-band rate "
                "(44.61 GB / 207 granules, same window+bands) by MPC's discovered count; "
                "the upper bound uses config.APPROX_GB_PER_TILE, which counts ALL bands. "
                "No wall-clock estimate: that needs a calibrated cost_model from a prior "
                "run's timings.json, and none exists yet (honest, not invented).",
    }, indent=2))


def _print_delete_command(prev_run_id: str, az_root: str):
    """D5: the script prints the delete, the operator runs it -- nothing here ever
    deletes a run prefix by itself. The prefix holds the run's inputs, cubes and outputs;
    the imagery lives in the shared archive (spec 59 D12), which this never touches.

    Account/filesystem/path are resolved from `AZ_ROOT` rather than emitted as
    `"$AZ_FS"`/`"$AZ_ACCOUNT"` shell references: those two are not part of §8.2's
    exported contract, so the quoted form pasted into a shell that never set them
    silently expands to empty. The directory is also `<AZ_ROOT's path>/demo_runs/<id>`,
    NOT a bare `demo_runs/<id>` -- `az storage fs directory delete -n` takes a path
    relative to the filesystem root, and AZ_ROOT carries its own prefix."""
    from rise_eo.storage.azure import account_from_url, to_vsi

    prev_root = f"{az_root.rstrip('/')}/demo_runs/{prev_run_id}"
    try:
        if not prev_root.startswith(("abfss://", "az://")):
            raise ValueError(prev_root)  # `to_vsi` passes non-blob paths through unchanged
        _, _, filesystem, path = to_vsi(prev_root).split("/", 3)
    except Exception:  # noqa: BLE001 - a non-blob root has no `az storage` command at all
        print(f"  ! previous run {prev_run_id!r} left data at {prev_root} -- delete it by hand "
              f"(fsd's own recursive delete is broken, TODO #50).")
        return
    account = account_from_url(prev_root)
    account_arg = f'--account-name "{account}"' if account else "--account-name <account>"
    print(f"  ! previous run {prev_run_id!r} left data on blob. To reclaim it (fsd's own "
          f"recursive delete is broken, TODO #50 -- the OPERATOR runs this, never this "
          f"script):\n"
          f"      az storage fs directory delete -f \"{filesystem}\" {account_arg} "
          f"-n \"{path}\" --auth-mode login -y")


def _last_run_marker_fp() -> str:
    """Which run last *spent* anything -- written when the download step starts, not
    when a run id is allocated. A run that dies in preflight has an empty prefix and
    must not overwrite the id of the run that actually holds the data; with fsd's
    recursive delete broken (TODO #50), a forgotten id is orphaned storage nobody can
    name."""
    return os.path.join(_LOCAL_OUTDIR_BASE, ".last_run_id")


def _allocate_run_id() -> str:
    return pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    run_group = ap.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--run-id", help="resume this demo run (completed steps skip, D5).")
    run_group.add_argument("--fresh", action="store_true",
                           help="allocate a NEW run-stamped prefix (D5: deletes nothing).")
    ap.add_argument("--dry-run", action="store_true",
                    help="print counts + GB estimate, zero side effects (D6), then exit.")
    ap.add_argument("--confirm-spend", action="store_true",
                    help="required to actually dispatch anything (D6).")
    args = ap.parse_args(argv)

    if args.dry_run:
        _dry_run_estimate()
        return

    if not args.confirm_spend:
        print("refusing to start without --confirm-spend (D6). Run --dry-run first.",
              file=sys.stderr)
        raise SystemExit(1)

    az_root = os.environ["AZ_ROOT"].rstrip("/")

    os.makedirs(_LOCAL_OUTDIR_BASE, exist_ok=True)
    marker = _last_run_marker_fp()
    if args.fresh:
        if os.path.exists(marker):
            with open(marker) as f:
                prev = f.read().strip()
            if prev:
                _print_delete_command(prev, az_root)
        run_id, resume = _allocate_run_id(), False
    else:
        run_id, resume = args.run_id, True

    demo = Demo(run_id, resume)
    root = f"{az_root}/demo_runs/{run_id}"
    _install_signal_handlers()

    print(f"run_id={run_id} root={root} outdir={demo.outdir}", flush=True)

    ml_client = _make_ml_client()

    try:
        demo.run_step("0_preflight", step_preflight, ml_client, root)

        demo.run_step("1_tiling", step_tiling, demo.outdir)

        # Claim the marker HERE, not at run-id allocation: from this step the run prefix
        # holds bytes (the uploaded ROI, then datacubes) worth a delete command. The
        # imagery itself goes to the shared archive, not under this prefix (spec 59 D12). A run that never got past preflight leaves the previous (spending)
        # run's id intact for the next `--fresh` to print.
        with open(marker, "w") as f:
            f.write(run_id)

        dl = demo.run_step("2_download", step_download, ml_client, root, az_root)
        # After the result is on disk (so a resume skips the download), before three more
        # dispatches spend ~20 min producing telemetry that cannot answer D11.
        _assert_dispatch_telemetry_complete(dl.get("dispatch_timings") or [],
                                            step="2_download")

        adapter = DemoRF()
        adapter.n_timestamps = rise_eo.compute_n_timestamps(START, END, MOSAIC_DAYS)
        local_export_fp = os.path.join(demo.outdir, "training_data")
        td_result = demo.run_step("3_training_data", step_training_data, ml_client, root,
                                  dl["catalog_fp"], adapter, local_export_fp)
        td = TrainingData(
            export_folderpath=td_result["export_folderpath"],
            run_folderpath=td_result["run_folderpath"], n_pixels=td_result["n_pixels"],
            n_timestamps=td_result["n_timestamps"], bands=td_result["bands"],
            feature_bands=td_result["feature_bands"],
        )
        d = td.load()

        train_result = demo.run_step("4_train_bundle", step_train, d, adapter, demo.outdir)

        infer_result = demo.run_step("5_run_inference", step_inference, ml_client, root,
                                     train_result["bundle_dir"], dl["catalog_fp"], demo.outdir)

        demo.run_step("6_plots", step_plots, d, infer_result["local_merged_filepath"],
                     train_result["classes"])

        demo.run_step("7_report", step_report, demo)

        log(f"DONE -- outputs under {demo.outdir}; send back timings.json (D9)")
    except DemoInterrupted:
        print(f"\ninterrupted -- {len(STEP_RESULTS)} step(s) recorded in timings.json. "
              f"Resume with --run-id {run_id}", file=sys.stderr)
        raise SystemExit(130)
    except PreflightFailure as exc:
        print(f"\npreflight/D14 failure: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
