"""Runner seam — execute the datacube task across many work-units.

Spec: specs/08-workflows.md, specs/10-storage-and-scale.md, specs/36-scale-runner.md.

Two backends: local (Snakemake), and Azure ML (`run_aml`) -- which shards `input_csv` and
dispatches each shard onto an AML cluster, where it calls back into this same module's
`run_local`. Same interface; runner is swappable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from importlib.resources import files

import pandas as pd

from rise_eo import config
from rise_eo import progress as _progress
from rise_eo import secrets as _secrets
from rise_eo.catalog import processing as _processing
from rise_eo.catalog.catalog import TileCatalog as _TileCatalog
from rise_eo.model import bundle as _bundle
from rise_eo.sources import mpc as _mpc
from rise_eo.sources.cdse import CdseCredentials as _CdseCredentials
from rise_eo.sources.cdse import query_catalog as _cdse_query_catalog
from rise_eo.storage import fs

_SNAKEFILE = "workflows/_snakefiles/create_datacube/Snakefile"
_INFER_SNAKEFILE = "workflows/_snakefiles/create_inference/Snakefile"
_INFER_ONLY_SNAKEFILE = "workflows/_snakefiles/infer_only/Snakefile"

# A unit's content identity. Kept in sync with `create_datacube._UNIT_IDENTITY_COLS` by
# hand, not imported from there: `create_datacube` already imports this module, so the
# import back would be circular (TODO #53).
_UNIT_IDENTITY_COLS = (
    "id", "startdate", "enddate", "bands", "mosaic_days", "mosaic_scheme", "collection",
    "properties_filter", "processing",
)


def _snakefile_path(rel: str = _SNAKEFILE) -> str:
    """Locate a bundled Snakefile (package-data) at runtime."""
    return str(files("rise_eo").joinpath(rel))


def _require_snakemake() -> None:
    """Fail with the install line rather than a subprocess `No module named snakemake`.

    Snakemake is only ever a subprocess, so a missing install surfaces as a returncode
    from a child process -- far from the call the user actually made.

    **This fires on AML nodes too, and that is not a contradiction.** `workflows/shard.py`
    and `workflows/infer_shard.py` are the in-job entrypoints, and both call straight back
    into `run_local`/`run_local_inference` -- an AML node runs the same Snakemake
    orchestration a laptop does. So a node image built without `[local]` fails HERE,
    ~30 min into a dispatch, which is exactly why the message names the image.
    """
    if importlib.util.find_spec("snakemake") is None:
        raise RuntimeError(
            "the Snakemake runner needs the optional '[local]' extra: "
            "pip install 'fsd[local]' (#80). "
            "On an AML node this means the node IMAGE was built without it -- add 'local' "
            "to ImageDefinition(extras=...); the in-job entrypoint runs the same runner."
        )


_STDERR_TAIL_CHARS = 4000


def _run(cmd: list[str]) -> tuple[int, str]:
    """Run `cmd` in its own process group, echo its stderr, and return (returncode, last 4000 stderr chars)."""
    process = subprocess.Popen(
        cmd, start_new_session=True, stderr=subprocess.PIPE, text=True, errors="replace"
    )
    tail: deque[str] = deque(maxlen=_STDERR_TAIL_CHARS)
    try:
        for line in process.stderr:
            sys.stderr.write(line)
            tail.append(line)
        process.communicate()
    except KeyboardInterrupt:
        print("\nInterrupt received, stopping Snakemake...")
        os.killpg(process.pid, signal.SIGINT)
        try:
            _, err = process.communicate(timeout=10)
            sys.stderr.write(err)
            tail.append(err)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
    return process.returncode, "".join(tail)[-_STDERR_TAIL_CHARS:]


def run_local(
    input_csv: str,
    *,
    cores: int,
    dry_run: bool = False,
    unlock: bool = False,
    njobs: int = 1,
    njobs_load_images: int = 1,
    jitter_span: int = 1,
) -> subprocess.CompletedProcess:
    """Local runner: drive the bundled Snakefile over `input_csv` rows.

    `cores` = Snakemake parallelism (how many datacubes build at once); `njobs` /
    `njobs_load_images` = intra-build parallelism passed to each task.
    """
    config = {
        "input_csv": input_csv,
        "njobs": njobs,
        "njobs_load_images": njobs_load_images,
        "jitter_span": jitter_span,
    }
    _require_snakemake()
    # Invoke via the running interpreter so it resolves regardless of PATH /
    # venv activation (and the task shells out with the same sys.executable).
    cmd = [
        sys.executable, "-m", "snakemake",
        "--snakefile", _snakefile_path(),
        "--cores", str(cores),
        "--config", *[f"{k}={v}" for k, v in config.items()],
    ]
    if dry_run:
        cmd.append("--dry-run")
    if unlock:
        cmd.append("--unlock")

    returncode, tail = _run(cmd)
    return subprocess.CompletedProcess(args=cmd, returncode=returncode, stderr=tail)


def run_local_inference(
    input_csv: str,
    *,
    cores: int,
    bundle_path: str,
    cubes_per_task: int = 1,
    predict_batch_size: int | None = None,
    skip_nan: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
    unlock: bool = False,
    njobs: int = 1,
    njobs_load_images: int = 1,
    jitter_span: int = 1,
) -> subprocess.CompletedProcess:
    """Local runner for ROI inference: drive the per-cell **build+infer** Snakefile
    over `input_csv` rows.

    Same seam as `run_local` — `cores` = how many groups run at once — but each job shells one
    `rise_eo.workflows.infer_task` group process (build each cell's datacube, then infer ->
    output.tif) instead of the build-only task. `bundle_path` is the model the workers reload.
    `cubes_per_task` groups K cells per job so the bundle loads once per group, not once per
    cell (TODO #25); default 1. `overwrite` forces a recompute of every cell (`--forceall`
    **and** `infer_task`'s own per-cell skip is bypassed, config `overwrite=1`); otherwise
    each cell's `output.tif` existence makes it resumable, decoupled from group size. Azure
    ML runs this same task on each node (`runner="aml"`, through `rise_eo.workflows.infer_shard`).

    Raises before dispatch if `input_csv` has two distinct-content rows sharing an
    `export_folderpath` -- a malformed manifest, the same exposure as
    `run_aml`/`run_aml_inference`.
    """
    with fs.open(input_csv, "r") as f:
        _dupe_errs = _duplicate_unit_errors(pd.read_csv(f).to_dict("records"))
    if _dupe_errs:
        raise ValueError("run_local_inference preflight failed:\n  - " + "\n  - ".join(_dupe_errs))

    conf = {
        "input_csv": input_csv,
        "bundle_path": bundle_path,
        "cubes_per_task": max(int(cubes_per_task), 1),
        "skip_nan": 1 if skip_nan else 0,
        "overwrite": 1 if overwrite else 0,
        "njobs": njobs,
        "njobs_load_images": njobs_load_images,
        "jitter_span": jitter_span,
    }
    if predict_batch_size is not None:  # snakemake parses an empty `key=` as None -> omit it
        conf["predict_batch_size"] = int(predict_batch_size)
    return _run_snakemake(_INFER_SNAKEFILE, cores, conf,
                          overwrite=overwrite, dry_run=dry_run, unlock=unlock)


def run_local_infer_only(
    input_csv: str,
    *,
    cores: int,
    bundle_path: str,
    cubes_per_task: int = 1,
    overwrite: bool = False,
    predict_batch_size: int | None = None,
    skip_nan: bool = True,
    dry_run: bool = False,
    unlock: bool = False,
) -> subprocess.CompletedProcess:
    """Local runner for **infer-only** fan-out over pre-built datacubes — the replacement
    for `engine.run_local`'s retired `mp.Pool`.

    `input_csv` has `datacube_filepath`, `output_filepath`. `cores` = how many groups run at once
    (Snakemake — the only parallel primitive); `cubes_per_task` groups K cubes per sequential job to
    amortise the one-per-job bundle load. `overwrite` forces recompute (`--forceall`); otherwise
    per-group sentinels + the task's skip-existing make it resumable. Local only: there is no
    AML dispatcher for pre-built cubes (`api.run_inference` keeps that path local).
    """
    conf = {
        "input_csv": input_csv,
        "bundle_path": bundle_path,
        "cubes_per_task": max(int(cubes_per_task), 1),
        "skip_nan": 1 if skip_nan else 0,
        "overwrite": 1 if overwrite else 0,
    }
    if predict_batch_size is not None:
        conf["predict_batch_size"] = int(predict_batch_size)
    return _run_snakemake(_INFER_ONLY_SNAKEFILE, cores, conf,
                          overwrite=overwrite, dry_run=dry_run, unlock=unlock)


def _run_snakemake(snakefile_rel, cores, conf, *, overwrite=False, dry_run=False, unlock=False):
    """Build + run a snakemake command over `conf` (shared by the inference runners)."""
    _require_snakemake()
    cmd = [
        sys.executable, "-m", "snakemake",
        "--snakefile", _snakefile_path(snakefile_rel),
        "--cores", str(cores),
        "--config", *[f"{k}={v}" for k, v in conf.items()],
    ]
    if overwrite:
        cmd.append("--forceall")
    if dry_run:
        cmd.append("--dry-run")
    if unlock:
        cmd.append("--unlock")
    returncode, tail = _run(cmd)
    return subprocess.CompletedProcess(args=cmd, returncode=returncode, stderr=tail)


# --- The Azure ML runner -----------------------------------------------------

_TERMINAL_JOB_STATUSES = {"Completed", "Failed", "Canceled"}


def _now_iso() -> str:
    """Driver-clock timestamp, ISO8601 UTC."""
    return pd.Timestamp.now(tz="UTC").isoformat()


def _seconds_between(start_iso: str | None, end_iso: str | None) -> float | None:
    """`end - start` in seconds between two ISO8601 timestamps; `None` if either is
    missing (e.g. a job whose `_status/<k>.json` never got written). Never floored at
    0: a negative result is the clock-skew bound being exceeded, not an
    error to hide."""
    if start_iso is None or end_iso is None:
        return None
    return (pd.Timestamp(end_iso) - pd.Timestamp(start_iso)).total_seconds()


def _derive_timing(
    *, run_id: str, t_start: str, t_first_submit: str | None = None,
    t_last_submit: str, t_end: str,
    submitted_at: dict, returned_at: dict, reports: dict, poll_interval_seconds: int,
) -> dict:
    """Per-job dispatch telemetry plus the additive wall-clock split, from the driver's own
    `submitted_at`/`returned_at` stamps and each job's in-job stamps (already read into
    `reports[k]` from `_status/<k>.json`).

    Pure, so it is unit-testable on hand-written stamps without a fake `ml_client`'s polling loop.

    The wall-clock split is five legs that telescope back to `t_end - t_start`:
      driver_prep         t_start          -> t_first_submit  (building jobs, before any submit)
      first_admission     t_first_submit   -> earliest process_start_at (time to the first node)
      execution_window    earliest process_start_at -> latest ended_at (the fleet finishing)
      teardown_detect     latest ended_at  -> latest returned_at (poll-quantized detection)
      post_collect        latest returned_at -> t_end (aggregating `_status/*.json`)

    ⚠️ **`first_admission` is measured from the FIRST submission, never the last.**
    Submitting N jobs is sequential and takes real time, while admission of the *early* jobs is
    already happening. Submission and admission overlap, so they cannot be adjacent legs: anchor
    on `t_last_submit` and the leg goes negative whenever a node starts before the final job is
    submitted. Those numbers still telescope correctly, so nothing catches it -- but neither leg
    means what its name says, and a negative here is RESERVED as the signal that the clock-skew
    bound was exceeded. The submission span is reported as `submission_span_seconds`,
    deliberately **outside** the additive split, because it overlaps `first_admission`.

    `t_first_submit` defaults to `t_last_submit`: a caller passing only `t_last_submit` gets
    the contiguous split rather than an error.
    """
    if t_first_submit is None:
        t_first_submit = t_last_submit
    jobs: dict = {}
    process_starts: list[str] = []
    ended_ats: list[str] = []
    for k in submitted_at:
        report = reports.get(k) or {}
        process_start_at = report.get("process_start_at")
        work_start_at = report.get("work_start_at")
        work_end_at = report.get("work_end_at")
        ended_at = report.get("ended_at")
        work_seconds = report.get("seconds")
        r_at = returned_at.get(k)

        total_job = _seconds_between(submitted_at[k], r_at)
        dispatch_overhead = (
            total_job - work_seconds
            if total_job is not None and work_seconds is not None else None
        )
        jobs[k] = {
            "submitted_at": submitted_at[k],
            "returned_at": r_at,
            "process_start_at": process_start_at,
            "work_start_at": work_start_at,
            "work_end_at": work_end_at,
            "ended_at": ended_at,
            "work_seconds": work_seconds,
            "job_admission_seconds": _seconds_between(submitted_at[k], process_start_at),
            "import_seconds": _seconds_between(process_start_at, work_start_at),
            "dispatch_overhead_seconds": dispatch_overhead,
        }
        if process_start_at is not None:
            process_starts.append(process_start_at)
        if ended_at is not None:
            ended_ats.append(ended_at)

    first_process_start = min(process_starts) if process_starts else None
    last_ended_at = max(ended_ats) if ended_ats else None
    last_returned_at = max(returned_at.values()) if returned_at else None

    return {
        "run_id": run_id,
        "poll_interval_seconds": poll_interval_seconds,
        "jobs": jobs,
        "wall": {
            "t_start": t_start,
            "t_first_submit": t_first_submit,
            "t_last_submit": t_last_submit,
            "t_end": t_end,
            "driver_prep_seconds": _seconds_between(t_start, t_first_submit),
            "first_admission_seconds": _seconds_between(t_first_submit, first_process_start),
            "execution_window_seconds": _seconds_between(first_process_start, last_ended_at),
            "teardown_detect_seconds": _seconds_between(last_ended_at, last_returned_at),
            "post_collect_seconds": _seconds_between(last_returned_at, t_end),
            # NOT one of the five legs: submitting overlaps the admission of jobs already
            # submitted, so it partitions nothing. Reported because it answers the obvious
            # follow-up to a large first_admission -- "how much of that was us still
            # submitting?"
            "submission_span_seconds": _seconds_between(t_first_submit, t_last_submit),
        },
    }


def _import_aml_command():
    """Lazy handle to `azure.ai.ml.command` -- imported inside a function, like every
    `azure.ai.ml` import in `fsd/` (here and in `model/verify_image.py`), so `import rise_eo`
    never needs the `[aml]` extra.

    Indirected through a helper so the AML job-builder sits on `run_aml`'s injection
    boundary: no test may require Azure, so tests substitute a fake here. Production
    behaviour is unchanged -- a real `runner="aml"` without the extra still raises
    ImportError at this line, exactly as a direct import would.
    """
    from azure.ai.ml import command

    return command


def _duplicate_export_folderpaths(rows: list[dict]) -> list[str]:
    """The `export_folderpath`s that two distinct-content rows collide on, or `[]`.

    `export_folderpath` is keyed by `id` ALONE (`create_datacube.setup`), a NARROWER key
    than the content-identity dedupe (`_UNIT_IDENTITY_COLS`) -- so two rows can pass that
    dedupe and still name the same folder, and nothing downstream can say which content the
    folder should hold (TODO #53).
    """
    seen: dict[str, set] = {}
    for r in rows:
        identity = tuple(str(r.get(c)) for c in _UNIT_IDENTITY_COLS)
        seen.setdefault(str(r.get("export_folderpath")), set()).add(identity)
    return sorted(p for p, ids in seen.items() if len(ids) > 1)


def _duplicate_unit_errors(rows: list[dict]) -> list[str]:
    """The aggregatable-error form of `_duplicate_export_folderpaths`, shared by every
    dispatcher (`run_aml`, `run_aml_inference`, `run_local_inference`) -- same exposure."""
    dupes = _duplicate_export_folderpaths(rows)
    if not dupes:
        return []
    shown = dupes[:5]
    more = f" (+{len(dupes) - 5} more)" if len(dupes) > 5 else ""
    return [
        "duplicate unit dispatch (D13): distinct-content rows share an export_folderpath "
        f"-- malformed manifest: {shown}{more}"
    ]


def shard_units(units: list, n_shards: int) -> list[list]:
    """Partition `units` into up to `n_shards` non-empty groups, round-robin.

    A true partition: every unit appears in exactly one shard. `n_shards > len(units)`
    degrades to `len(units)` shards -- an empty shard is never produced.
    """
    n_shards = max(int(n_shards), 1)
    n_groups = min(n_shards, len(units)) or 1
    groups: list[list] = [[] for _ in range(n_groups)]
    for i, unit in enumerate(units):
        groups[i % n_groups].append(unit)
    return [g for g in groups if g]


def _aml_preflight_common(ml_client, *, cluster: str, environment: str, root: str) -> list[str]:
    """Cluster/environment/storage-root checks shared by `run_aml`
    and `run_aml_download`. Returns error strings; never raises --
    each caller aggregates alongside its own source-specific checks."""
    errs = []
    try:
        compute = ml_client.compute.get(cluster)
        state = getattr(compute, "provisioning_state", None)
        if state not in (None, "Succeeded"):
            errs.append(f"cluster {cluster!r} not ready (provisioning_state={state!r}).")
    except Exception as exc:  # noqa: BLE001 - report, don't crash on a preflight check
        errs.append(f"cluster {cluster!r} not found or unreachable: {exc}")
    try:
        env_name, _, env_version = environment.partition(":")
        if env_version:
            ml_client.environments.get(name=env_name, version=env_version)
        else:
            ml_client.environments.get(name=env_name, label="latest")
    except Exception as exc:  # noqa: BLE001
        errs.append(f"environment {environment!r} does not resolve: {exc}")
    try:
        fs.makedirs(root)
        probe = f"{root.rstrip('/')}/.fsd_preflight_{uuid.uuid4().hex}"
        with fs.open(probe, "w") as f:
            f.write("preflight")
        fs.rm(probe)
    except Exception as exc:  # noqa: BLE001
        errs.append(f"storage root {root!r} not reachable/writable from the driver: {exc}")
    return errs


def _aml_preflight(ml_client, *, cluster: str, environment: str, root: str,
                    input_csv: str, n_shards: int | None) -> None:
    """Know before you spend: cheap checks that turn a 20-minutes-later cluster failure
    into an instant one."""
    errs = _aml_preflight_common(ml_client, cluster=cluster, environment=environment, root=root)
    if not fs.exists(input_csv):
        errs.append(f"input_csv does not exist: {input_csv!r}")
    else:
        with fs.open(input_csv, "r") as f:
            rows = pd.read_csv(f).to_dict("records")
        if len(rows) == 0:
            errs.append(f"input_csv is empty: {input_csv!r}")
        else:
            errs += _duplicate_unit_errors(rows)
    if n_shards is not None and int(n_shards) < 1:
        errs.append(f"n_shards must be >= 1, got {n_shards!r}.")
    if errs:
        raise ValueError("run_aml preflight failed:\n  - " + "\n  - ".join(errs))


def _aml_download_preflight(
    ml_client, *, cluster: str, environment: str, root: str, source: str,
    n_assets: int, vault_url: str | None, secret_name: str | None,
    get_secret, remaining_quota_gb: float | None, estimated_gb: float | None,
    creds_url: str | None = None,
    n_tiles: int | None = None,
    max_tiles: int | None = None,
) -> list[str]:
    """Know before you spend, for a download dispatch.

    Cluster/environment/root (shared, `_aml_preflight_common`) + discovery non-emptiness +
    the `max_tiles` guardrail + (CDSE-only) that the supplied creds source resolves/parses
    and its S3 keys are not expired. Raises on any hard failure; returns a possibly-empty
    list of non-fatal **warnings** -- just the CDSE quota estimate.

    CDSE creds come from **exactly one** of two mutually exclusive sources: Key Vault
    (`vault_url`+`secret_name`) or a blob JSON (`creds_url`). Neither or both is a hard
    preflight error. `source='mpc'` is anonymous and refuses all three (TODO #49).

    ⚠️ `max_tiles` is enforced **here, on the driver**, for BOTH sources, mirroring the
    guard the local paths apply (`sources/mpc.py`, `sources/cdse.py`). The runner must not
    change what a call means: enforce it on the node instead and MPC drops it entirely, so
    an `api.download(source='mpc', max_tiles=N)` that raises locally silently downloads
    everything on AML. Checking at dispatch also fails before a single node starts.
    """
    errs = _aml_preflight_common(ml_client, cluster=cluster, environment=environment, root=root)
    if n_assets < 1:
        errs.append("discovery matched 0 assets for this roi/date-window.")
    if n_tiles is not None and max_tiles is not None and n_tiles > max_tiles:
        errs.append(
            f"{n_tiles} matched tiles exceed max_tiles={max_tiles}. Narrow the query "
            "or raise max_tiles."
        )
    warnings: list[str] = []
    if source == "cdse":
        kv_given = bool(vault_url) or bool(secret_name)
        blob_given = bool(creds_url)
        if kv_given and blob_given:
            errs.append(
                "source='cdse' requires exactly one CDSE creds source, got both: "
                "vault_url/secret_name (Key Vault) and creds_url (blob JSON)."
            )
        elif not kv_given and not blob_given:
            errs.append(
                "source='cdse' requires exactly one CDSE creds source: "
                "vault_url+secret_name (Key Vault) or creds_url (blob JSON)."
            )
        elif kv_given and not (vault_url and secret_name):
            errs.append("source='cdse' requires both vault_url and secret_name (D5 Key Vault creds).")
        else:
            try:
                if blob_given:
                    creds = _CdseCredentials.from_json(creds_url)
                else:
                    creds = _CdseCredentials.from_json_str(get_secret(vault_url, secret_name))
                creds.require_s3()
                if creds.is_expired():
                    errs.append(
                        f"CDSE S3 keys expired (s3_keys_expire={creds.s3_keys_expire!r})."
                    )
            except Exception as exc:  # noqa: BLE001 - report, don't crash on a preflight check
                source_desc = f"blob {creds_url!r}" if blob_given else \
                    f"Key Vault secret {secret_name!r} at {vault_url!r}"
                errs.append(f"CDSE creds ({source_desc}) did not resolve/parse: {exc}")
        if remaining_quota_gb is not None and estimated_gb is not None and estimated_gb > remaining_quota_gb:
            warnings.append(
                f"estimated download (~{estimated_gb:.0f} GB) exceeds the ~{remaining_quota_gb:.0f} GB "
                "remaining CDSE 30-day quota -- expect throttling to 1 MB/s partway through "
                "(https://documentation.dataspace.copernicus.eu/Quotas.html)."
            )
    else:
        # MPC is anonymous: it reads no credentials at all, so a creds
        # argument here is not merely inert -- `creds_url` would put the secret on
        # blob for the whole run in exchange for nothing. Refuse it rather than
        # ignore it (TODO #49).
        supplied = [
            name for name, value in (
                ("creds_url", creds_url), ("vault_url", vault_url), ("secret_name", secret_name),
            ) if value
        ]
        if supplied:
            errs.append(
                f"source='mpc' is anonymous and reads no credentials, but {', '.join(supplied)} "
                "was supplied -- remove it. (Passing creds_url to an MPC run stages the secret on "
                "blob for the run's duration and never reads it.)"
            )
    if errs:
        raise ValueError("run_aml_download preflight failed:\n  - " + "\n  - ".join(errs))
    return warnings


def _aml_submit_and_wait(
    ml_client, jobs: dict, run_root: str, run_id: str, *, poll_interval_seconds: int = 30,
) -> dict:
    """Submit each prebuilt AML `command(...)` job in `jobs` (`{k: job}`), wait for
    all to reach a terminal status, aggregate `<run_root>/_status/<k>.json`, and
    raise on any failed/circuit-tripped job. Shared by `run_aml` (one job per datacube
    shard) and `run_aml_download` (one CDSE job, or N MPC shard jobs); the only difference
    between the two callers is how `jobs` gets built, not how submission, waiting and
    aggregation work.

    Also writes `<run_root>/_timing.json`: per-job `submitted_at` (as each
    `create_or_update` returns) and `returned_at` (the first poll at which that job is
    observed terminal -- so `teardown_detect` carries up to `poll_interval_seconds` of
    quantization error), plus the derived metrics from `_derive_timing`. Written **before**
    raising on failure, so a crashed dispatch still leaves every completed step's telemetry
    on disk. The return value is unchanged -- timing is a file, not a field.
    """
    # Name run_id + run_root BEFORE any job is submitted: it is the one line that answers
    # "is it stuck?" from outside the notebook, since `_status/*.json` and `_timing.json`
    # live under run_root.
    print(f"[aml] run_id={run_id} run_root={run_root}", flush=True)

    t_start = _now_iso()
    job_names: dict[int, str] = {}
    submitted_at: dict[int, str] = {}
    # Both ends of the submission loop: the early jobs are already being admitted while the
    # late ones are still going out, so `first_admission` anchors on the first (see
    # `_derive_timing`).
    t_first_submit = t_start
    t_last_submit = t_start
    for k, job in jobs.items():
        submitted = ml_client.jobs.create_or_update(job)
        t_last_submit = _now_iso()
        if not submitted_at:
            t_first_submit = t_last_submit
        submitted_at[k] = t_last_submit
        job_names[k] = submitted.name

    # Tick from the statuses dict this loop already maintains -- no extra AML calls. A
    # fan-out's per-second completion rate is not meaningful (show_rate=False), and a single
    # job has no rate to derive an ETA from, so it prints elapsed only rather than inventing
    # one (show_eta=False).
    n_jobs = len(job_names)
    tick = _progress.ticker(n_jobs, "aml", unit="jobs terminal", show_rate=False,
                            show_eta=(n_jobs > 1))
    statuses: dict[int, str] = {}
    returned_at: dict[int, str] = {}
    while True:
        for k, name in job_names.items():
            s = ml_client.jobs.get(name).status
            statuses[k] = s
            if s in _TERMINAL_JOB_STATUSES and k not in returned_at:
                returned_at[k] = _now_iso()
        n_terminal = sum(1 for s in statuses.values() if s in _TERMINAL_JOB_STATUSES)
        # One tick per poll, forced on the last so the 100% line always lands regardless of
        # the throttle. Force it as a SECOND call instead and the line prints twice whenever
        # every job is already terminal on the first poll.
        done = n_terminal == n_jobs
        tick(n_terminal, force=done, suffix=f"{n_jobs - n_terminal} running")
        if done:
            break
        time.sleep(poll_interval_seconds)

    failed = [k for k, s in statuses.items() if s != "Completed"]
    reports: dict[int, dict] = {}
    for k in job_names:
        status_url = f"{run_root}/_status/{k}.json"
        if fs.exists(status_url):
            with fs.open(status_url, "r") as f:
                report = json.load(f)
            reports[k] = report
            if (report.get("status") != "ok" or report.get("circuit_tripped")) and k not in failed:
                failed.append(k)
        else:
            reports[k] = {"unit": k, "aml_job_status": statuses[k]}
    failed = sorted(set(failed))

    t_end = _now_iso()
    timing = _derive_timing(
        run_id=run_id, t_start=t_start, t_first_submit=t_first_submit,
        t_last_submit=t_last_submit, t_end=t_end,
        submitted_at=submitted_at, returned_at=returned_at, reports=reports,
        poll_interval_seconds=poll_interval_seconds,
    )
    with fs.open(f"{run_root}/_timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    if failed:
        raise RuntimeError(f"job(s)/shard(s) failed: {failed} (run_id={run_id!r})")

    return {"run_id": run_id, "job_statuses": statuses, "reports": reports}


def run_aml(
    input_csv: str,
    *,
    cluster: str,
    environment: str,
    root: str,
    identity_client_id: str,
    run_id: str | None = None,
    n_shards: int | None = None,
    cores: int = 16,
    ml_client=None,
    subscription_id: str | None = None,
    resource_group_name: str | None = None,
    workspace_name: str | None = None,
    poll_interval_seconds: int = 30,
) -> dict:
    """AML runner: shard `input_csv`, submit one command job per shard onto `cluster`, wait,
    aggregate `_status/<k>.json`, raise on any failure.

    Each dispatched unit is a **shard**, not a cube: the job runs
    `python -m rise_eo.workflows.shard <shard_csv_url> --cores <cores>`, which calls back
    into this module's `run_local` -- the same Snakemake orchestration a laptop runs.
    No AML-specific pipeline code exists; only this dispatcher knows about AML.

    `identity_client_id` is set as the job's `AZURE_CLIENT_ID` env var -- the AML
    cluster carries only a user-assigned managed identity, which is never selected
    implicitly, so `fsd/storage/azure.py`'s bare `DefaultAzureCredential()` needs this
    to authenticate on the node. **fsd never hardcodes it** (a concrete `rise` identity
    id has no business in a public repo) -- the caller resolves it (e.g. via
    `az identity show --query clientId`) and passes it in.

    `ml_client` is the test/injection seam: pass a fake with
    `.compute.get`, `.environments.get`, `.jobs.create_or_update`, `.jobs.get` to avoid
    any network call; when omitted, a real `azure.ai.ml.MLClient` is constructed here
    (lazy import, so `import rise_eo` never needs the `[aml]` extra).

    `root` is the storage root (any `rise_eo.storage` URL, typically `abfss://...`) under
    which `runs/<run_id>/{shards,_status}/...` is laid out. `n_shards` defaults to
    the cluster's `max_instances`.
    """
    if ml_client is None:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group_name, workspace_name
        )

    _aml_preflight(ml_client, cluster=cluster, environment=environment, root=root,
                    input_csv=input_csv, n_shards=n_shards)

    if n_shards is None:
        compute = ml_client.compute.get(cluster)
        n_shards = getattr(compute, "max_instances", None) or 1

    run_id = run_id or pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    run_root = f"{root.rstrip('/')}/runs/{run_id}"

    with fs.open(input_csv, "r") as f:
        units = pd.read_csv(f).to_dict("records")
    shards = shard_units(units, n_shards)

    aml_command = _import_aml_command()

    jobs: dict[int, object] = {}
    for k, rows in enumerate(shards):
        shard_url = f"{run_root}/shards/{k}.csv"
        with fs.open(shard_url, "w") as f:
            pd.DataFrame(rows).to_csv(f, index=False)

        jobs[k] = aml_command(
            command=f"python -m rise_eo.workflows.shard {shard_url} --cores {cores}",
            environment=environment,
            compute=cluster,
            environment_variables={"AZURE_CLIENT_ID": identity_client_id},
            display_name=f"fsd-shard-{run_id}-{k}",
            experiment_name=f"fsd-{run_id}",
        )

    result = _aml_submit_and_wait(ml_client, jobs, run_root, run_id,
                                   poll_interval_seconds=poll_interval_seconds)
    return {"run_id": run_id, "n_shards": len(shards),
            "job_statuses": result["job_statuses"], "shards": result["reports"]}


# --- The Azure ML flatten reduce dispatcher -----------------------------------


def _aml_flatten_preflight(
    ml_client, *, cluster: str, environment: str, root: str,
    input_csv: str, id_col: str, label_col: str | None,
) -> None:
    """Cluster/environment/root (shared, `_aml_preflight_common`) + `input_csv` non-empty and
    carrying `id_col` (+ `label_col` iff requested).

    Deliberately no `_duplicate_unit_errors` check: unlike a build/inference fan-out, a
    flatten reduce reads every row into ONE job, so a duplicate `id` is a labeling question
    for the caller, not a dispatch hazard."""
    errs = _aml_preflight_common(ml_client, cluster=cluster, environment=environment, root=root)
    if not fs.exists(input_csv):
        errs.append(f"input_csv does not exist: {input_csv!r}")
    else:
        with fs.open(input_csv, "r") as f:
            df = pd.read_csv(f)
        if len(df) == 0:
            errs.append(f"input_csv is empty: {input_csv!r}")
        else:
            if id_col not in df.columns:
                errs.append(f"input_csv missing id_col {id_col!r}.")
            if label_col is not None and label_col not in df.columns:
                errs.append(f"input_csv missing label_col {label_col!r}.")
    if errs:
        raise ValueError("run_aml_flatten preflight failed:\n  - " + "\n  - ".join(errs))


def run_aml_flatten(
    input_csv: str,
    export_folderpath: str,
    *,
    id_col: str = "id",
    label_col: str | None = None,
    filepath_col: str = "datacube_filepath",
    nodata: int = config.NODATA,
    cluster: str,
    environment: str,
    root: str,
    identity_client_id: str,
    run_id: str | None = None,
    ml_client=None,
    subscription_id: str | None = None,
    resource_group_name: str | None = None,
    workspace_name: str | None = None,
    poll_interval_seconds: int = 30,
) -> dict:
    """AML flatten dispatcher: flatten concatenates ALL cubes into ONE array, so
    the cluster form is exactly **one** command job (`python -m rise_eo.workflows.flatten ...`) --
    no `shard_units`, no fan-out. Submits via the shared `_aml_submit_and_wait` and
    reuses `_aml_preflight_common`. Runs on the **general-purpose** fsd Environment --
    flatten is pure `fsd`, no adapter, so it needs no new image.

    `input_csv` is a blob url of `id`/[`label`]/`datacube_filepath` rows.
    `export_folderpath` is the **blob** prefix the reduce writes its raw output to
    (`data.npy`/`coords.npy`/`ids.npy`/`metadata.pickle.npy`/`labels.npy?`) -- landing it
    locally is `api._land_local`'s job, not this dispatcher's.

    `identity_client_id`/`ml_client`/`root` behave exactly as in `run_aml` (see its
    docstring).
    """
    if ml_client is None:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group_name, workspace_name
        )

    _aml_flatten_preflight(ml_client, cluster=cluster, environment=environment, root=root,
                           input_csv=input_csv, id_col=id_col, label_col=label_col)

    run_id = run_id or pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    run_root = f"{root.rstrip('/')}/runs/{run_id}"

    aml_command = _import_aml_command()

    command = (
        f"python -m rise_eo.workflows.flatten --input-csv {input_csv} "
        f"--filepath-col {filepath_col} --id-col {id_col} --export {export_folderpath} "
        f"--nodata {nodata} --status-url {run_root}/_status/0.json"
    )
    if label_col is not None:
        command += f" --label-col {label_col}"

    jobs = {0: aml_command(
        command=command, environment=environment, compute=cluster,
        environment_variables={"AZURE_CLIENT_ID": identity_client_id},
        display_name=f"fsd-flatten-{run_id}", experiment_name=f"fsd-flatten-{run_id}",
    )}

    result = _aml_submit_and_wait(ml_client, jobs, run_root, run_id,
                                   poll_interval_seconds=poll_interval_seconds)
    return {"run_id": run_id, "n_jobs": len(jobs),
            "job_statuses": result["job_statuses"], "reports": result["reports"]}


# --- The Azure ML inference dispatcher ----------------------------------------


def _stage_bundle(bundle_path: str, dst_url: str) -> str:
    """Stage a bundle (local, or on any `rise_eo.storage` backend) to `dst_url`; returns `dst_url`.

    Copies `bundle.json` plus every file its `artifacts` map and `code` block name. No
    directory listing and no new storage primitive: the manifest already enumerates every
    file the bundle needs, with relative hrefs and no absolute path baked in, so this is the
    same manifest-driven shape the node uses to fetch it back down
    (`infer_shard.fetch_bundle_to_scratch`).
    """
    manifest = _bundle.read_spec(bundle_path)
    with fs.open(os.path.join(bundle_path, _bundle.BUNDLE_MANIFEST), "r") as f:
        raw = f.read()
    with fs.open(os.path.join(dst_url, _bundle.BUNDLE_MANIFEST), "w") as f:
        f.write(raw)
    rels = list(manifest.get("artifacts", {}).values()) + _bundle.manifest_code_files(manifest)
    # Destination + total size print BEFORE the upload starts: this leg can be slow over VPN,
    # and it is the one silence costs most on.
    total_bytes = sum(fs.size(os.path.join(bundle_path, rel)) for rel in rels)
    print(f"[stage] bundle -> {dst_url} | {len(rels)} files, {total_bytes / 1e6:.1f} MB",
          flush=True)
    tick = _progress.ticker(len(rels), "stage", unit="files")
    tick(0, force=True)
    for i, rel in enumerate(rels, 1):
        fs.transfer(os.path.join(bundle_path, rel), os.path.join(dst_url, rel))
        tick(i)
    tick(len(rels), force=True)
    return dst_url


def _aml_inference_preflight(
    ml_client, *, cluster: str, environment: str, root: str,
    input_csv: str, n_shards: int | None, max_cells: int | None,
) -> None:
    """Every check that CAN run on the driver MUST run here, before any AML job is submitted
    -- node cold-start is 40-380 s (TODO #48).

    Covers cluster/environment/root (shared, `_aml_preflight_common`), input_csv
    non-emptiness, the duplicate-unit guard, and the `max_cells` guardrail: refuse an ROI
    that splits into more grid cells than intended, before dispatching thousands of jobs.
    Model-spec checks (bands/T) are hoisted into `api._run_inference_roi`'s own preflight,
    ahead of this call, rather than duplicated here.
    """
    errs = _aml_preflight_common(ml_client, cluster=cluster, environment=environment, root=root)
    if not fs.exists(input_csv):
        errs.append(f"input_csv does not exist: {input_csv!r}")
    else:
        with fs.open(input_csv, "r") as f:
            rows = pd.read_csv(f).to_dict("records")
        if len(rows) == 0:
            errs.append(f"input_csv is empty: {input_csv!r}")
        else:
            errs += _duplicate_unit_errors(rows)
            if max_cells is not None and len(rows) > max_cells:
                errs.append(
                    f"{len(rows)} cells exceed max_cells={max_cells}. Narrow the ROI or "
                    "raise max_cells."
                )
    if n_shards is not None and int(n_shards) < 1:
        errs.append(f"n_shards must be >= 1, got {n_shards!r}.")
    if errs:
        raise ValueError("run_aml_inference preflight failed:\n  - " + "\n  - ".join(errs))


def run_aml_inference(
    input_csv: str,
    bundle_path: str,
    *,
    cluster: str,
    environment: str,
    root: str,
    identity_client_id: str,
    run_id: str | None = None,
    n_shards: int | None = None,
    cores: int | None = None,
    cubes_per_task: int | None = None,
    predict_batch_size: int | None = None,
    skip_nan: bool = True,
    overwrite: bool = False,
    max_cells: int | None = None,
    skip_smoke: bool = False,
    ml_client=None,
    subscription_id: str | None = None,
    resource_group_name: str | None = None,
    workspace_name: str | None = None,
    poll_interval_seconds: int = 30,
) -> dict:
    """AML inference dispatcher: the thin runner swap under the per-cell build+infer unit.

    Receives the already-produced `input_csv` (tiling and `setup` ran on the driver, in
    `api._run_inference_roi`) plus `bundle_path`, and does only: stage the bundle to blob ->
    shard the cells (reusing `shard_units`) -> submit one job per shard running `python -m
    rise_eo.workflows.infer_shard` -> wait -> aggregate `_status/<k>.json` -> raise on any
    failure. Mirrors `run_aml`; the only differences are what each node runs and that a
    bundle is staged first.

    `identity_client_id`/`ml_client`/`root` behave exactly as in `run_aml` (see its
    docstring) -- inference changes neither the storage seam nor the identity mechanism.

    `skip_smoke=False` (default) runs a one-node adapter-import smoke BEFORE the N-node
    fan-out. It is the only preflight check that needs a real node, because the driver's venv
    is not guaranteed to mirror the inference Environment. Pass `True` once an Environment is
    proven, to skip the extra spin-up on repeat runs.

    `cores`/`cubes_per_task` default to `None` = **load-per-core**: the flag is left off the
    node command so `infer_shard` computes it from the node's own `os.cpu_count()` and the
    shard size -- the bundle loads once per core rather than once per cell (TODO #25), and
    the node stays fully busy. Pass `cores=1` for the heavy-model **load-once-per-node**
    opt-out (one whole-shard group, one bundle load); pass explicit values to override.
    """
    if ml_client is None:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group_name, workspace_name
        )

    _aml_inference_preflight(ml_client, cluster=cluster, environment=environment, root=root,
                             input_csv=input_csv, n_shards=n_shards, max_cells=max_cells)

    if n_shards is None:
        compute = ml_client.compute.get(cluster)
        n_shards = getattr(compute, "max_instances", None) or 1

    run_id = run_id or pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    run_root = f"{root.rstrip('/')}/runs/{run_id}"

    bundle_url = _stage_bundle(bundle_path, f"{run_root}/_bundle")

    aml_command = _import_aml_command()

    if not skip_smoke:
        smoke_status_url = f"{run_root}/_status/smoke.json"
        smoke_job = aml_command(
            command=(
                f"python -m rise_eo.workflows.adapter_smoke {bundle_url} "
                f"--status-url {smoke_status_url}"
            ),
            environment=environment, compute=cluster,
            environment_variables={"AZURE_CLIENT_ID": identity_client_id},
            display_name=f"fsd-infer-smoke-{run_id}", experiment_name=f"fsd-infer-{run_id}",
        )
        _aml_submit_and_wait(ml_client, {"smoke": smoke_job}, run_root, f"{run_id}-smoke",
                             poll_interval_seconds=poll_interval_seconds)

    with fs.open(input_csv, "r") as f:
        units = pd.read_csv(f).to_dict("records")
    shards = shard_units(units, n_shards)

    jobs: dict[int, object] = {}
    for k, rows in enumerate(shards):
        shard_url = f"{run_root}/shards/{k}.csv"
        with fs.open(shard_url, "w") as f:
            pd.DataFrame(rows).to_csv(f, index=False)

        # `cores`/`cubes_per_task` left off the command when None: the NODE then computes
        # the load-per-core default from `os.cpu_count()` + the shard size (`infer_shard`).
        cmd = f"python -m rise_eo.workflows.infer_shard {shard_url} {bundle_url}"
        if cores is not None:
            cmd += f" --cores {cores}"
        if cubes_per_task is not None:
            cmd += f" --cubes-per-task {cubes_per_task}"
        if predict_batch_size is not None:
            cmd += f" --predict-batch-size {predict_batch_size}"
        if not skip_nan:
            cmd += " --no-skip-nan"
        if overwrite:
            cmd += " --overwrite"

        jobs[k] = aml_command(
            command=cmd,
            environment=environment,
            compute=cluster,
            environment_variables={"AZURE_CLIENT_ID": identity_client_id},
            display_name=f"fsd-infer-{run_id}-{k}",
            experiment_name=f"fsd-infer-{run_id}",
        )

    result = _aml_submit_and_wait(ml_client, jobs, run_root, run_id,
                                   poll_interval_seconds=poll_interval_seconds)
    return {"run_id": run_id, "n_shards": len(shards),
            "job_statuses": result["job_statuses"], "shards": result["reports"]}


# --- The Azure ML download dispatcher ----------------------------------------


def _import_command_job_limits():
    """Lazy handle to `azure.ai.ml.entities.CommandJobLimits` -- same injection-boundary
    pattern as `_import_aml_command`, so tests substitute a fake and never require the
    `[aml]` extra."""
    from azure.ai.ml.entities import CommandJobLimits

    return CommandJobLimits


def _estimate_timeout_seconds(
    estimated_gb: float, *, conservative_mb_per_s: float = 10.0, floor_seconds: int = 1800,
) -> int:
    """Size an explicit job timeout from a GB estimate at a conservative throughput.

    Well under CDSE's 4x20 MB/s ceiling and MPC's blob throughput, so a healthy transfer
    never trips it, with a floor so a tiny or empty estimate still gets a sane timeout.
    """
    return max(int(estimated_gb * 1024 / conservative_mb_per_s), floor_seconds)


def _iso(dt) -> str:
    return pd.Timestamp(dt).isoformat()


def _mpc_catalog_shortfall(catalog_filepath: str, rows: list[dict]) -> list[dict]:
    """Which of `rows` (`_mpc.discover_shard_rows`'s shape -- one row per `(tile_id, band)`
    asset) the existing catalog does NOT already cover (#64).

    "Already present" means the catalog carries a row for `tile_id` whose `files` column
    covers `band`. That is a catalog READ, never a destination `fs.exists`/`fs.stat`: one WAN
    listing per asset would approach the very cold-start cost this diff exists to avoid.

    ⚠️ It therefore rests on the catalog's invariant that a row exists only if its file does
    -- an invariant an interrupted MPC transfer still violates by leaving a truncated file
    under the final name (#74). An absent catalog means nothing is present, so everything is
    dispatched.
    """
    if not fs.exists(catalog_filepath):
        return rows
    catalog_df = _TileCatalog(catalog_filepath).read()
    have: dict[str, set[str]] = {}
    for _, r in catalog_df.iterrows():
        files = str(r.get("files") or "")
        bands_present = {os.path.splitext(f)[0] for f in files.split(",") if f}
        have.setdefault(str(r["id"]), set()).update(bands_present)
    return [r for r in rows if r["band"] not in have.get(str(r["tile_id"]), set())]


def run_aml_download(
    roi: str,
    startdate,
    enddate,
    bands: list[str],
    dst_folderpath: str,
    catalog_filepath: str,
    *,
    source: str,
    cluster: str,
    environment: str,
    root: str,
    identity_client_id: str,
    max_tiles: int,
    collection: str = config.SATELLITE_S2L2A,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str = "latest",
    vault_url: str | None = None,
    secret_name: str | None = None,
    creds_url: str | None = None,
    max_cloudcover: float | None = None,
    cog: bool = True,
    n_shards: int | None = None,
    remaining_quota_gb: float | None = None,
    timeout_seconds: int | None = None,
    run_id: str | None = None,
    ml_client=None,
    subscription_id: str | None = None,
    resource_group_name: str | None = None,
    workspace_name: str | None = None,
    poll_interval_seconds: int = 30,
    get_secret=None,
) -> dict:
    """AML download dispatcher. Dispatch shape depends on the source: CDSE submits
    **exactly one** whole-ROI job; MPC discovers on the driver, `shard_units` the asset
    list, and submits **N** per-shard jobs. Both wait, aggregate `_status/<k>.json`, and
    raise on any failed/circuit-tripped job (`_aml_submit_and_wait`, shared with `run_aml`).

    `roi` must be a url (any `rise_eo.storage`/geopandas-readable path) rather than an
    in-memory GeoDataFrame -- the job that reads it runs on a different machine.

    `vault_url`/`secret_name` (CDSE only) are Key Vault coordinates and `creds_url` (CDSE
    only) is a blob JSON location -- **exactly one** of the two is required; preflight errs
    on neither and on both. Both are caller-supplied; fsd hardcodes no concrete identifier.

    ⚠️ Secrets never ride in the job spec. Only these non-secret names and locations go into
    the command args, and the node reads the value itself at run time -- via
    `rise_eo.secrets.get_secret` (Key Vault, substitutable here through `get_secret`) or
    `rise_eo.storage.fs.open` (blob, via `cdse.CdseCredentials.from_json`). The same
    `identity_client_id` that authorises blob also authorises Key Vault.

    `ml_client` is the test/injection seam (mirrors `run_aml`): pass a fake with
    `.compute.get`, `.environments.get`, `.jobs.create_or_update`, `.jobs.get` to avoid any
    network call.

    **MPC only** (#64): discovery runs on the driver here, so before any preflight or
    dispatch the discovered `(tile_id, band)` assets are diffed against `catalog_filepath`
    (`_mpc_catalog_shortfall` -- a catalog READ, not a per-asset destination stat). A
    shortfall of zero returns without calling `ml_client.jobs.create_or_update` at all; a
    partial shortfall shards and dispatches only the missing assets. **CDSE is out**: it
    submits one whole-ROI job and discovers on the node inside it, so the same diff would
    need a new driver-side CDSE discovery pass.
    """
    if source not in ("cdse", "mpc"):
        raise ValueError(f"source={source!r} must be one of 'cdse', 'mpc'.")

    get_secret = get_secret or _secrets.get_secret

    if ml_client is None:
        from azure.ai.ml import MLClient
        from azure.identity import DefaultAzureCredential

        ml_client = MLClient(
            DefaultAzureCredential(), subscription_id, resource_group_name, workspace_name
        )

    run_id = run_id or pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    run_root = f"{root.rstrip('/')}/runs/{run_id}"
    aml_command = _import_aml_command()
    limits_cls = _import_command_job_limits()

    if source == "cdse":
        tiles = _cdse_query_catalog(roi, startdate, enddate, max_cloudcover=max_cloudcover,
                                    processing=processing)
        touched_keys = set(tiles["acquisition_key"])
        n_assets = len(tiles)
        estimated_gb = len(tiles) * config.APPROX_GB_PER_TILE

        _aml_download_preflight(
            ml_client, cluster=cluster, environment=environment, root=root,
            source="cdse", n_assets=n_assets, vault_url=vault_url, secret_name=secret_name,
            get_secret=get_secret, remaining_quota_gb=remaining_quota_gb, estimated_gb=estimated_gb,
            creds_url=creds_url, n_tiles=len(tiles), max_tiles=max_tiles,
        )

        timeout = timeout_seconds or _estimate_timeout_seconds(estimated_gb)
        creds_arg = f"--creds-url {creds_url}" if creds_url else \
            f"--vault-url {vault_url} --secret-name {secret_name}"
        command = (
            f"python -m rise_eo.workflows.download --roi {roi} "
            f"--startdate {_iso(startdate)} --enddate {_iso(enddate)} "
            f"--bands {','.join(bands)} --dst {dst_folderpath} --catalog {catalog_filepath} "
            f"--max-tiles {max_tiles} {creds_arg} "
            # The specifier rides the command line as a plain string, quoted because
            # `>=05.00` is otherwise a shell redirect.
            f"--processing {shlex.quote(processing)} "
            f"--status-url {run_root}/_status/0.json"
        )
        if max_cloudcover is not None:
            command += f" --max-cloudcover {max_cloudcover}"
        if not cog:
            command += " --no-cog"

        jobs = {0: aml_command(
            command=command, environment=environment, compute=cluster,
            environment_variables={"AZURE_CLIENT_ID": identity_client_id},
            display_name=f"fsd-download-cdse-{run_id}", experiment_name=f"fsd-download-{run_id}",
            limits=limits_cls(timeout=timeout),
        )}
    else:
        rows = _mpc.discover_shard_rows(
            roi, startdate, enddate, bands, dst_folderpath, max_cloudcover=max_cloudcover,
            collection=collection, properties_filter=properties_filter,
            processing=processing,
        )
        touched_keys = {r["acquisition_key"] for r in rows}
        n_discovered = len(rows)

        # Diff against the existing catalog BEFORE any preflight or dispatch (#64). A
        # request whose every discovered asset is already catalogued must not submit a
        # single job (node cold start is minutes), and a 95%-present request must dispatch
        # the shortfall, not 100%.
        rows = _mpc_catalog_shortfall(catalog_filepath, rows)
        n_assets = len(rows)
        if n_assets == 0:
            print(f"[download] 0 of {n_discovered} assets missing; nothing to download",
                  flush=True)
            return {"run_id": run_id, "source": "mpc", "n_jobs": 0,
                    "job_statuses": {}, "reports": {}}
        if n_assets < n_discovered:
            print(f"[download] {n_assets} of {n_discovered} assets missing; "
                  f"dispatching {n_assets}", flush=True)

        _aml_download_preflight(
            ml_client, cluster=cluster, environment=environment, root=root,
            source="mpc", n_assets=n_assets, vault_url=vault_url, secret_name=secret_name,
            get_secret=get_secret, remaining_quota_gb=None, estimated_gb=None,
            creds_url=creds_url,
            # one row per asset -> collapse to distinct MGRS tiles, the unit
            # `max_tiles` counts in `sources/mpc.download`'s own guard.
            n_tiles=len({r["tile_id"] for r in rows}), max_tiles=max_tiles,
        )

        if n_shards is None:
            compute = ml_client.compute.get(cluster)
            n_shards = getattr(compute, "max_instances", None) or 1
        shards = shard_units(rows, n_shards)
        timeout = timeout_seconds or _estimate_timeout_seconds(
            n_assets * config.APPROX_GB_PER_TILE / max(len(bands), 1)
        )

        # MPC only: each shard writes its OWN catalog file -- single writer, no lock --
        # instead of N shards racing an unsynchronised read-whole-parquet -> concat ->
        # write-whole-parquet against the SAME `catalog_filepath`. `TileCatalog.append` is
        # last-writer-wins on blob, so that race is a SILENT lost update that under-declares
        # the archive. The driver merges the shard catalogs sequentially below, after every
        # shard has finished. CDSE runs as one job writing the canonical catalog directly,
        # so the race cannot occur there (TODO #51).
        shard_catalog_urls: dict[int, str] = {}
        jobs = {}
        for k, shard_rows in enumerate(shards):
            shard_url = f"{run_root}/shards/{k}.csv"
            with fs.open(shard_url, "w") as f:
                pd.DataFrame(shard_rows).to_csv(f, index=False)

            shard_catalog_url = f"{run_root}/shards/catalog-{k}.parquet"
            shard_catalog_urls[k] = shard_catalog_url
            jobs[k] = aml_command(
                command=(
                    f"python -m rise_eo.workflows.download --shard {shard_url} "
                    f"--dst {dst_folderpath} --catalog {shard_catalog_url} "
                    f"--status-url {run_root}/_status/{k}.json"
                ),
                environment=environment, compute=cluster,
                environment_variables={"AZURE_CLIENT_ID": identity_client_id},
                display_name=f"fsd-download-mpc-{run_id}-{k}",
                experiment_name=f"fsd-download-{run_id}",
                limits=limits_cls(timeout=timeout),
            )

    result = _aml_submit_and_wait(ml_client, jobs, run_root, run_id,
                                   poll_interval_seconds=poll_interval_seconds)

    if source == "mpc":
        _merge_shard_catalogs(shard_catalog_urls, catalog_filepath)

    # The driver knows which acquisitions this download touched, so it -- not a node --
    # reports whether any now holds more than one processing in the archive.
    if fs.exists(catalog_filepath):
        for line in _processing.ambiguity_lines(
            _TileCatalog(catalog_filepath).read(), touched_keys, n_matched=len(touched_keys),
            processing=processing, source=source,
        ):
            print(line, flush=True)

    return {"run_id": run_id, "source": source, "n_jobs": len(jobs),
            "job_statuses": result["job_statuses"], "reports": result["reports"]}


def _merge_shard_catalogs(shard_catalog_urls: dict[int, str], canonical_filepath: str) -> None:
    """Sequentially `TileCatalog.append` each MPC shard's catalog into the canonical one.

    A deliberate single-writer SERIALIZATION, in shard order: no lock and no ETag/lease
    (those go badly on `abfss://` -- TODO #50), run once after every shard has already
    finished, so there is nothing to race. A shard that produced no assets writes no catalog
    file at all and is skipped, not an error. TODO #51.
    """
    canonical = _TileCatalog(canonical_filepath)
    for k in sorted(shard_catalog_urls):
        shard_url = shard_catalog_urls[k]
        if not fs.exists(shard_url):
            continue
        shard_catalog = _TileCatalog(shard_url)
        rows = shard_catalog.read().to_dict("records")
        canonical.append(rows, declaration=shard_catalog.declaration)
