"""In-job entrypoint for the AML inference dispatcher.

A thin shim, mirroring `rise_eo.workflows.shard`'s role for the build fan-out: `fs.get` the shard CSV and
the staged bundle to node-local scratch, then call the **existing** `runners.run_local_inference`
over them. `rise_eo.workflows.infer_task` (the unit of work) and `runners.run_local_inference` are
unchanged; this file adds no pipeline logic of its own -- it exists only so the same local
orchestration can be invoked from an AML node instead of a laptop.

The bundle fetch is **manifest-driven** (`fs.open` the staged `bundle.json`, then `fs.get`
each file its `artifacts` map names). `rise_eo.storage.get` is single-file (`fs.get_file`), not
recursive, so this is a fetch that needs no directory listing -- keep it that way, or the node
starts paying a WAN listing per bundle. `bundle.load` itself is untouched: it only ever sees a
local directory.

Run as: python -m rise_eo.workflows.infer_shard <shard_csv_url> <bundle_url> --cores N
"""

from __future__ import annotations

import datetime as _dt

# Stamped before any heavy import, so it is process start (see workflows/shard.py).
_PROCESS_START_AT = _dt.datetime.now(_dt.timezone.utc).isoformat()

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402

from rise_eo.model import bundle as _bundle  # noqa: E402
from rise_eo.storage import fs  # noqa: E402
from rise_eo.workflows import runners  # noqa: E402

EXPORT_FOLDERPATH_COL = "export_folderpath"


def _status_url(shard_csv_url: str) -> str:
    """`<root>/runs/<run_id>/shards/<k>.csv` -> `<root>/runs/<run_id>/_status/<k>.json`
    (mirrors `workflows.shard._status_url`)."""
    root, name = shard_csv_url.rsplit("/shards/", 1)
    stem = name[:-4] if name.endswith(".csv") else name
    return f"{root}/_status/{stem}.json"


def fetch_bundle_to_scratch(bundle_url: str, local_dir: str) -> str:
    """Fetch a staged bundle to node-local scratch, manifest-driven -- no directory
    listing. Reads `bundle.json` (`fs.open`), then `fs.get`s each file its `artifacts` map
    and its `code` block name. Returns `local_dir`, ready for `bundle.load` --
    which is what puts the fetched `code/` on `sys.path`, so this node needs no
    adapter-specific image."""
    os.makedirs(local_dir, exist_ok=True)
    manifest_url = os.path.join(bundle_url, _bundle.BUNDLE_MANIFEST)
    with fs.open(manifest_url, "r") as f:
        manifest = json.load(f)
    with open(os.path.join(local_dir, _bundle.BUNDLE_MANIFEST), "w") as f:
        json.dump(manifest, f)
    rels = list(manifest.get("artifacts", {}).values()) + _bundle.manifest_code_files(manifest)
    for rel in rels:
        dst = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)   # code/ has subdirectories
        fs.get(os.path.join(bundle_url, rel), dst)
    return local_dir


def _n_final_exists(export_folderpaths) -> int:
    return sum(
        1 for p in export_folderpaths if fs.exists(os.path.join(p, "output.tif"))
    )


def _resolve_cores_and_group(
    n_units: int, cores: int | None, cubes_per_task: int | None
) -> tuple[int, int]:
    """The NODE computes the load-per-core default from its OWN core count and
    the shard size. `cores=None` => `os.cpu_count()` (run one group per core, so the bundle
    loads once per core and the node stays fully busy); `cubes_per_task=None` =>
    `ceil(n_units / cores)`, i.e. exactly `cores` groups. So the default (both None) is
    **load-per-core**, and the heavy-model **load-once-per-node** opt-out is simply `cores=1`
    (=> one whole-shard group, one bundle load). Explicit values override either knob."""
    eff_cores = (os.cpu_count() or 1) if cores is None else max(int(cores), 1)
    if cubes_per_task is None:
        eff_group = max(math.ceil(n_units / eff_cores), 1) if n_units else 1
    else:
        eff_group = max(int(cubes_per_task), 1)
    return eff_cores, eff_group


def run_infer_shard(
    shard_csv_url: str,
    bundle_url: str,
    *,
    cores: int | None = None,
    cubes_per_task: int | None = None,
    predict_batch_size: int | None = None,
    skip_nan: bool = True,
    overwrite: bool = False,
) -> dict:
    """Materialize the shard CSV + the staged bundle locally, run them via
    `runners.run_local_inference`, and publish a `_status/<k>.json`.

    `cores`/`cubes_per_task` default to `None` = the load-per-core default, computed here
    on the node from `os.cpu_count()` and the shard size (`_resolve_cores_and_group`)."""
    with fs.open(shard_csv_url, "r") as f:
        shard_df = pd.read_csv(f)

    fd, local_csv = tempfile.mkstemp(suffix=".csv", prefix="fsd-infer-shard-")
    os.close(fd)
    scratch_dir = tempfile.mkdtemp(prefix="fsd-infer-bundle-")
    try:
        shard_df.to_csv(local_csv, index=False)
        local_bundle = fetch_bundle_to_scratch(bundle_url, os.path.join(scratch_dir, "bundle"))

        n_units = len(shard_df)
        eff_cores, eff_group = _resolve_cores_and_group(n_units, cores, cubes_per_task)
        n_skipped = _n_final_exists(shard_df[EXPORT_FOLDERPATH_COL])
        t0 = time.time()
        work_start_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        error = None
        try:
            result = runners.run_local_inference(
                local_csv, cores=eff_cores, bundle_path=local_bundle,
                cubes_per_task=eff_group, predict_batch_size=predict_batch_size,
                skip_nan=skip_nan, overwrite=overwrite,
            )
            if result.returncode != 0:
                error = f"snakemake exited {result.returncode}"
                if result.stderr:
                    error += f":\n{result.stderr}"
        except Exception as exc:  # noqa: BLE001 - always report, never crash the job silently
            error = str(exc)
    finally:
        if os.path.exists(local_csv):
            os.remove(local_csv)
        shutil.rmtree(scratch_dir, ignore_errors=True)

    work_end_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    n_done = _n_final_exists(shard_df[EXPORT_FOLDERPATH_COL])
    n_failed = n_units - n_done
    status = {
        "shard": os.path.basename(shard_csv_url),
        "status": "ok" if (error is None and n_failed == 0) else "failed",
        "n_units": n_units,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        # Report the effective grouping this node ran, so bundle-loads
        # (== n_groups with a non-skipped cell) can be checked against the load-per-core intent.
        "cores": eff_cores,
        "cubes_per_task": eff_group,
        "n_groups": math.ceil(n_units / eff_group) if n_units else 0,
        "process_start_at": _PROCESS_START_AT,
        "work_start_at": work_start_at,
        "work_end_at": work_end_at,
        "seconds": round(time.time() - t0, 3),
        "error": error,
    }
    status["ended_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    with fs.open(_status_url(shard_csv_url), "w") as f:
        json.dump(status, f, indent=2)
    return status


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m rise_eo.workflows.infer_shard",
        description="Run one AML-dispatched shard of a ROI-inference work list (spec 38).",
    )
    p.add_argument("shard_csv_url", help="the shard's input.csv slice (any rise_eo.storage URL)")
    p.add_argument("bundle_url", help="the staged bundle folder (any rise_eo.storage URL)")
    p.add_argument("--cores", type=int, default=None,
                   help="groups run concurrently; default (unset) = os.cpu_count() (D7)")
    p.add_argument("--cubes-per-task", type=int, default=None,
                   help="cells per group; default (unset) = ceil(n_units/cores) (D7 load-per-core)")
    p.add_argument("--predict-batch-size", type=int, default=None)
    p.add_argument("--no-skip-nan", action="store_true",
                   help="predict on every pixel (default skips all-NaN pixels)")
    p.add_argument("--overwrite", action="store_true", help="re-infer even if output exists")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    status = run_infer_shard(
        args.shard_csv_url, args.bundle_url, cores=args.cores,
        cubes_per_task=args.cubes_per_task, predict_batch_size=args.predict_batch_size,
        skip_nan=not args.no_skip_nan, overwrite=args.overwrite,
    )
    if status["status"] != "ok":
        raise SystemExit(f"shard failed: {status}")


if __name__ == "__main__":
    main()
