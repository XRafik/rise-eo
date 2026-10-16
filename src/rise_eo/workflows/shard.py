"""In-job entrypoint for the AML runner.

A thin shim: resolve one shard CSV (any `rise_eo.storage` URL) to a local file, then call
the **existing** local runner over it. `rise_eo.workflows.task` (the unit of work) and
`rise_eo.workflows.runners.run_local` are unchanged; this file adds no pipeline logic of
its own -- it exists only so the same local orchestration can be invoked from an AML
node instead of a laptop.

Run as: python -m rise_eo.workflows.shard <shard_csv_url> --cores N
"""

from __future__ import annotations

import datetime as _dt

# Stamped before any heavy import, so it reflects process start, not "after
# argparse/pandas/rise_eo loaded".
_PROCESS_START_AT = _dt.datetime.now(_dt.timezone.utc).isoformat()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402

from rise_eo.storage import fs  # noqa: E402
from rise_eo.workflows import runners  # noqa: E402

EXPORT_FOLDERPATH_COL = "export_folderpath"


def _status_url(shard_csv_url: str) -> str:
    """`<root>/runs/<run_id>/shards/<k>.csv` -> `<root>/runs/<run_id>/_status/<k>.json`,
    derived from the shard's own path so the CLI stays the two arguments the shard url
    already implies -- no extra "where do I report" argument."""
    root, name = shard_csv_url.rsplit("/shards/", 1)
    stem = name[:-4] if name.endswith(".csv") else name
    return f"{root}/_status/{stem}.json"


def _n_final_exists(export_folderpaths) -> int:
    return sum(
        1 for p in export_folderpaths if fs.exists(os.path.join(p, "datacube.npy"))
    )


def run_shard(shard_csv_url: str, *, cores: int) -> dict:
    """Materialize `shard_csv_url` locally, run it via `runners.run_local`, and publish
    a `_status/<k>.json` shaped like a spec-24 `_result.json`."""
    with fs.open(shard_csv_url, "r") as f:
        shard_df = pd.read_csv(f)

    fd, local_csv = tempfile.mkstemp(suffix=".csv", prefix="rise-eo-shard-")
    os.close(fd)
    try:
        shard_df.to_csv(local_csv, index=False)

        n_units = len(shard_df)
        n_skipped = _n_final_exists(shard_df[EXPORT_FOLDERPATH_COL])
        t0 = time.time()
        work_start_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
        error = None
        try:
            result = runners.run_local(local_csv, cores=cores)
            if result.returncode != 0:
                error = f"snakemake exited {result.returncode}"
                if result.stderr:
                    error += f":\n{result.stderr}"
        except Exception as exc:  # noqa: BLE001 - always report, never crash the job silently
            error = str(exc)
    finally:
        if os.path.exists(local_csv):
            os.remove(local_csv)

    work_end_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    n_done = _n_final_exists(shard_df[EXPORT_FOLDERPATH_COL])
    n_failed = n_units - n_done
    status = {
        "shard": os.path.basename(shard_csv_url),
        "status": "ok" if (error is None and n_failed == 0) else "failed",
        "n_units": n_units,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
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
        prog="python -m rise_eo.workflows.shard",
        description="Run one AML-dispatched shard of a datacube-build work list (spec 36).",
    )
    p.add_argument("shard_csv_url", help="the shard's input.csv slice (any rise_eo.storage URL)")
    p.add_argument("--cores", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    status = run_shard(args.shard_csv_url, cores=args.cores)
    if status["status"] != "ok":
        raise SystemExit(f"shard failed: {status}")


if __name__ == "__main__":
    main()
