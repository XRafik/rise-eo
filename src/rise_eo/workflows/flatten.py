"""In-job entrypoint for the AML flatten reduce.

A thin CLI wrapping the unmodified `datacube.flatten.flatten` -- no pipeline logic of its
own, mirroring `rise_eo.workflows.download`'s shape. Reads the caller's `input_csv` (any
`rise_eo.storage` url) via the storage seam, concatenates every listed datacube into ONE array
under `--export` (a reduce, not a fan-out -- `runners.run_aml_flatten` submits exactly one
job), and writes a `_status/<k>.json` -- the same `_result.json` shape every fsd job uses.

Runs on the **general-purpose** fsd Environment (ADR-0020): no adapter import here, ever --
the feature transform stays on the driver, after land-local.

Run as:
  python -m rise_eo.workflows.flatten --input-csv <url> --filepath-col datacube_filepath \\
      --id-col id --label-col label --export <url> --status-url <url>
"""

from __future__ import annotations

import datetime as _dt

# Stamped before any heavy import, so it is process start (see workflows/shard.py).
_PROCESS_START_AT = _dt.datetime.now(_dt.timezone.utc).isoformat()

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402

from rise_eo import config  # noqa: E402
from rise_eo.datacube import flatten as _flatten  # noqa: E402
from rise_eo.storage import fs  # noqa: E402


def _write_status(status_url: str, status: dict) -> None:
    with fs.open(status_url, "w") as f:
        json.dump(status, f, indent=2)


def run(
    *,
    input_csv: str,
    export_folderpath: str,
    status_url: str,
    id_col: str = "id",
    label_col: str | None = None,
    filepath_col: str = "datacube_filepath",
    nodata: int = config.NODATA,
) -> dict:
    """Read `input_csv`, flatten every listed datacube into `export_folderpath`, publish
    status. Raises (loudly, ADR-0013) if any listed cube/metadata is missing -- a lost
    cube must not be silently dropped from the reduce."""
    start = time.monotonic()
    with fs.open(input_csv, "r") as f:
        df = pd.read_csv(f)

    work_start_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    _flatten.flatten(
        filepaths_df=df, filepath_col=filepath_col, id_col=id_col,
        export_folderpath=export_folderpath, label_col=label_col, nodata=nodata,
    )
    work_end_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

    status = {
        "unit": "flatten",
        "status": "ok",
        "n_cubes": len(df),
        "process_start_at": _PROCESS_START_AT,
        "work_start_at": work_start_at,
        "work_end_at": work_end_at,
        "seconds": round(time.monotonic() - start, 3),
        "error": None,
    }
    status["ended_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    _write_status(status_url, status)
    return status


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m rise_eo.workflows.flatten",
        description="Run the AML-dispatched flatten reduce (spec 39): concatenate every "
                     "datacube listed in --input-csv into one training array under --export.",
    )
    p.add_argument("--input-csv", required=True)
    p.add_argument("--filepath-col", default="datacube_filepath")
    p.add_argument("--id-col", default="id")
    p.add_argument("--label-col", default=None)
    p.add_argument("--export", required=True, dest="export_folderpath")
    p.add_argument("--nodata", type=int, default=config.NODATA)
    p.add_argument("--status-url", required=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    status = run(
        input_csv=args.input_csv, export_folderpath=args.export_folderpath,
        status_url=args.status_url, id_col=args.id_col, label_col=args.label_col,
        filepath_col=args.filepath_col, nodata=args.nodata,
    )
    if status["status"] != "ok":
        raise SystemExit(f"flatten reduce failed: {status}")


if __name__ == "__main__":
    main()
