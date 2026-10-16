"""The ROI-inference unit-of-work: build ONE datacube, then infer it to a COG.

Spec: specs/21-roi-inference-verb.md (P0.75).

A superset of `rise_eo.workflows.task`: the *same* datacube build, plus a model-inference step
that loads the bundle **once per task** and writes `output.tif` alongside the datacube. This is
the **per-cell build+infer** work-unit each Azure ML node runs unchanged (`runner="aml"`, through
`rise_eo.workflows.infer_shard`) — the reason inference goes through the runner seam rather than a
separate process pool.

The model must be a **bundle** (it crosses a subprocess boundary); `run_inference(roi=…)`
auto-saves a live adapter to one.

Run as:  python -m rise_eo.workflows.infer_task <shapefilepath> <catalog_filepath>
             <startdate> <enddate> <export_folderpath>
             --bands B04,B08,B8A,SCL --mosaic-days 20
             --declaration-filepath <run_folderpath>/<window>/declaration.json
             --bundle <bundle_path> --output <output.tif>
"""

from __future__ import annotations

import argparse
import datetime
import os

import pandas as pd

from rise_eo import config
from rise_eo.datacube import ops
from rise_eo.model import engine
from rise_eo.storage import fs
from rise_eo.workflows import task as _task


def run_infer_task(
    shapefilepath: str,
    catalog_filepath: str,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    export_folderpath: str,
    *,
    bands: list[str],
    mosaic_days: int,
    declaration_filepath: str,
    bundle_path: str,
    output_filepath: str,
    mosaic_scheme: str = config.MOSAIC_SCHEME,
    if_missing_files: str | None = "warn",
    njobs: int = 1,
    njobs_load_images: int = 1,
    predict_batch_size: int | None = None,
    skip_nan: bool = True,
    overwrite: bool = False,
) -> str:
    """Build the cell's datacube (reuses `task.run_task`) then infer it to `output_filepath`.

    Returns the output COG path.

    Returns immediately, without building or inferring, if `output_filepath`
    already exists and `overwrite=False` — the durable per-cell resume signal (mirrors
    `task.run_task`'s own `datacube.npy`-exists skip), so a group retried on a fresh node
    redoes only its unfinished cells.

    The adapter is resolved via the per-process bundle cache (TODO #25)
    (`engine._adapter_from_bundle_cached`), not a fresh `bundle.load` — a bundle loads once
    per process, not once per cell, when this is called in a loop over a group that shares
    one process (see `run_infer_group` / the `create_inference` Snakefile).
    """
    if not overwrite and fs.exists(output_filepath):
        return output_filepath
    _task.run_task(
        shapefilepath=shapefilepath,
        catalog_filepath=catalog_filepath,
        startdate=startdate,
        enddate=enddate,
        export_folderpath=export_folderpath,
        bands=bands,
        mosaic_days=mosaic_days,
        declaration_filepath=declaration_filepath,
        mosaic_scheme=mosaic_scheme,
        if_missing_files=if_missing_files,
        njobs=njobs,
        njobs_load_images=njobs_load_images,
    )
    datacube_filepath = os.path.join(export_folderpath, "datacube.npy")
    adapter = engine._adapter_from_bundle_cached(bundle_path)
    return engine.infer_datacube_to_cog(
        adapter, datacube_filepath, output_filepath,
        predict_batch_size=predict_batch_size, skip_nan=skip_nan,
    )


def run_infer_group(
    input_csv: str,
    rows: tuple[int, int],
    bundle_path: str,
    *,
    if_missing_files: str | None = "warn",
    njobs: int = 1,
    njobs_load_images: int = 1,
    predict_batch_size: int | None = None,
    skip_nan: bool = True,
    overwrite: bool = False,
) -> list[str]:
    """Build+infer rows `[lo, hi)` of `input_csv` **sequentially in one process**.

    The whole point of grouping: every row in the group shares the SAME process, so the
    bundle-cache in `run_infer_task` (`engine._adapter_from_bundle_cached`) loads the bundle
    once for the group, not once per cell -- this is the "cubes_per_task" amortiser the
    `create_inference` Snakefile groups rows into. Each row's own skip-if-exists still
    applies, so a group re-run after a partial failure only redoes its unfinished cells.
    """
    with fs.open(input_csv, "r") as f:
        df = pd.read_csv(f)
    lo, hi = rows
    written = []
    for _, r in df.iloc[lo:hi].iterrows():
        export_folderpath = str(r["export_folderpath"])
        output_filepath = os.path.join(export_folderpath, "output.tif")
        out = run_infer_task(
            str(r["shapefilepath"]), str(r["catalog_filepath"]),
            pd.to_datetime(r["startdate"]), pd.to_datetime(r["enddate"]),
            export_folderpath,
            bands=str(r["bands"]).split(","),
            mosaic_days=int(r["mosaic_days"]),
            declaration_filepath=str(r["declaration_filepath"]),
            bundle_path=bundle_path,
            output_filepath=output_filepath,
            mosaic_scheme=str(r["mosaic_scheme"]) if "mosaic_scheme" in r else config.MOSAIC_SCHEME,
            if_missing_files=if_missing_files,
            njobs=njobs, njobs_load_images=njobs_load_images,
            predict_batch_size=predict_batch_size, skip_nan=skip_nan,
            overwrite=overwrite,
        )
        written.append(out)
    return written


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m rise_eo.workflows.infer_task",
        description="Build a cell's datacube and infer it to a COG -- one cell (positional "
                     "args) or a group of cells sharing one bundle load (--input-csv/--rows, "
                     "spec 38 D7).",
    )
    p.add_argument("shapefilepath", nargs="?", help="geometry.geojson for this cell "
                   "(single-cell mode)")
    p.add_argument("catalog_filepath", nargs="?", help="per-cell catalog.parquet slice "
                   "(single-cell mode)")
    p.add_argument("startdate", nargs="?", help="calendar-window start; ISO or 'YYYY-MM-DD' "
                   "(single-cell mode)")
    p.add_argument("enddate", nargs="?", help="calendar-window end (single-cell mode)")
    p.add_argument("export_folderpath", nargs="?", help="where datacube.npy + metadata go "
                   "(single-cell mode)")
    p.add_argument("--bands", help="comma-separated, e.g. B04,B08,B8A,SCL (single-cell mode)")
    p.add_argument("--mosaic-days", type=int, default=config.MOSAIC_DAYS)
    p.add_argument("--mosaic-scheme", default=config.MOSAIC_SCHEME,
                   choices=list(ops.MOSAIC_SCHEMES))
    p.add_argument("--declaration-filepath",
                   help="control file written by setup(): the resolved "
                        "CollectionDeclaration as JSON (single-cell mode; spec 58 D13)")
    p.add_argument("--if-missing-files", default="warn",
                   choices=["raise_error", "warn", "none"])
    p.add_argument("--njobs", type=int, default=1)
    p.add_argument("--njobs-load-images", type=int, default=1)
    p.add_argument("--bundle", required=True, help="model bundle path")
    p.add_argument("--output", help="output COG path (single-cell mode)")
    p.add_argument("--input-csv", help="run manifest (group mode, spec 38 D7)")
    p.add_argument("--rows", help="row slice 'i:j' of --input-csv (group mode)")
    p.add_argument("--predict-batch-size", type=int, default=None)
    p.add_argument("--no-skip-nan", action="store_true",
                   help="predict on every pixel (default skips all-NaN pixels)")
    p.add_argument("--overwrite", action="store_true", help="re-infer even if output exists")
    return p.parse_args(argv)


def _parse_rows(s: str) -> tuple[int, int]:
    lo, hi = s.split(":")
    return int(lo), int(hi)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if bool(args.input_csv) == bool(args.shapefilepath):
        raise SystemExit(
            "exactly one of positional single-cell args or --input-csv/--rows is required"
        )
    if args.input_csv:
        if not args.rows:
            raise SystemExit("--input-csv requires --rows i:j")
        run_infer_group(
            args.input_csv, _parse_rows(args.rows), args.bundle,
            if_missing_files=None if args.if_missing_files == "none" else args.if_missing_files,
            njobs=args.njobs, njobs_load_images=args.njobs_load_images,
            predict_batch_size=args.predict_batch_size, skip_nan=not args.no_skip_nan,
            overwrite=args.overwrite,
        )
        return
    if not (args.bands and args.output and args.declaration_filepath):
        raise SystemExit(
            "single-cell mode requires --bands, --output and --declaration-filepath"
        )
    run_infer_task(
        shapefilepath=args.shapefilepath,
        catalog_filepath=args.catalog_filepath,
        startdate=pd.to_datetime(args.startdate),
        enddate=pd.to_datetime(args.enddate),
        export_folderpath=args.export_folderpath,
        bands=args.bands.split(","),
        mosaic_days=args.mosaic_days,
        declaration_filepath=args.declaration_filepath,
        bundle_path=args.bundle,
        output_filepath=args.output,
        mosaic_scheme=args.mosaic_scheme,
        if_missing_files=None if args.if_missing_files == "none" else args.if_missing_files,
        njobs=args.njobs,
        njobs_load_images=args.njobs_load_images,
        predict_batch_size=args.predict_batch_size,
        skip_nan=not args.no_skip_nan,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
