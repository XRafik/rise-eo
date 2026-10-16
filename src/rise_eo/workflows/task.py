"""The unit-of-work: build ONE datacube. Runner-agnostic, CLI-invokable.

Spec: specs/08-workflows.md, specs/10-storage-and-scale.md.

Azure ML runs this same task on each node (`runner="aml"`, through `rise_eo.workflows.shard`). It:
- reads its inputs (geometry + pre-sliced catalog) and writes its artifact via
  rise_eo.storage / the builder,
- knows nothing about how it was scheduled.

Run as:  python -m rise_eo.workflows.task <shapefilepath> <catalog_filepath>
             <startdate> <enddate> <export_folderpath>
             --bands B04,B08,B8A,SCL --mosaic-days 20
             --declaration-filepath <run_folderpath>/<window>/declaration.json
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import os

import geopandas as gpd
import pandas as pd

from rise_eo import config
from rise_eo.catalog import declaration as declaration_module
from rise_eo.datacube import builder, ops
from rise_eo.storage import fs


def run_task(
    shapefilepath: str,
    catalog_filepath: str,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    export_folderpath: str,
    *,
    bands: list[str],
    mosaic_days: int,
    declaration_filepath: str,
    mosaic_scheme: str = config.MOSAIC_SCHEME,
    if_missing_files: str | None = "warn",
    njobs: int = 1,
    njobs_load_images: int = 1,
    write_timings: bool = False,
    write_read_log: bool = False,
) -> None:
    """Build one datacube from a setup work-unit and save it to export_folderpath.

    `catalog_filepath` is the per-shape GeoParquet slice written by setup
    (`TileCatalog.filter` output — already date+overlap filtered, carries
    `area_contribution`); we band-flatten it and hand it to the builder.

    `declaration_filepath` is the control file `setup` wrote under the run folder's WINDOW
    segment: the `collection=` string resolved to a `CollectionDeclaration` and serialized to
    JSON **on the driver**, before this task was dispatched. This node reads that file and
    nothing else -- it never imports `rise_eo.collections` to resolve a collection string, because
    that registry is a plain in-process dict that does not exist inside a fresh AML job
    container (ADR 0031). The declaration is validated against the catalog's own stamp on
    ARTIFACT facts: a build-policy difference (mask, reference band, mosaic method) is fine;
    an artifact-fact difference raises.

    `write_timings` asks the builder to drop a `timings.json` sidecar (benchmark
    seam); `write_read_log` asks for a `reads.jsonl` per-read log.
    `main()` sets both from the `RISE_EO_WRITE_TIMINGS` / `RISE_EO_WRITE_READ_LOG` env vars so
    the harness can enable them without any runner/Snakefile plumbing.

    A task whose final artifact (`datacube.npy`) already exists at `export_folderpath`
    returns immediately -- the resume signal is the artifact's own existence, not a sentinel,
    so a re-dispatched (recovery-retried) shard skips every cube it already finished.

    Spec: 58 D13, D14.
    """
    if fs.exists(os.path.join(export_folderpath, "datacube.npy")):
        return

    with fs.open(declaration_filepath, "r") as f:
        declaration = declaration_module.from_json(json.load(f))

    # Read via rise_eo.storage + BytesIO, not gpd.read_file(path) (TODO #40)
    # directly -- a cluster node has no local checkout of the caller's geometry file.
    with fs.open(shapefilepath, "rb") as f:
        shape_gdf = gpd.read_file(io.BytesIO(f.read()))
    subset_gdf = fs.read_parquet(catalog_filepath)
    flattened = builder.flatten_catalog(subset_gdf)

    builder.build_datacube(
        catalog_subset=flattened,
        shape_gdf=shape_gdf,
        startdate=startdate,
        enddate=enddate,
        bands=bands,
        mosaic_days=mosaic_days,
        declaration=declaration,
        mosaic_scheme=mosaic_scheme,
        export_folderpath=export_folderpath,
        njobs=njobs,
        njobs_load_images=njobs_load_images,
        if_missing_files=if_missing_files,
        write_timings=write_timings,
        write_read_log=write_read_log,
    )


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m rise_eo.workflows.task",
        description="Build a single datacube (one work-unit), collection-agnostic.",
    )
    p.add_argument("shapefilepath", help="geometry.geojson for this shape")
    p.add_argument("catalog_filepath", help="per-shape catalog.parquet slice")
    p.add_argument("startdate", help="calendar-window start; ISO or 'YYYY-MM-DD'")
    p.add_argument("enddate", help="calendar-window end")
    p.add_argument("export_folderpath", help="where datacube.npy + metadata go")
    p.add_argument("--bands", required=True, help="comma-separated, e.g. B04,B08,B8A,SCL")
    p.add_argument("--mosaic-days", type=int, default=config.MOSAIC_DAYS)
    p.add_argument("--mosaic-scheme", default=config.MOSAIC_SCHEME,
                   choices=list(ops.MOSAIC_SCHEMES))
    p.add_argument("--declaration-filepath", required=True,
                   help="control file written by setup(): the resolved "
                        "CollectionDeclaration as JSON (spec 58 D13)")
    p.add_argument("--if-missing-files", default="warn",
                   choices=["raise_error", "warn", "none"])
    p.add_argument("--njobs", type=int, default=1)
    p.add_argument("--njobs-load-images", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run_task(
        shapefilepath=args.shapefilepath,
        catalog_filepath=args.catalog_filepath,
        startdate=pd.to_datetime(args.startdate),
        enddate=pd.to_datetime(args.enddate),
        export_folderpath=args.export_folderpath,
        bands=args.bands.split(","),
        mosaic_days=args.mosaic_days,
        mosaic_scheme=args.mosaic_scheme,
        declaration_filepath=args.declaration_filepath,
        if_missing_files=None if args.if_missing_files == "none" else args.if_missing_files,
        njobs=args.njobs,
        njobs_load_images=args.njobs_load_images,
        write_timings=os.environ.get("RISE_EO_WRITE_TIMINGS") == "1",
        write_read_log=os.environ.get("RISE_EO_WRITE_READ_LOG") == "1",
    )


if __name__ == "__main__":
    main()
