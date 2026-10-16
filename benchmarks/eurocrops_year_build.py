"""Full-year (2018) datacube build for the 1015-field EuroCrops training set.

Builds one datacube per field over the whole of 2018 (mosaic_days=20 -> ~19 calendar
mosaic timestamps, spec 15) via the real workflow (setup -> Snakemake -> task), with
per-cube `timings.json` sidecars enabled (RISE_EO_WRITE_TIMINGS) so the companion report
script can profile the build.

Dataset: shapefiles/austria_eurocrops_sampled_ethiopia_translated.geojson (id=`fid`,
label=`EC_hcat_n`, 11 crop classes, both UTM zones) against the COG archive
satellite_benchmark/. Cubes land under fsd/tests/outputs/datacube_year/ (gitignored);
the report + figures are produced by `eurocrops_year_report.py`.

Run from the workspace root (parent of fsd/):
    RISE_EO_WRITE_TIMINGS=1 fsd/.venv/bin/python fsd/benchmarks/eurocrops_year_build.py
"""

from __future__ import annotations

import datetime
import os
import time

import pandas as pd

# per-cube timings.json (propagates to the task subprocesses the workflow spawns)
os.environ.setdefault("RISE_EO_WRITE_TIMINGS", "1")

from rise_eo.workflows import create_datacube  # noqa: E402  (after env set)

SHAPES = "shapefiles/austria_eurocrops_sampled_ethiopia_translated.geojson"
CATALOG = "satellite_benchmark/sentinel-2-l2a/catalog.parquet"
RUN = "fsd/tests/outputs/datacube_year/run"
CSV = os.path.join(RUN, "input.csv")

STARTDATE = datetime.datetime(2018, 1, 1)
ENDDATE = datetime.datetime(2019, 1, 1)
BANDS = ["B04", "B08", "B8A", "SCL"]
SCL_MASK = [0, 1, 3, 7, 8, 9, 10]
MOSAIC_DAYS = 20
CORES = 8


def main() -> None:
    os.makedirs(RUN, exist_ok=True)
    t0 = time.time()
    print(f"[{datetime.datetime.now():%H:%M:%S}] full-year (2018) build, 1015 fields, "
          f"cores={CORES}, timings={os.environ.get('RISE_EO_WRITE_TIMINGS')}", flush=True)

    create_datacube.run_create_datacube(
        catalog_filepath=CATALOG, timestamp_col="timestamp", shapefilepath=SHAPES,
        id_col="fid", run_folderpath=RUN, startdate=STARTDATE, enddate=ENDDATE,
        bands=BANDS, scl_mask_classes=SCL_MASK, mosaic_days=MOSAIC_DAYS,
        csv_filepath=CSV, label_col="EC_hcat_n", cores=CORES,
    )

    df = pd.read_csv(CSV)
    done = sum(os.path.exists(os.path.join(os.path.dirname(f), "done.txt"))
               for f in df["datacube_filepath"])
    tim = sum(os.path.exists(os.path.join(os.path.dirname(f), "timings.json"))
              for f in df["datacube_filepath"])
    print(f"[{datetime.datetime.now():%H:%M:%S}] BUILD DONE: {len(df)} units | "
          f"{done} done.txt | {tim} timings.json | {(time.time() - t0) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
