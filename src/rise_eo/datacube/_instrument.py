"""The datacube builder's benchmark instrumentation: the phase timer and the `timings.json` /
`reads.jsonl` sidecars that `build_datacube(write_timings=, write_read_log=)` writes (specs 11
and 12). Off by default; `workflows.task` turns them on from `RISE_EO_WRITE_TIMINGS` /
`RISE_EO_WRITE_READ_LOG`."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager

from rise_eo.raster import images
from rise_eo.storage import fs

TIMINGS_FILENAME = "timings.json"
READ_LOG_FILENAME = "reads.jsonl"


@contextmanager
def timed(store: dict, name: str):
    """Record wall-seconds for a build phase into `store` -- the benchmark seam."""
    t0 = time.perf_counter()
    yield
    store[name] = round(time.perf_counter() - t0, 4)


def write_timings_sidecar(export_folderpath, timings, total_seconds, *, shape_gdf,
                          catalog_subset, catalog_gdf, n_resampled, datacube, metadata,
                          dst_crs):
    """Dump per-phase timings + a few sizing counts as `timings.json`."""
    payload = {
        "id": (str(shape_gdf["id"].iloc[0]) if "id" in shape_gdf.columns else None),
        "total_seconds": total_seconds,
        "phase_seconds": timings,
        "n_band_rows": int(len(catalog_subset)),
        "n_images_loaded": int(len(catalog_gdf)),
        "n_images_resampled": int(n_resampled),
        "n_mosaic_timestamps": int(len(metadata["timestamps"])),
        "datacube_shape": list(datacube.shape),
        "dst_crs": str(dst_crs),
    }
    with fs.open(os.path.join(export_folderpath, TIMINGS_FILENAME), "w") as f:
        json.dump(payload, f, indent=2)


def write_read_log_sidecar(export_folderpath, reads):
    """Dump one JSON row per windowed read as `reads.jsonl`."""
    fs.makedirs(export_folderpath)
    with fs.open(os.path.join(export_folderpath, READ_LOG_FILENAME), "w") as f:
        for r in reads:
            f.write(json.dumps(r) + "\n")


def _mgrs_tile(product_id, filepath) -> str | None:
    """Parse the MGRS tile (`..._T36NXF_...`) from the product id or the parent
    folder name. Returns None if neither carries the `_T<MGRS tile>` marker (e.g. the
    synthetic granules in tests) — the same-file key is `filepath`, not this."""
    for s in (product_id, os.path.basename(os.path.dirname(filepath))):
        s = str(s)
        if "_T" in s:
            cand = s.split("_T", 1)[1][:5]
            if len(cand) == 5:
                return cand
    return None


def load_images_logged(filepaths, shape_gdf, nodata, product_ids, bands):
    """Serial load+crop, timing each windowed read with wall-clock `time.time()`
    (comparable across processes). Returns (data_profile_list, reads)."""
    grid_id = (str(shape_gdf["id"].iloc[0]) if "id" in shape_gdf.columns else None)
    data_profile_list, reads = [], []
    for fp, pid, band in zip(filepaths, product_ids, bands):
        t0 = time.time()
        dp = images.load_image(
            src_filepath=fp, shapes_gdf=shape_gdf, nodata=nodata,
            all_touched=True, raise_error=False,
        )
        t1 = time.time()
        data_profile_list.append(dp)
        reads.append({
            "id": grid_id,
            "mgrs_tile": _mgrs_tile(pid, fp),
            "product_id": str(pid),
            "band": str(band),
            "filepath": fp,
            "start": round(t0, 6),
            "end": round(t1, 6),
            "duration": round(t1 - t0, 6),
        })
    return data_profile_list, reads
