"""Heavy full-year (2018) datacube build + 3 NDVI variants (raw / mosaic-no-mask /
mosaic-mask) + spaghetti + spatial triptych. Saves arrays + stats; plotting is separate.
"""
import datetime
import json
import multiprocessing
import os
import platform
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd
import numpy as np
import psutil

from rise_eo import config
from rise_eo.bands import modify
from rise_eo.catalog.catalog import TileCatalog
from rise_eo.datacube import builder, ops

ROOT = str(Path(__file__).resolve().parents[2])  # workspace root: the folder holding fsd/ and the archives
CATALOG = f"{ROOT}/satellite_benchmark/sentinel-2-l2a/catalog.parquet"
ROI = f"{ROOT}/shapefiles/s2grid=165bca4.geojson"
OUT = f"{ROOT}/fsd/tests/outputs/datacube_fullyear"

BANDS = ["B04", "B08", "B8A", "SCL"]
START, END = datetime.datetime(2018, 1, 1), datetime.datetime(2019, 1, 1)
MOSAIC_DAYS = 20
NJOBS_LOAD = NJOBS = 4

proc = psutil.Process()
_peak, _stop = [0], [False]
timings = {}


def _sample():
    while not _stop[0]:
        try:
            rss = proc.memory_info().rss + sum(
                c.memory_info().rss for c in proc.children(recursive=True))
        except Exception:
            rss = proc.memory_info().rss
        _peak[0] = max(_peak[0], rss)
        time.sleep(0.2)


@contextmanager
def phase(name):
    t0 = time.time()
    print(f"[start] {name}", flush=True)
    yield
    timings[name] = round(time.time() - t0, 1)
    print(f"[done ] {name}: {timings[name]}s | peak {_peak[0]/1e9:.2f} GB", flush=True)


def ndvi_hw(cube4d, bidx):
    """NDVI per timestamp as (t, h, w) float, nodata(0) -> nan."""
    b04 = cube4d[..., bidx["B04"]].astype(float)
    b08 = cube4d[..., bidx["B08"]].astype(float)
    denom = b08 + b04
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (b08 - b04) / denom, np.nan)


def area_median(ndvi_thw):
    flat = ndvi_thw.reshape(ndvi_thw.shape[0], -1)
    with np.errstate(invalid="ignore"):
        return np.nanmedian(flat, axis=1)


def main():
    os.makedirs(OUT, exist_ok=True)
    threading.Thread(target=_sample, daemon=True).start()
    t_all = time.time()
    nodata = config.NODATA

    with phase("catalog_filter"):
        sub = TileCatalog(CATALOG).filter(gpd.read_file(ROI), START, END)
    roi = gpd.read_file(ROI)
    print("tiles:", len(sub), "| dates:", sub["timestamp"].nunique(),
          "| zones:", sorted({t.split('_T')[1][:2] for t in sub['id']}), flush=True)

    with phase("flatten_catalog"):
        flat = builder.flatten_catalog(sub)
    with phase("missing_files_check"):
        builder._missing_files_action(catalog_gdf=flat, shape_gdf=roi, startdate=START,
                                      enddate=END, bands=BANDS, if_missing_files="warn",
                                      max_timedelta_days=config.MAX_TIMEDELTA_DAYS)
    with phase("load_images"):
        catalog_gdf, dpl = builder._load_images(flat, roi, nodata, njobs=NJOBS_LOAD)
    with phase("dst_crs"):
        dst_crs = builder._get_dst_crs(catalog_gdf)
    with phase("reference_profile"):
        ref_idx = catalog_gdf.loc[catalog_gdf["band"] == "B08", "image_index"]
        reference_profile = builder._get_merged_profile(ref_idx, dpl, dst_crs, nodata, NJOBS)
    with phase("resample_to_ref"):
        r_idx = builder._get_indices_to_resample(catalog_gdf["image_index"], dpl,
                                                 reference_profile)
        builder._resample_by_indices(r_idx, dpl, reference_profile, NJOBS)
    with phase("stack"):
        stacked, stack_md = builder._stack_datacube(catalog_gdf, dpl, BANDS,
                                                    reference_profile, roi, nodata)
    print("stacked (pre-ops):", stacked.shape, flush=True)
    bidx_s = {b: i for i, b in enumerate(BANDS)}
    raw_times = [str(t) for t in stack_md["timestamps"]]

    # --- three NDVI variants (derive from the single stacked cube) -----------
    with phase("variant_raw"):
        raw_ndvi = ndvi_hw(stacked, bidx_s)                 # (n_acq, h, w)
        raw_median = area_median(raw_ndvi)

    with phase("variant_mosaic_nomask"):
        dc_nm, md_nm = ops.drop_bands(stacked.copy(), dict(stack_md), bands_to_drop=["SCL"])
        dc_nm, md_nm = ops.median_mosaic(dc_nm, md_nm, startdate=START, enddate=END,
                                         mosaic_days=MOSAIC_DAYS)
        bidx_m = {b: i for i, b in enumerate(md_nm["bands"])}
        nomask_ndvi = ndvi_hw(dc_nm, bidx_m)
        nomask_median = area_median(nomask_ndvi)
        mosaic_times = [str(t) for t in md_nm["timestamps"]]

    with phase("variant_mosaic_mask"):
        dc_m, md_m = ops.run_ops(stacked.copy(), dict(stack_md), sequence=[
            (ops.apply_cloud_mask_scl, dict(mask_classes=config.SCL_MASK_CLASSES)),
            (ops.drop_bands, dict(bands_to_drop=["SCL"])),
            (ops.median_mosaic, dict(startdate=START, enddate=END, mosaic_days=MOSAIC_DAYS)),
        ])
        mask_ndvi = ndvi_hw(dc_m, bidx_m)
        mask_median = area_median(mask_ndvi)

    np.save(f"{OUT}/datacube.npy", dc_m)
    np.save(f"{OUT}/metadata.pickle.npy", md_m, allow_pickle=True)
    total = round(time.time() - t_all, 1)
    print(f"\nDATACUBE {dc_m.shape} {dc_m.dtype} | mosaic ts={len(md_m['timestamps'])} "
          f"| dst_crs={md_m['geotiff_metadata']['crs']}", flush=True)

    # --- spaghetti (final masked cube, demo plot_ndvi_data style) ------------
    with phase("spaghetti_interp"):
        data5d = np.expand_dims(dc_m, axis=0)
        interp, ibi = modify.modify_bands(
            bands=data5d, band_indices=dict(bidx_m),
            sequence=[(modify.mask_invalid_and_interpolate, {}),
                      (modify.compute_bands, dict(bands_to_compute=["NDVI"]))])
        ndvi_final = np.squeeze(interp)[:, :, :, ibi["NDVI"]]
        yx = np.where(~np.isnan(ndvi_final).any(axis=0))
        ndvi_2d = ndvi_final[:, yx[0], yx[1]]
        spaghetti = ndvi_2d.T[:100]
        spaghetti_median = np.median(ndvi_2d, axis=1)

    # --- spatial triptych: the season bucket where cloud masking mattered most
    # (largest mask - nomask NDVI gain), raw panel = a cloudy date inside it ----
    raw_ts = np.array(stack_md["timestamps"])
    scl = stacked[..., bidx_s["SCL"]]
    cloud_count = np.isin(scl, config.SCL_MASK_CLASSES).sum(axis=(1, 2)).astype(float)
    intervals = md_m["mosaic_index_intervals"]
    bmonth = np.array([t.month for t in md_m["timestamps"]])
    gain = np.where((bmonth >= 5) & (bmonth <= 10), mask_median - nomask_median, -np.inf)
    b_map = int(np.nanargmax(gain))
    s, e = intervals[b_map]
    in_bucket = np.zeros(len(raw_ts), bool)
    in_bucket[s:e + 1] = True
    t_raw = int(np.where(in_bucket, cloud_count, -1).argmax())
    print(f"map bucket {b_map} ({md_m['timestamps'][b_map]}) gain={gain[b_map]:.3f}; "
          f"raw acq idx {t_raw} ({raw_ts[t_raw]}) cloud px {int(cloud_count[t_raw])}",
          flush=True)

    np.save(f"{OUT}/map_raw.npy", raw_ndvi[t_raw])
    np.save(f"{OUT}/map_nomask.npy", nomask_ndvi[b_map])
    np.save(f"{OUT}/map_mask.npy", mask_ndvi[b_map])
    for nm, arr in [("raw_median", raw_median), ("raw_times", np.array(raw_times)),
                    ("nomask_median", nomask_median), ("mask_median", mask_median),
                    ("mosaic_times", np.array(mosaic_times)), ("spaghetti", spaghetti),
                    ("spaghetti_median", spaghetti_median)]:
        np.save(f"{OUT}/{nm}.npy", arr)

    _stop[0] = True
    stats = {
        "run_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "roi": "s2grid=165bca4.geojson", "window": [str(START.date()), str(END.date())],
        "bands": BANDS, "mosaic_days": MOSAIC_DAYS,
        "njobs_load": NJOBS_LOAD, "njobs": NJOBS,
        "n_tiles": int(len(sub)), "n_acquisition_dates": int(sub["timestamp"].nunique()),
        "n_band_rows": int(len(flat)), "n_images_resampled": int(len(r_idx)),
        "stacked_shape": list(stacked.shape), "datacube_shape": list(dc_m.shape),
        "n_mosaic_timestamps": int(len(md_m["timestamps"])),
        "dst_crs": str(md_m["geotiff_metadata"]["crs"]),
        "valid_ndvi_pixels": int(ndvi_2d.shape[1]),
        "map_selection": {"bucket": b_map, "bucket_date": str(md_m["timestamps"][b_map]),
                          "ndvi_gain": round(float(gain[b_map]), 3),
                          "raw_idx": t_raw, "raw_date": str(raw_ts[t_raw]),
                          "raw_cloud_px": int(cloud_count[t_raw])},
        "peak_rss_gb": round(_peak[0] / 1e9, 2),
        "total_seconds": total, "phase_seconds": timings,
    }
    json.dump(stats, open(f"{OUT}/heavy_stats.json", "w"), indent=2)
    print("\n=== STATS ===\n" + json.dumps(stats, indent=2))
    print("saved to", OUT, flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
