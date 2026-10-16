"""Spec 13 · Part A — build a COG mirror of the first N months of satellite_benchmark.

Converts the native CDSE JP2 tiles into base COGs (DEFLATE + PREDICTOR=2, tiled 512,
NO overviews — the build reads full-res and never uses them) and writes a parallel
`catalog.parquet` pointing at them, so the throughput harness can A/B JP2 vs COG by
just `--catalog`-switching (no `src/rise_eo/` change). Also emits the space-vs-time
*storage* side of the experiment: JP2 -> base COG -> COG+overviews (overview row
estimated from a sample, since we don't materialise overviews for the dataset).

Lossless: DEFLATE + PREDICTOR are reversible; S2 JP2 declares NBITS=15 in a uint16
container, so PREDICTOR=2 needs `NBITS=16` (promotes the *declared* bit depth only —
pixel values are unchanged). The script asserts bit-identical samples.

Run:  python -m benchmarks.prep_cog_dataset            # first 4 months
      python -m benchmarks.prep_cog_dataset --months 2 --jobs 6
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rise_eo.raster.cog import to_cog
from rise_eo.storage import fs

ROOT = str(Path(__file__).resolve().parents[2])  # workspace root: the folder holding fsd/ and the archives
JP2_ROOT_NAME = "satellite_benchmark"
COG_ROOT_NAME = "satellite_benchmark_cog"
JP2_CATALOG = f"{ROOT}/{JP2_ROOT_NAME}/sentinel-2-l2a/catalog.parquet"
COG_CATALOG = f"{ROOT}/{COG_ROOT_NAME}/sentinel-2-l2a/catalog.parquet"
STORAGE_MD = f"{ROOT}/fsd/benchmarks/cog_vs_jp2_storage.md"
STORAGE_JSON = f"{ROOT}/fsd/benchmarks/cog_vs_jp2_storage.json"

MONTHS = 4
DISK_MARGIN = 1.2       # require free >= estimate * this before bulk convert
SAMPLE_PRODUCTS = 2     # products used for the pre-flight ratio + overview delta


# --- pure helpers (unit-tested) ----------------------------------------------

def first_n_months(df: pd.DataFrame, months: int):
    """Rows whose timestamp is within `months` of the earliest acquisition.
    Returns (subset, first_ts, cutoff_ts)."""
    first = pd.Timestamp(df["timestamp"].min()).normalize()
    cutoff = first + pd.DateOffset(months=months)
    return df[df["timestamp"] < cutoff].copy(), first, cutoff


def jp2_to_tif(fname: str) -> str:
    """`B04.jp2` -> `B04.tif`; non-jp2 (e.g. MTD_TL.xml) unchanged."""
    return fname[:-4] + ".tif" if fname.endswith(".jp2") else fname


def rewrite_paths(local_folderpath: str, files, jp2_name: str, cog_name: str):
    """Map one catalog row's (local_folderpath, files) from the JP2 tree to the COG
    tree: swap the dataset-root dir in the absolute path and `.jp2`->`.tif`. PURE."""
    new_folder = local_folderpath.replace(
        f"{os.sep}{jp2_name}{os.sep}", f"{os.sep}{cog_name}{os.sep}"
    )
    new_files = ",".join(jp2_to_tif(f) for f in str(files).split(","))
    return new_folder, new_files


def rewrite_catalog(gdf: gpd.GeoDataFrame, jp2_name: str, cog_name: str):
    """Copy the catalog with `local_folderpath`/`files` pointed at the COG tree. PURE."""
    out = gdf.copy()
    folders, files = [], []
    for _, r in gdf.iterrows():
        nf, nfl = rewrite_paths(r["local_folderpath"], r["files"], jp2_name, cog_name)
        folders.append(nf)
        files.append(nfl)
    out["local_folderpath"] = folders
    out["files"] = files
    return out


def summarize_storage(by_band: dict) -> dict:
    """From {band: {jp2, cog, cog_ovr}} bytes -> totals + ratios. PURE."""
    tj = sum(v["jp2"] for v in by_band.values())
    tb = sum(v["cog"] for v in by_band.values())
    to = sum(v.get("cog_ovr", 0) for v in by_band.values())
    return {
        "per_band": by_band,
        "total_jp2": tj, "total_cog": tb, "total_cog_ovr": to,
        "cog_ratio": round(tb / tj, 3) if tj else None,
        "cog_ovr_ratio": round(to / tj, 3) if tj else None,
        "overview_delta_pct": round((to - tb) / tb * 100, 1) if tb else None,
    }


# --- conversion (raster I/O — the documented rasterio exception) --------------
# The COG creation profile lives in rise_eo.raster.cog.to_cog (spec 14); this benchmark
# just pins OVERVIEWS="NONE" (base COG — the build never reads overviews).

def _convert_file(src: str, dst: str, overviews: str = "NONE") -> int:
    """JP2 -> COG (or copy a non-raster sidecar). Returns bytes written."""
    if src.endswith(".jp2"):
        return to_cog(src, dst, overviews=overviews)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    return os.path.getsize(dst)


def _convert_one(task):
    src, dst = task
    return _convert_file(src, dst)


def _tasks(sub: gpd.GeoDataFrame):
    """(src, dst) pairs for every file in the subset (jp2 -> tif, sidecars copied)."""
    out = []
    for _, r in sub.iterrows():
        new_folder, _ = rewrite_paths(r["local_folderpath"], r["files"],
                                      JP2_ROOT_NAME, COG_ROOT_NAME)
        for f in str(r["files"]).split(","):
            out.append((os.path.join(r["local_folderpath"], f),
                        os.path.join(new_folder, jp2_to_tif(f))))
    return out


def _jp2_bytes_by_band(sub: gpd.GeoDataFrame) -> dict:
    """Actual on-disk JP2 bytes per band across the subset (fast; getsize only)."""
    by = {}
    for _, r in sub.iterrows():
        for f in str(r["files"]).split(","):
            if not f.endswith(".jp2"):
                continue
            p = os.path.join(r["local_folderpath"], f)
            by[f[:-4]] = by.get(f[:-4], 0) + os.path.getsize(p)
    return by


# --- pre-flight: sample ratio + disk check -----------------------------------

def _sample_ratios(sub: gpd.GeoDataFrame, tmpdir: str, n_products: int):
    """Convert a few products (base + overview variants) -> per-band cog/jp2 and
    cog_ovr/cog byte ratios, so we can size the full run before writing the bulk."""
    jp2, cog, ovr = {}, {}, {}
    for _, r in sub.head(n_products).iterrows():
        for f in str(r["files"]).split(","):
            if not f.endswith(".jp2"):
                continue
            band = f[:-4]
            src = os.path.join(r["local_folderpath"], f)
            b = os.path.join(tmpdir, f"{band}_{r['id']}.tif")
            o = os.path.join(tmpdir, f"{band}_{r['id']}_o.tif")
            _convert_file(src, b, overviews="NONE")
            to_cog(src, o, overviews="AUTO")
            jp2[band] = jp2.get(band, 0) + os.path.getsize(src)
            cog[band] = cog.get(band, 0) + os.path.getsize(b)
            ovr[band] = ovr.get(band, 0) + os.path.getsize(o)
    cog_ratio = {b: cog[b] / jp2[b] for b in jp2}
    ovr_ratio = {b: ovr[b] / cog[b] for b in cog}
    return cog_ratio, ovr_ratio


def _preflight(sub, tmpdir):
    jp2_by_band = _jp2_bytes_by_band(sub)
    cog_ratio, ovr_ratio = _sample_ratios(sub, tmpdir, SAMPLE_PRODUCTS)
    est_cog = sum(jp2_by_band[b] * cog_ratio.get(b, 1.3) for b in jp2_by_band)
    free = shutil.disk_usage(ROOT).free
    need = est_cog * DISK_MARGIN
    g = lambda x: f"{x / 2**30:.1f} GiB"  # noqa: E731
    print(f"[preflight] JP2 {g(sum(jp2_by_band.values()))} | est base-COG {g(est_cog)} "
          f"(x{est_cog / max(sum(jp2_by_band.values()), 1):.2f}) | free {g(free)}", flush=True)
    if free < need:
        raise SystemExit(
            f"[preflight] ABORT: need ~{g(need)} free (est x{DISK_MARGIN} margin), "
            f"only {g(free)} available. Reduce --months and retry."
        )
    return jp2_by_band, cog_ratio, ovr_ratio


# --- losslessness check ------------------------------------------------------

def _check_lossless(tasks, n=3):
    """Assert a spread of converted COGs are bit-identical to their JP2 sources."""
    jp2_tasks = [(s, d) for s, d in tasks if s.endswith(".jp2")]
    step = max(len(jp2_tasks) // n, 1)
    checked = 0
    for src, dst in jp2_tasks[::step][:n]:
        with rasterio.open(src) as s, rasterio.open(dst) as d:
            if not np.array_equal(s.read(), d.read()):
                raise SystemExit(f"[lossless] MISMATCH: {dst} != {src}")
        checked += 1
    print(f"[lossless] OK — {checked} sampled COGs bit-identical to JP2", flush=True)


# --- report ------------------------------------------------------------------

def _write_storage_report(summary, meta):
    import json

    with fs.open(STORAGE_JSON, "w") as f:
        json.dump({"meta": meta, "storage": summary}, f, indent=2, default=str)

    g = lambda x: f"{x / 2**30:.2f}" if x else "0.00"  # noqa: E731
    L = ["# COG vs JP2 — storage (spec 13)\n",
         f"_generated {meta['run_utc']} · first {meta['months']} months · "
         f"{meta['n_products']} products_\n",
         "Base COG = DEFLATE + PREDICTOR=2, tiled 512, **no overviews** (the build never "
         "reads overviews). The COG+overviews column is the *estimated* extra cost if these "
         "same files were also made XYZ-tiling-ready (e.g. for TiTiler) — measured from a "
         "sample, not materialised.\n\n",
         "| band | JP2 (GiB) | base COG (GiB) | COG ratio | +overviews (GiB, est) |\n",
         "|---|---|---|---|---|\n"]
    for band, v in sorted(summary["per_band"].items()):
        ratio = v["cog"] / v["jp2"] if v["jp2"] else 0
        L.append(f"| {band} | {g(v['jp2'])} | {g(v['cog'])} | {ratio:.2f}× | "
                 f"{g(v.get('cog_ovr', 0))} |\n")
    L.append(f"| **total** | **{g(summary['total_jp2'])}** | **{g(summary['total_cog'])}** "
             f"| **{summary['cog_ratio']}×** | **{g(summary['total_cog_ovr'])}** |\n")
    yr = 12 / meta["months"]
    L.append(f"\n- **Base COG costs {summary['cog_ratio']}× the JP2 storage** "
             f"(+{round((summary['cog_ratio'] - 1) * 100)}%). Overviews would add "
             f"~{summary['overview_delta_pct']}% on top (tiling-only; the build can't use them).\n")
    L.append(f"- Extrapolated to the full year (×{yr:.0f}): JP2 ≈ {g(summary['total_jp2'] * yr)} "
             f"GiB, base COG ≈ {g(summary['total_cog'] * yr)} GiB.\n")
    L.append("- **Lossless** (DEFLATE+PREDICTOR, NBITS=16 promotion): pixels are bit-identical.\n")
    with fs.open(STORAGE_MD, "w") as f:
        f.write("".join(L))
    print(f"[report] storage -> {STORAGE_MD}", flush=True)


# --- main --------------------------------------------------------------------

def _check_cog_driver():
    """Fail early (with guidance) if the .venv GDAL lacks the COG driver."""
    import tempfile

    from rasterio.transform import from_origin

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "probe.tif")
        arr = np.zeros((1, 8, 8), dtype="uint16")
        try:
            with rasterio.open(p, "w", driver="COG", height=8, width=8, count=1,
                               dtype="uint16", crs="EPSG:32637",
                               transform=from_origin(0, 0, 10, 10)) as dst:
                dst.write(arr)
        except Exception as e:  # pragma: no cover - environment guard
            raise SystemExit(
                "[preflight] GDAL COG driver unavailable in .venv "
                f"(GDAL {rasterio.__gdal_version__}): {e}. Need GDAL >= 3.1."
            ) from e


def main(argv=None):
    import datetime
    import json
    import tempfile

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--months", type=int, default=MONTHS)
    ap.add_argument("--jobs", type=int, default=4, help="parallel conversion workers")
    ap.add_argument("--force", action="store_true", help="reconvert even if COG exists")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap #products (smoke/incremental convert)")
    args = ap.parse_args(argv)

    _check_cog_driver()
    cat = gpd.read_parquet(JP2_CATALOG)
    sub, first, cutoff = first_n_months(cat, args.months)
    if args.limit is not None:
        sub = sub.head(args.limit)
    print(f"[setup] first {args.months} months: {first.date()} -> {cutoff.date()} | "
          f"{len(sub)} products | suggested timed window: {first.date()} -> "
          f"{(first + pd.DateOffset(weeks=6)).date()}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        jp2_by_band, _cog_ratio, ovr_ratio = _preflight(sub, tmp)

    tasks = _tasks(sub)
    if not args.force:
        tasks = [(s, d) for s, d in tasks if not os.path.exists(d)]
    print(f"[convert] {len(tasks)} files to write ({args.jobs} workers)...", flush=True)

    t0 = time.perf_counter()
    done = bytes_done = 0
    total = len(tasks)
    if total:
        with mp.Pool(args.jobs) as pool:
            for sz in pool.imap_unordered(_convert_one, tasks):
                done += 1
                bytes_done += sz
                if done % 20 == 0 or done == total:
                    el = time.perf_counter() - t0
                    eta = el / done * (total - done)
                    print(f"[convert] {done}/{total} | {bytes_done / 2**30:.1f} GiB | "
                          f"{el:.0f}s | ETA ~{eta:.0f}s", flush=True)

    _check_lossless(_tasks(sub))

    cog_cat = rewrite_catalog(sub, JP2_ROOT_NAME, COG_ROOT_NAME)
    fs.makedirs(os.path.dirname(COG_CATALOG))
    fs.write_parquet(COG_CATALOG, cog_cat)
    print(f"[catalog] COG catalog ({len(cog_cat)} rows) -> {COG_CATALOG}", flush=True)

    # actual base-COG sizes on disk + estimated overview column
    cog_by_band = {}
    for _, r in cog_cat.iterrows():
        for f in str(r["files"]).split(","):
            if not f.endswith(".tif"):
                continue
            cog_by_band[f[:-4]] = cog_by_band.get(f[:-4], 0) + os.path.getsize(
                os.path.join(r["local_folderpath"], f))
    by_band = {b: {"jp2": jp2_by_band[b], "cog": cog_by_band.get(b, 0),
                   "cog_ovr": round(cog_by_band.get(b, 0) * ovr_ratio.get(b, 1.35))}
               for b in jp2_by_band}
    summary = summarize_storage(by_band)
    meta = {"run_utc": datetime.datetime.utcnow().isoformat() + "Z", "months": args.months,
            "n_products": int(len(sub)), "window_start": str(first.date()),
            "window_cutoff": str(cutoff.date()), "cog_catalog": os.path.relpath(COG_CATALOG, ROOT)}
    _write_storage_report(summary, meta)
    print("[done] " + json.dumps({"cog_ratio": summary["cog_ratio"],
                                  "ovr_delta_pct": summary["overview_delta_pct"]}))


if __name__ == "__main__":
    mp.freeze_support()
    main()
