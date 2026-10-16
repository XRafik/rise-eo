"""End-to-end LOCAL-completeness gate (spec 23): download → jp2→COG → datacube → flatten → train →
bundle → ROI build+infer → COG/STAC/merged, on FRESH real CDSE data for an Austria ROI.

This is the go-to local run (see demos/E2E_AUSTRIA.md). It is a **reusable template**: point it at a
different `--roi` / `--train` (+ `--id-col`/`--label-col`) and it runs unchanged — **including
cross-UTM-zone ROIs** (merge="reproject" reprojects per-cell outputs to one CRS). The model
(adapter + bands) is the model-developer-owned part (adapters.py / the bundling guide in the doc).

Steps (each prints a ✓ check; a failed check aborts — this is a gate, not a best-effort demo):
  0. preflight (creds, inputs)          4. train RF + save model bundle
  1. ROI → S2 grid cells (+ CRS report) 5. run_inference(roi=…) → COG/cell + STAC + merged map
  2. DOWNLOAD (probe + resume + timing)  6. plots (per-class NDVI + crop map)
  3. training data (create_training_data) 7. timing report + cost_model + ETA estimator

Download is a SEPARATE step (CDSE serves quota'd .jp2 that rise-eo converts to COG); compute reads the
catalog and never touches CDSE. Needs CDSE creds + the isolated venv:

    python3.11 -m venv .venv-modeldeploy
    .venv-modeldeploy/bin/pip install -e ".[dev,grid,model-example]"
    .venv-modeldeploy/bin/python demos/e2e_austria.py --creds /path/to/cdse_credentials.json
    .venv-modeldeploy/bin/python demos/e2e_austria.py --fast            # short window + small infer ROI
    .venv-modeldeploy/bin/python demos/e2e_austria.py --tiny-download   # also clip the DOWNLOAD ROI
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

# make `adapters:DemoRF` importable both in-process AND in the runner's Snakemake subprocesses
# (cores>1 reloads the bundle by module:attr in a fresh worker, which reads PYTHONPATH).
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ["PYTHONPATH"] = _HERE + os.pathsep + os.environ.get("PYTHONPATH", "")

import estimate as _estimate  # noqa: E402  (demos/estimate.py — the no-download ETA helper)
from adapters import DemoRF  # noqa: E402

import rise_eo  # noqa: E402
from rise_eo import config, grid  # noqa: E402
from rise_eo.catalog.catalog import TileCatalog  # noqa: E402
from rise_eo.model import bundle  # noqa: E402
from rise_eo.sources import cdse  # noqa: E402
from rise_eo.sources.cdse import CdseCredentials  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))                     # workspace root

# --- the "swap these for your own region" config (also overridable by CLI; spec 23 D6) ---
ROI_FP = os.path.join(ROOT, "shapefiles/AT_ROI.geojson")
TRAIN_FP = os.path.join(ROOT, "shapefiles/AT_2018_TRAIN.geojson")
ID_COL = "fid"
LABEL_COL = "crop"

# --- the model-specific part (adapter DemoRF needs B04/B08; SCL masks; B8A is dropped) ---
# The SCL mask classes are no longer a verb parameter (spec 58 D3) -- they are declared
# on the `sentinel-2-l2a` CollectionDeclaration (`rise_eo.collections.get("sentinel-2-l2a")`),
# and happen to be exactly this list already, so this demo's behavior is unchanged.
# `collection=` (default "sentinel-2-l2a") and `properties_filter=` (spec 58 D9) are both
# left at their defaults here: S2 L2A declares `mosaic_partition=()`, so there is no
# partitioned property to select and an empty filter contributes nothing to the cube path.
BANDS = ["B04", "B08", "B8A", "SCL"]
MOSAIC_DAYS = 20

# --- download guardrails (spec 23 D6) ---
MAX_TILES = 207
MAX_CLOUDCOVER = 70

# --- crop-class colors: semantic where possible, spread across hue/lightness for separability.
#     Tweak freely; a class not listed falls back to a tab20 color. Used by BOTH plots so a class
#     has one consistent color across the crop map and the NDVI timeseries. ---
CLASS_COLORS = {
    "alfalfa_lucerne":                "#6A3D9A",  # violet — alfalfa's purple flowers
    "buckwheat":                      "#B15928",  # sienna — reddish buckwheat stems
    "grain_maize_corn_popcorn":       "#DAA520",  # goldenrod — ripe maize
    "hemp_cannabis":                  "#1B5E20",  # deep green — hemp
    "mustard":                        "#FFD500",  # bright yellow — mustard bloom
    "pasture_meadow_grassland_grass": "#7CB342",  # grass green
    "spring_common_soft_wheat":       "#F0E4B0",  # pale wheat
    "sunflower":                      "#FF7F00",  # orange — sunflower
    "winter_common_soft_wheat":       "#8B7500",  # dark khaki — winter wheat (vs pale spring)
}

OUTDIR = os.path.join(_HERE, "..", "tests/outputs/demo_e2e")       # heavy artifacts (gitignored)
# Spec 59 D12: ONE shared archive for every run (not under OUTDIR); the catalog lives at
# the archive layout (D5; see `rise_eo.archive_catalog_filepath`).
DATA_DIR = os.path.join(_HERE, "..", "tests/outputs/imagery")
FIGDIR = os.path.join(_HERE, "figures")                           # small PNGs (committable)

STEP_TIMES: list = []   # [(label, seconds), ...] — filled by timed_step, dumped at the end
COST_MODEL: dict = {}   # calibrated by the download + build + infer steps; used by the estimator


def log(msg):
    print(f"\n=== {msg} ===", flush=True)


def ok(msg):
    print(f"  ✓ {msg}", flush=True)


def _fail(msg):
    print(f"  ✗ {msg}", flush=True)
    raise SystemExit(1)


def timed_step(label, fn, *args, **kwargs):
    t = time.time()
    out = fn(*args, **kwargs)
    dt = time.time() - t
    STEP_TIMES.append((label, dt))
    print(f"  [{label}] took {dt:.1f}s", flush=True)
    return out


def _write_timing_breakdown(total):
    print("\n=== timing breakdown ===")
    print(f"{'step':<28}{'seconds':>10}{'share':>8}")
    for label, dt in STEP_TIMES:
        print(f"{label:<28}{dt:>10.1f}{100 * dt / total:>7.0f}%")
    print(f"{'TOTAL':<28}{total:>10.1f}{100:>7.0f}%")
    payload = {
        "total_seconds": round(total, 1),
        "steps": [{"step": s, "seconds": round(dt, 1)} for s, dt in STEP_TIMES],
        "cost_model": COST_MODEL,
    }
    with open(os.path.join(OUTDIR, "timings.json"), "w") as fh:
        json.dump(payload, fh, indent=2)


# --- step 0: preflight -------------------------------------------------------

def step_preflight(creds_fp):
    log("0. preflight (creds + inputs)")
    if not creds_fp or not os.path.exists(creds_fp):
        _fail(f"CDSE credentials not found: {creds_fp!r} — pass --creds or set "
              "$CDSE_CREDENTIALS_JSON (see demos/E2E_AUSTRIA.md §3).")
    creds = CdseCredentials.from_json(creds_fp)
    try:
        creds.require_s3()
    except ValueError as exc:
        _fail(str(exc))
    if creds.is_expired():
        print("  ! warning: CDSE S3 keys look expired (is_expired) — download may fail.", flush=True)
    for name, fp in [("ROI", ROI_FP), ("train", TRAIN_FP)]:
        if not os.path.exists(fp):
            _fail(f"{name} file not found: {fp}")
    ok(f"creds loaded ({creds!r}); inputs present")
    return creds


# --- step 1: tiling ----------------------------------------------------------

def step_tiling(fast, roi_run_fp):
    log("1. ROI → S2 grid cells (rise_eo.grid.roi_to_s2_grids)")
    grids = grid.roi_to_s2_grids(roi_run_fp, grid_size_km=5, scale_fact=1.1)
    grids_fp = os.path.join(OUTDIR, "inference_s2_grids.geojson")
    grids.to_file(grids_fp, driver="GeoJSON")
    print(f"  cells: {len(grids)} -> {grids_fp}")
    # report cell count — NO single-zone assertion (a cross-zone ROI is valid, D7).
    ok(f"{len(grids)} grid cells (Austria is single-zone UTM33; a cross-zone ROI would just work)")
    _plot_grids(grids, roi_run_fp)
    return grids, grids_fp


# --- step 2: download (the new heart — probe + resume + decomposed timing) ---

def step_download(creds, download_roi_fp):
    log("2. DOWNLOAD S2 L2A (probe throughput → resume-loop → jp2/COG timing)")
    os.makedirs(DATA_DIR, exist_ok=True)
    catalog_fp = rise_eo.archive_catalog_filepath(DATA_DIR, config.SATELLITE_S2L2A)

    # 2a) single-threaded baseline probe (achievable CDSE MB/s right now).
    print("  2a) baseline throughput probe (1 file, 1 thread) ...", flush=True)
    probe_mbps, probe_bytes, probe_s = cdse.probe_throughput(
        download_roi_fp, START, END, BANDS, creds, max_cloudcover=MAX_CLOUDCOVER)
    print(f"      probe: {probe_mbps:.2f} MB/s ({probe_bytes/1e6:.1f} MB in {probe_s:.1f}s)",
          flush=True)

    # 2b) idempotent, resumable download (skips files on disk; re-run to resume a bad window).
    # A download is a whole-asset byte copy: MAX_TILES bounds the granule count, not the bytes,
    # because a small ROI still pulls whole 110 km tiles (~122 MB per band per tile). Collections
    # with larger scenes cost far more -- Sentinel-1 RTC is ~3.7 GB per 250 km scene -- so on any
    # collection but this one, size the window before running it.
    print("  2b) download_resume (cog=True: jp2 → COG on arrival) ...", flush=True)
    cat = TileCatalog(catalog_fp)
    results = cdse.download_resume(
        download_roi_fp, START, END, BANDS, DATA_DIR, cat, creds,
        max_tiles=MAX_TILES, max_cloudcover=MAX_CLOUDCOVER, progress=True, cog=True)
    agg = cdse.sum_results(results)

    granules = len(cat.read())
    wall = max(agg.elapsed_s, 1e-9)
    gb = agg.bytes_downloaded / 1e9
    # aggregate (wall) rate = the honest all-streams throughput to compare vs the single-stream
    # probe (spec 25/26); per-stream rate is the thread-summed average, shown only for context.
    eff_mbps = agg.bytes_downloaded / 1e6 / max(agg.transfer_wall_seconds, 1e-9)   # aggregate
    per_stream_mbps = agg.bytes_downloaded / 1e6 / max(agg.transfer_seconds, 1e-9)  # per stream
    print(f"\n  transfer : {agg.transfer_wall_seconds:7.1f}s  ({gb:.2f} GB, "
          f"{eff_mbps:.1f} MB/s aggregate / {per_stream_mbps:.1f} per stream)")
    print(f"  convert  : {agg.convert_seconds:7.1f}s  (jp2 → COG, {agg.successful_count} files)")
    print(f"  wall     : {wall:7.1f}s  ({granules} granules, {len(results)} pass(es))")
    if probe_mbps and eff_mbps:
        verdict = ("CDSE/link-bound" if eff_mbps >= 0.75 * probe_mbps
                   else "local contention / concurrency-bound")
        print(f"  probe {probe_mbps:.1f} MB/s vs effective {eff_mbps:.1f} MB/s (aggregate) -> {verdict}")

    if agg.circuit_tripped or agg.failed_count:
        print("  ! CDSE was flaky (BUG-001): re-run this script to resume the remainder.", flush=True)
    if granules < 1:
        _fail("no granules downloaded — check creds / ROI / window.")

    # calibrate the cost_model for the estimator (best-effort; a fresh download makes it exact).
    if agg.bytes_downloaded > 0 and granules > 0:
        COST_MODEL.update({
            "transfer_mb_per_s": round(eff_mbps, 2),          # aggregate (wall) → realistic ETAs
            "per_stream_mb_per_s": round(per_stream_mbps, 2),  # per-stream, diagnostic only
            "probe_mb_per_s": round(probe_mbps, 2),
            "convert_s_per_file": round(agg.convert_seconds / max(agg.successful_count, 1), 3),
            "mean_bytes_by_band": {b: int(v / granules) for b, v in agg.bytes_by_band.items()},
            "t_calib": rise_eo.compute_n_timestamps(START, END, MOSAIC_DAYS),
        })
    ok(f"{granules} granules in catalog; every file on disk (re-run resumes)")
    return catalog_fp, granules


# --- step 3: training data ---------------------------------------------------

def step_training_data(fast, catalog_fp, adapter, clip_fp=None):
    log("3. training data (create_training_data → features.npy; builds cubes over fields, flattens)")
    fields_fp = TRAIN_FP
    if fast or clip_fp is not None:
        g = gpd.read_file(TRAIN_FP)
        if clip_fp is not None:
            # --tiny-download: only downloaded a small slice, so keep only fields inside it.
            box = gpd.read_file(clip_fp).to_crs(g.crs)
            g = g[g.intersects(box.unary_union)].reset_index(drop=True)
            if len(g) == 0:
                _fail("no training fields fall inside the --tiny-download slice; "
                      "enlarge the slice (edit _center_roi half_deg) or drop --tiny-download.")
        if fast:
            n = min(3, g.groupby(LABEL_COL).size().min())
            g = g.groupby(LABEL_COL, group_keys=False).sample(n=n, random_state=7)
        fields_fp = os.path.join(OUTDIR, "train_fields_subset.geojson")
        g.to_file(fields_fp, driver="GeoJSON")
        print(f"  training fields: {len(g)} ({'clipped ' if clip_fp else ''}"
              f"{'sampled' if fast else 'all-in-slice'})")
    t0 = time.time()
    td = rise_eo.create_training_data(
        label_polygons=fields_fp, catalog_filepath=catalog_fp,
        startdate=START, enddate=END, mosaic_days=MOSAIC_DAYS, bands=BANDS,
        id_col=ID_COL, label_col=LABEL_COL,
        export_folderpath=os.path.join(OUTDIR, "training_data"),
        run_folderpath=os.path.join(OUTDIR, "training_run"),
        adapter=adapter, cores=CORES,
        # The MODELLING UNIT, not a size trick (run-book 40): one `np.nanmedian` row per
        # labelled field instead of one per pixel. Labels are field-level, so training
        # per-pixel leaks a field's own pixels across the train/test split -- that is the
        # difference between a flattering ~0.70 and the honest field-wise ~0.29. It also
        # keeps the bundle small, which matters because inference fetches it to every node.
        # Training-only: inference still applies DemoRF's transform PER PIXEL, so the crop
        # map stays per-pixel. Matches `e2e_austria_aml.py` (spec 40 D1 amendment A2).
        aggregate="median_per_id",
    )
    d = td.load()
    n_cubes = len(pd.read_csv(os.path.join(OUTDIR, "training_run", "input.csv")))
    if n_cubes:
        COST_MODEL["build_s_per_cube"] = round((time.time() - t0) / n_cubes, 3)
    feats, T = d["features"], rise_eo.compute_n_timestamps(START, END, MOSAIC_DAYS)
    print(f"  features: {feats.shape} {td.feature_bands} | raw: {d['data'].shape}")
    if feats.shape[1] != T:
        _fail(f"features T={feats.shape[1]} != expected T={T}")
    ok(f"features {feats.shape}, T={T}, {len(set(d['feature_labels']))} classes")
    return td, d


# --- step 4: train + bundle --------------------------------------------------

def step_train(d, adapter):
    log("4. train RandomForest (sklearn — rise-eo does not train) + save model bundle")
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    X = d["features"].reshape(len(d["features"]), -1)
    y_raw = d["feature_labels"]
    keep = ~np.isnan(X).any(axis=1)
    X, y_raw = X[keep], np.asarray(y_raw)[keep]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42).fit(X, y)
    model_fp = os.path.join(OUTDIR, "rf.joblib")
    joblib.dump((clf, le), model_fp)

    adapter.artifacts = {"model": model_fp}
    bundle_dir = bundle.save(adapter, {"model": model_fp}, os.path.join(OUTDIR, "bundle"))
    spec = bundle.read_spec(bundle_dir)
    print(f"  trained on {len(X)} samples, {len(le.classes_)} classes -> {model_fp}")
    print(f"  bundle: adapter={spec['adapter']} n_timestamps={spec['n_timestamps']} "
          f"required_bands={spec['required_bands']}")
    if spec["n_timestamps"] != rise_eo.compute_n_timestamps(START, END, MOSAIC_DAYS):
        _fail("bundle n_timestamps disagrees with the window T")
    ok(f"model-free bundle spec reads back ({bundle_dir})")
    return bundle_dir, list(le.classes_)


# --- step 5: ROI inference (build + infer + save, one call) ------------------

def step_inference(bundle_dir, catalog_fp, roi_run_fp, grids):
    log("5. run_inference(roi=…) → COG per cell + STAC + merged crop map")
    t0 = time.time()
    # merge="reproject": cross-UTM-zone-safe (lossless for single-zone Austria — no cell changes
    # zone). rise-eo tiles the ROI, builds each cell's cube in its own CRS, runs the model, writes a COG
    # (spec 21); cores=INFER_CORES fans out via the Snakemake runner; cubes_per_task amortises the RF
    # load. Idempotent: a re-run skips cells whose output.tif exists.
    result = rise_eo.run_inference(
        model=bundle_dir, output_folderpath=os.path.join(OUTDIR, "model_outputs"),
        roi=roi_run_fp, catalog_filepath=catalog_fp,
        startdate=START, enddate=END, mosaic_days=MOSAIC_DAYS, bands=BANDS,
        merge="reproject",
        cores=INFER_CORES, cubes_per_task=20, overwrite=False, progress=True,
    )
    n = len(result.output_filepaths)
    if n:
        COST_MODEL["infer_s_per_cube"] = round((time.time() - t0) / n, 3)
    print(f"  outputs: {n} COGs | stac: {result.stac_catalog_filepath}")
    print(f"  merged (single-CRS display map): {result.merged_filepath}")
    if n < 1:
        _fail("no per-cell outputs produced")
    ok(f"{n} per-cell COGs + STAC + merged map (re-run is idempotent: 'Nothing to be done')")
    return result, result.merged_filepath


# --- step 6: plots -----------------------------------------------------------

def step_plots(d, merged_fp, classes):
    log("6. plots: per-class NDVI timeseries + categorical model-output map")
    _plot_ndvi_timeseries(d)
    if merged_fp:
        _plot_crop_map(merged_fp, classes)


# --- step 7: report + estimator ----------------------------------------------

def step_report(granules, grids):
    log("7. cost_model + no-download ETA estimator (spec 23 §7)")
    print("  cost_model (calibrated this run):", json.dumps(COST_MODEL, indent=2))
    if not COST_MODEL.get("mean_bytes_by_band"):
        print("  (re-run without existing imagery to (re)calibrate mean bytes/band)")
        return
    T = rise_eo.compute_n_timestamps(START, END, MOSAIC_DAYS)
    self_est = _estimate.estimate_from_counts(
        granules=granules, cells=len(grids), t=T, bands=BANDS, cost_model=COST_MODEL)
    print("  sanity — estimate for THIS run's own counts:", json.dumps(self_est, indent=2))
    print("  to estimate another region without downloading it, e.g. full France:")
    print("      from estimate import estimate_run")
    print("      estimate_run('FR_ROI.geojson', START, END, BANDS, creds=creds,")
    print("                   cost_model=cost_model, max_cloudcover=70)")


# --- plotting helpers --------------------------------------------------------

def _plot_grids(grids, roi_run_fp):
    import matplotlib.pyplot as plt
    roi = gpd.read_file(roi_run_fp)
    fig, ax = plt.subplots(figsize=(8, 7))
    roi.boundary.plot(ax=ax, color="black", linewidth=1.5, label="ROI")
    grids.boundary.plot(ax=ax, color="tab:blue", linewidth=0.6)
    ax.set_title(f"ROI → {len(grids)} S2 grids (5 km, 10% overlap, clipped)")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    _save(fig, "s2_grids.png")


def _plot_ndvi_timeseries(d):
    import matplotlib.pyplot as plt
    feats, labels = d["features"], np.asarray(d["feature_labels"])
    ts = d["metadata"]["timestamps"]
    fb = d["metadata"]["feature_bands"]
    ndvi = feats[:, :, fb.index("NDVI")]
    fig, ax = plt.subplots(figsize=(11, 6))
    for lab in sorted(set(labels)):
        med = np.nanmedian(ndvi[labels == lab], axis=0)
        ax.plot(ts, med, marker="o", markersize=3, linewidth=1.2, label=lab,
                color=CLASS_COLORS.get(lab))
    ax.set_title("Per-class median NDVI over the season (training features)")
    ax.set_ylabel("NDVI")
    ax.set_xlabel("mosaic window")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    fig.autofmt_xdate()
    _save(fig, "ndvi_timeseries.png")


def _plot_crop_map(merged_fp, classes):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib.colors import BoundaryNorm, ListedColormap

    with rasterio.open(merged_fp) as src:
        arr = src.read(1)
    arr = np.ma.masked_equal(arr, 255)
    values = list(range(len(classes)))
    fallback = plt.cm.tab20(np.linspace(0, 1, max(len(values), 1)))
    colors = [CLASS_COLORS.get(classes[i], fallback[i]) for i in values]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.array(values + [values[-1] + 1]) - 0.5, cmap.N)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(arr, cmap=cmap, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Model output — crop class map (merged over ROI)")
    handles = [mpatches.Patch(color=cmap(i), label=classes[i]) for i in values]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    _save(fig, "crop_map.png")


def _save(fig, name):
    import matplotlib.pyplot as plt
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> demos/figures/{name}")


def _center_roi(out_fp, half_deg=0.05):
    """A small central slice of the ROI (a `2*half_deg` box at the ROI centroid, clipped to it).
    Used for --fast to shrink the *inference* ROI (the full ROI is still downloaded)."""
    g = gpd.read_file(ROI_FP).to_crs(4326)
    c = g.unary_union.centroid
    box = gpd.GeoDataFrame(
        geometry=[shapely.geometry.box(c.x - half_deg, c.y - half_deg,
                                       c.x + half_deg, c.y + half_deg)], crs=4326)
    clip = gpd.overlay(box, g, how="intersection")
    clip.to_file(out_fp, driver="GeoJSON")
    return out_fp


def _single_tile_roi(out_fp, half_deg=0.03):
    """--tiny-download: a slice that lies inside a SINGLE MGRS tile, so the download is *actually*
    small. CDSE serves full ~110 km granules, so download volume scales with the number of MGRS
    **tiles**, not ROI area — and AT_ROI sits on a 4-tile junction, so a box at its centroid pulls
    all 4 tiles (~the whole ROI). Query the catalog (anonymous, no bytes), pick the tile with the
    most ROI overlap, and return a small box at that overlap's centroid clipped to the tile footprint
    (single-tile → typically ~1/N the granules)."""
    import re

    roi = gpd.read_file(ROI_FP).to_crs(4326).union_all()
    tiles = cdse.query_catalog(ROI_FP, START, END, max_cloudcover=MAX_CLOUDCOVER)
    if not len(tiles):
        _fail("no CDSE tiles intersect the ROI in-window — cannot pick a tiny slice.")
    foot: dict = {}
    for iid, geom in zip(tiles["id"], tiles.geometry):
        m = re.search(r"(\d{2}[A-Z]{3})", str(iid))
        foot.setdefault(m.group(1) if m else geom.wkt, geom)
    best = max(foot.values(), key=lambda g: g.intersection(roi).area)  # dominant MGRS tile
    c = best.intersection(roi).centroid
    box = shapely.geometry.box(c.x - half_deg, c.y - half_deg, c.x + half_deg, c.y + half_deg)
    slice_geom = box.intersection(best).intersection(roi)              # inside ONE tile AND the ROI
    gpd.GeoDataFrame(geometry=[slice_geom], crs=4326).to_file(out_fp, driver="GeoJSON")
    return out_fp


def main():
    global ROI_FP, TRAIN_FP, ID_COL, LABEL_COL, START, END, CORES, INFER_CORES
    ap = argparse.ArgumentParser(description="rise-eo end-to-end LOCAL gate (spec 23).")
    ap.add_argument("--fast", action="store_true",
                    help="quick smoke: 2-month window, 3 fields/class, small inference ROI "
                         "(still downloads the FULL ROI).")
    ap.add_argument("--tiny-download", action="store_true",
                    help="tiniest smoke: clip the DOWNLOAD ROI to a slice inside a SINGLE MGRS tile "
                         "(CDSE serves full ~110km granules, so fewer TILES = smaller download) + "
                         "train only on fields inside it. Implies --fast's window.")
    ap.add_argument("--cores", type=int, default=8,
                    help="parallelism for the training-data build (step 3). Fields are small.")
    ap.add_argument("--infer-cores", type=int, default=None,
                    help="parallelism for the inference build+infer (step 5). Each ~5x5km cube is "
                         "memory-heavy — keep LOW (default max(1, cores // 4)).")
    ap.add_argument("--creds", default=os.environ.get("CDSE_CREDENTIALS_JSON")
                    or os.path.join(ROOT, "cdse_credentials.json"),
                    help="path to cdse_credentials.json (or set $CDSE_CREDENTIALS_JSON).")
    ap.add_argument("--roi", default=ROI_FP, help="ROI GeoJSON (swap for your own region).")
    ap.add_argument("--train", default=TRAIN_FP, help="labelled training fields GeoJSON.")
    ap.add_argument("--id-col", default=ID_COL, help="field id column in --train.")
    ap.add_argument("--label-col", default=LABEL_COL, help="crop-label column in --train.")
    args = ap.parse_args()

    ROI_FP, TRAIN_FP, ID_COL, LABEL_COL = args.roi, args.train, args.id_col, args.label_col
    CORES = args.cores
    INFER_CORES = args.infer_cores if args.infer_cores is not None else max(1, CORES // 4)
    # --tiny-download implies the short window (fewer granules) and a sampled/clipped smoke.
    fast = args.fast or args.tiny_download
    START = datetime.datetime(2018, 4, 1)
    END = datetime.datetime(2018, 6, 1) if fast else datetime.datetime(2018, 9, 30)
    T = rise_eo.compute_n_timestamps(START, END, MOSAIC_DAYS)

    os.makedirs(OUTDIR, exist_ok=True)
    # ROIs: --tiny-download clips BOTH the download and the inference ROI to one small central slice
    # (and trains only on fields inside it); --fast clips only the inference ROI (full download).
    if args.tiny_download:
        slice_fp = _single_tile_roi(os.path.join(OUTDIR, "tiny_roi.geojson"))
        download_roi_fp = roi_run_fp = slice_fp
        clip_fp = slice_fp
    else:
        download_roi_fp = ROI_FP
        roi_run_fp = _center_roi(os.path.join(OUTDIR, "fast_roi.geojson")) if args.fast else ROI_FP
        clip_fp = None

    mode = "TINY" if args.tiny_download else ("FAST" if args.fast else "FULL")
    print(f"mode: {mode} | window {START:%Y-%m-%d}..{END:%Y-%m-%d} "
          f"| T={T} | cores={CORES} (train) / {INFER_CORES} (inference)"
          f"\nROI={ROI_FP} | download_roi={download_roi_fp}\nOUTDIR={os.path.abspath(OUTDIR)}")

    adapter = DemoRF()
    adapter.n_timestamps = T                              # model-determined; recorded in the bundle

    t0 = time.time()
    creds = timed_step("0_preflight", step_preflight, args.creds)
    grids, _grids_fp = timed_step("1_tiling", step_tiling, fast, roi_run_fp)
    catalog_fp, granules = timed_step("2_download", step_download, creds, download_roi_fp)
    _td, d = timed_step("3_training_data", step_training_data, fast, catalog_fp, adapter, clip_fp)
    bundle_dir, classes = timed_step("4_train_bundle", step_train, d, adapter)
    _result, merged_fp = timed_step(
        "5_run_inference", step_inference, bundle_dir, catalog_fp, roi_run_fp, grids)
    timed_step("6_plots", step_plots, d, merged_fp, classes)
    timed_step("7_report", step_report, granules, grids)
    total = time.time() - t0
    _write_timing_breakdown(total)
    log(f"DONE in {total:.0f}s — outputs under {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()
