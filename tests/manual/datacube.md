# Manual tests — real satellite data (datacube builder)

Validates `rise_eo.datacube.builder.build_datacube` against the **real, multi-CRS**
`satellite_benchmark/` tiles, with the output inspected **visually in QGIS** (LLMs
are unreliable on GeoTIFFs, so green unit tests are not enough — see issue
#8). Unit tests (`tests/test_datacube_*.py`) cover the logic on synthetic tiles; this
guide proves it on genuine bytes, real CRS, real nodata — and specifically that the
**single-CRS merge** + **reference-image resampling** behave on data that straddles a
UTM-zone boundary.

Data: the 1-year Ethiopia download (`benchmarks/download_report_2018_ethiopia.md`) —
catalog `satellite_benchmark/sentinel-2-l2a/catalog.parquet` (579 tiles), ROI
`shapefiles/s2grid=165bca4.geojson`. The ROI sits on **36°E**, so it pulls S2 tiles in
**both EPSG:32636 (zone 36) and EPSG:32637 (zone 37)** — the whole reason this ROI
exists. Bands present: **B04, B08, B8A, SCL** (B08 = 10 m reference; B8A native 20 m
exercises resampling; no B02/B03, so no true-color — we use B08 / FCC / NDVI).

> **Archive is now COG (2026-07-04, spec 14 migration).** `satellite_benchmark` band
> files are `Bxx.tif` (lossless COG + overviews), not `.jp2`, and the catalog `files`
> column reflects that. The build is format-agnostic (rasterio reads either), so every
> step below is unchanged — this just documents what you'll see on disk. Tool:
> `benchmarks/migrate_jp2_to_cog.py`.

Work top-to-bottom in one Python session. Tick a box when the result is confirmed.
Expected values below are from the reference run (window 2018-06-01 → 2018-07-10).

---

## 0. Setup

From the repo root (`fsd/`), with the dev env active (`source .venv/bin/activate`):

```python
import os, datetime
import numpy as np
import geopandas as gpd

from rise_eo.catalog.catalog import TileCatalog
from rise_eo.datacube import builder
from rise_eo.raster import images
from rise_eo.bands import modify
from rise_eo.storage import fs

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))   # holds satellite_benchmark/
CATALOG = os.path.join(ROOT, "satellite_benchmark/sentinel-2-l2a/catalog.parquet")
ROI = os.path.join(ROOT, "shapefiles/s2grid=165bca4.geojson")
OUT = os.path.join(os.getcwd(), "tests/outputs/datacube")
os.makedirs(OUT, exist_ok=True)

roi = gpd.read_file(ROI)
start = datetime.datetime(2018, 6, 1)
end   = datetime.datetime(2018, 7, 10)   # short window -> a small, few-timestamp cube
print("roi crs:", roi.crs, "| catalog exists:", os.path.exists(CATALOG))
```

- [ ] `roi crs: EPSG:4326` and `catalog exists: True`

> **Why a 40-day slice of a 1-year download?** The whole cube is built **in memory**
> (per-geometry). This window × `mosaic_days=20` → just **2** mosaic timestamps. The
> full year would be ~**18** mosaic timestamps (365/20) and a proportionally larger
> cube — fine for a real run, but overkill for a visual smoke test. Widen `start`/
> `end` when you want the full series.

---

## 1. Filter the catalog + confirm it is multi-CRS

`TileCatalog.filter` does the inclusive date range + spatial-overlap filter and adds
`area_contribution` (% of the ROI union each tile covers). This is exactly what the
builder consumes.

```python
sub = TileCatalog(CATALOG).filter(roi, start, end)
zones = sorted({tid.split("_T")[1][:2] for tid in sub["id"]})   # MGRS zone number
print("tiles:", len(sub), "| dates:", sub["timestamp"].nunique(), "| MGRS zones:", zones)
print("area_contribution: %.0f..%.0f%%" %
      (sub["area_contribution"].min(), sub["area_contribution"].max()))
```

- [ ] `MGRS zones: ['36', '37']` — the subset spans **two UTM zones** (the point of
      this ROI). Expected ~`tiles: 64 | dates: 16`, `area_contribution: 20..100%`.

---

## 2. Flatten + build the datacube

`flatten_catalog` explodes each tile row into one row per raster band file (skips
`MTD_TL.xml`). `build_datacube` then runs the full pipeline: missing-check →
load/crop → pick `dst_crs` (max mean area contribution) → merge B08 → resample all to
that reference grid → stack → SCL cloud-mask → drop SCL → median-mosaic → save.

```python
flat = builder.flatten_catalog(sub)
print("band-rows:", len(flat), "| bands:", sorted(flat["band"].unique()))

builder.build_datacube(
    catalog_subset=flat, shape_gdf=roi, startdate=start, enddate=end,
    bands=["B04", "B08", "B8A", "SCL"], mosaic_days=20,
    export_folderpath=OUT, if_missing_files="warn",   # ROI edges -> partial coverage
)
```

- [ ] `bands: ['B04', 'B08', 'B8A', 'SCL']`, band-rows ~`256`. `build_datacube`
      finishes and writes `datacube.npy` + `metadata.pickle.npy` to `OUT`
      (a `Missing files warning` about area/time is expected and fine).

> `if_missing_files="warn"` because a ROI on a tile edge rarely has 100% coverage or a
> perfectly even cadence; `"raise_error"` (the default) is for strict production runs.

---

## 3. Inspect the result

```python
dc = fs.load_npy(os.path.join(OUT, "datacube.npy"))
md = fs.load_npy(os.path.join(OUT, "metadata.pickle.npy"), allow_pickle=True)[()]
prof, bands = md["geotiff_metadata"], md["bands"]
print("datacube:", dc.shape, dc.dtype, "| bands:", bands, "| ts:", len(md["timestamps"]))
print("dst_crs:", prof["crs"], "| HxW:", prof["height"], prof["width"])
```

- [ ] `datacube: (2, 554, 533, 3) uint16`, `bands: ['B04', 'B08', 'B8A']` (SCL dropped),
      `dst_crs: EPSG:32636` (zone 36 won on mean area). **All 3 bands share one HxW** —
      proof B8A (native 20 m) was resampled onto the 10 m B08 grid.

---

## 4. Export GeoTIFFs for QGIS

The datacube is a bare array; rebuild `(data, profile)` from a timestamp slice + the
saved reference profile and write GeoTIFFs.

```python
t = 0  # first mosaic timestamp

def band_dp(name):
    arr = dc[t, :, :, bands.index(name)][np.newaxis, ...]        # (1, H, W)
    p = {**prof, "driver": "GTiff", "count": 1, "dtype": str(arr.dtype), "nodata": 0}
    return arr, p

# (a) B08 single band — geolocation / merge / coverage
images.save_geotiff(f"{OUT}/165bca4_B08.tif", *band_dp("B08"))

# (b) False-colour (B08/B8A/B04 -> R/G/B), vegetation reads bright red
images.save_rgb_geotiff(f"{OUT}/165bca4_FCC_8bit.tif",
                        [band_dp("B08"), band_dp("B8A"), band_dp("B04")], scale_max=4000)

# (c) NDVI via the 5-D bands.modify contract (as the notebooks do)
b04 = dc[t, :, :, bands.index("B04")].astype(float)
b08 = dc[t, :, :, bands.index("B08")].astype(float)
cube = np.stack([b04, b08], axis=-1)[np.newaxis, ...]            # (1, H, W, 2)
b5 = modify.expand_datacube(cube)                               # (1, 1, H, W, 2)
with np.errstate(divide="ignore", invalid="ignore"):
    out, obi = modify.compute_bands(b5, {"B04": 0, "B08": 1}, ["NDVI"])
ndvi = out[0, 0, :, :, obi["NDVI"]]
ndvi32 = np.where(np.isfinite(ndvi), ndvi, np.nan).astype("float32")[np.newaxis, ...]
images.save_geotiff(f"{OUT}/165bca4_NDVI.tif", ndvi32,
                    {**prof, "driver": "GTiff", "dtype": "float32",
                     "nodata": float("nan"), "count": 1})

v = ndvi32[np.isfinite(ndvi32)]
print("NDVI %.3f..%.3f mean %.3f" % (v.min(), v.max(), v.mean()))
```

- [ ] `NDVI` prints a plausible range (reference: `0.034..0.607 mean 0.236`).

---

## 5. QGIS validation checklist

Open the three GeoTIFFs (all in `dst_crs` EPSG:32636). Confirm each geospatial goal:

- [x] **Geolocation** — `165bca4_B08.tif` lands over the `s2grid=165bca4` shape
      (load the geojson too). CRS reads EPSG:32636. *(Position is correct; the raster
      does not tightly hug the shape edges — see the edge-tightness nit below. That's
      a known, deferred cosmetic artifact, not a geolocation error, so this box is
      about "is it in the right place," which it is.)*
- [x] **Single-CRS merge (the key test)** — the raster is **seamless across the 36°E
      boundary**: no gap, no doubling, no visible seam where zone-37 tiles were
      reprojected into zone 36. Coverage fills the ROI (edge nodata is fine; ~83% of
      the slice is non-zero).
- [ ] **Multi-tile same-acquisition merge (spec 20)** — since spec 20, when several tiles
      of the *same* acquisition cover the shape (it straddles an MGRS tile boundary), **all**
      are merged onto the reference grid (was: one kept → interior nodata holes; BUG-002, in `BUGS.md`
      at tag `docs-archive-2026`). Re-run this cube post-fix and confirm coverage did **not** drop and any prior
      interior holes are gone. The hardest stress case is a **tile-row-boundary corner** (4
      tiles/acquisition) — exercised by the spec-19 demo (`demos/`, grid `165b09c`: 0.6 % → 82.8 %).
- [x] **Reference-image resampling** — in `165bca4_FCC_8bit.tif` the three bands are
      **pixel-aligned** (no colour-fringing/offset), confirming B8A (20 m) landed on
      the B08 (10 m) grid.
- [x] **Cloud-mask + mosaic** — apply a singleband-pseudocolor ramp to
      `165bca4_NDVI.tif` (e.g. 0 → 0.7): vegetated fields read high, bare/water low,
      and there are **no cloud blobs / few nodata holes** (SCL masking + median mosaic
      did their job).

---

## Known nit — edge tightness (deferred, TODO #8)
The output rectangle does **not** hug the ROI's edges: it extends beyond the shape's
bbox (measured: left −4 m, right +55 m, bottom −48 m, **top +113 m**) with the overhang
being almost pure nodata (top/bottom edge strips ~98–100% zero). Along the bottom/left
slanted edges the valid-pixel staircase doesn't touch every point of the shape line.
**Why:** (1) the pipeline crops to the shape *once* up front, then reproject→merge→
resample onto a reference grid and **never re-crops to the polygon**, so the extent is
the merged bbox + a nodata halo; (2) the ROI is a **rotated parallelogram** (fills only
0.86 of its UTM bbox), and a north-up raster's edges can't coincide with slanted shape
edges. Note two *different* things get conflated here: the **file extent** (the bbox, which
carries the nodata halo above) vs. the **visible data boundary** QGIS actually draws
(nodata is hidden). The visible over/undershoot you see is from `all_touched=True`
(grabs pixels the polygon merely grazes → data ~1 px past the true edge), the rotated
staircase, and **no final clip to the polygon** — so the data edge follows the crop
footprints, not the shape. This is **legacy behaviour** (faithful port) and cosmetic —
pixels are correctly geolocated and `flatten()` drops all-nodata pixels, so it never
reaches training data.
A final `rasterio.mask` re-crop to the shape would trim the halo (tested: 554×533 →
539×528) but not the diagonal staircase; it's a deliberate behaviour change, so it's
parked under TODO #8 pending a decision.

## Notes
- **Mosaic timestamps are calendar boundaries (spec 15, 2026-07-05).** `median_mosaic`
  now defaults to `mosaic_scheme="calendar"`: the 2 output timestamps here are the
  **window-start dates** `2018-06-01` and `2018-06-21` (not the first-acquisition dates).
  The pixel groupings/medians for this dense window are unchanged, so the NDVI reference
  values above still hold. Pass `mosaic_scheme="acquisition"` for the legacy labeling.
- Outputs go to `tests/outputs/datacube/` (gitignored) — for QGIS, not commits.
- The **whole cube is built in memory** per geometry. Fine for a short window on this
  ROI; a full year / large ROI is where the future `xarray+zarr` chunked artifact
  (TODO #13) and the Snakemake `workflows` layer (module #6) come in.
- `dst_crs` is chosen in-situ (max mean `area_contribution`), never passed in — see
  `specs/03-datacube.md` and the geospatial rationale in `../../CLAUDE.md`.
