# Manual tests — real satellite data (raster + bands)

Validates `rise_eo.raster.images` and `rise_eo.bands.modify` against a **real** locally
downloaded Sentinel-2 L2A tile, with the outputs inspected **visually in QGIS**
(LLMs are unreliable on GeoTIFFs, so green unit tests are not enough — see
issue #8). Unit tests (`tests/test_raster.py`, `tests/test_bands.py`)
cover the logic; this guide proves it works on genuine tile bytes, real CRS, real
nodata.

> **DATASET MOVED (2026-07-02).** The `satellite/` folder (tile **T33UWP**, all 13
> bands) has been **deleted**. The current real dataset is **`satellite_benchmark/`**
> — the 1-year Ethiopia multi-CRS download (ROI `s2grid=165bca4`, 579 tiles across
> EPSG:32636 & 32637, bands **B04/B08/B8A/SCL** + `MTD_TL.xml`; see
> `benchmarks/download_report_2018_ethiopia.md`). The raster+bands ops below were
> already validated in QGIS on T33UWP and are kept as reference, but the **paths and
> the TCC/FCC examples are stale**: `satellite_benchmark` has no B02/B03, so only
> **NDVI** (B04/B08) applies here. The **multi-CRS datacube** build against
> `satellite_benchmark` now has its own runbook: **`datacube.md`**.
>
> **COG archive (2026-07-04, spec 14).** `satellite_benchmark` band files are now
> `Bxx.tif` (lossless COG + overviews), not `.jp2` — the `{band}.jp2` in the historical
> code below refers to the deleted `satellite/` layout. Adapt to `.tif` if you point any
> of these ops at `satellite_benchmark`. Tool: `benchmarks/migrate_jp2_to_cog.py`.

Data (historical): tile **T33UWP**, all 2018 dates, flattened EODATA layout under
`satellite/`. Crop geometry: `shapefiles/s2grid=476da24.geojson` (EPSG:4326; the
rasters are EPSG:32633, so this also exercises CRS-mismatch handling).

Work top-to-bottom in one Python session. Tick a box when QGIS confirms the result.

---

## 0. Setup

From the repo root (`fsd/`), with the dev env active (`source .venv/bin/activate`):

```python
import os
import numpy as np
import geopandas as gpd
from rise_eo.raster import images
from rise_eo.bands import modify

# Workspace root that holds satellite/ and shapefiles/ (parent of fsd/)
ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

PRODUCT = os.path.join(
    ROOT, "satellite/sentinel-2-l2a/Sentinel-2/MSI/L2A_N0500/2018/06/30/"
    "S2B_MSIL2A_20180630T100029_N0500_R122_T33UWP_20230804T104527",
)
SHAPEFILE = os.path.join(ROOT, "shapefiles/s2grid=476da24.geojson")
OUT = os.path.join(os.getcwd(), "tests/outputs/realdata")
os.makedirs(OUT, exist_ok=True)

roi = gpd.read_file(SHAPEFILE)
print("roi crs:", roi.crs)          # EPSG:4326
print("product exists:", os.path.isdir(PRODUCT))
```

- [ ] `roi crs: EPSG:4326` and `product exists: True`

> To test another date, point `PRODUCT` at any other `…/2018/MM/DD/<product>/`
> folder under `satellite/`.

---

## Section A — raster: crop a real tile + save composites

`crop_tif` reprojects the **vector** to each raster's CRS internally
(`shapes_gdf.to_crs(src.crs)`) and clips with `rasterio.mask` — the recommended
"reproject vector, not raster; clip-by-mask preserves resolution, no
interpolation" approach. We always crop first: a full 10 m band is 10980×10980 px
(~241 MB) and would exhaust laptop memory.

### A1. Load + crop the visible + NIR bands

```python
def crop(band):
    return images.crop_tif(f"{PRODUCT}/{band}.jp2", roi, nodata=0, all_touched=True)

bands = {b: crop(b) for b in ["B02", "B03", "B04", "B08"]}
data, profile = bands["B04"]
print("cropped shape:", data.shape, "crs:", profile["crs"], "dtype:", profile["dtype"])
```

- [ ] Prints a small shape like `(1, 550, 606)`, `crs: EPSG:32633`, `dtype: uint16`
      (the 6×5.5 km ROI at 10 m — not the full tile)

### A2. True-color composite (B04/B03/B02 → R/G/B)

`scale_max=4000` ≈ Sentinel Hub's `2.5 × reflectance` display gain
(`2.5 × DN/10000 = DN/4000`). `_native` keeps lossless uint16 (stretch it in QGIS);
`_8bit` is display-ready (lossy: rescaled to 0–255, clipped at 4000).

```python
images.save_rgb_geotiff(f"{OUT}/476da24_20180630_TCC_native.tif",
                        [bands["B04"], bands["B03"], bands["B02"]])
images.save_rgb_geotiff(f"{OUT}/476da24_20180630_TCC_8bit.tif",
                        [bands["B04"], bands["B03"], bands["B02"]], scale_max=4000)
```

- [ ] Open `…_TCC_8bit.tif` in QGIS → looks like a natural-color scene, and it
      geolocates exactly over the `476da24` shape.

### A3. False-color composite (B08/B04/B03 → R/G/B), vegetation = red

Standard vegetation FCC: NIR in red, red in green, green in blue.

```python
images.save_rgb_geotiff(f"{OUT}/476da24_20180630_FCC_native.tif",
                        [bands["B08"], bands["B04"], bands["B03"]])
images.save_rgb_geotiff(f"{OUT}/476da24_20180630_FCC_8bit.tif",
                        [bands["B08"], bands["B04"], bands["B03"]], scale_max=4000)
```

- [ ] Open `…_FCC_8bit.tif` in QGIS → healthy vegetation shows **red**; water dark.

---

## Section B — bands: NDVI on the real crop

NDVI = (NIR − Red)/(NIR + Red) = (B08 − B04)/(B08 + B04). We feed the cropped bands
through the 5-D `bands.modify` contract exactly as the notebooks do.

```python
b04, p04 = bands["B04"]
b08, _   = bands["B08"]

# 5-D contract: (samples, timestamps, height, width, bands)
cube = np.stack([b04[0], b08[0]], axis=-1)[np.newaxis, ...]   # (1, h, w, 2) = (t,h,w,b)
bands5d = modify.expand_datacube(cube).astype(float)          # (1, 1, h, w, 2)

with np.errstate(divide="ignore", invalid="ignore"):
    out, obi = modify.compute_bands(bands5d, {"B04": 0, "B08": 1}, ["NDVI"])
ndvi = out[0, 0, :, :, obi["NDVI"]]                           # (h, w)

# nodata pixels (B04+B08 == 0) -> NaN so QGIS leaves them uncolored
ndvi32 = np.where(np.isfinite(ndvi), ndvi, np.nan).astype("float32")[np.newaxis, ...]
images.save_geotiff(f"{OUT}/476da24_20180630_NDVI.tif",
                    ndvi32, {**p04, "dtype": "float32", "nodata": float("nan"), "count": 1})

v = ndvi32[np.isfinite(ndvi32)]
print(f"NDVI range {v.min():.3f}..{v.max():.3f}, mean {v.mean():.3f}")
```

- [ ] Prints a plausible range within roughly `-1..1` (e.g. `-0.079..0.633`).
- [ ] Open `…_NDVI.tif` in QGIS, apply a singleband-pseudocolor ramp (e.g.
      −0.1 → 0.7): vegetated fields read high, bare/built/water read low.

---

## Notes

- **`_native` vs `_8bit`**: native = lossless uint16 reflectance (analysis; needs a
  QGIS stretch); 8-bit = display-ready 0–255 (lossy, clipped at `scale_max`).
- **Crop first, always** — full tiles OOM a laptop; every step here runs on the
  small ROI crop.
- **CRS handling** is inside `crop_tif`; you never reproject the raster yourself.

## References
- FCC band order (NIR/Red/Green): GIS Geography, Sentinel Hub custom scripts.
- Reproject vector not raster; clip-by-mask preserves resolution: Earth Lab
  (earthdatascience.org), L. Lucchese (QGIS warp/clip).

## To extend
The datacube time-series test (crop a *time series* → SCL cloud-mask → median-mosaic
→ QGIS) now lives in its own runbook: **`datacube.md`** (real multi-CRS build).
