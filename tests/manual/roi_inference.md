# Manual test — `run_inference(roi=…)` (Mode A, one call: ROI → per-cell COGs)

Validates the **P0.75 ROI inference verb** (spec 21) on real data: a single
`rise_eo.run_inference(roi=…)` call **tiles an ROI into S2 grid cells, builds one datacube per
cell, infers each, and writes per-cell COGs + a STAC catalog (+ an optional merged map)** — all
local, via the **runner seam** (Snakemake now; Azure Batch swaps in at P4 unchanged).

Supersedes `deploy.md` §3's hand-rolled 3×3-bbox-grid stand-in — that existed *only because* this
verb didn't. Unit tests (`tests/test_workflows.py::test_infer_task_*`, `tests/test_api_roi.py`)
cover the per-cell task, the merge modes, and the preflight guards on synthetic data; this proves
the whole verb end-to-end on genuine bytes + real multi-zone CRS.

> **Env.** Needs the **`[grid]`** extra (`s2`+`s2cell`) — run in `.venv-modeldeploy`
> (`pip install -e ".[dev,grid,model-example]"`), *not* the lean `.venv`. QGIS eyeballing is the
> final gate (the visual-validation rule).

---

## 0. Setup + a rule-based adapter (no training needed)

The adapter must be **importable by `module:attr`** (F5) so the per-cell task can reload it in a
subprocess — put it in a real module on `PYTHONPATH`, **not** an interactive `__main__`. Write
`ndvi_thresh.py`:

```python
from rise_eo.bands import modify
from rise_eo.model import BaseModelAdapter

class NDVIThresh(BaseModelAdapter):
    required_bands = ["B04", "B08"]
    n_timestamps = 2                      # must match T from the dates below
    output_dtype = "uint8"; output_nodata = 255
    output_band_names = ["ndvi_class"]
    feature_sequence = [
        (modify.mask_invalid_and_interpolate, {}),
        (modify.compute_bands, dict(bands_to_compute=["NDVI"])),
        (modify.remove_bands, dict(bands_to_remove=["B04", "B08"])),
    ]
    def load(self): pass
    def predict(self, X):                 # NDVI mean > 0.3 -> class 1
        return (X.mean(axis=1) > 0.3).astype("uint8")
```

---

## 1. One call: ROI → per-cell COGs + STAC + merged map

From the repo root (`fsd/`), with `ndvi_thresh.py` importable (`PYTHONPATH=.`):

```python
import datetime, os
import geopandas as gpd
from shapely.geometry import box
import rise_eo
from ndvi_thresh import NDVIThresh

CAT = "../satellite_benchmark/sentinel-2-l2a/catalog.parquet"
# a small ~9 km sub-ROI keeps the cell count (and time) down; use inference_roi.geojson for the full run
roi = gpd.GeoDataFrame({"geometry": [box(36.20, 11.45, 36.28, 11.53)]}, crs="EPSG:4326")

res = rise_eo.run_inference(
    NDVIThresh(),
    output_folderpath=os.path.join("tests/outputs/roi_inference"),
    roi=roi, catalog_filepath=CAT,
    startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 11),  # T=2 @ 20d
    mosaic_days=20, bands=["B04", "B08", "B8A", "SCL"],
    grid_size_km=5, scale_fact=1.1,
    merge="reproject",                    # ROI may straddle the 36°E zone boundary -> display merge
    cores=2,
)
print("cells:", len(gpd.read_file(res.grids_filepath)))
print("outputs:", len(res.output_filepaths), "| stac:", res.stac_catalog_filepath)
print("merged:", res.merged_filepath)
```

- [ ] Snakemake runs **one job per cell** (you'll see `N of N steps done`), building the datacube
      then inferring each → `output.tif`. A live progress log, not just a spinner.
- [ ] `res.grids_filepath` = `…/grids.geojson`; `len(outputs) == number of cells with tiles`.
      (My ~9 km box → **10 cells / 10 COGs**, ~40 s at `cores=2`.)

---

## 2. Validate the artifacts

```python
import rasterio, pystac
with rasterio.open(res.output_filepaths[0]) as s:
    print("per-cell:", s.count, s.dtypes[0], "nodata", s.nodata, "crs", s.crs.to_epsg())
with rasterio.open(res.merged_filepath) as s:
    a = s.read(1)
    print("merged:", s.shape, "crs", s.crs.to_epsg(),
          "valid%", round(100*(a != 255).mean(), 1))
cat = pystac.Catalog.from_file(res.stac_catalog_filepath)
print("stac items:", len(list(cat.get_items(recursive=True))))
```

- [ ] Each per-cell output is a **1-band uint8 COG, nodata 255**, in the cell's UTM CRS.
- [ ] `merged.tif` is a single COG in the **dominant zone** (nearest-neighbour reproject; lossy —
      *display only*, the per-cell COGs stay authoritative). STAC item count == number of outputs.

**Resumability:** re-run the same call → Snakemake reports **"Nothing to be done"** (cells whose
`done_infer.txt` exists are skipped). Delete one cell's `done_infer.txt` + `output.tif` and re-run
→ only that cell rebuilds.

**Merge modes (spec 21):**
- [ ] `merge=True` on a **zone-straddling** ROI **raises** (strict single-CRS) with a message
      pointing at `merge="reproject"`. On a single-zone ROI it produces a data-faithful merge.
- [ ] `merge="reproject"` always yields a viewable map.

---

## 3. QGIS gate (required)

- [ ] Load `merged.tif` (or the per-cell `output.tif`s) + drop `grids.geojson` and the ROI on top —
      the class map tiles the ROI with **no seams** between cells (the `scale_fact=1.1` overlap) and
      **no dead cells** (spec-20 fix).
- [ ] Overlay a true-colour composite for the same window (see `datacube.md`) — output nodata (255)
      coincides with cloud/edge nodata, not valid land.

---

## Notes
- ROI mode requires a **bundle**; a live adapter is auto-saved to `…/_bundle` (needs an importable
  class). `cores` = how many cells build+infer at once (the runner seam); Batch (P4) dispatches the
  same per-cell task.
- **Imagery is assumed present** in `catalog_filepath` — inference never calls CDSE (conserve quota;
  spec 21 SO-6). Download first with `rise_eo.download` if a cell has no tiles (those cells warn + skip).
- Outputs are gitignored (`tests/outputs/`). Full-ROI run = swap in `shapefiles/inference_roi.geojson`
  and a full-year window (`T=19`); expect both UTM zones → `merge="reproject"`.
