# Manual test — Mode A end-to-end (train → adapter → inference → COG + STAC)

Validates the **ModelAdapter contract + local inference engine** (spec 18) on real data,
reproducing the legacy `demo_02_model_train` + `demo_03_model_deploy` through the fsd verbs:
`create_training_data(adapter=…)` → train your own model → wrap it in an adapter/bundle →
`run_inference(…)` → one **COG** per output + a **STAC** catalog (+ optional merged map). The whole
loop runs on a laptop (Mode A) with **one** feature-transform definition shared by training and
inference (the F1 anti-skew guarantee).

Unit tests (`tests/test_model.py`) cover the engine/bundle logic on synthetic cubes; this
proves it on genuine bytes + real CRS, and specifically that the **same adapter** drives both
`features.npy` at train time and the prediction at inference time.

> **Data note.** `satellite_benchmark/` only has bands **B04/B08/B8A/SCL**, so this runbook
> uses a small **NDVI** adapter (`required_bands=[B04,B08]`). The shipped 9-band
> `examples/eurocrops_rf.py` is the fuller illustration but needs a 9-band archive to run.

Work top-to-bottom. Outputs → `tests/outputs/deploy/` (gitignored). QGIS eyeballing is the
final gate (the visual-validation rule) — LLMs are unreliable on GeoTIFFs.

---

## 0. Setup + the adapter

From the repo root (`fsd/`), env active. The adapter is the only "model code":

```python
import os, datetime, warnings
import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import box
warnings.simplefilter("ignore")

import rise_eo
from rise_eo.bands import modify
from rise_eo.model import BaseModelAdapter
from rise_eo.workflows import create_datacube
from rise_eo.storage import fs

ROOT    = os.path.abspath(os.path.join(os.getcwd(), ".."))
CATALOG = os.path.join(ROOT, "satellite_benchmark/sentinel-2-l2a/catalog.parquet")
SHAPES  = os.path.join(ROOT, "shapefiles/austria_eurocrops_sampled_ethiopia_translated.geojson")
OUTDIR  = os.path.join(os.getcwd(), "tests/outputs/deploy")
os.makedirs(OUTDIR, exist_ok=True)

START, END, MOSAIC = datetime.datetime(2018, 6, 1), datetime.datetime(2018, 7, 10), 20
BANDS = ["B04", "B08", "B8A", "SCL"]              # SCL for cloud masking, dropped in the cube
T = rise_eo.compute_n_timestamps(START, END, MOSAIC)  # -> 2

class NDVIRandomForest(BaseModelAdapter):
    required_bands   = ["B04", "B08"]
    n_timestamps     = T
    output_dtype     = "uint8"
    output_nodata    = 255
    output_band_names = ["crop_class"]
    feature_sequence = [
        (modify.mask_invalid_and_interpolate, {}),
        (modify.compute_bands, dict(bands_to_compute=["NDVI"])),
        (modify.remove_bands, dict(bands_to_remove=["B04", "B08", "B8A"])),  # -> NDVI only
    ]
    def load(self):
        import joblib
        self.clf, self.le = joblib.load(self.artifacts["model"])
    def predict(self, X):
        return self.clf.predict(X).astype("uint8")

print("T =", T)
```

- [ ] `T = 2`.

---

## 1. Training data — the adapter runs the feature transform (F1)

`create_training_data(adapter=…)` builds one datacube per field, flattens to per-pixel arrays,
then runs **the adapter's `feature_sequence`** and writes `features.npy` (raw `data.npy` kept).
A class-stratified subset keeps it fast.

```python
g = gpd.read_file(SHAPES)
sub = g.groupby("EC_hcat_n", group_keys=False).sample(n=3, random_state=7)
SHAPES_SUB = os.path.join(OUTDIR, "train_fields.geojson"); sub.to_file(SHAPES_SUB, driver="GeoJSON")

td = rise_eo.create_training_data(
    label_polygons=SHAPES_SUB, catalog_filepath=CATALOG,
    startdate=START, enddate=END, mosaic_days=MOSAIC, bands=BANDS,
    id_col="fid", label_col="EC_hcat_n",
    export_folderpath=os.path.join(OUTDIR, "training_data"),
    adapter=NDVIRandomForest(), cores=8,
)
d = td.load()
print("features:", d["features"].shape, "| feature_bands:", td.feature_bands,
      "| raw kept:", d["data"].shape)
```

- [ ] `feature_bands: ['NDVI']`; `features` is `(pixels, 2, 1)`; raw `data` is `(pixels, 2, 3)`
      (B04/B08/B8A) — kept alongside. `feature_ids`/`feature_labels` present in `d`.

---

## 2. Train your model (your code — fsd does not train)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

X = d["features"].reshape(len(d["features"]), -1)   # (pixels, T*1)
le = LabelEncoder(); y = le.fit_transform(d["feature_labels"])
clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42).fit(X, y)
MODEL = os.path.join(OUTDIR, "rf.joblib"); joblib.dump((clf, le), MODEL)
print("classes:", list(le.classes_))
```

- [ ] Prints the 11 EuroCrops classes. (Accuracy is uninteresting on a 2-timestamp NDVI toy —
      the point is the *pipeline*, not the score.)

---

## 3. Build inference datacubes over a grid

Inference runs on **pre-built inference datacubes** (P0.5). Here we make a quick 3×3 grid over
the ROI bbox and build a cube per cell with the **same window/bands** (so `T` matches). The
real ROI→S2-grid tiling is the P4 front-end; this stand-in keeps the runbook self-contained.

```python
minx, miny, maxx, maxy = g.total_bounds
nx = ny = 3
cells = [box(minx + i*(maxx-minx)/nx, miny + j*(maxy-miny)/ny,
             minx + (i+1)*(maxx-minx)/nx, miny + (j+1)*(maxy-miny)/ny)
         for i in range(nx) for j in range(ny)]
grid = gpd.GeoDataFrame({"grid_id": range(len(cells)), "geometry": cells}, crs=g.crs)
GRID = os.path.join(OUTDIR, "grid.geojson"); grid.to_file(GRID, driver="GeoJSON")

INFER_RUN = os.path.join(OUTDIR, "inference_datacubes")
INFER_CSV = os.path.join(INFER_RUN, "input.csv")
create_datacube.run_create_datacube(
    catalog_filepath=CATALOG, timestamp_col="timestamp",
    shapefilepath=GRID, id_col="grid_id", run_folderpath=INFER_RUN,
    startdate=START, enddate=END, bands=BANDS,
    mosaic_days=MOSAIC, csv_filepath=INFER_CSV, cores=8,   # no label_col — inference set
    # SCL mask classes are declared on the collection (spec 58 D3), not passed here.
)
print("inference cubes:", len(pd.read_csv(INFER_CSV)))
```

- [ ] Builds ~9 cubes (some edge cells may be empty of tiles and skip). `input.csv` has a
      `datacube_filepath` column — exactly what `run_inference` consumes.

---

## 4. Run inference → COG + STAC (+ merged map)

```python
adapter = NDVIRandomForest()
adapter.artifacts = {"model": MODEL}          # a live adapter reads its artifact from here
                                              # (bundle.load injects this for you — see below)
result = rise_eo.run_inference(
    model=adapter,
    inference_datacubes=INFER_CSV,
    output_folderpath=os.path.join(OUTDIR, "model_outputs"),
    merge=True,
)
print("outputs:", len(result.output_filepaths))
print("stac:", result.stac_catalog_filepath)
print("merged:", result.merged_filepath)
```

> Simpler: just `model=NDVIRandomForest()`. (You can also `rise_eo.model.bundle.save(NDVIRandomForest(),
> {"model": MODEL}, ".../bundle")` and pass the **bundle path** — that's how the model travels to
> another machine or the cloud; see `specs/18-model-bundle-explainer.md`. A bundle needs the
> adapter class to be *importable* by `module:attr`, so put it in a `pip install`ed module, e.g.
> `examples/eurocrops_rf.py`, not an interactive `__main__`.)

- [ ] One `output.tif` per inference cube under `model_outputs/<grid_id>/`, a `stac/catalog.json`,
      and `merged.tif`. Each output is a **1-band uint8 COG, nodata 255**, in the cube's UTM CRS.
      Merge succeeds **only if all cubes share one CRS** — if the grid straddles the 36°E zone
      boundary it raises (by design); restrict the grid to one zone for a merged map, or use the
      per-output COGs + STAC.

---

## 5. Validate + QGIS

```python
import rasterio, pystac
with rasterio.open(result.output_filepaths[0]) as s:
    a = s.read(1)
    print("shape:", a.shape, "| dtype:", a.dtype, "| nodata:", s.nodata,
          "| classes present:", sorted(set(a.flatten()) - {255}))
cat = pystac.Catalog.from_file(result.stac_catalog_filepath)
print("stac items:", len(list(cat.get_items(recursive=True))))
```

- [ ] `dtype uint8`, `nodata 255`, class values are label-encoded ints within `range(11)`;
      STAC item count == number of outputs; each item has `proj:transform` + a COG asset.

**QGIS gate (required):**
- [ ] Load `merged.tif` (or the per-output `output.tif`s) — the class map aligns geographically
      with the ROI (drop the fields shapefile on top; classes fall on/near the training fields).
- [ ] Overlay a true-colour composite for the same window (see `datacube.md`) — the crop map's
      nodata (255) coincides with cloud/edge nodata, not with valid land.

---

## Notes
- The 3×3 bbox grid is a **runbook stand-in** for the P4 ROI→S2-grid tiling
  (`rsutils.s2_grid_utils.get_s2_grids_gdf`; recipe pinned in ROADMAP §4). Real inference will
  tile + download inside `run_inference`; here you supply pre-built cubes.
- `cores=1` infers **in-process** (a live adapter is fine, no bundle). `cores>1` fans out via the
  **Snakemake infer-only runner** (spec 22 — no `mp.Pool`) and needs a **bundle** (a live adapter is
  auto-saved); `cubes_per_task` groups cubes per job to amortise the model load. Inference is
  **idempotent** — a re-run skips existing outputs unless `overwrite=True` (so re-running this
  runbook's step 4 does nothing the second time).
- Outputs are gitignored (`tests/outputs/deploy/`). Scale up by using the full `SHAPES` set and a
  finer grid; a full-year (T=19) model just changes `START/END` and `n_timestamps`.
