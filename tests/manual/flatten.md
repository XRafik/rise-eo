# Manual test — real satellite data (flatten → training arrays)

Validates `rise_eo.datacube.flatten.flatten` against a **real, multi-tile / multi-CRS**
labelled field set: it builds one datacube per field via the workflow, then folds the
cubes into per-pixel training arrays. This is the last v1-pipeline stage to get a
real-data run (download → catalog → datacube → **flatten** → workflows). Unit tests
(`tests/test_datacube_flatten.py`) cover the logic on synthetic cubes; this proves it on
genuine bytes, real CRS, real labels — and specifically that cubes from **different tiles
and UTM zones share a `timestamps` axis** so `flatten` can concatenate them.

> **Depends on spec 15 (calendar mosaic).** `flatten` requires every datacube to share an
> identical `timestamps` axis. That only holds because `median_mosaic` defaults to
> `mosaic_scheme="calendar"` (fixed calendar windows off the caller's `startdate`), so a
> field in EPSG:32636 and one in EPSG:32637 both label their windows `2018-06-01` /
> `2018-06-21`. With the legacy `"acquisition"` scheme this run would raise on the
> consistency gate.

Data: the COG archive `satellite_benchmark/sentinel-2-l2a/catalog.parquet` (Ethiopia,
EPSG:32636 & 32637; bands **B04/B08/B8A/SCL**) + a **EuroCrops** training set,
`shapefiles/austria_eurocrops_sampled_ethiopia_translated.geojson` (1015 Austrian field
polygons translated onto the Ethiopia footprint; **id col `fid`**, **label col
`EC_hcat_n`** = 11 crop classes). The fields span lon 36.06–36.80 → **both UTM zones**.

Work top-to-bottom. Outputs go to `tests/outputs/flatten/` (gitignored). Expected values
below are from a **class-stratified 33-field subset** (3 per class); scale up by pointing
at the full shapefile.

---

## 0. Setup

From the repo root (`fsd/`), env active (`source .venv/bin/activate`). Paths are relative
to the workspace root (parent of `fsd/`):

```python
import os, datetime, warnings
import numpy as np, pandas as pd, geopandas as gpd
from rise_eo.workflows import create_datacube
from rise_eo.datacube import flatten
from rise_eo.storage import fs
warnings.simplefilter("ignore")

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
CATALOG = os.path.join(ROOT, "satellite_benchmark/sentinel-2-l2a/catalog.parquet")
SHAPES  = os.path.join(ROOT, "shapefiles/austria_eurocrops_sampled_ethiopia_translated.geojson")
RUN = os.path.join(os.getcwd(), "tests/outputs/flatten/run")     # per-field cubes
OUT = os.path.join(os.getcwd(), "tests/outputs/flatten/flattened")
CSV = os.path.join(RUN, "input.csv")
os.makedirs(RUN, exist_ok=True)

# (optional) class-stratified subset for a fast run — 3 fields per crop class
g = gpd.read_file(SHAPES)
sub = g.groupby("EC_hcat_n", group_keys=False).sample(n=3, random_state=7)
SHAPES_SUB = os.path.join(os.getcwd(), "tests/outputs/flatten/subset.geojson")
sub.to_file(SHAPES_SUB, driver="GeoJSON")
print("fields:", len(sub), "| classes:", sub["EC_hcat_n"].nunique())
```

- [ ] Prints e.g. `fields: 33 | classes: 11` (use `SHAPES` instead of `SHAPES_SUB` below
      for the full 1015-field run).

---

## 1. Build one datacube per field (the real workflow path)

`run_create_datacube` runs `setup` (per-field geometry + catalog slice + `input.csv`
row) then the local Snakemake runner. Since spec 15, `setup` anchors the mosaic at the
**caller's** `startdate`/`enddate`, so all fields mosaic on the same calendar grid.

```python
create_datacube.run_create_datacube(
    catalog_filepath=CATALOG, timestamp_col="timestamp",
    shapefilepath=SHAPES_SUB, id_col="fid", run_folderpath=RUN,
    startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 10),
    bands=["B04", "B08", "B8A", "SCL"],
    mosaic_days=20, csv_filepath=CSV, label_col="EC_hcat_n", cores=8,
    # SCL mask classes are declared on the collection (spec 58 D3), not passed here.
)
df = pd.read_csv(CSV)
print("cubes:", len(df), "| anchor:", df["startdate"].iloc[0], "->", df["enddate"].iloc[0])
```

- [ ] `cubes: 33` and **anchor `2018-06-01 … -> 2018-07-10`** (the caller's calendar
      window, *not* per-shape actual acquisition dates — the spec-15 change). Each field's
      folder has a `done.txt`. `input.csv` carries `datacube_filepath`, `id` (=`fid`),
      `label` (=`EC_hcat_n`) — exactly what `flatten` consumes.

---

## 2. Flatten the cubes into per-pixel training arrays

`flatten` reads each cube `(t, H, W, b)`, keeps pixels that are not entirely nodata across
time+band → `(pixels, t, b)`, concatenates across cubes, and repeats each field's `id` and
`label` per pixel. It first asserts all cubes agree on `bands` and `timestamps`.

```python
flatten.flatten(filepaths_df=df, filepath_col="datacube_filepath",
                id_col="id", label_col="label", export_folderpath=OUT)

data   = fs.load_npy(f"{OUT}/data.npy")
coords = fs.load_npy(f"{OUT}/coords.npy")
ids    = fs.load_npy(f"{OUT}/ids.npy")
labels = fs.load_npy(f"{OUT}/labels.npy")
md     = fs.load_npy(f"{OUT}/metadata.pickle.npy", allow_pickle=True)[()]
print("data:", data.shape, data.dtype, "| bands:", md["bands"],
      "| ts:", [str(t.date()) for t in md["timestamps"]])
```

- [ ] `data: (6502, 2, 3) uint16`, `bands: ['B04', 'B08', 'B8A']` (SCL dropped),
      `ts: ['2018-06-01', '2018-06-21']`. `coords`/`ids`/`labels` are each length 6502.
      The `flatten` call **does not raise** — the consistency gate passed across cubes in
      **both** UTM zones (the spec-15 payoff).

---

## 3. Validate the arrays

```python
from collections import Counter
# per-field kept-pixel recompute (independent of flatten)
def kept(fp):
    c = fs.load_npy(fp); nt, _, _, nb = c.shape
    return int((((c == 0).sum(axis=(0, 3))) < nt * nb).sum())
df["kept"] = df["datacube_filepath"].map(kept)

assert data.shape[0] == int(df["kept"].sum())                 # total pixels
idc = Counter(ids.tolist())
assert all(idc[k] == v for k, v in zip(df["id"], df["kept"])) # per-field tag counts
assert md["data_shape_desc"] == ("pixel", "timestamps", "bands")

# round-trip: first flattened row == first kept pixel of the first cube
r0 = df.iloc[0]; c0 = fs.load_npy(r0["datacube_filepath"])
nt, _, _, nb = c0.shape
hi, wi = np.where(((c0 == 0).sum(axis=(0, 3))) < nt * nb)
assert np.array_equal(data[0], c0[:, hi[0], wi[0], :]) and ids[0] == r0["id"]

print("labels:", sorted(set(labels.tolist())))
for lab, ct in sorted(Counter(labels.tolist()).items()):
    print(f"  {lab:34s} {ct}")
```

- [ ] All asserts pass: total-pixel count matches, per-field `id`/`label` tag counts
      match, `data_shape_desc == ('pixel','timestamps','bands')`, and the round-trip is
      exact. Label distribution covers all 11 classes (e.g. `alfalfa_lucerne 532`,
      `hemp_cannabis 1572`, `sunflower 744`, …).

---

## 4. Known caveats

- **Multi-zone `coords.npy` (TODO #16 — RESOLVED).** `coords` are now emitted as
  `(lon, lat)` in **EPSG:4326**: each cube's per-pixel easting/northing is reprojected
  from its native CRS before concatenation, so a west field (was EPSG:32636) and an east
  field (was EPSG:32637) share one common CRS and are geographically comparable. Expect
  lon/lat values (e.g. ~14–15°E / ~48–49°N for Austria), not raw UTM eastings. Still does
  **not** affect the spectral arrays (`data` / `ids` / `labels`).
- **`mosaic_days` vs revisit (spec 15).** With `mosaic_days` ≥ the S2 revisit you get real
  temporal compositing; set it *below* the cadence and calendar mode pads the axis with
  all-nodata slices (documented in `ops.median_mosaic`). `mosaic_days=20` here → 2 windows.
- **Edge tightness (TODO #8).** Cubes carry a small nodata halo (see `datacube.md`);
  `flatten` drops fully-nodata pixels, so the halo never reaches the training arrays.

## Notes
- Outputs are gitignored (`tests/outputs/flatten/`). The full 1015-field run is the same
  commands with `SHAPES` instead of `SHAPES_SUB` (serial cube build ≈ 9 min; the workflow
  parallelises at `cores`).
- `flatten` also works without `label_col` (drops `labels.npy`) — e.g. an inference set.
