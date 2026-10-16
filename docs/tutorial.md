# Tutorial — download to crop map, offline, in one sitting

> **Last verified:** 2026-07-31 @ `df98463` (spec 41 D5 tier 2 — "dated"). Re-verify by re-running
> this tutorial after any change to `rise_eo.create_training_data`, `rise_eo.run_inference`,
> `rise_eo.workflows.create_datacube`, or `tests/data/tutorial/`; the underlying pipeline is pinned by
> `tests/test_tutorial_fixture.py`, which runs in `pytest -q`.

**This tutorial cannot fail.** Every byte it reads is a real Sentinel-2 pixel already committed to
the repo — no download, no credentials, no cloud account. If a step below does not do what it
says, that is a bug in rise-eo or in this page, not something to work around: please open an issue with
the exact command and output (`gh issue create`, or however you were pointed here).

If you've finished this and want to point rise-eo at your own region, real imagery, a cluster, your own
model, or a map viewer, the next stop is [`docs/howto/`](howto/).

## What you'll build

Real Sentinel-2 L2A pixels over one 5 km grid cell in the Waldviertel region of Austria → a trained
3-class crop classifier → a predicted crop map (a GeoTIFF + a STAC item) for that same cell. Roughly
**4 minutes**, almost all of it one intentionally slow step (§3) that this page tells you about in
advance so it doesn't look like a hang.

## Prerequisites

- A clone of this repo and Python ≥ 3.11. Nothing else — no CDSE account, no Azure, no VPN.

```bash
git clone https://github.com/nikhilsrajan/fsd.git
cd rise-eo
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,local,model-example]"
```

`local` brings in `snakemake`, which runs the default `runner="local"` pipeline — steps 3 and 6
below both need it. `model-example` brings in `scikit-learn`/`joblib` for the classifier this
tutorial trains — rise-eo
itself never trains a model (permanent design choice, see [`ARCHITECTURE.md`](../ARCHITECTURE.md)).

## 1. The data you already have

`tests/data/tutorial/` is committed to this repo — real Sentinel-2 pixels, clipped and shipped so
this tutorial can promise success. Look at what's there before touching any code:

```bash
ls tests/data/tutorial/
cat tests/data/tutorial/NOTICE
```

| | |
|---|---|
| granules | 36, one MGRS tile (`T33UWP`) — this fixture never exercises multi-tile merge, see [`docs/howto/your-own-region.md`](howto/your-own-region.md) |
| bands | `B04`, `B08`, `SCL` (SCL is the cloud/shadow mask, dropped before training — 2 bands reach the model) |
| date span | 2018-04-01 → 2018-09-28 |
| grid cell | `4772924`, 15.3900–15.4717°E, 48.4821–48.5320°N |
| labelled fields | 43, collapsed to 3 classes (§2) |
| size on disk | 27 MB |

`tests/data/tutorial/README.md` records exactly which source granules these pixels came from and
how they were clipped (`tests/data/tutorial/build_fixture.py`). You do not need to run it. The
committed fixture predates spec 59's granule naming, so today's `build_fixture.py` does not
reproduce it byte for byte ([#121](https://github.com/nikhilsrajan/fsd/issues/121)).

**On radiometry, if you're curious:** these are 2018-era Microsoft Planetary Computer products,
generated before Sentinel-2's processing-baseline-04.00 offset convention existed. Their radiometric
offset is **declared by the source** as `0`, not the `-1000` you'll see on newer products — both are
correct, they're just different product versions of the same acquisitions. rise-eo reads whatever the
source declares; it never guesses.

## 2. Labels: three classes, derived from data

```bash
python -c "
import geopandas as gpd
fields = gpd.read_file('tests/data/tutorial/fields.geojson')
print(fields['label'].value_counts())
"
```

Expect:

```
label
grain_maize_corn_popcorn    20
hemp_cannabis                13
other                        10
Name: count, dtype: int64
```

Those two long names are not a display choice — they're the actual HCAT crop-type values in the
source labels, kept as-is because renaming them would be exactly the kind of hardcoded mapping that
silently broke this fixture once already (every field collapsed to `other` until the majors were
derived from clipped area instead of guessed). If you build your own fixture, print your own
`value_counts()` — don't copy these names.

## 3. Build training arrays — `rise_eo.create_training_data`

```python
import datetime
import rise_eo

startdate = datetime.datetime(2018, 4, 1)
enddate = datetime.datetime(2018, 9, 29)

training = rise_eo.create_training_data(
    label_polygons="tests/data/tutorial/fields.geojson",
    catalog_filepath="tests/data/tutorial/sentinel-2-l2a/catalog.parquet",
    startdate=startdate, enddate=enddate, mosaic_days=20,
    bands=["B04", "B08", "SCL"],
    id_col="fid", label_col="label",
    export_folderpath="/tmp/rise_eo_tutorial/training",
)

loaded = training.load()
data, labels = loaded["data"], loaded["labels"]
print(data.shape, sorted(set(labels)))
```

**This step takes 3–4 minutes and that is expected.** It builds one datacube per labelled
field — 43 fields × 36 granules × 3 bands, one at a time (`cores=1`, the default). Run it as a
plain script (as above) and you'll see live `[setup] N/43 shapes … ETA` progress. If you instead
run this inside `pytest -s` (e.g. the acceptance test this tutorial narrates,
`tests/test_tutorial_fixture.py`), use `-s` — plain `pytest` captures stdout, so without it the
progress line disappears and 3–4 minutes of silence looks like a hang. It is not one.

Expect `data.shape == (N, 10, 2)` and `sorted(set(labels))` to have **3** entries. `N` is the total
pixel count across all 43 labelled fields — this call returns one row **per pixel**, not per field
(pass `aggregate="median_per_id"` if you want one row per field instead). `10` is `T`, the mosaic
interval count (`ceil(181 days / 20) = 10`, from `startdate=2018-04-01` to `enddate=2018-09-29`);
`2` is the band count — `B04`, `B08` (`SCL` was consumed as the cloud/shadow mask and dropped, per
rise-eo's masking default).

## 4. Train a classifier (your code, not rise-eo's)

rise-eo hands you arrays; it never trains a model. Any framework works — this is deliberately the
simplest possible one:

```python
from sklearn.ensemble import RandomForestClassifier

X = data.reshape(data.shape[0], -1)
clf = RandomForestClassifier(n_estimators=5, random_state=0)
clf.fit(X, labels)
classes = sorted(set(labels))
```

`n_estimators=5` is chosen for tutorial speed on 43 samples, not accuracy — see
[`docs/howto/bundle-your-model.md`](howto/bundle-your-model.md) for a real model and how to bundle
it as a reusable `ModelAdapter`.

## 5. Build one inference datacube over the fixture's own cell

Inference needs a datacube built the same way training data was — same bands, same window, same
`mosaic_days`. The tutorial's ROI (`roi.geojson`) is exactly the one grid cell the fixture covers,
so this builds a single datacube via the same workflow entrypoint `create_training_data` uses under
the hood:

```python
from rise_eo.workflows import create_datacube

csv_filepath = "/tmp/rise_eo_tutorial/infer_build/input.csv"
create_datacube.run_create_datacube(
    catalog_filepath="tests/data/tutorial/sentinel-2-l2a/catalog.parquet", timestamp_col="timestamp",
    shapefilepath="tests/data/tutorial/roi.geojson", id_col="id",
    run_folderpath="/tmp/rise_eo_tutorial/infer_build",
    startdate=startdate, enddate=enddate, bands=["B04", "B08", "SCL"],
    mosaic_days=20, csv_filepath=csv_filepath, label_col=None, cores=1,
)
```

The SCL mask is not passed: it is declared on the `sentinel-2-l2a` collection (spec 58 D3).
`collection=` (default `"sentinel-2-l2a"`) and `properties_filter=` are the two knobs that
replace it — see `rise_eo.collections.known()`.

This takes seconds — it's one cell, not 43 fields.

## 6. Run inference — `rise_eo.run_inference`

Wrap the trained classifier as a `ModelAdapter` (the contract rise-eo calls at inference time — see
[`docs/howto/bundle-your-model.md`](howto/bundle-your-model.md) for the full guide) and run it over
the datacube you just built:

```python
import numpy as np
from rise_eo.model import BaseModelAdapter

class TutorialClassifier(BaseModelAdapter):
    required_bands = ["B04", "B08"]
    n_timestamps = 0          # model-determined: follow whatever T the input cube has
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["crop_class"]
    feature_sequence = None   # identity: raw bands, no transform

    def __init__(self, clf, classes):
        self.clf = clf
        self.classes = list(classes)

    def load(self):
        pass  # clf/classes were passed in directly for this tutorial

    def features(self, data5d, band_indices):
        return data5d, band_indices

    def predict(self, X_chunk):
        pred = self.clf.predict(X_chunk)
        return np.array([self.classes.index(p) for p in pred], dtype="uint8")

adapter = TutorialClassifier(clf, classes)
result = rise_eo.run_inference(adapter, csv_filepath, "/tmp/rise_eo_tutorial/out", progress=False)
print(result.output_filepaths)
print(result.stac_catalog_filepath)
```

## 7. Look at what you made

`result.output_filepaths[0]` (printed in §6) is a GeoTIFF — open it in QGIS (Layer → Add Raster
Layer) to see the three-class crop map, or inspect it in Python:

```python
import rasterio
with rasterio.open(result.output_filepaths[0]) as src:
    print(src.count, src.nodata, src.width, src.height)   # 1, 255, ...
```

One band (`crop_class`), `nodata = 255`, one pixel value per class index (`0`/`1`/`2` — look them up
against `classes` from §4). `result.stac_catalog_filepath` is a standard STAC catalog with one item
— see [`docs/howto/serve-xyz.md`](howto/serve-xyz.md) if you want to serve it on a map instead of
opening the file directly.

## What you didn't need

No CDSE account, no Azure subscription, no `AZ_*` environment variable, no GPU, no more than 27 MB
on disk. That's the point of the committed fixture (spec 42) — everything above is real Sentinel-2
physics on a real crop, just clipped small enough to ship.

## Where to go next

| you want | read |
|---|---|
| to run this on your own region instead of the fixture | [`docs/howto/your-own-region.md`](howto/your-own-region.md) |
| to actually download imagery (this tutorial didn't) | [`docs/howto/download-real-imagery.md`](howto/download-real-imagery.md) |
| to run the same pipeline on a cluster, not your laptop | [`docs/howto/run-at-scale.md`](howto/run-at-scale.md) |
| to package a real model as a reusable bundle | [`docs/howto/bundle-your-model.md`](howto/bundle-your-model.md) |
| to serve the output on an XYZ map viewer | [`docs/howto/serve-xyz.md`](howto/serve-xyz.md) |
| a complete, readable script instead of these fragments | [`examples/eurocrops_rf.py`](../examples/eurocrops_rf.py) |
