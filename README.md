# rise-eo

**RISE** (Rapid Intelligence System for the Environment) is NASA Harvest's toolkit for producing crop
and land maps quickly for a new region. **rise-eo is its satellite half: imagery in, model-ready data
out, your model's predictions back as maps.** The [RISE repository](https://github.com/nasaharvest/rise)
(NASA Harvest members only, for now) pins rise-eo next to RISE's reference-data module and shared
notebooks; rise-eo also works on its own.

The same calls run on your laptop or across an Azure ML cluster. Scaling out is one argument
(`runner="aml"`), not a rewrite.

**rise-eo does not train your model.** It hands you training arrays, takes back your trained model
wrapped in a small adapter, and runs it over any region. That line is deliberate and permanent.

## What v0.3.0 does

| stage | you give | you get | call |
|---|---|---|---|
| **Download** | an ROI, dates, bands | a shared imagery archive + a GeoParquet catalog | `rise_eo.download` |
| **Training data** | labelled polygons | `(samples, timestamps, bands)` arrays, with ids, labels and coordinates | `rise_eo.create_training_data` |
| **Check your model** | your model, wrapped as a `ModelAdapter` | a pass on one real grid cell, before any cluster spend | `rise_eo.verify_adapter` |
| **Bundle and publish** | the adapter + its files | a portable bundle folder; optionally a registry ref like `crop-rf:1` | `rise_eo.save_bundle`, `rise_eo.deploy` |
| **Inference** | a bundle or ref + an ROI | one COG per grid cell, a merged map, a STAC catalog | `rise_eo.run_inference` |
| **Scale out** | `runner="aml"` | the same calls, fanned out on Azure ML | every verb above |
| **Serve** | the COGs + STAC | XYZ map tiles, through stock open-source servers | none; see [below](#5-serve-the-maps) |

**Imagery:** Sentinel-2 L2A from the Microsoft Planetary Computer (MPC, no account needed) or the
Copernicus Data Space Ecosystem (CDSE), and Sentinel-1 RTC from MPC.

**Measured on real data:**
- **Inference at scale:** a 300-cell Austria ROI ran on Azure ML in **18.8 minutes across 32 nodes**:
  213 MPC granules in, 300 COGs + STAC + a merged map out ([report](demos/E2E_AUSTRIA_AML.md)).
- **Training data at scale:** 900 field datacubes on blob storage reduced into one training array on a
  single node.

## Install

```bash
pip install "rise-eo[local,mpc,grid] @ git+https://github.com/nasaharvest/rise-eo@v0.3.0"
```

Python ≥ 3.11. If you use RISE, it installs rise-eo for you. Install only the extras you need;
if one is missing, rise-eo's error names the extra to add.

| extra | you need it for |
|---|---|
| `local` | the default `runner="local"` pipeline (Snakemake), **and every AML node image** |
| `mpc` | `source="mpc"` |
| `s3` | `source="cdse"` downloads, and any `s3://` path |
| `grid` | splitting an ROI into grid cells (`roi=` on `run_inference` and `verify_adapter`) |
| `azure`, `aml` | `abfss://` storage and managed identity; `runner="aml"` |
| `titiler`, `serving` | serving the output maps |

For development: clone, then follow [`CONTRIBUTING.md`](CONTRIBUTING.md#setup).

## 1. Download imagery

```python
import datetime
import rise_eo

catalog = rise_eo.download(
    roi="my_roi.geojson",
    startdate=datetime.datetime(2018, 1, 1),
    enddate=datetime.datetime(2019, 1, 1),
    bands=["B04", "B08", "B8A", "SCL"],
    dst_folderpath="data/imagery",   # one shared archive for every run
    source="mpc",                    # no credentials; for CDSE: source="cdse", creds=CdseCredentials.from_env()
    max_tiles=20,                    # refuse the run if the ROI matches more MGRS granules than this
    max_cloudcover=60,               # optional: skip granules cloudier than this (%)
)
```

- **The archive is shared and lossless.** A repeat run downloads only what is missing. If two
  processings of one acquisition are both stored (an MPC copy and a CDSE copy, say), later steps ask you
  to pick one with `processing=` instead of silently choosing.
- **Preflight runs before any spend.** `max_tiles` is required; rise-eo counts the matching granules
  from metadata alone and stops if the ROI is bigger than you meant.
- **Sentinel-1 RTC** uses the same call with `collection="sentinel-1-rtc"`, `bands=["vv", "vh"]` and
  `source="mpc"`. Ascending and descending passes are never mixed in one datacube: when you build, pick
  one with `properties_filter={"sat:orbit_state": "ascending"}`.

## 2. Build training data

```python
training = rise_eo.create_training_data(
    label_polygons="my_labeled_fields.geojson",
    catalog_filepath=catalog,
    startdate=datetime.datetime(2018, 1, 1),
    enddate=datetime.datetime(2019, 1, 1),
    mosaic_days=20,
    bands=["B04", "B08", "B8A", "SCL"],
    id_col="fid",
    label_col="crop_type",
    export_folderpath="data/training",
)

arrays = training.load()   # {"data", "ids", "labels", "coords", "metadata"}
# arrays["data"] has shape (samples, timestamps, bands). Train any model you like on it.
```

For each labelled polygon, rise-eo builds a datacube and flattens the cubes into one array:
- **cloud-masked** with the Sentinel-2 scene classification (SCL);
- **mosaicked** into calendar intervals of `mosaic_days` (median), so every cube in a run has the same
  time axis;
- **merged across MGRS tiles**, after moving them all into one CRS, and resampled to a real 10 m band.

Pass your adapter (`adapter=`) to compute its features here; inference later computes the exact same
features.

## 3. Run your model

Wrap your trained model in a **`ModelAdapter`**. It declares the bands and number of timestamps it
needs and what it outputs, and implements `load` and `predict`. Subclass `rise_eo.BaseModelAdapter` and
the rest is filled in. [`examples/eurocrops_rf.py`](examples/eurocrops_rf.py) is a complete one to copy.

```python
rise_eo.verify_adapter(              # one real grid cell, end to end, before a large run
    my_adapter,
    roi="my_roi.geojson",
    catalog_filepath=catalog,
    startdate=datetime.datetime(2018, 1, 1),
    enddate=datetime.datetime(2019, 1, 1),
    mosaic_days=20,
    bands=["B04", "B08", "B8A", "SCL"],
    export_folderpath="data/verify",
)
rise_eo.save_bundle(my_adapter, {"model": "model.joblib"}, "my_bundle/")

result = rise_eo.run_inference(
    "my_bundle/",
    output_folderpath="data/predictions",
    roi="my_roi.geojson",            # split into ~5 km grid cells; one datacube per cell
    catalog_filepath=catalog,
    startdate=datetime.datetime(2018, 1, 1),
    enddate=datetime.datetime(2019, 1, 1),
    mosaic_days=20,
    bands=["B04", "B08", "B8A", "SCL"],
    merge="reproject",               # also write one merged map, in one CRS
)
```

`run_inference` checks that the adapter's bands and timestamp count match the run before any heavy
work, and writes one COG per grid cell plus a STAC catalog. To share a model, `rise_eo.deploy`
publishes a saved bundle to a registry, and refuses until the bundle has run on the node image you
name. `run_inference` then takes the returned ref (`"crop-rf:1"`) in place of the path. See
[`docs/howto/bundle-your-model.md`](docs/howto/bundle-your-model.md).

## 4. Scale out on Azure ML

Add `runner="aml"` and `runner_kwargs` (your cluster, node image, blob working root and identity) to
the same calls. ROI-mode inference also takes `storage="azure"`. Nothing else changes: every file goes
through one storage layer (`fsspec`), so local paths and `abfss://` paths are interchangeable.

```python
result = rise_eo.run_inference(
    "my_bundle/", output_folderpath="data/predictions",
    roi="my_roi.geojson", catalog_filepath=catalog,
    startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
    mosaic_days=20, bands=["B04", "B08", "B8A", "SCL"],
    merge="reproject",
    runner="aml", storage="azure", runner_kwargs=runner_kwargs,
)
```

Cluster settings live in a user-level config file, written once:

```bash
rise-eo init     # -> ~/.config/rise-eo/config.toml
rise-eo config   # print the resolved config and where each value came from
```

A crashed inference run resumes per grid cell: finished cells are skipped. Full setup:
[`docs/howto/run-at-scale.md`](docs/howto/run-at-scale.md).

> **Building an AML node image?** It must include the `local` extra, because nodes run the same
> Snakemake steps a laptop does. An image without it builds fine, then fails about 30 minutes into a
> run. See [`docs/howto/build-the-images.md`](docs/howto/build-the-images.md).

## 5. Serve the maps

rise-eo writes standard outputs (Cloud-Optimized GeoTIFFs and a STAC catalog) and hosts nothing
itself. Stock open-source servers turn them into XYZ tiles for any web map:
- **One map, no database:** serve the merged COG directly with TiTiler (the `titiler` extra).
- **Many runs, searchable:** load the STAC into pgSTAC and serve it with titiler-pgstac (the
  `serving` extra).

Both are worked through in [`docs/howto/serve-xyz.md`](docs/howto/serve-xyz.md).

## Not yet

- **Two collections only:** Sentinel-2 L2A and Sentinel-1 RTC. No HLS, Landsat, L1C, ERA5 or CHIRPS.
- **One cluster runner:** Azure ML. Storage is generic (`fsspec`, so S3 works), but job dispatch is not.
- **Fixed output grid:** datacubes are at the reference band's resolution (10 m for Sentinel-2), with
  median mosaics only. Downloads fetch whole MGRS granules.
- **No hosted map, and no colour scheme on the outputs** for a viewer to pick up.
- **The API is not stable yet.** rise-eo is `0.y.z`, and a minor release can break it; each release's
  notes list what broke.

Everything else, with what would trigger a fix: [`LIMITATIONS.md`](LIMITATIONS.md).

## Where to go next

| you want | read |
|---|---|
| a first run that works offline, with no credentials | [`docs/tutorial.md`](docs/tutorial.md) |
| your own region, real imagery, your own model, a cluster or a map | [`docs/howto/`](docs/howto/) |
| a complete script to copy | [`examples/eurocrops_rf.py`](examples/eurocrops_rf.py) |
| everything rise-eo does not do yet | [`LIMITATIONS.md`](LIMITATIONS.md) |
| to change rise-eo: setup, the review process, design docs | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## License

Code: MIT (`LICENSE`). Data derived from EuroCrops (`notebooks/shapefiles/AT_2018_TRAIN.geojson`,
`tests/data/tutorial/fields.geojson`) is CC BY 4.0; see each folder's `NOTICE` for the citation.
