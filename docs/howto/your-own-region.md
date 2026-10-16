# How to: point fsd at your own region

> **Last verified:** 2026-07-31 @ `df98463` (spec 41 D5 tier 2 — "dated"). Re-verify after any
> change to `rise_eo.download`, `rise_eo.create_training_data`, `rise_eo.grid`, or the merge logic in
> `rise_eo.run_inference`.

You've finished [`docs/tutorial.md`](../tutorial.md) on the committed fixture. This page is the
same pipeline over **your** ROI and **your** labels. Unlike the tutorial, this page **cannot
promise success** — your data has properties the fixture was built to avoid. Read the pitfalls
below before running.

## Prerequisites

- An ROI (`your_roi.geojson`) and, if you're training a model, a labelled-fields file
  (`your_fields.geojson`) with an id column and a label column.
- If you actually want to download imagery for the ROI (the tutorial didn't — it used a committed
  fixture), read [`download-real-imagery.md`](download-real-imagery.md) **first**: fsd reads whole
  ~110 km MGRS granules, so the byte budget does not shrink with a small ROI, and that page states
  the budget up front.

## What changes from the tutorial

Three inputs, same call shapes:

```python
import rise_eo
from rise_eo.sources.cdse import CdseCredentials

catalog = rise_eo.download(
    roi="your_roi.geojson",
    startdate=..., enddate=..., bands=["B04", "B08", "B8A", "SCL"],
    dst_folderpath="data/imagery",   # one shared archive, reused across runs
    creds=CdseCredentials.from_env(),   # or source="mpc" for anonymous MPC
    max_tiles=20,                       # required cost guardrail -- see below
)

training = rise_eo.create_training_data(
    label_polygons="your_fields.geojson",
    catalog_filepath=catalog,
    startdate=..., enddate=..., mosaic_days=20,
    bands=["B04", "B08", "B8A", "SCL"],
    id_col="your_id_column", label_col="your_label_column",
    export_folderpath="data/training",
)
```

Everything else — tiling, datacube build, masking, inference, merge — is unchanged, **including
ROIs that straddle multiple MGRS tiles or UTM zones**: each S2 grid cell builds a datacube in its
own tile/CRS, and `rise_eo.run_inference(..., merge="reproject")` reprojects all cells to one CRS for
the merged output (area-dominant target by default, or pass `merge_crs=` yourself). **The tutorial
fixture never exercises this path** — it is a single grid cell fully inside one MGRS tile
(`T33UWP`) by design (spec 42 §5), so a multi-tile ROI is genuinely new territory for you, not
something the tutorial already validated for your case.

## `max_tiles` — read it before you run it

`rise_eo.download` refuses to run if your ROI/window matches more MGRS granules than `max_tiles` — a
required argument, not a default, because a wide ROI over a long window can silently mean hundreds
of ~426 MB granules. Preflight checks the granule count **before any bytes move**. If it refuses,
either narrow the ROI/window or raise `max_tiles` deliberately — don't raise it reflexively; read
[`download-real-imagery.md`](download-real-imagery.md) for what a given granule count actually
costs.

## Sizing before you commit to a download

Query the granule and grid-cell counts without downloading anything:

```python
from rise_eo import grid

cells = grid.roi_to_s2_grids("your_roi.geojson", grid_size_km=5)
print(len(cells), "grid cells")
```

For the granule count and byte estimate, use your source's own STAC search (CDSE's OData API or
MPC's `pystac_client`) against the same ROI/window/bands before calling `rise_eo.download` — this is
exactly what `max_tiles`'s preflight check does internally, just without the guardrail stopping you.

## The label-collapse trap (this cost real time once)

If you're training a classifier and collapsing many raw crop values into a few classes, **derive
the mapping from your data — never hardcode class name literals against values you haven't
printed.** This is not a style preference: spec 42's own fixture was built once with a hardcoded
`{"maize", "hemp"}` collapse that matched nothing against the real HCAT compound values in the
label file, and silently produced a single-class fixture that looked fine until training data was
inspected. Before writing any collapse logic:

```python
import geopandas as gpd

fields = gpd.read_file("your_fields.geojson")
print(fields["your_label_column"].value_counts())
```

If your collapse rule is "top-N classes by area, everything else `other`", derive it from a
`clipped area per label` computation the way
`tests/data/tutorial/derive_roi_and_labels.py::pick_major_crops` does — don't guess names.

## Diagnosing a run that produces zero training rows

- **Check the label file actually intersects the ROI/cells** you built datacubes for — a field
  file drawn from a different region than the ROI produces zero labelled pixels, not an error.
  (This is exactly how the original grid-cell candidate for this fixture, `s2grid=476da24`, was
  rejected: it sat ~100 km from every labelled field.)
- **Check `id_col`/`label_col` actually name columns that exist** in your GeoJSON — a typo raises
  at `create_training_data`'s preflight, not silently.
- **Check your window actually covers imagery** — `startdate`/`enddate` outside what you downloaded
  produces an empty catalog slice, not a partial one.

## Where to go next

- [`download-real-imagery.md`](download-real-imagery.md) — the byte budget for a real download.
- [`run-at-scale.md`](run-at-scale.md) — the same calls, `runner="aml"`, on a cluster.
- [`bundle-your-model.md`](bundle-your-model.md) — package your trained model as a `ModelAdapter`.
