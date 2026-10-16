# How to: bundle your model as a `ModelAdapter`

> **Last verified:** 2026-07-31 @ `df98463` (spec 41 D5 tier 2 — "dated"). Re-verify after any
> change to `rise_eo.model.BaseModelAdapter`, `rise_eo.model.bundle`, or the feature-sequence contract in
> `rise_eo.bands.modify`.

**fsd owns the plumbing** — download, datacube build, tiling, COG/STAC/merge, the local and AML
runners. **You own two endpoints** that connect your trained model to that plumbing. This page is
the step-by-step version of what [`docs/tutorial.md`](../tutorial.md) §6 did with a five-line
throwaway class; [`examples/eurocrops_rf.py`](../../examples/eurocrops_rf.py) is a complete,
copy-paste-ready adapter for a real EuroCrops-style model.

## The shape of an adapter

`ModelAdapter` is a duck-typed Protocol — any ML framework works. Subclass `BaseModelAdapter` for
sane defaults; a minimal adapter is ~10–20 lines.

### 1. Declarations — read at preflight, before any heavy compute

```python
required_bands = ["B04", "B08", "B8A"]   # bands your model needs
n_timestamps = 19                        # T your model was trained on -- 0 if model-determined
output_dtype = "uint8"
output_nodata = 255
output_band_names = ["crop_class"]       # 1 name -> categorical map; N -> probabilities/regression
```

`run_inference`'s preflight checks `bands ⊇ required_bands` and `T == n_timestamps` (unless `0`)
**before building a single datacube** — a mismatched model fails fast, not after an expensive build.
(`create_training_data` checks `required_bands` too, but not `n_timestamps`: there, `T` is the
caller's window to choose.)

### 2. Endpoint ① — datacube → model input (your feature transform)

Declared **once**, run by fsd at **both** training and inference — the anti-skew guarantee: train
and serve see identical features, by construction, not by discipline.

**That guarantee only holds if you hand the adapter to the training call too:**
`rise_eo.create_training_data(..., adapter=my_adapter)`. Without it, `create_training_data` writes the
**raw bands** and your transform runs at inference only — which is exactly the train/serve skew this
section claims to rule out. (There is also a raw `feature_sequence=` argument for adapter-less
exploration; pass one or the other, never both.)

```python
from rise_eo.bands import modify

feature_sequence = [
    (modify.mask_invalid_and_interpolate, {}),
    (modify.compute_bands, dict(bands_to_compute=["NDVI", "NDRE", "GCVI", "SAVI"])),
    (modify.remove_bands, dict(bands_to_remove=required_bands)),
]
```

This is a `rise_eo.bands.modify` pipeline over the 5-D `(samples, timestamps, height, width, bands)`
contract — the primary, declarative way to specify a transform. If your feature logic can't be
expressed as a sequence of `(fn, kwargs)` steps, override `features(data5d, band_indices)` instead
(what the tutorial's trivial classifier does — `feature_sequence = None`, `features` returns the
raw bands unchanged).

`datacube_to_X(feats, band_indices)` reshapes `(T, H, W, B)` → your model's `(H*W, T*B)` input; the
default is provided, override only if your model wants a different layout.

### 3. `predict(X_chunk)` — your framework, unmodified

```python
def predict(self, X_chunk):
    return self.clf.predict(X_chunk).astype("uint8")
```

fsd hands you valid (non-NaN) rows already chunked (`predict_batch_size`) and scatters NaN → nodata
on the output side — `predict` never sees a NaN and never has to think about masking.

### 4. Endpoint ② — raw output → standard `Output` (your packaging)

`to_output(raw, hw)` → `Output((bands, H, W), dtype, nodata, band_names)` — how your model's numbers
become the COG bands fsd writes. A default is provided that maps one categorical/vector value per
pixel using the declarations from step 1; override only for custom band packing.

### 5. `load()` — read your artifact once per worker

```python
def load(self):
    self.clf, self.label_encoder = joblib.load(self.artifacts["model"])
```

`self.artifacts` is `{name: absolute path}`, injected by the bundle before `load()` runs. fsd never
trains a model — you bring one that's already fit.

## Bundle it

```python
from rise_eo.model import bundle

bundle_dir = bundle.save(adapter, {"model": "rf.joblib"}, "path/to/bundle")
bundle.read_spec(bundle_dir)   # the model-free manifest -- no import, no model load
```

The bundle is a folder: `bundle.json` (the `module:attr` adapter reference + the declarations from
step 1 + relative artifact paths) plus the artifact file(s). Two things make this the shippable
unit, not just a convenience:

- **`read_spec` validates a run without importing your model** — a model-free preflight check that
  works even if the model's dependencies aren't installed where the check runs.
- **The bundle carries your adapter's source** (spec 44), so the inference image needs no copy of
  it. The adapter must live in a module or package file: a class defined in `__main__` or a
  notebook cell cannot be reloaded on a node, and `save` raises with the fix.
  `examples/eurocrops_rf.py` is a standalone module for exactly this reason.

### What `code=` embeds

| you write | what lands in `code/` |
|---|---|
| `code=None` (default) | the adapter's own module (`['my_adapter.py']`), or for a package adapter (`my_pkg.adapters:X`) the whole `my_pkg/` tree, layout kept |
| `code=["./demo_model/my_adapter.py", "./demo_model/helper.py"]` | exactly these files |
| `code=["./demo_model"]` | the **whole folder**, minus caches, virtualenvs, compiled files (`.pyc`, `.so`, …) and dotfiles; a `Dockerfile` or a wheel rides along |
| `code=False` | nothing: the adapter must be pip-installed in the inference image |

The import root is found by walking up `module.count(".") + 1` folders from the adapter's file:
`my_adapter` → its own folder; `my_pkg.adapters` → the folder above `my_pkg/`. `<bundle>/code` goes
on `sys.path` on the node. Auto-detection follows the module, not its imports, so list a sibling
helper yourself. Keep every embedded file under one folder. `save` checks both before copying (spec
45: #71, #72): an embedded file's sibling import that is not embedded, or an adapter that is not at
the top of `code/`, raises a `ValueError` naming the fix.

Check a saved bundle without importing anything:

```python
from rise_eo.model import bundle
m = bundle.read_spec("path/to/bundle")
print(m["fsd_bundle_version"], m["adapter"], m.get("code"), m.get("requirements"))
```

`code` must be non-null for the bundle to run on a generic (adapter-free) inference image.

## Verify it, then run it: three gates, each answering a different question

Before dispatching a many-cell fan-out, run these **in order** — each is cheap relative to the one
after it, and each answers a question the others cannot:

1. **`rise_eo.verify_adapter`** — does my adapter's LOGIC compute the right thing? Builds ONE real
   grid cell's datacube (locally, or on AML for the case that matters), lands it on your laptop, and
   runs your adapter over it through the exact same code the cluster runs
   (`rise_eo.workflows.infer_only_task.run_infer_only`) — so you can open `output.tif` in QGIS before
   trusting the bundle at all. **Does NOT check:** the inference image (see step 2), scale (one
   cell is not the fan-out), or any cell but the one it ran.

   ```python
   report = rise_eo.verify_adapter(
       bundle_dir, roi="your_roi.geojson", catalog_filepath=catalog,
       startdate=..., enddate=..., mosaic_days=20, bands=required_bands,
       export_folderpath="data/verify_adapter",   # the cube + output.tif land HERE, locally
   )
   assert report["pass"], report["error"]
   ```

2. **`rise_eo.model.verify_image`** — will the inference IMAGE actually run this bundle? Submits ONE
   real Azure ML node running your bundle inside the target Environment (~40–380s). **Does NOT
   check:** whether the adapter's logic is correct (that's step 1 — a local run here would pass
   trivially, since the driver already has your adapter's source on `sys.path`) or scale.

   ```python
   from rise_eo.model import verify_image

   report = verify_image(
       bundle_dir, environment="fsd-infer-sklearn:3",
       runner_kwargs={"cluster": ..., "root": ..., "identity_client_id": ...},
   )
   assert report["pass"], report["error"]
   ```

3. **`rise_eo.run_inference`** — the fan-out, N nodes. **Does NOT re-check** what steps 1–2 already
   proved; it assumes the adapter and image are both sound.

## Run it

```python
result = rise_eo.run_inference(
    bundle_dir, output_folderpath="data/predictions",
    roi="your_roi.geojson", catalog_filepath=catalog,
    startdate=..., enddate=..., mosaic_days=20, bands=required_bands,
    merge=True,
)
```

`model` accepts either a live adapter instance or a bundle path — a live adapter is auto-saved to a
temp bundle for you, but a bundle is **required** for ROI mode with `cores > 1` or `runner="aml"`,
since both cross a subprocess/cluster boundary where a live Python object can't travel.

## Where to go next

- [`examples/eurocrops_rf.py`](../../examples/eurocrops_rf.py) — the complete adapter this page
  walks through, ready to copy and adapt.
- [`run-at-scale.md`](run-at-scale.md) — run the bundled model over an AML cluster instead of
  in-process.
- [`serve-xyz.md`](serve-xyz.md) — put the resulting COGs/STAC on a map.
