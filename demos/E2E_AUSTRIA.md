---
status: current
summary: The local end-to-end Austria walkthrough with real timings; a tutorial and a benchmark report fused into one doc (spec 41 D10) pending extraction into docs/tutorial.md. ⚠️ its step-3/4 numbers are stale — measured 2026-07-13, before `aggregate="median_per_id"` (spec 40 A2); see TODO #62.
---

# Running fsd locally — the end-to-end guide (Austria)

**This is the go-to doc for running fsd on your own machine.** It walks the whole local pipeline on
real, freshly-downloaded Sentinel-2 data, explains every stage (including **how to bundle your own
model**), and records real timings you can extrapolate from. Driver: `demos/e2e_austria.py`
(spec 23). Model quality here is not meaningful (a demo RF) — this validates the *plumbing + timings*.

---

## 1. What fsd does locally

```
  rise_eo.download            rise_eo.create_training_data                 rise_eo.run_inference(roi=…)
 ┌────────────┐   catalog  ┌──────────────────────────┐   bundle   ┌───────────────────────────┐
 │ CDSE STAC  │──────────▶ │ build datacubes over the │ ─────────▶ │ tile ROI → S2 cells;      │
 │ + S3 .jp2  │  (parquet  │ label fields → FLATTEN → │  (model)   │ per cell: build cube +    │
 │ → COG      │   + COGs)  │ features.npy  → train RF │            │ infer → COG; +STAC +merge │
 └────────────┘            └──────────────────────────┘            └───────────────────────────┘
   step 2                     steps 3–4                               step 5
```

The user-facing surface is **four verbs**: `rise_eo.download`, `rise_eo.create_training_data`,
`rise_eo.run_inference`, `rise_eo.deploy` (the last a P6 stub). Every verb **preflights** — it asserts
bands / T / inputs and fails fast with a clear message *before* any heavy compute, so most mistakes
surface in seconds, not after an hour.

---

## 2. Where the imagery comes from — `rise_eo.download` is a separate step, and why (CDSE now; MPC later)

- **Download is its own verb, run once, up front** (`rise_eo.download` → a `catalog.parquet` + local
  COGs). The compute verbs (`create_training_data`, `run_inference`) **read** that catalog and
  **never call CDSE themselves.** If imagery is missing they stop with an **actionable download
  plan** (the exact `rise_eo.download(...)` to run), never a silent auto-fetch. Step 2 of the demo *is*
  that download.
- **Why separate — three reasons, not just quota:** (1) **CDSE quota** — one controlled pull, not N
  pulls from N workers; (2) **redundant fan-out** — 1,000 grid cells over overlapping MGRS tiles
  would otherwise refetch the same bytes 1,000×; (3) **the Batch model** — download once → storage
  (local now, Azure Blob later) → compute reads from the storage seam.
- **CDSE needs the download because it serves quota'd S3 `.jp2`**, which fsd **converts to COG on
  arrival** (the native ingest format the datacube build reads fast). So with CDSE, `download` is
  **required** before the compute verbs.
- **How this changes with Microsoft Planetary Computer (MPC), later.** MPC serves **COGs over public
  HTTP + STAC** — no quota, and no conversion needed. fsd will read MPC pixels **in place** through
  the GDAL VSI path it already uses for raster reads; an MPC-backed catalog is just a `TileCatalog`
  of `/vsicurl/https://…` COG URLs. The verbs don't change and there's **no `if source==…` branch** —
  point them at an MPC catalog and it "just works" because there is nothing to materialize, so the
  missing-data guardrail never fires. **CDSE → `download` required; MPC → `download` optional.**
  *(MPC is forward design — not built yet; TODO #21.)*
- **CDSE is intermittently flaky (BUG-001), and that's expected.** Discovery (STAC) is **anonymous**;
  only the byte download needs **S3 keys**. The download is **idempotent** (skips files on disk),
  fail-fast with a **circuit breaker**, and wrapped in a **resume-loop** — if a pass trips on a bad
  CDSE window, **just re-run the script; it resumes** and completes the remainder. A slow/failing
  pass is not a fsd bug.

### The safe download runner (`rise_eo.sources.download_cli`)

The demo calls the download **engine** (`cdse.download_resume`) in-process, but for real / large pulls
there is a **standalone CLI over the same engine** — `python -m rise_eo.sources.download_cli` (spec 26) —
that adds the operational controls a one-off call doesn't need (it's also what the Batch runner will
dispatch later):

- **`--dry-run`** — metadata-only preview (STAC query + plan, **zero band bytes**): how many granules
  are *needed / present / missing*, so you **size a pull before committing** to hours of transfer.
- **`--stop-file PATH`** — graceful stop: `touch` the file and the run halts *new* submissions and
  drains what's in flight (no partial files left); delete it and re-run to resume.
- **`--max-concurrent-s3 N`** — number of parallel S3 transfer streams (default 4). More is **not**
  always faster — read the three rates below.
- **`--result-json` / `--expected-json`** — a machine-readable `_result.json` per run (status, counts,
  GB, throughput) and a pass/fail diff against expected criteria; the process **exit code** doubles as
  PASS/FAIL for scripting. Everything stays **idempotent + resumable**, same as the in-demo download.

```bash
# size the full pull first (no bytes fetched):
.venv-modeldeploy/bin/python -m rise_eo.sources.download_cli \
    --roi shapefiles/AT_ROI.geojson --start 2018-04-01 --end 2018-09-30 \
    --bands B04 B08 B8A SCL --max-cloudcover 70 \
    --dst tests/outputs/imagery --catalog tests/outputs/imagery/sentinel-2-l2a/catalog.parquet \
    --max-tiles 200 --dry-run
```

**Read the three throughput numbers** it (and the demo's step 2) report:
- **probe** — single-stream achievable MB/s *right now* (one file, one thread).
- **per-stream** — bytes ÷ thread-summed transfer time (the average each of the N streams saw).
- **wall / aggregate** — bytes ÷ transfer wall-clock (the honest *all-streams* effective rate).

Compare **probe vs aggregate**: aggregate ≈ probe (or below) ⇒ **link-bound** (one stream already
saturates your uplink; extra streams don't help — or hurt); aggregate ≫ probe ⇒ concurrency is paying
off. On the §8 run, probe 26 vs aggregate 17 MB/s → link-bound, and 4 streams were *slower* than 1.
**That verdict is environment-specific** — it inverts on a datacenter NIC, so `--max-concurrent-s3` is
tuned per environment, not baked in (§9; and the Azure re-tune, TODO #24).

---

## 3. Prerequisites

- **Python 3.11** and the isolated demo venv (keeps fsd core lean):
  ```bash
  python3.11 -m venv .venv-modeldeploy
  .venv-modeldeploy/bin/pip install -e ".[dev,grid,model-example]"
  ```
  The extras are split deliberately: **`[grid]`** = `s2`/`s2cell` (ROI→S2-cell tiling — *not* in
  core, which is why `rise_eo.grid` is an optional import); **`[model-example]`** =
  `scikit-learn`/`joblib`/`matplotlib` (the demo model + plots, *not* fsd itself — fsd never
  trains); `[dev]` = ruff/pytest; `[azure]` = `adlfs` (Phase-2, unused in v1).
- **CDSE credentials** — a gitignored `cdse_credentials.json` with the legacy keys
  (`sh_clientid`/`sh_clientsecret` for discovery, `s3_access_key`/`s3_secret_key` for download).
  Point the script at it with `--creds /path/to/cdse_credentials.json` or `$CDSE_CREDENTIALS_JSON`.
  S3 keys **can expire** — the script warns via `CdseCredentials.is_expired`.
- **No VPN required.** fsd talks to CDSE over the public internet; a VPN only affects *your* measured
  throughput (see §10) — it is never a setup step to reproduce.
- **(Optional, serving) `.venv-titiler`** — for §8's "serve the crop map to STACNotator" step
  (spec 29): a *separate* isolated venv, `python3.11 -m venv .venv-titiler && .venv-titiler/bin/pip
  install -e ".[titiler]"` (rio-tiler/fastapi/uvicorn). Not needed for the pipeline itself.

---

## 4. Reuse it for your own region (change 3 things)

```bash
.venv-modeldeploy/bin/python demos/e2e_austria.py \
    --roi   shapefiles/YOUR_ROI.geojson \
    --train shapefiles/YOUR_FIELDS.geojson \
    --id-col fid --label-col crop
```

Everything else — download, tiling, build, infer, merge — is unchanged, **including cross-UTM-zone
ROIs** (France, Russia): each cell builds a datacube in its own CRS, and the merged map reprojects
all cells to one CRS (`merge="reproject"`, area-dominant target, or a `merge_crs=` you choose). The
**model** (adapter + bands) is the separate, model-developer-owned part (§6). The "swap these"
inputs are also named constants at the top of the script (`ROI_FP`, `TRAIN_FP`, `ID_COL`,
`LABEL_COL`).

---

## 5. Run it

```bash
.venv-modeldeploy/bin/python demos/e2e_austria.py --creds /path/to/cdse_credentials.json
.venv-modeldeploy/bin/python demos/e2e_austria.py --fast            # 2-month window + small inference ROI
.venv-modeldeploy/bin/python demos/e2e_austria.py --tiny-download   # tiniest: also clip the DOWNLOAD
```

Smoke levels (increasing download cost): **`--tiny-download`** clips *both* the download and the
inference ROI to one small central slice (few granules, trains only on fields inside it — the
fastest true end-to-end); **`--fast`** shortens the window + shrinks the *inference* ROI but still
downloads the full ROI; no flag = full season over the whole ROI. `--cores` (training-data build
parallelism), `--infer-cores` (inference build+infer parallelism — **keep low**, each ~5×5 km cube
is memory-heavy; defaults to `max(1, cores//4)`).

**Tip:** before a large first pull, **size the download** with the runner's `--dry-run` (§2) — it
reports the granule count with zero bytes fetched, so you can sanity-check disk + time up front.

The 7 steps and what they produce (heavy artifacts under `tests/outputs/demo_e2e/`, gitignored — except the imagery, which lives in the ONE shared archive `tests/outputs/imagery/`, spec 59 D12):

| step | produces |
|------|----------|
| 0 preflight | validates creds + inputs |
| 1 tiling | `inference_s2_grids.geojson` + `figures/s2_grids.png` |
| 2 download | `tests/outputs/imagery/sentinel-2-l2a/catalog.parquet` + local COGs (shared archive); the timing report (§8) |
| 3 training data | `training_data/features.npy` — **one row per labelled field** (`aggregate="median_per_id"`) — (+ raw per-pixel `data.npy`) |
| 4 train + bundle | `rf.joblib` + `bundle/` (`bundle.json` + artifact) |
| 5 run_inference | `model_outputs/<cell>/output.tif` per cell + `stac/` + `merged.tif` |
| 6 plots | `figures/ndvi_timeseries.png`, `figures/crop_map.png` |
| 7 report | `timings.json` (incl. the `cost_model`) + the ETA estimator |

---

## 6. Bundling YOUR model — the step-by-step guide

**fsd owns the plumbing** (download, datacube, flatten, tiling, COG/STAC/merge, runner). **The model
developer owns the two endpoints** that connect a model to fsd. Write a small adapter (the
`ModelAdapter` Protocol is duck-typed — any framework; subclass `BaseModelAdapter` for defaults, so
it's ~10 lines), then `bundle.save`. See `demos/adapters.py::DemoRF` for a complete worked example.

**6.1 Declarations** (read at preflight, before any heavy compute):
`required_bands`, `n_timestamps` (T the model was trained on — leave `0`/model-determined if it
varies per model), `output_dtype`, `output_nodata`, `output_band_names` (1 name → categorical map;
N → probabilities/regression).

**6.2 Endpoint ① — datacube → model input** (developer-owned). The feature transform, declared
**once** and run by fsd at **both** training and inference (the anti-skew guarantee — train and
serve see identical features):
- `feature_sequence` — a `rise_eo.bands.modify` pipeline `[(fn, kwargs), …]` on the 5-D
  `(samples, T, H, W, bands)` contract (the primary, declarative way); **or** override
  `features(data5d, band_indices)` for logic the sequence can't express.
- `datacube_to_X(feats, band_indices)` — reshape features `(T,H,W,B)` → model input `(H*W, T*B)`.
  Default provided; override if your model wants a different layout.

**6.3 `predict(X_chunk)`** — your framework, model input → raw per-pixel predictions. fsd handles
chunking (`predict_batch_size`) and the NaN→nodata scatter.

**6.4 Endpoint ② — raw output → standard `Output`** (developer-owned). `to_output(raw, hw)` →
`Output((bands, H, W), dtype, nodata, band_names)` — how your model's numbers become the COG bands
fsd writes. Default maps a class/vector per pixel using the §6.1 declarations; override for custom
band packing.

**6.5 `load()`** — read your artifact(s) into memory once per worker; `self.artifacts`
`{name: absolute path}` is injected by the bundle before `load()`. (fsd never trains — you bring a
trained model.)

**6.6 Bundle it:**
```python
from rise_eo.model import bundle
bundle_dir = bundle.save(adapter, {"model": "rf.joblib"}, "…/bundle")
bundle.read_spec(bundle_dir)          # the model-free manifest (no import, no model load)
```
The bundle is a folder with `bundle.json` (the `module:attr` adapter ref + the §6.1 spec + relative
artifact hrefs) and the artifact(s). **Why it matters:** it's the shippable, self-describing unit;
`read_spec` validates a run **without importing the model** (model-free preflight); and the
`module:attr` ref must be **importable** (installed package / on `PYTHONPATH`) because it crosses a
subprocess/Batch boundary — a `__main__`/notebook class won't reload.

---

## 7. The bundle on disk

```json
{
  "fsd_bundle_version": 1,
  "adapter": "adapters:DemoRF",
  "artifacts": {"model": "rf.joblib"},
  "required_bands": ["B04", "B08"],
  "n_timestamps": 9,
  "output_dtype": "uint8", "output_nodata": 255, "output_band_names": ["crop_class"],
  "feature": {"kind": "sequence", "steps": ["mask_invalid_and_interpolate", "compute_bands", "remove_bands"]}
}
```
`read_spec` reads just this (fast, no imports). `bundle.load` resolves the adapter, injects absolute
artifact paths, checks the class's declared spec against the manifest (catches code/bundle drift),
and calls `.load()`. A field the class leaves **unset** (e.g. `n_timestamps=0`) is taken from the
bundle — this is what lets **one adapter class back models trained on different T**.

---

## 8. Results — a real full run

Measured **2026-07-13** on the full **Waldviertel AT_ROI** (single UTM-33), window **2018-04-01…09-30**,
`mosaic_days=20` (**T=10**), bands `B04 B08 B8A SCL`, `--cores 8` (macOS, university wifi). **207**
Sentinel-2 granules, **300** S2 grid cells, **900** EuroCrops training fields (9 classes). *Numbers are
stitched from a resumed run — the download + training were measured on the first pass, inference on a
clean re-pass over all 300 cells (see the note under the table). Your throughput will differ (§9).*

```
step                          seconds   share
0_preflight                       0.0      0%
1_tiling                          0.8      0%
2_download                     2732.8     45%    <- transfer / convert / wall below
3_training_data                 591.6     10%
4_train_bundle                   26.9      0%
5_run_inference                2683.5     44%    <- build + infer per cell, below
6_plots                           6.0      0%
7_report                          0.0      0%
TOTAL                          6041.6    100%   (~100 min)

download:
  transfer : 2666.7 s  (44.61 GB, 16.7 MB/s aggregate / 4.4 per stream)
  convert  : 11926.0 s (jp2 → COG; summed over the ~8-proc convert pool, fully OVERLAPPED with
             transfer — the download step wall is 2732.8 s ≈ transfer wall 2688.6 s, not the sum)
  wall     : 2688.6 s  (207 granules, 1 pass, ~0.216 GB/granule)
  probe 26.3 MB/s vs effective 16.7 MB/s (aggregate) -> link-bound: 4 parallel streams were
  SLOWER than 1 (per-stream 4.4). This is a property of the local uplink, not CDSE or fsd — it
  is expected to invert on a datacenter NIC; re-tune `max_concurrent_s3` per environment (TODO #24).
```

> ⚠️ **`3_training_data` and `4_train_bundle` above are STALE (as of 2026-07-29).** This run
> predates `aggregate="median_per_id"` (spec 40 D1 amendment A2): step 3 flattened **~172k
> per-pixel rows**, where it now reduces to **≤900 field medians** before the adapter's transform,
> and step 4 fit an RF on those pixels. Both numbers should drop; neither has been re-measured.
> Any accuracy read off this run is the **per-pixel** one, which leaks a field's own pixels across
> the split — the honest field-wise figure is far lower (~0.29). Download and inference are
> unaffected. **Re-run the demo before quoting steps 3–4 or comparing them to the cluster.**

**Two steps dominate (~90%): download (45%) and inference (44%).** Everything else is noise.

> **The same pipeline on the cluster:** [`E2E_AUSTRIA_AML.md`](E2E_AUSTRIA_AML.md) is this table's
> AML counterpart — the identical ROI and 300 grid cells run on Azure ML, the one genuinely
> apples-to-apples comparison (inference), and what the cluster's fixed overhead costs.

- **Download** is transfer-bound, not convert-bound — the jp2→COG conversion runs in a separate
  process pool that fully overlaps the transfers (§Appendix A), so the step wall tracks the network,
  not the CPU. At 16.7 MB/s aggregate the 44.6 GB took ~45 min.
- **Inference = build + infer, per cell.** The 2683.5 s over 300 cells ≈ **8.9 s/cell** at
  `INFER_CORES=2` (`max(1, cores//4)`; each ~5×5 km cube is memory-heavy). Decomposed (per-cell,
  reconstructed from the build/infer file timestamps): **build ≈ 8.0 s** (scales with a cell's image
  count, 23–203) and **infer ≈ 7.8 s** — where the "infer" time is *almost entirely fixed model-load
  overhead* (a flat 6–10 s regardless of cell size; an RF predict is sub-second), because ROI mode
  reloads the bundle once per cell. Grouping cells per worker would roughly halve it (TODO #25).

**Outputs** (under `tests/outputs/demo_e2e/model_outputs/`): one COG per cell
`cells/<window>/<cell_id>/output.tif` (300), a self-contained `stac/` catalog (300 distinct Items),
and the single-CRS display mosaic `merged.tif` — **6830×6868 px, EPSG:32633, 99.2% valid**. Class mix
of the map: pasture/grassland 49.7%, buckwheat 13.7%, hemp 6.4%, maize 5.8%, the rest 4–6% each.

**Figures** (committed under `demos/figures/`, so they survive clearing the outputs folder):
`s2_grids.png` (ROI→300 cells), `ndvi_timeseries.png` (per-class median NDVI), `crop_map.png`
(the merged class map). Model quality is illustrative — the point is the pipeline (§6, bring your
own model). Validate the map in QGIS (Appendix A: visual validation).

**Serve the crop map to STACNotator (BYO-XYZ, spec 29):** `merged.tif` above can be served as a
pre-styled XYZ layer and consumed by NASA Harvest's **STACNotator** viewer via its
Bring-Your-Own-XYZ mode — the Tier-1 rung of the serving contract (`PROGRESS.md` LATEST
2026-07-14). See `RECIPES.md` "Serve the crop map to STACNotator" and the full runbook
`runbooks/29-tier1-stacnotator-byo.md`. The output STAC's Item geometry is the true S2-cell
polygon, not a bbox (spec 28) — see `runbooks/28-stac-geometry-regen.md` to regenerate it over
this run's outputs.

---

## 9. Trusting the numbers & estimating other regions

- **Read probe-vs-effective.** The **probe** is single-threaded achievable CDSE MB/s right now; the
  **effective** rate is the run's bytes ÷ transfer-time at concurrency. probe ≈ effective → you're
  **CDSE/link-bound**; probe ≫ effective → **local contention** (background downloads) or the
  concurrency cap. A VPN or a busy link lowers *both* — it's a *your-machine* caveat, not CDSE. If
  the numbers look off, re-run without background load.
- **Estimate another region without downloading it** (`demos/estimate.py`). Runtime = counts ×
  per-unit costs; the counts are exact cheap queries and this run calibrates the costs
  (`timings.json → cost_model`):
  ```python
  from estimate import estimate_run
  # e.g. "how long for full France, same window/bands?" — no download:
  estimate_run("FR_ROI.geojson", START, END, BANDS, creds=creds, cost_model=cost_model,
               max_cloudcover=70)   # -> {granules, cells, GB, download_min, convert_min, compute_min, total_min}
  ```
  It STAC-queries the granule count, `roi_to_s2_grids` the cell count, and `compute_n_timestamps`
  the T — all without downloading — then multiplies by the cost model. **Caveats:** throughput is
  environment-bound (holds within the same link); bytes/granule vary at ROI edges; the cloud filter
  is queried exactly, not modelled; CDSE flaky windows add resume variance.

---

## Appendix A — Concepts & conventions you'll meet

- **The time axis is derived, not free-form.** `T = ceil((enddate − startdate) / mosaic_days)`;
  windows are half-open `[start, end)`, each labelled by its start boundary, and **empty windows are
  still emitted** (as nodata). So **every datacube over the same `(start, end, mosaic_days)` has a
  byte-identical `timestamps` axis** — which is why multi-tile / multi-CRS cubes can stack, and
  **flatten requires it**. Your adapter's `n_timestamps` must equal this T (or be model-determined).
  Dates are localized to **UTC** to compare against tz-aware catalog timestamps.
- **On-disk artifacts.** A datacube = `datacube.npy` + `metadata.pickle.npy` (metadata is
  `np.save`'d-pickle, a deliberate cross-platform fix). **`nodata = 0`** throughout. Inference = one
  COG per cube + a STAC catalog; `merged.tif` is a display product.
- **Everything is 10 m, resampled to a *real* B08 image** — not an abstract target grid (the
  reference-image resampling rule). Other resolutions need a different known-resolution reference
  band, not just different resample params.
- **Boundary cells don't lose data.** When a grid cell straddles an MGRS-tile boundary, all covering
  tiles of the same acquisition are nodata-fill merged onto the reference grid (not collapsed).
- **`flatten` drops all-nodata pixels** and keeps `coords.npy` (per-pixel easting/northing). *Caveat*
  (TODO #16): coords across two UTM zones are fine as per-pixel IDs, but not as geography.
- **Validate visually in QGIS** — the pipeline saves RGB/composite + categorical outputs precisely
  because rasters must be eyeballed, not just unit-tested.

## Appendix B — Known v1-local limitations (these are *known*, not bugs you found)

- **COG conversion is local-dst only** — both COG-on-download and the inference-output COG require a
  local path; remote/Blob is the deferred stage-local→convert→upload path (a P1/P4 item — the "who
  converts on Blob?" question).
- **`cores > 1` requires a model bundle**, not a live adapter (it crosses a subprocess). The demo
  saves a bundle anyway; a live adapter runs sequentially.
- **The datacube rectangle carries a nodata halo** and can't hug a slanted ROI edge — cosmetic;
  `flatten` drops the nodata. Don't be alarmed eyeballing a cube in QGIS.
- **`rise_eo.deploy` is a P6 stub** (registration/push); local train + bundle + infer is the complete
  path today.

## Appendix C — Why run the full ROI (real bugs it has caught)

Small synthetic / single-field tests pass while a real, full-ROI run surfaces coverage and
serialization bugs — which is exactly why this demo exists, and why outputs get **eyeballed in QGIS**,
not just unit-tested:

- **Tile-merge coverage bug (spec 20).** An early full run showed **9 of 300 cells as ~nodata holes**
  along an MGRS-tile-row boundary. The cause was a real datacube-builder bug (not the demo): the
  builder kept only **one** MGRS tile per `(timestamp, band)`, so a cell straddling a tile boundary
  lost every *other* same-acquisition tile's coverage (worst cell: 0.6% valid despite ~80% raw
  coverage). Fixed by nodata-fill **merging all same-`(timestamp, band)` tiles** onto the reference
  grid → the dead cells rebuilt to 80%+ valid and recovered ~6% more training pixels too. (This is
  the "boundary cells don't lose data" guarantee in Appendix A.)
- **STAC item-id collision (spec 26).** The full 300-cell run produced a `collection.json` with **300
  links all pointing to one item file** — every output COG is named `output.tif`, and the item id was
  derived from that constant filename. Fixed to derive the id from the per-cell folder, with a
  uniqueness guard that now fails loudly instead of silently overwriting. A 2-cube synthetic test had
  *masked* it (it counted duplicate link-follows as distinct items).
- **Multi-UTM-zone outputs → the display merge.** A cross-zone ROI's per-cell cubes land in
  *different* CRSs (e.g. a ROI near 36°E gets **both** EPSG:32636 and 32637, since S2 zone-36 tiles
  reach past 36°E). `rasterio.merge` needs one CRS (the single-CRS-merge principle), so
  `merge="reproject"` reprojects every output COG to the area-dominant zone before mosaicking the
  display map. Austria (§8) is single-zone UTM-33, so no reprojection was needed — but the path is
  exercised, and the `--roi` swap in §4 (France/Russia-style cross-zone ROIs) relies on it working
  unchanged.
