# LIMITATIONS — what rise-eo cannot do today

**The one-page honest answer to "can rise-eo do X?"** — written for a *user* of the system
(and for anyone sizing a demo), not for an implementer.

**This is an index, not a register.** One line per limitation, no detail. The detail
already lives elsewhere and must not be copied here:

- a GitHub issue — the deferred *work item* (why it's parked, what the fix looks like);
  `TODO #NN` in older text is issue #NN
- `specs/` and `docs/adr/` — the signed-off design that drew the boundary in the first place

**Maintenance rule (keep it stupid simple):** a limitation is worth a row here only if a
user could *hit* it. Add the row when you find it; delete the row when it's fixed. If a
row grows past two lines, its detail belongs in an issue and this row should just point there.
**We plug a limitation when we actually hit it**, not in advance (YAGNI) — the "Trigger"
column is what "hitting it" means for each row.

---

## Data sources

| Limitation | Trigger to fix | Detail |
|---|---|---|
| **Two collections: Sentinel-2 L2A and Sentinel-1 RTC.** No L1C, HLS (granule parsers exist, no declaration), CHIRPS or ERA5. | the first real use case for another collection | #11; spec 58; `docs/adding-a-source.md` |
| **Two providers: CDSE and MPC.** S1 RTC comes from MPC only. | a third provider | #11 |
| **Two processings of one acquisition coexist in the archive, and a build over both raises.** Pick one with `processing=`. | a window mixes baselines and you want both | spec 59; ADR 0032 |
| **CDSE discovery has no retry** — one transient API blip kills a run before any download. | a long/unattended run (i.e. Batch) | TODO #43 |
| **Downloads are whole-MGRS-tile.** No windowed/partial read of a granule. | download cost dominates a small-ROI job | TODO #36 |

## Datacube

| Limitation | Trigger to fix | Detail |
|---|---|---|
| **Output resolution is the reference band's** (10 m for S2) — not configurable. | a model wants 20 m/60 m native, or a non-10 m source | TODO #1 |
| **`mosaic_method="median"` is the only one implemented.** A declared-but-unimplemented value raises. | a source/model needs mean, max-NDVI, best-pixel… | spec 34 §2a |
| **`mask_type="categorical_classes"` is the only one implemented.** No bitmask (Landsat/HLS QA) or threshold (cloud-probability) masking. | first Landsat/HLS/probability-mask source | spec 34 `[G3]` |
| **`native_grid=True` raises `NotImplementedError`** — there is no non-tiled (global-grid) build path. | first ERA5/CHIRPS-style source | spec 34 `[G2]` |
| **Multi-CRS ROIs collapse to the single max-mean-area UTM zone** before merging; contributions from the other zone are dropped. | an ROI genuinely straddling a UTM boundary with data on both sides | TODO #5 |
| **The artifact is `datacube.npy` + `metadata.pickle.npy`**, not xarray/zarr — no lazy/chunked access, no partial read. | cubes stop fitting in memory | TODO #13 |

## Scale / cloud

| Limitation | Trigger to fix | Detail |
|---|---|---|
| **There will be no Azure *Batch* runner.** The project's Batch account has a 6 vCPU quota against a 64-core pool VM, so it cannot allocate a node; dropped rather than quota-requested. | someone needs Batch specifically (or a generic task-queue backend: AWS Batch, k8s) | `docs/reference/AZURE_INFRA.md` §3.1 |
| **Inference-on-blob (`run_inference(roi=…, runner="aml")`) is FULLY VALIDATED on the real cluster (2026-07-28) — Phases 0-3 all GREEN.** Phase 0 (env + adapter smoke, D3/D4), Phase 1 (single-MGRS-tile ROI → 9 cells → 9 COGs + STAC; found + fixed the grids.geojson GDAL-on-blob bug, `9422a1a`), Phase 2 (D6/D7 resume + D13 guard), and **Phase 3 (`AT_ROI` → 300 cells, 16 shards, 300 COGs + STAC, `n_failed == 0`, `bundle_loads == 16`, wall 2066.9 s)** and **Phase 4 (strict single-CRS `merge` -> one 14.1 MB `merged.tif` on blob, wall 1082.1 s)**. Phase 3's two earlier failures were a *label set* passed as `roi=` (spec 21 D-GRID-1), not an infrastructure limit. **Known cost shape: 52.5 % of wall was driver overhead, undecomposed** (TODO #59). `deploy` and the pre-built-cubes `run_inference` path stay local-only (unchanged, D14 scope). | the overhead share matters, or a non-demo-scale ROI | spec 38; spec 21 D-GRID-1; TODO #59 |
| **P4's crash-resume is per-cell, not per-shard-atomic**: a shard that crashes mid-cell loses only that cell's un-pushed scratch (each cell publishes atomically via D5's `to_cog` remote branch); a re-dispatch skips every cell whose `output.tif` already exists on blob (D6) and rebuilds only the unfinished tail. Same honest limitation shape as spec 37 D8's download resume, cheaper here (a crashed shard re-runs only its cells, not the whole download). | a crash-resume actually happens on the cluster | spec 38 D6/D12 |
| **An inference image serves one dependency family** (e.g. sklearn), not every model: the bundle carries the adapter's code, but a model needing new packages needs a new image. Images are declared and built with `rise_eo.image` (`docs/howto/build-the-images.md`); the build is an operator step, not automatic. | a model with a new dependency family | spec 44; spec 56; ADR 0002 |
| **A crashed AML download job loses its un-pushed scratch.** A fresh-node resume re-downloads the unpushed remainder: it cannot see COGs already on blob, since the push is whole-run. Cheap for MPC (only the crashed shard's slice re-runs); costs re-downloaded bytes for CDSE. | a crash-resume actually happens | spec 37 D8 |
| **CDSE creds delivered via blob JSON (`--creds-url`) sit as plaintext at rest on blob**, unlike the Key Vault path, because the operator has no Key Vault *write* role (`ForbiddenByRbac`). Mitigated by a `_secrets/` prefix and a file scoped to one run, deleted in a `finally`. Rotate the CDSE keys if a run was long or the prefix is broadly readable. | a Key Vault write role becomes available | spec 37 D5 REVISED |
| **`create_training_data(download=True, runner="aml")` end to end is unrun on the cluster.** Its flatten leg alone is validated (900 blob cubes → one single-node reduce, 2026-07-27). | someone runs it and reports back | spec 39 |
| **The single-node flatten reduce has a memory ceiling** — `np.concatenate` allocates one new contiguous array and copies every input, so peak memory ≈ 2x the flattened total (all per-cube arrays + the result). **MEASURED (runbook 39 Phase 1, 2026-07-27): 900 `AT_2018_TRAIN` cubes → `data.npy` = 8.29 MB (172,781×8×3 uint16), so peak ≈ ~16 MB** — an order of magnitude below the spec's "tens-to-low-hundreds of MB" estimate, trivially within one node's RAM. Untested at 10⁴–10⁵ cells, where a streaming/partial-reduce would be needed. | someone flattens 10⁴+ cells in one call | spec 39 D3 §9; TODO #56 |
| **There is no write retry in the storage seam**: a failed blob write raises on the first attempt. Deliberate; see `ARCHITECTURE.md` §4, "the `InvalidBlockList` write retry". | a genuine transient write error is observed on DISTINCT blobs | #57 / #58 |
| **`run_inference(roi=…)` tiles the ROI's CONVEX HULL into grid cells**, so a sparse ROI still yields a contiguous, partly-empty cell set — a region-wide crop-map fan-out, not one-cell-per-feature. Not a bug (it's how you get a contiguous map), but a cost surprise: each cell is a **full ~49 km² datacube** at `grid_size_km=5`. **`AT_ROI.geojson` → 300 cells** (measured 2026-07-28; **299 as of spec 46 D4, 2026-08-19** — one cell was fully covered by another and is now dropped). For fewer cells: a smaller/compacter ROI, a larger `grid_size_km`, or `inference_datacubes=` mode over pre-built cubes. **⚠️ `roi=` takes a REGION, not a label set** — passing `AT_2018_TRAIN.geojson` (900 field polygons) there is what produced the bogus "1167 cells" figure in earlier docs. | a roi-mode run costs more than expected | spec 21 D-GRID-1; spec 46 |

## Serving / outputs

| Limitation | Trigger to fix | Detail |
|---|---|---|
| **rise-eo serves nothing.** It emits COGs + STAC; a stock server turns them into XYZ tiles (`docs/howto/serve-xyz.md`), and none is hosted. | someone needs a hosted map | #26; #63; `ROADMAP.md` P5 |
| **No render config on outputs** — nothing tells a viewer how to colour a class raster. | first output shown to a non-author | TODO #28 |
| **The STAC Collection's `classification:classes` lists only the *masked* SCL values, with placeholder names.** Misleading to an external STAC consumer; rise-eo itself is unaffected. | an external tool actually reads our STAC | TODO #45 |

## Models

| Limitation | Trigger to fix | Detail |
|---|---|---|
| **A model adapter is hand-written Python.** No config-only path. | a non-programmer needs to plug a model in | TODO #19 |
| **✅ Root cause closed (spec 38 D7)**: the bundle now loads once per core per node (default, `cubes_per_task` groups cells) or once per node (`cores=1` heavy-model opt-out), not once per cell. The fine-grained per-phase timing breakdown (load/build/predict/save) TODO #25 also asked for is still open. | per-cell inference time is dominated by load | TODO #25 |
| **One worked example: single-band classification (EuroCrops RF).** No regression / multi-band-output example. | first regression or multi-output model | TODO #18 |
| **Don't publish to one registry name from two places at once** (`deploy`, `ensure_environment`): there is no lock, and one publisher can get back a version that holds the other's model or image. | two people build or deploy the same name at the same time | #175; spec 52 §5 |

## Data on disk (not code — state)

| Limitation | Trigger to fix | Detail |
|---|---|---|
| **The local Austria archive predates spec 59, and current rise-eo refuses to read it.** Every catalog written before spec 59 lacks its four columns and raises on read; there is no migration. | any real-data run | #119; `docs/reference/test-data.md` |
| **The `rise` blob COGs carry the pre-fix (wrong) GDAL offset tag** — a titiler `unscale=true` render of them would be all black. | before ever serving those blob COGs | TODO #44 |

## Legacy capabilities not carried over

From the pre-fsd repos (`fetch_satdata`, `rsutils`, `cdseutils`), left out on purpose. The decisions behind them are in ADR 0009 and the specs; the full table is `DROPPED.md` at tag `docs-archive-2026`.

| Capability | Why it was left out | Reconsider when |
|---|---|---|
| Sentinel-2 **L1C**, and **s2cloudless** cloud masking (L1C-only, heavy dependency) | rise-eo is L2A-first | an L1C use case returns |
| **Planet** datacube path | a same-CRS special case, not a download source | a uniform-grid source is needed |
| **SSH / cluster fetch** (`fetch_from_cluster`) | infrastructure-specific | a multi-machine workflow without blob storage |
| **ESA WorldCover / WorldCereal** TIF generators | reference-data tooling, not core | reference layers are needed in-repo |
| Sentinel-Hub **Process API** download (evalscripts) | rise-eo downloads whole granules | small on-the-fly composites are needed |
| `rsutils` plotting, `rich_data_filter`, `esa_download`, `utils_preprocess` (SAR scaling, patch-finding) | not on the data-prep path; plotting belongs in notebooks | per need |
| Band ops: rolling `median_mosaic` (window/step), `sav_gol` smoothing, `trim_bands`, `modify_bands_chunkwise` | not used by any band sequence | a training-time rolling mosaic, temporal smoothing, timestamp trimming, or out-of-memory band math |
| Preprocess-log save and replay | not on the band path | a preprocessing recipe must be persisted and replayed |
