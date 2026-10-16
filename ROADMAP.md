# fsd — north-star, roadmap & model contract

> **Status: DRAFT for sign-off (2026-07-05).** This is the strategic umbrella doc: where
> fsd is going (the north-star), how we get there in small releases (the roadmap), and the
> **model contract** that lets project teams plug their own models into fsd. It feeds the
> numbered specs in `fsd/specs/` — decisions here graduate into specs before implementation.
> Infra ground-truth for the cloud phases lives in **`docs/reference/AZURE_INFRA.md`** (the `rise`
> project). Current state: the pinned "Order of work" issue and the open PRs.
>
> The F1–F5 decisions baked into §3 are **proposed**, pending the user's sign-off.

---

## 1. North-star — THE GOAL

A researcher developing a crop/land model for some region (say Afghanistan) can, with
minimal cloud setup:

1. `pip install fsd` (a GitHub link first; PyPI later).
2. **Create training data** from a list of known-label polygons (e.g. farm plots), possibly
   across different years / date ranges / mosaic intervals depending on the model: fsd
   downloads the satellite data → builds datacubes for those polygons → flattens them into
   training arrays. The user never has to think about "flattening."
3. **Train their own model** — sklearn, PyTorch, TensorFlow, anything. **fsd does not care.**
   Training stays permanently on the user's side.
4. **Write a small model adapter** (see §3): declare the feature transform, load the model,
   run predict, shape the output.
5. **Run inference at scale** by giving fsd a **region of interest + startdate + enddate +
   mosaic interval**: fsd tiles the ROI into S2-geometry grids, downloads the imagery,
   builds inference datacubes for each grid, runs the model over them, and writes the
   results as **COGs + a STAC catalog**. The user never has to think about "tiling the ROI."
6. **View outputs** as XYZ map tiles through a hosted TiTiler (infra-side, raapid).

**The invariant that makes this tractable:** every stage works **locally (Mode A)** *and*
**scaled on the cloud (Mode B/C)** with the **same code**. Cloud is a *backend*, never a
rewrite. This is the promise we protect at every step.

---

## 2. Organizing principles (the anti-rewrite frame)

### 2.1 Three usage modes

**Moved to [`ARCHITECTURE.md` §5](ARCHITECTURE.md#5-the-three-modes)** (spec 41 D9: the code map,
invariants and modes have one home). Modes **A** (fully local), **B** (cloud data + compute, local
control and training) and **C** (fully cloud inference) — B and C were both proven on the real
cluster 2026-07-29.

### 2.2 Control plane vs data plane (data gravity)

**Moved to [`ARCHITECTURE.md` §2](ARCHITECTURE.md#2-containers--the-four-runtime-pieces).** The
driver is a thin, portable job submitter (laptop · VM · AML job · later a Function); the data plane
is heavy and must be cloud-colocated. What it costs when that line is crossed is measured in
[`docs/findings/cloud-overhead.md`](docs/findings/cloud-overhead.md).

### 2.3 Layers (swap backends without touching the core)

**Moved to [`ARCHITECTURE.md` §6](ARCHITECTURE.md#6-layers--swap-a-backend-without-touching-the-core)**
(same D9 reason as §2.1/2.2). L0 core · L1 seams · L2 deployment backends · L3 the project's
`ModelAdapter` · L4 product surfaces. The shipped L2 backends are **local (Snakemake) and Azure ML**
— this section previously named Azure *Batch*, which was the earlier plan, not what runs.

### 2.4 Two datacube types, one builder

**Moved to [`ARCHITECTURE.md` §3](ARCHITECTURE.md#3-code-map).** Training cubes come from
known-label polygons and are tiny; inference cubes come from S2 grid cells tiling an ROI and are
large. Same builder — what differs is the *source of geometries* and *what happens after*. The
opposite economics are measured in
[`docs/findings/workload-regimes.md`](docs/findings/workload-regimes.md).

### 2.5 Scope goes UP — high-level verbs hide the plumbing
Intended users shouldn't say "flatten" or "tile the ROI." fsd grows a high-level API (§4);
today's `flatten` / ROI-tiling / builder become its internals.

### 2.6 Preflight before the fan-out (fail cheap, before you pay)
Download + datacube creation are the **expensive, billable** operations. So fsd runs every
**cheaply-computable check up front** and **refuses to trigger** the heavy fan-out if any
fails — a "preflight." The goal: *know before you spend* whether a run can succeed. Checks
that cost nothing (no download) include:
- **`T` match** — inference `T = ceil((enddate-startdate)/mosaic_days)` (calendar mosaic)
  equals the model's `n_timestamps` (§3.3). This alone catches the most common deploy failure.
- **band availability** — `required_bands` are obtainable from the source.
- **ROI sanity** — non-empty, intersects the source's data coverage / catalog.
- **adapter loads** and declares a coherent, self-consistent spec (bands, output, T).
- **storage + credentials reachable** (blob writable, secrets present).

Preflight is a first-class product feature (users iterate on config safely) *and* a cost
guardrail (no surprise Batch bills for runs that were doomed at config time).

### 2.7 The five "get-it-right-early" surfaces (these gate specs)
A wrong early choice here forces broad rewrites, so each gets a deliberate spec **before**
we scale:
1. **Catalog format** → STAC (§6).
2. **Storage URI abstraction** → fsspec (good) + MSI/adlfs + GDAL-VSI auth (to prove).
3. **The model contract (ModelAdapter)** → §3.
4. **Runner / control-plane seam boundary** → `runner=local|batch` never leaks into the task CLI.
5. **Model-artifact packaging** → the model *bundle* (§3.4).

---

## 3. The model contract (L3 — `ModelAdapter`)

Extracted from the legacy `demo_02_model_train` + `model/demo_model_deploy.py`, generalized.

### 3.1 The anti-skew invariant
The **feature transform** (e.g. `mask_invalid_and_interpolate → compute NDVI/NDRE/GCVI/SAVI
→ remove raw bands`) appears **identically** in the legacy train and deploy code, copy-pasted
— a classic train/serve-skew trap. **Rule: exactly one feature-transform definition, run by
fsd in BOTH training-data-generation and inference.** [F1]

### 3.2 The interface (strawman)
```python
class ModelAdapter(Protocol):
    # --- declarations fsd reads to build compatible datacubes & outputs ---
    required_bands: list[str]            # e.g. ["B02","B03","B04","B05","B06","B07","B08","B11","B12"]
    n_timestamps: int                    # T — the number of mosaic windows the model was trained on.
                                         #     fsd DERIVES T from startdate/enddate/mosaic_days (calendar
                                         #     mosaic, spec 15) and requires an exact match. [contract = same T + bands]
    output_dtype: str                    # e.g. "uint8"
    output_nodata: int | float           # e.g. 255
    output_band_names: list[str]         # 1 name = categorical map; N = probabilities/regression

    # --- feature engineering: ONE definition, used at train AND inference [F1] ---
    feature_sequence: list[tuple[callable, dict]]        # fsd bands.modify vocabulary (primary)
    # def features(self, datacube, band_indices) -> (data, band_indices): ...   # arbitrary escape hatch

    # --- model lifecycle ---
    def load(self) -> None: ...                          # load artifact ONCE per worker
    def datacube_to_X(self, features, band_indices): ... # reshape features → model input
    def predict(self, X_chunk): ...                      # raw output; fsd owns the chunk loop [F2]
    def to_output(self, raw) -> "Output": ...            # → (bands, H, W) + dtype/nodata/names [F3]
```

**fsd owns** (both modes): datacube/array I/O via the storage seam; running `feature_sequence`;
the predict **chunking loop** (`predict_batch_size`, default = whole tile) [F2]; assembling
the output array into a **COG** using the datacube's `transform`/`crs`; and building the
**STAC catalog** over the outputs. **The user owns** only: the feature declaration, the
reshape, the model call, and the output shaping.

### 3.3 Requirements declaration + validation (preflight — see §2.7)
The model's input width is fixed at train time (e.g. RF sees `T×B`). **The contract is: same
`T` and same `bands`.** fsd **validates this before any download/build**:
- **`T` (timestamps):** thanks to spec-15 **calendar** mosaicing, `T` is a pure function of
  `startdate`/`enddate`/`mosaic_days` — `T = ceil((enddate - startdate) / mosaic_days)`
  (final window upper-inclusive). So fsd computes the inference `T` from the user's chosen
  dates/interval and checks `T == adapter.n_timestamps` **at config time, before a single
  tile is fetched.** (This is a concrete payoff of the calendar-mosaic decision.)
- **bands:** `adapter.required_bands` must be obtainable from the requested source.

A mismatch **refuses to trigger the run** with an explanatory error ("your startdate/enddate/
mosaic_days give T=18 but the model needs T=19") — never a silent reshape failure deep in a
fan-out after money has been spent on downloads.

**Honest asymmetry (surfaced, not prevented):** the demo trains on per-field *median* samples
[F4] but predicts per *pixel*. That's a legitimate modeling choice — fsd documents it, doesn't
block it.

### 3.4 The model bundle (packaging — surface #5)
For deploy, especially on cloud workers, the adapter **code** + model **artifact(s)** + the
**spec** (required bands, output spec, feature sequence) must travel together and be loadable
anywhere. fsd defines a self-describing **bundle** and can push/register it (later: to ACR /
blob / an AML-registry-like store). [F5: code-first — a Python class referenced by
import-path/entry-point in config; no no-code path early.]

### 3.5 Division of labor (summary)
| Concern | Owner |
|---|---|
| Feature engineering *definition* | user (declared) |
| Running the feature transform (train + inference) | **fsd** |
| Reshape features → model input | user (`datacube_to_X`) |
| Model load / predict | user (any framework) |
| Predict chunking/batching loop | **fsd** |
| Output → `(bands,H,W)` + dtype/nodata/names | user (`to_output`) |
| Array → COG (+ transform/crs) | **fsd** |
| STAC catalog of outputs | **fsd** |
| Training the model | user (permanent) |
| ROI → S2 tiles, download, datacube, flatten | **fsd** (hidden behind verbs) |

---

## 4. High-level user API (the new verbs — §2.5)

Sketch (signatures to be firmed in specs):
```python
rise_eo.create_training_data(
    label_polygons, dates, mosaic_days, bands,
    feature_sequence, aggregate=None,       # "median_per_id" | None | callable  [F4]
    runner="local", storage=...,            # runner + storage are seams (local|batch, local|blob)
) -> TrainingArrays                         # data/ids/labels/metadata  (hides flatten)

rise_eo.run_inference(
    roi, dates, mosaic_days, model_bundle,
    runner="local", storage=...,
) -> COGs + STAC catalog                    # hides ROI→S2 tiling, datacube build, COG/STAC

rise_eo.deploy(model_bundle, ...)               # register a bundle for scaled inference
```
`runner=` and `storage=` are the seams: the **same call** runs Mode A (local, local disk) or
Mode B/C (Batch, blob) by config alone — no code change.

**ROI → S2-grid tiling** (inside `run_inference(roi=…)`, **P0.75** — the tiling itself landed as
`rise_eo.grid`, spec 19) is a **port, not an invention** — the
legacy `rsutils.s2_grid_utils.get_s2_grids_gdf` already does it, and
`fetch_satdata/notebooks/demo_preparation.ipynb` pins the intended recipe:
1. `get_s2_grids_gdf(roi_geojson_4326, grid_size_km=5, scale_fact=1.1, res=None)` — map
   `grid_size_km`→S2 level (**5 km → res 11**, as in the legacy `100_random_grids`),
   `s2.polyfill` the ROI's **convex hull** at that level, keep cells that **intersect** the
   ROI, then **scale each cell by `scale_fact` (1.1 → 10 % overlap per side)** so adjacent
   grid cells don't leave seams at mosaic time.
2. `gpd.overlay(grids_gdf, roi_gdf)` — clip the scaled grids to the ROI → the per-cell
   geometries that feed datacube creation (each **grid cell** = one inference datacube = one
   **per-cell task**).
3. Deps: `s2` (`from s2 import s2`, gives `polyfill`) + `s2cell`. ✅ ported into `fsd/grid.py`
   (spec 19); consumed by the **P0.75** wrapper, not P4. Not needed for P0's `create_training_data`.

---

## 5. Roadmap — small, demoable releases

Each phase is a shippable release to show the team. **Infra-ask** flags where we must *propose*
a `raapid-infra` change to the admin (we never edit it ourselves).

### 5.0 ⭐ THE DEMO TARGET (locked 2026-07-21, user)

Everything below is background; **this is what we are driving at**, and anything not on this
path is a distraction until it ships:

> **A researcher runs the same fsd pipeline on Azure at scale — `runner="aml"` /
> `storage="abfss://…"` is *configuration*, not a rewrite.** That is **P2 → P4** (TODO #41).
> Audience: **a researcher who would actually use fsd**, so the self-serve bar applies —
> `pip install`, docs, adapter-writing, error messages that teach.

> **Wording widened 2026-07-21 (user), from "on Azure **Batch** at scale" / `runner="batch"`.**
> The promise being locked is **the seam** — cloud as a swappable backend — not a product name, and
> hardcoding the vendor's product into the north-star is precisely the failure mode that cost us a
> session to detect. *Why Batch was displaced:* run-book `36-runner-fork-probe` (tag `docs-archive-2026`) measured the
> project's Batch account at a **6 vCPU** dedicated core quota against a **64-core** pool VM — it
> cannot allocate one node — while the AML `d16` cluster offers 512 cores today under **the same
> managed identity spec 31 already proved against blob**. Full evidence: `docs/reference/AZURE_INFRA.md` §3.1.
> **Batch is dropped, not deferred** (user, 2026-07-21 — strict YAGNI): AML ships the demo, and the
> seam is already demonstrated by local-Snakemake ↔ AML. Revisit only if something actually needs it.

**Everything local is already done and proven on real data** (P0 → P0.9), and the storage half
of the cloud story is proven too (P1: runbook `31-p1-datacube-on-blob.md` green,
`34-download-to-blob.md` green on both sources). **The single missing piece is the runner.**

⚠️ **A tension to hold consciously, not resolve by accident.** "Researcher self-serve" and
"runs on Batch" pull apart: a researcher without `rise` credentials + VPN **cannot run the Batch
path at all**. So the honest shape of the demo is *"the local path is theirs to run; the cloud
path is the same call with two arguments changed, which we drive."* Protecting that — one code
path, cloud as a backend — **is** the demo. If the Batch runner ends up needing its own pipeline
code, the demo has failed even if it runs fast.

**What is left, in order** (detail: `docs/reference/AZURE_INFRA.md` §3.1/§7/§8, TODO #41):

1. ~~**Decide Batch vs AML.**~~ ✅ **RESOLVED 2026-07-21 → AML** (`cluster-<proj>-d16`), by measurement
   rather than argument: run-book `36-runner-fork-probe` (tag `docs-archive-2026`) + `docs/reference/AZURE_INFRA.md` §3.1. Batch dropped.
2. **Write spec 36 (the scale-runner spec)** — settles §7's remaining questions: where the driver
   runs, blob data layout, the job environment/image, the `--runner=` seam, idempotency under
   retries, telemetry. *(§7's task-granularity question is already answered — see below.)*
3. **Implement the runner seam** + fix the local Snakemake sentinels' blob-unsafety (both TODO #41;
   the second falls out of the first, since §8.1's atomic-rename publish replaces the sentinels).

**Two things got materially cheaper on 2026-07-21, and both are worth naming:**

- **Zero infra asks.** The old plan needed two platform-admin changes (a Batch core-quota bump and
  probably `max_tasks_per_node`). The AML clusters are provisioned, quota'd, and identity-attached
  **today** — so P2 no longer waits on anyone. The project's "first infra proposal" is deferred
  indefinitely.
- **No container build blocking the path.** "Container image + ACR push — the largest genuinely new
  build" was a Batch requirement. AML builds and versions the job environment itself, so an image is
  an optimization, not a gate. (`az acr build` is available server-side if we ever want one — the
  registry is Basic SKU with public access, so no local Docker is needed either.)

**Task granularity is decided (user, 2026-07-21): one dispatched unit = a *shard* of `input.csv`,**
executed by the **existing local Snakemake runner** inside the job. The cloud runner shards the work
list and launches copies of code that is already proven — the strongest form of "cloud is a backend,
not a rewrite." Retry/resume is per-shard, made safe by per-cube skip-if-exists (§8.1).

✅ **No spike is needed first.** GDAL/VSI auth under MSI — long carried as *the* technical
unknown (`docs/reference/AZURE_INFRA.md` §7.3) — was **solved and proven on real Azure by spec 31**
(`rise_eo.raster.rio_open` → `/vsiadls/` + fresh token; runbook `31-p1-datacube-on-blob.md` green
2026-07-18). §7.3/§8 are now marked resolved. The residual is narrower and lands in **P4**, not
here: GDAL *writes* to blob for inference-output COGs are still unproven (`rio_open` raises on a
remote `mode="w"` by design), i.e. TODO #39. **P2 is design + build, not discovery.**

**Not on the path** (real, but do them when we hit them — `LIMITATIONS.md`): more data sources,
xarray/zarr, the config-only adapter, tile serving. Serving (P5) is the natural *next* demo
after this one, not part of it.

| Phase | Ships | Demo | Infra-ask |
|---|---|---|---|
| **P0** | fsd pip-installable (GitHub); high-level verb *skeletons*; STAC-aligned catalog (§6) | `pip install fsd`, local download→datacube→flatten via `create_training_data` | none |
| **P0.5** | ✅ **DONE (spec 18, 2026-07-06)** — **ModelAdapter contract** + legacy train/deploy reimplemented on it, **fully local**. `src/rise_eo/model/` (adapter/features/engine/bundle), `run_inference` real, `create_training_data(adapter=…)`, self-describing bundle. | Mode A end-to-end: EuroCrops RF train → inference → COG + STAC, one plug-in adapter (`tests/manual/deploy.md`) | none |
| **P0.75** | ✅ **DONE (spec 21, 2026-07-07)** — **Local ROI inference verb (completes Mode A).** `run_inference(roi=…)` chains **tile the ROI (`rise_eo.grid.roi_to_s2_grids`) → build one datacube per grid cell → infer → COG + STAC (+ optional display merge)** in a single call, all local. The per-cell **build+infer** is one runner-dispatched unit-of-work (`workflows/infer_task.py` + Snakefile → **Batch swaps in at P4 unchanged**, not a second pool). `merge=True\|"reproject"` (strict vs lossy display merge). Imagery assumed present in the catalog — **inference never calls CDSE** (conserve quota, SO-6). *(`create_training_data(roi=…)` deferred — labelled field shapes need no ROI→cell tiling.)* | Mode A: one call turns an ROI GeoJSON into per-cell crop-class COGs (`tests/manual/roi_inference.md`) | none |
| **P0.9** | ✅ **DONE (spec 23, 2026-07-10)** — **Local-completeness gate + team run-book.** `demos/e2e_austria.py` runs the *whole* local pipeline on **fresh real CDSE data** (download → jp2→COG → datacube → flatten → train → bundle → ROI build+infer → COG/STAC/merged), a **reusable template** (swap `--roi/--train`), **cross-UTM-zone-safe** (`merge="reproject"` area-dominant/`merge_crs`). Adds **decomposed download timing** (transfer vs COG-convert) + a **throughput probe** (factor out link/VPN), the **`plan_download` guardrail** (missing imagery → actionable `rise_eo.download` plan; verbs never auto-fetch), and a **no-download `estimate_run`** (ETA for any region/window/bands). Doc: `demos/E2E_AUSTRIA.md`. | the go-to "how fsd runs locally" doc + trustworthy timings | none | 
| **P1** | Storage seam on Azure: adlfs/MSI read+write to the `rise` project storage; GDAL-VSI auth proven | build a datacube locally but I/O against `rise` blob (over VPN) | none (uses existing) |
| **P2** | Azure **AML** runner for datacube fan-out (the runner seam; spec 36) | N datacubes built across the autoscaled `rise` AML `d16` cluster | **none** (was: Batch quota bump + `max_tasks_per_node` — moot since the fork went to AML) |
| **P3** | 🟡 **PARTIALLY DONE (spec 39, 2026-07-24, pending cluster validation)** — **create-training-data-at-scale.** `create_training_data`/`flatten_training_data` gain `runner="aml"`: an optional download phase (`api.download`, reused), the spec-36 build fan-out (reused), and a new **single-node flatten reduce** (`workflows.runners.run_aml_flatten`, D3 — flatten is a reduce over ALL cubes, not a per-cell fan-out) whose compact output lands back on the laptop (`api._land_local`, D4, `storage.transfer`). The driver stays control-plane-only (ADR-0004) — only the flattened array (+ optional `features.npy`, computed on the driver, ADR-0020) comes home, never the raw cubes. All unit tests mocked at the AML-client boundary — run-book `39-training-data-on-aml` (tag `docs-archive-2026`) Phases 0-2 are the pending real-cluster validation. Docs: `docs/adr/0020`, `CONTEXT.md`. | Mode B: laptop triggers cloud download→build→flatten, pulls the flattened arrays | none new |
| **P4** | ✅ **DONE (spec 38, 2026-07-23, pending cluster validation)** — **Inference at scale.** `run_inference(roi=…, runner="aml")` dispatches the P0.75 per-cell build+infer unit-of-work onto the **`rise` AML cluster**, reusing the P2 datacube-fan-out runner (`workflows.runners.run_aml_inference` + a new node entrypoint `workflows/infer_shard.py`) — a **thin runner/dispatch swap** (`runner=`/`storage=`/`runner_kwargs=` config), no new pipeline algorithm, **plus** the I/O-seam fixes the swap exposed: remote-dst COG (`raster.cog.to_cog`, closes TODO #17), bundle-loaded-once-per-node (closes TODO #25's root cause), a dedicated inference AML Environment (adapter + deps, an operator run-book step), and a one-node adapter-import smoke before the fan-out. 31 tests, all mocked at the AML-client boundary (no test needs Azure) — run-book `38-inference-on-aml` (tag `docs-archive-2026`) Phases 0-3 are the pending real-cluster validation. Docs: `docs/adr/0001`, `docs/adr/0002`, `CONTEXT.md`. | Mode C: the P0.75 ROI verb fanned out across AML nodes | maybe scale `max_nodes` |
| **P5** | Output STAC + hosted TiTiler / XYZ | outputs viewable as web tiles | TiTiler hosting (infra) |
| **P6** | Deploy/registration UX; model-bundle push/register | one-command deploy of a bundle | model store (infra) |

**P0–P1 need zero infra changes** — pure de-risking, two team-visible releases before we're
ever blocked on someone else's `terraform apply`. The first infra proposal is P2 (Batch quota).

### 5.1 Goals after the demo target

The demo target (§5.0) was met on 2026-07-29, when the cluster demo ran. These goals followed it, in
order. The current goal's state lives in its tracking issue and the pinned "Order of work" issue.

| Goal | Set | Status |
|---|---|---|
| **fsd as an installed module, driven from a notebook** | 2026-08-21 | ✅ **Done** (maintainer, 2026-10-07). A notebook in a separate repository, with fsd installed as a pinned dependency, ran the full pipeline and the work was demoed. Built by specs 54–57 and issues #78–#82; `v0.1.0` was tagged on 2026-09-04. |
| **Collection-agnostic verbs** (spec 58) | 2026-09-04 | 🅿️ **Parked** (maintainer, 2026-10-07). Sentinel-2 L2A and Sentinel-1 RTC work through the same verbs; Sentinel-1 ran on AML on 2026-09-29. HLS (spec 58 P3) is parked: new satellite sources are meant to come from contributors, not from the maintainer alone. MODIS was deferred on 2026-09-04. |
| **Contributor readiness** (spec 102) | 2026-09-29 | Current. See #102. |

### 5.9 Post-v1 sequencing (user, 2026-07-02) — historical, moved from the old TODO file

> Moved here verbatim when the old TODO file became a stub (spec 41 D8/P2, 2026-07-30). **Point-in-time:
> this is what the ordering looked like on 2026-07-02** — steps 1 and 2 have since happened, and
> §5's phase table above is the live plan. Kept because it records *why* the order was chosen.

The item numbers are stable IDs, **not** priority order (they are now GitHub issue numbers). The
intended order of the big efforts:

1. **Finish v1 end-to-end** — download → datacube → flatten, local Snakemake runner
   (current focus; datacube module #5 is next).
2. **Azure setup + Batch processing** for datacube creation — the cloud-agnostic
   scale-out end goal (`specs/10`). Comes **before** any source extension.
3. **Source extension, incrementally** (issue #11; promotes the source contract to the
   `sources/base.py: Source` ABC, OQ-3): CDSE **S2 L2A** (done) → **MPC S2 L2A** →
   CDSE **S1 GRD / S1 RTC** → **MPC S1** → other products (CHIRPS, ERA5, …).
4. ~~**Benchmark against `rslearn`** (issue #12)~~ — **DONE and DECIDED 2026-07-31; issue #12
   closed.** See §7.1 below.

The **cross-cutting perf track** that used to sit under this heading (_datacube-creation speed at
scale_, the 3-part benchmark-first plan and its five parked optimization candidates) moved to
**issue #15**, which is that item — see its migration comment.

---

## 6. STAC decision — now, but deliberately thin

**Do it in P0, before Azure.** Rationale: the catalog format is surface #1 — it ripples into
inference outputs, cross-source catalogs (CDSE + future sources), and TiTiler (which expects
STAC). Deciding late = migrating data *and* rewriting the reader twice. It's cloud-agnostic and
locally testable, and there's a natural bridge (our catalog is GeoParquet; STAC has the
`stac-geoparquet` serialization; the legacy deploy notebook already builds a `pystac` catalog
of output COGs).

**Scope guard:** "our catalog is STAC-*valid* and round-trips to STAC Items" — **not** "we run
a STAC API server" (that's L4/infra/later). A tight spec will name exactly which STAC fields we
populate (`proj:epsg`, `proj:shape`, `proj:transform`, datetime, bbox, COG asset) and stop.
Applies to **both** the post-download catalog *and* the inference-output catalog (one catalog
abstraction).

---

## 7.1 ✅ RESOLVED 2026-07-31 — rslearn (Plan B vs Plan C)

**fsd will not use rslearn as a data-download source** — not wholesale, not as a hybrid source
behind an extra. **Plan B stands**; `spike/rslearn` does not merge into `main`. Issue #12 is
closed.

**The framing that decided it:** fsd is an alternative to **Google Earth Engine**, for running
*simpler* models at scale over satellite imagery. rslearn is a **foundation-model library** that
happens to ship an acquisition layer. Two tools, two jobs — merging the acquisition layers is a
category error.

**The successor project:** rslearn's real value is its **model library** and deploying heavy
foundation models. That becomes its own project — *run rslearn on Azure the way fsd now can*. The
transferable asset is fsd's **scale-out know-how** (the runner seam, the storage seam, the AML
environment and identity work), not fsd's pipeline code.

**What was measured** (rslearn v0.1.13 @ `a5c50c63`, Azure VM, 2026-07-31): rslearn's `T` is
data-dependent (9 groups where fsd gives 10; 7 where fsd gives 9 once a period is empty); its
period yields **one first-coverage scene** where fsd takes a per-pixel **median**; the install is
**5.3 GB** with no lite path and does not import out of the box; its blob reader runs **~22 MB/s
against fsd's ~107 MB/s** via `/vsiadls/`, which **vindicates spec 31's design** for fsd's
workload; and it ships **no distributed runner** at all. Against that, rslearn has **50** concrete
data-source entry points to fsd's 2 plus a foundation-model zoo fsd has no analogue for — the
strongest argument for adoption, and the reason the successor project exists.

**One finding carries straight into that project:** rslearn reads *and writes* ADLS Gen2 under
managed identity **unmodified**, for the price of `pip install adlfs`. So rslearn-on-Azure does
not begin with a storage problem — it begins at the runner.

**Two byproducts land on fsd regardless:** rslearn's lazy per-read MPC signing answers issue
**#32**'s design question, and its `class_path` + `init_args` pattern is a good model for the
`Source` ABC that issue **#11** still owns — which **does not exist yet** (`sources/__init__.py:5`;
`api.py:264,413` dispatch on a hardcoded `("cdse", "mpc")` check, duplicated).

**Evidence:** [`spike/RSLEARN_SPIKE_REPORT.md`](spike/RSLEARN_SPIKE_REPORT.md) — §6.4 is the
decision, §6.5 explains why it cut differently from the report's own recommendation. The report
also supersedes [`docs/findings/RSLEARN_COMPARISON.md`](docs/findings/RSLEARN_COMPARISON.md) (2026-07-06), which was wrong
about fsd's calendar-`T` contract being unique.

---

## 7. Open questions / to-confirm

- **F1–F5 sign-off** (§3) — the proposed contract decisions.
- ~~`timestamp_contract` strictness~~ — **RESOLVED (2026-07-06): same `T` + same bands**;
  `T` derived from calendar params and checked in preflight before download (§2.6, §3.3).
- **Coupling of `TileCatalog`** to its current schema — determines how cheap the STAC move is.
- **Where the model bundle is stored/registered** on cloud (ACR? blob? AML registry?) — P6.
- **Multi-band feature vocab limits** — is `bands.modify` expressive enough, or is the callable
  escape hatch load-bearing from day one?
- Everything in `docs/reference/AZURE_INFRA.md` §7–§8 (Batch task granularity, driver host, GDAL-VSI auth,
  data layout, container image, idempotency at scale).

---

*Maintenance: update phase status as releases ship; fold each resolved decision into its
numbered spec. Cross-refs: `docs/reference/AZURE_INFRA.md` (infra), the pinned "Order of work" issue (current state),
`specs/` (signed-off designs).*
