# ARCHITECTURE

The code map. **Where things live, what must stay true, and what runs where.**

Read this before changing code. For *why* a decision was made, see [`docs/adr/`](docs/adr/); for
how the code got this shape — the forks taken and dropped — [`docs/history.md`](docs/history.md);
for where fsd is going, [`ROADMAP.md`](ROADMAP.md); for what a term means,
[`CONTEXT.md`](CONTEXT.md).

Modules and types are **named, not deep-linked** — symbol-search beats a line number that rots.

---

## 1. Context — what fsd is, and what it is not

```mermaid
graph LR
  U["Researcher<br/>(writes a ModelAdapter,<br/>trains their own model)"]
  F["<b>fsd</b><br/>download · datacube · flatten<br/>· inference · STAC"]
  C["CDSE<br/>Sentinel-2 L2A"]
  M["Microsoft<br/>Planetary Computer"]
  A["Azure ML + Blob<br/>(the <i>rise</i> platform)"]
  S["pgSTAC + titiler<br/>→ STACNotator"]
  U -->|"ROI, dates, bands,<br/>labels, an adapter"| F
  C -->|granules| F
  M -->|granules| F
  F <-->|"jobs + artifacts"| A
  F -->|"COGs + STAC + render config"| S
```

**fsd never trains a model.** Training is permanently the user's side of the line — fsd produces
training arrays and consumes a bundled adapter. It also builds **no dashboard**: it emits standard
STAC + COGs, and stock infrastructure serves them.

## 2. Containers — the four runtime pieces

C4's "container" means *a separately runnable thing* — [its own docs say **"Not
Docker!"**](https://c4model.com/diagrams/container). fsd's containers are therefore these, and
emphatically **not** the AML Docker environments (those are packaging for the node).

```mermaid
graph TB
  subgraph Control plane
    D["<b>driver</b><br/>laptop · VM · AML job<br/>submits work, collects results<br/>never touches pixels"]
  end
  subgraph Data plane
    N["<b>node</b><br/>one CLI unit-of-work<br/>build a cube · infer a cell · download a shard"]
  end
  B[("<b>blob</b><br/>granules, cubes, outputs<br/>+ the model and image registries<br/>fsspec: local ≡ abfss://")]
  K[("<b>catalog</b><br/>GeoParquet<br/>+ STAC export view")]
  D -->|"dispatch (runner seam)"| N
  N -->|"read/write (storage seam)"| B
  N -->|rows| K
  D -->|query| K
  D -->|"collect + STAC"| B
```

**The driver/node split is the single most important thing in this document.** The driver is a thin,
portable, authenticated job submitter — laptop (VPN + `az login`), a VM, an AML job, later a
Function. The data plane is heavy and **must be cloud-colocated**: compute next to storage.

Keeping the driver thin is *why* all three driver locations are cheap to support — and its cost when
violated is measured: [`docs/findings/cloud-overhead.md`](docs/findings/cloud-overhead.md) shows a
per-output-unit collect loop running on the operator's laptop over VPN was **35 % of a 2067 s run**.
Removing those round-trips took that window from **777 s to 36 s** on a 299-cell run (spec 57).

**Two registries live on blob**, addressed through the same storage seam as everything else: the
**model registry** (a name and a version for a bundle) and the **image registry** (a name and a
version for a resolved image definition). Both are stores, not services — a directory layout plus
an all-or-nothing completion marker, readable by anything that can read a blob. fsd never hard-codes
where they are; the location is an argument, optionally defaulted from user config.

## 3. Code map

| module | responsibility |
|---|---|
| `fsd/api.py` | **the public verbs** — `download`, `create_training_data`, `flatten_training_data`, `run_inference`, `verify_adapter`, `deploy`. Preflight lives here: fail cheap, before any spend. |
| `fsd/sources/` | `cdse.py`, `mpc.py` — discover + fetch granules. `_s2_radiometry.py` derives offset/scale from the processing baseline. `download_cli.py` is the safe shell runner. |
| `fsd/catalog/` | `catalog.py` = `TileCatalog`, the GeoParquet query format. `stac.py` = an **additive export view**; `stac_geoparquet.py` exports it in the form pgSTAC loads. `declaration.py` carries radiometry through. `inspect_cli.py` / `restamp_cli.py` are operator tools. |
| `fsd/datacube/` | `builder.py` merges granules into one cube per geometry; `_instrument.py` the opt-in timing sidecars; `ops.py` array transforms; `flatten.py` the reduce into training arrays. |
| `fsd/bands/` | band math on the 5-D contract. |
| `fsd/raster/` | rasterio primitives: `cog.py` (COG conversion, remote-dst branch), `images.py`. **The home of the GDAL/VSI exception to invariant 1** — the mosaic path in `api.py` and the COG writer in `model/engine.py` also open rasters directly, and are the only other places that may. |
| `fsd/grid.py` | `roi_to_s2_grids` — an ROI becomes S2 grid cells, one cell = one work unit. |
| `fsd/model/` | `adapter.py` the `ModelAdapter` contract, `bundle.py` packaging (carries the adapter's own source since spec 44), `engine.py` inference, `features.py`. `registry.py` = **a name for a bundle**, on the storage seam. `verify_image.py` answers "does this image actually run this bundle?". |
| `fsd/workflows/` | `task.py` / `infer_task.py` / `infer_only_task.py` / `shard.py` / `infer_shard.py` / `create_datacube.py` / `download.py` / `flatten.py` = the CLI units-of-work. `runners.py` = the runner seam (local Snakemake, AML). `stamp.py` = **the one identity-stamp helper** — "were these artifacts derived from exactly *this* request?", never a modification time. `adapter_smoke.py` is the import-level gate. |
| `fsd/image/` | an AML node image **declared as data**: `definition.py` (a frozen `ImageDefinition` that renders a Dockerfile but never builds one), `digest.py` (resolve every moving reference to a fixed one, then hash), `registry.py` (the versioned store of resolved definitions). |
| `fsd/aml/` | `environment.py` — **the only module in fsd that shells out to `az`**. `ensure_environment` builds a declared image if that digest is not already registered, and reuses it otherwise. Azure coordinates are always arguments; fsd hard-codes none. |
| `fsd/registry/` | `_core.py` — generic version allocation, aliases, and the all-or-nothing completion marker shared by the model and image registries (each keeps its own ref parsing and name checks). |
| `fsd/config.py` · `fsd/cli.py` | the user-level config (`~/.config/fsd/config.toml`) and the **`fsd` console script** that writes it — `fsd init`, `fsd config`. Operator-facing: **the library itself never reads the config file**, it takes locations as arguments. |
| `fsd/progress.py` | one shared throttled ticker (rate + elapsed + ETA). Every driver-side loop that can run for minutes prints through it rather than growing a second copy. |
| `fsd/secrets.py` | a thin Key Vault read, authenticated by the same managed identity as blob. |
| `fsd/storage/` | `fs.py` the fsspec seam every module uses; `azure.py` the `az://` URL form. |

**Key types:** `TileCatalog` · `ModelAdapter` / `BaseModelAdapter` · `TrainingData` ·
`InferenceResult` · `Output` · `PreflightError` · `ImageDefinition` · `EnsureResult` (which carries
**both** versions, because AML versions *assets* while the registry versions *definitions* — they
are not the same number, and assuming they were cost a real run).

**Two datacube types, one builder:** training cubes (one per labelled field, tiny — a median cube is
14 × 15 px) and inference cubes (one per grid cell, large — 597 × 554 px). Same code, opposite
economics; see [`docs/findings/workload-regimes.md`](docs/findings/workload-regimes.md).

## 4. Invariants

Important invariants are **an absence of something** — that is what makes them checkable:

1. **No module reaches a *remote* path outside `fsd.storage`.** This is what makes local ≡
   `abfss://` a config change. Two documented exceptions, both narrow:
   - **raster pixel reads**, which go through rasterio/GDAL VSI (`raster/`, plus the mosaic path in
     `api.py`);
   - **node-local scratch and CLI-local files** — bare `open()` on a path that is always on the
     local disk (a staged bundle, a `_result.json`, a Snakemake sentinel, a scratch GeoTIFF).
     `api.py`'s scratch opens carry an inline comment saying so.

   The line to hold is therefore: **a bare open is only ever legal on a path that cannot be
   remote.** Anything that could be a URL goes through `fsd.storage`.
2. **No direct `boto3`.** S3 transport is `s3fs` with any `endpoint_url`; a tile download is
   `storage.transfer(src, dst)`.
3. **The unit-of-work never knows how it was scheduled.** `workflows/task.py` reads its inputs and
   writes its artifact; it cannot tell Snakemake from AML. This is what makes a new backend a
   dispatch swap, not a pipeline rewrite.
4. **fsd never trains a model.** It produces training arrays and consumes a bundle.
5. **fsd builds no dashboard.** It emits STAC + COGs + a render config; stock pgSTAC/titiler serves.
6. **Ingest stores raw DN.** Radiometry is *declared* as metadata, never baked into pixels.
7. **An ROI is one region, not a label set.** One cell = one row = one work unit; ids are unique.
   Violating this cost two failed cluster runs — issue #58, spec 21 D-GRID-1.
8. **Verbs never auto-fetch.** Missing imagery yields an actionable plan, not a surprise download.
9. **No skip decision reads a modification time.** Two unsynchronised clocks make a timestamp
   meaningless between a driver and a node, and a blob's `Last-Modified` is read-only — it cannot
   carry *when this content was produced* across a copy at all. Asserted directly rather than
   trusted: `test_no_mtime_read_in_build_skip_logic` and `test_no_mtime_read_in_flatten_skip_logic`
   scan the skip logic's own source for `getmtime` / `st_mtime` / `os.stat`.

   What replaces it is **identity** — a stamp recording the request that produced an artifact
   (`workflows/stamp.py`), used for the flatten skip and for reusing a landed cube. **Presence is
   still what the cube-build skip keys on**, deliberately and with a known limit: a *truncated*
   cube passes a presence test, which is why downloads and cube writes being atomic is the named
   prerequisite (#74, #76) rather than a size comparison bolted on now.

   The failure mode all of this exists to prevent is **existence standing in for identity**, and it
   has cost real runs: a resume that re-inferred a *different* ROI (#66), and a cube reused as
   "already landed" and then stamped with the wrong request's identity.

**Conventions that go with them:** raster ops take and return `(data, profile)` so they chain;
band math uses the 5-D contract `(samples, timestamps, height, width, bands)` plus a `band_indices`
dict; nodata is 0; cubes over one start/end/`mosaic_days` share an identical `timestamps` axis
(`T = ceil((end-start)/mosaic_days)`), which `flatten` requires.

**Code that looks simplifiable but is not:**

- **`_dt2ts` attaches UTC to a naive date** (`datacube/ops.py`). Catalog timestamps are tz-aware;
  a user's `startdate`/`enddate` arrive naive, and pandas raises when it compares the two. Any user
  date compared with catalog times goes through it.
- **`metadata.pickle.npy` is pickled through `np.save(allow_pickle=True)`**, not raw `pickle`. A
  cube's metadata (geometry, the timestamp of each slice, dimension names) is a dict that cannot
  live in the array. A raw-`pickle` file written on macOS failed to read on Ubuntu; going through
  `np.save` was stable across both. xarray could one day replace the dict + array pair.
- **Bands are resampled *to a real reference image* (B08, 10 m)**, never to an abstract target
  grid, because the resampler does not reliably line up with one. Another resolution means another
  reference band, not new resample parameters. Unmeasured so far: [#100](https://github.com/nikhilsrajan/fsd/issues/100).
- **MGRS tiles are moved into one CRS before `rasterio.merge`**, which needs a single CRS: the zone
  with the largest mean `area_contribution` wins.
- **Raster output is checked by eye in QGIS.** Unit tests do not prove pixels are right; a change
  to raster code ships with an inspectable output (gate 4 in `CONTRIBUTING.md`).

**Approaches tried and reverted** (each will look like a good idea again):

- **The `InvalidBlockList` write retry** (`storage/fs.py`, reverted 2026-07-28, #57 → #58). It
  retried writes that looked like a transient Azure block-commit race. The real cause was duplicate
  work-unit ids making 16 threads write one blob, which is deterministic, so every writer retried
  into every other and a fast, legible failure became a minutes-long error storm. The fix was to
  refuse duplicate ids at the source (`create_datacube.setup`). No transient race has been seen on
  distinct blobs. If one ever is, retry only on Azure storage error codes, never on adlfs's
  `"Failed to upload block"` prefix: a catch-all `except Exception` produces it, so auth failures
  wear it too.
- **A config file inside the checkout** (`env.example.sh` + `notebooks/_config.py`, retired by spec
  54, #78). Only a checkout could reach it, so a `pip install` consumer could not, and a project-local
  file is where someone eventually commits a live storage URL. Replaced by
  `~/.config/fsd/config.toml` + `fsd init`.
- **A checked-in image build folder** (`notebooks/images/` + `.last_registered.json`, retired by
  spec 56, #79). Same shape: only a checkout could build the image, and nobody else could answer
  "what was this image built from?". Replaced by `fsd.image.ImageDefinition`, a digest of the
  resolved fsd reference, and `fsd.image.registry`.

## 5. The three modes

| mode | who does what | state |
|---|---|---|
| **A — fully local** | the laptop does everything: download → datacube → flatten → train → inference | works today; the escape hatch for Azure-hesitant colleagues, and it never goes away |
| **B — cloud data + compute, local control and training** | the laptop is a thin remote control: it triggers download/build/flatten in the cloud, then pulls back only the **compact flattened arrays** | proven on the cluster 2026-07-29; re-run 2026-09-02 from a separate repo with fsd installed as a dependency |
| **C — fully cloud inference** | register a model + adapter, trigger by ROI + dates; the cloud fans out, runs the model, writes COGs + STAC | proven on the cluster 2026-07-29; re-run 2026-09-02 from a separate repo with fsd installed as a dependency |

The "downloading raw data to a laptop defeats the cloud speed-up" worry is really *Mode A data
locality with Mode C speed*, which is incoherent. The resolution is Mode B: **you download the
flattened result, not the raw imagery.**

## 6. Layers — swap a backend without touching the core

- **L0 — core library.** Pure pipeline functions. Cloud-agnostic; never imports Azure.
- **L1 — seams.** Storage (fsspec), runner (Snakemake → AML), control/trigger.
- **L2 — deployment backends.** Local, AML, later others.
- **L3 — project contract.** The user-supplied `ModelAdapter`.
- **L4 — product surfaces.** pip UX, config, hosted titiler/STAC. fsd *produces* the STAC + COG;
  infrastructure *hosts* the tiler.

## 7. Deployment

```mermaid
graph LR
  L["laptop / VM / AML job<br/><i>driver</i>"] -->|"az login + VPN"| W["AML workspace"]
  W -->|"scales 0→N"| P["d16 cluster<br/><i>nodes</i>"]
  P <-->|"managed identity"| S[("ADLS Gen2<br/>abfss://")]
  L -->|"reads results"| S
```

Every concrete name, id and URL lives in `AZURE_INFRA_PRIVATE.md` at the **workspace root**, never
in this repo — it is public MIT. The variables are named and verifiable in
[`docs/reference/environment.md`](docs/reference/environment.md). Bootstrap with **`fsd init`**,
which writes `~/.config/fsd/config.toml` (spec 54); `AZ_*` environment variables still override it
per shell, and `fsd config` shows the resolved value and where each one came from.

## 8. Contributing

Setup, the four gates and the review checklist are in [`CONTRIBUTING.md`](CONTRIBUTING.md); test
data is described in [`docs/reference/test-data.md`](docs/reference/test-data.md). What the
architecture adds:

- **Tests are synthetic and offline.** Anything needing credentials, a cluster or human eyes is a
  **run-book** (`runbooks/`, spec 24), not a test.
- **Docs can fail the suite** — `tests/test_docs.py` checks point-in-time status headers, config-key
  parity, that every relative link in the maintained docs resolves, and that every `fsd.<verb>(…)`
  the README calls really binds to the live signature. Adding or renaming a config key means editing
  `fsd/config.py` and `docs/reference/environment.md` in the same change.
- **Design lands as a spec first** (`specs/`), signed off before implementation.
- **Point-in-time documents are never edited after the fact** — specs, run-books, findings, ADRs,
  the progress archive. Supersede them with a new document instead.
- **Open work is GitHub Issues**, not a file. Issue numbers #1–#62 are aligned with the historical
  `TODO #NN` references.
- **Never commit a concrete Azure identifier.** Describe the identifier, never spell it.
  `tests/test_notebooks.py`'s identifier guard catches the usual shapes (GUIDs, home paths, resource
  names, personal blob prefixes) in the files it scans; anything else still needs a human eye. Prose
  about a real run has leaked identifiers four times, once in a write-up explaining that the
  identifier had been scrubbed.

