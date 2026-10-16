# fsd — domain glossary

The ubiquitous language for the fsd pipeline (download → datacube → flatten → inference at scale). A
glossary only: terms specific to this project, defined by what they *are*. Design lives in `specs/`,
decisions in `docs/adr/`, terminology gotchas also in `CLAUDE.md`.

## Model & packaging

**Bundle**:
A portable, self-describing folder — trained weights + a `module:attr` adapter code *reference* + the
model spec (bands, T, output dtype/nodata/names). fsd stages and loads it to run a model; it does **not**
contain the adapter code or its dependencies.
_Avoid_: model package, model artifact (overloaded).

**Manifest** (`bundle.json`):
The bundle's self-declaration: a `module:attr` adapter code reference, an `artifacts` map (name →
relative file path — the bundle's *table of contents*), and the model spec. Lets fsd list a bundle's
files and validate a run **without loading the model** (model-free preflight), and lets a node fetch
exactly the files the bundle names.
_Avoid_: index, metadata (too generic).

**Adapter** (`ModelAdapter`):
The user's class declaring required bands / T / output spec + the one feature transform + `load`/predict.
Referenced by a bundle; installed as a package in the inference Environment. Feature transform is run by
fsd at *both* training-data generation and inference (the anti-skew invariant).
_Avoid_: model (that's the weights), wrapper.

**Inference Environment** (a.k.a. inference image):
The AML Docker Environment a node runs inside = `fsd[azure,mpc]` + the user's adapter package + its
runtime deps. Built once per adapter version, referenced by name. Distinct from the generic datacube
build Environment.
_Avoid_: container (too generic), bundle (that's weights+reference, not the image).

## Sources & collections

**Source**:
The *provider* fsd fetches bytes from — `"cdse"`, `"mpc"`. Decides authentication, transport, and
whether the native bytes need converting. One source hosts many collections.
_Avoid_: satellite, provider (say Source), data source (redundant).

**Collection**:
The *product* being fetched, named by its STAC collection id — `"sentinel-2-l2a"`,
`"sentinel-1-rtc"`, `"hls2-s30"`. Decides bands, mask, radiometry and grid; it is what a
`SourceDeclaration` describes. Orthogonal to Source: one collection may be served by several
sources.
_Avoid_: satellite (S1 GRD and S1 RTC are the same satellite; HLS is two), product, dataset.

**Collection declaration** (`CollectionDeclaration`):
The collection-level facts the datacube builder needs — reference band, mask spec, nodata, mosaic
method — registered per collection id and stamped into the catalog so an artifact self-describes.
Provider-independent: `sentinel-2-l2a` declares the same thing whether it came from CDSE or MPC.
_Avoid_: source declaration (it describes a Collection, not a Source), config, schema.

**Canonical band name**:
The band vocabulary fsd addresses bands by — STAC EO `common_name` values (`red`, `nir`, `nir08`,
`swir16`, …). Each Collection declares its own mapping from these to its native asset keys, so one
adapter's `required_bands` works across collections. fsd declares the mapping itself; a provider's
published `common_name` is not trusted.
_Avoid_: band alias (that's the mapping, not the name), common band, band code.

**Mosaic partition**:
The catalog properties a Collection declares must hold a single value within one datacube build —
`sentinel-1-rtc` declares `sat:orbit_state`, optical collections declare nothing. It exists because
some products are harmonized for compositing and others are not.
_Avoid_: orbit filter (that's the user-facing knob), grouping key.

## Archive (spec 59, ADR 0032)

**Archive**:
The root a `download` writes into — `{root}/{collection}/YYYY/MM/DD/{granule}/{band}.tif`, one
`catalog.parquet` per collection directory. Shared across runs; lossless (nothing is ever replaced).
_Avoid_: imagery folder, download folder (both named one run's copy, which it no longer is), cache.

**Granule**:
One downloaded unit: one acquisition at one processing — one catalog row, one folder. For
Sentinel-2 it covers one MGRS tile.
_Avoid_: tile (ambiguous — see MGRS tile vs grid cell), product (in this glossary a Collection is
never called a product), scene, item (that's the provider's STAC record, whose id may differ).

**Canonical granule name**:
A granule's provider-independent name, keeping its processing fields — for Sentinel-2 the ESA product
name without `.SAFE` (`S2B_MSIL2A_20180918T100019_N0212_R122_T33UWP_20201009T023142`), whether MPC or
CDSE served it. The folder name and the catalog `id`. Equal across sources exactly when the processing
is the same.
_Avoid_: item id (MPC's drops the baseline field), SAFE name.

**Acquisition**:
The physical observation — one platform, one sensing time, one footprint — independent of how often
it was processed or who serves it. Several granules can share one.
_Avoid_: scene, capture, date (many acquisitions share a date).

**Acquisition key**:
The canonical granule name minus its processing fields (`S2B_MSIL2A_20180918T100019_R122_T33UWP`).
Stored as a catalog column; used only to **detect** that one acquisition has more than one processing
in a build, never to name a folder.
_Avoid_: dedup key (nothing is deduplicated by it), granule id.

**Processing version**:
Which processing produced a granule's bytes — S2 baseline `"05.00"`, HLS `"2.0"`; null where the
provider publishes none (S1 RTC). Named after STAC `processing:version`; compared as a PEP 440
version, which is what `processing=` specifiers (`">=04.00"`) select on — on `download` (what to
fetch) and on the build verbs (what to use).
_Avoid_: baseline (S2-only word), collection version, product version.

## Grids & work units

**Grid cell** (a.k.a. S2 grid cell):
The ~5 km ROI subdivision on the S2-geometry grid (`rise_eo.grid.roi_to_s2_grids`), id like `165b09c`. One
grid cell = one inference datacube = one build+infer unit-of-work = one `output.tif`. Never write a bare "tile": say MGRS tile or grid cell.
_Avoid_: bare "tile" (ambiguous with MGRS tile).

**MGRS tile** (a.k.a. satellite / Sentinel-2 tile):
The ~110 km source granule on the military MGRS grid, id like `T33UWP`. What fsd downloads and merges
across; catalog column `mgrs_tile`.
_Avoid_: bare "tile".

**Unit-of-work**:
The runner-dispatched task. For inference it is the per-cell **build-the-datacube-then-infer** step
(`workflows.infer_task`). The property that keeps "scale = swap the runner, not the code" honest.
_Avoid_: job (a job carries a *shard* of units, not one unit).

**Shard**:
A slice of the per-cell work list (`input.csv`) assigned to one node by `shard_units` (round-robin). One
shard = one AML job. Within a shard, cells are further grouped for bundle-load amortization.
_Avoid_: batch, partition (the shards *form* a partition; a shard is one slice of it).

**Run**:
One dispatched execution, identified by `run_id`, with all its inputs/outputs/telemetry laid out under
`<root>/runs/<run_id>/` (`shards/`, `_status/`, `_bundle/`).

**Demo run**:
One end-to-end execution of the whole pipeline for a region — the thing `demos/e2e_austria.py` (local)
and its cluster sibling perform. A demo run **contains** several *runs* in the sense below (the cluster
one dispatches four: download, cube build, flatten reduce, inference) plus driver-only work that
dispatches nothing.
_Avoid_: run (reserved, below), pipeline run.

## Documentation kinds (ADR 0026 — four things, never one)

**Benchmark harness**:
A script whose purpose is to **measure** the pipeline — stepwise timings, throughput probes, resume,
structured logging. `demos/e2e_austria.py` and `demos/e2e_austria_aml.py` are these (531 and 1056 lines,
of which 12 and comparably few touch `fsd`), as is everything under `benchmarks/`. A harness may
deliberately **bypass the public API** to reach a measurement — `e2e_austria.py` calls
`cdse.download_resume` directly rather than `rise_eo.download` — which is exactly why it cannot double as
an example. Its write-up is a **benchmark report**: point-in-time, statused, keeping results and
appendices.
_Avoid_: demo (the directory is misnamed and stays so, ADR 0026), example, guide.

**Example**:
A minimal, readable, copy-paste script that composes fsd's verbs and nothing else — no timing, resume,
signal handling or plotting. ~60–80 lines, lives in `examples/`. The artifact a reader **edits into
their own pipeline**. Correctness is demonstrated by being short enough to read.
_Avoid_: demo, benchmark, snippet.

**Tutorial**:
A **learning-oriented** document that narrates one example on **fixed** data the maintainers control,
and **must not fail** — responsibility for the reader's success lies with the teacher (Diátaxis).
`docs/tutorial.md`, driven by spec 42's committed offline fixture. Not for the reader's own region:
guaranteed success is impossible on data we have not seen.
_Avoid_: demo, guide, how-to (opposite reader), quickstart (implies the README's 60-second block).

**How-to guide**:
A **task-oriented** recipe for someone who already has a goal and some competence — "run at scale",
"use your own region", "rebuild the AML images". Lives in `docs/howto/`. **Cannot promise safety**
(Diátaxis): the user owns getting into and out of trouble, so a how-to states its prerequisites and how
to diagnose failure. Cloud how-tos may legitimately begin "file a ticket with your platform admin".
_Avoid_: tutorial (opposite reader), runbook (below), guide (ambiguous between the two).

**Run-book**:
A how-to **fused to acceptance criteria** — exact commands, expected outputs, PASS/FAIL, and a
`_result.json` the user pastes back (spec 24). It exists because Claude does not run pipeline/networked
scripts. This fusion is deliberate and makes run-books excellent *evidence* and poor *documentation*;
distilling their reusable parts into `docs/howto/` is spec 41 P7.
_Avoid_: how-to (a run-book additionally proves the step worked), tutorial.

**Step**:
One of the labelled parts a demo run is divided into and timed by — `0_preflight`, `1_tiling`,
`2_download`, `3_training_data`, `4_train_bundle`, `5_run_inference`, `6_plots`, `7_report`. The same
labels on the local and cluster sides, so the two are comparable line by line. A step may dispatch one
run, several, or none.
_Avoid_: phase (means a run-book's manual stage — "run-book 38 Phase 3" — and is not this), stage.

**Reduce job**:
A dispatched unit that collapses MANY inputs into ONE output, run as a single node — the opposite shape
from a fan-out. The flatten reduce (`workflows.runners.run_aml_flatten`, spec 39 D3) reads every cube
named in `input.csv` and writes one training array; no `shard_units`, `n=1` always.
_Avoid_: shard (a shard is one slice of a fan-out; a reduce job is not sharded).

## Training data

**Label** (as a separable overlay):
The class assigned to a training polygon. **Separable from training-data generation** — the flattened
training array is keyed by per-pixel `id`; labels attach by joining on `id`, and can be re-derived
(combine classes, split/invent classes) across experiment iterations **without re-flattening**. Required
for model deploy / supervised training; **optional** when *generating* training data (you often download
+ flatten once, then iterate labels many times).
_Avoid_: target, y (fine in code, not the domain term).

## Control plane vs data plane

**Driver**:
The machine that orchestrates a run — tiles the ROI, runs `setup`, stages inputs, submits jobs,
aggregates `_status/*.json`, assembles the run-level STAC. Either the operator's laptop on VPN or a VM
in-region; **which one is a property of the run, not a detail** — the driver does per-unit blob I/O, so
its distance from the data moves measured timings substantially. A run records where its driver ran.
Preflight and fan-in happen here.
_Avoid_: client, control node.

**Dispatch telemetry**:
The dispatcher's own record of a run's shape — when each job was submitted, admitted and finished —
written to `<run_root>/_timing.json` beside `_status/`, as the run proceeds. Durable by design: timings
held only in driver memory are lost on a crash and unrecoverable afterwards, which is why reconstructing
them once cost a session of forensics against job history and blob mtimes. Runner-agnostic — a non-AML
dispatcher writes the same file.
_Avoid_: metrics, profiling (this is a run's own record, not a sampling profiler).

**Node**:
An AML cluster worker executing one shard's units, reading imagery/catalog/bundle from blob and writing
COGs + STAC items back to blob. Runs the *same* local orchestration a laptop runs — the runner is the
only thing that differs.
_Avoid_: worker (ambiguous), instance.

**Dispatcher**:
The `runners.run_aml*` function on the driver that shards a work list, submits one job per shard, waits,
and raises on any failure. The only place that knows about AML; the unit-of-work never does.
_Avoid_: runner (the *local* Snakemake orchestration is also a "runner"; the dispatcher is the *cloud*
one).

**Job admission**:
The wait between the dispatcher submitting a job and that job's code beginning to execute on a node —
queueing, node allocation, container image pull, process start. A property of **each job**, not of the
cluster: on an already-scaled cluster nothing "starts up" and admission is still dominated by the image
pull. Cluster **scale-out** (the autoscaler adding nodes) is not measured separately; it is read off the
*spread* of admission times within a dispatch, since late-admitted jobs are the ones that waited for a
node.
_Avoid_: cluster start-up (wrong in the common warm-cluster case), cold start (conflates the image pull
with the node), spin-up.

**Land-local**:
Bringing a run's compact output home to the operator's laptop after a cloud dispatch, via
`storage.transfer` (single-object, atomic) — never the raw inputs, only the small result (spec 39 D4:
the flatten reduce's `data.npy`/`coords.npy`/`ids.npy`/`metadata.pickle.npy`/`labels.npy?`). Keeps the
driver control-plane-only: it orchestrates and receives results, it does not pull bulk data over the WAN.
_Avoid_: download (ambiguous with `rise_eo.download`, the imagery-fetch verb).
