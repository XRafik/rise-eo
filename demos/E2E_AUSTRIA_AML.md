---
status: current
summary: The same pipeline timed on Azure ML, laptop-vs-cluster side by side; every number traces to a _result.json (spec 40 A3).
---

# The same demo on the cluster — AML timings, and what they cost

The counterpart to [`E2E_AUSTRIA.md`](E2E_AUSTRIA.md) §8. That section times the whole pipeline on
one laptop; this one times it on **Azure Machine Learning** and puts the two side by side.

> **Every number here comes from a `_result.json` written by the run that produced it.** The source
> file is named next to each figure, and the full map is in [§7](#7-where-every-number-comes-from).
> Nothing is estimated, scaled, or reconstructed from memory. Where a measurement does not exist the
> cell says **not measured** — those cells are the honest part of this document, not a gap to be
> filled in later with something plausible.

---

## 1. The two runs

| | local | AML |
|---|---|---|
| when | 2026-07-13 | 2026-07-22 (download, build) / 2026-07-28 (inference) |
| where | 8-core macOS laptop, university wifi | up to 16 × **16-vCPU** nodes (the `d16` cluster), West Europe |
| imagery | 207 CDSE granules on local SSD, Apr–Sep 2018, 4 bands, **44.61 GB** | 576 MPC granules (3456 assets) on blob, full-year 2018, 6 bands, **418.0 GB** |
| window | 2018-04-01 … 2018-09-30, `mosaic_days=20` → **T=10** | 2018-04-01 … 2018-09-01, `mosaic_days=20` → **T=8** |
| ROI | `AT_ROI.geojson`, `grid_size_km=5`, `scale_fact=1.1` → **300 grid cells** | identical — same file, same parameters, **300 grid cells** |
| training set | `AT_2018_TRAIN.geojson`, **900** EuroCrops fields, 9 classes | identical |
| model | `DemoRF` bundle (RF over NDVI+SAVI) | identical |

**The ROI, the cell count, the field count and the model are the same on both sides.** The window,
the band set and the imagery source are not — which is why only §4's inference row is a fair
comparison and §5 says explicitly which rows are not.

---

## 2. Local — the reference run

Reproduced from `E2E_AUSTRIA.md` §8; the authoritative lines are in
`tests/outputs/demo_e2e/full_run.log` (steps 0–4) and `full_run_3.log` (steps 1, 5–7). It is a
stitched run: download and training were measured on the first pass, inference on a clean re-pass
over all 300 cells.

```
step                          seconds   share
0_preflight                       0.0      0%
1_tiling                          0.8      0%
2_download                     2732.8     45%
3_training_data                 591.6     10%
4_train_bundle                   26.9      0%
5_run_inference                2683.5     44%
6_plots                           6.0      0%
7_report                          0.0      0%
TOTAL                          6041.6    100%   (~100 min)
```

Download detail: **44.61 GB** transferred in 2666.7 s = **16.7 MB/s** aggregate over 4 streams
(4.4 MB/s per stream); the jp2→COG conversion (11926.0 s summed across a ~8-process pool) is fully
overlapped, so the step wall tracks the uplink, not the CPU.

---

## 3. AML — the same pipeline on the cluster

The cluster run is not one script; it is six run-books, each with its own `_result.json`. The table
follows the local step order so the two can be read against each other.

| # | step | run-book | AML wall | evidence |
|---|---|---|---|---|
| 0 | preflight | — | *not separately measured* | folded into each phase's driver time |
| 1 | tiling (ROI → 300 cells) | 38 P3 | *not separately measured* | runs inside `run_inference` preflight, so it is inside step 5's 1084.3 s |
| 2 | download archive → blob | 37 P3 | **not measured** — wall **≥ 354 s** | 16 shards, 3456 assets = **418.0 GB**, **slowest shard 192.1 s**, Σ 2434.7 node-s ⇒ **171.7 MB/s per node**, 0 failed. Jobs ran 15:12:12 → 15:17:40 UTC (2026-07-22); the driver wrote its result 26 s later |
| 3a | build 900 field cubes | 36 P3 | **not measured** — wall **≥ 343 s** | 16 shards × 16 cores, 900 units, **slowest shard 213.8 s**, Σ 2851.8 node-s, 0 failed. Jobs ran 17:49:57 → 17:55:21 UTC (2026-07-22); the driver wrote its result 19 s later |
| 3b | flatten reduce → training table | 39 P1 | **405.7 s** | one single-node job; 900 blob cubes → `(172781, 8, 3)` landed locally |
| 3c | features (NDVI+SAVI) | 40 P1 | **179.4 s** | **driver-side, not cluster** (ADR-0020) → `(900, 8, 2)` |
| 4 | train + bundle | 40 P2/P3 | *not timed* | both run locally; the local run measured the pair at 26.9 s |
| 5 | inference, 300 cells | 38 P3 | **2066.9 s** | = 982.7 slowest shard + **1084.3 driver overhead**; 16 shards, `cores=1`, 0 failed, 0 skipped |
| 5b | merge → one crop map | 38 P4 | **1082.1 s** | a **second** dispatch; all 300 cells skipped, 16 no-op jobs, `merged.tif` 14.1 MB |
| 6–7 | plots / report | — | *not run on AML* | |
| | **measured subtotal** | | **3734.1 s** | steps 3b + 3c + 5 + 5b only — **not** a pipeline total |

**The two "not measured" cells are not empty rows.** Both runs stored complete per-shard `seconds`;
what nobody recorded at the time was the **driver wall**, and therefore the driver overhead. The
run-books emit `wall_seconds` today (`runbooks/36-aml-runner.md:295`,
`runbooks/37-download-on-aml.md:399`) — the stored results simply predate the instrumentation. They
cannot be recovered by re-running: both phases are resumable and their outputs are already on blob,
so a re-run would time the *skip* path.

`runbooks/41-recover-aml-job-timings.md` Steps 1 + 1b (2026-07-28) recovered as much as is
recoverable without spending: AML keeps `StartTimeUtc`/`EndTimeUtc` on all 113 jobs, so both phases'
**job-execution spans** are known (36 P3 = 324 s over 16 jobs; 37 P3 = 328 s over 16, after
discarding an earlier 8-job attempt that was **cancelled**). Adding the gap to the driver's own
`_result.json` write gives a **lower bound** on each wall — 343 s and 354 s — and a tight one, since
the driver's post-run work was only 19 s and 26 s.

**A lower bound is not a wall, and it is never printed as one.** A real `wall_seconds` still
requires `runbooks/42-timed-cold-reruns.md`'s cold replays. What the bound *does* settle is the
price: those replays are ~6-minute cluster runs, not the ~20 min this document first guessed.

---

## 4. The one fair comparison — inference over 300 cells

Both sides tiled the same `AT_ROI` into the same 300 grid cells and ran the same bundle over them.

| | local | AML |
|---|---|---|
| wall | **2683.5 s** | **2066.9 s** |
| configuration | T=10, `INFER_CORES=2`, 8-core laptop, COGs on local SSD, merge included | T=8, `cores=1`, 16 shards on 16-vCPU nodes, COGs on blob, merge excluded |
| concurrent workers | 2 | 16 |
| worker-seconds of compute | 2 × 2683.5 = **5367** | 16 × 982.7 = **15 723** |
| per cell | **17.9 s** | **52.4 s** |
| **per cell per timestamp** | **1.79 s** | **6.55 s** |

**The cloud is ~3.7× slower per unit of work, and only 1.30× faster on the wall.** It wins by
throwing 16 workers at the problem and then hands roughly half of that back as driver overhead.

Two things make 3.7× a *conservative* figure:

- The local 2683.5 s **includes** the merge and the STAC build; the AML 2066.9 s **excludes** the
  merge entirely. The local per-cell number is therefore inflated by work AML did not do.
- AML loaded the model bundle **16 times** for 300 cells (`bundle_loads == n_shards`, spec 38 D7);
  locally the bundle reloads far more often (§8 attributes a flat 6–10 s per cell to model load).
  That advantage is AML's, and it still lost on per-unit throughput.

**Counted end to end, AML was slower than the laptop.** A viewable map needs the merge, and on AML
that cost a second dispatch: 2066.9 + 1082.1 = **3149.0 s** versus the laptop's 2683.5 s, i.e.
**1.17× slower** for the same artefact. A single `merge=True` call would not have cost the full
1082.1 s — most of that is a second cluster spin-up — but how much less **was not measured**.

**Confounds, stated plainly.** T=8 vs T=10 (normalised away in the per-cell-per-timestamp row, not
in the wall row); blob reads vs local SSD; a 16-vCPU Azure D-family core vs an Apple laptop core;
`cubes_per_task=20` locally vs one whole-shard group per node on AML; merge in/out as above; and the
two runs are **15 days and several bug-fixes apart** (the local inference figure predates the
D-GRID-1, `rio_env` and merge fixes). No single one of these explains a 3.7× gap; together they
bound how hard the number should be leaned on.

---

## 5. The other steps — what compares, what doesn't, and why

- **Download — the *walls* are not comparable, but the *rates* now are.** The two runs moved
  different data over different networks from different providers (CDSE / university uplink /
  207 granules / 4 bands / 6 months, vs MPC / same-region Azure / 576 granules / 6 bands /
  12 months), so no ratio between 2732.8 s and the AML download would mean anything. The MPC path
  also records `bytes_downloaded: 0` — byte accounting is CDSE-only
  (`workflows/download.py:50-58`) — so a throughput was impossible until the archive was measured
  directly: **418.0 GB, 3456 assets, 121 MB/asset** (run-book 41 Step 2). A rate *is* comparable,
  and it is the most lopsided number in this document:

  | | local (CDSE, laptop) | AML (MPC, 16 nodes) |
  |---|---|---|
  | moved | 44.61 GB | **418.0 GB** |
  | per worker | 16.7 MB/s (1 machine, 4 streams) | **171.7 MB/s** per node |
  | aggregate | 16.3 MB/s over the 2732.8 s step | **≤ 1.18 GB/s** over the ≥354 s wall |
  | seconds per GB | 61.3 | **0.85** |

  **~10× per worker and ~72× on the wall**, moving 9.4× more data. The per-worker gap is proximity
  — MPC's blobs and the destination account are both in West Europe, so the node copies inside the
  datacentre while the laptop pulls across the public internet. The wall gap is that plus 16 nodes.
  (The AML aggregate is quoted against the wall *lower bound*, so it is an upper bound on the rate;
  against the 328 s job span it is 1.27 GB/s.)
- **Training-data generation.** Local step 3 is one call (`create_training_data`) that builds 900
  cubes, flattens them and computes features; on AML those are three separate run-books (36 P3, 39
  P1, 40 P1) with different shapes — and one of the three does not even run on the cluster. The
  parts that *are* measured: local **591.6 s** for all three, against AML's 405.7 s (flatten) +
  179.4 s (features, driver-side) = **585.1 s for two of the three**, with the build's wall
  unmeasured. Read that as "the same order of magnitude", not as a ratio.
- **Train + bundle.** Identical on both sides — it runs on the operator's machine either way.

---

## 6. What this says about running fsd at scale

### 6.1 The overhead, fully decomposed — and it is not the cluster

> **Extracted 2026-07-30 (spec 41 D14 P3):** this analysis now also stands on its own as
> [`docs/findings/cloud-overhead.md`](../docs/findings/cloud-overhead.md), which is the citable
> home and carries what has been fixed since. The text below is unchanged — this document is
> point-in-time and is not edited to chase later work.

`PROGRESS.md` and TODO #59 read run-book 38 Phase 4 (16 no-op jobs, 1082.1 s ≈ Phase 3's computed
1084.3 s, within 0.2 %) as one *fixed* cost, mostly cluster spin-up. **That reading is wrong, and
the run's own records say so.** Every AML job carries `StartTimeUtc`/`EndTimeUtc`, and the driver
stamped its own last action by writing `_result.json`. Between them the wall closes exactly
(`runbooks/41-recover-aml-job-timings.md` Steps 1 + 1b, 2026-07-28):

| run | wall | driver **pre**-dispatch | job span | driver **post**-collect |
|---|---|---|---|---|
| 38 P3 — inference, 300 cells | 2066.9 s | 249 s | 1089 s | **729 s** |
| 38 P4 — merge, 300 cells, no-op jobs | 1082.1 s | 66 s | 44 s | **972 s** |
| 36 P3 — build 900 cubes | **≥ 343 s** | *not measured* | 324 s | 19 s |
| 37 P3 — download 3456 assets | **≥ 354 s** | *not measured* | 328 s | 26 s |

`post` is measured directly (the `_result.json` write time minus the last job's end); `pre` is the
residual against a known wall. **The arithmetic is the check**: had the timestamps been misread, the
residual would have come out grossly negative rather than a plausible 249 s / 66 s. The two rows
without a measured wall get a **lower bound** — first job start to the driver's own last write —
which is tight, because their `post` is only 19 s and 26 s.

**35 % of the inference run, and 90 % of the merge run, was the driver collecting results over
blob.** Every artefact the driver wrote carries a blob `last_modified`, so both windows decompose
directly (run-book 41 Step 3):

**The 249 s before dispatch** — all on one clock, so these are exact:

| | window | seconds |
|---|---|---|
| `setup()` — 300 cell slices + geometries + `input.csv` | 09:06:15 → 09:06:37 | **22** |
| stage the model bundle (13.26 MB) | 09:06:48 → 09:07:01 | **13** |
| dispatch — write 16 shard CSVs | 09:07:03 → 09:07:11 | **8** |
| **submit → the first job actually executes** | 09:07:11 → 09:10:32 | **201** |
| total | | **257** |

**The bundle upload was 13 seconds.** `PROGRESS.md`'s 627 s bundle stage — this document's earlier
suspect — is not what happened here, and the guess is now refuted by direct measurement rather than
by an inequality. What dominates the pre-window is **201 s between submitting the jobs and the first
one executing**: AML's own queue-and-allocate, i.e. TODO #48's cold start, measured.

**The 972 s after the last job** (from run-book 38 Phase 4, which is the run whose STAC survives —
see the caveat below). It sums to the measured window exactly:

| | window | seconds | per unit |
|---|---|---|---|
| collect — 300 `fs.exists`, then a COG open + `geometry.geojson` read each; **no writes at all** | 12:09:39 → 12:19:55 | **616** | 2.05 s/cell |
| STAC — write 300 Items + `catalog.json` | 12:19:55 → 12:22:36 | **161** | 0.53 s/item |
| merge — read 300 COGs → `merged.tif` | 12:22:36 → 12:25:49 | **193** | |
| write `_result.json` | 12:25:49 → 12:25:51 | **2** | |
| total | | **972** | |

616 s of pure reads with **zero writes** is the signature of the collect loop in `api.py:1228-1240` +
`catalog/stac.py:233` — about three sequential blob round-trips per output cell at ~0.68 s each. It
scales with the number of **output units**, which is why run-books 36 and 37 — whose collect is 16
`_status` reads — pay 19 s and 26 s while moving far more data. Phase 3's own post-window (729 s,
no merge) is consistent: 616 + 161 = 777 s of collect + STAC, within 7 % of it.

> **Caveat — this is Phase 4's post-window, not Phase 3's.** Phase 4 wrote its STAC into the *same*
> output folder, so Phase 3's Item timestamps were **overwritten**; the surviving `catalog.json` is
> stamped 12:22:36, which is Phase 4. The two runs do the same collect over the same 300 outputs, so
> the decomposition transfers — but the per-segment numbers above were measured on the merge run.

> **Clock note.** The pre-window total is 257 s on the blob clock against 249 s derived from the
> laptop's wall, i.e. **~8 s of skew** between the laptop and Azure. Every figure in this section is
> far larger than that, and each decomposition above is internally single-clock. The blob timestamps
> also settle the timezone question raised by run-book 41 Step 1: the 16 `_status/*.json` writes
> (09:24:13 → 09:28:33) fall inside the job window Step 1b reports (09:10:32 → 09:28:41), which
> could not be true under the other reading.

Inside the job span, the picture is equally clear:

| run | node stagger | longest job | in-job work (stored) | **container startup** |
|---|---|---|---|---|
| 36 P3 (16 nodes) | 32 s | 316 s | 213.8 s | **102 s** |
| 37 P3 (16 nodes) | 26 s | 328 s | 192.1 s | **136 s** |
| 38 P3 (16 nodes) | 22 s | 1089 s | 982.7 s | **106 s** |
| 37 P2 (1 node, **warm**) | 0 s | 592 s | 577.6 s | **14 s** |
| 37 P2 (8 nodes, scaling 1→8) | 203 s | 219 s | 113.7 s | **105 s** |

- **Container startup is ~100–135 s on a cold node and ~14 s on a warm one** — TODO #48's "40–380 s
  node cold start", now pinned. It is paid by every node *in parallel*, so it costs ~100 s of wall
  regardless of width; but on the training regime's millisecond-scale units it is the entire cost.
- **Node stagger is 22–48 s when the cluster is already scaled out**, and 203 s in the one case
  where it had to grow from 1 node to 8. So stagger is a *scale-up* cost, not a per-node one.
  **This corrects an earlier claim in this document** that ramp-up grows with fan-out width: the
  8-node run's 203 s was the cluster scaling from cold, and the three 16-node runs each ramped in
  under 50 s.

**Where that leaves run-book 37 Phase 2's headline** (8× the nodes → 1.42× speedup): still true,
still worth knowing, but the cause is not per-node overhead. It is that the 8-shard run paid a
one-off 203 s scale-up **and** was crippled by the band-stratified partition in §6.3 — its slowest
shard did 113.7 s of work while two shards did 7 s.

**Roadmap consequence: attack the driver, not the cluster.** The largest single line item in fsd's
cloud overhead is a per-output-unit collect that runs sequentially on the operator's laptop over
VPN: **616 s of reads and 161 s of writes for 300 cells**, none of it touched by any fan-out width,
node SKU or `cores` setting. One listing instead of 300 existence checks, threaded metadata reads,
and batched Item writes would take most of it out of a 2067 s run (TODO #61). The second-largest is
**201 s of AML queue-and-allocate**, which is Azure's, not ours — but it is a fixed toll per
dispatch, and it is the reason run-book 38 needs *one* call with `merge=True` rather than Phase 3
followed by Phase 4.

### 6.2 Training and inference are opposite regimes

> **Extracted 2026-07-30 (spec 41 D14 P3):** also stands on its own as
> [`docs/findings/workload-regimes.md`](../docs/findings/workload-regimes.md), which reconciles
> the 5.48 GB / 4.13 GB discrepancy noted below.

Measured 2026-07-28 over the same two runs:

| | training units (900 fields) | inference units (300 cells) |
|---|---|---|
| median cube | 14 × 15 px, 13 KB | 597 × 554 px, 21.2 MB |
| whole set | 0.02 GB | 5.48 GB * |

\* **Unreconciled.** The 300 inference cubes actually on blob total **4.13 GB** across 600 `.npy`
files (run-book 41 Step 3). The 5.48 GB figure in `PROGRESS.md` was measured the same day; the
likeliest explanation is that it was taken over the **local** run, whose T=10 axis carries 25 % more
data than the cluster's T=8 — but that is a guess, and one listing would settle it. The 781×
per-unit ratio is unaffected: it comes from pixel dimensions, not totals.
| work per unit | milliseconds of raster work | ~52 s of real work |

**781× more pixels per unit.** Inference is genuinely work-bound, so a 16-way fan-out earns its
keep — and it is well balanced (18.75 cells/shard × ~52 s = 975 s predicted vs 982.7 s observed; no
straggler). Training is overhead-bound end to end: the ~1084 s that a 16-node dispatch costs would
dwarf the entire 900-unit workload. Combined with §6.1, the two verbs want **different cluster
shapes**, which is exactly TODO #59.

### 6.3 A partitioner bug the timings exposed (TODO #60)

Run-book 37 Phase 2's 8 shards each got 120–121 assets, 0 skipped, 0 failed — and their per-shard
seconds were **109.9 / 113.7 / 62.4 / 6.8 / 107.7 / 97.0 / 56.3 / 7.1**. A 17× spread on an equal
unit count is not variance; it is arithmetic. `shard_units` assigns `groups[i % n_groups]`
(`runners.py:251-261`) over an asset list built **granule-major, band-minor** (`mpc.py:360-364`), so
when `n_shards % len(bands) == 0` **each shard receives exactly one band**: shards 0/4 were all B04
(10 m uint16), 1/5 all B08, 2/6 all B8A (20 m), 3/7 all **SCL** (20 m uint8, ~7 s of work). Phase 3
escaped it only because 16 shards over 6 bands does not divide evenly.

The run's wall is its slowest shard, so those 8 nodes delivered the throughput of about 5 — **~38 %
of allocated node-time idle**, and two 16-vCPU nodes were allocated, image-pulled and held for 7
seconds of work. A deterministic shuffle before `shard_units` would have recovered it. No existing
test catches this: they assert the *partition* property and never assert balance.

---

### 6.4 The structural result: the cloud wins on proximity, loses on compute

Put §4 and §5 side by side and the two headline steps point in opposite directions — and the reason
is the same in both cases.

| | per worker | on the wall |
|---|---|---|
| **download** (network-bound) | AML **10× faster** | AML **~72× faster** |
| **inference** (compute-bound) | AML **3.7× slower** | AML **1.30× faster** |

Where the bottleneck is **distance to the data**, the cloud wins enormously and the fan-out
compounds it: a node copying MPC→blob inside West Europe moves 171.7 MB/s against a laptop's
16.7 MB/s, and 16 of them turn a 45-minute download into minutes. Where the bottleneck is
**compute per unit**, a single cloud vCPU is *slower* than a laptop core on the same cube, and the
16-way fan-out is spent buying back that deficit plus the overhead of §6.1 — netting 1.30×.

**So "move it to the cloud" is not one decision.** For fsd's pipeline it is unambiguously right for
download (and for anything else that reads the archive), marginal for inference as currently
configured, and — per §6.2 — actively wrong for the 900-unit training regime. The lever that makes
inference worth it is not a bigger cluster: it is TODO #61's collect and the per-unit cost, both of
which are fsd's own code.

## 7. Where every number comes from

| figure | file |
|---|---|
| local step table, download detail | `tests/outputs/demo_e2e/full_run.log` (steps 0–4), `full_run_3.log` (steps 1, 5–7) |
| AML download, per-shard seconds | `tests/outputs/p2_download_aml/phase3_result.json` |
| AML fan-out sweep (1 vs 8 shards) | `tests/outputs/p2_download_aml/phase2_result.json` |
| AML 900-cube build, per-shard seconds | `tests/outputs/p2_aml_runner/phase3_result.json` |
| flatten reduce, 405.7 s | `tests/outputs/p39_training_data_aml/phase1_result.json` |
| features, 179.4 s | `tests/outputs/p40_train_and_bundle/phase1_result.json` |
| inference, 2066.9 / 982.7 / 1084.3 s | `tests/outputs/p4_inference_aml/phase3_result.json` |
| merge, 1082.1 s | `tests/outputs/p4_inference_aml/phase4_result.json` |
| archive inventory (3456 assets, 576 granules) | `tests/outputs/p2_verify_archive/step1_result.json`, `step2_result.json` |
| cube dimensions (both regimes) | `PROGRESS.md` "Cube sizes measured 2026-07-28"; TODO #59 |
| job spans, node ramp, container startup (§6.1) | `tests/outputs/p41_job_timings/_result.json` + `_result_step1b.json` (run-book 41 Steps 1 + 1b, 2026-07-28) |
| driver pre/post split — the `post` half | each run's `_result.json` **file write time** minus its last job's `EndTimeUtc`; a direct driver-side timestamp |
| archive bytes, 418.0 GB / 121 MB per asset / per-band totals | `tests/outputs/p41_job_timings/_result_step2.json` (run-book 41 Step 2, 2026-07-28) |

**Open cells, and how they close:** the two "not measured" walls in §3 are filled by
`runbooks/41-recover-aml-job-timings.md` (free — a lower bound reconstructed from AML job history)
or `runbooks/42-timed-cold-reruns.md` (paid — a real `wall_seconds` from a cold replay). Whichever
lands, it is recorded here **with its provenance**: a reconstructed span will be labelled as a lower
bound, never as `wall_seconds`.

## 8. Reproduce it

Everything above §7 was assembled by hand from six separate run-books (spec 40's "why"). Going
forward, one script replaces that: `demos/e2e_austria_aml.py` runs the same eight steps as
`demos/e2e_austria.py`, on the cluster, unattended, and emits one self-contained `timings.json`
(spec 40 D9) — no run-book paste-back required. **Claude does not run this script** (CLAUDE.md); it
is handed to the operator.

### 8.1 Prerequisites

1. **A VM inside the project's compute subnet** (`snet<proj>-compute`) or another subnet carrying
   the storage service endpoint. Project storage is deny-by-default firewalled
   (`AZURE_INFRA.md` "Firewalled storage"): a VM outside those ranges gets **403 on every blob call
   regardless of credentials**, because network rules and authorization are enforced independently
   (Microsoft Learn, "Azure Storage firewall rules and network access"). This is the one prerequisite
   that isn't obvious from the script failing — it looks like an auth bug and isn't.
2. **`az login`** once over SSH; the CLI refreshes tokens silently, so a multi-hour unattended run is
   fine. (Upgrade: a managed identity with Storage Blob Data Contributor + an AML submit role skips
   the login entirely — an admin action, not a prerequisite.)
3. **Clone and install — all six extras, not just `[dev,azure,aml]`:**

   ```bash
   python3.11 -m venv .venv
   .venv/bin/pip install -e ".[dev,azure,aml,mpc,grid,model-example]"
   ```

   | extra | supplies | needed by |
   |---|---|---|
   | `azure` | `adlfs`, `azure-identity` | every blob read/write |
   | `aml` | `azure-ai-ml` | dispatching every job |
   | `mpc` | `planetary-computer` | MPC discovery (`0_preflight`, `2_download`) |
   | `grid` | `s2`, `s2cell` | `1_tiling`, and `run_inference`'s ROI re-tiling |
   | `model-example` | `scikit-learn`, `joblib`, `matplotlib` | `4_train_bundle`, `6_plots` |
   | `dev` | `ruff`, `pytest` | not required to run; keep for the test suite |

   fsd core stays deliberately lean, so **none of the modelling stack is a base dependency** —
   `scikit-learn`/`joblib` live in `model-example` because fsd never trains a model (that is
   permanently the user's side). Miss them and the run dies at `4_train_bundle`, *after* the
   download and the training-data dispatch. `0_preflight` now checks all of these up front and
   prints the exact `pip install` line, so a short install can only cost you seconds — but only
   if you are on a build that has that check (2026-07-29 or later).
4. **Both AML Environments already built, from rise_eo ≥ spec 40 (2026-07-28).** The script's own
   preflight only verifies they *resolve* (D4); building one is a 10–20 min ACR build that must not
   risk killing a 40-minute unattended run. The general-purpose Environment (`AZ_ENV_NAME`,
   download + build + flatten, ADR-0020) and the inference Environment (`AZ_INFER_ENV_NAME`,
   carries the adapter package) are two different images — see `runbooks/36-aml-runner.md` /
   `runbooks/38-inference-on-aml.md` for how each was built.

   > ⚠️ **Reusing an image from run-books 36–39 is not enough — rebuild and bump the version.**
   > The four in-job stamps `job_admission_seconds` (D11) is computed from are written by the
   > `fsd` **inside the image**, not by your checkout. An older image runs fine and produces
   > correct science; it just emits `job_admission_seconds: null` on every job, silently voiding
   > the headline measurement. That cost a complete 25-minute run on 2026-07-29 — 97 jobs, four
   > dispatches, no admission data. `2_download` now checks this on the first dispatch and stops
   > before the other three spend another ~20 minutes.

### 8.2 Environment variables

```bash
export AZ_SUBSCRIPTION_ID='<subscription id>'
export AZ_RG='<resource group>'
export AZ_ML_WORKSPACE='<aml workspace>'
export AZ_CLUSTER='<the d16 cluster name>'
export AZ_UAMI_CLIENT_ID="$(az identity show -g "$AZ_RG" -n '<compute identity name>' --query clientId -o tsv)"

export AZ_ROOT="abfss://<filesystem>@<storage account>.dfs.core.windows.net/<your-user>/fsd-p40-demo"

export AZ_ENV_NAME='rise-eo-aml-env'                 # general-purpose: download + build + flatten
export AZ_ENV_VERSION='<version>'
export AZ_INFER_ENV_NAME='rise-eo-infer-env'         # carries the DemoRF adapter package
export AZ_INFER_ENV_VERSION='<version>'

# No CDSE credentials. `2_download` sources from MPC (spec 40 D13 amendment A1, matching
# run-book 37 Phase 3 and every cluster run since P1): MPC is anonymous, and
# `run_aml_download` REFUSES creds for an MPC run rather than staging a secret on blob
# for something that never reads it (TODO #49).
```

### 8.3 Run it

```bash
# 1. Estimate first -- zero side effects (D6).
python demos/e2e_austria_aml.py --fresh --dry-run

# 2. The real run, under tmux/nohup so it survives a dropped SSH session (D7).
tmux new -s rise-eo-demo
python demos/e2e_austria_aml.py --fresh --confirm-spend
# ^C / SIGTERM both exit cleanly, printing the resume line rather than a traceback
# (D7). Every step that already FINISHED keeps its numbers (D3, each wrote its own
# _result.json); the step that was in flight did not complete, so it has none, and
# resuming re-runs it from the start on a fresh prefix (D5).

# 3. If it stops partway (preflight failure, a failed dispatch, or an interrupt),
#    resume with the SAME run id it printed at the start -- completed steps skip
#    instantly, only the failed/remaining ones re-run (D5):
python demos/e2e_austria_aml.py --run-id <the id it printed> --confirm-spend
```

`--fresh` never deletes anything (fsd's own recursive delete is broken, TODO #50): if a *previous*
`--fresh` run left data on blob, this one prints the exact `az storage fs directory delete` command
for the operator to run by hand.

### 8.4 What to send back

**One file:** `tests/outputs/demo_e2e_aml/<run_id>/timings.json` — it embeds every step's own
`_result.json` (D9), including each dispatch's `_timing.json` (job admission, dispatch overhead, the
D11 additive split). Render the timing figures anywhere, offline, no cluster needed:

```bash
python demos/plot_aml_timings.py tests/outputs/demo_e2e_aml/<run_id>/timings.json
```

writes `demos/figures/aml_job_admission.png`, `aml_where_the_wall_went.png`, and
`aml_job_gantt.png` (dropped automatically above ~80 jobs — spec 40 D12).

Once a real `timings.json` lands, this document is rewritten around it (§7's promise): the
laptop-driver numbers in §3 move to a labelled appendix, and §1/§4 compare like-for-like runs on the
same code, same download scope (D13), same driver-location bookkeeping (D10) — for the first time.
