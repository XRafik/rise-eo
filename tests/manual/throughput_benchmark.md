# Manual runbook — datacube throughput benchmark (spec 11 · Part 1)

How to run the **parallelism-sweep** benchmark yourself, re-run it after an
optimization, and read the result. This is the reusable baseline for the
datacube-speed track (TODO #15): *how does the wall-time to build many datacubes
scale with build parallelism, and where does the time go per step?*

It runs the **real** Snakemake workflow (`run_create_datacube` path) over a fixed
set of geometries against the real `satellite_benchmark/` tiles — no synthetic data,
no downloads. Read-conflict logging is **not** here (that's Part 2 / spec 12); this
measures throughput + per-step timing + static grid×tile overlap.

> **NOTE — the archive is now COG (2026-07-04, spec 14 migration).**
> `satellite_benchmark` tiles are `Bxx.tif` (COG + overviews), not `.jp2`. **The
> Part-1/Part-2 findings in the specs/reports were measured on the *pre-migration
> JP2* archive** — re-running the sweep now reads COG, so expect the decode-bound
> slowdown to be largely gone (this is exactly the spec-13 result). The examples below
> still show `.jp2` filepaths as they appeared then.

---

## 0. Prerequisites

- Dev env active: from `fsd/`, `source .venv/bin/activate` (deps incl. `matplotlib`,
  `snakemake` are in `.venv`, not system Python).
- Data present at the workspace root (parent of `fsd/`):
  - `satellite_benchmark/sentinel-2-l2a/catalog.parquet` — the 1-year Ethiopia download.
  - `shapefiles/100_random_grids.geojson` — the 100 res-11 S2 grids, `id` column.
- Everything is parametrised by module constants at the top of
  `benchmarks/datacube_throughput_sweep.py` (`ROOT`, `CATALOG`, `GRIDS`, window, bands).
  Edit those if your paths/window differ.

---

## 1. Run it

From `fsd/` (the repo root):

```bash
# quick smoke — 3 grids, cores 1 & 2 (~30 s), proves the chain end-to-end
.venv/bin/python benchmarks/datacube_throughput_sweep.py --smoke

# full baseline — 100 grids, cores 1,2,4,6,8,10 (~25–40 min on a 10-core Mac)
.venv/bin/python benchmarks/datacube_throughput_sweep.py

# custom: e.g. 40 grids, only 1/2/4 cores, 2 repeats (min wall kept per setting)
.venv/bin/python benchmarks/datacube_throughput_sweep.py --n-grids 40 --cores 1,2,4 --repeats 2

# Part 2 (spec 12): also log every windowed read + analyse read conflicts
.venv/bin/python benchmarks/datacube_throughput_sweep.py --read-log
```

Flags: `--cores` (comma list), `--n-grids N` (first N grids), `--repeats R`, `--smoke`,
`--read-log` (Part 2 — see §7), and `--report-only` (rebuild `report.md` from the saved
`stats.json`, reusing the figures — no sweep; handy after tweaking the report text).

The harness prints a **live progress line every ~10 s** while each setting builds, plus a
running sweep-wide ETA, e.g.:

```
[sweep] START 6 runs × 100 grids | cores=[1, 2, 4, 6, 8, 10] repeats=1
[progress] run 1/6 cores=1 [########----------------] 33/100 | 3.1m elapsed | ETA ~6.2m
[sweep] DONE run 1/6 cores=1 rep=0: total=9.5m built=100/100 rc=0 | sweep ETA ~40m
```

> **Long run?** Launch detached and watch the log:
> ```bash
> nohup .venv/bin/python benchmarks/datacube_throughput_sweep.py > /tmp/sweep.log 2>&1 &
> grep -E "\[setup\]|\[progress\]|\[sweep\]|\[done\]" /tmp/sweep.log   # progress + ETA
> ```

---

## 2. What it does (the steps, in order)

1. **Setup once** — `create_datacube.setup(...)` slices the big catalog per grid
   (`TileCatalog.filter`) into `tests/outputs/throughput_sweep/<dates>/<id>/catalog.parquet`
   + `geometry.geojson`, and writes one `input.csv`. (Skips grids with no intersecting
   tiles — that's why "n grids with tiles" can be < 100.)
2. **Characterize (static, no raster reads)** — reads each grid's `catalog.parquet`
   slice → the MGRS tiles it touches → grid×tile overlap (how many grids share each tile
   = *potential* shared reads) and the tiles-per-grid distribution (build heaviness).
3. **Sweep** — for each `cores` value (× `repeats`):
   a. **wipe** each grid folder's artifacts + `start.txt`/`done.txt` sentinels so
      Snakemake actually rebuilds (setup slices are kept);
   b. set `RISE_EO_WRITE_TIMINGS=1` and run `runners.run_local(input.csv, cores=C)`
      (the real Snakemake local runner, jitter off);
   c. time the whole call = **total wall**; then **collect** per-grid `wall_seconds`
      (`done.txt − start.txt`) + the builder's `timings.json` sidecar (per-phase seconds).
4. **Aggregate + report** — per `cores`: throughput, speedup, efficiency, summed
   per-phase seconds, mean `load_images`/grid; write the report + stats + 4 plots.

**How the per-step timing works:** `build_datacube(write_timings=True)` times each phase
(missing_check · load_images · dst_crs · reference_profile · resample · stack · ops · save)
and drops a `timings.json` next to `datacube.npy`. It is **off by default**; the harness
turns it on for the whole run via the `RISE_EO_WRITE_TIMINGS` env var (inherited by the task
subprocesses), so nothing else in the workflow needs to change.

---

## 3. Outputs

| Path | What |
|------|------|
| `benchmarks/datacube_throughput_report.md` | the human report (regenerated every run) |
| `benchmarks/datacube_throughput_stats.json` | machine-readable — **diff this across runs** |
| `benchmarks/datacube_throughput_figures/*.png` | throughput, phase-breakdown, load_images, wall-vs-tiles |
| `tests/outputs/throughput_sweep/` | gitignored scratch (per-grid slices, sentinels, cubes) |
| `.../<grid>/reads.jsonl` | *(--read-log only)* one JSON row per windowed read (see §7) |
| `benchmarks/datacube_throughput_figures/read_*.png` | *(--read-log only)* the 4 read-contention plots |

---

## 4. How to read it

- **Throughput vs parallelism table** — `speedup` = wall(cores=1)/wall(cores);
  `efficiency` = speedup/cores. Efficiency dropping well below 1 as `cores` climbs means
  extra processes are **not** buying proportional speed → contention. The **sweet spot**
  is the `cores` with the best total wall.
- **`mean load/grid (s)` column + the load_images plot** — the key contention signal: if
  the *same* per-grid `load_images` gets slower as `cores` rises (while compute phases stay
  flat), parallel reads are blocking each other. That number climbing is exactly the
  hypothesis Part 2 (spec 12) will confirm by logging every individual read.
- **`load_images frac`** — how much of the build is I/O. On this pipeline it dominates;
  that's why the speed work targets reads, not CPU.
- **wall-vs-tiles plot** — per-grid cost scales with how many MGRS tiles a grid straddles
  (1-tile grids are cheap; boundary grids that hit 2–4 tiles cost more).

---

## 5. Re-running after an optimization (the point of the harness)

1. Keep the current `datacube_throughput_stats.json` as the baseline (copy it aside).
2. Make your change (e.g. a tile-split from Part 3, or a read-caching tweak).
3. Re-run the **same** command.
4. Compare `total_seconds` / `mean_load_per_grid` / `efficiency` per `cores` between the
   old and new `*_stats.json`. Faster load_images at high `cores` with flat compute = the
   contention was reduced.

---

## 6. Notes & caveats

- **Cache: measured, not forced** (spec 11 decision). Runs are warm-as-is; no `sudo purge`.
  Re-running the same grids across settings can warm shared file blocks, but the grids read
  mostly-disjoint windows so reuse is limited. Precise cache-vs-contention separation is
  Part 2's job. If you want a colder comparison, `sudo purge` between runs (macOS) — but
  that is not required and not what the current report claims.
- **Setup time is excluded** from the swept total (setup runs once, up front). Only the
  build/`run_local` wall is timed per `cores`.
- **Cleanup:** the harness removes its own `.snakemake/` scratch; the
  `tests/outputs/throughput_sweep/` folder is gitignored — delete it to reclaim disk.
- Lower-level pieces are importable if you want to poke by hand:
  `from benchmarks.datacube_throughput_sweep import characterize, overlap_stats, run_sweep`
  (or drive `create_datacube.setup` + `runners.run_local` directly).

---

## 7. Part 2 — read-contention instrumentation (`--read-log`, spec 12)

Adding `--read-log` turns on the builder's per-read log and enriches the same report with a
**Read contention** section — the direct measurement behind the Part-1 inference that
`load_images` slows under parallelism.

**What it logs.** With `--read-log`, `build_datacube(write_read_log=True)` (enabled per-task
via the `RISE_EO_WRITE_READ_LOG=1` env var, same mechanism as `RISE_EO_WRITE_TIMINGS`) writes a
`reads.jsonl` next to each grid's `datacube.npy`. One row per windowed read:

```json
{"id": "<grid>", "mgrs_tile": "37PBN", "product_id": "S2A_..._T37PBN_...",
 "band": "B08", "filepath": ".../<product>/B08.jp2",
 "start": 1783147061.845, "end": 1783147061.904, "duration": 0.0588}
```

Times are **wall-clock `time.time()`** (comparable across grid processes). The *only* disk reads
in a build are these `load_images` reads, so `sum(duration) ≈ the load_images phase`.

> **Requires `njobs_load_images == 1`** (the reads must run in the grid's own process to be
> timed). The sweep always uses 1 (parallelism is at the grid/process level), so this is fine;
> if you set it >1 with `--read-log`, the log is skipped with a warning.

**Three identifiers, don't conflate them:**
- `mgrs_tile` (`37PBN`) — geographic tile, **same across acquisition dates**. *Not* a file.
- `product_id` — one SAFE product = one (tile × datetime) folder.
- `filepath` — `<product>/B08.jp2` = **one physical file** (product × band). The same-file key.

**How to read the Read-contention section:**
- **conflicts table** (`cores → reads, conflicts, same-file / same-tile / diff-tile, max/mean
  concur`): a *conflict* = two reads from **different grids** whose intervals overlap. `max concur`
  = peak reads in flight at once (bounded by `cores`). Only **same-file** conflicts are what
  Part-3 tile-splitting can remove.
- **`read_duration_vs_concurrency.png`** (the money plot): mean read duration vs how many reads
  were in flight. Rising with concurrency = reads block each other = hypothesis **confirmed**.
- **`read_conflicts_vs_cores.png`**: conflict pairs stacked by class — is the contention on the
  same file or just general disk bandwidth?
- **`read_class_split.png`**: duration-vs-concurrency for all reads vs same-file reads only (a
  first cut at cache-vs-contention: a same-file read that gets *faster* hints cache reuse).
- **`read_concurrency_timeline.png`**: reads in flight over time, busiest `cores`.
- **Verdict paragraph**: states whether the slowdown is confirmed and whether conflicts are
  **same-file** (→ Part-3 tile-splitting helps) or **different-file/tile** disk-bandwidth (→ it
  won't — reconsider before building Part 3). *This is the go/no-go signal for Part 3.*

**Self-check:** compare a run's `sum_read_seconds` against the summed `load_images` phase in
`timings.json` — they should track (reads are the only I/O in a build).

**Note on smoke vs full:** a 3-grid `--smoke --read-log` can show *all different-tile* conflicts
simply because 3 random grids rarely share a tile; the full 100-grid run (where one tile is shared
by ~48 grids) is what surfaces same-file conflicts. Judge the same-file/different-file split from
the **full** run, not the smoke.
