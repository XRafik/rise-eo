# Manual runbook — COG vs JP2 storage/time experiment (spec 13)

Measure what switching Sentinel-2 tiles from native CDSE **JP2** to **COG** buys in
datacube-build **time**, and costs in **disk** — the numbers for a team space-vs-time call.
Runs entirely off a parallel COG dataset + catalog; **no `src/rise_eo/` change** (rasterio reads
`.tif` transparently). Three steps: **prep** (convert + storage) → **A/B sweep** (time) →
**compare** (the team report).

---

## 0. Prerequisites
- Dev env active (`source .venv/bin/activate` from `fsd/`); deps incl. `rasterio`/GDAL ≥3.1
  (COG driver — prep checks this and fails early if missing).
- The JP2 dataset present: `satellite_benchmark/sentinel-2-l2a/catalog.parquet`.
- **Disk:** first 4 months of base COG ≈ **1.2× the JP2** (~39 GiB here); prep runs a
  pre-flight and **aborts before writing** if free space < estimate × 1.2.

---

## 1. Prep — build the COG mirror + storage report
From `fsd/`:
```bash
# full experiment prep: convert the first 4 months (parallel, live progress + ETA)
.venv/bin/python benchmarks/prep_cog_dataset.py --months 4 --jobs 6

# smoke / incremental: cap #products
.venv/bin/python benchmarks/prep_cog_dataset.py --limit 2
```
What it does, in order: COG-driver check → select first-N-months rows → **pre-flight**
(sample-convert, estimate full COG size, check free disk, abort if short) → **convert**
JP2→base COG (DEFLATE + PREDICTOR=2, tiled 512, **no overviews**; `NBITS=16` promotes S2's
declared 15-bit depth so PREDICTOR is legal — **lossless**) → **bit-identical losslessness
assert** → write the parallel `satellite_benchmark_cog/…/catalog.parquet` → **storage report**.

Outputs: `satellite_benchmark_cog/` (the COG tree + catalog, *outside* the fsd repo, like
`satellite_benchmark/`), and `benchmarks/cog_vs_jp2_storage.{md,json}` (JP2 → base COG →
COG+overviews, the overview row estimated from a sample). It prints a **suggested ~6-week timed
window** for step 2.

Flags: `--months N`, `--jobs N` (workers), `--limit N` (cap products), `--force` (reconvert).

---

## 2. A/B sweep — time (the same Part-1/2 harness, `--catalog`-switched)
Run the throughput harness twice — identical grids/window/cores, only the catalog differs —
with `--tag` so outputs don't clobber and `--read-log` so we get the duration-vs-concurrency
curve. Use the **~6-week window** prep suggested (must sit inside the converted 4 months):
```bash
W_START=2018-01-02 ; W_END=2018-02-13   # example; use prep's suggestion

# JP2 baseline
.venv/bin/python benchmarks/datacube_throughput_sweep.py --tag jp2 --read-log \
  --catalog "$PWD/../satellite_benchmark/sentinel-2-l2a/catalog.parquet" \
  --start $W_START --end $W_END

# COG
.venv/bin/python benchmarks/datacube_throughput_sweep.py --tag cog --read-log \
  --catalog "$PWD/../satellite_benchmark_cog/sentinel-2-l2a/catalog.parquet" \
  --start $W_START --end $W_END
```
Each writes tagged `datacube_throughput_{report,stats}_<tag>.*` + `_figures_<tag>/`. New harness
flags: `--catalog`, `--start/--end`, `--tag` (see also the Part-1/2 runbook
`throughput_benchmark.md` for `--cores`, `--smoke`, progress lines). Two full sweeps take a while
— launch detached and watch the log if needed.

> **Fair A/B:** run the two back-to-back on the same machine (same cache regime). `--read-log`
> exposes any residual cache effect. Same grids+window ⇒ identical read *sets*, differing only in
> format.

---

## 3. Compare — the team report
```bash
.venv/bin/python benchmarks/compare_cog_jp2.py --jp2 jp2 --cog cog
```
Reads the two tagged `stats.json` + `cog_vs_jp2_storage.json` → `benchmarks/cog_vs_jp2_report.md`:
- **Time table** per `cores`: JP2 vs COG total wall, `load_images` sum, mean load/grid, speedups.
- **Duration-vs-concurrency overlay** (`cog_vs_jp2_figures/duration_overlay.png`): *the
  decode-bound test* — if COG's curve is **lower and flatter**, the JP2 wavelet **decode** was
  the bottleneck (COG frees the CPU); if the curves match, contention was not decode-bound.
- **Storage table** + the year extrapolation (×3 from 4 months).
- **Verdict:** the one-paragraph space-vs-time summary for the team.

---

## 4. Notes & cleanup
- **Lossless:** DEFLATE + PREDICTOR are reversible; `NBITS=16` only changes the *declared* bit
  depth (S2 packs 15 significant bits in uint16), not any pixel value. Prep asserts bit-identical
  samples and will abort on any mismatch.
- **Overviews are not materialised** — the build reads full-res and never uses them; the storage
  report *estimates* the +~38% they'd add if you later want these COGs XYZ-tiling-ready (TiTiler).
- **Scope:** local read regime only. On Azure/remote Blob, COG's fewer-bytes / HTTP-range-request
  advantage is *additional* (out of scope here). No production cutover — this only measures.
- **Reclaim disk:** `rm -rf satellite_benchmark_cog` removes the COG tree; the `tests/outputs/
  throughput_sweep_*` scratch is gitignored.
