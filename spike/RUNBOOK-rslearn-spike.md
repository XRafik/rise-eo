---
status: current
summary: Run the rslearn build-vs-borrow spike on the VM -- install weight, the calendar-T contract (offline, decisive), then the Azure and equivalence probes that Step 2 gates.
---

# Run-book: rslearn spike (Plan C evaluation)

> **Deliberately unnumbered and inside `spike/`.** Run-book numbers are a `main` sequence and
> `44` is already taken (`runbooks/44-todo-to-issues.md`). This branch may never merge, so it
> does not claim a number from that sequence. If the Plan-C decision goes to C, this gets a
> number on the way in. Same reasoning for not writing a `specs/44-*.md`: see `README.md`
> ("Does this need a spec?").
>
> Spec 24 rules apply: Claude does not run any of this. You run it, paste back each step's
> `_result.json`, Claude diffs it against the success criteria below and never reads live logs.

## Purpose

Answer the three charter questions with measurements instead of argument, cheapest-and-most-
decisive first. The static half is already done and written up in
[`RSLEARN_READ_2026-07-31.md`](RSLEARN_READ_2026-07-31.md) — **read section 6 before starting**;
it lists exactly what is still unknown and predicts what each probe should return.

## Where this runs, and why

**On the VM, not the laptop** (your call, 2026-07-31 — hotspot data). Note what that buys:

| step | network | satellite bytes |
|---|---|---|
| 1 — install | ~2–3 GB of wheels, once | none |
| 2 — T contract | **none** | **none** |
| 3 — Azure write | blob only | tiny |
| 4 — equivalence | CDSE/MPC | one small cell |

**Step 2 is the decision gate and it is free after Step 1.** It can invalidate the shape of
Steps 3–4, which is why those two are not written yet.

## Prerequisites

- On the VM, this branch: `git switch spike/rslearn && git pull`.
- `python3.11+` (rslearn requires `>=3.11`).
- The tutorial fixture present at `tests/data/tutorial/` (it is committed — Step 0 checks).
- Steps 3–4 only: `az login`, and `source env.local.sh` for the `AZ_*` variables
  (see `env.example.sh`). **Never paste concrete account names or URLs into a result file** —
  this branch is part of a public MIT repo.
- **Never install rslearn into `.venv`.** The isolated env is `.venv-rslearn` (gitignored). Main
  must stay lean, and install weight is itself a measurement.

---

## Steps

### Step 0 — confirm the branch and the ground truth

```bash
cd fsd
git rev-parse --abbrev-ref HEAD          # expect: spike/rslearn
ls tests/data/tutorial/catalog.parquet   # expect: the file exists
python3 -c "import pathlib,json; p=pathlib.Path('tests/data/tutorial'); print(json.dumps({'granule_dirs': len([d for d in p.iterdir() if d.is_dir()]), 'mb': round(sum(f.stat().st_size for f in p.rglob('*'))/1e6,1)}))"
```

- **Expect:** branch `spike/rslearn`; `catalog.parquet` exists; roughly `36` granule dirs, ~`27` MB.
- **PASS if:** all three hold. No `_result.json` — just paste the output.
- **If the fixture is missing:** stop. It is the comparison corpus for Step 4 and it replaces the
  deleted `satellite_benchmark/`. It should have come with the merge.

### Step 1 — install rslearn into an isolated venv, and weigh it

```bash
cd fsd
python3.11 -m venv .venv-rslearn
source .venv-rslearn/bin/activate
python -c "import sys; assert 'venv-rslearn' in sys.prefix, sys.prefix; print('isolated OK', sys.prefix)"

# `--report` needs pip >= 22.2. A fresh `python3.11 -m venv` seeds whatever pip the
# distro's ensurepip bundles, which on the Azure VM images is older -- it fails with
# `no such option: --report` (hit for real, 2026-07-31). Upgrade BEFORE the timed install.
pip install --upgrade pip && pip --version

# Time it and keep the log -- both numbers feed the probe.
time pip install --report /tmp/rslearn_install.json 'rslearn==0.1.13' 2>&1 | tee /tmp/rslearn_install.log
```

Then read the two numbers off it and run the probe.

> ⚠️ **Download bytes come from the LOG, not from `--report`.** This run-book originally summed
> `download_info.archive_info.size` out of the report JSON. **That field does not exist** — pip's
> installation report gives each entry a PEP 610 *Direct URL* object (`url` +
> `archive_info.hashes`), which carries hashes, not sizes. The sum was therefore `0` on every
> machine, and was measured as `0` on the VM on 2026-07-31 before anyone noticed the extraction
> was unsound rather than the install being cached. Check it yourself in one line:
> `python -c "import json;print(json.load(open('/tmp/rslearn_install.json'))['install'][0]['download_info'])"`

```bash
# MB actually downloaded -- parsed from pip's own progress lines
grep -oE '\(([0-9.]+) [kM]B\)' /tmp/rslearn_install.log \
  | tr -d '()' | awk '{ mb += ($2=="kB" ? $1/1024 : $1) } END { printf "download_mb ~= %.0f\n", mb }'

# package count -- this the report DOES carry
python -c "import json;print('packages =', len(json.load(open('/tmp/rslearn_install.json'))['install']))"

python spike/probes/probe_01_install_weight.py \
    --out tests/outputs/rslearn_spike \
    --install-seconds <the `real` seconds from `time`> \
    --download-mb <the number printed above>
```

> ⚠️ **Warm cache ⇒ neither timing nor bytes is a cold-install measurement.** Observed on the VM
> 2026-07-31: every wheel came from pip's cache, so `download_mb` parsed to ~0 *and* the `time`
> figure is a warm-cache install (unpack + link only, no transfer). Both are floors. **Pass
> neither flag rather than passing a number that will be cited as cold** — probe 01 records
> `null`, which is honest, and its two load-bearing metrics (`venv_size_mb`,
> `torch_free_acquisition_path`) are unaffected by caching.
>
> To get the cold figure, use a **throwaway** venv so the working one survives, and force real
> downloads with `--no-cache-dir` (inside the existing venv pip would just report "already
> satisfied" and measure nothing):
>
> ```bash
> deactivate
> python3.11 -m venv /tmp/venv-rslearn-cold && source /tmp/venv-rslearn-cold/bin/activate
> pip install --upgrade pip
> time pip install --no-cache-dir 'rslearn==0.1.13' 2>&1 | tee /tmp/rslearn_cold.log
> du -sh /tmp/venv-rslearn-cold
> deactivate && rm -rf /tmp/venv-rslearn-cold && source <fsd>/.venv-rslearn/bin/activate
> ```
>
> This costs a few GB of VM egress and ~5–15 min. It is **optional and non-blocking** — do it
> after Step 2, never before, since Step 2 is the gate that can veto the expensive half.

Two caveats on that number, both of which the report must state when it cites it:

- **It is a floor.** Anything pip served from its HTTP cache prints no `Downloading` line at all,
  so a second run on the same machine will under-report. A near-zero result means "cached", not
  "small" — re-run with `--no-cache-dir` if you need the true cold figure.
- **It is optional.** `--download-mb` defaults to `None` on the probe. The number that actually
  carries the install-weight argument is **venv size**, which probe 01 measures itself from
  `sys.prefix` (`probe_01_install_weight.py:46-51`). If the log parse is unavailable or
  untrustworthy, omit the flag and say so — do not paste a figure you don't trust.

**If `--report` is unavailable entirely** (locked-down pip, no upgrade path): drop it. Run
`time pip install 'rslearn==0.1.13' 2>&1 | tee /tmp/rslearn_install.log` and use the same log
parse above; you lose only the package count.

> 🔴 **The stock install does not import — you must add `einops` by hand.** Measured on the VM
> 2026-07-31: after a clean `pip install rslearn==0.1.13`, every acquisition import dies with
> `ModuleNotFoundError: No module named 'einops'`. This is an upstream packaging bug, not a
> mistake in your install — `einops>=0.8` is declared in rslearn's **`extra`** optional group
> (`pyproject.toml:39`), but `rslearn/utils/raster_format.py:9` imports it unconditionally and
> `rslearn.config` reaches it via `config/dataset.py:31`.
>
> ```bash
> pip install einops       # ~1 MB; does not disturb the weight measurement above
> ```
>
> **Install `einops` alone — not `rslearn[extra]`.** The extra group drags in cdsapi,
> earthengine, earthdaily, netCDF4, osmium, huggingface_hub and more, which would destroy the
> install-weight numbers you just paid a cold download for. Record the stock venv size *before*
> adding it (probe 01 already did, if you ran it first). **Probe 02 needs this too** — it imports
> `rslearn.config`.

- **Expect:** the install succeeds and pulls torch, torchvision, torchmetrics and lightning as
  **core** dependencies (`pyproject.toml:11-31`) — a multi-GB venv is the *expected* result here,
  not a problem with your machine. Measured: **5,289.5 MB venv, 2,892 MB downloaded cold,
  88.5 s** (Azure VM).
- **PASS if:** `_result_probe01.json` has `"status": "ok"`. The interesting field is
  `metrics.torch_free_acquisition_path` — predicted **`true`**.
- **If `torch_free_acquisition_path` is `false`:** that contradicts
  `RSLEARN_READ_2026-07-31.md` §4.3. Not a failure — paste it, it changes the write-up.
- **If pip backtracks for a long time:** known upstream hazard; their own comment at
  `pyproject.toml:44-47` describes botocore pinning causing exactly this. Add `--only-binary=:all:`
  or install with `uv pip install` and note which you used.

### Step 2 — the calendar-`T` contract (offline, decisive) 🚩 DECISION GATE

```bash
cd fsd && source .venv-rslearn/bin/activate
python spike/probes/probe_02_t_contract.py --out tests/outputs/rslearn_spike
```

**Zero network, zero satellite data.** Four synthetic cases isolate three predicted divergences
between rslearn's `period_duration` and fsd's `T = ceil(span / mosaic_days)`.

- **Expect** (predicted from `rslearn/data_sources/utils.py:434-485`):

  | case | fsd `T` | rslearn groups, predicted |
  |---|---|---|
  | `dense_tutorial_window` | 10 | **9** (trailing partial period dropped) |
  | `exact_multiple_no_partial` | 9 | 9 |
  | `two_empty_periods` | 9 | **7** (empty periods dropped) |
  | `default_reverse_time_order` | 9 | 9, but reverse-chronological + a `FutureWarning` |

- **PASS if:** `_result_probe02.json` has `"status": "ok"` and all four cases returned an integer
  group count. **A pass does not mean rslearn matched fsd** — read `metrics.findings`.
- **The result that matters:** `metrics.findings.T_matches_fsd_on_dense_window`.
  - **`false`** (predicted) → rslearn's `T` is data-dependent. Plan C then **requires a
    re-alignment shim in fsd** (map groups back onto their period index, fill gaps) before
    preflight or cross-cell flatten can work at all — i.e. adoption *adds* fsd code rather than
    deleting it. Steps 3–4 stay worth running, but the bar for Plan C has risen.
  - **`true`** → `RSLEARN_READ_2026-07-31.md` §4.2 is **wrong**. Stop and paste the result;
    the read gets re-derived before anything else runs.
- **If it errors on import or signature:** rslearn's API moved between 0.1.13 and whatever
  installed. Paste the traceback; the probe is pinned to the version in Step 1 for this reason.

### Step 3 — can rslearn read and write Azure blob under managed identity, and how fast?

**Written 2026-07-31, after Step 2 landed.** This was billed as the spike's highest-risk
question. **It is now a smaller question than that, and the reason is worth reading before you
run it** — see report §4.1.1.

The source read argued rslearn "inherits" fsd's GDAL/VSI-auth problem because it reads pixels
through rasterio. **It does not.** `rslearn/utils/fsspec.py:157-214` shows that for any
*non-local* path — both the reader and the writer — rslearn opens an fsspec **file object** and
hands *that* to `rasterio.open`. GDAL never receives a remote URL, so GDAL's credential
machinery is bypassed and `adlfs` + `DefaultAzureCredential` does all the work. The only
`rasterio.Env(session=…)` in 54,850 LOC is one AWS-specific source (`nasa_hls.py:244`).

So the question is no longer *"can it authenticate?"* but **"what does it give up by not using
`/vsiadls/`?"** — i.e. throughput.

#### Step 3a — local smoke first (5 seconds, no Azure, no credentials)

The probe calls four real rslearn APIs whose signatures were read from source but never
executed. Probes 01 and 02 each cost a VM round-trip to a wrong assumption. `UPath` on a local
path takes the `LocalFileSystem` branch, so this exercises **every rslearn call in the probe**
with zero Azure involvement:

```bash
cd fsd && source .venv-rslearn/bin/activate
python spike/probes/probe_03_azure.py \
    --out /tmp/rslearn_spike_smoke \
    --dst-prefix /tmp/rslearn_spike_smoke/local
```

- **PASS if:** `q2_raster_format` and `q3_tile_store` both report `"ok": true` and
  `"roundtrip_identical": true`. `q1_upath` will report `adlfs_installed` and a
  `LocalFileSystem` class — that is expected here and is not the real Q1.
- **If a signature is wrong**, it fails here in seconds with a `TypeError`/`AttributeError`
  naming the call. Paste it; do not proceed to 3b.

#### Step 3b — the real thing, on the VM

```bash
az login                       # or rely on the VM's managed identity
source env.local.sh            # AZ_ROOT, AZ_SCRATCH_PREFIX -- see env.example.sh

pip install adlfs              # NOT declared by rslearn -- installing it by hand IS the finding
python spike/probes/probe_03_azure.py \
    --out tests/outputs/rslearn_spike \
    --dst-prefix "$AZ_ROOT/$AZ_SCRATCH_PREFIX/rslearn_spike"
```

- **Moves ~12 MB total** (two 6 MB rasters), writes only under the scratch prefix, and deletes
  what it wrote unless you pass `--keep`.
- **PASS if:** `_result_probe03.json` has `"status": "ok"` and
  `metrics.write_read_works == true`.
- **The numbers that matter:** `q2_raster_format.read_mb_per_s` and
  `q3_tile_store.read_mb_per_s`. Compare them against fsd's `/vsiadls/` throughput on the same
  VM — that comparison is what §4.1.1 leaves open.
- **If `adlfs_installed` is false and Q1 fails:** that is the expected shape of the finding, not
  a mistake. Record it and install `adlfs`.
- **If Q2/Q3 fail with an auth error:** paste it. That would mean fsspec's Azure path needs more
  than `DefaultAzureCredential` on this VM, which is a real cost line for Options A and B.

> 🔒 **The result file records the URL's *shape*, never the URL** (`_redact` in the probe:
> `abfss://<fs>@<account>/<N path segments>`). Paste the JSON freely; do not paste your
> `--dst-prefix`. This branch is a public MIT repo.

#### Step 3c — fsd's `/vsiadls/` baseline over the same object

**The last open number in the spike.** Step 3b measured rslearn at 21.8–23.6 MB/s reading blob
through an fsspec file object. That is a datum, not a verdict — fsd's own route over the *same
object, same VM* is what turns it into a comparison. Report §4.1.2 will not call rslearn's number
fast or slow until this exists.

Two venvs, one object — deliberately, since the charter keeps fsd out of `.venv-rslearn`:

```bash
# 1. re-run probe 03 with --keep so the geotiff survives
source .venv-rslearn/bin/activate
python spike/probes/probe_03_azure.py --out tests/outputs/rslearn_spike --keep \
    --dst-prefix "$AZ_ROOT/$AZ_SCRATCH_PREFIX/rslearn_spike"

# 2. read it back through fsd's VSI route, in fsd's OWN venv
deactivate && source .venv/bin/activate
python spike/probes/probe_03b_fsd_vsi_baseline.py --out tests/outputs/rslearn_spike \
    --url "$AZ_ROOT/$AZ_SCRATCH_PREFIX/rslearn_spike/raster_format/geotiff.tif"

# 3. clean up
python -c "import fsspec,os; fsspec.filesystem('abfss').rm(os.environ['AZ_ROOT'].split('://',1)[1], recursive=True)" 2>/dev/null || \
  echo "clean up $AZ_ROOT/$AZ_SCRATCH_PREFIX/rslearn_spike by hand"
```

- The filename is `geotiff.tif` — `GeotiffRasterFormat.fname` (`raster_format.py:510`).
- Reads three times: the **cold** figure is the honest one for a fan-out that opens each COG
  once; the warm reads only show what caching would buy.
- **PASS if:** `_result_probe03b.json` has `"status": "ok"`. There is no pass/fail bar — it is a
  baseline. Compare `cold_read_mb_per_s` against probe 03's `q2_raster_format.read_mb_per_s`.
- **If it fails with `--url did not translate`:** you passed a local path or a non-`abfss://`
  URL; the probe refuses rather than silently measuring the wrong route.

> ⚠️ **Compare WARM against probe 03, not cold — the cold pair is not like for like.** Found
> after the first run (2026-07-31, cold 3.7 MB/s vs rslearn's 21.8). fsd's first `rio_open` pays
> a one-time AAD token round-trip inside the timed section (`storage_token()`,
> `src/fsd/storage/azure.py:48-56`; the credential is module-cached so calls 2..N are ~free),
> whereas **probe 03 resolves its filesystem in `q1_upath` before timing `q2`** — so rslearn's
> number is measured with auth already warm. Charging fsd one-time auth and not rslearn would
> overstate the gap. The probe now reports `aad_token_seconds` separately so the split is
> visible; use `warm_read_mb_per_s` for the comparison and quote the cold figure only as
> "first-open latency", which is its own real cost in a fan-out that opens one COG per task.

- ⚠️ **One 6 MB object is not a COG read pattern.** A fair test of what GDAL's remote reader
  actually buys — overviews, windowed reads, many small range requests — needs the real archive.
  Treat this as a floor on the question, and say so if the number gets quoted.

### Step 4 — pixel equivalence against the tutorial fixture *(optional; deliberately deferred)*

**Recommendation: do not run this yet, and possibly not at all.** Stating the reasoning rather
than quietly dropping the step.

Step 4 was designed to answer *"does rslearn reproduce fsd's datacube?"* Probe 02 established
that it **cannot**, for three independent reasons that all sit upstream of any pixel comparison:

1. **`T` differs** — 9 vs fsd's 10 on the tutorial window, 7 vs 9 once a period has no scene.
2. **Phase differs** — rslearn's first period starts 2018-04-02, not 04-01 (end-anchoring).
3. **The composite differs in kind** — `period_duration` + `MOSAIC` returns **one
   first-coverage scene per period** (`utils.py:438-442,464-468`), where fsd takes a per-pixel
   **median over every scene in the window**.

To make a pixel diff meaningful you must first write the re-alignment shim *and* find a
compositor combination that reproduces a median. **At that point the diff measures our shim, not
rslearn** — it answers "did I write the adapter correctly?", which is a useful test for an
adapter that exists, and not evidence for a decision about whether to build one.

**What to run instead, if Option B stays live** — the one cheap question §6.2 identifies as
load-bearing and nobody has asked:

> Take one source fsd lacks and genuinely wants (**ERA5-Land** is the best candidate — issue #11
> names it, and rslearn has three variants). Time-box an afternoon. Get it into an fsd-shaped
> array two ways: through rslearn, and by writing it directly against fsd's `Source` seam.
> Compare the LOC, the dependency cost, and how much of rslearn's model you had to learn.

That measures the hybrid's actual value — the *marginal* cost of one more source — which is what
Option B rests on. A pixel diff does not.

**If you do run Step 4 anyway**, one finding is worth capturing precisely: rslearn's
harmonization is opt-in (`harmonize: bool = False`, `copernicus.py:680`) and **hard-asserts
`offset == -1000`** (`copernicus.py:73`). The fixture's pre-Collection-1 products correctly
declare `0`, so **rslearn is predicted to `AssertionError`** against this archive with
harmonization on. That is a genuine, quotable fact about rslearn regardless of the pixel diff.

---

## Success criteria (`_result.json`)

Steps 1 and 2 each write `tests/outputs/rslearn_spike/_result_probe0N.json`:

```json
{ "step": "probe_0N_...", "status": "ok|fail", "pass": true,
  "metrics": { }, "expected": { }, "error": null }
```

**Paste those two files back, not the logs.** `tests/outputs/` is gitignored, so nothing here is
committed by accident.

The spike does **not** pass or fail as a whole — it is a measurement exercise. It is *complete*
when the charter's three questions have numbers against them and
`spike/RSLEARN_SPIKE_REPORT.md` carries a go/no-go on Plan C.

## Stop / observe

- Steps 0 and 2 are seconds. Step 1 is minutes and is the only large download.
- Nothing here is destructive; nothing writes outside `.venv-rslearn/` and
  `tests/outputs/rslearn_spike/`, both gitignored.
- Abort: Ctrl-C. To start over: `rm -rf .venv-rslearn` and redo Step 1.
