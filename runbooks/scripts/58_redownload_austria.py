"""Spec 58 P1 — re-download the Austria test archive from MPC under the new catalog schema.

Run-book: `runbooks/58-redownload-austria-mpc.md`. This is the networked/long step Claude
never runs itself (`CLAUDE.md`) — you run it, and paste back each step's `_result_<step>.json`.

Why it exists: spec 58 D12 renamed `satellite`→`collection` and added `scale`/`properties`
with **no read-time back-compat shim**, so every catalog written before P1 is invalidated.
The existing Austria archive is worse than one generation behind — it carries no `offset`
or `nodata` column at all. Re-ingesting under post-spec-34 code stamps a real, per-item
radiometric offset derived from the baseline the item declares, whatever that turns out to
be, instead of a hardcoded nothing.

Source is **MPC**, not the CDSE the old archive came from (spec 58 D1 made MPC the default:
anonymous, and its assets are already COG so there is no jp2→COG conversion leg). Item ids
and the on-disk layout therefore change — flat `<root>/<item_id>/` instead of CDSE's
`Sentinel-2/MSI/L2A_N0500/YYYY/MM/DD/<id>/`. Nothing reads those paths except the catalog
that is being rewritten anyway.

⚠️ **The two providers serve different processings of the same acquisitions, and both are
correct.** CDSE served the **2023 reprocessing** (baseline N0500 ≥ 04.00 → ESA offset
−1000); MPC serves the **original 2018 processing** (baseline < 04.00 → offset 0). The old
archive was wrong not because its offset was 0 but because it stamped 0 onto *reprocessed*
bytes that needed −1000. `verify` therefore checks the offset against each item's OWN
declared baseline rather than against a constant — see the note by `EXPECTED_SCALE`.

Self-contained by the spec-31 run-book pattern: every step is wrapped so `_result_<step>.json`
is written even on a hard failure, and the download step is idempotent + resume-safe
(`mpc.download` skips any file already on disk, so Ctrl-C then re-run costs only what was
in flight).

Steps, in order — each is a separate invocation:

    .venv/bin/python runbooks/scripts/58_redownload_austria.py reclaim --yes-delete
    .venv/bin/python runbooks/scripts/58_redownload_austria.py discover
    .venv/bin/python runbooks/scripts/58_redownload_austria.py free-disk --yes-delete-the-archive
    .venv/bin/python runbooks/scripts/58_redownload_austria.py download --max-concurrent 16
    .venv/bin/python runbooks/scripts/58_redownload_austria.py verify
    .venv/bin/python runbooks/scripts/58_redownload_austria.py build-cube

`reclaim` and `discover` take `--bands` / `--max-cloudcover`; whatever you price in
`discover`, pass the same values to `download`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import shutil
import sys
import traceback
import urllib.request


def _find_workspace() -> pathlib.Path:
    """The workspace root — the directory holding both `fsd/` and `shapefiles/`.

    Found by walking up from this file rather than by a fixed `parents[N]`: a plain
    `parents[2]` is `fsd/` from the normal checkout but `fsd/.claude/worktrees/<name>/`
    from a worktree, which silently resolves every ROI path to a file that does not
    exist. Walking up finds the real workspace from either. Override with
    `FSD_WORKSPACE=/path/to/fetch_satdata_claude` if you keep the checkout elsewhere.
    """
    override = os.environ.get("FSD_WORKSPACE")
    if override:
        return pathlib.Path(override).resolve()
    here = pathlib.Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "shapefiles" / "AT_ROI.geojson").exists() and (candidate / "fsd").is_dir():
            return candidate
    raise SystemExit(
        f"could not locate the workspace root above {here}: looked for an ancestor "
        "containing both 'shapefiles/AT_ROI.geojson' and 'fsd/'. Set FSD_WORKSPACE."
    )


WORKSPACE = _find_workspace()
# Always the real checkout, never a worktree copy -- the archive and its `tests/outputs/`
# live there and are gitignored, so a worktree has neither.
FSD_ROOT = WORKSPACE / "fsd"

ROI_PATH = WORKSPACE / "shapefiles" / "AT_ROI.geojson"
CELL_PATH = WORKSPACE / "shapefiles" / "s2grid=476da24.geojson"   # 100% inside T33UWP

# In place, exactly where `demos/e2e_austria.py` and `CLAUDE.md`'s real-data note point.
DATA_DIR = FSD_ROOT / "tests" / "outputs" / "demo_e2e" / "imagery"
OUT = FSD_ROOT / "tests" / "outputs" / "p58_redownload"

# The archive's own parameters, copied from `demos/e2e_austria.py` so the re-download
# reproduces the same window/bands rather than a new archive that happens to be nearby.
BANDS = ["B04", "B08", "B8A", "SCL"]
STARTDATE = datetime.datetime(2018, 4, 1)
ENDDATE = datetime.datetime(2018, 9, 30)
MAX_CLOUDCOVER = 70
# Headroom over the CDSE archive's 207: MPC dedupes reprocessed items differently and
# publishes its own `eo:cloud_cover`, so the count is expected to be near 207, not equal.
# `mpc.download` raises (before any bytes move) if discovery exceeds this.
MAX_TILES = 260
EXPECTED_MGRS = {"T33UVP", "T33UWP", "T33UVQ", "T33UWQ"}

EXPECTED_SCALE = 1e-4

# ⚠️ There is deliberately NO expected-offset constant. An earlier draft of this script
# asserted `offset == -1000` for every row, reasoning from the OLD archive: CDSE served the
# 2023 reprocessing of this window (ids like `..._N0500_..._20230710T001349`), and baseline
# >= 04.00 means ESA's -1000 shift. The `discover` step falsified that in seconds --
# MPC reports `offsets_declared: [0]`, because MPC serves the ORIGINAL 2018 processing of
# the same acquisitions, which predates baseline 04.00 and genuinely carries no shift.
#
# Both are correct for their own bytes. So `verify` derives the expected offset per row from
# the baseline the item itself declares (now available in the `properties` column, spec 58
# D12) and checks the catalog agrees -- which is a stronger check than any constant, and one
# that stays right when the provider or the window changes.

# Sampled in `discover` to size the delete decision; falls back to this if the HEADs fail.
# NOT a good default: the first real sample measured 0.549 GB/granule for B04+B08+B8A+SCL
# from MPC, ~43% above the CDSE-converted archive's 0.384. Always trust the sample.
FALLBACK_GB_PER_GRANULE = 0.55


def _result(step: str, expected: dict) -> dict:
    return {"step": step, "status": "ok", "pass": False,
            "metrics": {}, "expected": expected, "error": None}


def _write(result: dict) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"_result_{result['step']}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str), flush=True)
    print(f"\n[58] wrote {path}", flush=True)
    return 0 if result["pass"] else 1


def _free_gb(path: pathlib.Path) -> float:
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    return usage.free / 1e9


def _dir_gb(path: pathlib.Path) -> float:
    """Size of a directory tree **or a single file**, in GB.

    The file case is not hypothetical: `RECLAIM_CANDIDATES` names `rf.joblib` (1.2 GiB), and
    an `os.walk`-only version reports a plain file as 0.0 -- which would have quietly
    under-reported the reclaim by more than a gigabyte and made the dry run look pointless.
    """
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / 1e9
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


# --- step 0: reclaim stale derived artifacts -----------------------------------------

# Everything the old archive fed, which is stale the moment it is deleted. `imagery/` is
# NOT here (that is `free-disk`'s job and it needs its own confirmation), and neither is
# `mpc_baseline/` -- 1.6 GiB, but it is the reference cube this run-book's own GeoTIFF
# writer was validated against, and the only local MPC radiometry sample.
RECLAIM_CANDIDATES = (
    "tests/outputs/demo_e2e/model_outputs",
    "tests/outputs/demo_e2e/bundle",
    "tests/outputs/demo_e2e/rf.joblib",
    "tests/outputs/demo_e2e/training_data",
    "tests/outputs/demo_e2e/training_run",
    "tests/outputs/spec34_mixed_baseline",
)


def step_reclaim(confirmed: bool) -> int:
    """Free the derived artifacts of the archive being replaced. Reports what it would
    remove and refuses to act without `--yes-delete`, same shape as `free-disk`.

    This exists because the first real run of this run-book did not fit: the measured
    0.549 GB/granule put the archive at ~117 GB against ~104 GB of headroom. Reclaiming
    these buys ~10 GB, which is necessary but on its own may still not be enough -- re-run
    `discover` with `--bands`/`--max-cloudcover` to price the rest.
    """
    result = _result("reclaim", {"reclaimed_gb": "> 0", "candidates_removed": True})
    try:
        sizes = {c: _dir_gb(FSD_ROOT / c) for c in RECLAIM_CANDIDATES}
        total = sum(sizes.values())
        if not confirmed:
            result["status"] = "fail"
            result["metrics"] = {"would_reclaim_gb": round(total, 1),
                                 "per_path_gb": {k: round(v, 2) for k, v in sizes.items()},
                                 "free_gb_now": round(_free_gb(FSD_ROOT), 1)}
            result["error"] = ("dry run: pass --yes-delete to actually remove these. "
                               "Nothing was deleted.")
            return _write(result)

        free_before = _free_gb(FSD_ROOT)
        for c in RECLAIM_CANDIDATES:
            target = FSD_ROOT / c
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
        free_after = _free_gb(FSD_ROOT)
        result["metrics"] = {
            "reclaimed_gb": round(total, 1),
            "per_path_gb": {k: round(v, 2) for k, v in sizes.items()},
            "free_gb_before": round(free_before, 1),
            "free_gb_after": round(free_after, 1),
            "all_removed": not any((FSD_ROOT / c).exists() for c in RECLAIM_CANDIDATES),
        }
        result["pass"] = result["metrics"]["all_removed"]
        if not result["pass"]:
            result["status"] = "fail"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return _write(result)


# --- step 1: discover ---------------------------------------------------------------

def step_discover(bands: list[str], max_cloudcover: float) -> int:
    """Query MPC and size the run. **No bytes are transferred and nothing is deleted** --
    this runs BEFORE the destructive step on purpose, so a bad ROI/window/collection costs
    nothing but a STAC query.

    `bands` and `max_cloudcover` are overridable (`--bands`, `--max-cloudcover`) so a run
    that will not fit on disk can be re-priced in seconds instead of guessed at. Whatever
    you settle on here, pass the SAME values to `download`.
    """
    result = _result("discover", {
        "granules_between": [150, MAX_TILES],
        "mgrs_tiles": sorted(EXPECTED_MGRS),
        "collection": "sentinel-2-l2a",
        "free_gb_after_delete_exceeds_estimate": True,
    })
    try:
        from fsd import collections as _collections
        from fsd.sources import mpc

        declaration = _collections.get("sentinel-2-l2a")
        print("[58] querying MPC STAC (no bytes) ...", flush=True)
        gdf = mpc.query_catalog(str(ROI_PATH), STARTDATE, ENDDATE,
                                max_cloudcover=max_cloudcover, collection="sentinel-2-l2a")
        mgrs = sorted({i.split("_")[-2] for i in gdf["id"]})

        # Measure, don't guess: HEAD the band assets of a few real items so the delete
        # decision below rests on this run's own numbers, not on a constant in this file.
        gb_per_granule, sampled = _sample_granule_gb(mpc, declaration, bands, max_cloudcover)

        n = len(gdf)
        est_gb = n * gb_per_granule
        current_gb = _dir_gb(DATA_DIR)
        free_now = _free_gb(FSD_ROOT)
        free_after_delete = free_now + current_gb

        result["metrics"] = {
            "bands": list(bands),
            "max_cloudcover": max_cloudcover,
            "granules": n,
            "mgrs_tiles": mgrs,
            "collections": sorted(set(gdf["collection"])),
            "date_min": str(gdf["timestamp"].min()),
            "date_max": str(gdf["timestamp"].max()),
            "offsets_declared": sorted({int(v) for v in gdf["offset"]}),
            "scales_declared": sorted({float(v) for v in gdf["scale"]}),
            "gb_per_granule_sampled": round(gb_per_granule, 3),
            "granules_sampled_for_size": sampled,
            "estimated_download_gb": round(est_gb, 1),
            "existing_archive_gb": round(current_gb, 1),
            "free_gb_now": round(free_now, 1),
            "free_gb_after_delete": round(free_after_delete, 1),
        }
        result["pass"] = (
            150 <= n <= MAX_TILES
            and set(mgrs) == EXPECTED_MGRS
            and sorted(set(gdf["collection"])) == ["sentinel-2-l2a"]
            and free_after_delete > est_gb * 1.25
        )
        if not result["pass"]:
            result["status"] = "fail"
    except Exception as exc:  # noqa: BLE001 - the whole point is to record the failure
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return _write(result)


def _sample_granule_gb(mpc, declaration, bands: list[str], max_cloudcover: float,
                       n_sample: int = 3) -> tuple[float, int]:
    """Mean GB per granule across `n_sample` real items, from HTTP `Content-Length` on the
    signed asset hrefs. Returns `(gb_per_granule, n_sampled)`; falls back to
    `FALLBACK_GB_PER_GRANULE` (0 sampled) if MPC won't answer a HEAD."""
    try:
        roi_gdf = mpc._roi_gdf(str(ROI_PATH))
        items = mpc._dedupe_reprocessed_items(
            mpc._search_items(roi_gdf, STARTDATE, ENDDATE, max_cloudcover=max_cloudcover,
                              collection="sentinel-2-l2a")
        )[:n_sample]
        if not items:
            return FALLBACK_GB_PER_GRANULE, 0
        totals = []
        for it in items:
            nbytes = 0
            for href, _dst, _band in mpc._select_item_files(
                it, bands, str(DATA_DIR), collection="sentinel-2-l2a",
                declaration=declaration,
            ):
                req = urllib.request.Request(href, method="HEAD")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    nbytes += int(resp.headers.get("Content-Length") or 0)
            totals.append(nbytes)
        if not any(totals):
            return FALLBACK_GB_PER_GRANULE, 0
        return (sum(totals) / len(totals)) / 1e9, len(totals)
    except Exception as exc:  # noqa: BLE001 - an estimate is a nicety, not the step
        print(f"[58] size sampling failed ({type(exc).__name__}: {exc}); "
              f"falling back to {FALLBACK_GB_PER_GRANULE} GB/granule", flush=True)
        return FALLBACK_GB_PER_GRANULE, 0


# --- step 2: free the disk (DESTRUCTIVE) ---------------------------------------------

def step_free_disk(confirmed: bool) -> int:
    """Delete the old archive. **This is the only destructive step in the run-book** and it
    refuses to act without `--yes-delete-the-archive`.

    Only `imagery/` goes. `training_run/` is also stale but is left alone deliberately: the
    spec 58 D4 cube-path digest now keys on `collection` + the declaration hash, so cubes
    built under the old key land at a DIFFERENT window segment and can never be silently
    reused as if they were fresh. Deleting them is hygiene (60 MB), not correctness -- the
    run-book says so rather than making this script do it."""
    result = _result("free-disk", {
        "archive_removed": True,
        "free_gb_after": "> estimated_download_gb * 1.25 from the discover step",
    })
    try:
        before_gb = _dir_gb(DATA_DIR)
        free_before = _free_gb(FSD_ROOT)
        if not confirmed:
            result["status"] = "fail"
            result["error"] = (
                f"refusing to delete {DATA_DIR} ({before_gb:.1f} GB) without "
                "--yes-delete-the-archive. Re-run the `discover` step first and check its "
                "free_gb_after_delete against estimated_download_gb."
            )
            result["metrics"] = {"would_delete_gb": round(before_gb, 1),
                                 "path": str(DATA_DIR)}
            return _write(result)

        if DATA_DIR.exists():
            print(f"[58] deleting {DATA_DIR} ({before_gb:.1f} GB) ...", flush=True)
            shutil.rmtree(DATA_DIR)
        else:
            print(f"[58] {DATA_DIR} already absent -- nothing to delete", flush=True)

        free_after = _free_gb(FSD_ROOT)
        result["metrics"] = {
            "deleted_gb": round(before_gb, 1),
            "free_gb_before": round(free_before, 1),
            "free_gb_after": round(free_after, 1),
            "archive_removed": not DATA_DIR.exists(),
        }
        result["pass"] = not DATA_DIR.exists()
        if not result["pass"]:
            result["status"] = "fail"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return _write(result)


# --- step 3: download ----------------------------------------------------------------

def step_download(bands: list[str], max_cloudcover: float, max_concurrent: int) -> int:
    """The long leg. Resume-safe: `mpc.download` skips any file already on disk, so Ctrl-C
    and re-run costs only the transfers that were in flight.

    `max_concurrent` is why this used to crawl: `sources.mpc.download` defaults to
    `config.MPC_MAX_CONCURRENT` (4 -- a value whose own comment says it was picked for "a
    single tile/band runbook"), and `api.download` did not forward the parameter at all
    until this run-book needed it. 200+ granules x 4 bands over 4 threads is latency-bound,
    not bandwidth-bound. 16 is a reasonable default against MPC, a public Azure endpoint.

    Pass the SAME `bands`/`max_cloudcover` you priced in `discover`, or the archive will not
    be the one you sized the disk for.
    """
    result = _result("download", {
        "catalog_written": True,
        "granules_between": [150, MAX_TILES],
        "granules_incomplete": 0,
    })
    try:
        import time

        from fsd import api

        t0 = time.time()
        print(f"[58] downloading {bands} -> {DATA_DIR} "
              f"(max_concurrent={max_concurrent}; progress on; Ctrl-C is resume-safe)",
              flush=True)
        catalog_filepath = api.download(
            roi=str(ROI_PATH),
            startdate=STARTDATE,
            enddate=ENDDATE,
            bands=bands,
            dst_folderpath=str(DATA_DIR),
            source="mpc",
            collection="sentinel-2-l2a",
            max_tiles=MAX_TILES,
            max_cloudcover=max_cloudcover,
            progress=True,
            max_concurrent=max_concurrent,
        )
        elapsed = time.time() - t0

        from fsd.storage import fs
        gdf = fs.read_parquet(catalog_filepath)
        archive_bands, uneven = _bands_in_catalog(gdf)
        # A partial run is the normal outcome of a throttled or interrupted download, and
        # `granules` alone cannot tell "58 complete granules" from "58 granules of which 8
        # are missing their 10 m bands". Report the shortfall so the pasted `_result.json`
        # is diagnosable on its own -- `mpc.download` prints the failure REASONS
        # (`_print_failure_summary`), this records the SHAPE of what is missing.
        want = set(bands)
        per_granule = {
            r["id"]: want - {f.rsplit(".", 1)[0] for f in str(r["files"]).split(",")}
            for _, r in gdf.iterrows()
        }
        incomplete = {gid: sorted(missing) for gid, missing in per_granule.items() if missing}
        result["metrics"] = {
            "bands": list(bands),
            "max_cloudcover": max_cloudcover,
            "max_concurrent": max_concurrent,
            "catalog_filepath": catalog_filepath,
            "granules": len(gdf),
            "archive_bands": archive_bands,
            "granules_incomplete": len(incomplete),
            "incomplete_examples": dict(list(incomplete.items())[:5]),
            "granules_uneven": uneven[:5],
            "files_on_disk": sum(1 for _ in DATA_DIR.rglob("*.tif")),
            "archive_gb": round(_dir_gb(DATA_DIR), 1),
            "elapsed_s": round(elapsed, 1),
            "free_gb_after": round(_free_gb(FSD_ROOT), 1),
        }
        # Re-run the SAME command to resume: `mpc.download` skips what is already on disk,
        # so a partial run costs only the missing files next time.
        result["pass"] = (
            os.path.exists(catalog_filepath)
            and 150 <= len(gdf) <= MAX_TILES
            and not incomplete
        )
        if not result["pass"]:
            result["status"] = "fail"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return _write(result)


# --- step 4: verify ------------------------------------------------------------------

def step_verify() -> int:
    """Read the new archive back and assert it is what P1 says a catalog is now: the new
    columns, a v2 declaration stamp, a per-row offset that MATCHES THE BASELINE THE ITEM
    ITSELF DECLARES, and SCL left un-offset."""
    result = _result("verify", {
        "columns_match_catalog_COLUMNS": True,
        "declaration_stamp_is_s2_l2a": True,
        "declaration_version_is_2": True,
        "offset_matches_declared_baseline": True,
        "scale": EXPECTED_SCALE,
        "properties_non_empty": True,
        "scl_never_offset": True,
        "every_granule_has_the_same_bands": True,
    })
    try:
        from fsd.catalog import declaration as dm
        from fsd.catalog.catalog import COLUMNS
        from fsd.storage import fs

        catalog_filepath = str(DATA_DIR / "catalog.parquet")
        # Via the storage seam, not gpd.read_parquet: `fs.read_parquet` is what restores the
        # footer's declaration stamp into `.attrs` (spec 35), which is half of what this
        # step checks.
        gdf = fs.read_parquet(catalog_filepath)
        stamp_raw = gdf.attrs.get(dm.ATTRS_KEY)
        stamp = dm.from_json(stamp_raw) if stamp_raw else None

        offsets = sorted({int(v) for v in gdf["offset"]})
        scales = sorted({float(v) for v in gdf["scale"]})
        props_non_empty = int((gdf["properties"].astype(str) != "{}").sum())
        archive_bands, missing_bands = _bands_in_catalog(gdf)
        scl_offset_ok = _scl_gdal_tag_is_unoffset(gdf)
        version = (stamp_raw or {}).get("fsd_declaration_version")
        offset_mismatches, baselines = _offset_matches_baseline(gdf)

        checks = {
            "columns_match_catalog_COLUMNS": list(gdf.columns) == COLUMNS,
            "declaration_stamp_is_s2_l2a": stamp == dm.S2_L2A_DECLARATION,
            "declaration_version_is_2": version == 2,
            "offset_matches_declared_baseline": not offset_mismatches,
            "scale": scales == [EXPECTED_SCALE],
            "properties_non_empty": props_non_empty == len(gdf),
            "scl_never_offset": scl_offset_ok,
            "every_granule_has_the_same_bands": not missing_bands,
        }
        result["metrics"] = {
            "granules": len(gdf),
            "archive_bands": archive_bands,
            "columns": list(gdf.columns),
            "offsets_seen": offsets,
            "baselines_seen": baselines,
            "offset_mismatches": offset_mismatches[:5],
            "scales_seen": scales,
            "granules_with_properties": props_non_empty,
            "granules_missing_a_band": missing_bands[:5],
            "declaration_version": version,
            "checks": {k: bool(v) for k, v in checks.items()},
        }
        result["pass"] = all(checks.values())
        if not result["pass"]:
            result["status"] = "fail"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return _write(result)


def _bands_in_catalog(gdf) -> tuple[list[str], list[str]]:
    """The band set the archive actually holds, and any granule that disagrees with it.

    Read from the catalog's own `files` column rather than from this module's `BANDS`
    constant or a `--bands` flag. The constant is the *default* request; the archive on disk
    is whatever the run settled on -- the first real run of this run-book downloaded
    `B04,B08,SCL` (B8A dropped to fit the disk), and a `verify` that checked the constant
    would have reported all 184 granules as missing a band. Reading the artifact removes a
    whole class of "I passed different flags to different steps" failure.

    The real invariant is INTERNAL CONSISTENCY: every granule carries the same bands as
    every other. Which bands is the operator's choice; a granule short of one is the bug.
    Returns `(sorted band union, ids disagreeing with that union)`.
    """
    per_granule = {
        r["id"]: frozenset(f.rsplit(".", 1)[0] for f in str(r["files"]).split(",") if f)
        for _, r in gdf.iterrows()
    }
    if not per_granule:
        return [], []
    union: set[str] = set().union(*per_granule.values())
    # Non-raster sidecars (CDSE writes `MTD_TL.xml`) are not bands and are not required
    # to be uniform; the builder skips them too (`flatten_catalog`).
    union -= {"MTD_TL"}
    odd = [gid for gid, bands in per_granule.items() if not union.issubset(bands)]
    return sorted(union), odd


def _offset_matches_baseline(gdf) -> tuple[list[dict], list[str]]:
    """Per row: does the catalog's `offset` match what the item's OWN declared processing
    baseline implies? Returns `(mismatches, baselines_seen)`.

    ESA's rule (`fsd.sources._s2_radiometry`): offset is −1000 for baseline ≥ 04.00, else 0.
    Reading the baseline out of the `properties` column (spec 58 D12's addition) and
    re-deriving the offset checks the ingest end to end without assuming which provider or
    which reprocessing this archive came from — MPC's original-2018 items declare a pre-04.00
    baseline and correctly carry offset 0, while CDSE's 2023-reprocessed copies of the SAME
    acquisitions declare N0500 and correctly carry −1000. A constant would call one of those
    a failure.

    A row whose `properties` names no baseline is a mismatch, not a pass: `offset_for_item`
    raises rather than defaulting for exactly that reason (spec 34 §3a A1).
    """
    from fsd.sources._s2_radiometry import _BASELINE_PROPS, baseline_tuple

    mismatches: list[dict] = []
    baselines: set[str] = set()
    for _, row in gdf.iterrows():
        try:
            props = json.loads(str(row["properties"]) or "{}")
        except json.JSONDecodeError:
            mismatches.append({"id": row["id"], "why": "properties is not valid JSON"})
            continue
        baseline = next((props[p] for p in _BASELINE_PROPS if p in props), None)
        if baseline is None:
            mismatches.append({"id": row["id"],
                               "why": f"no baseline among {list(_BASELINE_PROPS)}"})
            continue
        baselines.add(str(baseline))
        want = -1000 if baseline_tuple(str(baseline)) >= (4, 0) else 0
        got = int(row["offset"])
        if got != want:
            mismatches.append({"id": row["id"], "baseline": baseline,
                               "offset_in_catalog": got, "offset_from_baseline": want})
    return mismatches, sorted(baselines)


def _scl_gdal_tag_is_unoffset(gdf) -> bool:
    """SCL is a classification, not a DN — `S2_L2A_DECLARATION.radiometry_bands` excludes it,
    so its COG must carry scale 1 / offset 0 while B04's carries 1e-4. This is the on-disk
    counterpart of `test_transfer_and_stamp_one_never_offsets_mask_band`, and it is the check
    that would have caught the `radiometry_bands=None` declaration P1's review re-derived."""
    try:
        import rasterio

        folder = gdf["local_folderpath"].iloc[0]
        with rasterio.open(os.path.join(folder, "SCL.tif")) as src:
            scl_ok = (src.scales[0] or 1) == 1 and (src.offsets[0] or 0) == 0
        with rasterio.open(os.path.join(folder, "B04.tif")) as src:
            b04_ok = abs((src.scales[0] or 1) - EXPECTED_SCALE) < 1e-12
        return bool(scl_ok and b04_ok)
    except Exception as exc:  # noqa: BLE001
        print(f"[58] SCL tag check could not run: {type(exc).__name__}: {exc}", flush=True)
        return False


# --- step 5: build a cube + a GeoTIFF to eyeball -------------------------------------

def step_build_cube() -> int:
    """Prove the archive is usable, not just well-formed: build one cube from a cell wholly
    inside T33UWP (the control) and one from the AT_ROI cell touching the most MGRS tiles
    (the multi-CRS seam), and write a GeoTIFF of each for QGIS.

    `CLAUDE.md` is explicit that raster work gets eyeballed in QGIS rather than trusted to a
    unit test — this step produces the files to open, it does not claim they look right."""
    result = _result("build-cube", {
        "control_cube_built": True,
        "seam_cube_built": True,
        "nodata_fraction_below": 0.9,
    })
    try:
        import geopandas as gpd

        from fsd.storage import fs

        cube_dir = OUT / "cubes"
        cube_dir.mkdir(parents=True, exist_ok=True)
        catalog_filepath = str(DATA_DIR / "catalog.parquet")

        # Build with the bands the archive HAS, not the ones this module defaults to --
        # same reasoning as `verify` (see `_bands_in_catalog`). Requesting a band the
        # catalog lacks would fail the build for a reason that has nothing to do with the
        # archive being good or bad.
        bands, _odd = _bands_in_catalog(fs.read_parquet(catalog_filepath))
        print(f"[58] building with the archive's own bands: {bands}", flush=True)

        control = gpd.read_file(CELL_PATH)
        seam, seam_tiles = _most_multi_tile_cell(catalog_filepath)
        cases = {"control_T33UWP": control, "seam_multi_tile": seam}

        metrics = {"seam_cell_mgrs_tile_count": seam_tiles, "bands": bands}
        for name, cell_gdf in cases.items():
            cell_fp = cube_dir / f"{name}.geojson"
            cell_gdf.to_file(cell_fp, driver="GeoJSON")
            metrics[name] = _build_one(str(cell_fp), catalog_filepath, cube_dir / name,
                                       bands)

        result["metrics"] = metrics
        result["pass"] = all(
            metrics[n].get("built") and metrics[n].get("nodata_fraction", 1.0) < 0.9
            for n in cases
        )
        if not result["pass"]:
            result["status"] = "fail"
    except Exception as exc:  # noqa: BLE001
        result["status"] = "fail"
        result["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return _write(result)


def _most_multi_tile_cell(catalog_filepath: str):
    """The AT_ROI grid cell whose filtered catalog spans the most distinct MGRS tiles — the
    multi-CRS merge path. Since `satellite_benchmark/` was deleted this is the only real-data
    cover for that seam (`CLAUDE.md`); the Ethiopia ROI has no imagery behind it any more."""
    import geopandas as gpd

    from fsd import grid as _grid
    from fsd.catalog.catalog import filter_gdf
    from fsd.storage import fs

    catalog_gdf = fs.read_parquet(catalog_filepath)
    grids = _grid.roi_to_s2_grids(str(ROI_PATH))
    best, best_n = None, 0
    for _, row in grids.iterrows():
        one = gpd.GeoDataFrame({"id": [row["id"]], "geometry": [row["geometry"]]},
                               crs=grids.crs)
        subset = filter_gdf(catalog_gdf, one, STARTDATE, ENDDATE)
        # MGRS tile straight off the item id (`..._T33UWP_...`) rather than adding a column
        # to the shared catalog -- `filter_gdf` explicitly does not mutate its input, and
        # this keeps it that way.
        n = len({i.split("_")[-2] for i in subset["id"]}) if len(subset) else 0
        if n > best_n:
            best, best_n = one, n
    print(f"[58] seam cell spans {best_n} MGRS tiles", flush=True)
    if best is None:
        raise RuntimeError("no AT_ROI grid cell had any catalog rows")
    return best, best_n


def _build_one(cell_filepath: str, catalog_filepath: str, out_dir: pathlib.Path,
               bands: list[str]) -> dict:
    """One cube through the real workflow entry point, then a GeoTIFF of its first
    timestamp for QGIS. Returns the metrics dict this cell contributes."""
    import numpy as np
    import pandas as pd

    from fsd.workflows import create_datacube

    out_dir.mkdir(parents=True, exist_ok=True)
    create_datacube.run_create_datacube(
        catalog_filepath=catalog_filepath,
        timestamp_col="timestamp",
        shapefilepath=cell_filepath,
        id_col="id",
        run_folderpath=str(out_dir / "run"),
        startdate=STARTDATE,
        enddate=ENDDATE,
        bands=bands,
        mosaic_days=20,
        csv_filepath=str(out_dir / "input.csv"),
        label_col=None,
        collection="sentinel-2-l2a",
        cores=1,
        runner="local",
    )

    row = pd.read_csv(out_dir / "input.csv").iloc[0]
    cube_fp = str(row["datacube_filepath"])
    if not os.path.exists(cube_fp):
        return {"built": False, "reason": f"no datacube at {cube_fp}"}

    # `datacube.npy` is 4-D `(timestamps|ids, height, width, bands)` and
    # `metadata.pickle.npy` is `{geotiff_metadata, timestamps, ids, bands, ...}` -- see
    # `builder.py`'s module docstring. The 5-D `(samples, timestamps, H, W, bands)` contract
    # in `CLAUDE.md` is the FLATTENED training array, a different artifact; don't index this
    # one as if it were that.
    cube = np.load(cube_fp)
    meta = np.load(cube_fp.replace("datacube.npy", "metadata.pickle.npy"),
                   allow_pickle=True).item()
    return {
        "built": True,
        "shape": list(cube.shape),
        "dtype": str(cube.dtype),
        "bands": list(meta.get("bands", [])),
        "timestamps": len(meta.get("timestamps", [])),
        "nodata_fraction": round(float((cube == 0).mean()), 4),
        "geotiff": _write_geotiff(cube, meta, out_dir / "first_timestamp_rgb.tif"),
        "export_folderpath": str(row["export_folderpath"]),
    }


def _write_geotiff(cube, meta, dst: pathlib.Path) -> str:
    """First timestamp, B08/B04/B8A as a 3-band GeoTIFF — the file to open in QGIS. Georeferenced
    from the cube's own `geotiff_metadata` (the reference-image profile the builder merged onto),
    so it lands in the right place on the map rather than at the origin.

    SCL is deliberately not written: it is dropped from the cube by the declared mask anyway,
    and a classification band in an RGB composite is noise."""
    import numpy as np
    import rasterio

    bands = list(meta["bands"])
    order = [bands.index(b) for b in ("B08", "B04", "B8A") if b in bands]
    profile = meta.get("geotiff_metadata") or {}
    arr = np.moveaxis(cube[0][..., order], -1, 0)   # (T,H,W,B) -> first T -> (3,H,W)
    with rasterio.open(
        dst, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
        count=arr.shape[0], dtype=arr.dtype,
        crs=profile.get("crs"), transform=profile.get("transform"), nodata=0,
    ) as out:
        out.write(arr)
    print(f"[58] wrote {dst} -- open this in QGIS", flush=True)
    return str(dst)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="58_redownload_austria.py",
        description="Spec 58 P1 re-download of the Austria archive from MPC.",
    )
    p.add_argument("step", choices=["reclaim", "discover", "free-disk", "download",
                                    "verify", "build-cube"])
    p.add_argument("--yes-delete-the-archive", action="store_true",
                   help="required by `free-disk`: confirms deleting the imagery archive")
    p.add_argument("--yes-delete", action="store_true",
                   help="required by `reclaim`: confirms removing stale derived artifacts")
    p.add_argument("--bands", default=",".join(BANDS),
                   help=f"comma-separated bands (default: {','.join(BANDS)}). Pass the SAME "
                        "value to `discover` and `download`.")
    p.add_argument("--max-cloudcover", type=float, default=MAX_CLOUDCOVER,
                   help=f"discovery cloud-cover ceiling (default: {MAX_CLOUDCOVER}). "
                        "Lowering it is the cheapest way to shrink a run that will not fit.")
    p.add_argument("--max-concurrent", type=int, default=16,
                   help="concurrent band-file transfers (default: 16). The source default "
                        "is 4, which is far too low for a whole archive.")
    args = p.parse_args(argv)
    bands = [b.strip() for b in args.bands.split(",") if b.strip()]

    if args.step == "reclaim":
        return step_reclaim(args.yes_delete)
    if args.step == "discover":
        return step_discover(bands, args.max_cloudcover)
    if args.step == "free-disk":
        return step_free_disk(args.yes_delete_the_archive)
    if args.step == "download":
        return step_download(bands, args.max_cloudcover, args.max_concurrent)
    if args.step == "verify":
        return step_verify()
    return step_build_cube()


if __name__ == "__main__":
    sys.exit(main())
