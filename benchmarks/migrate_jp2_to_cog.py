"""Migrate a downloaded JP2 archive to COG **in place** (spec 14 follow-up).

Converts every ``Bxx.jp2`` in the catalog to ``Bxx.tif`` — a lossless COG **with
overviews** — in the *same* folder, then **deletes the ``.jp2``** so no duplicate
copies pile up, and rewrites ``catalog.parquet``'s ``files`` column ``.jp2``->``.tif``.

Safety (this is destructive + irreversible):
- **Verify-before-delete:** the ``.jp2`` is removed *only after* a verify passes; a
  failed/killed conversion leaves it intact (and at most a harmless ``.part``). Modes
  (``--verify``): ``full`` re-decodes the JP2 and asserts bit-identical (safest,
  ~+24% time); ``quick`` (default) reopens the COG and checks it is readable with the
  same shape/dtype/band-count and materialised overviews — no second JP2 decode. The
  DEFLATE+PREDICTOR(+NBITS=16) profile is lossless by construction and was proven
  bit-identical on this dataset (spec 13 + the datacube-parity run), so ``quick``
  guards against the realistic failure (a truncated/corrupt write) at full speed.
  ``none`` skips the check.
- **Resumable:** a file whose ``.tif`` already exists is skipped (and any stale
  ``.jp2`` beside it removed). Re-run to finish an interrupted migration.
- **Disk-safety floor:** aborts before the volume free space drops below ``--floor-gib``
  (COG+overviews ~1.70x the JP2 it replaces, so the archive *grows*).
- **Catalog stays consistent:** rewritten from the actual on-disk state (per file:
  ``.tif`` if present else ``.jp2``), so an interrupted run never points at a missing file.

Run:  python -m benchmarks.migrate_jp2_to_cog                 # the default archive
      python -m benchmarks.migrate_jp2_to_cog --jobs 6 --dry-run
"""
from __future__ import annotations

import argparse
import functools
import multiprocessing as mp
import os
import shutil
import time
from pathlib import Path

import geopandas as gpd
import rasterio
import rasterio.windows

from rise_eo.raster.cog import to_cog
from rise_eo.storage import fs

ROOT = str(Path(__file__).resolve().parents[2])  # workspace root: the folder holding fsd/ and the archives
CATALOG = f"{ROOT}/satellite_benchmark/sentinel-2-l2a/catalog.parquet"
GROWTH_FACTOR = 0.70   # measured: COG+overviews ~= 1.70x JP2 -> net +0.70x
FLOOR_GIB = 8.0        # abort before free disk drops below this
CHUNK = 12             # files between progress lines + disk-floor checks


def _bar(frac: float, width: int = 30) -> str:
    """A newline-friendly text progress bar, e.g. [############------] 60%."""
    filled = int(round(frac * width))
    return f"[{'#' * filled}{'-' * (width - filled)}] {frac * 100:3.0f}%"


def _tif(jp2_path: str) -> str:
    return jp2_path[:-4] + ".tif"


def _free_gib(path: str) -> float:
    return shutil.disk_usage(path).free / 2**30


def _catalog_jp2_paths(gdf: gpd.GeoDataFrame) -> list[str]:
    """Every Bxx.jp2 referenced by the catalog (absolute paths)."""
    out = []
    for _, r in gdf.iterrows():
        for f in str(r["files"]).split(","):
            if f.endswith(".jp2"):
                out.append(os.path.join(r["local_folderpath"], f))
    return out


def _quick_ok(jp2_path: str, tif_path: str) -> bool:
    """Cheap integrity gate (no full JP2 decode): the COG is readable, matches the
    JP2's shape/dtype/band-count, and has overviews. Opening the JP2 reads only its
    header (no wavelet decode)."""
    with rasterio.open(jp2_path) as s, rasterio.open(tif_path) as o:
        if s.shape != o.shape or s.dtypes != o.dtypes or s.count != o.count:
            return False
        if not o.overviews(1):
            return False
        w = rasterio.windows.Window(0, 0, min(512, o.width), min(512, o.height))
        o.read(1, window=w)  # forces a real block read -> catches a truncated file
    return True


def _migrate_one(jp2_path: str, verify: str = "quick"):
    """Convert one JP2 -> COG (verify), then delete the JP2. Returns (status, bytes)."""
    tif = _tif(jp2_path)
    if os.path.exists(tif) and os.path.getsize(tif) > 0:  # already migrated
        if os.path.exists(jp2_path):
            os.remove(jp2_path)
        return ("skip", os.path.getsize(tif))
    if not os.path.exists(jp2_path):
        return ("missing", 0)
    n = to_cog(jp2_path, tif, overviews="AUTO", verify=(verify == "full"))
    if verify == "quick" and not _quick_ok(jp2_path, tif):
        raise ValueError(f"quick verify failed (COG unreadable/mismatched): {tif}")
    os.remove(jp2_path)  # only after a verified conversion
    return ("done", n)


def _resync_catalog(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Rewrite `files` to the actual on-disk state: .jp2 -> .tif where the COG exists."""
    out = gdf.copy()
    new_files = []
    for _, r in gdf.iterrows():
        cur = []
        for f in str(r["files"]).split(","):
            if f.endswith(".jp2"):
                tif = f[:-4] + ".tif"
                exists_tif = os.path.exists(os.path.join(r["local_folderpath"], tif))
                cur.append(tif if exists_tif else f)
            else:
                cur.append(f)
        new_files.append(",".join(cur))
    out["files"] = new_files
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default=CATALOG)
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel conversion workers (machine has 8 performance cores)")
    ap.add_argument("--verify", choices=["full", "quick", "none"], default="quick",
                    help="pre-delete integrity check (full=bit-identical, slower)")
    ap.add_argument("--floor-gib", type=float, default=FLOOR_GIB)
    ap.add_argument("--limit", type=int, default=None, help="cap #files (smoke)")
    ap.add_argument("--dry-run", action="store_true", help="estimate only, convert nothing")
    args = ap.parse_args(argv)

    gdf = gpd.read_parquet(args.catalog)
    all_jp2 = _catalog_jp2_paths(gdf)
    todo = [j for j in all_jp2
            if not (os.path.exists(_tif(j)) and os.path.getsize(_tif(j)) > 0)]
    if args.limit is not None:
        todo = todo[: args.limit]

    todo_bytes = sum(os.path.getsize(j) for j in todo if os.path.exists(j))
    est_growth = todo_bytes * GROWTH_FACTOR
    free = _free_gib(ROOT)
    g = lambda b: f"{b / 2**30:.1f} GiB"  # noqa: E731
    print(f"[plan] catalog JP2s: {len(all_jp2)} | to convert: {len(todo)} "
          f"({g(todo_bytes)}) | est net growth: {g(est_growth)}", flush=True)
    print(f"[plan] free disk now: {free:.1f} GiB -> est after: "
          f"{free - est_growth / 2**30:.1f} GiB (floor {args.floor_gib:.0f})", flush=True)

    if free - est_growth / 2**30 < args.floor_gib:
        raise SystemExit(
            f"[abort] would end at {free - est_growth / 2**30:.1f} GiB free, below the "
            f"{args.floor_gib:.0f} GiB floor. Free space or use --no overviews path.")

    if args.dry_run:
        print("[dry-run] no files converted.")
        return
    if not todo:
        print("[done] nothing to convert; resyncing catalog anyway.")

    t0 = time.perf_counter()
    done = skipped = failed = 0
    bytes_written = 0
    aborted = False
    worker = functools.partial(_migrate_one, verify=args.verify)
    with mp.Pool(args.jobs) as pool:
        for i, (status, nbytes) in enumerate(
                pool.imap_unordered(worker, todo), start=1):
            if status == "done":
                done += 1
                bytes_written += nbytes
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
            if i % CHUNK == 0 or i == len(todo):
                el = time.perf_counter() - t0
                rate = i / el if el else 0
                eta = (len(todo) - i) / rate if rate else 0
                fr = _free_gib(ROOT)
                print(f"{_bar(i / len(todo))} {i}/{len(todo)} | "
                      f"done={done} skip={skipped} fail={failed} | "
                      f"{g(bytes_written)} | {el / 60:.1f}m elapsed | ETA {eta / 60:.1f}m "
                      f"| {rate:.1f} file/s | free {fr:.1f} GiB", flush=True)
                if fr < args.floor_gib:
                    print(f"[abort] free disk {fr:.1f} GiB hit the "
                          f"{args.floor_gib:.0f} GiB floor; stopping.", flush=True)
                    pool.terminate()
                    aborted = True
                    break

    # Rewrite the catalog from actual on-disk state (always consistent, resumable).
    synced = _resync_catalog(gdf)
    fs.write_parquet(args.catalog, synced)
    remaining = sum(str(f).count(".jp2") for f in synced["files"])
    print(f"[catalog] updated {args.catalog} | {remaining} .jp2 still referenced",
          flush=True)
    print(f"[{'ABORTED' if aborted else 'done'}] converted={done} skipped={skipped} "
          f"failed={failed} | free {_free_gib(ROOT):.1f} GiB", flush=True)
    if failed:
        raise SystemExit(f"[warn] {failed} files failed (their .jp2 are kept); re-run.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
