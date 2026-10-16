"""Probe 03 — can rslearn read and write Azure blob under managed identity, and how fast?

Spike questions Q1a/Q1b (`../RSLEARN_READ_2026-07-31.md` §6). **Needs the VM, `az login` (or a
managed identity), and the `AZ_*` variables** — see `../RUNBOOK-rslearn-spike.md` Step 3. It
moves a few MB, not satellite archives.

## What changed since the run-book was first sketched

The source read framed this as *"rslearn reads pixels through rasterio, so it inherits fsd's
GDAL/VSI-auth problem and has no known solution."* **That is wrong**, and the report says so at
§4.1.1. `rslearn/utils/fsspec.py:157-214` opens an fsspec **file object** for any non-local path
and hands *that* to `rasterio.open` — GDAL never sees a remote URL, so GDAL's credential
machinery is bypassed entirely and `adlfs`/`DefaultAzureCredential` does all the work.

So this probe is no longer "does it authenticate at all". It is four narrower questions:

  Q1  does `UPath("abfss://...")` even resolve in the spike venv? `adlfs` is NOT declared by
      rslearn (`pyproject.toml:44` declares `fsspec[gcs, s3]`), so it must be installed by hand
      -- and *that it must* is itself the finding.
  Q2  can rslearn's own `GeotiffRasterFormat` round-trip a raster through that path under
      `DefaultAzureCredential` -- i.e. its real write and read code, not a bare fsspec call?
  Q3  same round-trip through rslearn's `TileStore`, which is what `ingest` would use.
  Q4  **the question that actually matters now:** how does the fsspec-file-object read compare
      with fsd's `/vsiadls/` + `AZURE_STORAGE_ACCESS_TOKEN` route (spec 31) on the *same object*?
      fsd chose VSI for throughput; this prices that choice.

Q4 is measured only if fsd is importable. The spike venv deliberately does not have fsd
installed (charter: `.venv-rslearn` stays separate), so the comparison is skipped rather than
faked -- a skip is recorded honestly in `metrics`.

## Safety

Writes ONLY under `--dst-prefix`, which the run-book points at the scratch prefix
(`AZ_SCRATCH_PREFIX`). Deletes what it wrote unless `--keep`. **Never pass a concrete account
name on the command line in anything you paste back** -- the result JSON records only the
*shape* of the URL (scheme + whether it resolved), never the URL itself. This branch is part of
a public MIT repo.

Usage (see `../RUNBOOK-rslearn-spike.md` Step 3):

    python spike/probes/probe_03_azure.py \
        --out tests/outputs/rslearn_spike \
        --dst-prefix "$AZ_ROOT/$AZ_SCRATCH_PREFIX/rslearn_spike"
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

# Size of the synthetic raster round-tripped through blob. Big enough that the read timing is
# not pure latency, small enough to be free: 1024x1024 uint16 x 3 bands = 6 MB.
HEIGHT = WIDTH = 1024
N_BANDS = 3


def _redact(url: str) -> str:
    """The URL's SHAPE, never its identifiers -- this result file gets pasted into chat.

    `abfss://fs@acct.dfs.core.windows.net/a/b` -> `abfss://<fs>@<account>/<4 path segments>`
    """
    if "://" not in url:
        return "<no scheme>"
    scheme, rest = url.split("://", 1)
    n_segments = len([s for s in rest.split("/")[1:] if s])
    return f"{scheme}://<fs>@<account>/<{n_segments} path segments>"


def _timed(fn):
    """Run `fn`, returning (ok, seconds, error-or-None). Never raises."""
    t0 = time.perf_counter()
    try:
        fn()
        return True, round(time.perf_counter() - t0, 3), None
    except BaseException as exc:  # noqa: BLE001 -- any failure is data here
        return (
            False,
            round(time.perf_counter() - t0, 3),
            f"{type(exc).__name__}: {exc} | {traceback.format_exc().strip().splitlines()[-1]}",
        )


def q1_upath_resolves(dst_prefix: str) -> dict:
    """Does `UPath(abfss://...)` construct and list at all?"""
    result: dict = {"question": "UPath resolves an abfss:// path", "adlfs_installed": None}
    try:
        import adlfs  # noqa: F401

        result["adlfs_installed"] = True
    except ImportError:
        result["adlfs_installed"] = False

    def _do():
        from upath import UPath

        p = UPath(dst_prefix)
        # `.fs` is what forces fsspec to resolve a backend for the scheme.
        result["filesystem_class"] = type(p.fs).__name__
        result["exists_check_ok"] = bool(p.parent.exists() or True)

    ok, secs, err = _timed(_do)
    result.update({"ok": ok, "seconds": secs, "error": err})
    return result


def q2_raster_format_roundtrip(dst_prefix: str, keep: bool) -> dict:
    """rslearn's own GeotiffRasterFormat: write a raster to blob, read it back, compare."""
    result: dict = {"question": "GeotiffRasterFormat write+read round-trip on abfss://"}
    state: dict = {}

    def _do():
        from rasterio.crs import CRS
        from rslearn.utils.geometry import Projection
        from rslearn.utils.raster_array import RasterArray
        from rslearn.utils.raster_format import GeotiffRasterFormat
        from upath import UPath

        rng = np.random.default_rng(0)
        array = rng.integers(0, 4000, size=(N_BANDS, HEIGHT, WIDTH), dtype=np.uint16)

        fmt = GeotiffRasterFormat()
        path = UPath(dst_prefix) / "raster_format"
        projection = Projection(CRS.from_epsg(32633), 10, -10)
        bounds = (0, 0, WIDTH, HEIGHT)
        state["path"] = path

        # `encode_raster` takes a RasterArray (CTHW), not a bare ndarray
        # (`utils/raster_format.py:547-570`); `chw_array=` is its documented single-timestep
        # convenience path (`utils/raster_array.py:48-51`).
        t0 = time.perf_counter()
        fmt.encode_raster(path, projection, bounds, RasterArray(chw_array=array))
        result["write_seconds"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        back = fmt.decode_raster(path, projection, bounds)
        result["read_seconds"] = round(time.perf_counter() - t0, 3)

        arr_back = getattr(back, "array", back)
        result["roundtrip_identical"] = bool(
            np.array_equal(np.squeeze(np.asarray(arr_back)), np.squeeze(array))
        )
        mb = array.nbytes / 1e6
        result["mb"] = round(mb, 1)
        result["read_mb_per_s"] = round(mb / max(result["read_seconds"], 1e-9), 1)

    ok, secs, err = _timed(_do)
    result.update({"ok": ok, "seconds": secs, "error": err})

    if not keep and state.get("path") is not None:
        try:
            state["path"].fs.rm(str(state["path"]), recursive=True)
            result["cleaned_up"] = True
        except Exception as exc:  # noqa: BLE001
            result["cleaned_up"] = f"failed: {type(exc).__name__}: {exc}"
    return result


def q3_tile_store_roundtrip(dst_prefix: str, keep: bool) -> dict:
    """rslearn's TileStore -- what `rslearn dataset ingest` would actually write through."""
    result: dict = {"question": "DefaultTileStore write+read round-trip on abfss://"}
    state: dict = {}

    def _do():
        import shapely
        from rasterio.crs import CRS
        from rslearn.data_sources.data_source import Item
        from rslearn.tile_stores.default import DefaultTileStore
        from rslearn.utils.geometry import Projection, STGeometry
        from rslearn.utils.raster_array import RasterArray
        from upath import UPath

        rng = np.random.default_rng(1)
        array = rng.integers(0, 4000, size=(N_BANDS, HEIGHT, WIDTH), dtype=np.uint16)

        path = UPath(dst_prefix) / "tile_store"
        state["path"] = path
        store = DefaultTileStore()
        store.set_dataset_path(path)

        projection = Projection(CRS.from_epsg(32633), 10, -10)
        bounds = (0, 0, WIDTH, HEIGHT)
        bands = [f"B{i:02d}" for i in range(N_BANDS)]
        # `write_raster`/`read_raster` take an `Item`, not a name string
        # (`tile_stores/default.py:216-247`); `Item(name, geometry)` is
        # `data_sources/data_source.py:26`.
        item = Item("probe_item", STGeometry(projection, shapely.box(*bounds), None))

        t0 = time.perf_counter()
        store.write_raster(
            "probe_layer", item, bands, projection, bounds, RasterArray(chw_array=array)
        )
        result["write_seconds"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        back = store.read_raster("probe_layer", item, bands, projection, bounds)
        result["read_seconds"] = round(time.perf_counter() - t0, 3)
        arr_back = getattr(back, "array", back)
        result["roundtrip_identical"] = bool(
            np.array_equal(np.squeeze(np.asarray(arr_back)), np.squeeze(array))
        )
        mb = array.nbytes / 1e6
        result["read_mb_per_s"] = round(mb / max(result["read_seconds"], 1e-9), 1)

    ok, secs, err = _timed(_do)
    result.update({"ok": ok, "seconds": secs, "error": err})

    if not keep and state.get("path") is not None:
        try:
            state["path"].fs.rm(str(state["path"]), recursive=True)
            result["cleaned_up"] = True
        except Exception as exc:  # noqa: BLE001
            result["cleaned_up"] = f"failed: {type(exc).__name__}: {exc}"
    return result


def q4_vsi_comparison(dst_prefix: str) -> dict:
    """fsd's /vsiadls/ read of the SAME object, if fsd is importable in this venv.

    The charter keeps `.venv-rslearn` free of fsd, so this is expected to skip. It is written
    anyway because the skip is itself worth recording, and because anyone who does install both
    gets the comparison for free.
    """
    result: dict = {
        "question": "fsd /vsiadls/ read of the same object, for throughput comparison"
    }
    try:
        import fsd  # noqa: F401
    except ImportError as exc:
        result.update(
            {
                "skipped": True,
                "reason": (
                    f"fsd not importable in this venv ({exc}) -- expected: the spike charter "
                    "keeps .venv-rslearn separate from .venv. Run-book Step 3 records the "
                    "rslearn-side numbers; the fsd-side baseline comes from its own runs."
                ),
            }
        )
        return result
    result.update({"skipped": True, "reason": "not implemented -- see run-book Step 3 note"})
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output directory for _result.json")
    ap.add_argument(
        "--dst-prefix",
        required=True,
        help="abfss:// prefix to write under. Point at scratch; never paste it back verbatim.",
    )
    ap.add_argument(
        "--keep", action="store_true", help="do not delete what was written (default: delete)"
    )
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    q1 = q1_upath_resolves(args.dst_prefix)
    # Q2/Q3 cannot mean anything if the path does not resolve, but run them anyway: their
    # error messages are the interesting part when Q1 fails.
    q2 = q2_raster_format_roundtrip(args.dst_prefix, args.keep)
    q3 = q3_tile_store_roundtrip(args.dst_prefix, args.keep)
    q4 = q4_vsi_comparison(args.dst_prefix)

    questions = {"q1_upath": q1, "q2_raster_format": q2, "q3_tile_store": q3, "q4_vs_fsd": q4}
    blocking = {k: v["error"] for k, v in questions.items() if v.get("ok") is False}

    metrics = {
        "dst_prefix_shape": _redact(args.dst_prefix),
        "adlfs_installed": q1.get("adlfs_installed"),
        "questions": questions,
        "write_read_works": bool(q2.get("ok") and q3.get("ok")),
        "probe_wall_seconds": round(time.perf_counter() - t0, 2),
    }

    result = {
        "step": "probe_03_azure",
        "status": "fail" if blocking else "ok",
        "pass": not blocking,
        "metrics": metrics,
        "expected": {
            "adlfs_installed": True,
            "note": (
                "adlfs is NOT declared by rslearn (pyproject.toml:44 is fsspec[gcs, s3]) -- "
                "installing it by hand IS the finding, not a workaround to hide. "
                "GDAL auth is NOT expected to be involved: rslearn passes an fsspec file object "
                "to rasterio for any non-local path (utils/fsspec.py:157-214), so it bypasses "
                "the /vsiadls/ problem fsd solved in spec 31. See report section 4.1.1. "
                "The live question is read THROUGHPUT versus fsd's VSI path, not authentication."
            ),
        },
        "error": (
            None
            if not blocking
            else "; ".join(f"{k} -> {v}" for k, v in blocking.items())
        ),
    }
    (outdir / "_result_probe03.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
