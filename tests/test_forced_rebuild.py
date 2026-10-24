"""#200: a forced local rebuild must actually rebuild.

`run_create_datacube(..., overwrite=True)` deletes each row's cube files
(`create_datacube._force_rebuild`) and dispatches every row again, but the local
runner drove the Snakefile with no force flag. The `done.txt` sentinel the FIRST
run left on node-local scratch survives the deletion, so Snakemake skips every
row -- the cubes are gone, no error, and a second forced build of the same rows
leaves zero cubes on disk (issue #200, seen in run-book 198: 0 of 900).

These tests drive the REAL Snakemake over a tiny synthetic catalog (same fixture
shape as `test_workflows.test_run_task_builds_one_datacube`), each with its own
`scratch_dir` so the sentinel locations are pinned and tests cannot share state:

- `test_forced_rebuild_rebuilds_after_sentinels` -- the issue's repro: build once,
  delete the cube, force a second build. Fails on pre-fix code (Snakemake finds
  the stale sentinel and does nothing).
- `test_plain_rerun_still_skips_via_sentinel` -- the opposite guard: a NON-forced
  re-run must keep skipping completed rows (the sentinel's documented within-run
  resumability must not be over-forced away).
- `test_run_local_forceall_seam` -- the runner seam itself: `--forceall` on the
  command iff `overwrite=True` (mirrors `run_local_inference`'s existing seam).
- `test_shard_forwards_overwrite` -- the AML sibling: `run_shard` must forward
  `overwrite` to `run_local`, or a forced `runner="aml"` rebuild has the same
  exposure on the node.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from rise_eo.catalog import declaration as declaration_module
from rise_eo.catalog.declaration import S2_L2A_DECLARATION
from rise_eo.storage import fs
from rise_eo.workflows import runners, shard

CRS = "EPSG:32633"
TRANSFORM = from_origin(500000, 5000000, 10, 10)
TILE_UTM = box(500000, 4999960, 500040, 5000000)  # 4x4 @ 10 m
TILE_4326 = gpd.GeoSeries([TILE_UTM], crs=CRS).to_crs("EPSG:4326").iloc[0]
TS = [pd.Timestamp("2018-06-01", tz="UTC"), pd.Timestamp("2018-06-11", tz="UTC")]


def _write_tile(path, value):
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1,
                       dtype="uint16", crs=CRS, transform=TRANSFORM, nodata=0) as dst:
        dst.write(np.full((1, 4, 4), value, dtype="uint16"))


def _make_catalog(path, tmp):
    rows = []
    for i, ts in enumerate(TS):
        rows.append({
            "id": f"T_{i}", "acquisition_key": f"T_{i}", "processing_version": None,
            "processing_datetime": None, "source": "mpc", "satellite": "sentinel-2-l2a",
            "timestamp": ts, "s3url": f"s3://eodata/x{i}", "local_folderpath": str(tmp / f"prod{i}"),
            "files": "B04.tif,B08.tif,SCL.tif", "cloud_cover": 0.0, "offset": 0,
            "nodata": 0, "geometry": TILE_4326, "area_contribution": 100.0,
        })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    declaration_module.to_attrs(gdf, S2_L2A_DECLARATION)
    fs.write_parquet(str(path), gdf)


@pytest.fixture()
def build_env(tmp_path):
    """One buildable row: synthetic tiles, catalog slice, shape, declaration, input.csv.

    The Snakefile's `done.txt`/`start.txt` sentinels land in the default scratch root,
    keyed by a sha1 of THIS test's unique `export_folderpath`, so sentinel locations are
    pinned per test with no shared state."""
    for i in range(2):
        d = tmp_path / f"prod{i}"
        d.mkdir()
        for band, val in [("B04", 100 + i), ("B08", 200 + i), ("SCL", 4)]:
            _write_tile(d / f"{band}.tif", val)
    cat = tmp_path / "catalog.parquet"
    _make_catalog(cat, tmp_path)
    shape = tmp_path / "geometry.geojson"
    gpd.GeoDataFrame({"id": ["s1"], "geometry": [TILE_4326]},
                     crs="EPSG:4326").to_file(str(shape), driver="GeoJSON")
    declaration_filepath = tmp_path / "declaration.json"
    with open(declaration_filepath, "w") as f:
        json.dump(declaration_module.to_json(S2_L2A_DECLARATION), f)
    out = tmp_path / "cube"
    csv = tmp_path / "input.csv"
    pd.DataFrame([{
        "id": "s1",
        "shapefilepath": str(shape),
        "catalog_filepath": str(cat),
        "startdate": "2018-06-01", "enddate": "2018-06-11",
        "export_folderpath": str(out),
        "datacube_filepath": str(out / "datacube.npy"),
        "mosaic_days": 20, "mosaic_scheme": "calendar",
        "declaration_filepath": str(declaration_filepath),
        "bands": "B04,B08,SCL",
    }]).to_csv(csv, index=False)
    return {"csv": str(csv), "out": str(out), "declaration": str(declaration_filepath)}


def _sentinel_paths(export_folderpath: str):
    """The Snakefile's default scratch locations for `export_folderpath`, computed the
    way the Snakefile computes them (abspath for local paths, sha1 key)."""
    key = hashlib.sha1(os.path.abspath(export_folderpath).encode()).hexdigest()
    scratch = os.path.join(tempfile.gettempdir(), "rise-eo-snakemake-scratch", key)
    return os.path.join(scratch, "done.txt"), os.path.join(scratch, "start.txt")


def _cube_files(out):
    return os.path.join(out, "datacube.npy"), os.path.join(out, "metadata.pickle.npy")


def _assert_built(out):
    dc, md = _cube_files(out)
    assert os.path.exists(dc) and os.path.getsize(dc) > 0, "datacube.npy missing/empty"
    assert os.path.exists(md) and os.path.getsize(md) > 0, "metadata.pickle.npy missing/empty"


SNAP = pytest.mark.skipif(importlib.util.find_spec("snakemake") is None,
                          reason="snakemake not installed")


@SNAP
def test_forced_rebuild_rebuilds_after_sentinels(build_env):
    """#200's repro, on the real Snakemake: a first run leaves the `done.txt`
    sentinel; the forced second build deletes the cube but must rebuild it instead of
    skipping on the stale sentinel."""
    csv, out = build_env["csv"], build_env["out"]

    first = runners.run_local(csv, cores=1)
    assert first.returncode == 0, first.stderr
    _assert_built(out)
    done_sentinel, _ = _sentinel_paths(out)
    assert os.path.exists(done_sentinel), "first run must have left the done.txt sentinel"

    # The issue: the forced rebuild deletes the artifacts but the sentinel survives.
    for p in _cube_files(out):
        os.remove(p)

    forced = runners.run_local(csv, cores=1, overwrite=True)
    assert forced.returncode == 0, forced.stderr
    _assert_built(out)  # pre-fix: Snakemake skips on the stale sentinel -> both missing


@SNAP
def test_plain_rerun_still_skips_via_sentinel(build_env):
    """The opposite guard: without `overwrite`, a re-run must STILL skip a completed
    row via its sentinel (the documented resumability) -- the fix must force only the
    forced path, not make every re-run rebuild everything."""
    csv, out = build_env["csv"], build_env["out"]

    assert runners.run_local(csv, cores=1).returncode == 0
    _assert_built(out)
    _, start_sentinel = _sentinel_paths(out)
    m_before = os.path.getmtime(start_sentinel)
    time.sleep(1.2)  # the rule's `touch(start_output)` would move the mtime

    second = runners.run_local(csv, cores=1)  # NO overwrite
    assert second.returncode == 0, second.stderr
    _assert_built(out)
    assert os.path.getmtime(start_sentinel) == m_before, \
        "a non-forced re-run re-executed the job; the sentinel skip is gone"


def test_run_local_forceall_seam(build_env, monkeypatch):
    """The runner seam: `--forceall` on the Snakemake command iff `overwrite=True` --
    the same seam `run_local_inference`/`run_local_infer_only` already carry."""
    cmds = []

    def fake_run(cmd):
        cmds.append(cmd)
        return 0, ""

    monkeypatch.setattr(runners, "_run", fake_run)
    monkeypatch.setattr(runners, "_require_snakemake", lambda: None)

    runners.run_local(build_env["csv"], cores=1)
    runners.run_local(build_env["csv"], cores=1, overwrite=True)

    assert "--forceall" not in cmds[0]
    assert "--forceall" in cmds[1]
    # a forced run must not silently change the non-force seams
    assert cmds[0][0] == cmds[1][0] and cmds[0][1] == cmds[1][1]


def test_shard_forwards_overwrite(tmp_path, monkeypatch):
    """The AML sibling (the class, not one call site): a forced `runner='aml'` rebuild
    clears the cubes on the driver, then the NODE's `run_shard` must force its own
    Snakemake -- or the same sentinel skip strands the shard. `run_shard` gets an
    `overwrite` flag and forwards it to `run_local`."""
    shard_root = tmp_path / "runs" / "r1"
    (shard_root / "shards").mkdir(parents=True)
    (shard_root / "_status").mkdir(parents=True)
    shard_csv = shard_root / "shards" / "0.csv"
    out = tmp_path / "cube"
    pd.DataFrame([{
        "id": "s1", "export_folderpath": str(out),
        "datacube_filepath": str(out / "datacube.npy"),
    }]).to_csv(shard_csv, index=False)

    seen = {}

    class _CP:
        returncode = 0
        stderr = ""

    def fake_run_local(local_csv, **kw):
        seen.update(kw)
        return _CP()

    monkeypatch.setattr(runners, "run_local", fake_run_local)

    shard.run_shard(str(shard_csv), cores=1, overwrite=True)
    assert seen.get("overwrite") is True, \
        "run_shard did not forward overwrite= to run_local (same sentinel exposure as #200)"
