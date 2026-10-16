"""Tests for spec 36 (the AML scale runner): sharding, shard.py, run_aml, D4/D7/D6a.

No test requires Azure (spec 36 §7): both halves of the AML submission surface are
substituted at the `runners.run_aml` boundary -- the client via `_FakeMLClient` (below)
and the job-builder `azure.ai.ml.command` via the `fake_aml_command` fixture -- so the
suite passes on a `pip install -e ".[dev]"` clone without the `[aml]` extra.
"""

from __future__ import annotations

import json
import os
import subprocess
import types

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from rise_eo import api
from rise_eo.datacube import builder
from rise_eo.storage import fs
from rise_eo.workflows import create_datacube, runners, shard, task

CRS = "EPSG:32633"
TRANSFORM = from_origin(500000, 5000000, 10, 10)
TILE_UTM = box(500000, 4999960, 500040, 5000000)
TILE_4326 = gpd.GeoSeries([TILE_UTM], crs=CRS).to_crs("EPSG:4326").iloc[0]
TS = [pd.Timestamp("2018-06-01", tz="UTC"), pd.Timestamp("2018-06-11", tz="UTC")]


def _write_tile(path, value):
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1,
                       dtype="uint16", crs=CRS, transform=TRANSFORM, nodata=0) as dst:
        dst.write(np.full((1, 4, 4), value, dtype=np.uint16))


def _make_catalog(path, tmp, files="B04.tif,B08.tif,SCL.tif", with_ac=True):
    from rise_eo.catalog import declaration as declaration_module
    from rise_eo.catalog.declaration import S2_L2A_DECLARATION

    rows = []
    for i, ts in enumerate(TS):
        r = {"id": f"T_{i}", "acquisition_key": f"T_{i}", "processing_version": None, "processing_datetime": None, "source": "mpc", "satellite": "sentinel-2-l2a", "timestamp": ts,
             "s3url": f"s3://eodata/x{i}", "local_folderpath": str(tmp / f"prod{i}"),
             "files": files, "cloud_cover": 0.0, "offset": 0, "nodata": 0, "geometry": TILE_4326}
        if with_ac:
            r["area_contribution"] = 100.0
        rows.append(r)
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    declaration_module.to_attrs(gdf, S2_L2A_DECLARATION)
    fs.write_parquet(str(path), gdf)


# --- test 1 / test 8: sharding is a partition, round-robin, K>N degrades ----------

def test_shard_units_is_a_partition_round_robin_and_degrades():
    units = list(range(7))
    shards = runners.shard_units(units, 3)

    assert len(shards) == 3
    flat = sorted(u for group in shards for u in group)
    assert flat == units                        # no unit lost, none duplicated
    assert shards == [[0, 3, 6], [1, 4], [2, 5]]  # round-robin assignment

    degraded = runners.shard_units(units, 100)   # K > N -> N non-empty shards
    assert len(degraded) == len(units)
    assert all(len(g) == 1 for g in degraded)
    assert all(g for g in degraded)              # every shard non-empty


def test_shard_units_partition_check_is_non_vacuous():
    """Review standard (spec 35's tests were verified this way): prove the partition
    assertion above actually catches a broken sharder by mutating it to drop a unit."""
    units = list(range(7))
    shards = runners.shard_units(units, 3)
    shards[0] = shards[0][1:]  # simulate a sharder that drops a unit

    flat = sorted(u for group in shards for u in group)
    assert flat != units


# --- test 2: shard.py resolves a remote (memory://) shard CSV, calls run_local -----

def test_run_shard_resolves_remote_csv_and_calls_run_local(tmp_path, monkeypatch):
    shard_url = "memory://runs/r1/shards/0.csv"
    rows = [{"export_folderpath": str(tmp_path / "cube1"), "shapefilepath": "g1.geojson",
             "catalog_filepath": "c1.parquet"}]
    with fs.open(shard_url, "w") as f:
        pd.DataFrame(rows).to_csv(f, index=False)

    calls = {}

    def _fake_run_local(csv_path, *, cores, **kw):
        calls["csv_path"] = csv_path
        calls["cores"] = cores
        assert os.path.exists(csv_path)  # materialized to a real local file
        assert len(pd.read_csv(csv_path)) == 1
        # a real run would leave datacube.npy behind for each unit it built
        fs.makedirs(str(tmp_path / "cube1"))
        fs.save_npy(str(tmp_path / "cube1" / "datacube.npy"), np.zeros((1,)))
        return subprocess.CompletedProcess(args=[], returncode=0)

    monkeypatch.setattr(runners, "run_local", _fake_run_local)

    status = shard.run_shard(shard_url, cores=4)

    assert calls["cores"] == 4
    assert not calls["csv_path"].startswith("memory://")  # local, not the remote URL
    assert status["n_units"] == 1
    assert status["status"] == "ok"

    with fs.open(shard._status_url(shard_url), "r") as f:
        published = json.load(f)
    assert published["shard"] == "0.csv"


def test_run_shard_puts_the_stderr_tail_in_the_failed_status(tmp_path, monkeypatch):
    shard_url = "memory://runs/r2/shards/0.csv"
    rows = [{"export_folderpath": str(tmp_path / "cube1"), "shapefilepath": "g1.geojson",
             "catalog_filepath": "c1.parquet"}]
    with fs.open(shard_url, "w") as f:
        pd.DataFrame(rows).to_csv(f, index=False)
    monkeypatch.setattr(
        runners, "run_local",
        lambda *a, **kw: subprocess.CompletedProcess(
            args=[], returncode=1, stderr="ModuleNotFoundError: No module named 'x'"),
    )

    shard.run_shard(shard_url, cores=1)

    with fs.open(shard._status_url(shard_url), "r") as f:
        error = json.load(f)["error"]
    assert "snakemake exited 1" in error
    assert "ModuleNotFoundError: No module named 'x'" in error


# --- fake AML client: the injection seam for run_aml (D3 invariant 3) -------------

class _NS(types.SimpleNamespace):
    pass


@pytest.fixture
def fake_aml_command(monkeypatch):
    """Substitute `azure.ai.ml.command` with a no-dependency job-builder (a
    `SimpleNamespace` carrying the same kwargs) so run_aml tests exercise the real
    dispatcher yet never require the `[aml]` extra ("no test may require Azure", §7).
    `job.environment_variables` etc. are exactly the kwargs run_aml passed -- which is
    the run_aml behaviour these tests pin (test 5), not azure-ai-ml's Command class."""
    def _cmd(**kwargs):
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(runners, "_import_aml_command", lambda: _cmd)
    return _cmd


class _FakeMLClient:
    """Fakes exactly the `MLClient` surface `run_aml` touches. `job_statuses` is the
    terminal status returned for the k-th submitted job, in submission order."""

    def __init__(self, job_statuses: list[str]):
        self._job_statuses = job_statuses
        self.submitted: list = []
        self.compute = _NS(get=lambda cluster: _NS(provisioning_state="Succeeded", max_instances=4))
        self.environments = _NS(get=lambda **kw: _NS())
        self.jobs = _NS(create_or_update=self._create_or_update, get=self._get)

    def _create_or_update(self, job):
        idx = len(self.submitted)
        self.submitted.append(job)
        return _NS(name=f"job-{idx}")

    def _get(self, name):
        idx = int(name.rsplit("-", 1)[1])
        return _NS(status=self._job_statuses[idx])


def _write_input_csv(url, n_units=2):
    rows = [{"id": f"u{i}", "export_folderpath": f"memory://cubes/u{i}"} for i in range(n_units)]
    with fs.open(url, "w") as f:
        pd.DataFrame(rows).to_csv(f, index=False)


# --- test 3: run_aml raises listing exactly the failed shards ---------------------

def test_run_aml_raises_listing_exactly_the_failed_shards(fake_aml_command):
    input_csv = "memory://run_aml_fail/input.csv"
    _write_input_csv(input_csv, n_units=2)
    ml_client = _FakeMLClient(["Completed", "Failed"])

    with pytest.raises(RuntimeError, match=r"\[1\]"):
        runners.run_aml(
            input_csv, cluster="c", environment="rise-eo-env:1",
            root="memory://run_aml_fail/root", identity_client_id="fake-id",
            n_shards=2, ml_client=ml_client, run_id="failrun",
        )
    assert len(ml_client.submitted) == 2


def test_run_aml_succeeds_when_all_shards_complete(fake_aml_command):
    input_csv = "memory://run_aml_ok/input.csv"
    _write_input_csv(input_csv, n_units=2)
    ml_client = _FakeMLClient(["Completed", "Completed"])

    result = runners.run_aml(
        input_csv, cluster="c", environment="rise-eo-env:1",
        root="memory://run_aml_ok/root", identity_client_id="fake-id",
        n_shards=2, ml_client=ml_client, run_id="okrun",
    )
    assert result["n_shards"] == 2
    assert len(ml_client.submitted) == 2


# --- test 4: _check_local_seams accepts "aml", still rejects unknowns -------------

def test_check_local_seams_accepts_aml_and_lists_valid_values():
    assert api._check_local_seams("local", None) == []
    assert api._check_local_seams("aml", None) == []
    errs = api._check_local_seams("batch", None)
    assert errs and "'local'" in errs[0] and "'aml'" in errs[0]


# --- test 5: D4 -- the job spec run_aml builds carries AZURE_CLIENT_ID ------------

def test_run_aml_job_carries_azure_client_id(fake_aml_command):
    """Nothing in rise_eo/ reads AZURE_CLIENT_ID -- this test is the only thing that
    explains why run_aml sets it (D4: the AML cluster's user-assigned identity is
    never selected implicitly)."""
    input_csv = "memory://run_aml_id/input.csv"
    _write_input_csv(input_csv, n_units=1)
    ml_client = _FakeMLClient(["Completed"])

    runners.run_aml(
        input_csv, cluster="c", environment="rise-eo-env:1",
        root="memory://run_aml_id/root", identity_client_id="deadbeef-guid",
        n_shards=1, ml_client=ml_client, run_id="idrun",
    )
    assert len(ml_client.submitted) == 1
    job = ml_client.submitted[0]
    assert job.environment_variables["AZURE_CLIENT_ID"] == "deadbeef-guid"


# --- test 6: D7 -- skip-if-final-exists, atomic temp-then-rename publish ----------

def test_run_task_skips_rebuild_if_final_artifact_exists(tmp_path, monkeypatch):
    out = tmp_path / "cube"
    fs.makedirs(str(out))
    fs.save_npy(str(out / "datacube.npy"), np.zeros((1, 1, 1, 1)))

    built = {}
    monkeypatch.setattr(builder, "build_datacube", lambda **kw: built.setdefault("called", True))

    task.run_task(
        str(tmp_path / "geometry.geojson"), str(tmp_path / "catalog.parquet"),
        TS[0], TS[1], str(out), bands=["B04"], mosaic_days=20,
        declaration_filepath="unused",  # never read: the early-return fires first
    )
    assert "called" not in built  # returned early, builder never invoked


def test_atomic_publish_leaves_no_temp_path_after_success(tmp_path):
    path = str(tmp_path / "x.npy")
    builder._save_npy_atomic(path, np.arange(3))

    assert fs.exists(path)
    leftovers = [p for p in os.listdir(tmp_path) if p.startswith("x.npy.tmp-")]
    assert leftovers == []


def test_atomic_publish_leaves_no_final_path_on_interrupted_write(tmp_path, monkeypatch):
    path = str(tmp_path / "y.npy")

    def _boom(p, arr, allow_pickle=False, **kw):
        raise RuntimeError("simulated interrupted write")

    monkeypatch.setattr(fs, "save_npy", _boom)
    with pytest.raises(RuntimeError, match="simulated interrupted write"):
        builder._save_npy_atomic(path, np.arange(3))

    assert not fs.exists(path)


# --- test 9: D6a -- geometry I/O round-trips through rise_eo.storage (memory://) ------

def test_geometry_io_round_trips_through_remote_storage(tmp_path):
    """setup() + run_task() read/write geometry through rise_eo.storage, not a raw path --
    proven here by putting every geometry file on `memory://` (no local checkout at
    all). The tile band files stay real local files (rasterio pixel reads are the
    documented direct-I/O exception, untouched by D6a)."""
    for i in range(2):
        d = tmp_path / f"prod{i}"
        d.mkdir()
        for band, val in [("B04", 100 + i), ("B08", 200 + i), ("SCL", 4)]:
            _write_tile(d / f"{band}.tif", val)

    cat_url = "memory://d6a/subset.parquet"
    _make_catalog(cat_url, tmp_path, files="B04.tif,B08.tif,SCL.tif", with_ac=True)

    shape_url = "memory://d6a/geometry.geojson"
    shape_gdf = gpd.GeoDataFrame({"id": ["s1"], "geometry": [TILE_4326]}, crs="EPSG:4326")
    with fs.open(shape_url, "w") as f:
        f.write(shape_gdf.to_json())

    from rise_eo.catalog import declaration as declaration_module

    declaration_url = "memory://d6a/declaration.json"
    with fs.open(declaration_url, "w") as f:
        f.write(json.dumps(declaration_module.to_json(declaration_module.S2_L2A_DECLARATION)))

    out_url = "memory://d6a/cube"
    task.run_task(
        shape_url, cat_url, TS[0], TS[1], out_url,
        bands=["B04", "B08", "SCL"], mosaic_days=20,
        declaration_filepath=declaration_url,
        if_missing_files="warn",
    )

    dc = fs.load_npy(out_url + "/datacube.npy")
    md = fs.load_npy(out_url + "/metadata.pickle.npy", allow_pickle=True)[()]
    assert dc.shape == (1, 4, 4, 2)
    assert md["bands"] == ["B04", "B08"]


def test_setup_reads_caller_geometry_from_remote_storage(tmp_path):
    """D6a site 1: setup()'s `gpd.read_file(shapefilepath)` -> fs.open + BytesIO."""
    cat = tmp_path / "catalog.parquet"
    _make_catalog(cat, tmp_path)

    shapes_url = "memory://d6a_setup/shapes.geojson"
    g1 = gpd.GeoSeries([box(500005, 4999965, 500035, 4999995)], crs=CRS).to_crs("EPSG:4326")
    gdf = gpd.GeoDataFrame({"id": ["s1"], "label": [0], "geometry": [g1.iloc[0]]}, crs="EPSG:4326")
    with fs.open(shapes_url, "w") as f:
        f.write(gdf.to_json())

    csv = tmp_path / "run" / "input.csv"
    create_datacube.setup(
        catalog_filepath=str(cat), timestamp_col="timestamp",
        shapefilepath=shapes_url, id_col="id", run_folderpath=str(tmp_path / "run"),
        startdate=datetime_start(), enddate=datetime_end(),
        bands=["B04", "B08", "SCL"], mosaic_days=20,
        csv_filepath=str(csv), label_col="label",
    )
    df = pd.read_csv(csv)
    assert len(df) == 1


def datetime_start():
    import datetime
    return datetime.datetime(2018, 1, 1)


def datetime_end():
    import datetime
    return datetime.datetime(2019, 1, 1)
