"""Tests for spec 38 (P4, inference at scale on AML): `infer_shard` (D2/D3/D9),
`adapter_smoke` (D11), `runners.run_aml_inference` (D1/D1a/D11), `raster.cog.to_cog`'s
remote-dst branch (D5), the `create_inference` Snakefile's D6/D7 fixes, the D9 date
boundary, the D8 MPC catalog-merge fix, the D13 dedupe/guard, and `api.run_inference`'s
`runner="aml"` seam (D14).

No test requires Azure (spec 38 §7, mirrors spec 36/37): `_FakeMLClient` + a fake
`azure.ai.ml.command`; a trivial `BaseModelAdapter` + `bundle.save`/`load` (no real ML
deps) for the engine/Snakefile paths; `memory://` stands in for blob.
"""

from __future__ import annotations

import json
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
from rise_eo.catalog.catalog import TileCatalog
from rise_eo.model import BaseModelAdapter, bundle, engine
from rise_eo.raster.cog import to_cog
from rise_eo.storage import fs
from rise_eo.workflows import adapter_smoke, create_datacube, infer_shard, infer_task, runners

CRS = "EPSG:32633"
TRANSFORM = from_origin(500000, 5000000, 10, 10)
TILE_UTM = box(500000, 4999960, 500040, 5000000)
TILE_4326 = gpd.GeoSeries([TILE_UTM], crs=CRS).to_crs("EPSG:4326").iloc[0]
TS = [pd.Timestamp("2018-06-01", tz="UTC"), pd.Timestamp("2018-06-11", tz="UTC")]


class _NDVIUp(BaseModelAdapter):
    """Tiny adapter: NDVI>0 -> class 1 (mirrors tests/test_workflows.py)."""

    required_bands = ["B04", "B08"]
    n_timestamps = 1
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["cls"]
    feature_sequence = []

    def load(self):
        pass

    def predict(self, X):
        return np.ones(X.shape[0], dtype="uint8")


def _write_tile(path, value):
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1,
                       dtype="uint16", crs=CRS, transform=TRANSFORM, nodata=0) as dst:
        dst.write(np.full((1, 4, 4), value, dtype=np.uint16))


def _write_local_cog(path):
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1,
                       dtype="uint16", crs=CRS, transform=TRANSFORM, nodata=0) as dst:
        dst.write(np.arange(16, dtype="uint16").reshape(1, 4, 4))


# --- fake AML client: the injection seam (D3 invariant 3, mirrors spec 36/37) ------

class _NS(types.SimpleNamespace):
    pass


@pytest.fixture
def fake_aml_command(monkeypatch):
    def _cmd(**kwargs):
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(runners, "_import_aml_command", lambda: _cmd)
    monkeypatch.setattr(
        runners, "_import_command_job_limits",
        lambda: (lambda timeout: types.SimpleNamespace(timeout=timeout)),
    )
    return _cmd


class _FakeMLClient:
    def __init__(self, job_statuses: dict[str, str]):
        self._job_statuses = job_statuses
        self.submitted: list = []
        self.compute = _NS(get=lambda cluster: _NS(provisioning_state="Succeeded", max_instances=4))
        self.environments = _NS(get=lambda **kw: _NS())
        self.jobs = _NS(create_or_update=self._create_or_update, get=self._get)

    def _create_or_update(self, job):
        idx = len(self.submitted)
        name = f"job-{idx}"
        self.submitted.append((name, job))
        return _NS(name=name)

    def _get(self, name):
        idx = int(name.rsplit("-", 1)[1])
        return _NS(status=list(self._job_statuses.values())[idx])


def _write_bundle(tmp_path, name="bundle"):
    return bundle.save(_NDVIUp(), {}, str(tmp_path / name))


def _write_input_csv(url, n_units=2, export_prefix="memory://cells"):
    rows = [{
        "id": f"u{i}", "shapefilepath": f"{export_prefix}/u{i}/geometry.geojson",
        "catalog_filepath": f"{export_prefix}/u{i}/catalog.parquet",
        "startdate": "2018-06-01", "enddate": "2018-06-11",
        "export_folderpath": f"{export_prefix}/u{i}", "mosaic_days": 20,
        "mosaic_scheme": "calendar", "declaration_filepath": "unused", "bands": "B04,B08",
    } for i in range(n_units)]
    with fs.open(url, "w") as f:
        pd.DataFrame(rows).to_csv(f, index=False)
    return rows


# --- test 1: infer_shard fs.gets CSV + bundle, calls run_local_inference -----------

def test_infer_shard_fetches_csv_and_bundle_and_calls_run_local_inference(tmp_path, monkeypatch):
    bundle_dir = _write_bundle(tmp_path)
    staged_bundle_url = "memory://run1/_bundle"
    runners._stage_bundle(bundle_dir, staged_bundle_url)

    shard_url = "memory://run1/shards/0.csv"
    _write_input_csv(shard_url, n_units=1)

    calls = {}

    def _fake_run_local_inference(csv_path, *, cores, bundle_path, cubes_per_task, **kw):
        calls["csv_path"] = csv_path
        calls["cores"] = cores
        calls["bundle_path"] = bundle_path
        calls["cubes_per_task"] = cubes_per_task
        assert not csv_path.startswith("memory://")          # materialized locally
        assert not bundle_path.startswith("memory://")        # bundle fetched locally
        assert len(pd.read_csv(csv_path)) == 1
        with open(f"{bundle_path}/bundle.json") as f:
            assert json.load(f)["adapter"]                    # manifest actually landed
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runners, "run_local_inference", _fake_run_local_inference)

    status = infer_shard.run_infer_shard(shard_url, staged_bundle_url, cores=2, cubes_per_task=3)

    assert calls["cores"] == 2 and calls["cubes_per_task"] == 3
    assert status["n_units"] == 1
    with fs.open(infer_shard._status_url(shard_url), "r") as f:
        published = json.load(f)
    assert published["shard"] == "0.csv"


def test_infer_shard_status_reports_failed_when_snakemake_exits_nonzero(tmp_path, monkeypatch):
    bundle_dir = _write_bundle(tmp_path)
    staged_bundle_url = "memory://run1b/_bundle"
    runners._stage_bundle(bundle_dir, staged_bundle_url)
    shard_url = "memory://run1b/shards/0.csv"
    _write_input_csv(shard_url, n_units=1)

    monkeypatch.setattr(runners, "run_local_inference",
                        lambda *a, **kw: subprocess.CompletedProcess(
                            args=[], returncode=1,
                            stderr="ModuleNotFoundError: No module named 'x'"))
    status = infer_shard.run_infer_shard(shard_url, staged_bundle_url, cores=1)
    assert status["status"] == "failed"
    with fs.open(infer_shard._status_url(shard_url), "r") as f:
        error = json.load(f)["error"]
    assert "snakemake exited 1" in error
    assert "ModuleNotFoundError: No module named 'x'" in error


# --- D11: adapter_smoke -- the one-node import/predict-callable check --------------

def test_adapter_smoke_passes_for_a_loadable_adapter(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    staged_bundle_url = "memory://smoke_ok/_bundle"
    runners._stage_bundle(bundle_dir, staged_bundle_url)
    status_url = "memory://smoke_ok/_status/smoke.json"

    status = adapter_smoke.run_smoke(staged_bundle_url, status_url)

    assert status["status"] == "ok" and status["error"] is None
    with fs.open(status_url, "r") as f:
        assert json.load(f)["status"] == "ok"


def test_adapter_smoke_fails_for_an_unresolvable_adapter_ref(tmp_path):
    bundle_dir = _write_bundle(tmp_path)
    # corrupt the staged manifest's adapter ref -- simulates the exact failure this
    # smoke exists to catch: a module that doesn't import in the node's Environment.
    with fs.open(f"{bundle_dir}/bundle.json", "r") as f:
        manifest = json.load(f)
    manifest["adapter"] = "no_such_module:NoSuchClass"
    with fs.open(f"{bundle_dir}/bundle.json", "w") as f:
        json.dump(manifest, f)

    staged_bundle_url = "memory://smoke_bad/_bundle"
    runners._stage_bundle(bundle_dir, staged_bundle_url)
    status = adapter_smoke.run_smoke(staged_bundle_url, "memory://smoke_bad/_status/smoke.json")

    assert status["status"] == "failed" and "no_such_module" in status["error"]


# --- test 2/3: run_aml_inference stages the bundle, shards cells, carries identity --

def test_run_aml_inference_stages_bundle_shards_cells_and_carries_identity(
    tmp_path, fake_aml_command,
):
    bundle_dir = _write_bundle(tmp_path)
    input_csv = "memory://run2/cells/input.csv"
    _write_input_csv(input_csv, n_units=5)
    ml_client = _FakeMLClient({"smoke": "Completed", "0": "Completed", "1": "Completed"})

    result = runners.run_aml_inference(
        input_csv, bundle_dir, cluster="c", environment="fsd-infer-env:1",
        root="memory://run2/root", identity_client_id="deadbeef-guid", n_shards=2,
        ml_client=ml_client, run_id="infrun",
    )

    assert result["n_shards"] == 2                                  # K=5 cells -> 2 shards
    assert fs.exists("memory://run2/root/runs/infrun/_bundle/bundle.json")   # D3 staged
    names, jobs = zip(*ml_client.submitted)
    assert len(jobs) == 3                                            # smoke + 2 shards
    for job in jobs:
        assert job.environment_variables["AZURE_CLIENT_ID"] == "deadbeef-guid"   # D4'
        assert "memory://run2/root/runs/infrun/_bundle" in job.command  # references staged bundle
        assert "deadbeef" not in job.command                          # no secret in the command
    assert "adapter_smoke" in jobs[0].command                        # smoke runs first


def test_run_aml_inference_shard_count_is_non_vacuous_across_cell_counts(tmp_path, fake_aml_command):
    """Non-vacuousness (project standard): shard count actually tracks cell count,
    not a constant -- K > N degrades to N shards (reuses shard_units, proven a
    partition in tests/test_scale_runner.py)."""
    bundle_dir = _write_bundle(tmp_path)
    ml_client = _FakeMLClient({"smoke": "Completed", "0": "Completed"})
    input_csv = "memory://run2c/cells/input.csv"
    _write_input_csv(input_csv, n_units=1)

    result = runners.run_aml_inference(
        input_csv, bundle_dir, cluster="c", environment="fsd-infer-env:1",
        root="memory://run2c/root", identity_client_id="x", n_shards=4,
        ml_client=ml_client, run_id="degraderun", skip_smoke=True,
    )
    assert result["n_shards"] == 1   # 1 cell, n_shards=4 -> degrades to 1 non-empty shard


def test_run_aml_inference_skip_smoke_submits_no_smoke_job(tmp_path, fake_aml_command):
    bundle_dir = _write_bundle(tmp_path)
    ml_client = _FakeMLClient({"0": "Completed"})
    input_csv = "memory://run2d/cells/input.csv"
    _write_input_csv(input_csv, n_units=1)

    runners.run_aml_inference(
        input_csv, bundle_dir, cluster="c", environment="fsd-infer-env:1",
        root="memory://run2d/root", identity_client_id="x", n_shards=1,
        ml_client=ml_client, run_id="noSmokeRun", skip_smoke=True,
    )
    assert len(ml_client.submitted) == 1   # just the one shard, no smoke job


# --- spec 47 D5/D6: the four silent legs now print --------------------------------

def test_run_aml_inference_prints_stage_and_aml_progress(tmp_path, fake_aml_command, capsys):
    """D5/D6: the stage-bundle upload and the AML poll loop must each print at least one
    progress line; the run_id/run_root line comes before any job is submitted; the
    single-job smoke wait omits an ETA (D5) while the 2-shard fan-out shows one."""
    bundle_dir = _write_bundle(tmp_path)
    input_csv = "memory://run5/cells/input.csv"
    _write_input_csv(input_csv, n_units=5)
    ml_client = _FakeMLClient({"smoke": "Completed", "0": "Completed", "1": "Completed"})

    runners.run_aml_inference(
        input_csv, bundle_dir, cluster="c", environment="fsd-infer-env:1",
        root="memory://run5/root", identity_client_id="x", n_shards=2,
        ml_client=ml_client, run_id="progrun",
    )
    out = capsys.readouterr().out
    lines = out.splitlines()

    # [stage]: destination + size announced before upload, then a completed tick
    assert any(ln.startswith("[stage] bundle -> ") and "files," in ln for ln in lines)
    stage_ticks = [ln for ln in lines if ln.startswith("[stage] ") and "files (" in ln]
    assert any("100%" in ln for ln in stage_ticks)

    # [aml]: run_id/run_root named before any job is submitted (appears at least twice --
    # once for the smoke sub-run, once for the 2-shard main run).
    id_lines = [ln for ln in lines if ln.startswith("[aml] run_id=")]
    assert any("run_id=progrun-smoke" in ln and "run_root=" in ln for ln in id_lines)
    assert any("run_id=progrun " in ln and "run_root=" in ln for ln in id_lines)

    # single-job smoke wait: no rate, no eta
    smoke_ticks = [ln for ln in lines if ln.startswith("[aml] 0/1 jobs terminal")
                  or ln.startswith("[aml] 1/1 jobs terminal")]
    assert smoke_ticks
    for ln in smoke_ticks:
        assert "/s" not in ln
        assert "eta" not in ln

    # 2-shard main wait: eta present, no rate, "running" suffix present
    fanout_ticks = [ln for ln in lines if ln.startswith("[aml] 0/2 jobs terminal")
                    or ln.startswith("[aml] 2/2 jobs terminal")]
    assert fanout_ticks
    for ln in fanout_ticks:
        assert "/s" not in ln
        assert "eta" in ln
        assert "running" in ln


# --- test 4: run_aml_inference raises on Failed / status!="ok" ---------------------

def test_run_aml_inference_raises_on_failed_job(tmp_path, fake_aml_command):
    bundle_dir = _write_bundle(tmp_path)
    input_csv = "memory://run4/cells/input.csv"
    _write_input_csv(input_csv, n_units=2)
    ml_client = _FakeMLClient({"smoke": "Completed", "0": "Completed", "1": "Failed"})

    with pytest.raises(RuntimeError, match=r"\[1\]"):
        runners.run_aml_inference(
            input_csv, bundle_dir, cluster="c", environment="fsd-infer-env:1",
            root="memory://run4/root", identity_client_id="x", n_shards=2,
            ml_client=ml_client, run_id="failrun",
        )


def test_run_aml_inference_raises_when_status_not_ok_even_if_aml_completed(
    tmp_path, fake_aml_command,
):
    bundle_dir = _write_bundle(tmp_path)
    input_csv = "memory://run4b/cells/input.csv"
    _write_input_csv(input_csv, n_units=1)
    ml_client = _FakeMLClient({"smoke": "Completed", "0": "Completed"})

    with fs.open("memory://run4b/root/runs/badrun/_status/0.json", "w") as f:
        json.dump({"unit": 0, "status": "failed", "error": "boom"}, f)

    with pytest.raises(RuntimeError, match=r"\[0\]"):
        runners.run_aml_inference(
            input_csv, bundle_dir, cluster="c", environment="fsd-infer-env:1",
            root="memory://run4b/root", identity_client_id="x", n_shards=1,
            ml_client=ml_client, run_id="badrun",
        )


# --- test 5: api.run_inference accepts/rejects runner="aml" -----------------------

def test_run_inference_rejects_unknown_runner():
    with pytest.raises(api.PreflightError, match="not supported"):
        api.run_inference(
            _NDVIUp(), roi="x.geojson", output_folderpath="out",
            catalog_filepath="c.parquet", startdate="2018-01-01", enddate="2018-02-01",
            mosaic_days=20, bands=["B04", "B08"], runner="batch",
        )


def test_run_inference_rejects_storage_azure_for_prebuilt_cubes_even_with_aml(tmp_path):
    # a real, empty folder -- so preflight aggregation reaches _check_local_seams'
    # rejection rather than crashing on a missing input.csv (a separate, unrelated
    # early-read quirk of the pre-built-cubes path).
    with pytest.raises(api.PreflightError, match="not supported"):
        api.run_inference(
            _NDVIUp(), inference_datacubes=str(tmp_path / "no_cubes_here"),
            output_folderpath=str(tmp_path / "out"), runner="aml", storage="azure",
        )


def test_run_inference_roi_mode_threads_runner_kwargs_to_run_aml_inference(tmp_path, monkeypatch):
    bundle_dir = _write_bundle(tmp_path)
    calls = {}

    def _fake_run_aml_inference(input_csv, bundle_path, **kw):
        calls["input_csv"] = input_csv
        calls["kw"] = kw
        # simulate the AML dispatch having produced each cell's output.tif on blob,
        # so step 4 (collect) has something to find -- this test is about the
        # dispatch swap, not a real inference run.
        with fs.open(input_csv, "r") as f:
            for exp in pd.read_csv(f)["export_folderpath"]:
                _write_local_cog(f"{exp}/output.tif")
        return {"run_id": "r"}

    monkeypatch.setattr(runners, "run_aml_inference", _fake_run_aml_inference)
    # avoid a real ROI tiling (needs the [grid] extra) -- feed setup() an ALREADY-tiled
    # cells csv the same way `_run_inference_roi` would produce it, by monkeypatching
    # the grid step to a single already-known cell.
    monkeypatch.setattr(
        "rise_eo.grid.roi_to_s2_grids",
        lambda roi, **kw: gpd.GeoDataFrame({"id": ["s1"], "geometry": [TILE_4326]}, crs="EPSG:4326"),
    )

    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    from rise_eo.catalog import declaration as declaration_module
    from rise_eo.catalog.declaration import S2_L2A_DECLARATION
    gdf = gpd.GeoDataFrame(
        [{"id": "T_0", "acquisition_key": "T_0", "processing_version": None, "processing_datetime": None, "source": "mpc", "collection": "sentinel-2-l2a", "timestamp": TS[0], "s3url": "s3://x",
          "local_folderpath": str(tmp_path), "files": "B04.tif,B08.tif", "cloud_cover": 0.0,
          "geometry": TILE_4326, "area_contribution": 100.0}],
        crs="EPSG:4326",
    )
    declaration_module.to_attrs(gdf, S2_L2A_DECLARATION)
    fs.write_parquet(str(cat), gdf)

    roi_gdf = gpd.GeoDataFrame({"geometry": [TILE_4326]}, crs="EPSG:4326")
    api.run_inference(
        bundle_dir, roi=roi_gdf, output_folderpath=str(tmp_path / "out"),
        catalog_filepath=str(cat), startdate="2018-06-01", enddate="2018-06-11",
        mosaic_days=20, bands=["B04", "B08"], runner="aml",
        runner_kwargs={"cluster": "c", "environment": "e:1", "root": "memory://r",
                       "identity_client_id": "id"},
        storage="azure",
    )
    assert calls["kw"]["cluster"] == "c"
    assert calls["kw"]["identity_client_id"] == "id"


def test_run_inference_roi_stages_grids_geojson_via_seam_on_non_gdal_dst(tmp_path, monkeypatch):
    """Regression: a blob/seam-only `output_folderpath` made `grids.to_file(grids_filepath)` fail
    (GDAL/pyogrio has no abfss:// write driver -- "Failed to create GeoJSON datasource"). The
    gridded ROI must be staged through `rise_eo.storage`. `memory://` stands in for a GDAL-unwritable,
    seam-writable dst (same trick as the D5 to_cog tests). Non-vacuous: on the pre-fix `grids.to_file`
    this asserts False (grids.geojson never lands on the memory fs)."""
    bundle_dir = _write_bundle(tmp_path)

    def _fake_run_aml_inference(input_csv, bundle_path, **kw):
        # collect (step 4) only fs.exists()-checks output.tif; a 1-byte seam write is enough.
        with fs.open(input_csv, "r") as f:
            for exp in pd.read_csv(f)["export_folderpath"]:
                with fs.open(f"{exp}/output.tif", "w") as g:
                    g.write("x")
        return {"run_id": "r"}

    monkeypatch.setattr(runners, "run_aml_inference", _fake_run_aml_inference)
    monkeypatch.setattr(
        "rise_eo.grid.roi_to_s2_grids",
        lambda roi, **kw: gpd.GeoDataFrame({"id": ["s1"], "geometry": [TILE_4326]}, crs="EPSG:4326"),
    )
    # short-circuit the STAC/merge tail -- rasterio on a memory:// COG is out of scope for THIS test.
    monkeypatch.setattr(api, "_finalize_outputs", lambda outs, folder, *a, **k: api.InferenceResult(
        output_folderpath=folder, output_filepaths=list(outs), stac_catalog_filepath="memory://x"))

    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    from rise_eo.catalog import declaration as declaration_module
    from rise_eo.catalog.declaration import S2_L2A_DECLARATION
    gdf = gpd.GeoDataFrame(
        [{"id": "T_0", "acquisition_key": "T_0", "processing_version": None, "processing_datetime": None, "source": "mpc", "collection": "sentinel-2-l2a", "timestamp": TS[0], "s3url": "s3://x",
          "local_folderpath": str(tmp_path), "files": "B04.tif,B08.tif", "cloud_cover": 0.0,
          "geometry": TILE_4326, "area_contribution": 100.0}],
        crs="EPSG:4326",
    )
    declaration_module.to_attrs(gdf, S2_L2A_DECLARATION)
    fs.write_parquet(str(cat), gdf)

    out = "memory://roiout"
    api.run_inference(
        bundle_dir, roi=gpd.GeoDataFrame({"geometry": [TILE_4326]}, crs="EPSG:4326"),
        output_folderpath=out, catalog_filepath=str(cat),
        startdate="2018-06-01", enddate="2018-06-11", mosaic_days=20,
        bands=["B04", "B08"], runner="aml",
        runner_kwargs={"cluster": "c", "environment": "e:1", "root": "memory://r",
                       "identity_client_id": "id"},
        storage="azure",
    )
    import io
    grids_url = f"{out}/grids.geojson"
    assert fs.exists(grids_url), "grids.geojson was not staged through the storage seam"
    with fs.open(grids_url, "rb") as f:
        back = gpd.read_file(io.BytesIO(f.read()))
    assert list(back["id"]) == ["s1"]


# --- test 6: D5 -- to_cog remote-dst; engine + api._merge_outputs unchanged callers -

def test_to_cog_remote_dst_writes_a_valid_cog_via_transfer(tmp_path):
    src = str(tmp_path / "raw.tif")
    _write_local_cog(src)
    dst = "memory://cogtest1/out.tif"

    nbytes = to_cog(src, dst)

    assert nbytes > 0 and fs.exists(dst)
    assert not fs.exists(f"{dst}.part")   # no leftover temp


def test_to_cog_local_dst_is_unchanged(tmp_path):
    src = str(tmp_path / "raw.tif")
    _write_local_cog(src)
    dst = str(tmp_path / "out.tif")

    nbytes = to_cog(src, dst, verify=True)   # verify=True: bit-identical, would raise otherwise

    assert nbytes == __import__("os").path.getsize(dst)


def test_engine_write_output_cog_lands_on_memory_dst_via_to_cog(tmp_path, monkeypatch):
    from rise_eo.model.adapter import Output

    # Run in a clean, empty cwd so the non-vacuous assertion below can see any junk a
    # local-path-op-on-a-remote-URL would scatter.
    monkeypatch.chdir(tmp_path)
    out = Output(array=np.ones((1, 4, 4), dtype="uint8"), dtype="uint8", nodata=255,
                band_names=["cls"])
    dst = "memory://cogtest2/output.tif"

    engine._write_output_cog(out, TRANSFORM, CRS, dst)

    assert fs.exists(dst)
    # Non-vacuous (D5): a REMOTE dst must not trigger `os.makedirs`/`rasterio.open(mode="w")`
    # on the remote-looking URL -- the raw scratch tif is node-local, published via the seam.
    # Before the fix, `os.makedirs(os.path.dirname(dst))` created a `./memory:/cogtest2` junk
    # tree in cwd; assert cwd stays empty.
    assert list(tmp_path.iterdir()) == []


def test_merge_outputs_lands_on_memory_dst_via_to_cog(tmp_path):
    f0, f1 = str(tmp_path / "a.tif"), str(tmp_path / "b.tif")
    _write_local_cog(f0)
    _write_local_cog(f1)
    dst = "memory://cogtest3/merged.tif"

    result = api._merge_outputs([f0, f1], dst, nodata=0)

    assert result == dst
    assert fs.exists(dst)


def test_unit_identity_cols_do_not_drift_between_modules():
    # D13: the content-identity tuple is duplicated in `create_datacube` (the dedupe) and
    # `runners` (the dispatch guard) to avoid a circular import -- a silent divergence would
    # let a manifest dedupe on one key but be guarded on another. Pin them equal.
    assert create_datacube._UNIT_IDENTITY_COLS == runners._UNIT_IDENTITY_COLS


# --- test 7: D6 -- skip-if-output-exists; a remote export_folderpath dry-run plans -

def test_run_infer_task_returns_early_when_output_exists_and_not_overwrite(tmp_path, monkeypatch):
    output_fp = str(tmp_path / "output.tif")
    with open(output_fp, "w") as f:
        f.write("already there")

    called = {}
    monkeypatch.setattr("rise_eo.workflows.task.run_task", lambda **kw: called.setdefault("built", True))

    result = infer_task.run_infer_task(
        "g.geojson", "c.parquet", TS[0], TS[1], str(tmp_path),
        bands=["B04", "B08"], mosaic_days=20, declaration_filepath="unused",
        bundle_path="unused", output_filepath=output_fp,
    )
    assert result == output_fp
    assert "built" not in called   # no build, no infer -- returned on the first line


def test_run_infer_task_rebuilds_when_overwrite_true(tmp_path, monkeypatch):
    output_fp = str(tmp_path / "output.tif")
    with open(output_fp, "w") as f:
        f.write("stale")

    called = {}
    monkeypatch.setattr("rise_eo.workflows.task.run_task", lambda **kw: called.setdefault("built", True))
    monkeypatch.setattr(engine, "_adapter_from_bundle_cached", lambda p: _NDVIUp())
    monkeypatch.setattr(engine, "infer_datacube_to_cog", lambda *a, **kw: output_fp)

    infer_task.run_infer_task(
        "g.geojson", "c.parquet", TS[0], TS[1], str(tmp_path),
        bands=["B04", "B08"], mosaic_days=20, declaration_filepath="unused",
        bundle_path="unused", output_filepath=output_fp, overwrite=True,
    )
    assert called.get("built") is True


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("snakemake") is None, reason="snakemake not installed",
)
def test_create_inference_snakefile_plans_a_remote_export_folderpath(tmp_path):
    """D6: the grouped Snakefile never touches export_folderpath (it lives inside
    infer_task, resolved from the CSV), so a remote/abfss row needs no is_local guard
    to plan cleanly -- dry-run proves it (mirrors
    test_snakefile_plans_a_remote_export_folderpath for the datacube Snakefile)."""
    csv = tmp_path / "input.csv"
    pd.DataFrame([{
        "shapefilepath": str(tmp_path / "geometry.geojson"),
        "catalog_filepath": str(tmp_path / "catalog.parquet"),
        "startdate": "2018-01-01", "enddate": "2019-01-01",
        "export_folderpath": "abfss://data@acct.dfs.core.windows.net/p1-demo/run/x/s1",
        "mosaic_days": 20, "mosaic_scheme": "calendar", "declaration_filepath": "unused",
        "bands": "B04,B08",
    }]).to_csv(csv, index=False)

    result = runners.run_local_inference(
        str(csv), cores=1, bundle_path=str(tmp_path / "bundle"), dry_run=True)
    assert result.returncode == 0


# --- test 8: D7 -- a K-cell group loads the bundle once; cubes_per_task forwarded --

def test_run_infer_group_loads_bundle_once_for_the_whole_group(tmp_path, monkeypatch):
    csv = tmp_path / "input.csv"
    rows = []
    for i in range(3):
        exp = tmp_path / f"u{i}"
        exp.mkdir()
        with open(exp / "output.tif", "w") as f:
            f.write("prebuilt")   # D6 skip fires -> run_task/adapter never touched
        rows.append({
            "shapefilepath": "g.geojson", "catalog_filepath": "c.parquet",
            "startdate": "2018-06-01", "enddate": "2018-06-11", "export_folderpath": str(exp),
            "mosaic_days": 20, "mosaic_scheme": "calendar", "declaration_filepath": "unused",
            "bands": "B04,B08",
        })
    pd.DataFrame(rows).to_csv(csv, index=False)

    load_calls = []
    real_cached = engine._adapter_from_bundle_cached

    def _counting_cached(bundle_path):
        load_calls.append(bundle_path)
        return real_cached(bundle_path)

    monkeypatch.setattr(engine, "_adapter_from_bundle_cached", _counting_cached)
    bundle_dir = _write_bundle(tmp_path)

    written = infer_task.run_infer_group(str(csv), (0, 3), bundle_dir)

    assert len(written) == 3
    # D6's per-cell skip means the cache is never even consulted here (every output
    # pre-exists) -- the amortisation claim is about the NON-skip path, so assert the
    # opposite explicitly: skip-only groups touch it zero times, proving the skip (not
    # the cache) fired first.
    assert load_calls == []


def test_run_infer_group_shares_one_cached_adapter_across_a_group(tmp_path, monkeypatch):
    """The actual D7 amortisation: when cells DO need building, the cache is
    consulted once per cell but returns the SAME object each time within a group
    (one process, module-level cache) -- not a fresh bundle.load per cell."""
    bundle_dir = _write_bundle(tmp_path)
    csv = tmp_path / "input.csv"
    rows = []
    for i in range(2):
        (tmp_path / f"u{i}").mkdir()
        rows.append({
            "shapefilepath": "g.geojson", "catalog_filepath": "c.parquet",
            "startdate": "2018-06-01", "enddate": "2018-06-11",
            "export_folderpath": str(tmp_path / f"u{i}"),
            "mosaic_days": 20, "mosaic_scheme": "calendar", "declaration_filepath": "unused",
            "bands": "B04,B08",
        })
    pd.DataFrame(rows).to_csv(csv, index=False)

    monkeypatch.setattr("rise_eo.workflows.task.run_task", lambda **kw: None)
    monkeypatch.setattr(engine, "infer_datacube_to_cog",
                        lambda adapter, dc, out, **kw: __import__("pathlib").Path(out).write_text("x") or out)

    load_calls = []
    real_load = bundle.load

    def _counting_load(path, **kw):
        load_calls.append(path)
        return real_load(path, **kw)

    monkeypatch.setattr(bundle, "load", _counting_load)
    engine._BUNDLE_CACHE.clear()

    infer_task.run_infer_group(str(csv), (0, 2), bundle_dir)

    assert len(load_calls) == 1   # ONE bundle.load for both cells in the group


def test_run_local_inference_forwards_cubes_per_task(tmp_path):
    """Regression (D7, TODO #25's root cause): a mutation dropping cubes_per_task from
    the Snakemake config would fail this."""
    csv = tmp_path / "input.csv"
    pd.DataFrame([{
        "shapefilepath": "g.geojson", "catalog_filepath": "c.parquet",
        "startdate": "2018-01-01", "enddate": "2018-02-01",
        "export_folderpath": str(tmp_path / "u0"), "mosaic_days": 20,
        "mosaic_scheme": "calendar", "declaration_filepath": "unused", "bands": "B04,B08",
    }]).to_csv(csv, index=False)

    result = runners.run_local_inference(
        str(csv), cores=1, bundle_path=str(tmp_path / "bundle"),
        cubes_per_task=7, dry_run=True,
    )
    assert result.returncode == 0   # config accepted (the Snakefile reads cubes_per_task)


def test_infer_shard_default_is_load_per_core_computed_from_node_cores(monkeypatch):
    """D7 default: with cores/cubes_per_task unset, the NODE picks cores=os.cpu_count()
    and groups cells so the bundle loads once PER CORE (not once per cell -- TODO #25),
    computed from the shard size. Non-vacuous: pin cpu_count and assert the arithmetic."""
    monkeypatch.setattr(infer_shard.os, "cpu_count", lambda: 4)
    # 10 cells, 4 cores -> 4 groups of ceil(10/4)=3 -> 4 bundle loads/node (== cores), not 10.
    assert infer_shard._resolve_cores_and_group(10, None, None) == (4, 3)
    # load-once-per-node opt-out: cores=1 -> one whole-shard group -> one bundle load.
    assert infer_shard._resolve_cores_and_group(10, 1, None) == (1, 10)
    # explicit values override either knob.
    assert infer_shard._resolve_cores_and_group(10, 2, 5) == (2, 5)
    # a mutation reverting the default to a constant 1 (the old per-cell pathology) would
    # make this (4, 3) assertion fail -- 1 cell per group means 10 loads, not 4.
    assert infer_shard._resolve_cores_and_group(10, None, None) != (4, 1)


def test_run_aml_inference_omits_group_flags_by_default_so_node_computes_them(
    tmp_path, fake_aml_command,
):
    """D7: run_aml_inference leaves --cores/--cubes-per-task OFF the node command when the
    caller didn't set them, so the node computes the load-per-core default; explicit values
    are threaded through. Non-vacuous across both the default and the override."""
    bundle_dir = _write_bundle(tmp_path)
    input_csv = "memory://run_d7flags/cells/input.csv"
    _write_input_csv(input_csv, n_units=3)

    ml_client = _FakeMLClient({"0": "Completed"})
    runners.run_aml_inference(
        input_csv, bundle_dir, cluster="c", environment="e:1",
        root="memory://run_d7flags/root", identity_client_id="x", n_shards=1,
        ml_client=ml_client, run_id="d7default", skip_smoke=True,
    )
    default_cmd = ml_client.submitted[0][1].command
    assert "--cores" not in default_cmd and "--cubes-per-task" not in default_cmd

    ml_client2 = _FakeMLClient({"0": "Completed"})
    runners.run_aml_inference(
        input_csv, bundle_dir, cluster="c", environment="e:1",
        root="memory://run_d7flags/root2", identity_client_id="x", n_shards=1,
        ml_client=ml_client2, run_id="d7explicit", skip_smoke=True, cores=1, cubes_per_task=8,
    )
    explicit_cmd = ml_client2.submitted[0][1].command
    assert "--cores 1" in explicit_cmd and "--cubes-per-task 8" in explicit_cmd


# --- test 9: D9 -- date-window normalization, fail-fast on the driver -------------

def test_download_coerces_string_and_timestamp_startdate_to_the_same_timestamp(monkeypatch, tmp_path):
    calls = []

    # `**_rest` on purpose: this test pins how `startdate` is COERCED, not the full kwarg
    # set `api.download` forwards. Spelling every parameter out made it fail the moment
    # `max_concurrent` was added -- a false negative about date handling.
    def _fake_mpc_download(*, roi, startdate, enddate, bands, root_folderpath, catalog,
                           max_tiles, max_cloudcover=None, progress=True, collection=None,
                           **_rest):
        calls.append((startdate, enddate))

    monkeypatch.setattr(api, "_mpc_download", _fake_mpc_download)

    for start in ("2018-01-01", pd.Timestamp("2018-01-01")):
        api.download(
            "roi.geojson", start, "2018-02-01", ["B04"], str(tmp_path / "dl"),
            source="mpc", max_tiles=10,
        )

    assert calls[0] == calls[1]                       # same forwarded value regardless of type
    assert isinstance(calls[0][0], pd.Timestamp)       # and it's the natural type, not a string


def test_download_invalid_startdate_raises_preflight_before_any_download_call(monkeypatch, tmp_path):
    called = {}
    monkeypatch.setattr(api, "_mpc_download", lambda **kw: called.setdefault("hit", True))

    with pytest.raises(api.PreflightError, match="not a valid date"):
        api.download(
            "roi.geojson", "2018-13-01", "2018-02-01", ["B04"], str(tmp_path / "dl"),
            source="mpc", max_tiles=10,
        )
    assert "hit" not in called   # never reached the source call -- failed on the driver


def test_run_inference_roi_invalid_startdate_raises_preflight():
    with pytest.raises(api.PreflightError, match="not a valid date"):
        api.run_inference(
            _NDVIUp(), roi="x.geojson", output_folderpath="out",
            catalog_filepath="c.parquet", startdate="2018-13-01", enddate="2018-02-01",
            mosaic_days=20, bands=["B04", "B08"],
        )


# --- test 10: D8 -- per-shard MPC catalog files + driver sequential merge ---------

def test_run_aml_download_mpc_merges_per_shard_catalogs_into_the_canonical(
    fake_aml_command, monkeypatch,
):
    from rise_eo.catalog import declaration as declaration_module
    from rise_eo.catalog.declaration import S2_L2A_DECLARATION

    rows = [
        {"tile_id": "T0", "acquisition_key": "T0", "band": "B04", "href": "h0", "dst": "d0", "offset": 0,
         "collection": "sentinel-2-l2a", "timestamp": TS[0].isoformat(), "s3url": "s3://x0",
         "cloud_cover": 0.0, "nodata": 0, "geometry": TILE_4326.wkt},
        {"tile_id": "T1", "acquisition_key": "T1", "band": "B04", "href": "h1", "dst": "d1", "offset": 0,
         "collection": "sentinel-2-l2a", "timestamp": TS[1].isoformat(), "s3url": "s3://x1",
         "cloud_cover": 0.0, "nodata": 0, "geometry": TILE_4326.wkt},
    ]
    monkeypatch.setattr(runners._mpc, "discover_shard_rows", lambda *a, **kw: rows)
    ml_client = _FakeMLClient({"0": "Completed", "1": "Completed"})

    root = "memory://aml_dl_d8/root"
    run_id = "d8run"
    canonical = "memory://aml_dl_d8/data/catalog.parquet"

    # Simulate each node writing its OWN shard catalog (what workflows.download.run_shard
    # does when the dispatcher hands it a per-shard --catalog url, D8).
    def _shard_catalog(k, tile_id, ts):
        gdf = gpd.GeoDataFrame([{
            "id": tile_id, "collection": "sentinel-2-l2a", "timestamp": ts, "s3url": f"s3://{tile_id}",
            "local_folderpath": "/x", "files": "B04.tif", "cloud_cover": 0.0, "offset": 0,
            "nodata": 0, "acquisition_key": tile_id, "processing_version": None,
            "processing_datetime": None, "source": "mpc", "geometry": TILE_4326,
        }], crs="EPSG:4326")
        declaration_module.to_attrs(gdf, S2_L2A_DECLARATION)
        fs.write_parquet(f"{root}/runs/{run_id}/shards/catalog-{k}.parquet", gdf)
        with fs.open(f"{root}/runs/{run_id}/_status/{k}.json", "w") as f:
            json.dump({"unit": k, "status": "ok"}, f)

    _shard_catalog(0, "T0", TS[0])
    _shard_catalog(1, "T1", TS[1])

    runners.run_aml_download(
        "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
        "memory://aml_dl_d8/data", canonical,
        source="mpc", cluster="c", environment="fsd-env:1", root=root,
        identity_client_id="x", max_tiles=10, ml_client=ml_client, run_id=run_id,
    )

    # both shards' catalogs went to DISTINCT files (proves the shared-write race is gone)
    submitted_commands = [j.command for _, j in ml_client.submitted]
    assert any("catalog-0.parquet" in c for c in submitted_commands)
    assert any("catalog-1.parquet" in c for c in submitted_commands)
    assert not any(canonical in c for c in submitted_commands)   # never the shared path

    merged = TileCatalog(canonical).read()
    assert set(merged["id"]) == {"T0", "T1"}   # BOTH tiles' rows landed (the lost update didn't fire)


def test_merge_shard_catalogs_is_non_vacuous(tmp_path):
    """A mutation reverting to a shared writer would leave only ONE shard catalog to
    merge -- prove the merge helper actually needs both files by dropping one."""
    from rise_eo.catalog import declaration as declaration_module
    from rise_eo.catalog.declaration import S2_L2A_DECLARATION

    def _make(url, tile_id):
        gdf = gpd.GeoDataFrame([{
            "id": tile_id, "collection": "sentinel-2-l2a", "timestamp": TS[0], "s3url": "s3://x",
            "local_folderpath": "/x", "files": "B04.tif", "cloud_cover": 0.0, "offset": 0,
            "nodata": 0, "acquisition_key": tile_id, "processing_version": None,
            "processing_datetime": None, "source": "mpc", "geometry": TILE_4326,
        }], crs="EPSG:4326")
        declaration_module.to_attrs(gdf, S2_L2A_DECLARATION)
        fs.write_parquet(url, gdf)

    _make("memory://merge_nv/shards/catalog-0.parquet", "T0")
    _make("memory://merge_nv/shards/catalog-1.parquet", "T1")
    canonical = "memory://merge_nv/catalog.parquet"

    runners._merge_shard_catalogs(
        {0: "memory://merge_nv/shards/catalog-0.parquet",
         1: "memory://merge_nv/shards/catalog-1.parquet"}, canonical,
    )
    assert set(TileCatalog(canonical).read()["id"]) == {"T0", "T1"}

    # now the mutated (single-writer) shape: only ONE shard file exists
    canonical2 = "memory://merge_nv/catalog2.parquet"
    runners._merge_shard_catalogs({0: "memory://merge_nv/shards/catalog-0.parquet"}, canonical2)
    assert set(TileCatalog(canonical2).read()["id"]) == {"T0"}   # T1 would be silently missing


# --- test 11: D13 -- setup dedupe on identity + dispatch-time duplicate guard -----

def _two_shapes(path):
    g1 = gpd.GeoSeries([box(500005, 4999965, 500035, 4999995)], crs=CRS).to_crs("EPSG:4326")
    g2 = gpd.GeoSeries([box(500010, 4999970, 500030, 4999990)], crs=CRS).to_crs("EPSG:4326")
    gdf = gpd.GeoDataFrame({"id": ["s1", "s2"], "geometry": [g1.iloc[0], g2.iloc[0]]}, crs="EPSG:4326")
    gdf.to_file(str(path), driver="GeoJSON")


def _make_catalog(path, tmp):
    from rise_eo.catalog import declaration as declaration_module
    from rise_eo.catalog.declaration import S2_L2A_DECLARATION

    rows = []
    for i, ts in enumerate(TS):
        rows.append({"id": f"T_{i}", "acquisition_key": f"T_{i}", "processing_version": None,
                     "processing_datetime": None, "source": "mpc",
                     "collection": "sentinel-2-l2a", "timestamp": ts,
                     "s3url": f"s3://eodata/x{i}", "local_folderpath": str(tmp / f"prod{i}"),
                     "files": "B04.tif,B08.tif,SCL.tif", "cloud_cover": 0.0, "geometry": TILE_4326,
                     "area_contribution": 100.0})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    declaration_module.to_attrs(gdf, S2_L2A_DECLARATION)
    fs.write_parquet(str(path), gdf)


def test_setup_called_twice_dedupes_to_one_row_per_unit_order_preserved(tmp_path):
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _two_shapes(shapes)
    csv = tmp_path / "run" / "input.csv"

    kwargs = dict(
        catalog_filepath=str(cat), timestamp_col="timestamp", shapefilepath=str(shapes),
        id_col="id", run_folderpath=str(tmp_path / "run"),
        startdate=pd.Timestamp("2018-01-01"), enddate=pd.Timestamp("2019-01-01"),
        bands=["B04", "B08", "SCL"], mosaic_days=20,
        csv_filepath=str(csv), label_col=None,
    )
    create_datacube.setup(**kwargs)
    create_datacube.setup(**kwargs)   # re-run with IDENTICAL params -- idempotent re-run

    df = pd.read_csv(csv)
    assert len(df) == 2                          # one row per unit, not four
    assert df["id"].tolist() == ["s1", "s2"]      # manifest order preserved


def test_run_aml_inference_raises_on_duplicate_export_folderpaths(tmp_path, fake_aml_command):
    bundle_dir = _write_bundle(tmp_path)
    input_csv = "memory://run13/cells/input.csv"
    rows = [
        {"id": "s1", "shapefilepath": "g1.geojson", "catalog_filepath": "c1.parquet",
         "startdate": "2018-01-01", "enddate": "2018-02-01",
         "export_folderpath": "memory://run13/cells/shared", "mosaic_days": 20,
         "mosaic_scheme": "calendar", "declaration_filepath": "unused", "bands": "B04,B08"},
        {"id": "s1", "shapefilepath": "g2.geojson", "catalog_filepath": "c2.parquet",
         "startdate": "2018-03-01", "enddate": "2018-04-01",   # DIFFERENT content, SAME folder
         "export_folderpath": "memory://run13/cells/shared", "mosaic_days": 20,
         "mosaic_scheme": "calendar", "declaration_filepath": "unused", "bands": "B04,B08"},
    ]
    with fs.open(input_csv, "w") as f:
        pd.DataFrame(rows).to_csv(f, index=False)
    ml_client = _FakeMLClient({"smoke": "Completed", "0": "Completed"})

    with pytest.raises(ValueError, match="duplicate unit dispatch"):
        runners.run_aml_inference(
            input_csv, bundle_dir, cluster="c", environment="fsd-infer-env:1",
            root="memory://run13/root", identity_client_id="x", n_shards=1,
            ml_client=ml_client, run_id="duprun",
        )
    assert ml_client.submitted == []   # never dispatched -- caught on the driver (D11)


def test_run_local_inference_raises_on_duplicate_export_folderpaths(tmp_path):
    csv = tmp_path / "input.csv"
    shared = str(tmp_path / "shared")
    rows = [
        {"id": "s1", "shapefilepath": "g1.geojson", "catalog_filepath": "c1.parquet",
         "startdate": "2018-01-01", "enddate": "2018-02-01", "export_folderpath": shared,
         "mosaic_days": 20, "mosaic_scheme": "calendar", "declaration_filepath": "unused", "bands": "B04,B08"},
        {"id": "s1", "shapefilepath": "g2.geojson", "catalog_filepath": "c2.parquet",
         "startdate": "2018-03-01", "enddate": "2018-04-01", "export_folderpath": shared,
         "mosaic_days": 20, "mosaic_scheme": "calendar", "declaration_filepath": "unused", "bands": "B04,B08"},
    ]
    pd.DataFrame(rows).to_csv(csv, index=False)

    with pytest.raises(ValueError, match="duplicate unit dispatch"):
        runners.run_local_inference(str(csv), cores=1, bundle_path=str(tmp_path / "bundle"))


def test_dedupe_on_unit_identity_is_non_vacuous(tmp_path):
    """A mutation removing the dedupe would leave 4 rows (2 shapes x 2 runs) -- prove
    the row-count assertion above actually catches that by calling the helper directly
    with pre-deduped-vs-not input."""
    from rise_eo.workflows.create_datacube import _dedupe_on_unit_identity

    rows = pd.DataFrame([
        {"id": "s1", "startdate": "2018-01-01", "enddate": "2019-01-01", "bands": "B04",
         "mosaic_days": 20, "mosaic_scheme": "calendar", "declaration_filepath": "unused",
         "added_on": "2026-01-01T00:00:00+00:00"},
        {"id": "s1", "startdate": "2018-01-01", "enddate": "2019-01-01", "bands": "B04",
         "mosaic_days": 20, "mosaic_scheme": "calendar", "declaration_filepath": "unused",
         "added_on": "2026-01-02T00:00:00+00:00"},
    ])
    deduped = _dedupe_on_unit_identity(rows)
    assert len(deduped) == 1                       # the dupe collapsed
    assert len(rows) == 2                           # ...and the input really did have two


# --- E1 (#145): skip_nan / predict_batch_size / overwrite travel the whole chain ----------
# (The Snakefile -> infer_task CLI hop is not pinned here: the existing Snakefile tests are
# dry-runs that assert only `returncode == 0`, so no test reads the shell command they plan.)

_E1_KW = dict(skip_nan=False, predict_batch_size=16, overwrite=True)


def test_run_aml_inference_command_carries_skip_nan_batch_size_overwrite(
    tmp_path, fake_aml_command,
):
    input_csv = "memory://run_e1/cells/input.csv"
    _write_input_csv(input_csv, n_units=1)
    ml_client = _FakeMLClient({"0": "Completed"})
    runners.run_aml_inference(
        input_csv, _write_bundle(tmp_path), cluster="c", environment="e:1",
        root="memory://run_e1/root", identity_client_id="x", n_shards=1,
        ml_client=ml_client, run_id="e1", skip_smoke=True, **_E1_KW,
    )
    cmd = ml_client.submitted[0][1].command
    assert "--no-skip-nan" in cmd and "--predict-batch-size 16" in cmd and "--overwrite" in cmd


def test_infer_shard_main_forwards_flags_to_run_infer_shard(monkeypatch):
    seen = {}
    monkeypatch.setattr(infer_shard, "run_infer_shard",
                        lambda *a, **kw: seen.update(kw) or {"status": "ok"})
    infer_shard.main(["s.csv", "b", "--no-skip-nan", "--predict-batch-size", "16", "--overwrite"])
    assert {k: seen[k] for k in _E1_KW} == _E1_KW


def test_run_infer_shard_forwards_flags_to_run_local_inference(tmp_path, monkeypatch):
    staged_bundle_url = "memory://run_e1b/_bundle"
    runners._stage_bundle(_write_bundle(tmp_path), staged_bundle_url)
    shard_url = "memory://run_e1b/shards/0.csv"
    _write_input_csv(shard_url, n_units=1)
    seen = {}
    monkeypatch.setattr(runners, "run_local_inference",
                        lambda *a, **kw: seen.update(kw) or types.SimpleNamespace(returncode=0))
    infer_shard.run_infer_shard(shard_url, staged_bundle_url, cores=1, **_E1_KW)
    assert {k: seen[k] for k in _E1_KW} == _E1_KW


@pytest.mark.parametrize("runner", ["run_local_inference", "run_local_infer_only"])
def test_local_inference_runners_put_flags_in_snakemake_config(tmp_path, monkeypatch, runner):
    input_csv = "memory://run_e1c/input.csv"
    _write_input_csv(input_csv, n_units=1)
    seen = {}
    monkeypatch.setattr(runners, "_run_snakemake",
                        lambda snakefile, cores, conf, **kw: seen.update(conf))
    getattr(runners, runner)(input_csv, cores=1, bundle_path=str(tmp_path / "b"), **_E1_KW)
    assert (seen["skip_nan"], seen["predict_batch_size"], seen["overwrite"]) == (0, 16, 1)
