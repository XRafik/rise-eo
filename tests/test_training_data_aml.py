"""Tests for spec 39 (create_training_data e2e on AML: flatten -> land-local): the
`workflows/flatten.py` CLI (D3), `runners.run_aml_flatten` (D3), `api._land_local` (D4),
`api.flatten_training_data` (D5), and `create_training_data`'s download phase (D1) +
driver-side features (D2) + optional `label_col` (D-labels) + dropped `n_timestamps`
preflight (D6).

No test requires Azure (spec 39 §7): the AML submission surface is substituted exactly as
specs 36/37/38 do it (`_FakeMLClient` + a fake `azure.ai.ml.command`), and blob is
`memory://`. Fast + synthetic.
"""

from __future__ import annotations

import datetime
import os
import types

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry
from rasterio.transform import from_origin

import rise_eo
from rise_eo import api
from rise_eo.datacube import flatten as flatten_lib
from rise_eo.storage import fs
from rise_eo.workflows import flatten as flatten_cli
from rise_eo.workflows import runners

JAN1 = datetime.datetime(2018, 1, 1)
JAN1_NEXT = datetime.datetime(2019, 1, 1)
TS = [pd.Timestamp("2018-06-01", tz="UTC"), pd.Timestamp("2018-07-01", tz="UTC")]
TRANSFORM = from_origin(500000, 5000000, 10, 10)


# --- shared fakes: AML client + job-builder (mirrors specs 36/37/38) --------------

class _NS(types.SimpleNamespace):
    pass


@pytest.fixture
def fake_aml_command(monkeypatch):
    def _cmd(**kwargs):
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(runners, "_import_aml_command", lambda: _cmd)
    return _cmd


class _FakeMLClient:
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


def _write_flatten_input_csv(url, n_rows=2, with_label=True):
    rows = [
        {"id": f"u{i}", "datacube_filepath": f"memory://cubes/u{i}/datacube.npy"}
        for i in range(n_rows)
    ]
    if with_label:
        for r in rows:
            r["label"] = "crop"
    with fs.open(url, "w") as f:
        pd.DataFrame(rows).to_csv(f, index=False)


# --- synthetic cube fixture (mirrors test_datacube_flatten.py) --------------------

def _save_cube(folder, arr, bands=("B04", "B08"), timestamps=TS, crs="EPSG:32633"):
    fs.makedirs(folder)
    fs.save_npy(f"{folder}/datacube.npy", arr)
    md = {"bands": list(bands), "timestamps": list(timestamps),
          "geotiff_metadata": {"width": arr.shape[2], "height": arr.shape[1],
                               "transform": TRANSFORM, "crs": crs}}
    fs.save_npy(f"{folder}/metadata.pickle.npy", md, allow_pickle=True)
    return f"{folder}/datacube.npy"


def _polys(id_col="fid", label_col="crop", n=2):
    gdf = gpd.GeoDataFrame(
        {
            id_col: list(range(n)),
            label_col: ["a", "b"][:n],
            "geometry": [shapely.geometry.box(i, 0, i + 1, 1) for i in range(n)],
        },
        crs="EPSG:4326",
    )
    return gdf


# --- test 1: flatten_training_data(runner="local") over synthetic cubes -----------

def test_flatten_training_data_local_stacks_cubes_and_labels_are_optional(tmp_path):
    a = np.ones((2, 2, 2, 2), dtype=np.uint16)
    b = np.ones((2, 2, 2, 2), dtype=np.uint16)
    fp_a = _save_cube("memory://flatten_local/A", a)
    fp_b = _save_cube("memory://flatten_local/B", b)

    input_csv = "memory://flatten_local/input.csv"
    df = pd.DataFrame({"id": ["A", "B"], "label": ["x", "y"],
                       "datacube_filepath": [fp_a, fp_b]})
    with fs.open(input_csv, "w") as f:
        df.to_csv(f, index=False)

    export = tmp_path / "export"
    td = api.flatten_training_data(input_csv, str(export), label_col="label")

    assert isinstance(td, rise_eo.TrainingData)
    assert td.n_pixels == 8  # 4 pixels/cube * 2 cubes, none nodata
    assert fs.exists(str(export / "labels.npy"))
    assert not fs.exists(str(export / "features.npy"))  # no adapter/feature_sequence given

    # D-labels: label_col=None -> no labels.npy at all
    export2 = tmp_path / "export_nolabel"
    td2 = api.flatten_training_data(input_csv, str(export2))
    assert not fs.exists(str(export2 / "labels.npy"))
    assert "labels" not in td2.load()


# --- test 2: run_aml_flatten builds exactly ONE job (a reduce, not a fan-out) -----

@pytest.mark.parametrize("n_rows", [1, 900])
def test_run_aml_flatten_submits_exactly_one_job_non_vacuous(fake_aml_command, n_rows):
    input_csv = f"memory://flatten_aml_{n_rows}/input.csv"
    _write_flatten_input_csv(input_csv, n_rows=n_rows)
    ml_client = _FakeMLClient(["Completed"])

    result = runners.run_aml_flatten(
        input_csv, f"memory://flatten_aml_{n_rows}/out",
        id_col="id", label_col="label", cluster="c", environment="rise-eo-env:1",
        root=f"memory://flatten_aml_{n_rows}/root", identity_client_id="deadbeef",
        ml_client=ml_client, run_id="r1",
    )

    assert result["n_jobs"] == 1
    assert len(ml_client.submitted) == 1  # non-vacuous at both n=1 and n=900
    job = ml_client.submitted[0]
    assert "--input-csv" in job.command and input_csv in job.command
    assert "--export" in job.command
    assert job.environment_variables["AZURE_CLIENT_ID"] == "deadbeef"


# --- test 3: land-local (D4) -------------------------------------------------------

def test_land_local_transfers_files_and_labels_iff_present(tmp_path):
    blob_prefix = "memory://land_local/reduce_out"
    fs.save_npy(f"{blob_prefix}/data.npy", np.zeros((3, 2, 2)))
    fs.save_npy(f"{blob_prefix}/coords.npy", np.zeros((3, 2)))
    fs.save_npy(f"{blob_prefix}/ids.npy", np.array(["a", "a", "b"]))
    fs.save_npy(f"{blob_prefix}/metadata.pickle.npy", {"bands": ["B04"]}, allow_pickle=True)
    fs.save_npy(f"{blob_prefix}/labels.npy", np.array(["x", "x", "y"]))

    local = tmp_path / "landed"
    api._land_local(
        blob_prefix, str(local),
        ["data.npy", "coords.npy", "ids.npy", "metadata.pickle.npy", "labels.npy"],
    )

    for name in ("data.npy", "coords.npy", "ids.npy", "metadata.pickle.npy", "labels.npy"):
        assert fs.exists(str(local / name))
    landed = fs.load_npy(str(local / "ids.npy"))
    assert list(landed) == ["a", "a", "b"]

    # labels_col not present -> not requested, not landed
    local2 = tmp_path / "landed_nolabel"
    api._land_local(blob_prefix, str(local2), ["data.npy", "coords.npy", "ids.npy",
                                               "metadata.pickle.npy"])
    assert not fs.exists(str(local2 / "labels.npy"))


def test_land_local_leaves_no_partial_file_when_a_transfer_fails(tmp_path, monkeypatch):
    blob_prefix = "memory://land_local_fail/reduce_out"
    fs.save_npy(f"{blob_prefix}/data.npy", np.zeros((1, 1, 1)))

    def _boom(src, dst, **kw):
        raise RuntimeError("simulated transfer failure")

    monkeypatch.setattr(fs, "transfer", _boom)
    local = tmp_path / "landed"
    with pytest.raises(RuntimeError, match="simulated transfer failure"):
        api._land_local(blob_prefix, str(local), ["data.npy"])
    assert not fs.exists(str(local / "data.npy"))


# --- test 4: create_training_data(download=True) orchestrates in order ------------

def test_create_training_data_download_orchestrates_download_build_flatten_in_order(
    tmp_path, monkeypatch,
):
    order = []
    n_px, T, bands = 3, 2, ["B04", "B08"]
    catalog = str(tmp_path / "data" / "sentinel-2-l2a" / "catalog.parquet")

    def fake_download_verb(**kwargs):
        order.append("download")
        # a real download would write the catalog; simulate it existing afterward
        fs.makedirs(os.path.dirname(catalog))
        os.makedirs(os.path.dirname(catalog), exist_ok=True)
        with open(catalog, "w") as f:
            f.write("")
        assert isinstance(kwargs["roi"], str) and kwargs["roi"].endswith(".geojson")
        return catalog

    def fake_run_create_datacube(*, csv_filepath, **kw):
        order.append("build")
        assert fs.exists(catalog)  # catalog exists BEFORE the build reads it
        pd.DataFrame(
            {"datacube_filepath": ["x/datacube.npy"], "id": [0], "label": ["a"]}
        ).to_csv(csv_filepath, index=False)

    def fake_flatten(*, export_folderpath, **kw):
        order.append("flatten")
        fs.makedirs(export_folderpath)
        fs.save_npy(f"{export_folderpath}/data.npy", np.zeros((n_px, T, len(bands))))
        fs.save_npy(f"{export_folderpath}/ids.npy", np.zeros(n_px))
        fs.save_npy(f"{export_folderpath}/coords.npy", np.zeros((n_px, 2)))
        fs.save_npy(f"{export_folderpath}/labels.npy", np.array(["a"] * n_px))
        fs.save_npy(
            f"{export_folderpath}/metadata.pickle.npy",
            {"timestamps": list(range(T)), "bands": bands}, allow_pickle=True,
        )

    monkeypatch.setattr(api, "_download_verb", fake_download_verb)
    monkeypatch.setattr(api._create_datacube, "run_create_datacube", fake_run_create_datacube)
    monkeypatch.setattr(api._flatten, "flatten", fake_flatten)

    td = rise_eo.create_training_data(
        label_polygons=_polys(), catalog_filepath=catalog,
        startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=bands,
        id_col="fid", label_col="crop", export_folderpath=str(tmp_path / "export"),
        source="mpc", download=True, max_tiles=5,
    )

    assert order == ["download", "build", "flatten"]
    assert isinstance(td, rise_eo.TrainingData)
    assert td.n_pixels == n_px


def test_create_training_data_download_stages_gdf_once_for_download_and_build(
    tmp_path, monkeypatch,
):
    """Q3: an in-memory gdf is staged to ONE geojson that serves both the download roi
    and the build shapefile."""
    catalog = str(tmp_path / "data" / "sentinel-2-l2a" / "catalog.parquet")
    seen = {}

    def fake_download_verb(*, roi, **kw):
        seen["download_roi"] = roi
        fs.makedirs(os.path.dirname(catalog))
        os.makedirs(os.path.dirname(catalog), exist_ok=True)
        with open(catalog, "w") as f:
            f.write("")
        return catalog

    def fake_run_create_datacube(*, csv_filepath, shapefilepath, **kw):
        seen["build_shapefilepath"] = shapefilepath
        pd.DataFrame(
            {"datacube_filepath": ["x/datacube.npy"], "id": [0], "label": ["a"]}
        ).to_csv(csv_filepath, index=False)

    def fake_flatten(*, export_folderpath, **kw):
        fs.makedirs(export_folderpath)
        fs.save_npy(f"{export_folderpath}/data.npy", np.zeros((1, 1, 1)))
        fs.save_npy(f"{export_folderpath}/ids.npy", np.zeros(1))
        fs.save_npy(f"{export_folderpath}/coords.npy", np.zeros((1, 2)))
        fs.save_npy(
            f"{export_folderpath}/metadata.pickle.npy",
            {"timestamps": [0], "bands": ["B04"]}, allow_pickle=True,
        )

    monkeypatch.setattr(api, "_download_verb", fake_download_verb)
    monkeypatch.setattr(api._create_datacube, "run_create_datacube", fake_run_create_datacube)
    monkeypatch.setattr(api._flatten, "flatten", fake_flatten)

    rise_eo.create_training_data(
        label_polygons=_polys(), catalog_filepath=catalog,
        startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=["B04"],
        id_col="fid", label_col="crop", export_folderpath=str(tmp_path / "export"),
        source="mpc", download=True, max_tiles=5,
    )

    assert seen["download_roi"] == seen["build_shapefilepath"]  # ONE staged geojson, reused


def test_create_training_data_stages_gdf_with_timestamp_column(tmp_path, monkeypatch):
    """Regression (runbook 39 Phase 2, 2026-07-27): staging an in-memory gdf that carries a
    Timestamp/datetime property column (e.g. EuroCrops' obs date) must not crash. The staging
    write uses `gdf.to_json()`, which routes through json.dumps and raises
    `TypeError: Object of type Timestamp is not JSON serializable` without `default=str`.
    The pre-fix code (and every existing test's clean int/str-only fixture) let this slip."""
    catalog = str(tmp_path / "data" / "sentinel-2-l2a" / "catalog.parquet")

    def fake_download_verb(*, roi, **kw):
        fs.makedirs(os.path.dirname(catalog))
        os.makedirs(os.path.dirname(catalog), exist_ok=True)
        with open(catalog, "w") as f:
            f.write("")
        # the staged geojson must be real, readable, and preserve id/label
        staged = gpd.read_file(roi)
        assert "fid" in staged.columns and "crop" in staged.columns
        return catalog

    def fake_run_create_datacube(*, csv_filepath, **kw):
        pd.DataFrame(
            {"datacube_filepath": ["x/datacube.npy"], "id": [0], "label": ["a"]}
        ).to_csv(csv_filepath, index=False)

    def fake_flatten(*, export_folderpath, **kw):
        fs.makedirs(export_folderpath)
        fs.save_npy(f"{export_folderpath}/data.npy", np.zeros((1, 1, 1)))
        fs.save_npy(f"{export_folderpath}/ids.npy", np.zeros(1))
        fs.save_npy(f"{export_folderpath}/coords.npy", np.zeros((1, 2)))
        fs.save_npy(
            f"{export_folderpath}/metadata.pickle.npy",
            {"timestamps": [0], "bands": ["B04"]}, allow_pickle=True,
        )

    monkeypatch.setattr(api, "_download_verb", fake_download_verb)
    monkeypatch.setattr(api._create_datacube, "run_create_datacube", fake_run_create_datacube)
    monkeypatch.setattr(api._flatten, "flatten", fake_flatten)

    gdf = _polys()
    gdf["obs_date"] = [pd.Timestamp("2018-05-01"), pd.Timestamp("2018-06-01")]  # the trap

    # must NOT raise TypeError: Object of type Timestamp is not JSON serializable
    rise_eo.create_training_data(
        label_polygons=gdf, catalog_filepath=catalog,
        startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=["B04"],
        id_col="fid", label_col="crop", export_folderpath=str(tmp_path / "export"),
        source="mpc", download=True, max_tiles=5,
    )


# --- test 5: driver-side features (D2/ADR-0020) ------------------------------------

class _FakeAdapter:
    required_bands = ["B04", "B08"]
    feature_sequence = None

    def features(self, data5d, band_indices):
        return data5d, band_indices


def test_create_training_data_aml_features_written_locally_after_land_local(
    tmp_path, monkeypatch,
):
    n_px, T, bands = 4, 2, ["B04", "B08"]
    catalog = str(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    os.makedirs(os.path.dirname(catalog), exist_ok=True)
    with open(catalog, "w") as f:
        f.write("")

    def fake_run_create_datacube(*, csv_filepath, **kw):
        pd.DataFrame(
            {"datacube_filepath": ["x/datacube.npy"], "id": [0], "label": ["a"]}
        ).to_csv(csv_filepath, index=False)

    calls = {}

    def fake_run_aml_flatten(input_csv, export_folderpath, *, id_col, label_col,
                              filepath_col, nodata, root, run_id, **kw):
        calls["kwargs"] = kw  # ADR-0020 pin: must never contain "adapter"
        fs.save_npy(f"{export_folderpath}/data.npy", np.ones((n_px, T, len(bands))))
        fs.save_npy(f"{export_folderpath}/coords.npy", np.zeros((n_px, 2)))
        fs.save_npy(f"{export_folderpath}/ids.npy", np.array(["a"] * n_px))
        fs.save_npy(
            f"{export_folderpath}/metadata.pickle.npy",
            {"timestamps": list(range(T)), "bands": bands}, allow_pickle=True,
        )
        if label_col is not None:
            fs.save_npy(f"{export_folderpath}/labels.npy", np.array(["a"] * n_px))
        return {"run_id": run_id, "n_jobs": 1}

    monkeypatch.setattr(api._create_datacube, "run_create_datacube", fake_run_create_datacube)
    monkeypatch.setattr(runners, "run_aml_flatten", fake_run_aml_flatten)

    export = tmp_path / "export"
    td = rise_eo.create_training_data(
        label_polygons=_polys(), catalog_filepath=catalog,
        startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=bands,
        id_col="fid", label_col="crop", export_folderpath=str(export),
        runner="aml", runner_kwargs={
            "cluster": "c", "environment": "rise-eo-env:1", "root": "memory://features_aml/root",
            "identity_client_id": "id",
        },
        adapter=_FakeAdapter(),
    )

    assert fs.exists(str(export / "features.npy"))  # written locally after land-local
    assert td.feature_bands is not None
    assert "adapter" not in calls["kwargs"]  # never reaches the faked job's kwargs


# --- test 6: D6 -- no n_timestamps preflight; cross-cube consistency still enforced -

def test_create_training_data_accepts_any_t_no_adapter_n_timestamps_preflight(
    tmp_path, monkeypatch,
):
    catalog = str(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    os.makedirs(os.path.dirname(catalog), exist_ok=True)
    with open(catalog, "w") as f:
        f.write("")

    class _MismatchedTAdapter:
        required_bands = ["B04"]
        n_timestamps = 999  # deliberately does not match the window's computed T
        feature_sequence = None

        def features(self, data5d, band_indices):
            return data5d, band_indices

    def fake_run_create_datacube(*, csv_filepath, **kw):
        pd.DataFrame(
            {"datacube_filepath": ["x/datacube.npy"], "id": [0], "label": ["a"]}
        ).to_csv(csv_filepath, index=False)

    def fake_flatten(*, export_folderpath, **kw):
        fs.makedirs(export_folderpath)
        fs.save_npy(f"{export_folderpath}/data.npy", np.zeros((1, 19, 1)))
        fs.save_npy(f"{export_folderpath}/ids.npy", np.zeros(1))
        fs.save_npy(f"{export_folderpath}/coords.npy", np.zeros((1, 2)))
        fs.save_npy(
            f"{export_folderpath}/metadata.pickle.npy",
            {"timestamps": list(range(19)), "bands": ["B04"]}, allow_pickle=True,
        )

    monkeypatch.setattr(api._create_datacube, "run_create_datacube", fake_run_create_datacube)
    monkeypatch.setattr(api._flatten, "flatten", fake_flatten)

    # no PreflightError despite adapter.n_timestamps=999 != computed T=19
    rise_eo.create_training_data(
        label_polygons=_polys(), catalog_filepath=catalog,
        startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=["B04"],
        id_col="fid", label_col="crop", export_folderpath=str(tmp_path / "export"),
        adapter=_MismatchedTAdapter(),
    )


def test_flatten_training_data_raises_on_cross_cube_timestamp_mismatch(tmp_path):
    a = np.ones((2, 2, 2, 1), dtype=np.uint16)
    b = np.ones((3, 2, 2, 1), dtype=np.uint16)  # different T
    fp_a = _save_cube("memory://ts_mismatch/A", a, bands=("B04",), timestamps=TS)
    fp_b = _save_cube(
        "memory://ts_mismatch/B", b, bands=("B04",),
        timestamps=[TS[0], TS[1], pd.Timestamp("2018-08-01", tz="UTC")],
    )
    input_csv = "memory://ts_mismatch/input.csv"
    with fs.open(input_csv, "w") as f:
        pd.DataFrame({"id": ["A", "B"], "datacube_filepath": [fp_a, fp_b]}).to_csv(
            f, index=False
        )

    with pytest.raises(ValueError, match="not consistent"):
        api.flatten_training_data(input_csv, str(tmp_path / "export"))


# --- test 7: workflows/flatten.py CLI -----------------------------------------------

def test_flatten_cli_reads_csv_and_calls_datacube_flatten(tmp_path, monkeypatch):
    input_csv = "memory://flatten_cli/input.csv"
    with fs.open(input_csv, "w") as f:
        pd.DataFrame({"id": ["A"], "datacube_filepath": ["memory://cubes/A/datacube.npy"]}).to_csv(
            f, index=False
        )

    calls = {}

    def fake_flatten(*, filepaths_df, filepath_col, id_col, export_folderpath, label_col, nodata):
        calls["n_rows"] = len(filepaths_df)
        calls["filepath_col"] = filepath_col
        calls["id_col"] = id_col
        calls["label_col"] = label_col
        calls["export_folderpath"] = export_folderpath

    monkeypatch.setattr(flatten_cli, "_flatten", types.SimpleNamespace(flatten=fake_flatten))

    status_url = "memory://flatten_cli/status.json"
    status = flatten_cli.main([
        "--input-csv", input_csv, "--export", "memory://flatten_cli/out",
        "--status-url", status_url,
    ])
    assert status is None  # main() returns None; raises on failure

    assert calls["n_rows"] == 1
    assert calls["filepath_col"] == "datacube_filepath"
    assert calls["id_col"] == "id"
    assert calls["label_col"] is None
    assert calls["export_folderpath"] == "memory://flatten_cli/out"

    import json
    with fs.open(status_url, "r") as f:
        published = json.load(f)
    assert published["status"] == "ok"
    assert published["n_cubes"] == 1


def test_flatten_cli_uses_the_real_flatten_module_by_default():
    """Non-vacuousness for test 7: the CLI's `run()` really does delegate to
    `datacube.flatten.flatten`, not a stub -- proven by checking the module reference
    the CLI calls is exactly `rise_eo.datacube.flatten`."""
    assert flatten_cli._flatten is flatten_lib


# --- test 8: non-vacuousness -------------------------------------------------------

def test_run_aml_flatten_one_job_assertion_is_non_vacuous(fake_aml_command):
    input_csv = "memory://flatten_vacuous/input.csv"
    _write_flatten_input_csv(input_csv, n_rows=3)
    ml_client = _FakeMLClient(["Completed"])

    runners.run_aml_flatten(
        input_csv, "memory://flatten_vacuous/out", id_col="id", label_col="label",
        cluster="c", environment="rise-eo-env:1", root="memory://flatten_vacuous/root",
        identity_client_id="id", ml_client=ml_client, run_id="r1",
    )
    with pytest.raises(AssertionError):
        assert len(ml_client.submitted) == 2  # deliberately wrong: only 1 job is ever built


def test_features_npy_absent_without_adapter_assertion_is_non_vacuous(tmp_path):
    fp = _save_cube("memory://vacuous_features/A", np.ones((2, 2, 2, 2), dtype=np.uint16))
    input_csv = "memory://vacuous_features/input.csv"
    with fs.open(input_csv, "w") as f:
        pd.DataFrame({"id": ["A"], "datacube_filepath": [fp]}).to_csv(f, index=False)

    export = tmp_path / "export"
    api.flatten_training_data(input_csv, str(export))  # no adapter given
    with pytest.raises(AssertionError):
        assert fs.exists(str(export / "features.npy"))  # deliberately wrong: never written
