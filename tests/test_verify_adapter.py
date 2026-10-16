"""Fast + synthetic tests for `rise_eo.verify_adapter` (spec 48) — preflight, cell selection,
resume-identity, and the returned verdict shape, with the build/land/infer legs
monkeypatched. The real end-to-end (`runner="local"` against the tutorial fixture, AC10)
lives in `test_tutorial_fixture.py::test_verify_adapter_real_fixture_local_runner`.
"""

from __future__ import annotations

import datetime
import inspect
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry

import rise_eo
from rise_eo import api
from rise_eo.model.adapter import BaseModelAdapter
from rise_eo.storage import fs
from rise_eo.workflows import create_datacube as _create_datacube
from rise_eo.workflows import infer_only_task as _infer_only_task

JAN1 = datetime.datetime(2018, 1, 1)
JAN60 = datetime.datetime(2018, 3, 2)  # 60 days later
BANDS = ["B04", "B08"]
T = 3


class _FakeAdapter(BaseModelAdapter):
    """Module-level (spec 44's bundle auto-detect needs a real importable class)."""

    required_bands = BANDS
    n_timestamps = 0  # model-determined -- no T preflight (mirrors the tutorial fixture)
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["output"]
    feature_sequence = None

    def features(self, data5d, band_indices):
        return data5d, band_indices

    def predict(self, X_chunk):
        return np.zeros(X_chunk.shape[0], dtype="uint8")


class _WrongTAdapter(_FakeAdapter):
    n_timestamps = T + 5  # the cube these tests build has T timestamps


ROI = gpd.GeoDataFrame({"geometry": [shapely.geometry.box(0, 0, 2, 1)]}, crs="EPSG:4326")

# Two disjoint grid cells covering the roi -- cell_a intersects more catalog tiles than
# cell_b, so the deterministic default (D3) must always pick cell_a.
_GRIDS = gpd.GeoDataFrame(
    {"id": ["cell_a", "cell_b"]},
    geometry=[shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(1, 0, 2, 1)],
    crs="EPSG:4326",
)


def _catalog(tmp_path, n_cell_a=3, n_cell_b=1):
    """Tiles whose geometry intersects cell_a more than cell_b -- no imagery, just enough
    for `_filter_gdf`'s date+overlap logic (geometry + timestamp)."""
    rows = []
    for i in range(n_cell_a):
        rows.append({"id": f"a{i}", "acquisition_key": f"a{i}", "processing_version": None,
                     "processing_datetime": None, "source": "mpc",
                     "geometry": shapely.geometry.box(0, 0, 1, 1), "timestamp": JAN1})
    for i in range(n_cell_b):
        rows.append({"id": f"b{i}", "acquisition_key": f"b{i}", "processing_version": None,
                     "processing_datetime": None, "source": "mpc",
                     "geometry": shapely.geometry.box(1, 0, 2, 1), "timestamp": JAN1})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], utc=True)
    fp = str(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    fs.write_parquet(fp, gdf)
    return fp


def _patch_grid(monkeypatch):
    import rise_eo.grid as grid_mod

    monkeypatch.setattr(grid_mod, "roi_to_s2_grids", lambda *a, **kw: _GRIDS.copy())


def _patch_build_and_infer(monkeypatch, *, output_dtype="uint8"):
    """Stand in for the real per-cell build (`create_datacube.run_create_datacube`) and the
    real inference leg (`infer_only_task.run_infer_only`) so these tests stay fast and
    network-free -- the wiring under test is preflight/selection/resume/report, not the
    (already-tested) build or inference units themselves."""
    calls = {"build": 0, "infer": 0}

    def fake_run_create_datacube(*, csv_filepath, **kw):
        calls["build"] += 1
        cube_dir = os.path.join(os.path.dirname(csv_filepath), "cellbuild")
        fs.makedirs(cube_dir)
        fs.save_npy(os.path.join(cube_dir, "datacube.npy"),
                    np.zeros((T, 2, 2, len(BANDS)), dtype="float32"))
        fs.save_npy(os.path.join(cube_dir, "metadata.pickle.npy"),
                    {"timestamps": list(range(T)), "bands": BANDS}, allow_pickle=True)
        pd.DataFrame(
            {"datacube_filepath": [os.path.join(cube_dir, "datacube.npy")]}
        ).to_csv(csv_filepath, index=False)

    def fake_run_infer_only(input_csv, rows, bundle_path, **kw):
        calls["infer"] += 1
        import rasterio
        from rasterio.transform import from_origin

        df = pd.read_csv(input_csv)
        out_fp = str(df["output_filepath"].iloc[0])
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        profile = {
            "driver": "GTiff", "height": 2, "width": 2, "count": 1, "dtype": output_dtype,
            "crs": "EPSG:4326", "transform": from_origin(0, 1, 0.5, 0.5), "nodata": 255,
        }
        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(np.zeros((1, 2, 2), dtype=output_dtype))
        return [out_fp]

    monkeypatch.setattr(_create_datacube, "run_create_datacube", fake_run_create_datacube)
    monkeypatch.setattr(_infer_only_task, "run_infer_only", fake_run_infer_only)
    return calls


# --- AC1/AC4/AC8/AC9/AC12/AC14: happy path + report shape --------------------

def test_verify_adapter_happy_path_report_shape(tmp_path, monkeypatch):
    _patch_grid(monkeypatch)
    calls = _patch_build_and_infer(monkeypatch)
    cat = _catalog(tmp_path)
    export = str(tmp_path / "export")

    result = rise_eo.verify_adapter(
        _FakeAdapter(), roi=ROI, catalog_filepath=cat,
        startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
        export_folderpath=export, cell="cell_a",
    )

    assert calls["build"] == 1  # AC1: exactly one build
    assert calls["infer"] == 1
    assert result["step"] == "verify_adapter"
    assert result["pass"] is True
    assert result["status"] == "ok"
    for key in ("metrics", "expected", "error"):
        assert key in result
    for key in ("cube_shape", "cube_t", "adapter_n_timestamps", "post_feature_sequence_bands",
                "required_bands", "output_dtype", "output_value_min", "output_value_max",
                "output_nodata_fraction", "cube_filepath", "output_filepath", "grids_filepath"):
        assert key in result["metrics"], key

    # AC4: grids.geojson always written and its path printed/returned.
    assert os.path.exists(os.path.join(export, "grids.geojson"))
    assert result["metrics"]["grids_filepath"] == os.path.join(export, "grids.geojson")

    # The auto-saved bundle is reported, and is a real persistent bundle on disk -- not a
    # temp one -- so "reuse the bundle gate 1 verified" is a supported move rather than a
    # guess at the folder layout.
    assert result["metrics"]["bundle_path"] == os.path.join(export, "_bundle")
    assert os.path.exists(os.path.join(export, "_bundle", "bundle.json"))

    # AC14: no flattened/feature array.
    assert not os.path.exists(os.path.join(export, "features.npy"))
    assert not os.path.exists(os.path.join(export, "data.npy"))
    assert not os.path.exists(os.path.join(export, "ids.npy"))

    # D8: the cube + output land locally.
    assert os.path.exists(os.path.join(export, "datacube.npy"))
    assert os.path.exists(os.path.join(export, "metadata.pickle.npy"))
    assert os.path.exists(os.path.join(export, "output.tif"))


# --- AC2/AC5: deterministic cell selection + resume by identity --------------

def test_verify_adapter_deterministic_cell_and_resume_submits_no_job(tmp_path, monkeypatch, capsys):
    _patch_grid(monkeypatch)
    calls = _patch_build_and_infer(monkeypatch)
    cat = _catalog(tmp_path, n_cell_a=3, n_cell_b=1)
    export = str(tmp_path / "export")

    result1 = rise_eo.verify_adapter(
        _FakeAdapter(), roi=ROI, catalog_filepath=cat,
        startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS, export_folderpath=export,
    )
    assert result1["metrics"]["cell"] == "cell_a"  # largest in-window coverage
    out = capsys.readouterr().out
    assert "cell='cell_a'" in out or 'cell_a' in out
    assert calls["build"] == 1

    # AC5: a second call, same roi/window/bands/mosaic_days -> no rebuild, straight to
    # inference (though inference itself always reruns, D-not-in-scope).
    result2 = rise_eo.verify_adapter(
        _FakeAdapter(), roi=ROI, catalog_filepath=cat,
        startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS, export_folderpath=export,
    )
    assert result2["metrics"]["cell"] == "cell_a"
    assert calls["build"] == 1  # still 1 -- no second build job
    assert calls["infer"] == 2


def test_verify_adapter_changed_window_refuses_resume(tmp_path, monkeypatch):
    _patch_grid(monkeypatch)
    _patch_build_and_infer(monkeypatch)
    cat = _catalog(tmp_path)
    export = str(tmp_path / "export")

    rise_eo.verify_adapter(
        _FakeAdapter(), roi=ROI, catalog_filepath=cat,
        startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
        export_folderpath=export, cell="cell_a",
    )
    with pytest.raises(api.PreflightError, match="DIFFERENT request"):
        rise_eo.verify_adapter(
            _FakeAdapter(), roi=ROI, catalog_filepath=cat,
            startdate=JAN1, enddate=datetime.datetime(2018, 4, 1), mosaic_days=20,
            bands=BANDS, export_folderpath=export, cell="cell_a",
        )


# --- AC3: explicit bad cell id ------------------------------------------------

def test_verify_adapter_unknown_cell_raises_with_available_ids(tmp_path, monkeypatch):
    _patch_grid(monkeypatch)
    _patch_build_and_infer(monkeypatch)
    cat = _catalog(tmp_path)
    export = str(tmp_path / "export")

    with pytest.raises(api.PreflightError, match="cell_a"):
        rise_eo.verify_adapter(
            _FakeAdapter(), roi=ROI, catalog_filepath=cat,
            startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
            export_folderpath=export, cell="not-a-real-cell",
        )


# --- AC13: cell="random" prints + is pinnable ---------------------------------

def test_verify_adapter_random_cell_prints_and_pins(tmp_path, monkeypatch, capsys):
    _patch_grid(monkeypatch)
    _patch_build_and_infer(monkeypatch)
    cat = _catalog(tmp_path)
    export = str(tmp_path / "export")

    result = rise_eo.verify_adapter(
        _FakeAdapter(), roi=ROI, catalog_filepath=cat,
        startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
        export_folderpath=export, cell="random",
    )
    chosen = result["metrics"]["cell"]
    assert chosen in ("cell_a", "cell_b")
    out = capsys.readouterr().out
    assert "random" in out and chosen in out

    # pinned: passing the printed id back as cell= reproduces the same cell.
    result2 = rise_eo.verify_adapter(
        _FakeAdapter(), roi=ROI, catalog_filepath=cat,
        startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
        export_folderpath=str(tmp_path / "export2"), cell=chosen,
    )
    assert result2["metrics"]["cell"] == chosen


# --- AC9: a T mismatch is a verdict, not a raise ------------------------------

def test_verify_adapter_t_mismatch_is_pass_false_not_raised(tmp_path, monkeypatch):
    _patch_grid(monkeypatch)
    _patch_build_and_infer(monkeypatch)
    cat = _catalog(tmp_path)
    export = str(tmp_path / "export")

    result = rise_eo.verify_adapter(
        _WrongTAdapter(), roi=ROI, catalog_filepath=cat,
        startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
        export_folderpath=export, cell="cell_a",
    )
    assert result["pass"] is False
    assert result["status"] == "fail"
    assert str(T) in result["error"] and str(T + 5) in result["error"]


# --- AC7: the bundle verify_adapter produces == the one run_inference would use ----

def test_verify_adapter_bundle_matches_ensure_bundle(tmp_path, monkeypatch):
    _patch_grid(monkeypatch)
    _patch_build_and_infer(monkeypatch)
    cat = _catalog(tmp_path)
    export = str(tmp_path / "export")

    rise_eo.verify_adapter(
        _FakeAdapter(), roi=ROI, catalog_filepath=cat,
        startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
        export_folderpath=export, cell="cell_a",
    )
    from rise_eo.model import bundle as _bundle

    got = _bundle.read_spec(os.path.join(export, "_bundle"))
    want = _bundle.read_spec(
        api._ensure_bundle(_FakeAdapter(), str(tmp_path / "elsewhere"), why="test")
    )
    assert got["adapter"] == want["adapter"]
    assert got["required_bands"] == want["required_bands"] == BANDS


# --- AC6: no branch anywhere says "if verify_adapter" -------------------------

def test_no_verify_adapter_branch_in_shared_inference_code():
    from rise_eo.model import bundle as bundle_mod
    from rise_eo.model import engine as engine_mod

    for mod in (engine_mod, _infer_only_task, bundle_mod):
        src = inspect.getsource(mod)
        assert "verify_adapter" not in src


def test_export_folderpath_required(tmp_path, monkeypatch):
    _patch_grid(monkeypatch)
    _patch_build_and_infer(monkeypatch)
    cat = _catalog(tmp_path)
    with pytest.raises(api.PreflightError, match="export_folderpath"):
        rise_eo.verify_adapter(
            _FakeAdapter(), roi=ROI, catalog_filepath=cat,
            startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
            export_folderpath="",
        )


# --- D8: the verdict is a FILE, not only a return value ----------------------

def test_result_json_is_written_on_pass_and_on_fail(tmp_path, monkeypatch):
    """D8 lists `_result.json` among the artifacts `export_folderpath` holds, and spec 24's
    whole run-book protocol is the user pasting that file back -- so it must land on disk,
    on the failing exits as much as the passing one."""
    _patch_grid(monkeypatch)
    _patch_build_and_infer(monkeypatch)
    cat = _catalog(tmp_path)

    export_ok = str(tmp_path / "export_ok")
    result = rise_eo.verify_adapter(
        _FakeAdapter(), roi=ROI, catalog_filepath=cat,
        startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
        export_folderpath=export_ok, cell="cell_a",
    )
    result_filepath = os.path.join(export_ok, "_result.json")
    assert os.path.exists(result_filepath)
    with open(result_filepath) as f:
        on_disk = json.load(f)
    assert on_disk["pass"] is True
    assert on_disk["step"] == "verify_adapter"
    assert on_disk["metrics"]["cell"] == result["metrics"]["cell"]

    export_fail = str(tmp_path / "export_fail")
    rise_eo.verify_adapter(
        _WrongTAdapter(), roi=ROI, catalog_filepath=cat,
        startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
        export_folderpath=export_fail, cell="cell_a",
    )
    with open(os.path.join(export_fail, "_result.json")) as f:
        failed = json.load(f)
    assert failed["pass"] is False
    assert failed["status"] == "fail"
    assert str(T + 5) in failed["error"]


# --- D5: existence is not identity, on the LANDING too -----------------------

def test_cube_without_a_matching_stamp_is_relanded_not_reused(tmp_path, monkeypatch):
    """A cube file left in `export_folderpath` with no stamp beside it (e.g. the caller
    deleted `_cube_stamp.json` to get past the "different request" refusal) must be
    OVERWRITTEN by the fresh build, never skipped as "already landed" -- otherwise the
    stamp written afterwards would claim this request's identity over the old cube's
    pixels."""
    _patch_grid(monkeypatch)
    calls = _patch_build_and_infer(monkeypatch)
    cat = _catalog(tmp_path)
    export = str(tmp_path / "export")

    kwargs = dict(
        roi=ROI, catalog_filepath=cat, startdate=JAN1, enddate=JAN60,
        mosaic_days=20, bands=BANDS, export_folderpath=export, cell="cell_a",
    )
    rise_eo.verify_adapter(_FakeAdapter(), **kwargs)
    assert calls["build"] == 1

    # Stand in for "a cube from some other request is sitting here, unstamped".
    landed = os.path.join(export, "datacube.npy")
    fs.save_npy(landed, np.full((T, 2, 2, len(BANDS)), 7.0, dtype="float32"))
    os.remove(os.path.join(export, "_cube_stamp.json"))

    rise_eo.verify_adapter(_FakeAdapter(), **kwargs)
    assert calls["build"] == 2                       # no stamp -> rebuild
    assert float(fs.load_npy(landed).max()) == 0.0   # and the sentinel was overwritten


# --- D4/D5: runner="aml" builds where the NODE can write, not on the driver ----

def _patch_build_capturing_paths(monkeypatch):
    """Like `_patch_build_and_infer`'s build half, but records the run/csv paths the verb
    hands to the per-cell unit -- what the NODE would be told to write."""
    seen = {}

    def fake_run_create_datacube(*, csv_filepath, run_folderpath, shapefilepath, **kw):
        seen["csv_filepath"] = csv_filepath
        seen["run_folderpath"] = run_folderpath
        seen["shapefilepath"] = shapefilepath
        cube_dir = os.path.join(os.path.dirname(csv_filepath), "cellbuild")
        fs.makedirs(cube_dir)
        fs.save_npy(os.path.join(cube_dir, "datacube.npy"),
                    np.zeros((T, 2, 2, len(BANDS)), dtype="float32"))
        fs.save_npy(os.path.join(cube_dir, "metadata.pickle.npy"),
                    {"timestamps": list(range(T)), "bands": BANDS}, allow_pickle=True)
        pd.DataFrame(
            {"datacube_filepath": [os.path.join(cube_dir, "datacube.npy")]}
        ).to_csv(csv_filepath, index=False)

    monkeypatch.setattr(_create_datacube, "run_create_datacube", fake_run_create_datacube)
    return seen


def test_aml_builds_under_the_blob_root_not_the_local_export(tmp_path, monkeypatch):
    """`create_datacube.setup` turns a LOCAL `run_folderpath` into an absolute DRIVER path
    and writes it into `input.csv`; on `runner="aml"` the node reads that csv, so the build
    must be rooted on blob (`runner_kwargs["root"]`) or the node is sent to write the cube
    at a path that does not exist on it. `export_folderpath` stays local -- it is where the
    cube is LANDED, never where it is built."""
    _patch_grid(monkeypatch)
    _patch_build_and_infer(monkeypatch)          # installs the inference fake
    seen = _patch_build_capturing_paths(monkeypatch)   # ... then override the build half
    cat = _catalog(tmp_path)
    export = str(tmp_path / "export")
    root = str(tmp_path / "blobroot")

    result = rise_eo.verify_adapter(
        _FakeAdapter(), roi=ROI, catalog_filepath=cat,
        startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
        export_folderpath=export, cell="cell_a",
        runner="aml", runner_kwargs={"root": root, "run_id": "RUNID"},
    )

    assert seen["run_folderpath"] == f"{root}/runs/RUNID/_verify_adapter"
    assert not seen["run_folderpath"].startswith(export)
    assert seen["csv_filepath"].startswith(root)
    # and the cube still LANDS locally, in export_folderpath (D5/D8)
    assert os.path.exists(os.path.join(export, "datacube.npy"))
    assert os.path.exists(os.path.join(export, "metadata.pickle.npy"))
    assert result["metrics"]["cube_filepath"] == os.path.join(export, "datacube.npy")


def test_aml_without_root_raises_naming_it(tmp_path, monkeypatch):
    _patch_grid(monkeypatch)
    _patch_build_and_infer(monkeypatch)
    cat = _catalog(tmp_path)
    with pytest.raises(api.PreflightError, match=r"root"):
        rise_eo.verify_adapter(
            _FakeAdapter(), roi=ROI, catalog_filepath=cat,
            startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
            export_folderpath=str(tmp_path / "export"), cell="cell_a",
            runner="aml", runner_kwargs={"cluster": "c"},
        )


def test_verify_adapter_forwards_skip_nan_and_batch_size_to_run_infer_only(tmp_path, monkeypatch):
    """D10 (#145): the inference leg gets the caller's `skip_nan` and `predict_batch_size`."""
    _patch_grid(monkeypatch)
    _patch_build_and_infer(monkeypatch)
    seen = {}
    fake_infer = _infer_only_task.run_infer_only

    def spy(*a, **kw):
        seen.update(kw)
        return fake_infer(*a, **kw)

    monkeypatch.setattr(_infer_only_task, "run_infer_only", spy)
    rise_eo.verify_adapter(
        _FakeAdapter(), roi=ROI, catalog_filepath=_catalog(tmp_path),
        startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=BANDS,
        export_folderpath=str(tmp_path / "export"), cell="cell_a",
        skip_nan=False, predict_batch_size=16,
    )
    assert (seen["skip_nan"], seen["predict_batch_size"]) == (False, 16)


# --- #145 P10: alias spelling of a band is not a missing band ----------------

@pytest.mark.parametrize("required, bands", [(["B8A"], ["nir08"]), (["nir08"], ["B8A"])])
def test_alias_spelling_is_not_a_missing_band(tmp_path, monkeypatch, required, bands):
    _patch_grid(monkeypatch)
    _patch_build_and_infer(monkeypatch)
    cat = _catalog(tmp_path)

    class _Alias(_FakeAdapter):
        required_bands = required

    with pytest.raises(api.PreflightError) as ei:
        rise_eo.verify_adapter(
            _Alias(), roi=ROI, catalog_filepath=cat,
            startdate=JAN1, enddate=JAN60, mosaic_days=20, bands=bands,
            export_folderpath="",
        )
    assert str(ei.value) == "preflight failed:\n  - export_folderpath is required."
