"""Tests for spec 49 D3/D4/D6: `flatten_training_data` skips the reduce when
`_flatten_stamp.json` already records the identity of the current cube set + run
parameters, and `create_training_data` forwards `overwrite=` to the right leg(s). Fast +
synthetic (the reduce itself, `datacube.flatten.flatten`, is monkeypatched)."""

from __future__ import annotations

import inspect
import os

import numpy as np
import pandas as pd
import pytest

from fsd import api
from fsd.storage import fs
from fsd.workflows import stamp as stamp_mod

T = 2
BANDS = ["B04", "B08"]


def _write_flatten_outputs(export_folderpath, *, n_px=3):
    fs.makedirs(export_folderpath)
    fs.save_npy(os.path.join(export_folderpath, "data.npy"), np.zeros((n_px, T, len(BANDS))))
    fs.save_npy(os.path.join(export_folderpath, "ids.npy"), np.arange(n_px))
    fs.save_npy(os.path.join(export_folderpath, "coords.npy"), np.zeros((n_px, 2)))
    fs.save_npy(
        os.path.join(export_folderpath, "metadata.pickle.npy"),
        {"timestamps": list(range(T)), "bands": list(BANDS)}, allow_pickle=True,
    )


def _input_csv(tmp_path, rows, name="input.csv"):
    fp = str(tmp_path / name)
    pd.DataFrame(rows).to_csv(fp, index=False)
    return fp


def _row(id_, mosaic_days=20, bands="B04,B08"):
    return {
        "id": id_, "datacube_filepath": f"x/{id_}/datacube.npy",
        "bands": bands, "mosaic_days": mosaic_days,
        "startdate": "2018-01-01", "enddate": "2018-02-01",
        "processing": "",  # spec 59: real `setup` always writes this column
    }


def _patch_flatten(monkeypatch):
    calls = {"n": 0}

    def fake_flatten(*, export_folderpath, **kw):
        calls["n"] += 1
        _write_flatten_outputs(export_folderpath)

    monkeypatch.setattr(api._flatten, "flatten", fake_flatten)
    return calls


# --- AC4: matching stamp skips; arrays still land, TrainingData still works ---

def test_flatten_skips_when_stamp_matches(tmp_path, monkeypatch):
    calls = _patch_flatten(monkeypatch)
    csv_fp = _input_csv(tmp_path, [_row("a")])
    export = str(tmp_path / "export")

    td1 = api.flatten_training_data(csv_fp, export)
    assert calls["n"] == 1

    td2 = api.flatten_training_data(csv_fp, export)
    assert calls["n"] == 1  # AC4: second call skipped

    assert td1.n_pixels == td2.n_pixels
    assert td1.n_timestamps == td2.n_timestamps
    assert td1.bands == td2.bands


# --- AC5: each of these invalidates the stamp -> flatten reruns ---------------

@pytest.mark.parametrize("mutate", [
    "add_cell", "remove_cell", "change_bands", "change_mosaic_days", "change_aggregate",
])
def test_flatten_reruns_on_any_identity_change(tmp_path, monkeypatch, mutate):
    calls = _patch_flatten(monkeypatch)
    export = str(tmp_path / "export")

    csv_fp = _input_csv(tmp_path, [_row("a"), _row("b")], name="input1.csv")
    kwargs = {}
    if mutate == "change_aggregate":
        kwargs["aggregate"] = None  # baseline call carries no aggregate
    api.flatten_training_data(csv_fp, export, **kwargs)
    assert calls["n"] == 1

    if mutate == "add_cell":
        csv_fp2 = _input_csv(tmp_path, [_row("a"), _row("b"), _row("c")], name="input2.csv")
        api.flatten_training_data(csv_fp2, export)
    elif mutate == "remove_cell":
        csv_fp2 = _input_csv(tmp_path, [_row("a")], name="input2.csv")
        api.flatten_training_data(csv_fp2, export)
    elif mutate == "change_bands":
        csv_fp2 = _input_csv(
            tmp_path, [_row("a", bands="B04,B08,B8A"), _row("b", bands="B04,B08,B8A")],
            name="input2.csv",
        )
        api.flatten_training_data(csv_fp2, export)
    elif mutate == "change_mosaic_days":
        csv_fp2 = _input_csv(
            tmp_path, [_row("a", mosaic_days=10), _row("b", mosaic_days=10)], name="input2.csv",
        )
        api.flatten_training_data(csv_fp2, export)
    elif mutate == "change_aggregate":
        api.flatten_training_data(csv_fp, export, aggregate="median_per_id")

    assert calls["n"] == 2  # AC5: reran, not skipped


def test_flatten_rebuild_with_identical_ids_and_params_reruns_when_forced(tmp_path, monkeypatch):
    """A cube REBUILT under the same id/path set is not distinguishable from an unrelated
    call by identity alone (D3 knowingly declines a content digest by default) -- the
    caller's own `overwrite=True` is the honest way to force it, exercised here."""
    calls = _patch_flatten(monkeypatch)
    csv_fp = _input_csv(tmp_path, [_row("a")])
    export = str(tmp_path / "export")

    api.flatten_training_data(csv_fp, export)
    assert calls["n"] == 1
    api.flatten_training_data(csv_fp, export, overwrite=True)
    assert calls["n"] == 2


# --- D6: a missing array under a matching stamp fails towards running --------

def test_flatten_reruns_when_stamp_matches_but_array_missing(tmp_path, monkeypatch):
    calls = _patch_flatten(monkeypatch)
    csv_fp = _input_csv(tmp_path, [_row("a")])
    export = str(tmp_path / "export")

    api.flatten_training_data(csv_fp, export)
    assert calls["n"] == 1
    os.remove(os.path.join(export, "data.npy"))  # simulate a half-cleaned export folder

    api.flatten_training_data(csv_fp, export)
    assert calls["n"] == 2  # fail towards running, not towards trusting a partial skip


def test_flatten_reruns_when_stamp_corrupt(tmp_path, monkeypatch):
    calls = _patch_flatten(monkeypatch)
    csv_fp = _input_csv(tmp_path, [_row("a")])
    export = str(tmp_path / "export")

    api.flatten_training_data(csv_fp, export)
    assert calls["n"] == 1
    fs.write_text(os.path.join(export, api._FLATTEN_STAMP_NAME), "{not json")

    api.flatten_training_data(csv_fp, export)
    assert calls["n"] == 2


# --- AC7/AC8: create_training_data's overwrite= plumbing ----------------------

def _polys(tmp_path, n=1):
    import geopandas as gpd
    import shapely.geometry

    return gpd.GeoDataFrame(
        {"fid": list(range(n)), "crop": ["a"] * n,
         "geometry": [shapely.geometry.box(i, 0, i + 1, 1) for i in range(n)]},
        crs="EPSG:4326",
    )


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return str(path)


@pytest.mark.parametrize("overwrite,expect_build,expect_flatten", [
    (False, False, False),
    (True, True, True),
    ("datacubes", True, True),
    ("flatten", False, True),
])
def test_create_training_data_overwrite_forwarding(
    tmp_path, monkeypatch, overwrite, expect_build, expect_flatten,
):
    import datetime

    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    export = tmp_path / "export"
    seen = {}

    def fake_run_create_datacube(*, csv_filepath, overwrite, **kw):
        seen["build_overwrite"] = overwrite
        pd.DataFrame(
            {"datacube_filepath": ["x/datacube.npy"], "id": [0], "label": ["a"]}
        ).to_csv(csv_filepath, index=False)

    def fake_flatten_training_data(input_csv, export_folderpath, *, overwrite=False, **kw):
        seen["flatten_overwrite"] = overwrite
        fs.makedirs(export_folderpath)
        fs.save_npy(os.path.join(export_folderpath, "data.npy"), np.zeros((1, 1, 1)))
        fs.save_npy(
            os.path.join(export_folderpath, "metadata.pickle.npy"),
            {"timestamps": [0], "bands": ["B04"]}, allow_pickle=True,
        )
        return api.TrainingData(
            export_folderpath=export_folderpath, run_folderpath=os.path.dirname(input_csv),
            n_pixels=1, n_timestamps=1, bands=["B04"],
        )

    monkeypatch.setattr(api._create_datacube, "run_create_datacube", fake_run_create_datacube)
    monkeypatch.setattr(api, "flatten_training_data", fake_flatten_training_data)

    api.create_training_data(
        label_polygons=_polys(tmp_path), catalog_filepath=cat,
        startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
        mosaic_days=20, bands=["B04"], id_col="fid", label_col="crop",
        export_folderpath=str(export), overwrite=overwrite,
    )
    assert seen["build_overwrite"] == expect_build
    assert seen["flatten_overwrite"] == expect_flatten


def test_invalid_overwrite_raises(tmp_path):
    import datetime

    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    with pytest.raises(api.PreflightError, match="overwrite"):
        api.create_training_data(
            label_polygons=_polys(tmp_path), catalog_filepath=cat,
            startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
            mosaic_days=20, bands=["B04"], id_col="fid", label_col="crop",
            export_folderpath=str(tmp_path / "export"), overwrite="bogus",
        )


# --- AC9: skip prints a line naming what was skipped --------------------------

def test_flatten_skip_prints_a_line(tmp_path, monkeypatch, capsys):
    _patch_flatten(monkeypatch)
    csv_fp = _input_csv(tmp_path, [_row("a")])
    export = str(tmp_path / "export")

    api.flatten_training_data(csv_fp, export)
    capsys.readouterr()
    api.flatten_training_data(csv_fp, export)
    assert "[flatten]" in capsys.readouterr().out


# --- AC6: no modification time read anywhere in the flatten-skip logic -------

def test_no_mtime_read_in_flatten_skip_logic():
    src = "".join(inspect.getsource(fn) for fn in (
        api._flatten_identity, api._flatten_outputs_present, api.flatten_training_data,
        stamp_mod.write_stamp, stamp_mod.read_stamp, stamp_mod.matches_stamp,
    ))
    for forbidden in ("getmtime", "st_mtime", "os.stat", ".stat()"):
        assert forbidden not in src


# --- Spec 59 A2 X2: an `input.csv` from before the `processing` column is refused ----

def _setup_shaped_row(id_, cube_fp):
    """A row shaped like `setup`'s output as of spec 58 -- every run-parameter column
    except `processing`."""
    return {
        "id": id_, "datacube_filepath": cube_fp, "bands": "B04,B08", "mosaic_days": 20,
        "startdate": "2018-01-01", "enddate": "2018-02-01",
        "collection": "sentinel-2-l2a", "mosaic_scheme": "median", "properties_filter": "",
    }


def test_flatten_refuses_input_csv_without_processing_column(tmp_path, monkeypatch):
    calls = _patch_flatten(monkeypatch)
    csv_fp = _input_csv(tmp_path, [_setup_shaped_row("a", "x/a/datacube.npy")])
    export = tmp_path / "export"

    with pytest.raises(api.PreflightError) as ei:
        api.flatten_training_data(csv_fp, str(export))

    assert csv_fp in str(ei.value)
    assert "processing" in str(ei.value)
    assert calls["n"] == 0
    assert not (export / api._FLATTEN_STAMP_NAME).exists()
    assert not list(export.glob("*.npy"))


def test_create_training_data_refuses_run_folder_with_old_input_csv(tmp_path, monkeypatch):
    import datetime

    import geopandas as gpd
    from shapely.geometry import box

    from fsd import config

    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    export = tmp_path / "export"
    state = {"csv": None}

    def fake_run_create_datacube(*, csv_filepath, run_folderpath, **kw):
        if fs.exists(csv_filepath):
            return  # every cube already exists: the real build leaves `input.csv` alone
        cube = os.path.join(run_folderpath, "cube0", "datacube.npy")
        fs.makedirs(os.path.dirname(cube))
        fs.save_npy(cube, np.zeros((1, 1, 1)))
        row = _setup_shaped_row(0, cube)
        row["processing"] = ""
        fs.makedirs(os.path.dirname(csv_filepath))
        pd.DataFrame([row]).to_csv(csv_filepath, index=False)
        state["csv"] = csv_filepath

    monkeypatch.setattr(api._create_datacube, "run_create_datacube", fake_run_create_datacube)
    _patch_flatten(monkeypatch)

    gdf = gpd.GeoDataFrame(
        {"fid": [0], "crop": ["a"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326",
    )
    kwargs = dict(
        label_polygons=gdf, catalog_filepath=cat,
        startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
        mosaic_days=20, bands=["B04"], id_col="fid", label_col="crop",
        export_folderpath=str(export), collection=config.SATELLITE_S2L2A,
    )
    api.create_training_data(**kwargs)

    # Simulate a run folder `setup` wrote before the `processing` column existed.
    df = pd.read_csv(state["csv"]).drop(columns=["processing"])
    df.to_csv(state["csv"], index=False)
    # Delete the stamp so the top-level short-circuit (X2.4) cannot answer from it
    # before the flatten leg reads `input.csv`.
    os.remove(export / api._FLATTEN_STAMP_NAME)

    with pytest.raises(api.PreflightError, match="processing"):
        api.create_training_data(**kwargs)


def test_flatten_input_csv_without_run_columns_has_no_processing_param(
    tmp_path, monkeypatch, capsys,
):
    """A hand-built `input.csv` (only `id` + `datacube_filepath`) is not a `setup`
    file: it flattens, and its stamp records no `processing` key."""
    calls = _patch_flatten(monkeypatch)
    csv_fp = _input_csv(tmp_path, [{"id": "a", "datacube_filepath": "x/a/datacube.npy"}])
    export = str(tmp_path / "export")

    api.flatten_training_data(csv_fp, export)
    stamp = stamp_mod.read_stamp(os.path.join(export, api._FLATTEN_STAMP_NAME))
    assert "processing" not in stamp["params"]

    capsys.readouterr()
    api.flatten_training_data(csv_fp, export)
    assert calls["n"] == 1
    assert "[flatten] arrays match" in capsys.readouterr().out
