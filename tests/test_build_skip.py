"""Tests for spec 49 D1/D2/D4: `create_datacube.run_create_datacube` skips the build leg
when the cubes `input.csv` names are already present, dispatches only the shortfall on a
partial re-run, and `overwrite=True` forces a rebuild. Fast + synthetic (no catalog, no
imagery -- these exercise the driver-side shortfall logic, not the builder).
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pandas as pd

from fsd import api
from fsd.storage import fs
from fsd.workflows import create_datacube as cd


def _write_cube(folder, *, empty_metadata=False):
    fs.makedirs(folder)
    fs.save_npy(os.path.join(folder, "datacube.npy"), np.zeros((2, 2, 2, 1)))
    if empty_metadata:
        # a truncated write: present but zero bytes (#74's class of defect, D2).
        with open(os.path.join(folder, "metadata.pickle.npy"), "wb"):
            pass
    else:
        fs.save_npy(os.path.join(folder, "metadata.pickle.npy"), {"a": 1}, allow_pickle=True)


_WINDOW = dict(startdate="2018-01-01", enddate="2018-02-01", bands=["B04"],
               collection="sentinel-2-l2a", mosaic_days=20)


def _cube_dir(run_folder, id_value):
    """The folder THIS request derives for `id_value` (spec 50 D6, plus the 2026-08-21
    fix that makes a row current only if it names that path). These tests used to
    hand-build `run_folder/<id>/`, which a real `setup` has never written -- such a row
    is now purged as stale, so the fixtures use the derived path like everything else."""
    from fsd import collections as _collections
    from fsd import config
    segment = cd.window_folder_segment(
        _WINDOW["startdate"], _WINDOW["enddate"], _WINDOW["mosaic_days"],
        bands=_WINDOW["bands"], mosaic_scheme=config.MOSAIC_SCHEME,
        collection=_WINDOW["collection"],
        declaration=_collections.get(_WINDOW["collection"]),
    )
    return cd.cube_export_folderpath(str(run_folder), segment, id_value)


def _row(run_folder, id_value):
    folder = _cube_dir(run_folder, id_value)
    return {"id": id_value, "export_folderpath": folder,
            "datacube_filepath": os.path.join(folder, "datacube.npy")}


def _write_csv(tmp_path, rows, name="input.csv"):
    csv_fp = str(tmp_path / name)
    pd.DataFrame(rows).to_csv(csv_fp, index=False)
    return csv_fp


def _shapefile_with_ids(tmp_path, ids, name="shapes.geojson"):
    """A real shapefile naming exactly `ids` -- spec 50's `build_shortfall_only` (reached
    via `overwrite_setup_csv=False`, below) enumerates its targets from the REQUEST (D3),
    so it always reads `shapefilepath`; `overwrite_setup_csv=False` no longer means "input
    .csv is already complete, don't look at the shapefile at all" the way it did under
    spec 49's dispatch-only test isolation."""
    import geopandas as gpd
    import shapely.geometry

    fp = str(tmp_path / name)
    gpd.GeoDataFrame(
        {"id": list(ids), "geometry": [shapely.geometry.box(i, 0, i + 1, 1) for i in range(len(ids))]},
        crs="EPSG:4326",
    ).to_file(fp, driver="GeoJSON")
    return fp


def _call(csv_fp, run_folder, *, ids=(), **kw):
    defaults = dict(
        catalog_filepath="unused", timestamp_col="timestamp",
        shapefilepath=_shapefile_with_ids(run_folder.parent, ids) if ids else "unused",
        id_col="id", run_folderpath=str(run_folder), startdate="2018-01-01",
        enddate="2018-02-01", bands=["B04"], mosaic_days=20,
        csv_filepath=csv_fp, label_col=None, cores=1, overwrite_setup_csv=False,
    )
    defaults.update(kw)
    return cd.run_create_datacube(**defaults)


# --- AC1: shortfall 0 -> no job ------------------------------------------------

def test_build_skip_when_all_cubes_present(tmp_path, monkeypatch, capsys):
    run_folder = tmp_path / "run"
    cube_dir = _cube_dir(run_folder, "cellA")
    _write_cube(cube_dir)
    csv_fp = _write_csv(tmp_path, [_row(run_folder, "cellA")])

    called = []
    monkeypatch.setattr(cd.runners, "run_local", lambda *a, **kw: called.append(a))
    monkeypatch.setattr(cd.runners, "run_aml", lambda *a, **kw: called.append(a))

    result = _call(csv_fp, run_folder, ids=["cellA"])
    assert result is None
    assert called == []
    assert "[build] 0 of 1 cubes missing; nothing to build" in capsys.readouterr().out


# --- AC2: partial shortfall dispatches only the missing rows ------------------

def test_build_skip_partial_dispatches_only_missing(tmp_path, monkeypatch, capsys):
    run_folder = tmp_path / "run"
    _write_cube(_cube_dir(run_folder, "present"))
    csv_fp = _write_csv(tmp_path, [_row(run_folder, "present"), _row(run_folder, "missing")])

    seen = {}

    def fake_run_local(dispatch_csv, **kw):
        with fs.open(dispatch_csv, "r") as f:
            seen["csv"] = pd.read_csv(f)
        return None

    monkeypatch.setattr(cd.runners, "run_local", fake_run_local)

    _call(csv_fp, run_folder, ids=["present", "missing"])
    assert list(seen["csv"]["id"]) == ["missing"]
    out = capsys.readouterr().out
    assert "[build] 1 of 2 cubes missing; dispatching 1" in out


def test_build_full_dispatch_when_nothing_present_prints_nothing(tmp_path, monkeypatch, capsys):
    """D1: shortfall == total (today's full-dispatch shape) prints nothing -- mirrors
    spec 47's `_mpc_catalog_shortfall` call site exactly."""
    run_folder = tmp_path / "run"
    csv_fp = _write_csv(tmp_path, [_row(run_folder, "a"), _row(run_folder, "b")])
    seen = {}

    def fake_run_local(dispatch_csv, **kw):
        seen["csv"] = dispatch_csv
        return None

    monkeypatch.setattr(cd.runners, "run_local", fake_run_local)
    _call(csv_fp, run_folder, ids=["a", "b"])
    assert seen["csv"] == csv_fp  # dispatched the ORIGINAL csv, no shortfall temp file
    assert "[build]" not in capsys.readouterr().out


# --- AC3: presence needs BOTH files, non-empty (D2) ---------------------------

def test_cube_present_requires_both_files_non_empty(tmp_path):
    cube_dir = tmp_path / "cube"
    _write_cube(cube_dir)
    assert cd._cube_present(str(cube_dir / "datacube.npy"))

    empty_dir = tmp_path / "empty_meta"
    _write_cube(empty_dir, empty_metadata=True)
    assert not cd._cube_present(str(empty_dir / "datacube.npy"))

    missing_dir = tmp_path / "missing"
    assert not cd._cube_present(str(missing_dir / "datacube.npy"))


# --- AC (D4): overwrite=True forces a rebuild ---------------------------------

def test_overwrite_forces_full_rebuild(tmp_path, monkeypatch):
    run_folder = tmp_path / "run"
    cube_dir = _cube_dir(run_folder, "cellA")
    _write_cube(cube_dir)
    csv_fp = _write_csv(tmp_path, [_row(run_folder, "cellA")])

    dispatched = []

    def fake_run_local(dispatch_csv, **kw):
        dispatched.append(dispatch_csv)
        return None

    monkeypatch.setattr(cd.runners, "run_local", fake_run_local)

    _call(csv_fp, run_folder, ids=["cellA"], overwrite=True)
    assert dispatched == [csv_fp]  # every row dispatched again
    # the stale artifacts were cleared, not merely ignored.
    assert not os.path.exists(os.path.join(cube_dir, "datacube.npy"))
    assert not os.path.exists(os.path.join(cube_dir, "metadata.pickle.npy"))


# --- AC6: no modification time is read anywhere in the build-skip logic ------

def test_no_mtime_read_in_build_skip_logic():
    src = "".join(inspect.getsource(fn) for fn in (
        cd._cube_present, cd._build_shortfall, cd._force_rebuild, cd.run_create_datacube,
        cd.build_shortfall_only, cd._read_known_empty, cd._load_manifest, cd._record_known_empty,
        cd._row_matches_window, cd._row_matches_path, cd._present_cube_ids_at,
        cd._presence_for_paths, cd._cube_present_many,
    ))
    for forbidden in ("getmtime", "st_mtime", "os.stat", ".stat()"):
        assert forbidden not in src


# --- #83: do two identical calls address the same cube paths? -----------------
#
# The build skip compares `input.csv`'s cube paths against what exists, so it can only fire
# if two identical calls NAME the same paths. These characterise where that holds and where
# it does not -- "will this resurface elsewhere?" made executable.

def _polys(n=1):
    import geopandas as gpd
    import shapely.geometry

    return gpd.GeoDataFrame(
        {"fid": list(range(n)), "crop": ["a"] * n,
         "geometry": [shapely.geometry.box(i, 0, i + 1, 1) for i in range(n)]},
        crs="EPSG:4326",
    )


def _run_folderpaths_of_two_calls(tmp_path, monkeypatch, **extra):
    """`run_folderpath` as handed to the build unit, on two identical back-to-back calls."""
    import datetime

    seen = []

    def fake_run_create_datacube(*, csv_filepath, run_folderpath, **kw):
        seen.append(run_folderpath)
        fs.makedirs(os.path.dirname(csv_filepath))
        pd.DataFrame({"datacube_filepath": [f"{run_folderpath}/w/0/datacube.npy"],
                      "id": [0], "label": ["a"]}).to_csv(csv_filepath, index=False)

    def fake_flatten_training_data(input_csv, export_folderpath, **kw):
        fs.makedirs(export_folderpath)
        fs.save_npy(os.path.join(export_folderpath, "data.npy"), np.zeros((1, 1, 1)))
        fs.save_npy(os.path.join(export_folderpath, "metadata.pickle.npy"),
                    {"timestamps": [0], "bands": ["B04"]}, allow_pickle=True)
        return api.TrainingData(
            export_folderpath=export_folderpath, run_folderpath=os.path.dirname(input_csv),
            n_pixels=1, n_timestamps=1, bands=["B04"],
        )

    monkeypatch.setattr(api._create_datacube, "run_create_datacube", fake_run_create_datacube)
    monkeypatch.setattr(api, "flatten_training_data", fake_flatten_training_data)

    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    cat.parent.mkdir(parents=True, exist_ok=True)
    cat.write_text("")
    for _ in range(2):
        api.create_training_data(
            label_polygons=_polys(), catalog_filepath=str(cat),
            startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
            mosaic_days=20, bands=["B04"], id_col="fid", label_col="crop",
            export_folderpath=str(tmp_path / "export"), **extra,
        )
    return seen


def test_local_runner_addresses_the_same_cube_paths_twice(tmp_path, monkeypatch):
    """`runner="local"` was never affected by #83: its run_folderpath is
    `export_folderpath/run`, which carries no clock."""
    a, b = _run_folderpaths_of_two_calls(tmp_path, monkeypatch)
    assert a == b


def test_aml_run_folderpath_no_longer_derived_from_the_clock(tmp_path, monkeypatch):
    """#83, fixed by spec 50 D6. `run_folderpath` used to default to `{root}/runs/{run_id}`
    where run_id was a fresh UTC timestamp, so two calls seconds apart addressed different
    cubes, the shortfall was always N of N, and the build/flatten skips could never fire.
    Flipped deliberately (not deleted) per the spec 50 handoff: the default is now a plain
    stable name ("train"), not a hash of the request (Q1 rejected set-level addressing) and
    not the clock."""
    a, b = _run_folderpaths_of_two_calls(
        tmp_path, monkeypatch,
        runner="aml",
        runner_kwargs={"root": str(tmp_path / "root"), "cluster": "c",
                       "environment": "e:1", "identity_client_id": "i"},
    )
    assert a == b == f"{tmp_path}/root/runs/train"


def test_aml_is_stable_when_run_folderpath_is_given(tmp_path, monkeypatch):
    """The #83 workaround, locked in: an explicit `run_folderpath` makes two identical calls
    address identical cube paths, which is what the build + flatten skips require."""
    pinned = str(tmp_path / "root" / "runs" / "train")
    a, b = _run_folderpaths_of_two_calls(
        tmp_path, monkeypatch,
        runner="aml", run_folderpath=pinned,
        runner_kwargs={"root": str(tmp_path / "root"), "cluster": "c",
                       "environment": "e:1", "identity_client_id": "i"},
    )
    assert a == b == pinned


def test_run_inference_run_folderpath_carries_no_clock():
    """`run_inference` is structurally immune to #83: its run folder is derived from the
    caller's `output_folderpath`, never from a timestamp -- which is why spec 47 D5's
    per-cell output skip works today. Asserted on the source so a refactor that introduces a
    run_id here has to notice."""
    src = inspect.getsource(api._run_inference_roi)
    assert 'os.path.join(output_folderpath, "cells")' in src
    for clock in ("Timestamp.now", "datetime.now", "utcnow"):
        assert clock not in src
