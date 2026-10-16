"""Spec 50 — resolve `create_training_data` backwards from the target.

Step 0 (D6/#83): the deterministic run folder. Step 1 (D3): request-derived flatten
identity. Step 2 (phase 1): the top-level short-circuit. Step 4 (phase 2): the full
backward walk. Step 3 (D9) is blocked on #84 and is NOT implemented here.
"""

from __future__ import annotations

import datetime
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from rise_eo import api, config
from rise_eo import collections as _collections
from rise_eo.storage import fs
from rise_eo.workflows import create_datacube


def _make_catalog(cat_path, tmp_path):
    rows = []
    for i, mgrs in enumerate(["T33UVP", "T33UVP"]):
        tif = tmp_path / f"tile_{i}.tif"
        tif.write_bytes(b"\x00")
        rows.append({
            "id": f"g{i}", "acquisition_key": f"g{i}", "processing_version": None,
            "processing_datetime": None, "source": "mpc",
            "mgrs_tile": mgrs, "timestamp": pd.Timestamp("2018-04-01", tz="UTC"),
            "band": "B04", "filepath": str(tif),
            "geometry": box(0, 0, 10, 10),
        })
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf["area_contribution"] = 1.0
    fs.write_parquet(str(cat_path), gdf)


def _two_shapes(path):
    gdf = gpd.GeoDataFrame(
        {"id": [0, 1], "label": ["a", "b"],
         "geometry": [box(1, 1, 2, 2), box(3, 3, 4, 4)]},
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GeoJSON")


def _shapes(path, ids, *, outside_ids=()):
    """`ids` inside the `_make_catalog` bbox (`box(0, 0, 10, 10)`), except any in
    `outside_ids` -- placed far away so they have no intersecting tile (D5's known-empty
    case)."""
    geoms = []
    for i in ids:
        if i in outside_ids:
            geoms.append(box(1000 + i, 1000, 1001 + i, 1001))
        else:
            geoms.append(box(i + 1, i + 1, i + 2, i + 2))
    gpd.GeoDataFrame(
        {"id": list(ids), "label": [f"l{i}" for i in ids], "geometry": geoms}, crs="EPSG:4326",
    ).to_file(path, driver="GeoJSON")


# --- AC7b: two requests differing only in `bands` resolve to different cube paths -----

def test_params_key_differs_when_bands_differ():
    decl = _collections.get(config.SATELLITE_S2L2A)
    a = create_datacube.params_key(["B04"], config.MOSAIC_SCHEME,
                                    collection=config.SATELLITE_S2L2A, declaration=decl)
    b = create_datacube.params_key(["B04", "B08"], config.MOSAIC_SCHEME,
                                    collection=config.SATELLITE_S2L2A, declaration=decl)
    assert a != b


def test_params_key_stable_for_same_params():
    decl = _collections.get(config.SATELLITE_S2L2A)
    a = create_datacube.params_key(["B04", "B08"], config.MOSAIC_SCHEME,
                                    collection=config.SATELLITE_S2L2A, declaration=decl)
    b = create_datacube.params_key(["B04", "B08"], config.MOSAIC_SCHEME,
                                    collection=config.SATELLITE_S2L2A, declaration=decl)
    assert a == b


def test_params_key_differs_when_collection_differs():
    """Spec 58 AC4: two builds differing only in `collection` resolve to different cube
    paths. The throwaway second collection is registered with a declaration IDENTICAL to
    S2's, so the two `params_key` digests can only differ because `collection` itself is
    folded into the key -- not because the declaration digest happened to differ too.

    Registration is torn down in a `finally`: `rise_eo.collections.REGISTRY` is a global
    in-process dict, and a leaked test-only id would leak into every later test in the
    session (and into `restamp_cli`'s `--declaration` choices, which are a view over it)."""
    import dataclasses

    other_id = "ac4-test-collection"
    s2 = _collections.get(config.SATELLITE_S2L2A)
    _collections.register(other_id, dataclasses.replace(s2), force=True)
    try:
        decl = s2
        a = create_datacube.params_key(["B04"], config.MOSAIC_SCHEME,
                                        collection=config.SATELLITE_S2L2A, declaration=decl)
        b = create_datacube.params_key(["B04"], config.MOSAIC_SCHEME,
                                        collection=other_id, declaration=decl)
        assert a != b
    finally:
        _collections.REGISTRY.pop(other_id, None)


def test_bands_canonicalize_to_the_same_path_as_native_names(tmp_path):
    """Spec 58 AC5: `bands=["B8A"]` and `bands=["nir08"]` against a collection declaring
    that alias resolve to the SAME cube path -- `setup` canonicalizes bands to native
    asset keys before either the digest or `input.csv` ever see them."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _two_shapes(shapes)

    csv_native = tmp_path / "native.csv"
    create_datacube.setup(
        catalog_filepath=str(cat), timestamp_col="timestamp",
        shapefilepath=str(shapes), id_col="id", run_folderpath=str(tmp_path / "run_native"),
        startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
        bands=["B8A"], mosaic_days=20, csv_filepath=str(csv_native), label_col="label",
    )
    csv_alias = tmp_path / "alias.csv"
    create_datacube.setup(
        catalog_filepath=str(cat), timestamp_col="timestamp",
        shapefilepath=str(shapes), id_col="id", run_folderpath=str(tmp_path / "run_alias"),
        startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
        bands=["nir08"], mosaic_days=20, csv_filepath=str(csv_alias), label_col="label",
    )

    native_segment = os.path.basename(os.path.dirname(
        pd.read_csv(csv_native)["export_folderpath"].iloc[0]))
    alias_segment = os.path.basename(os.path.dirname(
        pd.read_csv(csv_alias)["export_folderpath"].iloc[0]))
    assert native_segment == alias_segment
    assert pd.read_csv(csv_native)["bands"].iloc[0] == "B8A"
    assert pd.read_csv(csv_alias)["bands"].iloc[0] == "B8A"  # canonicalized, not "nir08"


def test_setup_with_different_bands_writes_to_different_window_folders(tmp_path):
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _two_shapes(shapes)
    run_folder = tmp_path / "run"

    csv_a = tmp_path / "a.csv"
    create_datacube.setup(
        catalog_filepath=str(cat), timestamp_col="timestamp",
        shapefilepath=str(shapes), id_col="id", run_folderpath=str(run_folder),
        startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
        bands=["B04"], mosaic_days=20,
        csv_filepath=str(csv_a), label_col="label",
    )
    csv_b = tmp_path / "b.csv"
    create_datacube.setup(
        catalog_filepath=str(cat), timestamp_col="timestamp",
        shapefilepath=str(shapes), id_col="id", run_folderpath=str(run_folder),
        startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
        bands=["B04", "B08"], mosaic_days=20,
        csv_filepath=str(csv_b), label_col="label",
    )

    paths_a = set(pd.read_csv(csv_a)["export_folderpath"])
    paths_b = set(pd.read_csv(csv_b)["export_folderpath"])
    assert paths_a.isdisjoint(paths_b), (
        "two requests differing only in bands must resolve to different cube paths (D6)"
    )


# --- Step 1 / D3: the flatten identity is computable from the REQUEST -----------------

def test_request_identity_matches_the_input_csv_identity(tmp_path):
    """D3's load-bearing claim: `_flatten_identity_from_request` (reads nothing) must
    compute the SAME identity `_flatten_identity` computes from a real `input.csv` --
    otherwise a walk that trusts the request-derived one would (mis)match a stamp the
    old, csv-derived one would not."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _two_shapes(shapes)
    run_folder = tmp_path / "run"
    csv = tmp_path / "input.csv"

    kwargs = dict(
        startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
        mosaic_days=20, bands=["B04", "B08"], collection=config.SATELLITE_S2L2A,
    )
    create_datacube.setup(
        catalog_filepath=str(cat), timestamp_col="timestamp",
        shapefilepath=str(shapes), id_col="id", run_folderpath=str(run_folder),
        csv_filepath=str(csv), label_col="label", **kwargs,
    )
    input_df = pd.read_csv(csv)

    from_csv = api._flatten_identity(
        input_df, id_col="id", filepath_col="datacube_filepath",
        adapter=None, feature_sequence=None, aggregate=None,
    )

    gdf = gpd.read_file(shapes)
    from_request = api._flatten_identity_from_request(
        gdf, id_col="id", run_folderpath=str(run_folder),
        mosaic_scheme=config.MOSAIC_SCHEME,
        adapter=None, feature_sequence=None, aggregate=None, **kwargs,
    )

    assert from_request == from_csv


def test_request_identity_reads_no_input_csv(tmp_path):
    """AC4: computed with no `input.csv` on disk at all -- proof the identity no longer
    needs `setup` to have run."""
    assert not (tmp_path / "input.csv").exists()
    gdf = gpd.GeoDataFrame(
        {"id": [0, 1], "geometry": [box(1, 1, 2, 2), box(3, 3, 4, 4)]}, crs="EPSG:4326",
    )
    identity = api._flatten_identity_from_request(
        gdf, id_col="id", run_folderpath=str(tmp_path / "run"),
        startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
        mosaic_days=20, bands=["B04"], collection=config.SATELLITE_S2L2A,
        mosaic_scheme=config.MOSAIC_SCHEME,
        adapter=None, feature_sequence=None, aggregate=None,
    )
    assert not (tmp_path / "input.csv").exists()
    assert len(identity["cubes"]) == 2


# --- Step 2 / phase 1: the top-level short-circuit ------------------------------------

def _fake_run_create_datacube(
    *, csv_filepath, run_folderpath, startdate, enddate, mosaic_days, bands,
    collection=config.SATELLITE_S2L2A, mosaic_scheme=config.MOSAIC_SCHEME, **kw,
):
    """A `run_create_datacube` stand-in that writes an `input.csv` row whose
    `datacube_filepath` is the SAME path `create_datacube.cube_export_folderpath` (and
    therefore `_flatten_identity_from_request`) would derive -- otherwise the two
    identities would legitimately disagree and the short-circuit correctly would not
    fire. Real `setup` is not exercised here; only its output shape is."""
    window_segment = create_datacube.window_folder_segment(
        startdate, enddate, mosaic_days, bands=bands, mosaic_scheme=mosaic_scheme,
        collection=collection, declaration=_collections.get(collection),
    )
    cube_folder = create_datacube.cube_export_folderpath(run_folderpath, window_segment, 0)
    fs.makedirs(os.path.dirname(csv_filepath))
    pd.DataFrame({
        "datacube_filepath": [os.path.join(cube_folder, "datacube.npy")],
        "id": [0], "label": ["a"], "bands": [",".join(bands)], "mosaic_days": [mosaic_days],
        "startdate": [str(pd.to_datetime(startdate, utc=True))],
        "enddate": [str(pd.to_datetime(enddate, utc=True))],
        "collection": [collection],
        "mosaic_scheme": [mosaic_scheme],
        "properties_filter": [""],  # spec 58 D9: real `setup` always writes this column
        "processing": [""],  # spec 59: likewise
    }).to_csv(csv_filepath, index=False)


def _fake_flatten_training_data(
    input_csv, export_folderpath, *, id_col="id", filepath_col="datacube_filepath",
    label_col=None, adapter=None, feature_sequence=None, aggregate=None, **kw,
):
    """A `flatten_training_data` stand-in, patched at this level (not
    `_flatten.flatten`) so it is exercised identically under `runner="local"` and
    `runner="aml"` (AC11) -- the aml branch dispatches through `runners.run_aml_flatten`,
    a code path this fake must not need to know about. Writes the REAL stamp via
    `api._flatten_identity`, so the top-level short-circuit's independently-computed
    `_flatten_identity_from_request` genuinely has something correct to match against."""
    fs.makedirs(export_folderpath)
    fs.save_npy(os.path.join(export_folderpath, "data.npy"), np.zeros((1, 1, 1)))
    fs.save_npy(os.path.join(export_folderpath, "ids.npy"), np.array([0]))
    fs.save_npy(os.path.join(export_folderpath, "coords.npy"), np.zeros((1, 2)))
    fs.save_npy(os.path.join(export_folderpath, "metadata.pickle.npy"),
                {"timestamps": [0], "bands": ["B04"]}, allow_pickle=True)
    if label_col is not None:
        fs.save_npy(os.path.join(export_folderpath, "labels.npy"), np.array(["a"]))

    with fs.open(input_csv, "r") as f:
        input_df = pd.read_csv(f)
    identity = api._flatten_identity(
        input_df, id_col=id_col, filepath_col=filepath_col,
        adapter=adapter, feature_sequence=feature_sequence, aggregate=aggregate,
    )
    api._stamp.write_stamp(os.path.join(export_folderpath, api._FLATTEN_STAMP_NAME), identity)

    return api.TrainingData(
        export_folderpath=export_folderpath, run_folderpath=os.path.dirname(input_csv),
        n_pixels=1, n_timestamps=1, bands=["B04"],
    )


@pytest.mark.parametrize("runner", ["local", "aml"])
def test_top_level_short_circuit_skips_setup_and_catalog(tmp_path, monkeypatch, runner):
    """AC1/AC2/AC11: a fully-resumed `create_training_data` call performs no catalog
    read, no `setup` call, and no dispatch (D2) -- it lands the arrays and returns, off
    the request-derived identity alone, and the returned `TrainingData` matches the
    first run's. Identical under `runner="local"` and `runner="aml"` -- the walk is
    above the runner seam (AC11); only `run_folderpath`'s default differs."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    cat.parent.mkdir(exist_ok=True)
    cat.write_text("")  # exists for call #1's preflight; contents are never real parquet
    export = tmp_path / "export"

    monkeypatch.setattr(api._create_datacube, "run_create_datacube", _fake_run_create_datacube)
    monkeypatch.setattr(api, "flatten_training_data", _fake_flatten_training_data)

    gdf = gpd.GeoDataFrame(
        {"fid": [0], "crop": ["a"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326",
    )
    kwargs = dict(
        label_polygons=gdf, catalog_filepath=str(cat),
        startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
        mosaic_days=20, bands=["B04"], id_col="fid", label_col="crop",
        export_folderpath=str(export), runner=runner,
    )
    if runner == "aml":
        kwargs["runner_kwargs"] = {
            "root": str(tmp_path / "root"), "cluster": "c", "environment": "e:1",
            "identity_client_id": "i",
        }
    td1 = api.create_training_data(**kwargs)

    # Prove call #2 needs neither `setup` nor the catalog: move the catalog file away,
    # and make `run_create_datacube` (the build leg, which is what would call `setup`)
    # and `TileCatalog.filter` (the D13 guardrail's catalog read) both raise if reached.
    cat.rename(tmp_path / "moved.parquet")

    def _raise_build(*a, **kw):
        raise AssertionError("run_create_datacube must not run when the target is CURRENT")

    def _raise_filter(*a, **kw):
        raise AssertionError("the catalog must not be read when the target is CURRENT")

    monkeypatch.setattr(api._create_datacube, "run_create_datacube", _raise_build)
    monkeypatch.setattr(api.TileCatalog, "filter", _raise_filter)

    td2 = api.create_training_data(**kwargs)

    assert td2.n_pixels == td1.n_pixels
    assert td2.n_timestamps == td1.n_timestamps
    assert td2.bands == td1.bands


def test_top_level_short_circuit_prints_plan_and_fetch(tmp_path, monkeypatch, capsys):
    """AC9/D7: the satisfied case prints `[plan] ... CURRENT` then `[fetch] ...`."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    cat.parent.mkdir(exist_ok=True)
    cat.write_text("")
    export = tmp_path / "export"

    monkeypatch.setattr(api._create_datacube, "run_create_datacube", _fake_run_create_datacube)
    monkeypatch.setattr(api, "flatten_training_data", _fake_flatten_training_data)

    gdf = gpd.GeoDataFrame(
        {"fid": [0], "crop": ["a"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326",
    )
    kwargs = dict(
        label_polygons=gdf, catalog_filepath=str(cat),
        startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
        mosaic_days=20, bands=["B04"], id_col="fid", label_col="crop",
        export_folderpath=str(export),
    )
    api.create_training_data(**kwargs)
    capsys.readouterr()
    api.create_training_data(**kwargs)
    out = capsys.readouterr().out
    assert "[plan] target:" in out and "CURRENT" in out
    assert "[fetch] export ->" in out


def test_stale_target_prints_plan_before_any_work(tmp_path, monkeypatch, capsys):
    """AC9/D7, the other branch: the FIRST call (nothing built yet) prints `[plan]
    target: ... -> STALE (...)`, the flatten cube count, and `[plan] will run: ...`
    before `run_create_datacube` is ever invoked."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    cat.parent.mkdir(exist_ok=True)
    cat.write_text("")
    export = tmp_path / "export"

    monkeypatch.setattr(api._create_datacube, "run_create_datacube", _fake_run_create_datacube)
    monkeypatch.setattr(api, "flatten_training_data", _fake_flatten_training_data)

    gdf = gpd.GeoDataFrame(
        {"fid": [0], "crop": ["a"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326",
    )
    api.create_training_data(
        label_polygons=gdf, catalog_filepath=str(cat),
        startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
        mosaic_days=20, bands=["B04"], id_col="fid", label_col="crop",
        export_folderpath=str(export),
    )
    out = capsys.readouterr().out
    assert "[plan] target:" in out and "STALE" in out and "no stamp" in out
    assert "[plan]   flatten: 1 cubes required" in out
    assert "[plan] will run: build -> flatten -> land" in out
    # D7: announced before any work -- the STALE line precedes the will-run line.
    assert out.index("STALE") < out.index("[plan] will run")


# --- Step 4 / phase 2: the full backward walk (D2/D4/D5/D7) ---------------------------

def _walk_kwargs(cat, run_folder, csv_fp, **extra):
    kwargs = dict(
        catalog_filepath=str(cat), timestamp_col="timestamp", id_col="id",
        run_folderpath=str(run_folder),
        startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
        bands=["B04"], mosaic_days=20,
        csv_filepath=csv_fp, label_col="label",
    )
    kwargs.update(extra)
    return kwargs


def _spy_on_setup(monkeypatch):
    """Records the ids `setup` was called with. Reads `shapefilepath` INSIDE the spy --
    `build_shortfall_only` writes it to a `tempfile.TemporaryDirectory()` that is gone by
    the time `setup` returns."""
    calls = []
    orig_setup = create_datacube.setup

    def spy(**kw):
        calls.append(sorted(gpd.read_file(kw["shapefilepath"])["id"]))
        return orig_setup(**kw)

    monkeypatch.setattr(create_datacube, "setup", spy)
    return calls


def test_build_shortfall_only_calls_setup_for_missing_ids_only(tmp_path, monkeypatch):
    """AC3: a partial run calls `setup` only for the missing ids -- scaled down from
    spec 50's own 900/40 example: 3 already have rows, 2 more are requested, `setup`
    receives exactly those 2 shapes."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    _make_catalog(cat, tmp_path)
    run_folder = tmp_path / "run"
    csv_fp = str(run_folder / "input.csv")
    kwargs = _walk_kwargs(cat, run_folder, csv_fp)

    shapes3 = tmp_path / "shapes3.geojson"
    _shapes(shapes3, [0, 1, 2])
    create_datacube.build_shortfall_only(shapefilepath=str(shapes3), **kwargs)

    calls = _spy_on_setup(monkeypatch)
    shapes5 = tmp_path / "shapes5.geojson"
    _shapes(shapes5, [0, 1, 2, 3, 4])
    n_present, n_missing, n_known_empty = create_datacube.build_shortfall_only(
        shapefilepath=str(shapes5), **kwargs,
    )

    assert (n_present, n_missing, n_known_empty) == (3, 2, 0)
    assert calls == [[3, 4]]


_FORWARDED = dict(
    collection="fwd-coll", properties_filter={"s2:foo": "bar"}, processing="latest",
    mosaic_scheme="acquisition",
)


@pytest.mark.parametrize("hop", [
    "run_create_datacube", "run_create_datacube_incremental", "build_shortfall_only",
])
def test_build_legs_forward_collection_filter_processing_scheme_to_setup(
    tmp_path, monkeypatch, hop,
):
    """E3 (#145): `run_create_datacube` (both legs) and `build_shortfall_only` hand
    `setup` the request's `collection`, `properties_filter`, `processing` and
    `mosaic_scheme`."""
    monkeypatch.setitem(_collections.REGISTRY, "fwd-coll",
                        _collections.get(config.SATELLITE_S2L2A))
    shapes = tmp_path / "shapes.geojson"
    _shapes(shapes, [0, 1])
    got = []

    def fake_setup(**kw):
        got.append(kw)
        raise RuntimeError("stop")

    monkeypatch.setattr(create_datacube, "setup", fake_setup)
    kwargs = _walk_kwargs(tmp_path / "c.parquet", tmp_path / "run",
                          str(tmp_path / "run" / "input.csv"),
                          shapefilepath=str(shapes), **_FORWARDED)
    with pytest.raises(RuntimeError, match="stop"):
        if hop == "build_shortfall_only":
            create_datacube.build_shortfall_only(**kwargs)
        else:
            create_datacube.run_create_datacube(
                cores=1, overwrite_setup_csv=(hop == "run_create_datacube"), **kwargs)
    assert {k: got[0][k] for k in _FORWARDED} == _FORWARDED


def test_build_shortfall_only_no_setup_call_when_nothing_missing(tmp_path, monkeypatch):
    """AC5, in spirit: cube targets are enumerated with no catalog access -- `setup` (the
    only catalog reader in this module) is proven never called on a fully-satisfied
    request."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _shapes(shapes, [0, 1])
    run_folder = tmp_path / "run"
    csv_fp = str(run_folder / "input.csv")
    kwargs = _walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes))

    create_datacube.build_shortfall_only(**kwargs)  # first call builds the real rows

    def _raise(*a, **kw):
        raise AssertionError("setup (the only catalog reader here) must not run")

    monkeypatch.setattr(create_datacube, "setup", _raise)
    n_present, n_missing, n_known_empty = create_datacube.build_shortfall_only(**kwargs)
    assert n_missing == 0


def test_adding_one_polygon_rebuilds_exactly_one_cube(tmp_path, monkeypatch):
    """AC7a/D6 Q1: 3 shapes built, then a 4th requested -> the shortfall is 1 and `setup`
    receives exactly 1 shape. No set-level hash appears anywhere: the new id gets its own
    leaf, everything else is untouched."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    _make_catalog(cat, tmp_path)
    run_folder = tmp_path / "run"
    csv_fp = str(run_folder / "input.csv")
    kwargs = _walk_kwargs(cat, run_folder, csv_fp)

    shapes3 = tmp_path / "shapes3.geojson"
    _shapes(shapes3, [0, 1, 2])
    create_datacube.build_shortfall_only(shapefilepath=str(shapes3), **kwargs)
    with fs.open(csv_fp, "r") as f:
        paths_before = set(pd.read_csv(f)["export_folderpath"])

    calls = _spy_on_setup(monkeypatch)
    shapes4 = tmp_path / "shapes4.geojson"
    _shapes(shapes4, [0, 1, 2, 3])
    n_present, n_missing, n_known_empty = create_datacube.build_shortfall_only(
        shapefilepath=str(shapes4), **kwargs,
    )

    assert n_missing == 1
    assert calls == [[3]]

    with fs.open(csv_fp, "r") as f:
        paths_after = set(pd.read_csv(f)["export_folderpath"])
    assert paths_before <= paths_after  # the 3 existing cube paths are untouched
    assert len(paths_after) == 4


# --- Opus review 2026-08-21 (`/tmp/review-fsd-spec-50.md`): F1-F5 -----------------------

def _materialize_cubes(csv_fp):
    """Stand in for the runner: write `datacube.npy` + `metadata.pickle.npy` for every
    row currently in `csv_fp`, so `_cube_present` sees them as built."""
    with fs.open(csv_fp, "r") as f:
        df = pd.read_csv(f)
    for datacube_filepath in df["datacube_filepath"]:
        fs.makedirs(os.path.dirname(datacube_filepath))
        fs.save_npy(datacube_filepath, np.zeros((1, 1, 1, 1, 1)))
        fs.save_npy(os.path.join(os.path.dirname(datacube_filepath), "metadata.pickle.npy"),
                    {"timestamps": [0], "bands": ["B04"]}, allow_pickle=True)


def test_F1_a_rowless_but_built_cube_regenerates_its_row(tmp_path):
    """F1a: cubes built, then `input.csv` lost (e.g. a "force a clean rebuild"
    instinct). A re-run must regenerate the row for every cube that is genuinely on
    disk -- otherwise the id is invisible to everything downstream (`_build_shortfall`,
    `flatten`) and the next `_build_shortfall` call crashes on a missing `input.csv`."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _shapes(shapes, [0, 1])
    run_folder = tmp_path / "run"
    csv_fp = str(run_folder / "input.csv")
    kwargs = _walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes))

    create_datacube.build_shortfall_only(**kwargs)
    _materialize_cubes(csv_fp)
    os.remove(csv_fp)  # rows lost, cubes intact

    create_datacube.build_shortfall_only(**kwargs)
    assert fs.exists(csv_fp), "input.csv was never regenerated for present-but-rowless cubes"
    # No FileNotFoundError, and the (real) cubes are correctly seen as already built.
    _, n_total, n_missing = create_datacube._build_shortfall(csv_fp, force=False)
    assert (n_total, n_missing) == (2, 0)


def test_F1_b_window_purge_never_loses_a_present_windows_rows(tmp_path):
    """F1b: the shared aml run folder (D6) means every request against one root shares
    one `input.csv`. Switching between two windows' worth of requests against that one
    root must not permanently lose the first window's rows once its cubes are built."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    rows = []
    for i, ts in enumerate(["2018-04-01", "2019-04-01"]):
        tif = tmp_path / f"t{i}.tif"
        tif.write_bytes(b"\x00")
        rows.append({"id": f"g{i}", "acquisition_key": f"g{i}", "processing_version": None,
                     "processing_datetime": None, "source": "mpc",
                     "mgrs_tile": "T33UVP", "timestamp": pd.Timestamp(ts, tz="UTC"),
                     "band": "B04", "filepath": str(tif), "geometry": box(0, 0, 10, 10)})
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf["area_contribution"] = 1.0
    fs.write_parquet(str(cat), gdf)
    shapes = tmp_path / "shapes.geojson"
    _shapes(shapes, [0, 1])
    run_folder = tmp_path / "runs" / "train"  # shared across requests, D6
    csv_fp = str(run_folder / "input.csv")
    base = _walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes))
    window_a = dict(base, startdate=datetime.datetime(2018, 1, 1),
                    enddate=datetime.datetime(2018, 7, 1))
    window_b = dict(base, startdate=datetime.datetime(2019, 1, 1),
                    enddate=datetime.datetime(2019, 7, 1))

    create_datacube.build_shortfall_only(**window_a)
    _materialize_cubes(csv_fp)
    create_datacube.build_shortfall_only(**window_b)
    _materialize_cubes(csv_fp)
    create_datacube.build_shortfall_only(**window_a)  # back to window A

    with fs.open(csv_fp, "r") as f:
        assert len(pd.read_csv(f)) == 2, "window A's rows were purged and never restored"


def test_F2_all_out_of_coverage_shortfall_does_not_raise(tmp_path):
    """F2: a shortfall whose shapes ALL lack imagery must not crash the caller --
    `setup` raising `ValueError` is fine when it sees the whole shapefile (one bad
    polygon among hundreds can't trigger it), but `build_shortfall_only` can hand it a
    shortfall that is ENTIRELY out-of-coverage (e.g. one new polygon added to an
    otherwise-complete run)."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    _make_catalog(cat, tmp_path)
    run_folder = tmp_path / "run"
    csv_fp = str(run_folder / "input.csv")

    shapes3 = tmp_path / "shapes3.geojson"
    _shapes(shapes3, [0, 1, 2])
    create_datacube.build_shortfall_only(
        **_walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes3)))

    shapes4 = tmp_path / "shapes4.geojson"
    _shapes(shapes4, [0, 1, 2, 3], outside_ids=[3])  # id 3 is the ENTIRE shortfall
    n_present, n_missing, n_known_empty = create_datacube.build_shortfall_only(
        **_walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes4)))
    assert n_known_empty == 0  # not recorded as known-empty on THIS call yet
    assert n_missing == 1

    # Converges on the next call, same as the mixed-shortfall AC6 case.
    n_present2, n_missing2, n_known_empty2 = create_datacube.build_shortfall_only(
        **_walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes4)))
    assert (n_present2, n_missing2, n_known_empty2) == (3, 0, 1)


def test_F3_plan_missing_count_matches_what_build_shortfall_will_dispatch(tmp_path, capsys):
    """F3/D7/AC9: the printed `[plan] build:` line must agree with what
    `_build_shortfall` (the actual dispatch decision) finds -- an interrupted run (rows
    written, no cubes built yet) must not print `0 missing` while the build leg then
    dispatches every row."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _shapes(shapes, [0, 1, 2])
    run_folder = tmp_path / "run"
    csv_fp = str(run_folder / "input.csv")
    kwargs = _walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes))

    create_datacube.build_shortfall_only(**kwargs)  # 3 rows written, 0 cubes built
    capsys.readouterr()
    create_datacube.build_shortfall_only(**kwargs)
    plan = capsys.readouterr().out
    _, n_total, n_missing = create_datacube._build_shortfall(csv_fp, force=False)

    assert n_missing == 3  # the build leg really does need to dispatch all 3
    assert f"{n_missing} missing" in plan and f"will build {n_missing}" in plan, (
        f"plan said {plan.strip()!r}, but the build leg dispatches {n_missing}/{n_total}"
    )


def test_F4_known_empty_cell_is_excluded_from_request_identity(tmp_path):
    """F4/AC1: once `build_shortfall_only` has recorded a cell as known-empty (D5),
    `_flatten_identity_from_request` must exclude it too -- otherwise the request-side
    identity always names one more id than `input.csv` (which `setup` never wrote a row
    for) and the top-level short-circuit can never match its own stamp, forever."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _shapes(shapes, [0, 1], outside_ids=[1])
    run_folder = tmp_path / "run"
    csv_fp = str(run_folder / "input.csv")
    kwargs = _walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes))

    create_datacube.build_shortfall_only(**kwargs)  # records id 1 as known-empty

    with fs.open(csv_fp, "r") as f:
        from_csv = api._flatten_identity(
            pd.read_csv(f), id_col="id", filepath_col="datacube_filepath",
            adapter=None, feature_sequence=None, aggregate=None,
        )
    from_request = api._flatten_identity_from_request(
        gpd.read_file(shapes), id_col="id", run_folderpath=str(run_folder),
        mosaic_scheme=config.MOSAIC_SCHEME, adapter=None, feature_sequence=None,
        aggregate=None,
        startdate=kwargs["startdate"], enddate=kwargs["enddate"],
        mosaic_days=kwargs["mosaic_days"], bands=kwargs["bands"],
        collection=kwargs.get("collection", config.SATELLITE_S2L2A),
    )
    assert from_request == from_csv
    assert [c[0] for c in from_request["cubes"]] == ["0"]


def test_F5_repeated_identical_call_round_trips_without_purging(tmp_path):
    """F5 (spec 58-adapted): a second call with IDENTICAL params must not purge the
    rows the first call just wrote."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _shapes(shapes, [0, 1])
    run_folder = tmp_path / "run"
    csv_fp = str(run_folder / "input.csv")
    kwargs = _walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes))

    create_datacube.build_shortfall_only(**kwargs)
    n_present, n_missing, n_known_empty = create_datacube.build_shortfall_only(**kwargs)
    assert (n_present, n_missing, n_known_empty) == (2, 0, 0)


def test_known_empty_recorded_once_and_shortfall_converges_to_zero(tmp_path):
    """AC6/D5: a cell with no in-window imagery is recorded once and reported as
    known-empty on the next run; two consecutive identical runs both report a shortfall
    of 0 -- the non-convergence case the manifest exists to prevent."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _shapes(shapes, [0, 1], outside_ids=[1])  # id 1 has no intersecting tile
    run_folder = tmp_path / "run"
    csv_fp = str(run_folder / "input.csv")
    kwargs = _walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes))

    n_present1, n_missing1, n_known_empty1 = create_datacube.build_shortfall_only(**kwargs)
    assert (n_present1, n_missing1, n_known_empty1) == (0, 2, 0)  # neither seen before

    n_present2, n_missing2, n_known_empty2 = create_datacube.build_shortfall_only(**kwargs)
    assert (n_present2, n_missing2, n_known_empty2) == (1, 0, 1)  # id 1 now known-empty

    n_present3, n_missing3, n_known_empty3 = create_datacube.build_shortfall_only(**kwargs)
    assert n_missing3 == 0  # converges: stays 0, never rediscovered


# --- Opus re-review 2026-08-21: two defects introduced by the F2/F4 fixes -------------

def test_duplicate_ids_still_raise_and_are_not_recorded_as_known_empty(tmp_path):
    """F2's catch must be `NoWorkUnitsError`, not a bare `ValueError`: `setup`'s
    duplicate-`id_col` guard raises `ValueError` too, and swallowing it would silently
    record the caller's duplicated shapes as "no imagery" -- turning a deliberate loud
    refusal (added 2026-07-28, after a multi-polygon ROI repeated cell ids) into missing
    training data."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    _make_catalog(cat, tmp_path)
    shapes = tmp_path / "dupes.geojson"
    gpd.GeoDataFrame(
        {"id": [7, 7], "label": ["a", "b"], "geometry": [box(1, 1, 2, 2), box(3, 3, 4, 4)]},
        crs="EPSG:4326",
    ).to_file(shapes, driver="GeoJSON")
    run_folder = tmp_path / "run"
    csv_fp = str(run_folder / "input.csv")

    with pytest.raises(ValueError, match="duplicate"):
        create_datacube.build_shortfall_only(
            **_walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes)))
    assert not create_datacube._read_known_empty(
        str(run_folder),
        create_datacube.window_folder_segment(
            datetime.datetime(2018, 1, 1), datetime.datetime(2019, 1, 1), 20,
            bands=["B04"], mosaic_scheme=config.MOSAIC_SCHEME,
            collection=config.SATELLITE_S2L2A,
            declaration=_collections.get(config.SATELLITE_S2L2A)),
    )


def test_known_empty_is_forgotten_once_the_cell_has_a_row_again(tmp_path):
    """F4's subtraction makes the manifest load-bearing for identity equality, so it
    must not be write-only. A cell recorded known-empty that later gains an `input.csv`
    row must be forgotten -- otherwise `_flatten_identity_from_request` subtracts an id
    `input.csv` genuinely names, the two identities can never agree again, and the
    top-level short-circuit is dead for that request forever: F4 relocated, not fixed.

    The route that regains the row is the forced rebuild (`overwrite="datacubes"`/`True`
    -> the legacy full-`setup` pass), which is D5's documented escape hatch from a stale
    manifest. The scoped walk deliberately does NOT rediscover a known-empty cell -- that
    is D5 working as designed, and D5's own risk note says so."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _shapes(shapes, [0, 1], outside_ids=[1])  # id 1 has no imagery yet
    run_folder = tmp_path / "run"
    csv_fp = str(run_folder / "input.csv")
    kwargs = _walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes))

    create_datacube.build_shortfall_only(**kwargs)
    create_datacube.build_shortfall_only(**kwargs)  # id 1 now recorded known-empty

    _shapes(shapes, [0, 1])  # "re-ingest": id 1 now overlaps the catalog
    create_datacube.run_create_datacube(**kwargs, cores=1, overwrite_setup_csv=True,
                                        dry_run=True)

    from_request = api._flatten_identity_from_request(
        gpd.read_file(shapes), id_col="id", run_folderpath=str(run_folder),
        mosaic_scheme=config.MOSAIC_SCHEME, adapter=None, feature_sequence=None,
        aggregate=None, startdate=kwargs["startdate"], enddate=kwargs["enddate"],
        mosaic_days=kwargs["mosaic_days"], bands=kwargs["bands"],
        collection=kwargs.get("collection", config.SATELLITE_S2L2A),
    )
    with fs.open(csv_fp, "r") as f:
        from_csv = api._flatten_identity(
            pd.read_csv(f), id_col="id", filepath_col="datacube_filepath",
            adapter=None, feature_sequence=None, aggregate=None,
        )
    assert [c[0] for c in from_request["cubes"]] == ["0", "1"]
    assert from_request == from_csv


def test_forced_rebuild_clears_a_stale_known_empty_entry(tmp_path):
    """D5's risk note names `overwrite="datacubes"`/`True` (the legacy full-`setup`
    path) as the escape hatch from a stale manifest. That only holds if the escape hatch
    actually clears it: that pass re-derives every shape from the catalog, so
    `input.csv` becomes the authority and any surviving known-empty record would leave
    the manifest disagreeing with it."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _shapes(shapes, [0, 1], outside_ids=[1])
    run_folder = tmp_path / "run"
    csv_fp = str(run_folder / "input.csv")
    window_segment = create_datacube.window_folder_segment(
        datetime.datetime(2018, 1, 1), datetime.datetime(2019, 1, 1), 20,
        bands=["B04"], mosaic_scheme=config.MOSAIC_SCHEME,
        collection=config.SATELLITE_S2L2A,
        declaration=_collections.get(config.SATELLITE_S2L2A))

    create_datacube.build_shortfall_only(
        **_walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes)))
    create_datacube.build_shortfall_only(
        **_walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes)))
    assert create_datacube._read_known_empty(str(run_folder), window_segment) == {"1"}

    _shapes(shapes, [0, 1])  # id 1 has imagery again
    create_datacube.run_create_datacube(
        **_walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes)),
        cores=1, overwrite_setup_csv=True, dry_run=True,
    )
    assert create_datacube._read_known_empty(str(run_folder), window_segment) == set()


# --- 2026-08-21, first real AML run after D6: stale rows + the silent sweep -----------

def test_a_row_naming_the_old_path_shape_is_purged_and_regenerated(tmp_path, capsys):
    """Spec 50 D6 changed the cube path shape (`<window>` gained a `_<params_key>`
    digest). A row written BEFORE that change matches on every run PARAMETER and still
    names the OLD folder, so matching on params alone adopts it: the plan then announces
    a full rebuild while `_build_shortfall`, reading the row's own stale
    `datacube_filepath`, finds the old cube present and dispatches nothing -- and the
    flatten stamp records paths the request-derived identity can never reproduce, killing
    the short-circuit for good. The path is part of what makes a row current."""
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _shapes(shapes, [0, 1])
    run_folder = tmp_path / "runs" / "train"
    csv_fp = str(run_folder / "input.csv")
    kwargs = _walk_kwargs(cat, run_folder, csv_fp, shapefilepath=str(shapes))

    # A pre-D6 input.csv: identical params, OLD path shape, cubes really there.
    old_rows = []
    for i in (0, 1):
        folder = os.path.abspath(os.path.join(str(run_folder), "20180101_20190101_m20", str(i)))
        fs.makedirs(folder)
        fs.save_npy(os.path.join(folder, "datacube.npy"), np.zeros((1, 1, 1, 1, 1)))
        fs.save_npy(os.path.join(folder, "metadata.pickle.npy"),
                    {"timestamps": [0], "bands": ["B04"]}, allow_pickle=True)
        old_rows.append({
            "id": i, "label": f"l{i}", "export_folderpath": folder,
            "datacube_filepath": os.path.join(folder, "datacube.npy"),
            "startdate": pd.Timestamp("2018-01-01", tz="UTC"),
            "enddate": pd.Timestamp("2019-01-01", tz="UTC"),
            "mosaic_days": 20, "mosaic_scheme": config.MOSAIC_SCHEME,
            "collection": "sentinel-2-l2a", "bands": "B04",
            "added_on": pd.Timestamp.now(tz="UTC"), "images_count": 1,
        })
    fs.makedirs(str(run_folder))
    with fs.open(csv_fp, "w") as f:
        pd.DataFrame(old_rows).to_csv(f, index=False)

    create_datacube.build_shortfall_only(**kwargs)
    plan = capsys.readouterr().out

    window_segment = create_datacube.window_folder_segment(
        kwargs["startdate"], kwargs["enddate"], kwargs["mosaic_days"],
        bands=kwargs["bands"], mosaic_scheme=config.MOSAIC_SCHEME,
        collection=kwargs.get("collection", config.SATELLITE_S2L2A),
        declaration=_collections.get(kwargs.get("collection", config.SATELLITE_S2L2A)))
    with fs.open(csv_fp, "r") as f:
        after = pd.read_csv(f)
    assert len(after) == 2
    for path in after["datacube_filepath"]:
        assert window_segment in path, f"row still names the old path shape: {path}"

    # ...and the announced plan agrees with what the build leg will actually dispatch.
    _, n_total, n_missing = create_datacube._build_shortfall(csv_fp, force=False)
    assert (n_total, n_missing) == (2, 2)
    assert "2 missing" in plan and "will build 2" in plan


def test_presence_is_resolved_by_listing_not_one_path_at_a_time(tmp_path, monkeypatch):
    """The presence sweep must not be a per-path walk: at 900 cells `_cube_present` is
    ~3600 sequential blob round-trips (~20 min over the WAN) with no output, which is
    indistinguishable from a hang. One recursive listing answers it instead, compared in
    memory by the `<id>` path leaf."""
    run_folder = tmp_path / "runs" / "train"
    folders = []
    for i in range(4):
        folder = os.path.join(str(run_folder), "win", str(i))
        fs.makedirs(folder)
        folders.append(folder)
        if i < 3:  # id 3 is deliberately un-built
            fs.save_npy(os.path.join(folder, "datacube.npy"), np.zeros((1, 1, 1, 1, 1)))
            fs.save_npy(os.path.join(folder, "metadata.pickle.npy"), {"a": 1},
                        allow_pickle=True)

    def _boom(*a, **kw):
        raise AssertionError("per-path _cube_present must not be reached when listing works")

    monkeypatch.setattr(create_datacube, "_cube_present", _boom)
    presence = create_datacube._presence_for_paths(
        [os.path.join(f, "datacube.npy") for f in folders], label="plan")
    assert [presence[os.path.join(f, "datacube.npy")] for f in folders] == \
        [True, True, True, False]


def test_presence_falls_back_to_per_path_checks_when_the_folder_is_unlistable(tmp_path):
    """The listing is a fast path, never a new source of truth: nothing built yet (so
    nothing to list) must still resolve, via the concurrent per-path fallback."""
    missing = os.path.join(str(tmp_path), "runs", "train", "win", "0", "datacube.npy")
    assert create_datacube._presence_for_paths([missing], label="plan") == {missing: False}


# --- AC9a: the control file carries the resolved declaration, per WINDOW not per run ---

def test_setup_writes_a_window_scoped_declaration_control_file(tmp_path):
    """Spec 58 AC9 (driver half) + D13: `setup` resolves `collection=` on the driver and
    writes the resolved `CollectionDeclaration` as JSON to a control file every shard
    reads; `input.csv` carries that file's path per row.

    The control file is scoped to the WINDOW segment, not the run-folder root. One run
    folder holds rows from many `setup` calls (`_UNIT_IDENTITY_COLS` carries `collection`
    precisely so different collections coexist in one `input.csv`, and `_build_shortfall`
    dispatches every still-missing row in that file regardless of which call wrote it).
    A single run-root `declaration.json` would be silently overwritten by the second
    `setup` and then read back by the FIRST call's nodes -- the wrong mask/radiometry, at
    a cube path that names a different collection. So: two collections, one run folder,
    two distinct control files that each still hold their own declaration.
    """
    import dataclasses
    import json

    from rise_eo.catalog import declaration as declaration_module

    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    shapes = tmp_path / "shapes.geojson"
    _make_catalog(cat, tmp_path)
    _two_shapes(shapes)

    # A variant registered ONLY here, on the "driver" -- a node must never need the
    # registry to resolve it (ADR 0031); the JSON control file is the whole transport.
    variant_id = "ac9-test-variant"
    s2 = _collections.get(config.SATELLITE_S2L2A)
    variant = dataclasses.replace(s2, mask_keep=True)  # build policy, not an artifact fact
    _collections.register(variant_id, variant, force=True)
    run_folder = tmp_path / "run"
    try:
        paths = {}
        for name, collection in (("s2", config.SATELLITE_S2L2A), ("variant", variant_id)):
            csv = tmp_path / f"{name}.csv"
            create_datacube.setup(
                catalog_filepath=str(cat), timestamp_col="timestamp",
                shapefilepath=str(shapes), id_col="id", run_folderpath=str(run_folder),
                startdate=datetime.datetime(2018, 1, 1), enddate=datetime.datetime(2019, 1, 1),
                bands=["B04"], mosaic_days=20, csv_filepath=str(csv), label_col="label",
                collection=collection,
            )
            df = pd.read_csv(csv)
            assert df["collection"].unique().tolist() == [collection]
            declaration_paths = set(df["declaration_filepath"])
            assert len(declaration_paths) == 1
            paths[name] = declaration_paths.pop()

        # Each control file sits under its OWN window segment, beside the cubes it
        # governs -- not at the run-folder root where the second call would clobber it.
        assert paths["s2"] != paths["variant"]
        for name, collection in (("s2", config.SATELLITE_S2L2A), ("variant", variant_id)):
            df = pd.read_csv(tmp_path / f"{name}.csv")
            window = os.path.basename(os.path.dirname(df["export_folderpath"].iloc[0]))
            assert paths[name] == os.path.join(str(run_folder), window,
                                               create_datacube.DECLARATION_FILENAME)

        # And each still holds its own declaration after BOTH calls have run -- the node
        # reads plain JSON, never `rise_eo.collections`.
        with open(paths["s2"]) as f:
            assert declaration_module.from_json(json.load(f)) == s2
        with open(paths["variant"]) as f:
            assert declaration_module.from_json(json.load(f)) == variant
    finally:
        _collections.REGISTRY.pop(variant_id, None)


# --- P11 (#145 E4/E11): pin the geometry read and the manifest readers ---------------

def _memory_copy(local_path):
    """Copy a local GeoJSON to `memory://` (GDAL cannot read it, so a pin there fails if
    anyone hands the URL to `gpd.read_file`)."""
    import uuid

    url = f"memory://{uuid.uuid4()}/shapes.geojson"
    with open(local_path) as f:
        fs.write_text(url, f.read())
    return url


def test_build_shortfall_only_reads_a_memory_shapefile_like_a_local_one(tmp_path):
    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    _make_catalog(cat, tmp_path)
    shapes = tmp_path / "shapes.geojson"
    _shapes(shapes, [0, 1, 2])
    url = _memory_copy(shapes)

    out = {}
    for tag, shp in (("local", str(shapes)), ("memory", url)):
        run_folder = tmp_path / f"run_{tag}"
        csv_fp = str(run_folder / "input.csv")
        counts = create_datacube.build_shortfall_only(
            shapefilepath=shp, **_walk_kwargs(cat, run_folder, csv_fp))
        out[tag] = (counts, pd.read_csv(csv_fp))
    assert out["local"][0] == out["memory"][0] == (0, 3, 0)
    assert list(out["local"][1].columns) == list(out["memory"][1].columns)
    assert sorted(out["local"][1]["id"]) == sorted(out["memory"][1]["id"]) == [0, 1, 2]


_SEG = "seg"


def _write_manifest(run_folder, text):
    os.makedirs(run_folder, exist_ok=True)
    with open(os.path.join(run_folder, "_manifest.json"), "w") as f:
        f.write(text)


@pytest.mark.parametrize("text", [
    None, "{not json", "[1, 2]", '{"other": {"ids": [1]}}', '{"seg": [1]}', '{"seg": {}}',
])
def test_read_known_empty_is_empty_for_missing_or_unusable_manifests(tmp_path, text):
    run_folder = str(tmp_path / "run")
    os.makedirs(run_folder)
    if text is not None:
        _write_manifest(run_folder, text)
    assert create_datacube._read_known_empty(run_folder, _SEG) == set()


def test_read_known_empty_returns_the_ids_as_strings(tmp_path):
    run_folder = str(tmp_path / "run")
    _write_manifest(run_folder, '{"seg": {"ids": [1, "2"]}}')
    assert create_datacube._read_known_empty(run_folder, _SEG) == {"1", "2"}


@pytest.mark.parametrize("text", [None, "{not json", "[1, 2]"])
def test_load_manifest_is_empty_for_missing_or_unusable_manifests(tmp_path, text):
    run_folder = str(tmp_path / "run")
    os.makedirs(run_folder)
    if text is not None:
        _write_manifest(run_folder, text)
    assert create_datacube._load_manifest(run_folder) == {}


def test_load_manifest_returns_a_dict_manifest_as_is(tmp_path):
    run_folder = str(tmp_path / "run")
    _write_manifest(run_folder, '{"seg": {"ids": [1]}}')
    assert create_datacube._load_manifest(run_folder) == {"seg": {"ids": [1]}}
