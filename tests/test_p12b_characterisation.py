"""Characterisation pins for #145 slice P12b (datacube / catalog / raster tidy-up).

Written against the code as it was BEFORE the tidy-up and never edited afterwards: each
pin must pass on both sides of the refactor. Messages are compared in full (``str(exc)``),
not with ``match=``.
"""

import datetime
import json
import os
import pathlib

import numpy as np
import pandas as pd
import pytest
import rasterio
import rasterio.crs
import rasterio.shutil
import rasterio.transform
import shapely.geometry as sg

from rise_eo import config
from rise_eo.catalog import declaration as declaration_module
from rise_eo.catalog.catalog import COLUMNS, TileCatalog
from rise_eo.catalog.declaration import CollectionDeclaration, MaskSpec
from rise_eo.datacube import ops
from rise_eo.raster import cog, images, rio_env, rio_open

FORMATS = pathlib.Path(__file__).parent / "data" / "formats"
UTM = rasterio.crs.CRS.from_epsg(32643)


# --- B7: load_images ----------------------------------------------------------------


def _write_tif(path, fill):
    data = np.full((1, 6, 6), fill, dtype="uint16")
    profile = {
        "driver": "GTiff", "dtype": "uint16", "nodata": 0, "width": 6, "height": 6,
        "count": 1, "crs": UTM,
        "transform": rasterio.transform.from_origin(0.0, 60.0, 10.0, 10.0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return str(path)


@pytest.mark.parametrize("njobs", [1, 2])
def test_b7_load_images_matches_load_image_per_path(tmp_path, njobs):
    paths = [
        _write_tif(tmp_path / "a.tif", 3),
        str(tmp_path / "missing.tif"),
        _write_tif(tmp_path / "b.tif", 9),
    ]
    got = images.load_images(paths, raise_error=False, njobs=njobs, print_messages=False)
    want = [images.load_image(p, raise_error=False) for p in paths]
    assert len(got) == len(want) == 3
    assert got[1] == (None, None)
    for (g_data, g_profile), (w_data, w_profile) in zip(got, want):
        if w_data is None:
            assert g_data is None and g_profile is None
        else:
            assert np.array_equal(g_data, w_data)
            assert g_profile == w_profile
    assert int(got[0][0][0, 0, 0]) == 3 and int(got[2][0][0, 0, 0]) == 9


def test_b7_load_images_raise_error_true_raises_same_type_as_load_image(tmp_path):
    missing = str(tmp_path / "missing.tif")
    with pytest.raises(Exception) as direct:
        images.load_image(missing, raise_error=True)
    with pytest.raises(Exception) as batch:
        images.load_images([missing], raise_error=True, njobs=1, print_messages=False)
    assert type(batch.value) is type(direct.value)
    assert str(batch.value) == str(direct.value)


def test_b7_load_images_forwards_crop_arguments(tmp_path):
    import geopandas as gpd

    path = _write_tif(tmp_path / "a.tif", 5)
    roi = gpd.GeoDataFrame(geometry=[sg.box(10, 10, 30, 30)], crs=UTM)
    (got,) = images.load_images(
        [path], shapes_gdf=roi, nodata=0, all_touched=False, njobs=1, print_messages=False
    )
    want = images.load_image(path, shapes_gdf=roi, nodata=0, all_touched=False)
    assert np.array_equal(got[0], want[0])
    assert got[1] == want[1]
    assert got[0].shape[1] < 6


# --- B8: declaration.from_json ------------------------------------------------------


def test_b8_golden_v1_equals_literal():
    raw = json.loads((FORMATS / "declaration.v1.json").read_text())
    assert declaration_module.from_json(raw) == CollectionDeclaration(
        reference_band="B08",
        native_grid=False,
        mask_spec=MaskSpec(band="SCL", mask_type="categorical_classes",
                           classes=(0, 1, 3, 8, 9, 10)),
        mask_keep=False,
        nodata=0,
        mosaic_method="median",
    )


def test_b8_golden_v2_equals_literal():
    raw = json.loads((FORMATS / "declaration.v2.json").read_text())
    assert declaration_module.from_json(raw) == CollectionDeclaration(
        reference_band="B08",
        native_grid=True,
        mask_spec=MaskSpec(band="SCL", mask_type="bitmask", classes=(3, 8), bits=(1, 2)),
        mask_keep=True,
        nodata=7,
        mosaic_method="mean",
        scale=0.5,
        radiometry_bands=("B04", "B08"),
        band_aliases=(("red", "B04"), ("nir", "B08")),
        requires_subscription_key=True,
        supports_cloud_cover=True,
        mosaic_partition=("sat:orbit_state",),
        partition_policy="auto",
    )


def test_b8_version_only_gives_the_default():
    assert declaration_module.from_json({"fsd_declaration_version": 2}) == (
        CollectionDeclaration()
    )


def test_b8_null_radiometry_bands_and_mask_spec_are_none():
    decl = declaration_module.from_json(
        {"fsd_declaration_version": 2, "radiometry_bands": None, "mask_spec": None}
    )
    assert decl.radiometry_bands is None
    assert decl.mask_spec is None


def test_b8_band_aliases_keep_json_order():
    raw = {"fsd_declaration_version": 2,
           "band_aliases": {"nir": "B08", "red": "B04", "blue": "B02"}}
    decl = declaration_module.from_json(raw)
    assert decl.band_aliases == (("nir", "B08"), ("red", "B04"), ("blue", "B02"))
    assert isinstance(decl.band_aliases, tuple)


def _from_json_error(raw):
    with pytest.raises(ValueError) as exc:
        declaration_module.from_json(raw)
    return str(exc.value)


def test_b8_error_messages_in_full():
    assert _from_json_error([1, 2]) == (
        "declaration JSON must be a JSON object, got list: [1, 2]."
    )
    assert _from_json_error({}) == (
        "declaration JSON is missing required field 'fsd_declaration_version'."
    )
    assert _from_json_error({"fsd_declaration_version": 3}) == (
        "declaration JSON was written by a newer rise-eo (version 3) than this one supports "
        "(version 2); upgrade rise-eo before reading this catalog."
    )
    assert _from_json_error({"fsd_declaration_version": 2, "bogus": 1}) == (
        "declaration JSON (version 2) has unknown field(s) ['bogus']; known fields are "
        "('reference_band', 'native_grid', 'mask_spec', 'mask_keep', 'nodata', "
        "'mosaic_method', 'scale', 'radiometry_bands', 'band_aliases', "
        "'requires_subscription_key', 'supports_cloud_cover', 'mosaic_partition', "
        "'partition_policy')."
    )


def test_b8_mask_spec_error_wins_over_a_later_bad_field():
    # mask_spec comes before band_aliases in the field order, so its error is the one raised.
    raw = {"fsd_declaration_version": 2, "mask_spec": {}, "band_aliases": "not-an-object"}
    assert _from_json_error(raw) == (
        "declaration.mask_spec is missing required field 'band'."
    )


# --- B9: TileCatalog.append defaults ------------------------------------------------


def _min_row(id="t1", **kw):
    row = dict(
        id=id, collection=config.SATELLITE_S2L2A, timestamp="2024-08-01T10:00:00Z",
        s3url=f"s3://eodata/{id}.SAFE", local_folderpath=f"/data/{id}", files="B02.jp2",
        cloud_cover=0.0, geometry=sg.box(0, 0, 1, 1),
    )
    row.update(kw)
    return row


def test_b9_minimal_row_gets_the_remaining_defaults(tmp_path):
    # offset and nodata are required since spec 59 A3, so they are supplied; the other
    # defaults (scale 1.0 with no declaration, properties, acquisition_key, ...) still fill.
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    cat.append([_min_row(offset=0, nodata=0)])
    got = cat.read()
    assert list(got.columns) == COLUMNS
    assert pd.api.types.is_string_dtype(got["id"])
    assert isinstance(got["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(got["timestamp"].dtype.tz) == "UTC"
    assert got["cloud_cover"].dtype == np.dtype("float64")
    assert got["offset"].dtype == np.dtype("int64")
    assert got["scale"].dtype == np.dtype("float64")
    assert got["nodata"].dtype == np.dtype("int64")
    assert isinstance(got["processing_datetime"].dtype, pd.DatetimeTZDtype)
    assert str(got["processing_datetime"].dtype.tz) == "UTC"
    row = got.iloc[0]
    assert row["offset"] == 0
    assert row["scale"] == 1.0
    assert row["nodata"] == 0
    assert row["properties"] == "{}"
    assert row["acquisition_key"] == "t1"
    assert row["processing_version"] is None
    assert pd.isna(row["processing_datetime"])
    assert row["source"] == ""


def test_b9_supplied_optional_columns_are_kept(tmp_path):
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    cat.append([_min_row(
        offset=-1000, scale=0.0001, nodata=7, properties='{"a": 1}', acquisition_key="k",
        processing_version="05.11", processing_datetime="2024-08-02T00:00:00Z", source="cdse",
    )])
    row = cat.read().iloc[0]
    assert row["offset"] == -1000
    assert row["scale"] == 0.0001
    assert row["nodata"] == 7
    assert row["properties"] == '{"a": 1}'
    assert row["acquisition_key"] == "k"
    assert row["processing_version"] == "05.11"
    assert row["processing_datetime"] == pd.Timestamp("2024-08-02", tz="UTC")
    assert row["source"] == "cdse"


# --- B14: datacube.ops guards and windows -------------------------------------------


def _day(n):
    """2018-01-01 + (n - 1) days, tz-aware UTC."""
    return pd.Timestamp("2018-01-01", tz="UTC") + pd.Timedelta(days=n - 1)


_START = datetime.datetime(2018, 1, 1)
_END = datetime.datetime(2018, 4, 1)
_BOTH = [
    pytest.param(ops._get_mosaic_ts_index_ranges, id="legacy"),
    pytest.param(ops._calendar_windows, id="calendar"),
]


@pytest.mark.parametrize("func", _BOTH)
def test_b14_guard_messages_in_full(func):
    with pytest.raises(ValueError) as exc:
        func([_day(10), _day(5)], _START, _END, 20)
    assert str(exc.value) == "timestamps is not sorted."
    with pytest.raises(ValueError) as exc:
        func([_day(5), _day(10)], datetime.datetime(2018, 1, 8), _END, 20)
    assert str(exc.value) == "startdate must be before or on the first timestamp"
    with pytest.raises(ValueError) as exc:
        func([_day(5), _day(10)], _START, datetime.datetime(2018, 1, 8), 20)
    assert str(exc.value) == "enddate must be after or on the last timestamp"


@pytest.mark.parametrize("func", _BOTH)
def test_b14_guard_order(func):
    # unsorted AND startdate too late AND enddate too early: "not sorted" is raised first.
    with pytest.raises(ValueError) as exc:
        func([_day(10), _day(5)], datetime.datetime(2018, 1, 20),
             datetime.datetime(2018, 1, 2), 20)
    assert str(exc.value) == "timestamps is not sorted."
    # sorted, startdate too late AND enddate too early: startdate is raised first.
    with pytest.raises(ValueError) as exc:
        func([_day(5), _day(10)], datetime.datetime(2018, 1, 20),
             datetime.datetime(2018, 1, 2), 20)
    assert str(exc.value) == "startdate must be before or on the first timestamp"


def test_b14_legacy_gap_opens_one_interval_per_step():
    # day 2, then a gap of two whole windows, then days 64 and 65: the legacy bucketing
    # advances ONE bucket per out-of-window timestamp, so day 65 opens a third interval
    # even though it shares a calendar window with day 64.
    ts = [_day(2), _day(64), _day(65)]
    assert ops._get_mosaic_ts_index_ranges(ts, _START, _END, 20) == [(0, 0), (1, 1), (2, 2)]
    labels, intervals = ops._calendar_windows(ts, _START, _END, 20)
    assert labels == [_day(1), _day(21), _day(41), _day(61), _day(81)]
    assert intervals == [(0, 0), None, None, (1, 2), None]


def test_b14_timestamp_on_a_window_boundary():
    ts = [_day(2), _day(21)]  # day 21 == startdate + 20 days
    assert ops._get_mosaic_ts_index_ranges(ts, _START, _END, 20) == [(0, 1)]
    labels, intervals = ops._calendar_windows(ts, _START, _END, 20)
    assert labels == [_day(1), _day(21), _day(41), _day(61), _day(81)]
    assert intervals == [(0, 0), (1, 1), None, None, None]


def test_b14_timestamp_exactly_at_enddate():
    end = datetime.datetime(2018, 2, 10)  # day 41, the end of the second window
    ts = [_day(2), _day(41)]
    assert ops._get_mosaic_ts_index_ranges(ts, _START, end, 20) == [(0, 0), (1, 1)]
    labels, intervals = ops._calendar_windows(ts, _START, end, 20)
    assert labels == [_day(1), _day(21)]
    assert intervals == [(0, 0), (1, 1)]


# --- B16: rio_env / rio_open build the same env -------------------------------------


class _FakeEnv:
    created: list = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeEnv.created.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeSrc:
    def close(self):
        pass


@pytest.fixture
def fake_rasterio(monkeypatch):
    _FakeEnv.created = []
    opened = []
    monkeypatch.setattr("rise_eo.raster.storage_token", lambda: "TOKEN")
    monkeypatch.setattr(rasterio, "Env", _FakeEnv)
    monkeypatch.setattr(rasterio, "open", lambda *a, **k: opened.append((a, k)) or _FakeSrc())
    return opened


_WITH_ACCOUNT = "abfss://cont@acct.dfs.core.windows.net/a/b.tif"
_NO_ACCOUNT = "az://cont/a/b.tif"
_OPEN_CONFIG = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.jp2",
}


@pytest.mark.parametrize("url, account", [(_WITH_ACCOUNT, "acct"), (_NO_ACCOUNT, None)])
def test_b16_rio_env_and_rio_open_build_the_same_ordered_kwargs(fake_rasterio, url, account):
    rio_env([url])
    rio_open(url)
    env_from_rio_env, env_from_rio_open = _FakeEnv.created
    expected = {"AZURE_STORAGE_ACCESS_TOKEN": "TOKEN", **_OPEN_CONFIG}
    if account is not None:
        expected["AZURE_STORAGE_ACCOUNT"] = account
    assert env_from_rio_env.kwargs == expected
    assert list(env_from_rio_env.kwargs) == list(expected)
    assert list(env_from_rio_open.kwargs) == list(expected)
    assert env_from_rio_open.kwargs == expected


def test_b16_local_paths_build_no_env(fake_rasterio):
    with rio_env(["/tmp/a.tif", "/tmp/b.tif"]):
        pass
    rio_open("/tmp/a.tif")
    assert _FakeEnv.created == []
    assert fake_rasterio == [(("/tmp/a.tif", "r"), {})]


# --- B17: .part cleanup -------------------------------------------------------------


def _fake_copy_writing_part_then_raising(src, dst, **opts):
    with open(dst, "wb") as f:
        f.write(b"partial")
    raise RuntimeError("copy failed")


def test_b17_to_cog_failure_removes_the_part_and_reraises(tmp_path, monkeypatch):
    src = _write_tif(tmp_path / "src.tif", 4)
    dst = str(tmp_path / "out" / "dst.tif")
    monkeypatch.setattr(rasterio.shutil, "copy", _fake_copy_writing_part_then_raising)
    with pytest.raises(RuntimeError) as exc:
        cog.to_cog(src, dst)
    assert str(exc.value) == "copy failed"
    assert not os.path.exists(dst + ".part")
    assert not os.path.exists(dst)


def test_b17_to_cog_still_reraises_when_the_cleanup_itself_fails(tmp_path, monkeypatch):
    src = _write_tif(tmp_path / "src.tif", 4)
    dst = str(tmp_path / "out" / "dst.tif")
    monkeypatch.setattr(rasterio.shutil, "copy", _fake_copy_writing_part_then_raising)

    def failing_remove(path):
        raise OSError("cannot remove")

    monkeypatch.setattr(cog.os, "remove", failing_remove)
    with pytest.raises(RuntimeError) as exc:
        cog.to_cog(src, dst)
    assert str(exc.value) == "copy failed"


def test_b17_stamp_or_reencode_removes_its_part_when_the_replace_fails(tmp_path, monkeypatch):
    path = _write_tif(tmp_path / "x.tif", 4)
    calls = []

    def stamp_fails_once(filepath, **kwargs):
        calls.append(filepath)
        raise RuntimeError("stamp failed")

    def failing_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr(cog, "stamp_gdal_tags", stamp_fails_once)
    monkeypatch.setattr(cog.os, "replace", failing_replace)
    with pytest.raises(OSError) as exc:
        cog.stamp_or_reencode(path)
    assert str(exc.value) == "replace failed"
    assert calls == [path]
    assert not os.path.exists(path + ".reencode.part")
