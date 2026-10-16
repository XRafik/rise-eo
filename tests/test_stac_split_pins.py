"""Characterisation pins for `rise_eo.catalog.stac`'s two item builders (refactor audit #145, P14a).

Literals were computed once from the code BEFORE `tile_catalog_to_items` and
`cog_outputs_to_items` were split into helpers, and are never regenerated: the split must
leave every Item's `to_dict()` (key order included), every error message and every exception
type/cause exactly as they were.
"""

from __future__ import annotations

import datetime as dt
import json
import os

import geopandas as gpd
import pandas as pd
import pytest
import shapely.geometry

from rise_eo.catalog import stac
from rise_eo.catalog.declaration import CollectionDeclaration, MaskSpec


# Copied from tests/test_catalog_stac.py: CI runs bare `pytest`, which cannot import `tests.*`.
def _make_output_cog(folder, *, epsg=32637):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    os.makedirs(folder, exist_ok=True)
    fp = os.path.join(folder, "output.tif")
    data = np.zeros((1, 4, 4), dtype="uint8")
    with rasterio.open(
        fp, "w", driver="GTiff", height=4, width=4, count=1, dtype="uint8",
        crs=f"EPSG:{epsg}", transform=from_origin(500000, 4000000, 10, 10),
    ) as dst:
        dst.write(data)
    return fp


_SLANTED_POLYGON = shapely.geometry.Polygon(
    [(14.766, 48.492), (14.789, 48.534), (14.847, 48.526), (14.825, 48.484)]
)


def _write_geometry_geojson(folder, *, feature_id="cell-1", geom=_SLANTED_POLYGON):
    fp = os.path.join(folder, "geometry.geojson")
    fc = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"id": feature_id},
            "geometry": shapely.geometry.mapping(geom),
        }],
    }
    with open(fp, "w") as f:
        json.dump(fc, f)
    return fp


_DT = dt.datetime(2020, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)

# A None in a float column of a GeoDataFrame is NaN, and the builder passes a NaN offset /
# scale straight through to `raster:bands` today; the literals below pin that as-is.
_NAN = float("nan")


def _dump(items, tmp=None):
    """Item dicts as a JSON-round-tripped list (tuples -> lists), tmp_path masked."""
    s = json.dumps([it.to_dict() for it in items])
    if tmp is not None:
        s = s.replace(str(tmp), "<tmp>")
    return json.loads(s)


def _same(actual, expected):
    # json.dumps keeps dict key order, so this pins `assets` / property order too.
    assert json.dumps(actual) == json.dumps(expected)


# --- tile_catalog_to_items ---------------------------------------------------

_MGRS_A = "S2B_MSIL2A_20181231T080329_N0500_R035_T37PBP_20230726T205809"
_NO_MGRS = "custom_granule_1"
_MGRS_C = "S2A_MSIL2A_20180601T075611_N0500_R035_T36GZU_20230101T000000"


def _gdf_full():
    rows = [
        {   # both processing fields set; every file kind; offset set, scale 0, nodata None
            "id": _MGRS_A, "collection": "sentinel-2-l2a",
            "timestamp": pd.Timestamp("2018-12-31T08:03:29", tz="UTC"),
            "processing_version": "N0500",
            "processing_datetime": pd.Timestamp("2023-07-26T22:58:09+02:00"),
            "s3url": "s3://eodata/A.SAFE", "local_folderpath": "/data/A",
            "files": "B04.tif,,SCL.tif,B08.tif,B8A.jp2,MTD_TL.xml,notes.txt,AOT.tiff",
            "cloud_cover": 1.5, "offset": -1000, "scale": 0, "nodata": None,
            "geometry": shapely.geometry.box(36.6, 12.6, 37.0, 13.0),
        },
        {   # no MGRS in id; only the version set; offset None, scale None; empty s3url
            "id": _NO_MGRS, "collection": "other-coll",
            "timestamp": pd.Timestamp("2018-06-01T07:56:11", tz="UTC"),
            "processing_version": "N0212", "processing_datetime": None,
            "s3url": "", "local_folderpath": "/data/B",
            "files": "B04.jp2,MTD.xml,AOT.foo",
            "cloud_cover": 0.0, "offset": None, "scale": None, "nodata": 7,
            "geometry": shapely.geometry.box(35.6, 11.6, 36.0, 12.0),
        },
        {   # only the datetime set; scale given; offset 0
            "id": _MGRS_C, "collection": "sentinel-2-l2a",
            "timestamp": pd.Timestamp("2018-06-01T07:56:11", tz="UTC"),
            "processing_version": None,
            "processing_datetime": pd.Timestamp("2023-01-01T00:00:00", tz="UTC"),
            "s3url": "s3://eodata/C.SAFE", "local_folderpath": "/data/C",
            "files": "B04.tif,B08.tif,SCL.tif",
            "cloud_cover": 20.25, "offset": 0, "scale": 0.5, "nodata": 0,
            "geometry": shapely.geometry.box(10.0, 10.0, 11.0, 11.0),
        },
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _gdf_minimal():
    # No offset / scale / nodata / s3url columns; both processing fields NaN.
    rows = [
        {
            "id": _MGRS_A, "collection": "sentinel-2-l2a",
            "timestamp": pd.Timestamp("2018-12-31T08:03:29", tz="UTC"),
            "processing_version": float("nan"), "processing_datetime": pd.NaT,
            "local_folderpath": "/data/M", "files": "B04.tif,SCL.tif,B08.tif",
            "cloud_cover": 3.0,
            "geometry": shapely.geometry.box(36.6, 12.6, 37.0, 13.0),
        },
        {
            "id": _NO_MGRS, "collection": "sentinel-2-l2a",
            "timestamp": pd.Timestamp("2018-12-31T08:03:29", tz="UTC"),
            "processing_version": None, "processing_datetime": None,
            "local_folderpath": "/data/N", "files": "B04.tif",
            "cloud_cover": 4.0,
            "geometry": shapely.geometry.box(1.0, 2.0, 3.0, 4.0),
        },
    ]
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _gdf_nan_cols():
    # Columns present but empty (None / NaN), like a catalog read back from parquet.
    gdf = _gdf_full().iloc[:1].copy()
    gdf["offset"] = float("nan")
    gdf["nodata"] = float("nan")
    gdf["scale"] = float("nan")
    return gdf


_NON_S2 = CollectionDeclaration(
    reference_band="B2",
    mask_spec=MaskSpec(band="QA", mask_type="categorical_classes", classes=(1, 2)),
    scale=0.25,
    radiometry_bands=("B2", "B4"),
)


def _nonS2_gdf():
    gdf = _gdf_full().iloc[:2].copy()
    gdf["files"] = ["B2.tif,B3.tif,B4.tif,QA.tif,B2.xml", "B2.jp2,QA.jp2,B9.tif"]
    return gdf


_TILE_CASES = {
    "full_default": lambda: stac.tile_catalog_to_items(_gdf_full()),
    "full_collection_id": lambda: stac.tile_catalog_to_items(_gdf_full(), collection_id="forced"),
    "minimal": lambda: stac.tile_catalog_to_items(_gdf_minimal()),
    "minimal_collection_id": lambda: stac.tile_catalog_to_items(_gdf_minimal(), collection_id="c"),
    "nan_cols": lambda: stac.tile_catalog_to_items(_gdf_nan_cols()),
    "non_s2_declaration": lambda: stac.tile_catalog_to_items(_nonS2_gdf(), declaration=_NON_S2),
}


@pytest.mark.parametrize("name", list(_TILE_CASES))
def test_tile_catalog_to_items_pinned(name):
    _same(_dump(_TILE_CASES[name]()), TILE_EXPECTED[name])


def test_tile_catalog_to_items_read_proj_pinned(monkeypatch):
    calls = []

    def spy(href):
        calls.append(href)
        return {"shape": [2, 3], "transform": [10.0, 0.0, 5.0, 0.0, -10.0, 6.0]}

    monkeypatch.setattr(stac, "_read_proj_fields", spy)
    items = stac.tile_catalog_to_items(_gdf_full(), read_proj=True)
    assert calls == READ_PROJ_CALLS
    _same(_dump(items), READ_PROJ_EXPECTED)


# --- cog_outputs_to_items ----------------------------------------------------

def _two_cogs(tmp):
    fp1 = _make_output_cog(str(tmp / "cell-1"))
    fp2 = _make_output_cog(str(tmp / "cell-2"), epsg=32636)
    gj1 = _write_geometry_geojson(str(tmp / "cell-1"), feature_id="cell-1")
    gj2 = _write_geometry_geojson(
        str(tmp / "cell-2"), feature_id="cell-2",
        geom=shapely.geometry.box(14.0, 48.0, 14.5, 48.5),
    )
    return fp1, fp2, gj1, gj2


def _allow_path_fps(monkeypatch):
    # `rio_open` itself takes only str, so a pathlib.Path fp can reach the footprint lookup
    # only through a wrapper; `cog_outputs_to_items` imports `rio_open` on each call, so this applies.
    import rise_eo.raster

    real = rise_eo.raster.rio_open
    monkeypatch.setattr(rise_eo.raster, "rio_open", lambda p, *a, **k: real(str(p), *a, **k))


def _cog_ok(name, tmp):
    fp1, fp2, gj1, gj2 = _two_cogs(tmp)
    cogs = [fp1, fp2]
    if name == "none_nobands":
        items = stac.cog_outputs_to_items(cogs, dt=_DT)
    elif name == "none_bands_collection":
        items = stac.cog_outputs_to_items(
            cogs, band_names=["a", "b"], collection_id="c2", dt=_DT)
    elif name == "path_form":
        items = stac.cog_outputs_to_items(
            cogs, geometries={fp1: gj1, fp2: gj2}, band_names=["x"], dt=_DT)
    elif name == "shapely_form":
        items = stac.cog_outputs_to_items(
            cogs, geometries={fp1: shapely.geometry.box(14.0, 48.0, 14.1, 48.1),
                              fp2: shapely.geometry.box(15.0, 49.0, 15.2, 49.4)}, dt=_DT)
    elif name == "feature_without_id":
        fc = {"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": {},
            "geometry": shapely.geometry.mapping(shapely.geometry.box(14.0, 48.0, 14.1, 48.1)),
        }]}
        gj = os.path.join(str(tmp / "cell-1"), "noid.geojson")
        with open(gj, "w") as f:
            json.dump(fc, f)
        items = stac.cog_outputs_to_items([fp1], geometries={fp1: gj}, dt=_DT)
    elif name == "feature_null_properties":
        fc = {"type": "FeatureCollection", "features": [{
            "type": "Feature", "properties": None,
            "geometry": shapely.geometry.mapping(shapely.geometry.box(14.0, 48.0, 14.1, 48.1)),
        }]}
        gj = os.path.join(str(tmp / "cell-1"), "nullprops.geojson")
        with open(gj, "w") as f:
            json.dump(fc, f)
        items = stac.cog_outputs_to_items([fp1], geometries={fp1: gj}, dt=_DT)
    elif name == "str_key_wins_over_path_key":
        import pathlib

        fpp = pathlib.Path(fp1)
        items = stac.cog_outputs_to_items(
            [fpp], dt=_DT, geometries={
                str(fpp): shapely.geometry.box(14.0, 48.0, 14.1, 48.1),
                fpp: shapely.geometry.box(20.0, 50.0, 20.5, 50.5),
            })
    elif name == "path_key_when_no_str_key":
        import pathlib

        fpp = pathlib.Path(fp1)
        items = stac.cog_outputs_to_items(
            [fpp], dt=_DT, geometries={fpp: shapely.geometry.box(20.0, 50.0, 20.5, 50.5)})
    else:
        raise AssertionError(name)
    return _dump(items, tmp)


_COG_OK_CASES = [
    "none_nobands", "none_bands_collection", "path_form", "shapely_form",
    "feature_without_id", "feature_null_properties", "str_key_wins_over_path_key",
    "path_key_when_no_str_key",
]


@pytest.mark.parametrize("name", _COG_OK_CASES)
def test_cog_outputs_to_items_pinned(name, tmp_path, monkeypatch):
    _allow_path_fps(monkeypatch)
    _same(_cog_ok(name, tmp_path), COG_EXPECTED[name])


def _write_text(path, text):
    with open(path, "w") as f:
        f.write(text)
    return path


def _cog_err(name, tmp):
    """Run a failing case; return (exception type name, str(exc) with tmp masked, cause type name)."""
    fp1, fp2, gj1, gj2 = _two_cogs(tmp)
    try:
        if name == "no_entry":
            stac.cog_outputs_to_items([fp1], geometries={}, dt=_DT)
        elif name == "empty_shapely":
            stac.cog_outputs_to_items([fp1], geometries={fp1: shapely.geometry.Polygon()}, dt=_DT)
        elif name == "path_missing":
            stac.cog_outputs_to_items(
                [fp1], geometries={fp1: str(tmp / "nope" / "geometry.geojson")}, dt=_DT)
        elif name == "path_invalid_json":
            bad = _write_text(str(tmp / "bad.geojson"), "{not json")
            stac.cog_outputs_to_items([fp1], geometries={fp1: bad}, dt=_DT)
        elif name == "no_features":
            empty = _write_text(
                str(tmp / "empty.geojson"), json.dumps({"type": "FeatureCollection", "features": []}))
            stac.cog_outputs_to_items([fp1], geometries={fp1: empty}, dt=_DT)
        elif name == "id_mismatch":
            wrong = _write_geometry_geojson(str(tmp / "cell-2"), feature_id="other")
            stac.cog_outputs_to_items([fp1], geometries={fp1: wrong}, dt=_DT)
        elif name == "duplicate_ids":
            other = _make_output_cog(str(tmp / "elsewhere" / "cell-1"))
            stac.cog_outputs_to_items([fp1, other], dt=_DT)
        elif name == "missing_cog_and_no_entry":
            stac.cog_outputs_to_items([str(tmp / "ghost" / "output.tif")], geometries={}, dt=_DT)
        elif name == "second_entry_missing":
            stac.cog_outputs_to_items([fp1, fp2], geometries={fp1: gj1}, dt=_DT)
        else:
            raise AssertionError(name)
    except Exception as exc:
        cause = exc.__cause__
        return (type(exc).__name__, str(exc).replace(str(tmp), "<tmp>"),
                None if cause is None else type(cause).__name__)
    raise AssertionError(f"{name}: did not raise")


_COG_ERR_CASES = [
    "no_entry", "empty_shapely", "path_missing", "path_invalid_json", "no_features",
    "id_mismatch", "duplicate_ids", "missing_cog_and_no_entry", "second_entry_missing",
]


@pytest.mark.parametrize("name", _COG_ERR_CASES)
def test_cog_outputs_to_items_errors_pinned(name, tmp_path):
    assert _cog_err(name, tmp_path) == COG_ERR_EXPECTED[name]


def test_footprint_read_goes_through_module_global(tmp_path, monkeypatch):
    fp1, _fp2, gj1, _gj2 = _two_cogs(tmp_path)
    seen = []

    def fake(path):
        seen.append(path)
        return shapely.geometry.box(1, 2, 3, 4), "cell-1"

    monkeypatch.setattr(stac, "_read_footprint_geometry", fake)
    items = stac.cog_outputs_to_items([fp1], geometries={fp1: gj1}, dt=_DT)
    assert seen == [gj1]
    assert items[0].bbox == [1.0, 2.0, 3.0, 4.0]


# --- literals (computed once from the unchanged code) --------------------

TILE_EXPECTED = {'full_default': [{'type': 'Feature',
                   'stac_version': '1.1.0',
                   'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                       'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                                       'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
                                       'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
                                       'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
                   'id': 'S2B_MSIL2A_20181231T080329_N0500_R035_T37PBP_20230726T205809',
                   'geometry': {'type': 'Polygon',
                                'coordinates': [[[37.0, 12.6],
                                                 [37.0, 13.0],
                                                 [36.6, 13.0],
                                                 [36.6, 12.6],
                                                 [37.0, 12.6]]]},
                   'bbox': [36.6, 12.6, 37.0, 13.0],
                   'properties': {'eo:cloud_cover': 1.5,
                                  'processing:version': 'N0500',
                                  'processing:datetime': '2023-07-26T20:58:09Z',
                                  'proj:code': 'EPSG:32637',
                                  'grid:code': 'MGRS-37PBP',
                                  'datetime': '2018-12-31T08:03:29Z'},
                   'links': [{'rel': 'via', 'href': 's3://eodata/A.SAFE'}],
                   'assets': {'B04': {'href': '/data/A/B04.tif',
                                      'type': 'image/tiff; application=geotiff; '
                                              'profile=cloud-optimized',
                                      'title': 'B04',
                                      'eo:bands': [{'name': 'B04'}],
                                      'raster:bands': [{'nodata': _NAN,
                                                        'scale': 0.0001,
                                                        'offset': -0.1}],
                                      'roles': ['data', 'reflectance']},
                              'SCL': {'href': '/data/A/SCL.tif',
                                      'type': 'image/tiff; application=geotiff; '
                                              'profile=cloud-optimized',
                                      'title': 'SCL',
                                      'eo:bands': [{'name': 'SCL'}],
                                      'raster:bands': [{'nodata': _NAN, 'scale': 1, 'offset': 0}],
                                      'roles': ['data', 'mask']},
                              'B08': {'href': '/data/A/B08.tif',
                                      'type': 'image/tiff; application=geotiff; '
                                              'profile=cloud-optimized',
                                      'title': 'B08',
                                      'eo:bands': [{'name': 'B08'}],
                                      'raster:bands': [{'nodata': _NAN,
                                                        'scale': 0.0001,
                                                        'offset': -0.1}],
                                      'roles': ['data', 'reference']},
                              'B8A': {'href': '/data/A/B8A.jp2',
                                      'type': 'image/jp2',
                                      'title': 'B8A',
                                      'eo:bands': [{'name': 'B8A'}],
                                      'raster:bands': [{'nodata': _NAN,
                                                        'scale': 0.0001,
                                                        'offset': -0.1}],
                                      'roles': ['data', 'reflectance']},
                              'MTD_TL': {'href': '/data/A/MTD_TL.xml',
                                         'type': 'application/xml',
                                         'title': 'MTD_TL',
                                         'roles': ['metadata']},
                              'notes': {'href': '/data/A/notes.txt',
                                        'title': 'notes',
                                        'eo:bands': [{'name': 'notes'}],
                                        'roles': ['data']},
                              'AOT': {'href': '/data/A/AOT.tiff',
                                      'type': 'image/tiff; application=geotiff; '
                                              'profile=cloud-optimized',
                                      'title': 'AOT',
                                      'eo:bands': [{'name': 'AOT'}],
                                      'raster:bands': [{'nodata': _NAN, 'scale': 1, 'offset': 0}],
                                      'roles': ['data']}},
                   'collection': 'sentinel-2-l2a'},
                  {'type': 'Feature',
                   'stac_version': '1.1.0',
                   'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                       'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                                       'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
                   'id': 'custom_granule_1',
                   'geometry': {'type': 'Polygon',
                                'coordinates': [[[36.0, 11.6],
                                                 [36.0, 12.0],
                                                 [35.6, 12.0],
                                                 [35.6, 11.6],
                                                 [36.0, 11.6]]]},
                   'bbox': [35.6, 11.6, 36.0, 12.0],
                   'properties': {'eo:cloud_cover': 0.0,
                                  'processing:version': 'N0212',
                                  'datetime': '2018-06-01T07:56:11Z'},
                   'links': [],
                   'assets': {'B04': {'href': '/data/B/B04.jp2',
                                      'type': 'image/jp2',
                                      'title': 'B04',
                                      'eo:bands': [{'name': 'B04'}],
                                      'raster:bands': [{'nodata': 7.0,
                                                        'scale': _NAN,
                                                        'offset': _NAN}],
                                      'roles': ['data', 'reflectance']},
                              'MTD': {'href': '/data/B/MTD.xml',
                                      'type': 'application/xml',
                                      'title': 'MTD',
                                      'roles': ['metadata']},
                              'AOT': {'href': '/data/B/AOT.foo',
                                      'title': 'AOT',
                                      'eo:bands': [{'name': 'AOT'}],
                                      'roles': ['data']}},
                   'collection': 'other-coll'},
                  {'type': 'Feature',
                   'stac_version': '1.1.0',
                   'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                       'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                                       'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
                                       'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
                                       'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
                   'id': 'S2A_MSIL2A_20180601T075611_N0500_R035_T36GZU_20230101T000000',
                   'geometry': {'type': 'Polygon',
                                'coordinates': [[[11.0, 10.0],
                                                 [11.0, 11.0],
                                                 [10.0, 11.0],
                                                 [10.0, 10.0],
                                                 [11.0, 10.0]]]},
                   'bbox': [10.0, 10.0, 11.0, 11.0],
                   'properties': {'eo:cloud_cover': 20.25,
                                  'processing:datetime': '2023-01-01T00:00:00Z',
                                  'proj:code': 'EPSG:32736',
                                  'grid:code': 'MGRS-36GZU',
                                  'datetime': '2018-06-01T07:56:11Z'},
                   'links': [{'rel': 'via', 'href': 's3://eodata/C.SAFE'}],
                   'assets': {'B04': {'href': '/data/C/B04.tif',
                                      'type': 'image/tiff; application=geotiff; '
                                              'profile=cloud-optimized',
                                      'title': 'B04',
                                      'eo:bands': [{'name': 'B04'}],
                                      'raster:bands': [{'nodata': 0.0,
                                                        'scale': 0.5,
                                                        'offset': 0.0}],
                                      'roles': ['data', 'reflectance']},
                              'B08': {'href': '/data/C/B08.tif',
                                      'type': 'image/tiff; application=geotiff; '
                                              'profile=cloud-optimized',
                                      'title': 'B08',
                                      'eo:bands': [{'name': 'B08'}],
                                      'raster:bands': [{'nodata': 0.0,
                                                        'scale': 0.5,
                                                        'offset': 0.0}],
                                      'roles': ['data', 'reference']},
                              'SCL': {'href': '/data/C/SCL.tif',
                                      'type': 'image/tiff; application=geotiff; '
                                              'profile=cloud-optimized',
                                      'title': 'SCL',
                                      'eo:bands': [{'name': 'SCL'}],
                                      'raster:bands': [{'nodata': 0.0, 'scale': 1, 'offset': 0}],
                                      'roles': ['data', 'mask']}},
                   'collection': 'sentinel-2-l2a'}],
 'full_collection_id': [{'type': 'Feature',
                         'stac_version': '1.1.0',
                         'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                             'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                                             'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
                                             'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
                                             'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
                         'id': 'S2B_MSIL2A_20181231T080329_N0500_R035_T37PBP_20230726T205809',
                         'geometry': {'type': 'Polygon',
                                      'coordinates': [[[37.0, 12.6],
                                                       [37.0, 13.0],
                                                       [36.6, 13.0],
                                                       [36.6, 12.6],
                                                       [37.0, 12.6]]]},
                         'bbox': [36.6, 12.6, 37.0, 13.0],
                         'properties': {'eo:cloud_cover': 1.5,
                                        'processing:version': 'N0500',
                                        'processing:datetime': '2023-07-26T20:58:09Z',
                                        'proj:code': 'EPSG:32637',
                                        'grid:code': 'MGRS-37PBP',
                                        'datetime': '2018-12-31T08:03:29Z'},
                         'links': [{'rel': 'via', 'href': 's3://eodata/A.SAFE'}],
                         'assets': {'B04': {'href': '/data/A/B04.tif',
                                            'type': 'image/tiff; application=geotiff; '
                                                    'profile=cloud-optimized',
                                            'title': 'B04',
                                            'eo:bands': [{'name': 'B04'}],
                                            'raster:bands': [{'nodata': _NAN,
                                                              'scale': 0.0001,
                                                              'offset': -0.1}],
                                            'roles': ['data', 'reflectance']},
                                    'SCL': {'href': '/data/A/SCL.tif',
                                            'type': 'image/tiff; application=geotiff; '
                                                    'profile=cloud-optimized',
                                            'title': 'SCL',
                                            'eo:bands': [{'name': 'SCL'}],
                                            'raster:bands': [{'nodata': _NAN,
                                                              'scale': 1,
                                                              'offset': 0}],
                                            'roles': ['data', 'mask']},
                                    'B08': {'href': '/data/A/B08.tif',
                                            'type': 'image/tiff; application=geotiff; '
                                                    'profile=cloud-optimized',
                                            'title': 'B08',
                                            'eo:bands': [{'name': 'B08'}],
                                            'raster:bands': [{'nodata': _NAN,
                                                              'scale': 0.0001,
                                                              'offset': -0.1}],
                                            'roles': ['data', 'reference']},
                                    'B8A': {'href': '/data/A/B8A.jp2',
                                            'type': 'image/jp2',
                                            'title': 'B8A',
                                            'eo:bands': [{'name': 'B8A'}],
                                            'raster:bands': [{'nodata': _NAN,
                                                              'scale': 0.0001,
                                                              'offset': -0.1}],
                                            'roles': ['data', 'reflectance']},
                                    'MTD_TL': {'href': '/data/A/MTD_TL.xml',
                                               'type': 'application/xml',
                                               'title': 'MTD_TL',
                                               'roles': ['metadata']},
                                    'notes': {'href': '/data/A/notes.txt',
                                              'title': 'notes',
                                              'eo:bands': [{'name': 'notes'}],
                                              'roles': ['data']},
                                    'AOT': {'href': '/data/A/AOT.tiff',
                                            'type': 'image/tiff; application=geotiff; '
                                                    'profile=cloud-optimized',
                                            'title': 'AOT',
                                            'eo:bands': [{'name': 'AOT'}],
                                            'raster:bands': [{'nodata': _NAN,
                                                              'scale': 1,
                                                              'offset': 0}],
                                            'roles': ['data']}},
                         'collection': 'forced'},
                        {'type': 'Feature',
                         'stac_version': '1.1.0',
                         'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                             'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                                             'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
                         'id': 'custom_granule_1',
                         'geometry': {'type': 'Polygon',
                                      'coordinates': [[[36.0, 11.6],
                                                       [36.0, 12.0],
                                                       [35.6, 12.0],
                                                       [35.6, 11.6],
                                                       [36.0, 11.6]]]},
                         'bbox': [35.6, 11.6, 36.0, 12.0],
                         'properties': {'eo:cloud_cover': 0.0,
                                        'processing:version': 'N0212',
                                        'datetime': '2018-06-01T07:56:11Z'},
                         'links': [],
                         'assets': {'B04': {'href': '/data/B/B04.jp2',
                                            'type': 'image/jp2',
                                            'title': 'B04',
                                            'eo:bands': [{'name': 'B04'}],
                                            'raster:bands': [{'nodata': 7.0,
                                                              'scale': _NAN,
                                                              'offset': _NAN}],
                                            'roles': ['data', 'reflectance']},
                                    'MTD': {'href': '/data/B/MTD.xml',
                                            'type': 'application/xml',
                                            'title': 'MTD',
                                            'roles': ['metadata']},
                                    'AOT': {'href': '/data/B/AOT.foo',
                                            'title': 'AOT',
                                            'eo:bands': [{'name': 'AOT'}],
                                            'roles': ['data']}},
                         'collection': 'forced'},
                        {'type': 'Feature',
                         'stac_version': '1.1.0',
                         'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                             'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                                             'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
                                             'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
                                             'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
                         'id': 'S2A_MSIL2A_20180601T075611_N0500_R035_T36GZU_20230101T000000',
                         'geometry': {'type': 'Polygon',
                                      'coordinates': [[[11.0, 10.0],
                                                       [11.0, 11.0],
                                                       [10.0, 11.0],
                                                       [10.0, 10.0],
                                                       [11.0, 10.0]]]},
                         'bbox': [10.0, 10.0, 11.0, 11.0],
                         'properties': {'eo:cloud_cover': 20.25,
                                        'processing:datetime': '2023-01-01T00:00:00Z',
                                        'proj:code': 'EPSG:32736',
                                        'grid:code': 'MGRS-36GZU',
                                        'datetime': '2018-06-01T07:56:11Z'},
                         'links': [{'rel': 'via', 'href': 's3://eodata/C.SAFE'}],
                         'assets': {'B04': {'href': '/data/C/B04.tif',
                                            'type': 'image/tiff; application=geotiff; '
                                                    'profile=cloud-optimized',
                                            'title': 'B04',
                                            'eo:bands': [{'name': 'B04'}],
                                            'raster:bands': [{'nodata': 0.0,
                                                              'scale': 0.5,
                                                              'offset': 0.0}],
                                            'roles': ['data', 'reflectance']},
                                    'B08': {'href': '/data/C/B08.tif',
                                            'type': 'image/tiff; application=geotiff; '
                                                    'profile=cloud-optimized',
                                            'title': 'B08',
                                            'eo:bands': [{'name': 'B08'}],
                                            'raster:bands': [{'nodata': 0.0,
                                                              'scale': 0.5,
                                                              'offset': 0.0}],
                                            'roles': ['data', 'reference']},
                                    'SCL': {'href': '/data/C/SCL.tif',
                                            'type': 'image/tiff; application=geotiff; '
                                                    'profile=cloud-optimized',
                                            'title': 'SCL',
                                            'eo:bands': [{'name': 'SCL'}],
                                            'raster:bands': [{'nodata': 0.0,
                                                              'scale': 1,
                                                              'offset': 0}],
                                            'roles': ['data', 'mask']}},
                         'collection': 'forced'}],
 'minimal': [{'type': 'Feature',
              'stac_version': '1.1.0',
              'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                  'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
                                  'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
                                  'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
              'id': 'S2B_MSIL2A_20181231T080329_N0500_R035_T37PBP_20230726T205809',
              'geometry': {'type': 'Polygon',
                           'coordinates': [[[37.0, 12.6],
                                            [37.0, 13.0],
                                            [36.6, 13.0],
                                            [36.6, 12.6],
                                            [37.0, 12.6]]]},
              'bbox': [36.6, 12.6, 37.0, 13.0],
              'properties': {'eo:cloud_cover': 3.0,
                             'proj:code': 'EPSG:32637',
                             'grid:code': 'MGRS-37PBP',
                             'datetime': '2018-12-31T08:03:29Z'},
              'links': [],
              'assets': {'B04': {'href': '/data/M/B04.tif',
                                 'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                                 'title': 'B04',
                                 'eo:bands': [{'name': 'B04'}],
                                 'raster:bands': [{'nodata': 0, 'scale': 0.0001, 'offset': 0.0}],
                                 'roles': ['data', 'reflectance']},
                         'SCL': {'href': '/data/M/SCL.tif',
                                 'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                                 'title': 'SCL',
                                 'eo:bands': [{'name': 'SCL'}],
                                 'raster:bands': [{'nodata': 0, 'scale': 1, 'offset': 0}],
                                 'roles': ['data', 'mask']},
                         'B08': {'href': '/data/M/B08.tif',
                                 'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                                 'title': 'B08',
                                 'eo:bands': [{'name': 'B08'}],
                                 'raster:bands': [{'nodata': 0, 'scale': 0.0001, 'offset': 0.0}],
                                 'roles': ['data', 'reference']}},
              'collection': 'sentinel-2-l2a'},
             {'type': 'Feature',
              'stac_version': '1.1.0',
              'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                  'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
              'id': 'custom_granule_1',
              'geometry': {'type': 'Polygon',
                           'coordinates': [[[3.0, 2.0],
                                            [3.0, 4.0],
                                            [1.0, 4.0],
                                            [1.0, 2.0],
                                            [3.0, 2.0]]]},
              'bbox': [1.0, 2.0, 3.0, 4.0],
              'properties': {'eo:cloud_cover': 4.0, 'datetime': '2018-12-31T08:03:29Z'},
              'links': [],
              'assets': {'B04': {'href': '/data/N/B04.tif',
                                 'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                                 'title': 'B04',
                                 'eo:bands': [{'name': 'B04'}],
                                 'raster:bands': [{'nodata': 0, 'scale': 0.0001, 'offset': 0.0}],
                                 'roles': ['data', 'reflectance']}},
              'collection': 'sentinel-2-l2a'}],
 'minimal_collection_id': [{'type': 'Feature',
                            'stac_version': '1.1.0',
                            'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                                'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
                                                'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
                                                'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
                            'id': 'S2B_MSIL2A_20181231T080329_N0500_R035_T37PBP_20230726T205809',
                            'geometry': {'type': 'Polygon',
                                         'coordinates': [[[37.0, 12.6],
                                                          [37.0, 13.0],
                                                          [36.6, 13.0],
                                                          [36.6, 12.6],
                                                          [37.0, 12.6]]]},
                            'bbox': [36.6, 12.6, 37.0, 13.0],
                            'properties': {'eo:cloud_cover': 3.0,
                                           'proj:code': 'EPSG:32637',
                                           'grid:code': 'MGRS-37PBP',
                                           'datetime': '2018-12-31T08:03:29Z'},
                            'links': [],
                            'assets': {'B04': {'href': '/data/M/B04.tif',
                                               'type': 'image/tiff; application=geotiff; '
                                                       'profile=cloud-optimized',
                                               'title': 'B04',
                                               'eo:bands': [{'name': 'B04'}],
                                               'raster:bands': [{'nodata': 0,
                                                                 'scale': 0.0001,
                                                                 'offset': 0.0}],
                                               'roles': ['data', 'reflectance']},
                                       'SCL': {'href': '/data/M/SCL.tif',
                                               'type': 'image/tiff; application=geotiff; '
                                                       'profile=cloud-optimized',
                                               'title': 'SCL',
                                               'eo:bands': [{'name': 'SCL'}],
                                               'raster:bands': [{'nodata': 0,
                                                                 'scale': 1,
                                                                 'offset': 0}],
                                               'roles': ['data', 'mask']},
                                       'B08': {'href': '/data/M/B08.tif',
                                               'type': 'image/tiff; application=geotiff; '
                                                       'profile=cloud-optimized',
                                               'title': 'B08',
                                               'eo:bands': [{'name': 'B08'}],
                                               'raster:bands': [{'nodata': 0,
                                                                 'scale': 0.0001,
                                                                 'offset': 0.0}],
                                               'roles': ['data', 'reference']}},
                            'collection': 'c'},
                           {'type': 'Feature',
                            'stac_version': '1.1.0',
                            'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                                'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
                            'id': 'custom_granule_1',
                            'geometry': {'type': 'Polygon',
                                         'coordinates': [[[3.0, 2.0],
                                                          [3.0, 4.0],
                                                          [1.0, 4.0],
                                                          [1.0, 2.0],
                                                          [3.0, 2.0]]]},
                            'bbox': [1.0, 2.0, 3.0, 4.0],
                            'properties': {'eo:cloud_cover': 4.0,
                                           'datetime': '2018-12-31T08:03:29Z'},
                            'links': [],
                            'assets': {'B04': {'href': '/data/N/B04.tif',
                                               'type': 'image/tiff; application=geotiff; '
                                                       'profile=cloud-optimized',
                                               'title': 'B04',
                                               'eo:bands': [{'name': 'B04'}],
                                               'raster:bands': [{'nodata': 0,
                                                                 'scale': 0.0001,
                                                                 'offset': 0.0}],
                                               'roles': ['data', 'reflectance']}},
                            'collection': 'c'}],
 'nan_cols': [{'type': 'Feature',
               'stac_version': '1.1.0',
               'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                   'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                                   'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
                                   'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
                                   'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
               'id': 'S2B_MSIL2A_20181231T080329_N0500_R035_T37PBP_20230726T205809',
               'geometry': {'type': 'Polygon',
                            'coordinates': [[[37.0, 12.6],
                                             [37.0, 13.0],
                                             [36.6, 13.0],
                                             [36.6, 12.6],
                                             [37.0, 12.6]]]},
               'bbox': [36.6, 12.6, 37.0, 13.0],
               'properties': {'eo:cloud_cover': 1.5,
                              'processing:version': 'N0500',
                              'processing:datetime': '2023-07-26T20:58:09Z',
                              'proj:code': 'EPSG:32637',
                              'grid:code': 'MGRS-37PBP',
                              'datetime': '2018-12-31T08:03:29Z'},
               'links': [{'rel': 'via', 'href': 's3://eodata/A.SAFE'}],
               'assets': {'B04': {'href': '/data/A/B04.tif',
                                  'type': 'image/tiff; application=geotiff; '
                                          'profile=cloud-optimized',
                                  'title': 'B04',
                                  'eo:bands': [{'name': 'B04'}],
                                  'raster:bands': [{'nodata': _NAN, 'scale': _NAN, 'offset': _NAN}],
                                  'roles': ['data', 'reflectance']},
                          'SCL': {'href': '/data/A/SCL.tif',
                                  'type': 'image/tiff; application=geotiff; '
                                          'profile=cloud-optimized',
                                  'title': 'SCL',
                                  'eo:bands': [{'name': 'SCL'}],
                                  'raster:bands': [{'nodata': _NAN, 'scale': 1, 'offset': 0}],
                                  'roles': ['data', 'mask']},
                          'B08': {'href': '/data/A/B08.tif',
                                  'type': 'image/tiff; application=geotiff; '
                                          'profile=cloud-optimized',
                                  'title': 'B08',
                                  'eo:bands': [{'name': 'B08'}],
                                  'raster:bands': [{'nodata': _NAN, 'scale': _NAN, 'offset': _NAN}],
                                  'roles': ['data', 'reference']},
                          'B8A': {'href': '/data/A/B8A.jp2',
                                  'type': 'image/jp2',
                                  'title': 'B8A',
                                  'eo:bands': [{'name': 'B8A'}],
                                  'raster:bands': [{'nodata': _NAN, 'scale': _NAN, 'offset': _NAN}],
                                  'roles': ['data', 'reflectance']},
                          'MTD_TL': {'href': '/data/A/MTD_TL.xml',
                                     'type': 'application/xml',
                                     'title': 'MTD_TL',
                                     'roles': ['metadata']},
                          'notes': {'href': '/data/A/notes.txt',
                                    'title': 'notes',
                                    'eo:bands': [{'name': 'notes'}],
                                    'roles': ['data']},
                          'AOT': {'href': '/data/A/AOT.tiff',
                                  'type': 'image/tiff; application=geotiff; '
                                          'profile=cloud-optimized',
                                  'title': 'AOT',
                                  'eo:bands': [{'name': 'AOT'}],
                                  'raster:bands': [{'nodata': _NAN, 'scale': 1, 'offset': 0}],
                                  'roles': ['data']}},
               'collection': 'sentinel-2-l2a'}],
 'non_s2_declaration': [{'type': 'Feature',
                         'stac_version': '1.1.0',
                         'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                             'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                                             'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
                                             'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
                                             'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
                         'id': 'S2B_MSIL2A_20181231T080329_N0500_R035_T37PBP_20230726T205809',
                         'geometry': {'type': 'Polygon',
                                      'coordinates': [[[37.0, 12.6],
                                                       [37.0, 13.0],
                                                       [36.6, 13.0],
                                                       [36.6, 12.6],
                                                       [37.0, 12.6]]]},
                         'bbox': [36.6, 12.6, 37.0, 13.0],
                         'properties': {'eo:cloud_cover': 1.5,
                                        'processing:version': 'N0500',
                                        'processing:datetime': '2023-07-26T20:58:09Z',
                                        'proj:code': 'EPSG:32637',
                                        'grid:code': 'MGRS-37PBP',
                                        'datetime': '2018-12-31T08:03:29Z'},
                         'links': [{'rel': 'via', 'href': 's3://eodata/A.SAFE'}],
                         'assets': {'B2': {'href': '/data/A/B2.xml',
                                           'type': 'application/xml',
                                           'title': 'B2',
                                           'roles': ['metadata']},
                                    'B3': {'href': '/data/A/B3.tif',
                                           'type': 'image/tiff; application=geotiff; '
                                                   'profile=cloud-optimized',
                                           'title': 'B3',
                                           'eo:bands': [{'name': 'B3'}],
                                           'raster:bands': [{'nodata': _NAN,
                                                             'scale': 1,
                                                             'offset': 0}],
                                           'roles': ['data']},
                                    'B4': {'href': '/data/A/B4.tif',
                                           'type': 'image/tiff; application=geotiff; '
                                                   'profile=cloud-optimized',
                                           'title': 'B4',
                                           'eo:bands': [{'name': 'B4'}],
                                           'raster:bands': [{'nodata': _NAN,
                                                             'scale': 0.25,
                                                             'offset': -250.0}],
                                           'roles': ['data', 'reflectance']},
                                    'QA': {'href': '/data/A/QA.tif',
                                           'type': 'image/tiff; application=geotiff; '
                                                   'profile=cloud-optimized',
                                           'title': 'QA',
                                           'eo:bands': [{'name': 'QA'}],
                                           'raster:bands': [{'nodata': _NAN,
                                                             'scale': 1,
                                                             'offset': 0}],
                                           'roles': ['data', 'mask']}},
                         'collection': 'sentinel-2-l2a'},
                        {'type': 'Feature',
                         'stac_version': '1.1.0',
                         'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                                             'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                                             'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
                         'id': 'custom_granule_1',
                         'geometry': {'type': 'Polygon',
                                      'coordinates': [[[36.0, 11.6],
                                                       [36.0, 12.0],
                                                       [35.6, 12.0],
                                                       [35.6, 11.6],
                                                       [36.0, 11.6]]]},
                         'bbox': [35.6, 11.6, 36.0, 12.0],
                         'properties': {'eo:cloud_cover': 0.0,
                                        'processing:version': 'N0212',
                                        'datetime': '2018-06-01T07:56:11Z'},
                         'links': [],
                         'assets': {'B2': {'href': '/data/B/B2.jp2',
                                           'type': 'image/jp2',
                                           'title': 'B2',
                                           'eo:bands': [{'name': 'B2'}],
                                           'raster:bands': [{'nodata': 7.0,
                                                             'scale': _NAN,
                                                             'offset': _NAN}],
                                           'roles': ['data', 'reference']},
                                    'QA': {'href': '/data/B/QA.jp2',
                                           'type': 'image/jp2',
                                           'title': 'QA',
                                           'eo:bands': [{'name': 'QA'}],
                                           'raster:bands': [{'nodata': 7.0,
                                                             'scale': 1,
                                                             'offset': 0}],
                                           'roles': ['data', 'mask']},
                                    'B9': {'href': '/data/B/B9.tif',
                                           'type': 'image/tiff; application=geotiff; '
                                                   'profile=cloud-optimized',
                                           'title': 'B9',
                                           'eo:bands': [{'name': 'B9'}],
                                           'raster:bands': [{'nodata': 7.0,
                                                             'scale': 1,
                                                             'offset': 0}],
                                           'roles': ['data']}},
                         'collection': 'other-coll'}]}

READ_PROJ_CALLS = ['/data/A/B04.tif',
 '/data/A/SCL.tif',
 '/data/A/B08.tif',
 '/data/A/B8A.jp2',
 '/data/A/notes.txt',
 '/data/A/AOT.tiff',
 '/data/B/B04.jp2',
 '/data/B/AOT.foo',
 '/data/C/B04.tif',
 '/data/C/B08.tif',
 '/data/C/SCL.tif']

READ_PROJ_EXPECTED = [{'type': 'Feature',
  'stac_version': '1.1.0',
  'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                      'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                      'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
                      'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
                      'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
  'id': 'S2B_MSIL2A_20181231T080329_N0500_R035_T37PBP_20230726T205809',
  'geometry': {'type': 'Polygon',
               'coordinates': [[[37.0, 12.6],
                                [37.0, 13.0],
                                [36.6, 13.0],
                                [36.6, 12.6],
                                [37.0, 12.6]]]},
  'bbox': [36.6, 12.6, 37.0, 13.0],
  'properties': {'eo:cloud_cover': 1.5,
                 'processing:version': 'N0500',
                 'processing:datetime': '2023-07-26T20:58:09Z',
                 'proj:code': 'EPSG:32637',
                 'grid:code': 'MGRS-37PBP',
                 'datetime': '2018-12-31T08:03:29Z'},
  'links': [{'rel': 'via', 'href': 's3://eodata/A.SAFE'}],
  'assets': {'B04': {'href': '/data/A/B04.tif',
                     'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                     'title': 'B04',
                     'eo:bands': [{'name': 'B04'}],
                     'proj:shape': [2, 3],
                     'proj:transform': [10.0, 0.0, 5.0, 0.0, -10.0, 6.0],
                     'raster:bands': [{'nodata': _NAN, 'scale': 0.0001, 'offset': -0.1}],
                     'roles': ['data', 'reflectance']},
             'SCL': {'href': '/data/A/SCL.tif',
                     'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                     'title': 'SCL',
                     'eo:bands': [{'name': 'SCL'}],
                     'proj:shape': [2, 3],
                     'proj:transform': [10.0, 0.0, 5.0, 0.0, -10.0, 6.0],
                     'raster:bands': [{'nodata': _NAN, 'scale': 1, 'offset': 0}],
                     'roles': ['data', 'mask']},
             'B08': {'href': '/data/A/B08.tif',
                     'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                     'title': 'B08',
                     'eo:bands': [{'name': 'B08'}],
                     'proj:shape': [2, 3],
                     'proj:transform': [10.0, 0.0, 5.0, 0.0, -10.0, 6.0],
                     'raster:bands': [{'nodata': _NAN, 'scale': 0.0001, 'offset': -0.1}],
                     'roles': ['data', 'reference']},
             'B8A': {'href': '/data/A/B8A.jp2',
                     'type': 'image/jp2',
                     'title': 'B8A',
                     'eo:bands': [{'name': 'B8A'}],
                     'proj:shape': [2, 3],
                     'proj:transform': [10.0, 0.0, 5.0, 0.0, -10.0, 6.0],
                     'raster:bands': [{'nodata': _NAN, 'scale': 0.0001, 'offset': -0.1}],
                     'roles': ['data', 'reflectance']},
             'MTD_TL': {'href': '/data/A/MTD_TL.xml',
                        'type': 'application/xml',
                        'title': 'MTD_TL',
                        'roles': ['metadata']},
             'notes': {'href': '/data/A/notes.txt',
                       'title': 'notes',
                       'eo:bands': [{'name': 'notes'}],
                       'proj:shape': [2, 3],
                       'proj:transform': [10.0, 0.0, 5.0, 0.0, -10.0, 6.0],
                       'roles': ['data']},
             'AOT': {'href': '/data/A/AOT.tiff',
                     'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                     'title': 'AOT',
                     'eo:bands': [{'name': 'AOT'}],
                     'proj:shape': [2, 3],
                     'proj:transform': [10.0, 0.0, 5.0, 0.0, -10.0, 6.0],
                     'raster:bands': [{'nodata': _NAN, 'scale': 1, 'offset': 0}],
                     'roles': ['data']}},
  'collection': 'sentinel-2-l2a'},
 {'type': 'Feature',
  'stac_version': '1.1.0',
  'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                      'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                      'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
  'id': 'custom_granule_1',
  'geometry': {'type': 'Polygon',
               'coordinates': [[[36.0, 11.6],
                                [36.0, 12.0],
                                [35.6, 12.0],
                                [35.6, 11.6],
                                [36.0, 11.6]]]},
  'bbox': [35.6, 11.6, 36.0, 12.0],
  'properties': {'eo:cloud_cover': 0.0,
                 'processing:version': 'N0212',
                 'datetime': '2018-06-01T07:56:11Z'},
  'links': [],
  'assets': {'B04': {'href': '/data/B/B04.jp2',
                     'type': 'image/jp2',
                     'title': 'B04',
                     'eo:bands': [{'name': 'B04'}],
                     'proj:shape': [2, 3],
                     'proj:transform': [10.0, 0.0, 5.0, 0.0, -10.0, 6.0],
                     'raster:bands': [{'nodata': 7.0, 'scale': _NAN, 'offset': _NAN}],
                     'roles': ['data', 'reflectance']},
             'MTD': {'href': '/data/B/MTD.xml',
                     'type': 'application/xml',
                     'title': 'MTD',
                     'roles': ['metadata']},
             'AOT': {'href': '/data/B/AOT.foo',
                     'title': 'AOT',
                     'eo:bands': [{'name': 'AOT'}],
                     'proj:shape': [2, 3],
                     'proj:transform': [10.0, 0.0, 5.0, 0.0, -10.0, 6.0],
                     'roles': ['data']}},
  'collection': 'other-coll'},
 {'type': 'Feature',
  'stac_version': '1.1.0',
  'stac_extensions': ['https://stac-extensions.github.io/eo/v1.1.0/schema.json',
                      'https://stac-extensions.github.io/processing/v1.2.0/schema.json',
                      'https://stac-extensions.github.io/projection/v2.0.0/schema.json',
                      'https://stac-extensions.github.io/grid/v1.1.0/schema.json',
                      'https://stac-extensions.github.io/raster/v1.1.0/schema.json'],
  'id': 'S2A_MSIL2A_20180601T075611_N0500_R035_T36GZU_20230101T000000',
  'geometry': {'type': 'Polygon',
               'coordinates': [[[11.0, 10.0],
                                [11.0, 11.0],
                                [10.0, 11.0],
                                [10.0, 10.0],
                                [11.0, 10.0]]]},
  'bbox': [10.0, 10.0, 11.0, 11.0],
  'properties': {'eo:cloud_cover': 20.25,
                 'processing:datetime': '2023-01-01T00:00:00Z',
                 'proj:code': 'EPSG:32736',
                 'grid:code': 'MGRS-36GZU',
                 'datetime': '2018-06-01T07:56:11Z'},
  'links': [{'rel': 'via', 'href': 's3://eodata/C.SAFE'}],
  'assets': {'B04': {'href': '/data/C/B04.tif',
                     'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                     'title': 'B04',
                     'eo:bands': [{'name': 'B04'}],
                     'proj:shape': [2, 3],
                     'proj:transform': [10.0, 0.0, 5.0, 0.0, -10.0, 6.0],
                     'raster:bands': [{'nodata': 0.0, 'scale': 0.5, 'offset': 0.0}],
                     'roles': ['data', 'reflectance']},
             'B08': {'href': '/data/C/B08.tif',
                     'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                     'title': 'B08',
                     'eo:bands': [{'name': 'B08'}],
                     'proj:shape': [2, 3],
                     'proj:transform': [10.0, 0.0, 5.0, 0.0, -10.0, 6.0],
                     'raster:bands': [{'nodata': 0.0, 'scale': 0.5, 'offset': 0.0}],
                     'roles': ['data', 'reference']},
             'SCL': {'href': '/data/C/SCL.tif',
                     'type': 'image/tiff; application=geotiff; profile=cloud-optimized',
                     'title': 'SCL',
                     'eo:bands': [{'name': 'SCL'}],
                     'proj:shape': [2, 3],
                     'proj:transform': [10.0, 0.0, 5.0, 0.0, -10.0, 6.0],
                     'raster:bands': [{'nodata': 0.0, 'scale': 1, 'offset': 0}],
                     'roles': ['data', 'mask']}},
  'collection': 'sentinel-2-l2a'}]

COG_EXPECTED = {'none_nobands': [{'type': 'Feature',
                   'stac_version': '1.1.0',
                   'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json'],
                   'id': 'cell-1',
                   'geometry': {'type': 'Polygon',
                                'coordinates': [[[39.000444628882065, 36.144357468939475],
                                                 [39.000444628882065, 36.14471809881776],
                                                 [38.99999999999999, 36.14471809881776],
                                                 [38.99999999999999, 36.144357468939475],
                                                 [39.000444628882065, 36.144357468939475]]]},
                   'bbox': [38.99999999999999,
                            36.144357468939475,
                            39.000444628882065,
                            36.14471809881776],
                   'properties': {'proj:code': 'EPSG:32637',
                                  'proj:shape': [4, 4],
                                  'proj:transform': [10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0],
                                  'datetime': '2020-01-02T03:04:05Z'},
                   'links': [],
                   'assets': {'output': {'href': '<tmp>/cell-1/output.tif',
                                         'type': 'image/tiff; application=geotiff; '
                                                 'profile=cloud-optimized',
                                         'title': 'output',
                                         'roles': ['data']}},
                   'collection': 'fsd-inference'},
                  {'type': 'Feature',
                   'stac_version': '1.1.0',
                   'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json'],
                   'id': 'cell-2',
                   'geometry': {'type': 'Polygon',
                                'coordinates': [[[33.00044462888208, 36.144357468939475],
                                                 [33.00044462888208, 36.14471809881776],
                                                 [33.00000000000001, 36.14471809881776],
                                                 [33.00000000000001, 36.144357468939475],
                                                 [33.00044462888208, 36.144357468939475]]]},
                   'bbox': [33.00000000000001,
                            36.144357468939475,
                            33.00044462888208,
                            36.14471809881776],
                   'properties': {'proj:code': 'EPSG:32636',
                                  'proj:shape': [4, 4],
                                  'proj:transform': [10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0],
                                  'datetime': '2020-01-02T03:04:05Z'},
                   'links': [],
                   'assets': {'output': {'href': '<tmp>/cell-2/output.tif',
                                         'type': 'image/tiff; application=geotiff; '
                                                 'profile=cloud-optimized',
                                         'title': 'output',
                                         'roles': ['data']}},
                   'collection': 'fsd-inference'}],
 'none_bands_collection': [{'type': 'Feature',
                            'stac_version': '1.1.0',
                            'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json'],
                            'id': 'cell-1',
                            'geometry': {'type': 'Polygon',
                                         'coordinates': [[[39.000444628882065, 36.144357468939475],
                                                          [39.000444628882065, 36.14471809881776],
                                                          [38.99999999999999, 36.14471809881776],
                                                          [38.99999999999999, 36.144357468939475],
                                                          [39.000444628882065,
                                                           36.144357468939475]]]},
                            'bbox': [38.99999999999999,
                                     36.144357468939475,
                                     39.000444628882065,
                                     36.14471809881776],
                            'properties': {'proj:code': 'EPSG:32637',
                                           'proj:shape': [4, 4],
                                           'proj:transform': [10.0,
                                                              0.0,
                                                              500000.0,
                                                              0.0,
                                                              -10.0,
                                                              4000000.0],
                                           'datetime': '2020-01-02T03:04:05Z'},
                            'links': [],
                            'assets': {'output': {'href': '<tmp>/cell-1/output.tif',
                                                  'type': 'image/tiff; application=geotiff; '
                                                          'profile=cloud-optimized',
                                                  'title': 'output',
                                                  'eo:bands': [{'name': 'a'}, {'name': 'b'}],
                                                  'roles': ['data']}},
                            'collection': 'c2'},
                           {'type': 'Feature',
                            'stac_version': '1.1.0',
                            'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json'],
                            'id': 'cell-2',
                            'geometry': {'type': 'Polygon',
                                         'coordinates': [[[33.00044462888208, 36.144357468939475],
                                                          [33.00044462888208, 36.14471809881776],
                                                          [33.00000000000001, 36.14471809881776],
                                                          [33.00000000000001, 36.144357468939475],
                                                          [33.00044462888208,
                                                           36.144357468939475]]]},
                            'bbox': [33.00000000000001,
                                     36.144357468939475,
                                     33.00044462888208,
                                     36.14471809881776],
                            'properties': {'proj:code': 'EPSG:32636',
                                           'proj:shape': [4, 4],
                                           'proj:transform': [10.0,
                                                              0.0,
                                                              500000.0,
                                                              0.0,
                                                              -10.0,
                                                              4000000.0],
                                           'datetime': '2020-01-02T03:04:05Z'},
                            'links': [],
                            'assets': {'output': {'href': '<tmp>/cell-2/output.tif',
                                                  'type': 'image/tiff; application=geotiff; '
                                                          'profile=cloud-optimized',
                                                  'title': 'output',
                                                  'eo:bands': [{'name': 'a'}, {'name': 'b'}],
                                                  'roles': ['data']}},
                            'collection': 'c2'}],
 'path_form': [{'type': 'Feature',
                'stac_version': '1.1.0',
                'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json'],
                'id': 'cell-1',
                'geometry': {'type': 'Polygon',
                             'coordinates': [[[14.766, 48.492],
                                              [14.789, 48.534],
                                              [14.847, 48.526],
                                              [14.825, 48.484],
                                              [14.766, 48.492]]]},
                'bbox': [14.766, 48.484, 14.847, 48.534],
                'properties': {'proj:code': 'EPSG:32637',
                               'proj:shape': [4, 4],
                               'proj:transform': [10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0],
                               'datetime': '2020-01-02T03:04:05Z'},
                'links': [],
                'assets': {'output': {'href': '<tmp>/cell-1/output.tif',
                                      'type': 'image/tiff; application=geotiff; '
                                              'profile=cloud-optimized',
                                      'title': 'output',
                                      'eo:bands': [{'name': 'x'}],
                                      'roles': ['data']}},
                'collection': 'fsd-inference'},
               {'type': 'Feature',
                'stac_version': '1.1.0',
                'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json'],
                'id': 'cell-2',
                'geometry': {'type': 'Polygon',
                             'coordinates': [[[14.5, 48.0],
                                              [14.5, 48.5],
                                              [14.0, 48.5],
                                              [14.0, 48.0],
                                              [14.5, 48.0]]]},
                'bbox': [14.0, 48.0, 14.5, 48.5],
                'properties': {'proj:code': 'EPSG:32636',
                               'proj:shape': [4, 4],
                               'proj:transform': [10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0],
                               'datetime': '2020-01-02T03:04:05Z'},
                'links': [],
                'assets': {'output': {'href': '<tmp>/cell-2/output.tif',
                                      'type': 'image/tiff; application=geotiff; '
                                              'profile=cloud-optimized',
                                      'title': 'output',
                                      'eo:bands': [{'name': 'x'}],
                                      'roles': ['data']}},
                'collection': 'fsd-inference'}],
 'shapely_form': [{'type': 'Feature',
                   'stac_version': '1.1.0',
                   'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json'],
                   'id': 'cell-1',
                   'geometry': {'type': 'Polygon',
                                'coordinates': [[[14.1, 48.0],
                                                 [14.1, 48.1],
                                                 [14.0, 48.1],
                                                 [14.0, 48.0],
                                                 [14.1, 48.0]]]},
                   'bbox': [14.0, 48.0, 14.1, 48.1],
                   'properties': {'proj:code': 'EPSG:32637',
                                  'proj:shape': [4, 4],
                                  'proj:transform': [10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0],
                                  'datetime': '2020-01-02T03:04:05Z'},
                   'links': [],
                   'assets': {'output': {'href': '<tmp>/cell-1/output.tif',
                                         'type': 'image/tiff; application=geotiff; '
                                                 'profile=cloud-optimized',
                                         'title': 'output',
                                         'roles': ['data']}},
                   'collection': 'fsd-inference'},
                  {'type': 'Feature',
                   'stac_version': '1.1.0',
                   'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json'],
                   'id': 'cell-2',
                   'geometry': {'type': 'Polygon',
                                'coordinates': [[[15.2, 49.0],
                                                 [15.2, 49.4],
                                                 [15.0, 49.4],
                                                 [15.0, 49.0],
                                                 [15.2, 49.0]]]},
                   'bbox': [15.0, 49.0, 15.2, 49.4],
                   'properties': {'proj:code': 'EPSG:32636',
                                  'proj:shape': [4, 4],
                                  'proj:transform': [10.0, 0.0, 500000.0, 0.0, -10.0, 4000000.0],
                                  'datetime': '2020-01-02T03:04:05Z'},
                   'links': [],
                   'assets': {'output': {'href': '<tmp>/cell-2/output.tif',
                                         'type': 'image/tiff; application=geotiff; '
                                                 'profile=cloud-optimized',
                                         'title': 'output',
                                         'roles': ['data']}},
                   'collection': 'fsd-inference'}],
 'feature_without_id': [{'type': 'Feature',
                         'stac_version': '1.1.0',
                         'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json'],
                         'id': 'cell-1',
                         'geometry': {'type': 'Polygon',
                                      'coordinates': [[[14.1, 48.0],
                                                       [14.1, 48.1],
                                                       [14.0, 48.1],
                                                       [14.0, 48.0],
                                                       [14.1, 48.0]]]},
                         'bbox': [14.0, 48.0, 14.1, 48.1],
                         'properties': {'proj:code': 'EPSG:32637',
                                        'proj:shape': [4, 4],
                                        'proj:transform': [10.0,
                                                           0.0,
                                                           500000.0,
                                                           0.0,
                                                           -10.0,
                                                           4000000.0],
                                        'datetime': '2020-01-02T03:04:05Z'},
                         'links': [],
                         'assets': {'output': {'href': '<tmp>/cell-1/output.tif',
                                               'type': 'image/tiff; application=geotiff; '
                                                       'profile=cloud-optimized',
                                               'title': 'output',
                                               'roles': ['data']}},
                         'collection': 'fsd-inference'}],
 'feature_null_properties': [{'type': 'Feature',
                              'stac_version': '1.1.0',
                              'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json'],
                              'id': 'cell-1',
                              'geometry': {'type': 'Polygon',
                                           'coordinates': [[[14.1, 48.0],
                                                            [14.1, 48.1],
                                                            [14.0, 48.1],
                                                            [14.0, 48.0],
                                                            [14.1, 48.0]]]},
                              'bbox': [14.0, 48.0, 14.1, 48.1],
                              'properties': {'proj:code': 'EPSG:32637',
                                             'proj:shape': [4, 4],
                                             'proj:transform': [10.0,
                                                                0.0,
                                                                500000.0,
                                                                0.0,
                                                                -10.0,
                                                                4000000.0],
                                             'datetime': '2020-01-02T03:04:05Z'},
                              'links': [],
                              'assets': {'output': {'href': '<tmp>/cell-1/output.tif',
                                                    'type': 'image/tiff; application=geotiff; '
                                                            'profile=cloud-optimized',
                                                    'title': 'output',
                                                    'roles': ['data']}},
                              'collection': 'fsd-inference'}],
 'str_key_wins_over_path_key': [{'type': 'Feature',
                                 'stac_version': '1.1.0',
                                 'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json'],
                                 'id': 'cell-1',
                                 'geometry': {'type': 'Polygon',
                                              'coordinates': [[[14.1, 48.0],
                                                               [14.1, 48.1],
                                                               [14.0, 48.1],
                                                               [14.0, 48.0],
                                                               [14.1, 48.0]]]},
                                 'bbox': [14.0, 48.0, 14.1, 48.1],
                                 'properties': {'proj:code': 'EPSG:32637',
                                                'proj:shape': [4, 4],
                                                'proj:transform': [10.0,
                                                                   0.0,
                                                                   500000.0,
                                                                   0.0,
                                                                   -10.0,
                                                                   4000000.0],
                                                'datetime': '2020-01-02T03:04:05Z'},
                                 'links': [],
                                 'assets': {'output': {'href': '<tmp>/cell-1/output.tif',
                                                       'type': 'image/tiff; application=geotiff; '
                                                               'profile=cloud-optimized',
                                                       'title': 'output',
                                                       'roles': ['data']}},
                                 'collection': 'fsd-inference'}],
 'path_key_when_no_str_key': [{'type': 'Feature',
                               'stac_version': '1.1.0',
                               'stac_extensions': ['https://stac-extensions.github.io/projection/v2.0.0/schema.json'],
                               'id': 'cell-1',
                               'geometry': {'type': 'Polygon',
                                            'coordinates': [[[20.5, 50.0],
                                                             [20.5, 50.5],
                                                             [20.0, 50.5],
                                                             [20.0, 50.0],
                                                             [20.5, 50.0]]]},
                               'bbox': [20.0, 50.0, 20.5, 50.5],
                               'properties': {'proj:code': 'EPSG:32637',
                                              'proj:shape': [4, 4],
                                              'proj:transform': [10.0,
                                                                 0.0,
                                                                 500000.0,
                                                                 0.0,
                                                                 -10.0,
                                                                 4000000.0],
                                              'datetime': '2020-01-02T03:04:05Z'},
                               'links': [],
                               'assets': {'output': {'href': '<tmp>/cell-1/output.tif',
                                                     'type': 'image/tiff; application=geotiff; '
                                                             'profile=cloud-optimized',
                                                     'title': 'output',
                                                     'roles': ['data']}},
                               'collection': 'fsd-inference'}]}

COG_ERR_EXPECTED = {'no_entry': ('ValueError',
              'cog_outputs_to_items: geometries has no entry for output COG '
              "'<tmp>/cell-1/output.tif'; the manifest-driven contract requires every output to "
              'have a footprint (pass geometries=None to fall back to the raster bbox for ALL '
              'outputs).',
              None),
 'empty_shapely': ('ValueError',
                   'cog_outputs_to_items: geometries entry for output COG '
                   "'<tmp>/cell-1/output.tif' is an empty geometry.",
                   None),
 'path_missing': ('ValueError',
                  "cog_outputs_to_items: could not read geometry '<tmp>/nope/geometry.geojson' for "
                  "output COG '<tmp>/cell-1/output.tif': [Errno 2] No such file or directory: "
                  "'<tmp>/nope/geometry.geojson'",
                  'FileNotFoundError'),
 'path_invalid_json': ('ValueError',
                       "cog_outputs_to_items: could not read geometry '<tmp>/bad.geojson' for "
                       "output COG '<tmp>/cell-1/output.tif': Expecting property name enclosed in "
                       'double quotes: line 1 column 2 (char 1)',
                       'JSONDecodeError'),
 'no_features': ('ValueError',
                 "cog_outputs_to_items: geometry.geojson at '<tmp>/empty.geojson' (for "
                 "'<tmp>/cell-1/output.tif') has no readable polygon feature.",
                 None),
 'id_mismatch': ('ValueError',
                 "cog_outputs_to_items: geometry.geojson id 'other' at "
                 "'<tmp>/cell-2/geometry.geojson' disagrees with output item id 'cell-1' for "
                 "'<tmp>/cell-1/output.tif'.",
                 None),
 'duplicate_ids': ('ValueError',
                   "cog_outputs_to_items: item ids are not unique (['cell-1']); each output must "
                   'live in its own per-cube folder (<cube_id>/output.tif).',
                   None),
 'missing_cog_and_no_entry': ('RasterioIOError',
                              '<tmp>/ghost/output.tif: No such file or directory',
                              None),
 'second_entry_missing': ('ValueError',
                          'cog_outputs_to_items: geometries has no entry for output COG '
                          "'<tmp>/cell-2/output.tif'; the manifest-driven contract requires every "
                          'output to have a footprint (pass geometries=None to fall back to the '
                          'raster bbox for ALL outputs).',
                          None)}


