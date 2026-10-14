"""Spec 59 Amendment A3 (#145 X4, #173): a missing or empty `offset` / `nodata` fails loudly.

T1: `append` refuses it. T2: `append` fills `scale` from the declaration. T3: `read()` refuses a
file whose `offset` / `scale` / `nodata` column is missing or NaN. T4: the two readers
(`flatten_catalog`, `tile_catalog_to_items`) no longer default a missing column.
"""

from __future__ import annotations

import os

import geopandas as gpd
import pandas as pd
import pytest
import shapely.geometry as sg

from fsd import config
from fsd.catalog import stac
from fsd.catalog.catalog import TileCatalog
from fsd.catalog.declaration import S2_L2A_DECLARATION
from fsd.datacube import builder
from fsd.storage import fs


# Copied from tests/test_p12b_characterisation.py: CI runs bare `pytest`, which cannot import `tests.*`.
def _min_row(id="t1", **kw):
    row = dict(
        id=id, collection=config.SATELLITE_S2L2A, timestamp="2024-08-01T10:00:00Z",
        s3url=f"s3://eodata/{id}.SAFE", local_folderpath=f"/data/{id}", files="B02.jp2",
        cloud_cover=0.0, geometry=sg.box(0, 0, 1, 1),
    )
    row.update(kw)
    return row


_OTHER = {"offset": "nodata", "nodata": "offset"}


# --- T1: append refuses a missing / empty offset or nodata -------------------


@pytest.mark.parametrize("column", ["offset", "nodata"])
@pytest.mark.parametrize("shape", ["missing_from_every_row", "none_in_one_row"])
def test_append_refuses_missing_or_empty_offset_nodata(tmp_path, column, shape):
    other = _OTHER[column]
    if shape == "missing_from_every_row":
        rows = [_min_row("a", **{other: 0}), _min_row("b", **{other: 0})]
    else:
        rows = [
            _min_row("a", **{other: 0, column: None}),
            _min_row("b", **{other: 0, column: 0}),
        ]
    path = str(tmp_path / "c.parquet")
    with pytest.raises(ValueError, match=column) as exc:
        TileCatalog(path).append(rows)
    # The first row lacks the value in both shapes, so its id is named.
    assert "a" in str(exc.value)
    if shape == "none_in_one_row":
        assert "'b'" not in str(exc.value)
    assert not os.path.exists(path)


# --- T2: append fills scale from the declaration -----------------------------


@pytest.mark.parametrize("shape", ["missing_from_every_row", "none_in_one_row"])
def test_append_fills_scale_from_the_declaration(tmp_path, shape):
    if shape == "missing_from_every_row":
        rows = [_min_row("a", offset=0, nodata=0), _min_row("b", offset=0, nodata=0)]
    else:
        rows = [
            _min_row("a", offset=0, nodata=0, scale=None),
            _min_row("b", offset=0, nodata=0, scale=config.S2_REFLECTANCE_SCALE),
        ]
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    cat.append(rows, declaration=S2_L2A_DECLARATION)
    got = cat.read()
    assert len(got) == 2
    for value in got["scale"]:
        assert value == pytest.approx(config.S2_REFLECTANCE_SCALE)


# --- T3: read() refuses a file with a missing / NaN offset, scale or nodata --


def _write_mutated(tmp_path, mutate):
    path = str(tmp_path / "c.parquet")
    cat = TileCatalog(path)
    cat.append([_min_row("a", offset=-1000, scale=0.0001, nodata=0)])
    gdf = cat.read()
    mutate(gdf)
    fs.write_parquet(path, gdf)
    return path


def _drop_offset(gdf):
    gdf.drop(columns=["offset"], inplace=True)


def _nan_column(column):
    def mutate(gdf):
        gdf[column] = float("nan")

    return mutate


@pytest.mark.parametrize(
    "column, mutate",
    [
        ("offset", _drop_offset),
        ("offset", _nan_column("offset")),
        ("scale", _nan_column("scale")),
        ("nodata", _nan_column("nodata")),
    ],
    ids=["no_offset_column", "nan_offset", "nan_scale", "nan_nodata"],
)
def test_read_refuses_missing_or_nan_offset_scale_nodata(tmp_path, column, mutate):
    path = _write_mutated(tmp_path, mutate)
    with pytest.raises(ValueError) as exc:
        TileCatalog(path).read()
    assert path in str(exc.value)
    assert column in str(exc.value)


# --- T4: the readers no longer default a missing column ----------------------

_CRS = "EPSG:32633"
_TILE_BOX = sg.box(500000, 4999960, 500040, 5000000)


def test_flatten_catalog_raises_keyerror_without_offset(tmp_path):
    gdf = gpd.GeoDataFrame(
        [{"id": "x", "local_folderpath": str(tmp_path), "timestamp": 0,
          "files": "B04.jp2", "area_contribution": 50.0, "nodata": 0,
          "geometry": _TILE_BOX}], crs=_CRS,
    )
    with pytest.raises(KeyError, match="offset"):
        builder.flatten_catalog(gdf)


def test_tile_catalog_to_items_raises_keyerror_without_nodata():
    gdf = gpd.GeoDataFrame(
        [{
            "id": "S2B_MSIL2A_20181231T080329_N0500_R035_T37PBP_20230726T205809",
            "collection": "sentinel-2-l2a",
            "timestamp": pd.Timestamp("2018-12-31T08:03:29", tz="UTC"),
            "processing_version": "N0500", "processing_datetime": None,
            "s3url": "s3://eodata/A.SAFE", "local_folderpath": "/data/A",
            "files": "B04.tif,SCL.tif", "cloud_cover": 1.5,
            "offset": -1000, "scale": 0.0001,
            "geometry": sg.box(36.6, 12.6, 37.0, 13.0),
        }],
        geometry="geometry", crs="EPSG:4326",
    )
    with pytest.raises(KeyError, match="nodata"):
        stac.tile_catalog_to_items(gdf)
