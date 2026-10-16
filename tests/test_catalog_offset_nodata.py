"""Spec 59 Amendment A3 (#145 X4, #173): a missing or empty `offset` / `nodata` fails loudly.

T1: `append` refuses it. T2: `append` fills `scale` from the declaration. T4: `flatten_catalog`
no longer defaults a missing column. (T3 was dropped when A3 was narrowed.)
"""

from __future__ import annotations

import os

import geopandas as gpd
import pytest
import shapely.geometry as sg

from rise_eo import config
from rise_eo.catalog.catalog import TileCatalog
from rise_eo.catalog.declaration import S2_L2A_DECLARATION
from rise_eo.datacube import builder


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
    # Quoted: the message also says "offset/nodata" unquoted, which would match either column.
    with pytest.raises(ValueError, match=f"'{column}'") as exc:
        TileCatalog(path).append(rows)
    # The first row lacks the value in both shapes, so its id is named (quoted, as repr does).
    assert "'a'" in str(exc.value)
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


def test_append_fills_scale_from_the_existing_files_stamp(tmp_path):
    # X4.2: with no `declaration=` and no constructor declaration, the fill comes from the
    # declaration already stamped on the file.
    path = str(tmp_path / "c.parquet")
    TileCatalog(path).append(
        [_min_row("a", offset=0, nodata=0, scale=config.S2_REFLECTANCE_SCALE)],
        declaration=S2_L2A_DECLARATION,
    )
    TileCatalog(path).append([_min_row("b", offset=0, nodata=0)])
    got = TileCatalog(path).read().set_index("id")
    assert got.loc["b", "scale"] == pytest.approx(config.S2_REFLECTANCE_SCALE)


# --- T4: flatten_catalog no longer defaults a missing column ----------------

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
