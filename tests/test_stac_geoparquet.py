"""Round-trip test for the stac-geoparquet export (spec 30 Deliverable B).

Skips cleanly in the core `.venv` (`pytest.importorskip`) — run from a `[serving]` venv
(`python3.11 -m venv .venv-serving && .venv-serving/bin/pip install -e ".[dev,serving]"`).
"""

from __future__ import annotations

import datetime as dt
import json
import os

import pystac
import pytest
import shapely.geometry

pytest.importorskip("stac_geoparquet", reason="needs the [serving] extra (stac-geoparquet)")

from rise_eo.catalog import stac, stac_geoparquet  # noqa: E402


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


def _write_geometry_geojson(folder, *, feature_id, geom=_SLANTED_POLYGON):
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


def _make_items(tmp_path, n=2):
    cogs, geometries = [], {}
    for i in range(n):
        cell_dir = str(tmp_path / f"cell-{i}")
        cog = _make_output_cog(cell_dir)
        geom_path = _write_geometry_geojson(cell_dir, feature_id=f"cell-{i}")
        cogs.append(cog)
        geometries[cog] = geom_path
    when = dt.datetime(2018, 6, 1, tzinfo=dt.timezone.utc)
    return stac.cog_outputs_to_items(
        cogs, geometries=geometries, band_names=["crop_class"], dt=when
    )


def test_round_trip_preserves_core_fields(tmp_path):
    items = _make_items(tmp_path, n=2)
    dst_fp = str(tmp_path / "catalog.parquet")

    out_fp = stac_geoparquet.items_to_stac_geoparquet(items, dst_fp)
    assert out_fp == dst_fp
    assert os.path.exists(dst_fp)

    read_back = stac_geoparquet.stac_geoparquet_to_items(dst_fp)
    assert len(read_back) == len(items)

    by_id = {it.id: it for it in read_back}
    for original in items:
        got = by_id[original.id]
        assert isinstance(got, pystac.Item)
        assert got.id == original.id
        assert shapely.geometry.shape(got.geometry).equals(shapely.geometry.shape(original.geometry))
        assert list(got.bbox) == list(original.bbox)
        assert got.datetime == original.datetime
        assert got.properties["proj:shape"] == original.properties["proj:shape"]
        assert got.properties["proj:transform"] == original.properties["proj:transform"]
        assert got.ext.proj.epsg == original.ext.proj.epsg
        assert "output" in got.assets
        assert got.assets["output"].href == original.assets["output"].href
        assert got.assets["output"].extra_fields.get("eo:bands") == \
            original.assets["output"].extra_fields.get("eo:bands")


def test_items_to_stac_geoparquet_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="no items"):
        stac_geoparquet.items_to_stac_geoparquet([], str(tmp_path / "catalog.parquet"))
