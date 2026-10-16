"""Tests for rise_eo.catalog.TileCatalog (spec 02).

Covers: append -> read round-trip, `files` union on re-append, and `filter`
(inclusive date range + spatial overlap + area_contribution).
"""

import datetime

import geopandas as gpd
import pytest
import shapely.geometry as sg

from rise_eo import config
from rise_eo.catalog.catalog import TileCatalog


def _box(x0, y0, x1, y1):
    return sg.box(x0, y0, x1, y1)


def _row(id, ts, files, geom, **kw):
    base = dict(
        id=id,
        collection=config.SATELLITE_S2L2A,
        timestamp=ts,
        s3url=f"s3://eodata/{id}.SAFE",
        local_folderpath=f"/data/{id}",
        files=files,
        cloud_cover=0.0,
        geometry=geom,
    )
    base.update(kw)
    return base


def test_append_read_roundtrip(tmp_path):
    cat = TileCatalog(str(tmp_path / "catalog.parquet"))
    cat.append(
        [
            _row("t1", "2024-08-01T10:00:00Z", "B02.jp2,SCL.jp2", _box(0, 0, 1, 1), offset=0, nodata=0),
            _row("t2", "2024-08-05T10:00:00Z", "B02.jp2", _box(1, 1, 2, 2), offset=0, nodata=0),
        ]
    )
    gdf = cat.read()
    assert len(gdf) == 2
    assert set(gdf["id"]) == {"t1", "t2"}
    assert gdf.crs.to_epsg() == 4326
    assert str(gdf["timestamp"].dt.tz) == "UTC"


def test_files_union_on_reappend(tmp_path):
    cat = TileCatalog(str(tmp_path / "catalog.parquet"))
    cat.append([_row("t1", "2024-08-01T10:00:00Z", "B02.jp2", _box(0, 0, 1, 1), offset=0, nodata=0)])
    # Re-download more bands for the same tile + a higher cloud_cover value.
    cat.append(
        [
            _row(
                "t1",
                "2024-08-01T10:00:00Z",
                "B08.jp2,SCL.jp2",
                _box(0, 0, 1, 1),
                cloud_cover=12.5,
                offset=0,
                nodata=0,
            )
        ]
    )
    gdf = cat.read()
    assert len(gdf) == 1  # upsert, not duplicate
    files = gdf.loc[gdf["id"] == "t1", "files"].iloc[0]
    assert files == "B02.jp2,B08.jp2,SCL.jp2"  # unioned + sorted
    # other columns take the newest value
    assert gdf.loc[gdf["id"] == "t1", "cloud_cover"].iloc[0] == 12.5


def test_filter_date_and_spatial(tmp_path):
    cat = TileCatalog(str(tmp_path / "catalog.parquet"))
    cat.append(
        [
            # in range + overlaps ROI
            _row("hit", "2024-08-10T10:00:00Z", "B02.jp2", _box(0, 0, 2, 2), offset=0, nodata=0),
            # in range but no spatial overlap
            _row("far", "2024-08-11T10:00:00Z", "B02.jp2", _box(10, 10, 11, 11), offset=0, nodata=0),
            # overlaps but before the date range
            _row("early", "2024-07-01T10:00:00Z", "B02.jp2", _box(0, 0, 2, 2), offset=0, nodata=0),
            # overlaps but after the date range
            _row("late", "2024-09-01T10:00:00Z", "B02.jp2", _box(0, 0, 2, 2), offset=0, nodata=0),
        ]
    )
    roi = gpd.GeoDataFrame(geometry=[_box(0, 0, 1, 1)], crs="EPSG:4326")
    out = cat.filter(
        roi,
        datetime.datetime(2024, 8, 1),
        datetime.datetime(2024, 8, 31),
    )
    assert list(out["id"]) == ["hit"]
    # ROI (1x1) fully inside the 2x2 tile -> 100% contribution.
    assert abs(out["area_contribution"].iloc[0] - 100.0) < 1e-6


def test_filter_partial_area_contribution(tmp_path):
    cat = TileCatalog(str(tmp_path / "catalog.parquet"))
    # Tile covers the left half of a 2x2 ROI -> 50% contribution.
    cat.append([_row("half", "2024-08-10T10:00:00Z", "B02.jp2", _box(0, 0, 1, 2), offset=0, nodata=0)])
    roi = gpd.GeoDataFrame(geometry=[_box(0, 0, 2, 2)], crs="EPSG:4326")
    out = cat.filter(
        roi,
        datetime.datetime(2024, 8, 1),
        datetime.datetime(2024, 8, 31),
    )
    assert abs(out["area_contribution"].iloc[0] - 50.0) < 1e-6


def test_filter_inclusive_bounds(tmp_path):
    cat = TileCatalog(str(tmp_path / "catalog.parquet"))
    cat.append(
        [
            _row("start", "2024-08-01T00:00:00Z", "B02.jp2", _box(0, 0, 1, 1), offset=0, nodata=0),
            _row("end", "2024-08-31T00:00:00Z", "B02.jp2", _box(0, 0, 1, 1), offset=0, nodata=0),
        ]
    )
    roi = gpd.GeoDataFrame(geometry=[_box(0, 0, 1, 1)], crs="EPSG:4326")
    out = cat.filter(
        roi,
        datetime.datetime(2024, 8, 1),
        datetime.datetime(2024, 8, 31),
    )
    assert set(out["id"]) == {"start", "end"}


# --- declaration stamping / conflict rule (spec 35 §4/§8.7) -------------------


def test_append_stamps_declaration_on_new_catalog(tmp_path):
    from rise_eo.catalog.declaration import S2_L2A_DECLARATION

    cat = TileCatalog(str(tmp_path / "catalog.parquet"))
    cat.append(
        [_row("t1", "2024-08-01T10:00:00Z", "B02.jp2", _box(0, 0, 1, 1), offset=0, nodata=0)],
        declaration=S2_L2A_DECLARATION,
    )
    assert cat.declaration == S2_L2A_DECLARATION


def test_constructor_declaration_is_appends_default(tmp_path):
    from rise_eo.catalog.declaration import S2_L2A_DECLARATION

    cat = TileCatalog(str(tmp_path / "catalog.parquet"), declaration=S2_L2A_DECLARATION)
    cat.append([_row("t1", "2024-08-01T10:00:00Z", "B02.jp2", _box(0, 0, 1, 1), offset=0, nodata=0)])
    assert cat.declaration == S2_L2A_DECLARATION


def test_append_conflicting_declaration_raises(tmp_path):
    from rise_eo.catalog.declaration import CollectionDeclaration, MaskSpec

    a = CollectionDeclaration(reference_band="B04")
    b = CollectionDeclaration(reference_band="B08", mask_spec=MaskSpec(band="SCL", classes=(1,)))

    cat = TileCatalog(str(tmp_path / "catalog.parquet"))
    cat.append([_row("t1", "2024-08-01T10:00:00Z", "B02.jp2", _box(0, 0, 1, 1), offset=0, nodata=0)], declaration=a)
    with pytest.raises(ValueError, match="declaration conflict"):
        cat.append([_row("t2", "2024-08-02T10:00:00Z", "B02.jp2", _box(1, 1, 2, 2), offset=0, nodata=0)], declaration=b)
    # the original stamp is untouched by the rejected append.
    assert cat.declaration == a


def test_append_declaration_none_preserves_existing_stamp(tmp_path):
    from rise_eo.catalog.declaration import CollectionDeclaration

    a = CollectionDeclaration(reference_band="B04")
    cat = TileCatalog(str(tmp_path / "catalog.parquet"))
    cat.append([_row("t1", "2024-08-01T10:00:00Z", "B02.jp2", _box(0, 0, 1, 1), offset=0, nodata=0)], declaration=a)
    # an rise-eo-agnostic top-up (no declaration= given) must not erase the stamp.
    cat.append([_row("t2", "2024-08-02T10:00:00Z", "B02.jp2", _box(1, 1, 2, 2), offset=0, nodata=0)])
    assert cat.declaration == a
    assert len(cat.read()) == 2


def test_append_empty_is_noop(tmp_path):
    cat = TileCatalog(str(tmp_path / "catalog.parquet"))
    cat.append([])
    from rise_eo.storage import fs

    assert not fs.exists(cat.filepath)


# --- offset / nodata (spec 34 §1, retiring spec 32's boa_add_offset) -----------


def test_offset_and_nodata_round_trip(tmp_path):
    cat = TileCatalog(str(tmp_path / "catalog.parquet"))
    cat.append([
        _row("t1", "2024-08-01T10:00:00Z", "B04.tif", _box(0, 0, 1, 1),
             offset=-1000, nodata=0),
    ])
    gdf = cat.read()
    assert gdf.loc[gdf["id"] == "t1", "offset"].iloc[0] == -1000
    assert gdf.loc[gdf["id"] == "t1", "nodata"].iloc[0] == 0


def test_read_does_not_backfill_a_legacy_catalog_missing_offset_nodata(tmp_path):
    # spec 34 [G4] / spec 58 D12: no back-compat shim — a catalog written before
    # offset/nodata/scale/properties existed is disposable (re-ingest, don't migrate),
    # so `.read()` must NOT silently patch the columns back in.
    import geopandas as gpd
    import pandas as pd

    from rise_eo.storage import fs

    # `_row()` itself produces none of offset/nodata/scale/properties (they are only
    # added by `TileCatalog.append`'s backfill or an explicit kwarg), so this gdf is
    # already exactly what a pre-spec-34/pre-spec-58 catalog would have looked like.
    row = _row("t1", "2024-08-01T10:00:00Z", "B02.jp2", _box(0, 0, 1, 1))
    gdf = gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326")
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], utc=True)
    fp = str(tmp_path / "old_catalog.parquet")
    fs.write_parquet(fp, gdf)

    # spec 59 D8/AC13: a catalog without the acquisition columns is refused, naming them
    # and saying "re-download" -- no shim.
    with pytest.raises(ValueError, match=r"acquisition_key.*re-download"):
        TileCatalog(fp).read()

    # ...but a catalog that HAS them is still not backfilled for the older columns.
    gdf["acquisition_key"] = gdf["id"]
    gdf["processing_version"] = None
    gdf["processing_datetime"] = None
    gdf["source"] = "mpc"
    fs.write_parquet(fp, gdf)
    read_back = TileCatalog(fp).read()
    assert "offset" not in read_back.columns
    assert "nodata" not in read_back.columns
    assert "scale" not in read_back.columns
    assert "properties" not in read_back.columns


def test_declaration_survives_catalog_roundtrip(tmp_path):
    """spec 35 §8.1 — footer round-trip: a non-S2 declaration stamped through
    `TileCatalog.append` survives write->read byte-identical, tuples still
    tuples (replaces the deleted TODO-#42 pin, which this closes)."""
    import geopandas as gpd

    from rise_eo.catalog.declaration import CollectionDeclaration, MaskSpec

    custom = CollectionDeclaration(
        reference_band="B04",
        mask_spec=MaskSpec(band="QA", classes=(1, 2, 3)),
        mask_keep=True,
        mosaic_method="median",
    )
    cat = TileCatalog(str(tmp_path / "catalog.parquet"), declaration=custom)
    cat.append([
        _row("t1", "2024-08-01T10:00:00Z", "B04.tif", _box(0, 0, 1, 1),
             offset=-1000, nodata=7),
    ])

    back = cat.read()
    assert isinstance(back, gpd.GeoDataFrame)
    # per-row declarations survive...
    assert back["offset"].iloc[0] == -1000
    assert back["nodata"].iloc[0] == 7
    # ...and now so does the collection-level one (spec 35, closing TODO #42).
    assert cat.declaration == custom
    assert isinstance(cat.declaration.mask_spec.classes, tuple)
