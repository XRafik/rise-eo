"""Tests for rise_eo.sources.mpc (spec 32/34).

No network — duck-typed fake items (mirrors tests/test_cdse.py's `_FakeItem`).
"""

import datetime
import types

import geopandas as gpd
import pytest
import shapely.geometry as sg
from pystac.extensions.raster import RasterExtension

from rise_eo import collections as _collections
from rise_eo import config
from rise_eo.catalog.declaration import S2_L2A_DECLARATION
from rise_eo.sources import _s2_radiometry, mpc


def _product_uri(id, dt, baseline, mgrs_tile, generation_time):
    """A valid ESA product name for a fake item, unique per id (spec 59 D3 needs one)."""
    import hashlib

    disc = generation_time.replace("-", "").replace(":", "").rstrip("Z") if generation_time \
        else "20240101T" + f"{int(hashlib.sha1(id.encode()).hexdigest(), 16) % 10**6:06d}"
    base = (baseline or "05.09").replace(".", "")
    return (f"S2B_MSIL2A_{dt:%Y%m%dT%H%M%S}_N{base}_R122_{mgrs_tile or 'T33UWP'}_{disc}.SAFE")


def _canon(item):
    """The canonical granule name (spec 59 D3) of a fake item."""
    return item.properties["s2:product_uri"].removesuffix(".SAFE")


def _granule_dir(root, item):
    """Where spec 59 D2 files this item: {root}/{collection}/YYYY/MM/DD/{canonical}/."""
    d = item.datetime
    return (root / config.SATELLITE_S2L2A / f"{d:%Y}" / f"{d:%m}" / f"{d:%d}" / _canon(item))


class _FakeItem:
    """Duck-typed stand-in for an MPC pystac `Item` (no network)."""

    def __init__(self, id, dt, geom, cloud, baseline, mgrs_tile=None, assets=None,
                 generation_time=None, processing_version=None, product_uri=None):
        self.id = id
        self.datetime = dt
        self.geometry = sg.mapping(geom)
        self.properties = {"eo:cloud_cover": cloud}
        if baseline is not None:
            self.properties["s2:processing_baseline"] = baseline
        if processing_version is not None:
            self.properties["processing:version"] = processing_version
        if mgrs_tile is not None:
            self.properties["s2:mgrs_tile"] = mgrs_tile
        if generation_time is not None:
            self.properties["s2:generation_time"] = generation_time
        if product_uri is not False:
            self.properties["s2:product_uri"] = product_uri or _product_uri(
                id, dt, baseline, mgrs_tile, generation_time)
        self.assets = {k: types.SimpleNamespace(href=v) for k, v in (assets or {}).items()}


def _fake_item(id, dt, lon, lat, cloud, baseline="05.09", mgrs_tile=None, assets=None,
                generation_time=None, processing_version=None, product_uri=None):
    if assets is None:
        assets = {"B04": f"https://example/{id}/B04.tif?sig=abc"}
    dt = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00"))
    return _FakeItem(id, dt, sg.box(lon, lat, lon + 1, lat + 1), cloud, baseline,
                     mgrs_tile=mgrs_tile, assets=assets, generation_time=generation_time,
                     processing_version=processing_version, product_uri=product_uri)


# --- baseline -> offset (spec 34 §1, generalizing spec 32 D2/D3) -------------


def test_baseline_tuple_parses_major_minor():
    assert _s2_radiometry.baseline_tuple("04.00") == (4, 0)
    assert _s2_radiometry.baseline_tuple("05.09") == (5, 9)
    assert _s2_radiometry.baseline_tuple("02.14") == (2, 14)


def test_offset_for_item_pre_and_post_04():
    pre = _fake_item("pre", "2021-06-01T00:00:00Z", 0, 0, 5.0, baseline="02.14")
    post = _fake_item("post", "2022-06-01T00:00:00Z", 0, 0, 5.0, baseline="04.00")
    assert mpc.offset_for_item(pre) == 0
    assert mpc.offset_for_item(post) == -1000


def test_offset_for_item_reprocessed_pre_2022_date_still_yields_offset():
    # the date-vs-baseline trap: an old acquisition reprocessed at a >=04.00
    # baseline must still get the offset (keyed on baseline, not date).
    reprocessed = _fake_item(
        "old-but-reprocessed", "2019-01-01T00:00:00Z", 0, 0, 5.0, baseline="05.09",
    )
    assert mpc.offset_for_item(reprocessed) == -1000


def test_offset_for_item_missing_baseline_raises():
    it = _fake_item("no-baseline", "2021-06-01T00:00:00Z", 0, 0, 5.0, baseline=None)
    with pytest.raises(ValueError, match="s2:processing_baseline"):
        mpc.offset_for_item(it)


# --- provider-specific baseline property (spec 34 §3a Amendment A1) ----------


def test_offset_for_item_resolves_from_processing_version_alone():
    pre = _fake_item(
        "cdse-pre", "2021-06-01T00:00:00Z", 0, 0, 5.0, baseline=None,
        processing_version="02.14",
    )
    post = _fake_item(
        "cdse-post", "2022-06-01T00:00:00Z", 0, 0, 5.0, baseline=None,
        processing_version="05.10",
    )
    assert mpc.offset_for_item(pre) == 0
    assert mpc.offset_for_item(post) == -1000


def test_offset_for_item_prefers_s2_processing_baseline_when_both_present():
    # both properties present and disagreeing — s2:processing_baseline wins
    # (pins the ordering as a decision, not an accident).
    it = _fake_item(
        "both", "2022-06-01T00:00:00Z", 0, 0, 5.0, baseline="02.14",
        processing_version="05.10",
    )
    assert mpc.offset_for_item(it) == 0


def test_offset_for_item_missing_both_baseline_props_raises():
    it = _fake_item(
        "no-baseline-either", "2021-06-01T00:00:00Z", 0, 0, 5.0, baseline=None,
        processing_version=None,
    )
    with pytest.raises(ValueError, match="s2:processing_baseline"):
        mpc.offset_for_item(it)


# --- items -> catalog gdf -----------------------------------------------------


def test_items_to_gdf_carries_offset_and_nodata():
    items = [
        _fake_item("pre", "2021-06-01T00:00:00Z", 16.0, 48.0, 5.0, baseline="02.14"),
        _fake_item("post", "2022-06-01T00:00:00Z", 16.0, 48.0, 5.0, baseline="04.00"),
    ]
    gdf = mpc._items_to_gdf(items, collection=config.SATELLITE_S2L2A,
                         declaration=_collections.get(config.SATELLITE_S2L2A))
    assert list(gdf["id"]) == [f"S2B_MSIL2A_20210601T000000_N0214_R122_T33UWP_{d}"
                               for d in [gdf["id"].iloc[0][-15:]]] + [gdf["id"].iloc[1]]
    assert list(gdf["offset"]) == [0, -1000]
    assert list(gdf["nodata"]) == [0, 0]
    assert gdf.crs.to_epsg() == 4326
    assert str(gdf["timestamp"].dt.tz) == "UTC"


# --- per-acquisition selection (spec 33 rule, generalized by spec 59 D7) -----


def _select(items, processing="latest"):
    from rise_eo.catalog import processing as processing_module

    gdf = mpc._items_to_gdf(items, collection=config.SATELLITE_S2L2A,
                            declaration=_collections.get(config.SATELLITE_S2L2A))
    return processing_module.select_processing(gdf, processing).kept


def test_selection_no_duplicates_is_noop():
    items = [
        _fake_item("a", "2021-06-01T00:00:00Z", 0, 0, 1.0, mgrs_tile="T33UWP"),
        _fake_item("b", "2021-06-08T00:00:00Z", 0, 0, 1.0, mgrs_tile="T33UWP"),
    ]
    assert len(_select(items)) == 2


def test_selection_duplicate_pair_latest_generation_time_wins_in_any_order():
    same_dt = "2022-03-01T10:00:29Z"
    original = _fake_item("o", same_dt, 0, 0, 1.0, mgrs_tile="T33UWP",
                          generation_time="2022-03-03T18:25:40Z")
    reprocessed = _fake_item("r", same_dt, 0, 0, 1.0, mgrs_tile="T33UWP",
                             generation_time="2024-06-04T18:03:22Z")
    for order in ([original, reprocessed], [reprocessed, original]):
        out = _select(order)
        assert list(out["id"]) == [
            "S2B_MSIL2A_20220301T100029_N0509_R122_T33UWP_20240604T180322"]


def test_selection_three_way_group_latest_wins_regardless_of_order():
    same_dt = "2022-03-01T10:00:29Z"
    v = [_fake_item(f"v{i}", same_dt, 0, 0, 1.0, mgrs_tile="T33UWP", generation_time=g)
         for i, g in enumerate(["2022-03-03T18:25:40Z", "2023-01-01T00:00:00Z",
                                "2024-06-04T18:03:22Z"], 1)]
    for order in (v, [v[2], v[0], v[1]], [v[1], v[2], v[0]]):
        out = _select(order)
        assert list(out["processing_datetime"].dt.year) == [2024]


def test_generation_time_comes_from_the_property_not_the_name():
    """The row's `processing_datetime` is `s2:generation_time` (spec 59 D8)."""
    it = _fake_item("x", "2022-03-01T10:00:29Z", 0, 0, 1.0,
                    generation_time="2024-06-04T18:03:22Z")
    gdf = mpc._items_to_gdf([it], collection=config.SATELLITE_S2L2A,
                            declaration=_collections.get(config.SATELLITE_S2L2A))
    assert gdf["processing_datetime"].iloc[0].isoformat().startswith("2024-06-04T18:03:22")


def test_missing_product_uri_raises_never_falls_back_to_the_mpc_id():
    """spec 59 section 6: an id without the baseline field would silently break the
    cross-source equality of the canonical name."""
    it = _fake_item("MPC-ID-WITHOUT-BASELINE", "2022-03-01T10:00:29Z", 0, 0, 1.0,
                    product_uri=False)
    with pytest.raises(ValueError, match="s2:product_uri"):
        mpc._items_to_gdf([it], collection=config.SATELLITE_S2L2A,
                          declaration=_collections.get(config.SATELLITE_S2L2A))


def test_select_item_files_maps_requested_bands_to_asset_hrefs(tmp_path):
    it = _fake_item(
        "t1", "2021-06-01T00:00:00Z", 0, 0, 1.0,
        assets={"B04": "https://example/t1/B04.tif?sig=1",
                "SCL": "https://example/t1/SCL.tif?sig=2"},
    )
    selected = mpc._select_item_files(it, ["B04", "SCL"], str(tmp_path))
    folder = _granule_dir(tmp_path, it)
    assert selected == [
        ("https://example/t1/B04.tif?sig=1", str(folder / "B04.tif"), "B04"),
        ("https://example/t1/SCL.tif?sig=2", str(folder / "SCL.tif"), "SCL"),
    ]


def test_select_item_files_raises_on_a_band_the_item_lacks(tmp_path):
    """spec 58 D8: a missing band raises, naming the band and collection -- it used to
    silently skip, which let a cube quietly build with a band missing."""
    it = _fake_item(
        "t1", "2021-06-01T00:00:00Z", 0, 0, 1.0,
        assets={"B04": "https://example/t1/B04.tif?sig=1"},
    )
    with pytest.raises(ValueError, match="B02.*not available"):
        mpc._select_item_files(it, ["B04", "B02"], str(tmp_path))


def test_finalize_filters_cloud_and_roi_reused_from_cdse():
    items = [
        _fake_item("hit", "2021-06-01T00:00:00Z", 0.0, 0.0, 10.0),
        _fake_item("cloudy", "2021-06-01T00:00:00Z", 0.0, 0.0, 90.0),
    ]
    gdf = mpc._items_to_gdf(items, collection=config.SATELLITE_S2L2A,
                         declaration=_collections.get(config.SATELLITE_S2L2A))
    roi = gpd.GeoDataFrame(geometry=[sg.box(0.2, 0.2, 0.5, 0.5)], crs="EPSG:4326")
    out = mpc._finalize_catalog_gdf(gdf, roi, max_cloudcover=50.0)
    assert list(out["id"]) == [_canon(items[0])]


# --- download (byte-copy + GDAL tag stamp, spec 34 §3) -----------------------


def _write_fake_cog(path, value=100, nodata=None):
    """A minimal real single-band uint16 GeoTIFF — stand-in for an MPC asset,
    so `stamp_or_reencode` (real GDAL open) has something valid to open."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        str(path), "w", driver="GTiff", height=2, width=2, count=1,
        dtype="uint16", crs="EPSG:32633", transform=from_origin(0, 2, 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(np.full((1, 2, 2), value, dtype="uint16"))


def test_transfer_one_skips_existing_final(tmp_path):
    dst = tmp_path / "B04.tif"
    dst.write_bytes(b"already-here")
    ok, reason = mpc._transfer_and_stamp_one(
        "https://example/B04.tif", str(dst), band="B04", offset=0,
        declaration=_collections.get(config.SATELLITE_S2L2A),
    )
    assert ok is True
    assert reason == "skipped"


def test_transfer_and_stamp_one_stamps_reflectance_offset_and_nodata(tmp_path, monkeypatch):
    dst = tmp_path / "B04.tif"

    def _fake_transfer(src_url, dst_url, **kw):
        _write_fake_cog(dst_url, value=1500)

    monkeypatch.setattr(mpc.fs, "transfer", _fake_transfer)
    ok, reason = mpc._transfer_and_stamp_one(
        "https://example/B04.tif", str(dst), band="B04", offset=-1000,
        declaration=_collections.get(config.SATELLITE_S2L2A),
    )
    assert ok is True and reason == "ok"
    import rasterio

    with rasterio.open(str(dst)) as d:
        # reflectance-unit tag (spec 34 §1a): -1000 DN * 1/10000 -> -0.1, paired with
        # scale=1/10000 so unscale=true yields physical reflectance (not the black-tile
        # unit mismatch that stamped -1000 alongside scale=1/10000).
        assert d.offsets[0] == pytest.approx(-1000 * config.S2_REFLECTANCE_SCALE)
        assert d.scales[0] == pytest.approx(config.S2_REFLECTANCE_SCALE)
        assert d.nodata == 0  # stamped, was missing


def test_transfer_and_stamp_one_never_offsets_mask_band(tmp_path, monkeypatch):
    dst = tmp_path / "SCL.tif"

    def _fake_transfer(src_url, dst_url, **kw):
        _write_fake_cog(dst_url, value=4)

    monkeypatch.setattr(mpc.fs, "transfer", _fake_transfer)
    ok, _ = mpc._transfer_and_stamp_one(
        "https://example/SCL.tif", str(dst), band="SCL", offset=-1000,
        declaration=_collections.get(config.SATELLITE_S2L2A),
    )
    assert ok is True
    import rasterio

    with rasterio.open(str(dst)) as d:
        assert d.offsets[0] == 0  # SCL is never radiometrically offset


def _reprocessing_pair_plus_control(cloud=5.0):
    """The real spec-32 runbook duplicate pair (fabricated generation_times
    matching the real 20220303/20240604 ordering) plus one distinct control
    item, all in the ROI used by these tests."""
    same_dt = "2022-03-01T10:00:29Z"
    original = _fake_item(
        "S2B_MSIL2A_20220301T100029_R122_T33UWP_20220303T182540", same_dt, 0.0, 0.0, cloud,
        mgrs_tile="T33UWP", generation_time="2022-03-03T18:25:40Z",
        assets={"B04": "https://example/orig/B04.tif?sig=1"},
    )
    reprocessed = _fake_item(
        "S2B_MSIL2A_20220301T100029_R122_T33UWP_20240604T180322", same_dt, 0.0, 0.0, cloud,
        mgrs_tile="T33UWP", generation_time="2024-06-04T18:03:22Z",
        assets={"B04": "https://example/reproc/B04.tif?sig=2"},
    )
    control = _fake_item(
        "control", "2022-06-01T00:00:00Z", 0.0, 0.0, cloud, mgrs_tile="T34UWA",
        assets={"B04": "https://example/control/B04.tif?sig=3"},
    )
    return [original, reprocessed, control]


def test_query_catalog_drops_the_duplicate(monkeypatch):
    items = _reprocessing_pair_plus_control()
    monkeypatch.setattr(mpc, "_search_items", lambda *a, **k: items)
    # `download()` discovers unsigned now (spec: sign per transfer, not at discovery),
    # so stub both search entry points -- `query_catalog` still uses the signed one.
    monkeypatch.setattr(mpc, "_search_items_unsigned", lambda *a, **k: items)

    roi = gpd.GeoDataFrame(geometry=[sg.box(0.2, 0.2, 0.5, 0.5)], crs="EPSG:4326")
    gdf = mpc.query_catalog(roi, datetime.datetime(2021, 1, 1), datetime.datetime(2022, 12, 31))

    assert len(gdf) == 2
    assert set(gdf["id"]) == {_canon(items[1]), _canon(items[2])}


def test_download_drops_the_duplicate_before_transfer(monkeypatch, tmp_path):
    items = _reprocessing_pair_plus_control()
    monkeypatch.setattr(mpc, "_search_items", lambda *a, **k: items)
    # `download()` discovers unsigned now (spec: sign per transfer, not at discovery),
    # so stub both search entry points -- `query_catalog` still uses the signed one.
    monkeypatch.setattr(mpc, "_search_items_unsigned", lambda *a, **k: items)

    written = []

    def _fake_transfer(src_url, dst_url, **kw):
        written.append((src_url, dst_url))
        import os
        os.makedirs(os.path.dirname(dst_url), exist_ok=True)
        _write_fake_cog(dst_url)

    monkeypatch.setattr(mpc.fs, "transfer", _fake_transfer)

    from rise_eo.catalog.catalog import TileCatalog

    catalog = TileCatalog(str(tmp_path / "catalog.parquet"))
    roi = gpd.GeoDataFrame(geometry=[sg.box(0.2, 0.2, 0.5, 0.5)], crs="EPSG:4326")

    result = mpc.download(
        roi, datetime.datetime(2021, 1, 1), datetime.datetime(2022, 12, 31),
        ["B04"], str(tmp_path / "imagery"), catalog, max_tiles=10,
    )
    assert result.successful_count == 2  # winner + control, never the loser

    gdf = catalog.read()
    assert set(gdf["id"]) == {_canon(items[1]), _canon(items[2])}
    # loser's asset href was never even queued for transfer
    written_srcs = {src for src, _ in written}
    assert "https://example/orig/B04.tif?sig=1" not in written_srcs
    assert "https://example/reproc/B04.tif?sig=2" in written_srcs
    assert "https://example/control/B04.tif?sig=3" in written_srcs


def test_download_end_to_end_mocked(monkeypatch, tmp_path):
    items = [
        _fake_item(
            "pre", "2021-06-01T00:00:00Z", 0.0, 0.0, 5.0, baseline="02.14",
            assets={"B04": "https://example/pre/B04.tif?sig=1"},
        ),
        _fake_item(
            "post", "2022-06-01T00:00:00Z", 0.0, 0.0, 5.0, baseline="04.00",
            assets={"B04": "https://example/post/B04.tif?sig=2"},
        ),
    ]
    monkeypatch.setattr(mpc, "_search_items", lambda *a, **k: items)
    # `download()` discovers unsigned now (spec: sign per transfer, not at discovery),
    # so stub both search entry points -- `query_catalog` still uses the signed one.
    monkeypatch.setattr(mpc, "_search_items_unsigned", lambda *a, **k: items)

    written = []

    def _fake_transfer(src_url, dst_url, **kw):
        written.append((src_url, dst_url))
        import os
        os.makedirs(os.path.dirname(dst_url), exist_ok=True)
        _write_fake_cog(dst_url)

    monkeypatch.setattr(mpc.fs, "transfer", _fake_transfer)

    from rise_eo.catalog.catalog import TileCatalog

    catalog_fp = str(tmp_path / "catalog.parquet")
    catalog = TileCatalog(catalog_fp)
    roi = gpd.GeoDataFrame(geometry=[sg.box(0.2, 0.2, 0.5, 0.5)], crs="EPSG:4326")

    result = mpc.download(
        roi, datetime.datetime(2021, 1, 1), datetime.datetime(2022, 12, 31),
        ["B04"], str(tmp_path / "imagery"), catalog, max_tiles=10,
    )
    assert result.successful_count == 2
    assert result.failed_count == 0
    assert len(written) == 2

    gdf = catalog.read()
    assert set(gdf["id"]) == {_canon(items[0]), _canon(items[1])}
    offsets = dict(zip(gdf["id"], gdf["offset"]))
    assert offsets == {_canon(items[0]): 0, _canon(items[1]): -1000}


def test_download_accepts_remote_root_and_stamps_via_local_scratch(tmp_path, monkeypatch):
    """spec 34 §3/§5: lifts spec 32's local-only guard — a remote (here,
    fsspec `memory://`, standing in for `abfss://`) root_folderpath must still
    get a stamped COG, staged through local scratch."""
    items = [
        _fake_item("t1", "2022-06-01T00:00:00Z", 0.0, 0.0, 5.0, baseline="04.00",
                   assets={"B04": "https://example/t1/B04.tif?sig=1"}),
    ]
    monkeypatch.setattr(mpc, "_search_items", lambda *a, **k: items)
    # `download()` discovers unsigned now (spec: sign per transfer, not at discovery),
    # so stub both search entry points -- `query_catalog` still uses the signed one.
    monkeypatch.setattr(mpc, "_search_items_unsigned", lambda *a, **k: items)

    def _fake_transfer(src_url, dst_url, **kw):
        _write_fake_cog(dst_url, value=1500)

    monkeypatch.setattr(mpc.fs, "transfer", _fake_transfer)

    from rise_eo.catalog.catalog import TileCatalog

    catalog = TileCatalog(str(tmp_path / "catalog.parquet"))
    roi = gpd.GeoDataFrame(geometry=[sg.box(0.2, 0.2, 0.5, 0.5)], crs="EPSG:4326")
    remote_root = "memory://fsd-mpc-test/imagery"

    result = mpc.download(
        roi, datetime.datetime(2021, 1, 1), datetime.datetime(2022, 12, 31),
        ["B04"], remote_root, catalog, max_tiles=10,
    )
    assert result.successful_count == 1
    assert result.failed_count == 0

    import fsspec

    memfs = fsspec.filesystem("memory")
    d = items[0].datetime
    assert memfs.exists(f"fsd-mpc-test/imagery/{config.SATELLITE_S2L2A}/{d:%Y/%m/%d}/"
                        f"{_canon(items[0])}/B04.tif")


def test_gdal_tag_and_stac_raster_bands_agree(tmp_path, monkeypatch):
    """spec 34 §4: ingest writes offset/scale/nodata to **both** the COG's GDAL tag
    and STAC `raster:bands`, with equal values — the two declarations are written
    from the same source of truth, so a viewer reading the tag (`unscale=true`) and
    a tool reading the STAC item can never disagree.

    Drives the real ingest stamp (`mpc._transfer_and_stamp_one`) and the real STAC
    export (`stac.tile_catalog_to_items`) over the same declared offset, rather than
    asserting each side separately against a literal.
    """
    import geopandas as gpd
    import pandas as pd
    import rasterio
    import shapely.geometry

    from rise_eo.catalog import stac

    offset = -1000  # baseline >= 04.00
    dst = tmp_path / "B04.tif"

    def _fake_transfer(src_url, dst_url, **kw):
        _write_fake_cog(dst_url, value=1500)

    monkeypatch.setattr(mpc.fs, "transfer", _fake_transfer)
    ok, _ = mpc._transfer_and_stamp_one(
        "https://example/B04.tif", str(dst), band="B04", offset=offset,
        declaration=_collections.get(config.SATELLITE_S2L2A),
    )
    assert ok is True

    row = {
        "id": "S2A_MSIL2A_20220601T075611_N0500_R035_T33UWP_20220601T120000",
        "collection": "sentinel-2-l2a",
        "timestamp": pd.Timestamp("2022-06-01T07:56:11", tz="UTC"),
        "s3url": "", "local_folderpath": str(tmp_path), "files": "B04.tif",
        "cloud_cover": 0.0, "offset": offset, "nodata": 0,
        "geometry": shapely.geometry.box(15.0, 48.0, 15.4, 48.4),
    }
    item = stac.tile_catalog_to_items(
        gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326")
    )[0]
    stac_band = RasterExtension.ext(item.assets["B04"]).bands[0]

    with rasterio.open(str(dst)) as d:
        # both declarations carry the SAME reflectance-unit offset (-1000 DN * 1/10000
        # = -0.1), not the DN offset — so a viewer (GDAL tag) and a STAC reader agree
        # AND are unit-consistent with scale=1/10000 (spec 34 §1a).
        expected_refl_offset = offset * config.S2_REFLECTANCE_SCALE
        assert d.offsets[0] == pytest.approx(stac_band.offset)
        assert d.offsets[0] == pytest.approx(expected_refl_offset)
        assert d.scales[0] == pytest.approx(stac_band.scale)
        assert d.nodata == stac_band.nodata == 0


def test_stamped_tag_unscales_to_physical_reflectance_not_black(tmp_path, monkeypatch):
    """Regression for the black-tile bug found in runbook 34b (2026-07-20). The GDAL
    SCALE/OFFSET a viewer's `unscale=true` reads must be UNIT-CONSISTENT: unscale
    computes ``DN*scale + offset``, and with ``scale=1/10000`` (physical reflectance,
    spec 34 §1a) the stamped offset must be reflectance-unit too. The bug stamped the
    raw DN offset (-1000) alongside ``scale=1/10000``, so unscale gave
    ``1500/10000 - 1000 ~= -1000`` for *every* pixel → every tile rendered pure black.
    This asserts the actual unscale arithmetic, which the agreement test above cannot
    (both sides shared the same wrong value, so they agreed while both being wrong)."""
    import rasterio

    dst = tmp_path / "B04.tif"
    monkeypatch.setattr(mpc.fs, "transfer", lambda s, d, **k: _write_fake_cog(d, value=1500))
    ok, _ = mpc._transfer_and_stamp_one(
        "https://example/B04.tif", str(dst), band="B04", offset=-1000,
        declaration=_collections.get(config.SATELLITE_S2L2A),
    )
    assert ok is True
    with rasterio.open(str(dst)) as d:
        unscaled = 1500 * d.scales[0] + d.offsets[0]   # what titiler unscale=true computes
        assert unscaled == pytest.approx((1500 - 1000) / 10000)   # 0.05 reflectance
        assert 0.0 <= unscaled <= 1.0                              # sane, NOT ~-1000 (black)


def test_stac_roundtrip_preserves_dn_offset_for_builder(tmp_path):
    """The catalog `offset` column is DN-unit (the builder applies it in DN space,
    `clip(DN + offset)`), but raster:bands stores it reflectance-unit (spec 34 §1a).
    A ``to_stac`` → ``items_to_rows`` round-trip must recover the DN offset (-1000), or
    a datacube built from a re-imported catalog would silently be ~1000 DN high — the
    exact #10/#30 failure spec 34 exists to close."""
    import geopandas as gpd
    import pandas as pd
    import shapely.geometry

    from rise_eo.catalog import stac

    row = {
        "id": "S2A_MSIL2A_20220601T075611_N0500_R035_T33UWP_20220601T120000",
        "collection": "sentinel-2-l2a",
        "timestamp": pd.Timestamp("2022-06-01T07:56:11", tz="UTC"),
        "s3url": "", "local_folderpath": str(tmp_path), "files": "B04.tif",
        "cloud_cover": 0.0, "offset": -1000, "nodata": 0,
        "geometry": shapely.geometry.box(15.0, 48.0, 15.4, 48.4),
    }
    items = stac.tile_catalog_to_items(
        gpd.GeoDataFrame([row], geometry="geometry", crs="EPSG:4326")
    )
    back = stac.items_to_rows(items)
    assert back.iloc[0]["offset"] == pytest.approx(-1000)   # DN-unit recovered, not -0.1


# --- the [mpc] extra names itself, like [grid]/[local]/[s3] ------------------------
# `planetary_computer` is imported lazily at two points, so without this a user who
# installed fsd without `[mpc]` sees a bare ModuleNotFoundError naming a package they
# never typed -- from a call they made as source="mpc".


def test_missing_planetary_computer_names_the_mpc_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_pc(name, *args, **kwargs):
        if name == "planetary_computer":
            raise ImportError("No module named 'planetary_computer'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(__import__("sys").modules, "planetary_computer", raising=False)
    monkeypatch.setattr(builtins, "__import__", _no_pc)
    with pytest.raises(ImportError, match=r"fsd\[mpc\]"):
        mpc._import_pc()


def test_import_pc_sign_goes_through_the_same_guard(monkeypatch):
    """`_import_pc_sign` is the download-path entry; it must not bypass the message."""
    monkeypatch.setattr(mpc, "_import_pc", lambda: types.SimpleNamespace(sign="SIGN"))
    assert mpc._import_pc_sign() == "SIGN"


def test_download_prints_why_transfers_failed_not_just_how_many(capsys):
    """A failed download must say WHY, grouped by reason.

    `DownloadResult.failures` has always carried `(src_url, reason)`, but nothing printed
    it: the progress line shows `fail=N`, and `api.download` discards the result entirely.
    A 2026-09-05 run of `runbooks/58-redownload-austria-mpc.md` lost 393 of 552 files and
    left no way to distinguish throttling from an expired token from a network fault --
    the diagnosis was in memory and thrown away.

    Grouped by distinct reason, because a throttled run yields hundreds of copies of one
    message; the distinct set is the finding, the counts are the scale.
    """
    # Reasons carry the exception TYPE first (`_failure_reason`), and the message is the
    # asset url -- unique per file. Grouping must key on the type or nothing collapses.
    failures = [
        (f"https://x/{i}/B04.tif", f"FileNotFoundError: https://x/{i}/B04.tif")
        for i in range(300)
    ] + [
        (f"https://x/{i}/B08.tif", "TimeoutError: read timed out") for i in range(9)
    ]
    mpc._print_failure_summary(failures, total=552)
    out = capsys.readouterr().out

    assert "309/552 transfers FAILED" in out
    assert "300 x FileNotFoundError" in out
    assert "9 x TimeoutError" in out
    # One example message + url per kind: the count says how bad, the example says what to
    # do about it, and the url is what you retry or paste into a bug report.
    assert "e.g. TimeoutError: read timed out" in out
    assert "https://x/0/B04.tif" in out


def test_failure_summary_is_silent_when_nothing_failed(capsys):
    """A clean run must not print a failure block at all."""
    mpc._print_failure_summary([], total=10)
    assert capsys.readouterr().out == ""


def test_failure_summary_truncates_a_multiline_reason(capsys):
    """A rasterio/adlfs failure arrives as a whole traceback string, kilobytes long. Only
    the first line is the diagnosis, so the summary stays readable at 300 failures."""
    reason = "RuntimeError: boom\n" + "\n".join(f"  frame {i}" for i in range(50))
    mpc._print_failure_summary([("https://x/a.tif", reason)], total=1)
    out = capsys.readouterr().out

    assert "1 x RuntimeError" in out
    assert "e.g. RuntimeError: boom" in out
    assert "frame 7" not in out


def test_failure_reason_puts_the_exception_type_first():
    """`str(exc)` alone is useless on the real failure path: fsspec/adlfs raise
    `FileNotFoundError(url)`, so the message IS the asset url -- unique per file. A
    2026-09-06 run reported 74 failures as 74 one-off "reasons", each a different url.
    The type is the diagnosis, so it goes first."""
    assert mpc._failure_reason(FileNotFoundError("http://x/a.tif")) == \
        "FileNotFoundError: http://x/a.tif"
    # An exception with no message must not produce a dangling "OSError: ".
    assert mpc._failure_reason(OSError()) == "OSError"
    assert mpc._failure_reason(None) == "unknown"


def test_failure_kind_groups_on_the_type_not_the_message():
    """The grouping key is the type, so N failures whose messages are N distinct urls
    collapse to one line."""
    urls = [f"FileNotFoundError: http://x/{i}.tif" for i in range(50)]
    assert {mpc._failure_kind(u) for u in urls} == {"FileNotFoundError"}
    assert mpc._failure_kind("OSError") == "OSError"
    # A reason with no type prefix keeps its own text -- never mangled into a fake type.
    assert mpc._failure_kind("connection reset by peer") == "connection reset by peer"
    assert mpc._failure_kind("") == "unknown"


def test_transfer_signs_per_attempt_so_a_retry_never_reuses_a_dead_token(monkeypatch, tmp_path):
    """`sign` runs inside the retry loop, once per attempt -- not once per submission.

    This is the fix for the 2026-09-05 archive run: an MPC SAS token lives ~45 min, and
    `download()` signed every href at DISCOVERY, so 159 of 552 files landed over 44 minutes
    and the remaining 393 -- still queued behind them -- failed at once when the token aged
    out. A retry that reuses the same expired token is the same bug one level down, so the
    signature is minted per attempt.
    """
    attempts = []

    def _fake_sign(url):
        attempts.append(url)
        return f"{url}?sig={len(attempts)}"

    transferred = []

    def _flaky_transfer(src, dst):
        transferred.append(src)
        if len(transferred) < 3:
            raise RuntimeError("403 Server failed to authenticate")
        open(dst, "wb").write(b"cog")

    monkeypatch.setattr(mpc.fs, "transfer", _flaky_transfer)
    monkeypatch.setattr(mpc, "stamp_or_reencode", lambda *a, **kw: None)

    ok, reason = mpc._transfer_and_stamp_one(
        "https://mpc/B04.tif", str(tmp_path / "B04.tif"),
        band="B04", offset=0, declaration=S2_L2A_DECLARATION,
        sign=_fake_sign, tries=3, base_delay=0,
    )

    assert (ok, reason) == (True, "ok")
    # Three attempts, three DISTINCT signatures -- not one token reused three times.
    assert transferred == [
        "https://mpc/B04.tif?sig=1",
        "https://mpc/B04.tif?sig=2",
        "https://mpc/B04.tif?sig=3",
    ]


def test_transfer_without_a_signer_passes_the_url_through(monkeypatch, tmp_path):
    """`sign=None` (CDSE, a local file, any already-signed url) must not be touched."""
    seen = []
    monkeypatch.setattr(mpc.fs, "transfer",
                        lambda src, dst: (seen.append(src), open(dst, "wb").write(b"cog")))
    monkeypatch.setattr(mpc, "stamp_or_reencode", lambda *a, **kw: None)

    ok, _ = mpc._transfer_and_stamp_one(
        "https://mpc/B04.tif", str(tmp_path / "B04.tif"),
        band="B04", offset=0, declaration=S2_L2A_DECLARATION,
    )
    assert ok and seen == ["https://mpc/B04.tif"]
