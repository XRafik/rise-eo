"""Spec 58 P2 acceptance criteria — `sentinel-1-rtc` + `properties_filter` (D9/D17).

AC numbering follows specs/58-collection-agnostic-verbs.md §5 (the amended P2 section).
AC15 is a run-book (`runbooks/58-p2-window-a.md`), not a pytest AC -- see it for the
"S1 and S2 build/create_training_data with no verb-signature difference" proof.
"""

from __future__ import annotations

import datetime
import json
import pathlib

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
from rasterio.transform import from_origin

from rise_eo import api, config
from rise_eo import collections as _collections
from rise_eo.catalog import catalog
from rise_eo.catalog import declaration as declaration_module
from rise_eo.datacube import builder
from rise_eo.sources import mpc
from rise_eo.storage import fs
from rise_eo.workflows import create_datacube


def _s1_flattened_gdf(rows: list[tuple[str, str, "shapely.geometry.base.BaseGeometry"]]):
    """A band-flattened (`flatten_catalog`-shaped) GeoDataFrame for one requested band
    (`vv`), given `(id, orbit_state, geometry)` triples -- just enough columns for
    `builder._enforce_mosaic_partition` (and, transitively, `build_datacube`'s
    pre-image-load checks) to run without any real raster file."""
    data = {
        "id": [r[0] for r in rows],
        "filepath": ["" for _ in rows],
        "band": ["vv" for _ in rows],
        "timestamp": [pd.Timestamp("2018-04-01", tz="UTC") for _ in rows],
        "geometry": [r[2] for r in rows],
        "area_contribution": [1.0 for _ in rows],
        "offset": [0 for _ in rows],
        "nodata": [-32768 for _ in rows],
        "properties": [json.dumps({"sat:orbit_state": r[1]}) for r in rows],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


# --- D17: the sentinel-1-rtc declaration ---------------------------------------------

def test_s1_rtc_declaration_matches_spec_table():
    decl = _collections.get("sentinel-1-rtc")
    assert decl.reference_band is None
    assert decl.native_grid is False
    assert decl.mask_spec is None
    assert decl.nodata == -32768
    assert decl.scale == 1.0
    assert decl.radiometry_bands == ()  # empty, not None (D17)
    assert decl.band_aliases == ()
    assert decl.supports_cloud_cover is False
    assert decl.requires_subscription_key is False
    assert decl.mosaic_partition == ("sat:orbit_state",)
    assert decl.partition_policy == "raise"


def test_s1_rtc_declaration_round_trips_through_json():
    decl = _collections.get("sentinel-1-rtc")
    assert declaration_module.from_json(declaration_module.to_json(decl)) == decl


def test_mpc_serves_sentinel_1_rtc():
    assert "sentinel-1-rtc" in mpc.SERVED_COLLECTIONS


# --- bug found while wiring S1 ingest: offset_for_item raised for non-S2 items --------

def test_mpc_items_to_gdf_does_not_derive_offset_for_a_non_radiometric_collection():
    """`_items_to_gdf` used to call `_s2_radiometry.offset_for_item` unconditionally,
    which raises `ValueError` for any item lacking S2's processing-baseline properties
    -- every sentinel-1-rtc item. A collection declaring `radiometry_bands=()` has
    nothing to derive, so ingest must not even try."""
    class _Item:
        id = "S1B_..._rtc"
        datetime = pd.Timestamp("2018-04-01", tz="UTC")
        properties = {"sat:orbit_state": "ascending"}  # no s2:processing_baseline
        geometry = shapely.geometry.mapping(shapely.geometry.box(0, 0, 1, 1))

        def get_self_href(self):
            return "https://example/x"

    decl = _collections.get("sentinel-1-rtc")
    gdf = mpc._items_to_gdf([_Item()], collection="sentinel-1-rtc", declaration=decl)
    assert gdf["offset"].iloc[0] == 0


# --- bug found while drafting the AC15 run-book: reference_band=None never built ----

def test_reference_band_none_builds_a_real_cube_not_just_declaration_shape(tmp_path):
    """D11 says `reference_band=None` (S1 RTC, HLS: bands are already grid-uniform)
    means "use the first requested band for the merge geometry, run no resample step" --
    but `build_datacube` compared `catalog_gdf["band"] == reference_band` where
    `reference_band` was still `None`, which matches NOTHING, leaving `ref_indices`
    empty and the merge failing on zero images. No P1 test built a real cube with
    `reference_band=None`; every P1 AC checked the declaration/preflight shape, not a
    pixel. This is the missing real-run proof, for the collection that actually needs it."""
    crs = "EPSG:32633"
    transform = from_origin(500000, 5000000, 10, 10)
    tile_box = shapely.geometry.box(500000, 4999960, 500040, 5000000)

    def _write(path, value):
        with rasterio.open(
            path, "w", driver="GTiff", height=4, width=4, count=1,
            dtype="float32", crs=crs, transform=transform, nodata=-32768,
        ) as dst:
            dst.write(np.full((1, 4, 4), value, dtype=np.float32))

    ts = pd.Timestamp("2018-06-01", tz="UTC")
    rows = []
    for band, val in [("vv", 0.1), ("vh", 0.05)]:
        p = tmp_path / f"{band}.tif"
        _write(p, val)
        rows.append({
            "id": "s1_item", "filepath": str(p), "band": band, "timestamp": ts,
            "geometry": tile_box, "area_contribution": 100.0,
            "properties": json.dumps({"sat:orbit_state": "ascending"}),
        })
    catalog_subset = gpd.GeoDataFrame(rows, crs=crs)
    shape_gdf = gpd.GeoDataFrame({"geometry": [tile_box]}, crs=crs)
    decl = _collections.get("sentinel-1-rtc")
    assert decl.reference_band is None  # the case under test

    out = tmp_path / "cube"
    builder.build_datacube(
        catalog_subset=catalog_subset, shape_gdf=shape_gdf,
        startdate=datetime.datetime(2018, 5, 31), enddate=datetime.datetime(2018, 6, 20),
        bands=["vv", "vh"], mosaic_days=20, declaration=decl,
        export_folderpath=str(out), if_missing_files=None,
    )
    dc = fs.load_npy(str(out / "datacube.npy"))
    md = fs.load_npy(str(out / "metadata.pickle.npy"), allow_pickle=True)[()]
    assert dc.shape == (1, 4, 4, 2)
    assert md["bands"] == ["vv", "vh"]
    assert dc[0, 0, 0, 0] == pytest.approx(0.1)
    assert dc[0, 0, 0, 1] == pytest.approx(0.05)


# --- AC11: rows spanning both orbit states raise, enumerating pairs + counts ---------

def test_ac11_mixed_orbit_states_raises_enumerating_counts_and_coverage():
    decl = _collections.get("sentinel-1-rtc")
    box1 = shapely.geometry.box(0, 0, 10, 10)
    catalog_subset = _s1_flattened_gdf([
        ("a1", "ascending", box1), ("a2", "ascending", box1),
        ("d1", "descending", box1),
    ])
    shape_gdf = gpd.GeoDataFrame({"geometry": [box1]}, crs="EPSG:4326")

    with pytest.raises(ValueError) as exc_info:
        builder.build_datacube(
            catalog_subset=catalog_subset, shape_gdf=shape_gdf,
            startdate=datetime.datetime(2018, 4, 1), enddate=datetime.datetime(2018, 9, 30),
            bands=["vv"], declaration=decl, export_folderpath="/tmp/unused-ac11",
            if_missing_files=None,
        )
    msg = str(exc_info.value)
    assert "sat:orbit_state" in msg
    assert "ascending" in msg and "descending" in msg
    assert "2 acquisition" in msg
    assert "1 acquisition" in msg
    assert "coverage" in msg.lower()


def test_ac11_single_orbit_state_does_not_raise():
    decl = _collections.get("sentinel-1-rtc")
    box1 = shapely.geometry.box(0, 0, 10, 10)
    catalog_subset = _s1_flattened_gdf([("a1", "ascending", box1), ("a2", "ascending", box1)])
    shape_gdf = gpd.GeoDataFrame({"geometry": [box1]}, crs="EPSG:4326")
    builder._enforce_mosaic_partition(catalog_subset, shape_gdf, decl)  # must not raise


def test_ac11_s2_never_enforces_partition():
    """`sentinel-2-l2a` declares `mosaic_partition=()` -- rows carrying any (or no)
    properties never trigger enforcement; P1's behaviour is unchanged."""
    decl = _collections.get(config.SATELLITE_S2L2A)
    box1 = shapely.geometry.box(0, 0, 10, 10)
    catalog_subset = _s1_flattened_gdf([("a1", "ascending", box1), ("d1", "descending", box1)])
    shape_gdf = gpd.GeoDataFrame({"geometry": [box1]}, crs="EPSG:4326")
    builder._enforce_mosaic_partition(catalog_subset, shape_gdf, decl)  # no-op, no raise


# --- AC12: properties_filter given succeeds; an uncarried key raises -----------------

def test_ac12_properties_filter_narrows_to_one_orbit_then_enforcement_succeeds():
    decl = _collections.get("sentinel-1-rtc")
    box1 = shapely.geometry.box(0, 0, 10, 10)
    catalog_subset = _s1_flattened_gdf([
        ("a1", "ascending", box1), ("a2", "ascending", box1),
        ("d1", "descending", box1),
    ])
    filtered = catalog.filter_by_properties(catalog_subset, {"sat:orbit_state": "descending"})
    assert sorted(filtered["id"]) == ["d1"]
    shape_gdf = gpd.GeoDataFrame({"geometry": [box1]}, crs="EPSG:4326")
    builder._enforce_mosaic_partition(filtered, shape_gdf, decl)  # must not raise


def test_ac12_properties_filter_uncarried_key_raises_naming_carried_keys():
    gdf = gpd.GeoDataFrame(
        {"properties": ['{"sat:orbit_state": "ascending"}', '{"sat:orbit_state": "descending"}']}
    )
    with pytest.raises(ValueError) as exc_info:
        catalog.filter_by_properties(gdf, {"sat:relative_orbit": 146})
    msg = str(exc_info.value)
    assert "sat:relative_orbit" in msg
    assert "sat:orbit_state" in msg


def test_ac12_properties_filter_value_matching_zero_rows_is_not_an_error():
    """A KNOWN key whose requested value matches nothing is an ordinary empty result,
    not the "uncarried key" error -- only a key absent from every row raises."""
    gdf = gpd.GeoDataFrame({"properties": ['{"sat:orbit_state": "ascending"}']})
    out = catalog.filter_by_properties(gdf, {"sat:orbit_state": "descending"})
    assert len(out) == 0


def test_ac12_empty_or_none_properties_filter_is_a_no_op():
    gdf = gpd.GeoDataFrame({"properties": ['{"sat:orbit_state": "ascending"}']})
    assert len(catalog.filter_by_properties(gdf, None)) == 1
    assert len(catalog.filter_by_properties(gdf, {})) == 1


# --- AC13: properties_filter selection changes the digest; empty selection doesn't ---

def test_ac13_empty_properties_filter_does_not_perturb_the_s2_digest():
    """Reimplements the PRE-P2 `params_key` formula independently (not by calling the
    SUT with a different code path) and checks the SUT's default (`properties_filter`
    omitted) still produces exactly that digest -- catching a regression where an empty
    selection sneaks in an extra `|` or empty component."""
    import hashlib

    decl = _collections.get(config.SATELLITE_S2L2A)
    bands = ["B04", "B08"]
    raw = "|".join([
        ",".join(bands), config.MOSAIC_SCHEME, config.SATELLITE_S2L2A,
        declaration_module.digest(decl),
    ])
    expected = hashlib.sha1(raw.encode()).hexdigest()[:8]

    assert create_datacube.params_key(
        bands, config.MOSAIC_SCHEME, collection=config.SATELLITE_S2L2A, declaration=decl,
    ) == expected
    assert create_datacube.params_key(
        bands, config.MOSAIC_SCHEME, collection=config.SATELLITE_S2L2A, declaration=decl,
        properties_filter=None,
    ) == expected
    assert create_datacube.params_key(
        bands, config.MOSAIC_SCHEME, collection=config.SATELLITE_S2L2A, declaration=decl,
        properties_filter={},
    ) == expected


def test_ac13_differing_properties_filter_selection_changes_the_digest():
    decl = _collections.get("sentinel-1-rtc")
    bands = ["vv", "vh"]
    none_key = create_datacube.params_key(
        bands, config.MOSAIC_SCHEME, collection="sentinel-1-rtc", declaration=decl,
    )
    asc = create_datacube.params_key(
        bands, config.MOSAIC_SCHEME, collection="sentinel-1-rtc", declaration=decl,
        properties_filter={"sat:orbit_state": "ascending"},
    )
    desc = create_datacube.params_key(
        bands, config.MOSAIC_SCHEME, collection="sentinel-1-rtc", declaration=decl,
        properties_filter={"sat:orbit_state": "descending"},
    )
    assert len({none_key, asc, desc}) == 3


def test_ac13_properties_filter_canonicalization_is_order_independent():
    """Key order and per-key value-list order must not change the digest -- one
    selection is one stable string (D9.3)."""
    decl = _collections.get("sentinel-1-rtc")
    bands = ["vv"]
    a = create_datacube.params_key(
        bands, config.MOSAIC_SCHEME, collection="sentinel-1-rtc", declaration=decl,
        properties_filter={"sat:orbit_state": "ascending", "z": ["b", "a"]},
    )
    b = create_datacube.params_key(
        bands, config.MOSAIC_SCHEME, collection="sentinel-1-rtc", declaration=decl,
        properties_filter={"z": ["a", "b"], "sat:orbit_state": "ascending"},
    )
    assert a == b


# --- AC14: max_cloudcover against sentinel-1-rtc raises; no S1 key plumbing ----------

def test_ac14_max_cloudcover_against_s1_raises_naming_collection():
    errs = api._check_cloudcover_capability("sentinel-1-rtc", 50.0)
    assert errs and "sentinel-1-rtc" in errs[0]


def test_ac14_download_preflight_raises_for_s1_max_cloudcover():
    with pytest.raises(api.PreflightError, match="sentinel-1-rtc"):
        api.download(
            roi=None, startdate=datetime.datetime(2018, 4, 1),
            enddate=datetime.datetime(2018, 9, 30), bands=["vv"],
            dst_folderpath="/tmp/unused-ac14", source="mpc", collection="sentinel-1-rtc",
            max_tiles=1, max_cloudcover=50.0,
        )


def test_ac14_no_rise_eo_code_reads_pc_sdk_subscription_key():
    """D10 (retracted): sentinel-1-rtc needs no key, and P2 builds no preflight for
    one. `planetary_computer` (the third-party package) reads the env var itself; rise-eo
    is free to MENTION it in comments/docstrings (it does, documenting that the
    package reads it) but no rise-eo code may itself read `os.environ`/`os.getenv` for it
    -- that would be a second source of truth for the same secret (D10's own text)."""
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "rise_eo"
    read_patterns = (
        'os.environ["PC_SDK_SUBSCRIPTION_KEY"]', "os.environ['PC_SDK_SUBSCRIPTION_KEY']",
        'os.environ.get("PC_SDK_SUBSCRIPTION_KEY"', "os.environ.get('PC_SDK_SUBSCRIPTION_KEY'",
        'os.getenv("PC_SDK_SUBSCRIPTION_KEY"', "os.getenv('PC_SDK_SUBSCRIPTION_KEY'",
    )
    hits = [
        str(p) for p in root.rglob("*.py")
        if any(pat in p.read_text() for pat in read_patterns)
    ]
    assert hits == []


# --- review findings (Opus review of P2, 2026-09-12) ---------------------------------

class _FakeS1Item:
    """The minimum duck-typed STAC item `mpc._items_to_gdf` reads (id/datetime/
    geometry/properties/get_self_href) -- no `s2:processing_baseline`, like every real
    sentinel-1-rtc item."""

    id = "S1B_IW_GRDH_1SDV_20180601T165014_rtc"
    datetime = pd.Timestamp("2018-06-01", tz="UTC")
    properties = {"sat:orbit_state": "ascending", "sat:relative_orbit": 146}
    geometry = shapely.geometry.mapping(shapely.geometry.box(0, 0, 1, 1))

    def get_self_href(self):
        return "https://example/x"


def _write_float_tif(path, value, *, crs, transform):
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=1,
        dtype="float32", crs=crs, transform=transform, nodata=-32768,
    ) as dst:
        dst.write(np.full((1, 4, 4), value, dtype=np.float32))


def test_s1_source_nodata_is_translated_to_the_build_sentinel_not_medianed(tmp_path):
    """RTC writes its nodata as `-32768`, but the build's nodata is the CATALOG COLUMN's
    value (`declaration.nodata` is only the fallback for rows with no such column), and
    `mpc._items_to_gdf` writes `config.NODATA` (0) for every collection -- so a real S1
    cube's sentinel is 0, not the declared -32768.

    That is safe, and this pins WHY: `raster.images.crop_tif` hands `rasterio.mask` the
    build's nodata as the fill, so pixels at the SOURCE nodata are masked on read and
    come back as the build sentinel -- and `median_mosaic(mask_value=...)` then excludes
    them. Two acquisitions in one window, one all-nodata and one valid: the median must
    be the valid value, never a blend with -32768 (~ -16384).

    (`rise_eo.config.NODATA = 0` is a workspace-wide convention, `CLAUDE.md`; D17's declared
    -32768 therefore documents the SOURCE's value, and does not reach the cube.)"""
    decl = _collections.get("sentinel-1-rtc")
    ingested_nodata = int(
        mpc._items_to_gdf([_FakeS1Item()], collection="sentinel-1-rtc",
                          declaration=decl)["nodata"].iloc[0]
    )
    assert ingested_nodata == config.NODATA  # the build sentinel, not decl.nodata

    crs = "EPSG:32633"
    transform = from_origin(500000, 5000000, 10, 10)
    tile_box = shapely.geometry.box(500000, 4999960, 500040, 5000000)

    rows = []
    for n, (ts, value) in enumerate([
        (pd.Timestamp("2018-06-02", tz="UTC"), -32768.0),
        (pd.Timestamp("2018-06-04", tz="UTC"), 0.2),
    ]):
        p = tmp_path / f"vv_{n}.tif"
        _write_float_tif(p, value, crs=crs, transform=transform)
        rows.append({
            "id": f"s1_{n}", "filepath": str(p), "band": "vv", "timestamp": ts,
            "geometry": tile_box, "area_contribution": 100.0,
            "offset": 0, "nodata": ingested_nodata,
            "properties": json.dumps({"sat:orbit_state": "ascending"}),
        })
    catalog_subset = gpd.GeoDataFrame(rows, crs=crs)
    shape_gdf = gpd.GeoDataFrame({"geometry": [tile_box]}, crs=crs)

    out = tmp_path / "cube"
    builder.build_datacube(
        catalog_subset=catalog_subset, shape_gdf=shape_gdf,
        startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 6, 21),
        bands=["vv"], mosaic_days=20, declaration=decl,
        export_folderpath=str(out), if_missing_files=None,
    )
    dc = fs.load_npy(str(out / "datacube.npy"))
    assert dc.shape == (1, 4, 4, 1)
    assert dc[0, 0, 0, 0] == pytest.approx(0.2)


def test_properties_filter_matches_a_non_string_property_value():
    """D9 names `sat:relative_orbit` as the other thing `properties_filter` serves
    ("with no extra code"). It is an INTEGER in STAC. Passing the int raised
    `TypeError: 'int' object is not iterable` (`list(146)`); passing the string "146"
    matched zero rows silently -- the exact "user concludes their ROI has no coverage"
    failure D9.1 exists to prevent. Both spellings must select the row."""
    gdf = gpd.GeoDataFrame({"properties": [
        json.dumps({"sat:orbit_state": "ascending", "sat:relative_orbit": 146}),
        json.dumps({"sat:orbit_state": "ascending", "sat:relative_orbit": 44}),
    ]})
    assert len(catalog.filter_by_properties(gdf, {"sat:relative_orbit": 146})) == 1
    assert len(catalog.filter_by_properties(gdf, {"sat:relative_orbit": "146"})) == 1
    assert len(catalog.filter_by_properties(gdf, {"sat:relative_orbit": [146, 44]})) == 2


def test_properties_filter_digest_accepts_a_non_string_value():
    """The same scalar bug on the identity side: `_canonicalize_properties_filter` did
    `list(want)` for anything non-`str`, so an int selection raised before it could
    reach a digest. An int and its string spelling must canonicalize alike, or one
    selection would be two cube paths."""
    key_int = create_datacube._canonicalize_properties_filter({"sat:relative_orbit": 146})
    key_str = create_datacube._canonicalize_properties_filter({"sat:relative_orbit": "146"})
    assert key_int == key_str != ""


def test_build_datacube_applies_properties_filter_itself(tmp_path):
    """D9: "The filter is applied both there [build_datacube] and upstream where the
    catalog is queried ... applying it twice is idempotent." P2 implemented enforcement
    only, so a caller handing `build_datacube` an unfiltered mixed-orbit subset -- the
    direct-builder path, which is what run-book 58-p2's QGIS step does -- could be told
    it was wrong but had no way to narrow it. Filtering here makes the raise
    recoverable in place, and stays idempotent for rows `setup` already filtered."""
    crs = "EPSG:32633"
    transform = from_origin(500000, 5000000, 10, 10)
    tile_box = shapely.geometry.box(500000, 4999960, 500040, 5000000)
    p = tmp_path / "vv.tif"
    _write_float_tif(p, 0.3, crs=crs, transform=transform)

    rows = [
        {"id": f"s1_{n}", "filepath": str(p), "band": "vv",
         "timestamp": pd.Timestamp("2018-06-02", tz="UTC"), "geometry": tile_box,
         "area_contribution": 100.0, "offset": 0, "nodata": -32768,
         "properties": json.dumps({"sat:orbit_state": orbit})}
        for n, orbit in enumerate(["ascending", "descending"])
    ]
    catalog_subset = gpd.GeoDataFrame(rows, crs=crs)
    shape_gdf = gpd.GeoDataFrame({"geometry": [tile_box]}, crs=crs)
    decl = _collections.get("sentinel-1-rtc")
    kwargs = dict(
        catalog_subset=catalog_subset, shape_gdf=shape_gdf,
        startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 6, 21),
        bands=["vv"], mosaic_days=20, declaration=decl, if_missing_files=None,
    )
    with pytest.raises(ValueError, match="sat:orbit_state"):
        builder.build_datacube(export_folderpath=str(tmp_path / "a"), **kwargs)

    builder.build_datacube(
        export_folderpath=str(tmp_path / "b"),
        properties_filter={"sat:orbit_state": "ascending"}, **kwargs,
    )
    dc = fs.load_npy(str(tmp_path / "b" / "datacube.npy"))
    assert dc[0, 0, 0, 0] == pytest.approx(0.3)


# --- download-time properties_filter (spec 58 D9; the P2 scope cut, reversed after the
# --- first real Window A run: 104 RTC granules for ONE ~6 km cell, whole scenes each) ---

def _fake_s1_item(item_id, orbit, rel_orbit=146):
    class _Asset:
        def __init__(self, href):
            self.href = href

    class _Item:
        id = item_id
        datetime = pd.Timestamp("2018-06-01", tz="UTC")
        properties = {"sat:orbit_state": orbit, "sat:relative_orbit": rel_orbit}
        geometry = shapely.geometry.mapping(shapely.geometry.box(0, 0, 1, 1))
        assets = {"vv": _Asset(f"https://example/{item_id}_vv.tif"),
                  "vh": _Asset(f"https://example/{item_id}_vh.tif")}

        def get_self_href(self):
            return f"https://example/{item_id}"

    return _Item()


def _download_with(monkeypatch, tmp_path, items, **kwargs):
    """Drive `mpc.download` with discovery stubbed out and transfers replaced by a
    recorder -- so the cap/filter logic is exercised with no network and no bytes."""
    moved = []
    monkeypatch.setattr(mpc, "_search_items_unsigned", lambda *a, **k: items)
    monkeypatch.setattr(mpc, "_import_pc_sign", lambda: (lambda u: u))
    monkeypatch.setattr(
        mpc, "_transfer_and_stamp_one",
        lambda src, dst, **kw: (moved.append(src), (True, "ok"))[1],
    )
    roi = tmp_path / "roi.geojson"
    gpd.GeoDataFrame({"geometry": [shapely.geometry.box(0, 0, 1, 1)]},
                     crs="EPSG:4326").to_file(roi, driver="GeoJSON")
    cat = catalog.TileCatalog(str(tmp_path / "catalog.parquet"))
    mpc.download(
        roi=str(roi), startdate=datetime.datetime(2018, 4, 1),
        enddate=datetime.datetime(2018, 9, 30), bands=["vv", "vh"],
        root_folderpath=str(tmp_path / "imagery"), catalog=cat,
        collection="sentinel-1-rtc", **kwargs,
    )
    return moved


def test_download_properties_filter_drops_the_unused_orbit_before_transfer(monkeypatch, tmp_path):
    """The concrete need that justified adding this (Window A, 2026-09-12): a build can use
    only ONE orbit state (D9 enforcement), and a transfer is a whole-asset byte copy, so
    fetching the other orbit is pure waste."""
    items = [_fake_s1_item("asc1", "ascending"), _fake_s1_item("asc2", "ascending"),
             _fake_s1_item("desc1", "descending")]
    moved = _download_with(monkeypatch, tmp_path, items, max_tiles=10,
                           properties_filter={"sat:orbit_state": "descending"})
    assert all("desc1" in src for src in moved), moved
    assert len(moved) == 2  # vv + vh of the one descending scene


def test_download_properties_filter_applies_before_the_max_tiles_cap(monkeypatch, tmp_path):
    """`max_tiles` guards BYTES ABOUT TO MOVE, so it must count post-filter tiles. Three
    discovered, cap of 2: unfiltered it raises; narrowed to one orbit it proceeds."""
    items = [_fake_s1_item("asc1", "ascending"), _fake_s1_item("asc2", "ascending"),
             _fake_s1_item("desc1", "descending")]
    with pytest.raises(ValueError, match="exceed max_tiles"):
        _download_with(monkeypatch, tmp_path, items, max_tiles=2)

    moved = _download_with(monkeypatch, tmp_path, items, max_tiles=2,
                           properties_filter={"sat:orbit_state": "descending"})
    assert len(moved) == 2


def test_download_max_tiles_message_says_the_filter_already_narrowed(monkeypatch, tmp_path):
    """Still over the cap after filtering -- the message must not send the user back to
    add a filter they already have."""
    items = [_fake_s1_item(f"asc{i}", "ascending") for i in range(4)]
    with pytest.raises(ValueError, match="already narrowed from 4"):
        _download_with(monkeypatch, tmp_path, items, max_tiles=2,
                       properties_filter={"sat:orbit_state": "ascending"})


def test_download_properties_filter_unknown_key_raises_naming_carried_keys(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="sat:orbit_state"):
        _download_with(monkeypatch, tmp_path, [_fake_s1_item("asc1", "ascending")],
                       max_tiles=10, properties_filter={"no:such:key": "x"})


def test_api_download_rejects_properties_filter_for_cdse():
    """Silently ignoring it would only show up on the disk bill."""
    with pytest.raises(api.PreflightError, match="not implemented for source='cdse'"):
        api.download(
            roi=None, startdate=datetime.datetime(2018, 4, 1),
            enddate=datetime.datetime(2018, 9, 30), bands=["B04"],
            dst_folderpath="/tmp/unused-cdse-pf", source="cdse",
            collection=config.SATELLITE_S2L2A, max_tiles=1,
            properties_filter={"sat:orbit_state": "ascending"},
        )


# --- the AML download path must carry collection + properties_filter (first real S1 AML
# --- run, 2026-09-28: `run_aml_download` had neither, so MPC discovery fell back to its
# --- sentinel-2-l2a default and raised "band 'vv' is not available on item S2B_MSIL2A_...") --

_S1_FILTER = {"sat:orbit_state": "descending"}


def test_discover_shard_rows_honours_collection_and_properties_filter(monkeypatch, tmp_path):
    """Driver-side discovery for the AML fan-out must search the requested collection and
    drop the unused orbit BEFORE rows are sharded -- the same narrowing `mpc.download` does."""
    items = [_fake_s1_item("asc1", "ascending"), _fake_s1_item("desc1", "descending")]
    seen = {}

    def _search(*a, collection, **k):
        seen["collection"] = collection
        return items

    monkeypatch.setattr(mpc, "_search_items_unsigned", _search)
    roi = tmp_path / "roi.geojson"
    gpd.GeoDataFrame({"geometry": [shapely.geometry.box(0, 0, 1, 1)]},
                     crs="EPSG:4326").to_file(roi, driver="GeoJSON")

    rows = mpc.discover_shard_rows(
        str(roi), datetime.datetime(2018, 6, 1), datetime.datetime(2018, 7, 1),
        ["vv", "vh"], str(tmp_path / "imagery"),
        collection="sentinel-1-rtc", properties_filter=_S1_FILTER,
    )

    assert seen["collection"] == "sentinel-1-rtc"
    assert {r["tile_id"] for r in rows} == {"desc1"}
    assert sorted(r["band"] for r in rows) == ["vh", "vv"]
    assert {r["collection"] for r in rows} == {"sentinel-1-rtc"}


def test_run_aml_download_mpc_forwards_collection_and_properties_filter(monkeypatch):
    from rise_eo.workflows import runners
    seen = {}

    def _discover(*a, **kw):
        seen.update(kw)
        return []   # nothing to download -> returns before any job is built

    monkeypatch.setattr(runners._mpc, "discover_shard_rows", _discover)
    monkeypatch.setattr(runners, "_import_aml_command", lambda: None)
    monkeypatch.setattr(runners, "_import_command_job_limits", lambda: None)

    runners.run_aml_download(
        "memory://roi.geojson", "2018-06-01", "2018-07-01", ["vv", "vh"],
        "memory://s1dl/data", "memory://s1dl/data/catalog.parquet",
        source="mpc", cluster="c", environment="e:1", root="memory://s1dl/root",
        identity_client_id="id", max_tiles=10, ml_client=object(),
        collection="sentinel-1-rtc", properties_filter=_S1_FILTER,
    )

    assert seen["collection"] == "sentinel-1-rtc"
    assert seen["properties_filter"] == _S1_FILTER


def test_api_download_aml_forwards_collection_and_properties_filter(monkeypatch, tmp_path):
    from rise_eo.workflows import runners
    seen = {}
    monkeypatch.setattr(runners, "run_aml_download", lambda **kw: seen.update(kw))

    api.download(
        "memory://roi.geojson", datetime.datetime(2018, 6, 1), datetime.datetime(2018, 7, 1),
        ["vv", "vh"], str(tmp_path / "imagery"), source="mpc",
        collection="sentinel-1-rtc", properties_filter=_S1_FILTER, max_tiles=10,
        runner="aml", runner_kwargs={"cluster": "c", "environment": "e:1",
                                     "root": "memory://r", "identity_client_id": "id"},
    )

    assert seen["collection"] == "sentinel-1-rtc"
    assert seen["properties_filter"] == _S1_FILTER


def test_create_training_data_forwards_properties_filter_to_its_download(monkeypatch, tmp_path):
    """Without it the download pulls BOTH orbits -- whole ~3.7 GB scenes -- of which the
    build can use only one."""
    seen = {}

    class _Stop(Exception):
        pass

    def _fake_download(**kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(api, "_download_verb", _fake_download)
    polys = gpd.GeoDataFrame(
        {"fid": [1, 2], "crop": ["a", "b"],
         "geometry": [shapely.geometry.box(0, 0, 0.01, 0.01),
                      shapely.geometry.box(0.02, 0, 0.03, 0.01)]},
        crs="EPSG:4326",
    )
    with pytest.raises(_Stop):
        api.create_training_data(
            label_polygons=polys, catalog_filepath=str(tmp_path / "data" / "sentinel-1-rtc" / "catalog.parquet"),
            startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 1),
            mosaic_days=10, bands=["vv", "vh"], id_col="fid", label_col="crop",
            export_folderpath=str(tmp_path / "export"),
            collection="sentinel-1-rtc", properties_filter=_S1_FILTER,
            source="mpc", download=True, max_tiles=10,
        )

    assert seen["collection"] == "sentinel-1-rtc"
    assert seen["properties_filter"] == _S1_FILTER


def test_run_aml_download_mpc_forwards_max_cloudcover_to_discovery(monkeypatch):
    from rise_eo.workflows import runners
    seen = {}

    def _discover(*a, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(runners._mpc, "discover_shard_rows", _discover)
    monkeypatch.setattr(runners, "_import_aml_command", lambda: None)
    monkeypatch.setattr(runners, "_import_command_job_limits", lambda: None)

    runners.run_aml_download(
        "memory://roi.geojson", "2018-06-01", "2018-07-01", ["vv", "vh"],
        "memory://s1dl/data", "memory://s1dl/data/catalog.parquet",
        source="mpc", cluster="c", environment="e:1", root="memory://s1dl/root",
        identity_client_id="id", max_tiles=10, ml_client=object(), max_cloudcover=12.5,
    )

    assert seen["max_cloudcover"] == 12.5
