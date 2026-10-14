"""Tests for fsd.datacube.builder (spec 03). Synthetic GeoTIFF tiles on disk."""

import dataclasses
import datetime
import inspect
import json
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.crs import CRS as RioCRS
from rasterio.transform import from_origin
from shapely.geometry import box

from fsd.catalog.declaration import CollectionDeclaration, MaskSpec
from fsd.datacube import builder
from fsd.storage import fs

CRS = "EPSG:32633"
TRANSFORM = from_origin(500000, 5000000, 10, 10)  # 10 m pixels
TILE_BOX = box(500000, 4999960, 500040, 5000000)  # 4x4 @ 10 m


def _write_tile(path, arr):
    arr = arr[np.newaxis].astype(np.uint16)  # (1, H, W)
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1,
                       dtype="uint16", crs=CRS, transform=TRANSFORM, nodata=0) as dst:
        dst.write(arr)


def _band_row(id_, path, band, ts):
    return {"id": id_, "filepath": str(path), "band": band, "timestamp": ts,
            "geometry": TILE_BOX, "area_contribution": 100.0}


def _make_catalog(tmp_path):
    """One tile, two timestamps, bands B04/B08/SCL. ts1 has a cloud (SCL=8) at (0,0)."""
    ts1 = pd.Timestamp("2018-06-01", tz="UTC")
    ts2 = pd.Timestamp("2018-07-01", tz="UTC")
    rows = []
    for ts, tag, b04v, b08v in [(ts1, "t1", 100, 200), (ts2, "t2", 110, 210)]:
        scl = np.full((4, 4), 4, dtype=np.uint16)  # 4 = vegetation (not masked)
        if tag == "t1":
            scl[0, 0] = 8  # cloud medium prob -> masked
        for band, val in [("B04", b04v), ("B08", b08v)]:
            p = tmp_path / f"{tag}_{band}.tif"
            _write_tile(p, np.full((4, 4), val, dtype=np.uint16))
            rows.append(_band_row(f"tile_{tag}", p, band, ts))
        p = tmp_path / f"{tag}_SCL.tif"
        _write_tile(p, scl)
        rows.append(_band_row(f"tile_{tag}", p, "SCL", ts))
    gdf = gpd.GeoDataFrame(rows, crs=CRS)
    shape = gpd.GeoDataFrame({"geometry": [TILE_BOX]}, crs=CRS)
    return gdf, shape



def test_build_datacube_records_provenance_in_the_cube_metadata(tmp_path):
    """Spec 59 AC14 through the real builder: the rows a cube used -> D9's provenance."""
    catalog, shape = _make_catalog(tmp_path)
    catalog["collection"] = "sentinel-2-l2a"
    catalog["processing_version"] = catalog["id"].map({"tile_t1": "02.12", "tile_t2": "05.00"})
    catalog["source"] = catalog["id"].map({"tile_t1": "mpc", "tile_t2": "cdse,mpc"})
    out = tmp_path / "cube"
    builder.build_datacube(
        catalog_subset=catalog, shape_gdf=shape,
        startdate=datetime.datetime(2018, 5, 31), enddate=datetime.datetime(2018, 7, 2),
        bands=["B04", "B08", "SCL"], mosaic_days=20,
        export_folderpath=str(out), if_missing_files=None,
    )
    md = fs.load_npy(str(out / "metadata.pickle.npy"), allow_pickle=True)[()]
    assert md["provenance"] == {"sentinel-2-l2a": {
        "ids": ["tile_t1", "tile_t2"], "processing_version": ["02.12", "05.00"],
        "source": ["cdse", "mpc"]}}

def test_build_datacube_end_to_end(tmp_path):
    catalog, shape = _make_catalog(tmp_path)
    out = tmp_path / "cube"
    builder.build_datacube(
        catalog_subset=catalog, shape_gdf=shape,
        startdate=datetime.datetime(2018, 5, 31), enddate=datetime.datetime(2018, 7, 2),
        bands=["B04", "B08", "SCL"], mosaic_days=20,
        export_folderpath=str(out), if_missing_files=None,
    )

    dc = fs.load_npy(str(out / "datacube.npy"))
    md = fs.load_npy(str(out / "metadata.pickle.npy"), allow_pickle=True)[()]

    assert dc.shape == (2, 4, 4, 2)          # 2 mosaic buckets, SCL dropped
    assert md["bands"] == ["B04", "B08"]
    assert len(md["timestamps"]) == 2
    # ts1: cloud pixel (0,0) masked to 0; a clear pixel keeps its value
    assert dc[0, 0, 0, 0] == 0 and dc[0, 0, 0, 1] == 0
    assert dc[0, 1, 1, 0] == 100 and dc[0, 1, 1, 1] == 200
    # ts2: clear, values preserved
    assert dc[1, 0, 0, 0] == 110 and dc[1, 2, 2, 1] == 210
    assert md["geotiff_metadata"]["height"] == 4
    # spec 46 Q3 (#68): the shape's OWN actual acquisition range moved into the cube's
    # metadata once the run FOLDER stopped naming it (spec 46 D1) -- this fixture's
    # tiles are 2018-06-01 and 2018-07-01, distinct from the requested
    # startdate/enddate window (2018-05-31 / 2018-07-02) passed above.
    assert md["actual_start"] == pd.Timestamp("2018-06-01", tz="UTC")
    assert md["actual_end"] == pd.Timestamp("2018-07-01", tz="UTC")


def test_build_datacube_no_mask_band_requested_closes_35(tmp_path):
    """spec 34 §2b/§35: bands=['B04'] (no SCL requested) builds without raising —
    the mask/drop ops are skipped entirely, not just tolerated with a no-op."""
    ts = pd.Timestamp("2018-06-01", tz="UTC")
    p = tmp_path / "t1_B04.tif"
    _write_tile(p, np.full((4, 4), 111, dtype=np.uint16))
    catalog = gpd.GeoDataFrame([_band_row("tile_t1", p, "B04", ts)], crs=CRS)
    shape = gpd.GeoDataFrame({"geometry": [TILE_BOX]}, crs=CRS)
    out = tmp_path / "cube"

    builder.build_datacube(
        catalog_subset=catalog, shape_gdf=shape,
        startdate=datetime.datetime(2018, 5, 31), enddate=datetime.datetime(2018, 6, 2),
        bands=["B04"], mosaic_days=20, export_folderpath=str(out),
        if_missing_files=None, reference_band="B04",
    )
    dc = fs.load_npy(str(out / "datacube.npy"))
    md = fs.load_npy(str(out / "metadata.pickle.npy"), allow_pickle=True)[()]
    assert md["bands"] == ["B04"]      # no SCL to drop -- never requested
    assert (dc[0, :, :, 0] == 111).all()


def test_build_datacube_native_grid_declaration_raises_not_implemented(tmp_path):
    """spec 34 [G2]: a native-single-grid declaration is a loud, documented gap."""
    ts = pd.Timestamp("2018-06-01", tz="UTC")
    p = tmp_path / "t1_B04.tif"
    _write_tile(p, np.full((4, 4), 1, dtype=np.uint16))
    catalog = gpd.GeoDataFrame([_band_row("tile_t1", p, "B04", ts)], crs=CRS)
    shape = gpd.GeoDataFrame({"geometry": [TILE_BOX]}, crs=CRS)
    native_grid_declaration = CollectionDeclaration(reference_band=None, native_grid=True)

    with pytest.raises(NotImplementedError, match="native single-grid"):
        builder.build_datacube(
            catalog_subset=catalog, shape_gdf=shape,
            startdate=datetime.datetime(2018, 5, 31), enddate=datetime.datetime(2018, 6, 2),
            bands=["B04"], export_folderpath=str(tmp_path / "cube"),
            if_missing_files=None, declaration=native_grid_declaration,
        )


def test_build_datacube_unimplemented_mask_type_raises_not_implemented(tmp_path):
    """spec 34 [G3]: a mask_type other than categorical_classes is a loud gap, not a
    silently wrong mask."""
    ts = pd.Timestamp("2018-06-01", tz="UTC")
    rows = []
    for band, val in [("B04", 5), ("QA", 0)]:
        p = tmp_path / f"t1_{band}.tif"
        _write_tile(p, np.full((4, 4), val, dtype=np.uint16))
        rows.append(_band_row("tile_t1", p, band, ts))
    catalog = gpd.GeoDataFrame(rows, crs=CRS)
    shape = gpd.GeoDataFrame({"geometry": [TILE_BOX]}, crs=CRS)
    bitmask_declaration = CollectionDeclaration(
        reference_band="B04", mask_spec=MaskSpec(band="QA", mask_type="bitmask"),
    )

    with pytest.raises(NotImplementedError, match="bitmask"):
        builder.build_datacube(
            catalog_subset=catalog, shape_gdf=shape,
            startdate=datetime.datetime(2018, 5, 31), enddate=datetime.datetime(2018, 6, 2),
            bands=["B04", "QA"], export_folderpath=str(tmp_path / "cube"),
            if_missing_files=None, declaration=bitmask_declaration,
        )


def test_build_datacube_mask_keep_retains_mask_band(tmp_path):
    """spec 34 §2c: mask_keep=True masks but does not drop the mask band."""
    ts = pd.Timestamp("2018-06-01", tz="UTC")
    rows = []
    scl = np.full((4, 4), 4, dtype=np.uint16)
    scl[0, 0] = 8  # masked
    for band, arr in [("B04", np.full((4, 4), 50, dtype=np.uint16)), ("SCL", scl)]:
        p = tmp_path / f"t1_{band}.tif"
        _write_tile(p, arr)
        rows.append(_band_row("tile_t1", p, band, ts))
    catalog = gpd.GeoDataFrame(rows, crs=CRS)
    shape = gpd.GeoDataFrame({"geometry": [TILE_BOX]}, crs=CRS)
    keep_declaration = CollectionDeclaration(
        reference_band="B04",
        mask_spec=MaskSpec(band="SCL", mask_type="categorical_classes", classes=(8,)),
        mask_keep=True,
    )
    out = tmp_path / "cube"

    builder.build_datacube(
        catalog_subset=catalog, shape_gdf=shape,
        startdate=datetime.datetime(2018, 5, 31), enddate=datetime.datetime(2018, 6, 2),
        bands=["B04", "SCL"], export_folderpath=str(out),
        if_missing_files=None, declaration=keep_declaration,
    )
    md = fs.load_npy(str(out / "metadata.pickle.npy"), allow_pickle=True)[()]
    assert "SCL" in md["bands"]     # kept, not dropped
    dc = fs.load_npy(str(out / "datacube.npy"))
    b04 = dc[0, :, :, md["bands"].index("B04")]
    assert b04[0, 0] == 0 and b04[1, 1] == 50   # masked pixel zeroed, others intact


def test_build_datacube_reads_declared_nodata_not_config_default(tmp_path):
    """spec 34: the catalog's declared `nodata` (not `config.NODATA`) decides which
    pixel value the mosaic treats as missing. Two acquisitions in one mosaic window:
    one real (42), one all-declared-nodata (77) — if the declared value (77) is
    correctly used as the mask value, the 77s are excluded and the window median is
    42; if the code fell back to `config.NODATA` (0), 77 would NOT be excluded and
    the median would be (42+77)/2=59 instead."""
    ts1 = pd.Timestamp("2018-06-01", tz="UTC")
    ts2 = pd.Timestamp("2018-06-05", tz="UTC")
    rows = []
    for ts, tag, val in [(ts1, "a", 42), (ts2, "b", 77)]:
        p = tmp_path / f"{tag}_B04.tif"
        _write_tile(p, np.full((4, 4), val, dtype=np.uint16))
        rows.append({**_band_row(f"tile_{tag}", p, "B04", ts), "nodata": 77})
    catalog = gpd.GeoDataFrame(rows, crs=CRS)
    shape = gpd.GeoDataFrame({"geometry": [TILE_BOX]}, crs=CRS)
    out = tmp_path / "cube"

    builder.build_datacube(
        catalog_subset=catalog, shape_gdf=shape,
        startdate=datetime.datetime(2018, 5, 31), enddate=datetime.datetime(2018, 6, 10),
        bands=["B04"], mosaic_days=20, export_folderpath=str(out),
        if_missing_files=None, reference_band="B04",
    )
    dc = fs.load_npy(str(out / "datacube.npy"))
    assert dc.shape[0] == 1               # both dates land in the same window
    assert (dc[0, :, :, 0] == 42).all()   # 77 excluded as the declared nodata


def test_build_datacube_harmonizes_offset_before_median_mosaic(tmp_path):
    """spec 34 §1f / spec 32 #10 guard: a calendar window straddling the baseline-
    04.00 cutover must align every image to the pre-baseline scale BEFORE the
    median, not after — else the mosaic silently mixes incomparable reflectances."""
    ts_old = pd.Timestamp("2021-06-01", tz="UTC")
    ts_new = pd.Timestamp("2022-06-01", tz="UTC")
    rows = []
    for ts, tag, b04v, offset in [(ts_old, "old", 200, 0), (ts_new, "new", 1200, -1000)]:
        p = tmp_path / f"{tag}_B04.tif"
        _write_tile(p, np.full((4, 4), b04v, dtype=np.uint16))
        rows.append({**_band_row(f"tile_{tag}", p, "B04", ts), "offset": offset})
        scl_p = tmp_path / f"{tag}_SCL.tif"
        _write_tile(scl_p, np.full((4, 4), 4, dtype=np.uint16))  # 4 = vegetation, unmasked
        rows.append({**_band_row(f"tile_{tag}", scl_p, "SCL", ts), "offset": 0})
    catalog = gpd.GeoDataFrame(rows, crs=CRS)
    shape = gpd.GeoDataFrame({"geometry": [TILE_BOX]}, crs=CRS)
    out = tmp_path / "cube"

    builder.build_datacube(
        catalog_subset=catalog, shape_gdf=shape,
        startdate=datetime.datetime(2021, 1, 1), enddate=datetime.datetime(2023, 1, 1),
        bands=["B04", "SCL"], mosaic_days=1000,
        export_folderpath=str(out), if_missing_files=None, reference_band="B04",
    )
    dc = fs.load_npy(str(out / "datacube.npy"))
    assert dc.shape[0] == 1  # both dates land in the same (single) calendar window
    # harmonized: 200 and 1200-1000=200 -> median 200; unharmonized would median to 700.
    assert (dc[0, :, :, 0] == 200).all()


def test_build_datacube_writes_timings_only_when_flagged(tmp_path):
    catalog, shape = _make_catalog(tmp_path)
    kw = dict(
        catalog_subset=catalog, shape_gdf=shape,
        startdate=datetime.datetime(2018, 5, 31), enddate=datetime.datetime(2018, 7, 2),
        bands=["B04", "B08", "SCL"], mosaic_days=20,
        if_missing_files=None,
    )
    off = tmp_path / "off"
    builder.build_datacube(export_folderpath=str(off), **kw)
    assert not (off / builder.TIMINGS_FILENAME).exists()   # off by default

    on = tmp_path / "on"
    builder.build_datacube(export_folderpath=str(on), write_timings=True, **kw)
    payload = json.loads((on / builder.TIMINGS_FILENAME).read_text())
    assert set(payload["phase_seconds"]) == {
        "missing_check", "load_images", "dst_crs", "reference_profile",
        "resample", "stack", "ops", "save"}
    assert payload["total_seconds"] >= 0
    assert payload["n_band_rows"] == 6 and payload["datacube_shape"] == [2, 4, 4, 2]


def test_build_datacube_writes_read_log_only_when_flagged(tmp_path):
    catalog, shape = _make_catalog(tmp_path)
    shape = shape.copy()
    shape["id"] = "grid_007"
    kw = dict(
        catalog_subset=catalog, shape_gdf=shape,
        startdate=datetime.datetime(2018, 5, 31), enddate=datetime.datetime(2018, 7, 2),
        bands=["B04", "B08", "SCL"], mosaic_days=20,
        if_missing_files=None,
    )
    off = tmp_path / "off"
    builder.build_datacube(export_folderpath=str(off), **kw)
    assert not (off / builder.READ_LOG_FILENAME).exists()   # off by default

    on = tmp_path / "on"
    builder.build_datacube(export_folderpath=str(on), write_read_log=True, **kw)
    lines = (on / builder.READ_LOG_FILENAME).read_text().strip().splitlines()
    rows = [json.loads(ln) for ln in lines]
    assert len(rows) == len(catalog)          # one row per band file read (6)
    r = rows[0]
    assert set(r) == {"id", "mgrs_tile", "product_id", "band", "filepath",
                      "start", "end", "duration"}
    assert r["id"] == "grid_007"              # grid id from shape_gdf
    assert r["product_id"].startswith("tile_")
    assert r["mgrs_tile"] is None             # synthetic ids carry no _T marker
    assert all(row["duration"] >= 0 for row in rows)
    assert all(row["end"] >= row["start"] for row in rows)


def test_load_images_read_log_noop_when_parallel(tmp_path):
    catalog, shape = _make_catalog(tmp_path)
    shape = shape.copy()
    shape["id"] = "g"
    out = tmp_path / "par"
    with pytest.warns(RuntimeWarning, match="njobs_load_images == 1"):
        builder.build_datacube(
            catalog_subset=catalog, shape_gdf=shape,
            startdate=datetime.datetime(2018, 5, 31),
            enddate=datetime.datetime(2018, 7, 2),
            bands=["B04", "B08", "SCL"], mosaic_days=20,
            export_folderpath=str(out), if_missing_files=None,
            njobs_load_images=2, write_read_log=True,
        )
    assert not (out / builder.READ_LOG_FILENAME).exists()   # skipped, not written


def test_flatten_catalog_skips_non_raster(tmp_path):
    gdf = gpd.GeoDataFrame(
        [{"id": "x", "local_folderpath": str(tmp_path), "timestamp": 0,
          "files": "B04.jp2,B08.jp2,MTD_TL.xml", "area_contribution": 50.0,
          "offset": 0, "nodata": 0, "geometry": TILE_BOX}], crs=CRS,
    )
    flat = builder.flatten_catalog(gdf)
    assert sorted(flat["band"]) == ["B04", "B08"]      # xml skipped
    assert flat["filepath"].iloc[0].endswith("B04.jp2")


def test_flatten_catalog_offset_exempts_non_reflectance_bands(tmp_path):
    gdf = gpd.GeoDataFrame(
        [{"id": "x", "local_folderpath": str(tmp_path), "timestamp": 0,
          "files": "B04.tif,SCL.tif", "area_contribution": 50.0,
          "offset": -1000, "nodata": 0, "geometry": TILE_BOX}], crs=CRS,
    )
    flat = builder.flatten_catalog(gdf)
    offsets = dict(zip(flat["band"], flat["offset"]))
    assert offsets == {"B04": -1000, "SCL": 0}


def test_flatten_catalog_attaches_declaration():
    """spec 35 §2a: the resolved declaration is attached as the JSON-able
    `attrs["fsd:declaration"]`, never the dataclass under a bare `"declaration"`
    key -- `declaration.from_attrs` is how a caller reads it back typed."""
    gdf = gpd.GeoDataFrame(
        [{"id": "x", "local_folderpath": "/tmp", "timestamp": 0,
          "files": "B04.jp2", "area_contribution": 50.0, "offset": 0, "nodata": 0,
          "geometry": TILE_BOX}], crs=CRS,
    )
    from fsd.catalog import declaration as declaration_module
    from fsd.catalog.declaration import S2_L2A_DECLARATION

    flat = builder.flatten_catalog(gdf)
    assert "declaration" not in flat.attrs
    assert flat.attrs[declaration_module.ATTRS_KEY] == declaration_module.to_json(S2_L2A_DECLARATION)
    assert declaration_module.from_attrs(flat) == S2_L2A_DECLARATION


# --- declaration persistence, the three hops end to end (spec 35 §8.2) -------


def test_declaration_survives_ingest_filter_slice_reread_flatten():
    """The regression test that actually matters (spec 35 §8.2): a non-S2
    declaration stamped at ingest -> `TileCatalog.filter` -> `fs.write_parquet`
    slice -> a *fresh* `fs.read_parquet` (simulating the per-cell task's separate
    process) -> `flatten_catalog` -> the declaration reaching `build_datacube` is
    the stamped one, NOT `S2_L2A_DECLARATION`.

    Deliberately non-S2 (`reference_band="B04"`, no B08 anywhere) so a silent
    fallback to the S2 default cannot pass by coincidence -- an agreement test
    can't catch a shared error (the black-tile-bug lesson): if `build_datacube`
    ignored the resolved declaration and used the S2 default's `reference_band=
    "B08"` instead, `catalog_gdf.loc[catalog_gdf["band"] == "B08"]` would be
    empty and the merge below would raise, so this test fails loudly rather than
    passing vacuously.
    """
    import tempfile

    from fsd.catalog import declaration as declaration_module
    from fsd.catalog.catalog import TileCatalog
    from fsd.storage import fs as fs_module

    ts = pd.Timestamp("2018-06-01", tz="UTC")
    custom = CollectionDeclaration(reference_band="B04", mask_spec=None, mosaic_method="median")

    with tempfile.TemporaryDirectory() as tmp:
        import os

        tile_path = os.path.join(tmp, "B04.tif")
        _write_tile(tile_path, np.full((4, 4), 42, dtype=np.uint16))

        geom_4326 = gpd.GeoSeries([TILE_BOX], crs=CRS).to_crs("EPSG:4326").iloc[0]
        cat_path = os.path.join(tmp, "catalog.parquet")
        cat = TileCatalog(cat_path, declaration=custom)
        cat.append([{
            "id": "tile_t1", "collection": "s2-like-but-not", "timestamp": ts,
            "s3url": "s3://x", "local_folderpath": tmp, "files": "B04.tif",
            "cloud_cover": 0.0, "offset": 0, "nodata": 0, "geometry": geom_4326,
        }])

        shape_gdf = gpd.GeoDataFrame({"geometry": [TILE_BOX], "id": ["cell1"]}, crs=CRS)
        subset = cat.filter(shape_gdf, datetime.datetime(2018, 5, 31), datetime.datetime(2018, 6, 2))
        assert len(subset) == 1

        slice_path = os.path.join(tmp, "slice.parquet")
        fs_module.write_parquet(slice_path, subset)  # hop 2 (setup's write)

        fresh_subset = fs_module.read_parquet(slice_path)  # hop 2's reader, "a separate process"
        flat = builder.flatten_catalog(fresh_subset)  # hop 3

        resolved = declaration_module.from_attrs(flat)
        assert resolved == custom
        assert resolved != builder.S2_L2A_DECLARATION

        out = os.path.join(tmp, "cube")
        builder.build_datacube(
            catalog_subset=flat, shape_gdf=shape_gdf,
            startdate=datetime.datetime(2018, 5, 31), enddate=datetime.datetime(2018, 6, 2),
            bands=["B04"], export_folderpath=out, if_missing_files=None,
        )
        md = fs.load_npy(os.path.join(out, "metadata.pickle.npy"), allow_pickle=True)[()]
        assert md["bands"] == ["B04"]


def test_unstamped_file_read_catalog_raises_at_flatten(tmp_path):
    """spec 35 §5a: a catalog gdf that came from a file (`fs.read_parquet` stamps
    `attrs[fs.SOURCE_PATH_ATTRS_KEY]`) with no declaration stamp is an error, not
    a silent S2 fallback."""
    from fsd.storage import fs as fs_module

    gdf = gpd.GeoDataFrame(
        [{"id": "x", "local_folderpath": "/tmp", "timestamp": 0,
          "files": "B04.jp2", "area_contribution": 50.0, "geometry": TILE_BOX}], crs=CRS,
    )
    p = str(tmp_path / "unstamped.parquet")
    fs_module.write_parquet(p, gdf)  # no attrs -> no stamp
    from_file = fs_module.read_parquet(p)

    with pytest.raises(ValueError, match="restamp_cli"):
        builder.flatten_catalog(from_file)


def test_hand_built_gdf_keeps_s2_default_no_raise():
    """spec 35 §5a: a hand-built GeoDataFrame (never through fs.read_parquet, so
    no fsd:source_path) keeps the S2 default -- an explicit in-process call is
    an explicit choice, and this preserves synthetic-test/notebook ergonomics."""
    gdf = gpd.GeoDataFrame(
        [{"id": "x", "local_folderpath": "/tmp", "timestamp": 0,
          "files": "B04.jp2", "area_contribution": 50.0, "offset": 0, "nodata": 0,
          "geometry": TILE_BOX}], crs=CRS,
    )
    flat = builder.flatten_catalog(gdf)
    from fsd.catalog import declaration as declaration_module

    assert declaration_module.from_attrs(flat) == builder.S2_L2A_DECLARATION


def test_get_dst_crs_picks_max_mean_area():
    gdf = gpd.GeoDataFrame({
        "crs": ["EPSG:32636", "EPSG:32636", "EPSG:32637"],
        "area_contribution": [10.0, 10.0, 80.0],
        "geometry": [TILE_BOX] * 3,
    }, crs=CRS)
    # 32637 has the higher *mean* (80 vs 10) despite fewer tiles
    assert builder._get_dst_crs(gdf).to_epsg() == 32637


def test_stack_merges_multiple_tiles_same_timestamp():
    """spec 20: two tiles of the SAME acquisition covering complementary halves of a
    shape are merged onto the reference grid, not collapsed to one (legacy dict kept the
    last -> the other half was silently nodata; the exact bug behind the demo gaps)."""
    ts = pd.Timestamp("2018-06-01", tz="UTC")
    ref = {"height": 1, "width": 4, "crs": RioCRS.from_epsg(32636)}
    left = np.array([[[10, 10, 0, 0]]], dtype=np.uint16)   # (1,1,4): valid left half
    right = np.array([[[0, 0, 20, 20]]], dtype=np.uint16)  #          valid right half
    dpl = [(left, {}), (right, {})]
    cat = gpd.GeoDataFrame({
        "timestamp": [ts, ts], "band": ["B04", "B04"], "image_index": [0, 1],
        "crs": ["EPSG:32636", "EPSG:32636"], "id": ["p", "p"],
        "geometry": [TILE_BOX, TILE_BOX]}, crs=CRS)
    shape = gpd.GeoDataFrame({"geometry": [TILE_BOX]}, crs=CRS)
    dc, md = builder._stack_datacube(cat, dpl, ["B04"], ref, shape, nodata=0)
    assert dc.shape == (1, 1, 4, 1)
    assert (dc[0, 0, :, 0] == [10, 10, 20, 20]).all()   # full coverage (legacy: [0,0,20,20])
    assert md["timestamps"] == [ts]


def test_stack_overlap_tiebreak_prefers_native_crs():
    """spec 20 SO-1: where two tiles both have valid data (overlap), the dst_crs-native
    tile wins; a reprojected tile only fills the pixels the native left as nodata."""
    ts = pd.Timestamp("2018-06-01", tz="UTC")
    ref = {"height": 1, "width": 2, "crs": RioCRS.from_epsg(32636)}
    native = np.array([[[0, 5]]], dtype=np.uint16)     # valid only at col 1
    reproj = np.array([[[7, 7]]], dtype=np.uint16)     # valid at both cols
    # list the reprojected tile FIRST to prove ordering (not list order) decides the winner
    dpl = [(reproj, {}), (native, {})]
    cat = gpd.GeoDataFrame({
        "timestamp": [ts, ts], "band": ["B04", "B04"], "image_index": [0, 1],
        "crs": ["EPSG:32637", "EPSG:32636"], "id": ["p", "p"],
        "geometry": [TILE_BOX, TILE_BOX]}, crs=CRS)
    shape = gpd.GeoDataFrame({"geometry": [TILE_BOX]}, crs=CRS)
    dc, _ = builder._stack_datacube(cat, dpl, ["B04"], ref, shape, nodata=0)
    # col 1 = 5 (native wins the overlap, not 7); col 0 = 7 (reprojected fills the gap)
    assert (dc[0, 0, :, 0] == [7, 5]).all()


def test_missing_files_raises_on_incomplete_area():
    tile = gpd.GeoDataFrame(
        [{"band": "B04", "timestamp": pd.Timestamp("2018-06-01", tz="UTC"),
          "geometry": box(0, 0, 1, 1)}], crs="EPSG:32633",
    )
    shape = gpd.GeoDataFrame({"geometry": [box(0, 0, 4, 4)]}, crs="EPSG:32633")
    with pytest.raises(ValueError, match="area coverage"):
        builder._missing_files_action(
            catalog_gdf=tile, shape_gdf=shape,
            startdate=datetime.datetime(2018, 5, 30),
            enddate=datetime.datetime(2018, 6, 2), bands=["B04"],
            if_missing_files="raise_error", max_timedelta_days=5,
        )


def test_build_datacube_clips_low_dn_to_zero_pinned_not_a_bug(tmp_path):
    """spec 34 §1f `[G1]` — PINNED INTENDED BEHAVIOR, NOT A BUG. DO NOT "FIX".

    The datacube is `uint16`, so applying a post-baseline offset (-1000) to a true
    reflectance DN in `(0,1000]` clips it to 0. On disk the raw DN is preserved
    intact (spec 34 §1 — see `test_stamped_cog_preserves_low_dn_on_disk`); the clip
    is a *cube-representation* consequence the spec accepts consciously, because a
    dtype change here would balloon every cube's footprint.

    This test exists so a future "fix" that silently changes the cube dtype (or
    re-introduces a signed/float intermediate) is caught here and forces a
    deliberate spec revision rather than a quiet behavior change.
    """
    ts = pd.Timestamp("2022-06-01", tz="UTC")
    rows = []
    b04_p = tmp_path / "clip_B04.tif"
    _write_tile(b04_p, np.full((4, 4), 400, dtype=np.uint16))  # 400 + (-1000) -> clip 0
    rows.append({**_band_row("tile_clip", b04_p, "B04", ts), "offset": -1000})
    scl_p = tmp_path / "clip_SCL.tif"
    _write_tile(scl_p, np.full((4, 4), 4, dtype=np.uint16))
    rows.append({**_band_row("tile_clip", scl_p, "SCL", ts), "offset": 0})
    catalog = gpd.GeoDataFrame(rows, crs=CRS)
    shape = gpd.GeoDataFrame({"geometry": [TILE_BOX]}, crs=CRS)
    out = tmp_path / "cube"

    builder.build_datacube(
        catalog_subset=catalog, shape_gdf=shape,
        startdate=datetime.datetime(2022, 1, 1), enddate=datetime.datetime(2022, 12, 1),
        bands=["B04", "SCL"], mosaic_days=1000,
        export_folderpath=str(out), if_missing_files=None, reference_band="B04",
    )
    dc = fs.load_npy(str(out / "datacube.npy"))
    assert dc.dtype == np.uint16          # the pinned dtype (§1f [G1])
    assert (dc[0, :, :, 0] == 0).all()    # 400 - 1000 clipped to 0, not wrapped to 64936


def test_build_datacube_uses_declared_reference_band_without_kwarg(tmp_path):
    """spec 34 §4 "reads the declaration, not `config`": with NO `reference_band=`
    kwarg, the build must take the reference band from the passed `CollectionDeclaration`.

    Proven by grid, not by introspection: B04 is written at 10 m and B08 at 20 m over
    the same extent, so whichever band is the resample reference decides the cube's
    height/width. A declaration naming B08 must yield the 20 m (2x2) grid — the
    `config.REFERENCE_BAND` default (B08 at 10 m in the S2 declaration) cannot
    produce that, so this can only pass if the declared value was honored.
    """
    ts = pd.Timestamp("2022-06-01", tz="UTC")
    b04_p = tmp_path / "ref_B04.tif"   # 4x4 @ 10 m
    _write_tile(b04_p, np.full((4, 4), 100, dtype=np.uint16))
    b08_p = tmp_path / "ref_B08.tif"   # 2x2 @ 20 m, same extent
    with rasterio.open(
        b08_p, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint16",
        crs=CRS, transform=from_origin(500000, 5000000, 20, 20), nodata=0,
    ) as dst:
        dst.write(np.full((1, 2, 2), 200, dtype=np.uint16))
    rows = [
        {**_band_row("tile_ref", b04_p, "B04", ts), "offset": 0},
        {**_band_row("tile_ref", b08_p, "B08", ts), "offset": 0},
    ]
    catalog = gpd.GeoDataFrame(rows, crs=CRS)
    shape = gpd.GeoDataFrame({"geometry": [TILE_BOX]}, crs=CRS)

    def _build(declaration, out_name):
        out = tmp_path / out_name
        builder.build_datacube(
            catalog_subset=catalog, shape_gdf=shape,
            startdate=datetime.datetime(2022, 1, 1),
            enddate=datetime.datetime(2022, 12, 1),
            bands=["B04", "B08"], mosaic_days=1000,
            export_folderpath=str(out), if_missing_files=None,
            declaration=declaration,   # NOTE: no reference_band= kwarg
        )
        return fs.load_npy(str(out / "datacube.npy"))

    no_mask = CollectionDeclaration(mask_spec=None, nodata=0)
    dc_20m = _build(dataclasses.replace(no_mask, reference_band="B08"), "ref_b08")
    dc_10m = _build(dataclasses.replace(no_mask, reference_band="B04"), "ref_b04")

    assert dc_20m.shape[1:3] == (2, 2)   # declared B08 (20 m) drove the grid
    assert dc_10m.shape[1:3] == (4, 4)   # declared B04 (10 m) drove the grid
    assert dc_20m.shape[1:3] != dc_10m.shape[1:3]  # the declaration is load-bearing


# --- instrumentation pins (#145 P12a): the sidecars' exact outputs -----------------

_INSTRUMENT_KW = dict(
    startdate=datetime.datetime(2018, 5, 31), enddate=datetime.datetime(2018, 7, 2),
    bands=["B04", "B08", "SCL"], mosaic_days=20, if_missing_files=None,
)


def _build_instrumented(tmp_path, catalog, shape, name="out", **flags):
    out = tmp_path / name
    builder.build_datacube(
        catalog_subset=catalog, shape_gdf=shape, export_folderpath=str(out),
        **_INSTRUMENT_KW, **flags,
    )
    return out


def _read_log_rows(out):
    text = (out / builder.READ_LOG_FILENAME).read_text()
    return [json.loads(ln) for ln in text.splitlines()]


@pytest.mark.parametrize("grid_id", [None, "grid_007"])
def test_timings_sidecar_payload_keys_values_and_layout(tmp_path, grid_id):
    catalog, shape = _make_catalog(tmp_path)
    if grid_id is not None:
        shape = shape.copy()
        shape["id"] = grid_id
    out = _build_instrumented(tmp_path, catalog, shape, write_timings=True)
    text = (out / builder.TIMINGS_FILENAME).read_text()
    payload = json.loads(text)
    assert text == json.dumps(payload, indent=2)
    assert list(payload) == [
        "id", "total_seconds", "phase_seconds", "n_band_rows", "n_images_loaded",
        "n_images_resampled", "n_mosaic_timestamps", "datacube_shape", "dst_crs"]
    assert list(payload["phase_seconds"]) == [
        "missing_check", "load_images", "dst_crs", "reference_profile", "resample", "stack",
        "ops", "save"]
    assert payload["id"] == grid_id
    assert payload["n_band_rows"] == 6
    assert payload["n_images_loaded"] == 6
    assert payload["n_images_resampled"] == 0
    assert payload["n_mosaic_timestamps"] == 2
    assert payload["datacube_shape"] == [2, 4, 4, 2]
    assert payload["dst_crs"] == "EPSG:32633"


def test_read_log_rows_order_keys_values_and_layout(tmp_path):
    catalog, shape = _make_catalog(tmp_path)
    shape = shape.copy()
    shape["id"] = "grid_007"
    out = _build_instrumented(tmp_path, catalog, shape, write_read_log=True)
    text = (out / builder.READ_LOG_FILENAME).read_text()
    assert text.endswith("\n")
    lines = text.splitlines()
    assert len(lines) == 6
    rows = [json.loads(ln) for ln in lines]
    for row, ln in zip(rows, lines):
        assert list(row) == ["id", "mgrs_tile", "product_id", "band", "filepath", "start",
                             "end", "duration"]
        assert ln == json.dumps(row)
    assert [(r["id"], r["mgrs_tile"], r["product_id"], r["band"]) for r in rows] == [
        ("grid_007", None, pid, band)
        for pid in ("tile_t1", "tile_t2") for band in ("B04", "B08", "SCL")]
    assert [r["filepath"] for r in rows] == [
        str(tmp_path / f"{tag}_{band}.tif")
        for tag in ("t1", "t2") for band in ("B04", "B08", "SCL")]


def test_read_log_mgrs_tile_from_the_product_id(tmp_path):
    catalog, shape = _make_catalog(tmp_path)
    catalog["id"] = catalog["id"].map({
        "tile_t1": "S2A_MSIL2A_20180406T100031_R122_T33UVP_20201012T185711",
        "tile_t2": "S2A_MSIL2A_20180506T100031_R122_T33UVP_20201012T185712"})
    out = _build_instrumented(tmp_path, catalog, shape, write_read_log=True)
    assert [r["mgrs_tile"] for r in _read_log_rows(out)] == ["33UVP"] * 6


def test_read_log_mgrs_tile_from_the_folder_name(tmp_path):
    catalog, shape = _make_catalog(tmp_path)
    folder = tmp_path / "X_T33UWP_Y"
    folder.mkdir()
    new_paths = []
    for p in catalog["filepath"]:
        moved = folder / os.path.basename(p)
        os.rename(p, moved)
        new_paths.append(str(moved))
    catalog["filepath"] = new_paths
    out = _build_instrumented(tmp_path, catalog, shape, write_read_log=True)
    assert [r["mgrs_tile"] for r in _read_log_rows(out)] == ["33UWP"] * 6


def test_read_log_with_parallel_load_warns_once_at_the_build_call(tmp_path):
    catalog, shape = _make_catalog(tmp_path)
    with pytest.warns(RuntimeWarning) as record:
        out = _build_instrumented(
            tmp_path, catalog, shape, write_read_log=True, njobs_load_images=2)
    # Only the builder's own warnings: rasterio adds others on some platforms (CI's Linux).
    ours = [w for w in record if w.filename == builder.__file__]
    assert len(ours) == 1
    assert ours[0].category is RuntimeWarning
    assert str(ours[0].message) == (
        "write_read_log requires njobs_load_images == 1; skipping read log.")
    # stacklevel=2: the warning points at build_datacube's `_load_images(` call.
    call = "catalog_gdf, data_profile_list, reads = _load_images("
    source = inspect.getsource(builder).splitlines()
    assert [i + 1 for i, line in enumerate(source) if call in line] == [ours[0].lineno]
    assert not (out / builder.READ_LOG_FILENAME).exists()


def test_both_flags_off_writes_only_the_cube_and_its_metadata(tmp_path):
    catalog, shape = _make_catalog(tmp_path)
    out = _build_instrumented(tmp_path, catalog, shape)
    assert sorted(p.name for p in out.iterdir()) == ["datacube.npy", "metadata.pickle.npy"]


def test_both_flags_on_leave_the_cube_and_metadata_unchanged(tmp_path):
    catalog, shape = _make_catalog(tmp_path)
    off = _build_instrumented(tmp_path, catalog, shape, name="off")
    on = _build_instrumented(
        tmp_path, catalog, shape, name="on", write_timings=True, write_read_log=True)
    assert np.array_equal(fs.load_npy(str(off / "datacube.npy")),
                          fs.load_npy(str(on / "datacube.npy")))
    md_off = fs.load_npy(str(off / "metadata.pickle.npy"), allow_pickle=True)[()]
    md_on = fs.load_npy(str(on / "metadata.pickle.npy"), allow_pickle=True)[()]
    assert md_off.keys() == md_on.keys()
    for key in md_off:
        a, b = md_off[key], md_on[key]
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), key
        elif hasattr(a, "equals"):
            assert a.equals(b), key
        else:
            assert a == b, key
