"""Tests for tests/data/tutorial/build_fixture.py (spec 42 §3) against a
synthetic mini-archive -- the generator is not run for real here (that needs
the blob archive, spec 42 D4 / run-book 43), only its selection/clipping/
cataloguing logic, mirroring `tests/test_datacube_builder.py`'s `_write_tile`/
`_make_catalog` shape."""

import importlib.util
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from rise_eo.catalog import declaration as declaration_module
from rise_eo.catalog.declaration import S2_L2A_DECLARATION
from rise_eo.storage import fs

_MODULE_PATH = os.path.join(
    os.path.dirname(__file__), "data", "tutorial", "build_fixture.py"
)
_spec = importlib.util.spec_from_file_location("build_fixture", _MODULE_PATH)
build_fixture_mod = importlib.util.module_from_spec(_spec)
sys.modules["build_fixture"] = build_fixture_mod
_spec.loader.exec_module(build_fixture_mod)

CRS = "EPSG:32633"
TRANSFORM = from_origin(500000, 5000000, 10, 10)  # 10 m pixels
GRANULE_SIZE = 50  # 50x50 px = 500x500 m granule footprint
GRANULE_BOX = box(500000, 4999500, 500500, 5000000)


def _write_band(path, val, size=GRANULE_SIZE):
    arr = np.full((1, size, size), val, dtype=np.uint16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="uint16", crs=CRS, transform=TRANSFORM, nodata=0,
    ) as dst:
        dst.write(arr)


def _make_archive(tmp_path, *, n_granules=2, with_offset_col=True, baseline_token="N0500"):
    root = tmp_path / "archive"
    rows = []
    for i in range(n_granules):
        tag = f"_{baseline_token}_" if baseline_token else "_"
        gid = f"S2B_MSIL2A_2018010{i + 1}T100019{tag}R122_T33UWP_20230101T00000{i}"
        folder = root / gid
        for band, val in [("B04", 100 + i), ("B08", 200 + i), ("SCL", 4)]:
            _write_band(str(folder / f"{band}.tif"), val)
        rows.append({
            "id": gid,
            "satellite": "sentinel-2-l2a",
            "timestamp": pd.Timestamp(f"2018-01-0{i + 1}", tz="UTC"),
            "s3url": "",
            "local_folderpath": str(folder),
            "files": "B04.tif,B08.tif,SCL.tif",
            "cloud_cover": 5.0,
            "geometry": GRANULE_BOX,
        })
    catalog = gpd.GeoDataFrame(rows, crs=CRS)
    if with_offset_col:
        catalog["offset"] = -1000
        catalog["nodata"] = 0
    fs.write_parquet(str(root / "catalog.parquet"), catalog)
    return str(root)


def _make_roi(tmp_path, name="roi.geojson"):
    # Inside GRANULE_BOX with margin on every side for the buffer.
    cell = box(500100, 4999600, 500400, 4999900)
    roi = gpd.GeoDataFrame({"id": ["testcell"]}, geometry=[cell], crs=CRS)
    path = str(tmp_path / name)
    roi.to_file(path, driver="GeoJSON")
    return path


# --- pure helpers -------------------------------------------------------


def test_mgrs_tile_of():
    assert build_fixture_mod.mgrs_tile_of(
        "S2B_MSIL2A_20180928T100019_N0500_R122_T33UWP_20230710T001349"
    ) == "33UWP"
    assert build_fixture_mod.mgrs_tile_of("no-tile-here") is None


def test_derive_offset_from_id():
    assert build_fixture_mod.derive_offset_from_id(
        "S2B_MSIL2A_20180928T100019_N0500_R122_T33UWP_x"
    ) == -1000
    assert build_fixture_mod.derive_offset_from_id(
        "S2A_MSIL2A_20180101T100019_N0212_R122_T33UWP_x"
    ) == 0
    # MPC ids carry no baseline token at all (A1).
    assert build_fixture_mod.derive_offset_from_id(
        "S2A_MSIL2A_20220601T075611_R035_T33UWP_20220601T120000"
    ) is None


def test_collapse_radiometry_column_prefers_offset_then_boa_add_offset():
    gdf = gpd.GeoDataFrame({"offset": [-1000], "boa_add_offset": [0]}, geometry=[GRANULE_BOX], crs=CRS)
    assert build_fixture_mod.collapse_radiometry_column(gdf) == "offset"
    gdf2 = gpd.GeoDataFrame({"boa_add_offset": [0]}, geometry=[GRANULE_BOX], crs=CRS)
    assert build_fixture_mod.collapse_radiometry_column(gdf2) == "boa_add_offset"
    gdf3 = gpd.GeoDataFrame({"other": [0]}, geometry=[GRANULE_BOX], crs=CRS)
    assert build_fixture_mod.collapse_radiometry_column(gdf3) is None


# --- select / check -------------------------------------------------------


def test_check_archive_declared_radiometry(tmp_path):
    root = _make_archive(tmp_path, n_granules=2, with_offset_col=True)
    roi_path = _make_roi(tmp_path)
    catalog_gdf = fs.read_parquet(os.path.join(root, "catalog.parquet"))
    roi_gdf = fs.read_geo(roi_path)

    summary = build_fixture_mod.check_archive(catalog_gdf, roi_gdf)
    assert summary["granules"] == 2
    assert summary["single_tile"] is True
    assert summary["mgrs_tiles"] == ["33UWP"]
    assert summary["bands"] == ["B04", "B08", "SCL"]
    assert summary["declaration_column"] == "offset"
    assert summary["declared_non_null"] == 2


def test_check_archive_falls_back_to_derived_offset_when_unstamped(tmp_path):
    root = _make_archive(tmp_path, n_granules=2, with_offset_col=False, baseline_token="N0500")
    roi_path = _make_roi(tmp_path)
    catalog_gdf = fs.read_parquet(os.path.join(root, "catalog.parquet"))
    roi_gdf = fs.read_geo(roi_path)

    summary = build_fixture_mod.check_archive(catalog_gdf, roi_gdf)
    assert summary["declaration_column"] == "derived-from-id"
    assert summary["declared_non_null"] == 2


# --- dry run --------------------------------------------------------------


def test_build_fixture_dry_run_is_zero_writes(tmp_path):
    root = _make_archive(tmp_path)
    roi_path = _make_roi(tmp_path)
    fields_path = _make_roi(tmp_path, name="fields.geojson")
    out_dir = str(tmp_path / "out")

    summary = build_fixture_mod.build_fixture(
        root, roi_path, fields_path, out_dir, ["B04", "B08", "SCL"],
        max_bytes=30 * 1024 * 1024, dry_run=True, progress=False,
    )
    assert summary["granules"] == 2
    assert set(summary["per_band_bytes"]) == {"B04", "B08", "SCL"}
    assert summary["total_bytes"] > 0
    assert summary["under_cap"] is True
    assert not os.path.exists(out_dir)  # zero side effects


def test_build_fixture_dry_run_over_cap(tmp_path):
    root = _make_archive(tmp_path)
    roi_path = _make_roi(tmp_path)
    fields_path = _make_roi(tmp_path, name="fields.geojson")
    out_dir = str(tmp_path / "out")

    summary = build_fixture_mod.build_fixture(
        root, roi_path, fields_path, out_dir, ["B04", "B08", "SCL"],
        max_bytes=1, dry_run=True, progress=False,
    )
    assert summary["under_cap"] is False


# --- real build -------------------------------------------------------------


def test_build_fixture_writes_stamped_catalog_and_notice(tmp_path):
    root = _make_archive(tmp_path, n_granules=2)
    roi_path = _make_roi(tmp_path)
    fields_path = _make_roi(tmp_path, name="fields.geojson")
    out_dir = str(tmp_path / "out")

    summary = build_fixture_mod.build_fixture(
        root, roi_path, fields_path, out_dir, ["B04", "B08", "SCL"],
        max_bytes=30 * 1024 * 1024, dry_run=False, progress=False,
    )
    assert summary["granules"] == 2
    assert summary["under_cap"] is True
    assert summary["offsets"] == [-1000]

    out_gdf = fs.read_parquet(os.path.join(out_dir, "catalog.parquet"))
    assert len(out_gdf) == 2
    assert (out_gdf["offset"] == -1000).all()
    assert (out_gdf["nodata"] == 0).all()
    assert out_gdf.geometry.notna().all()
    stamped = declaration_module.from_attrs(out_gdf)
    assert stamped == S2_L2A_DECLARATION

    notice_path = os.path.join(out_dir, "NOTICE")
    with open(notice_path) as f:
        assert f.read() == "Contains modified Copernicus Sentinel data 2018\n"

    assert os.path.exists(os.path.join(out_dir, "README.md"))
    assert os.path.exists(os.path.join(out_dir, "fields.geojson"))
    assert os.path.exists(os.path.join(out_dir, "roi.geojson"))

    for _, row in out_gdf.iterrows():
        for band in ("B04", "B08", "SCL"):
            fp = os.path.join(row["local_folderpath"], f"{band}.tif")
            assert os.path.exists(fp)
            with rasterio.open(fp) as src:
                assert src.width < GRANULE_SIZE  # actually clipped, not a copy
                assert src.height < GRANULE_SIZE


def test_build_fixture_geometry_recomputed_from_clip_not_source(tmp_path):
    root = _make_archive(tmp_path, n_granules=1)
    roi_path = _make_roi(tmp_path)
    fields_path = _make_roi(tmp_path, name="fields.geojson")
    out_dir = str(tmp_path / "out")

    build_fixture_mod.build_fixture(
        root, roi_path, fields_path, out_dir, ["B04", "B08", "SCL"],
        max_bytes=30 * 1024 * 1024, dry_run=False, progress=False,
    )
    out_gdf = fs.read_parquet(os.path.join(out_dir, "catalog.parquet"))
    clipped_geom_utm = gpd.GeoSeries([out_gdf.geometry.iloc[0]], crs="EPSG:4326").to_crs(CRS)
    # The clipped footprint must be strictly smaller than the source granule's
    # footprint -- spec 42 §3 step 3 (never copy the source geometry).
    assert clipped_geom_utm.area.iloc[0] < gpd.GeoSeries([GRANULE_BOX], crs=CRS).area.iloc[0]


def test_build_fixture_is_idempotent_per_file(tmp_path):
    root = _make_archive(tmp_path, n_granules=1)
    roi_path = _make_roi(tmp_path)
    fields_path = _make_roi(tmp_path, name="fields.geojson")
    out_dir = str(tmp_path / "out")

    build_fixture_mod.build_fixture(
        root, roi_path, fields_path, out_dir, ["B04", "B08"],
        max_bytes=30 * 1024 * 1024, dry_run=False, progress=False,
    )
    out_gdf = fs.read_parquet(os.path.join(out_dir, "catalog.parquet"))
    fp = os.path.join(out_gdf.iloc[0]["local_folderpath"], "B04.tif")
    mtime_before = os.path.getmtime(fp)

    build_fixture_mod.build_fixture(
        root, roi_path, fields_path, out_dir, ["B04", "B08"],
        max_bytes=30 * 1024 * 1024, dry_run=False, progress=False,
    )
    assert os.path.getmtime(fp) == mtime_before  # skipped, not re-clipped


def test_build_fixture_max_timestamps_fallback(tmp_path):
    root = _make_archive(tmp_path, n_granules=2)
    roi_path = _make_roi(tmp_path)
    fields_path = _make_roi(tmp_path, name="fields.geojson")
    out_dir = str(tmp_path / "out")

    summary = build_fixture_mod.build_fixture(
        root, roi_path, fields_path, out_dir, ["B04", "B08", "SCL"],
        max_bytes=30 * 1024 * 1024, max_timestamps=1, dry_run=False, progress=False,
    )
    assert summary["granules"] == 1


def test_readme_redacts_the_archive_url(tmp_path):
    """`tests/data/tutorial/README.md` is committed to a PUBLIC MIT repo (spec 42
    D6) and run-book 43 Step 4 is invoked with the shell-expanded
    `--archive-root abfss://...`, so a verbatim sys.argv would publish a concrete
    infrastructure identifier."""
    argv = [
        "tests/data/tutorial/build_fixture.py",
        "--archive-root", "abfss://myfs@myaccount.dfs.core.windows.net/secret/archive",
        "--roi", "tests/data/tutorial/roi.geojson",
        "--bands", "B04", "B08", "SCL",
    ]
    redacted = " ".join(build_fixture_mod.redact_argv(argv))
    assert "abfss://" not in redacted
    assert "myaccount" not in redacted
    assert "--archive-root <archive-root>" in redacted
    # non-secret flags survive verbatim
    assert "--bands B04 B08 SCL" in redacted
    # the `--flag=value` spelling is redacted too
    joined = " ".join(build_fixture_mod.redact_argv(
        ["x.py", "--archive-root=abfss://a@b.dfs.core.windows.net/c"]
    ))
    assert "abfss://" not in joined


def test_build_fixture_readme_has_no_archive_url(tmp_path, monkeypatch):
    root = _make_archive(tmp_path, n_granules=1)
    roi_path = _make_roi(tmp_path)
    fields_path = _make_roi(tmp_path, name="fields.geojson")
    out_dir = str(tmp_path / "out")
    monkeypatch.setattr(
        build_fixture_mod.sys, "argv",
        ["build_fixture.py", "--archive-root", "abfss://fs@acct.dfs.core.windows.net/p"],
    )
    build_fixture_mod.build_fixture(
        root, roi_path, fields_path, out_dir, ["B04", "B08"],
        max_bytes=30 * 1024 * 1024, dry_run=False, progress=False,
    )
    with open(os.path.join(out_dir, "README.md")) as f:
        readme = f.read()
    assert "abfss://" not in readme
    assert "acct" not in readme


def test_build_fixture_does_not_rewrite_its_own_inputs(tmp_path):
    """Run-book 43 Step 4 passes --roi/--fields INSIDE --out, so the generator's
    inputs are its outputs. The hazard is not that the bytes would differ (a
    GeoJSON round-trip is often byte-identical) -- it is that ANY write there
    truncates the destination first, so a crash mid-write destroys Step 0's
    output, which cannot be regenerated on the VM (Step 0 needs `shapefiles/`
    from the workspace root). The only safe behavior is: do not write at all.

    Asserted on mtime, not content: content equality passes even when the file
    IS rewritten, which makes it a vacuous guard (verified by negative control).
    """
    root = _make_archive(tmp_path, n_granules=1)
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)
    roi_path = os.path.join(out_dir, "roi.geojson")
    fields_path = os.path.join(out_dir, "fields.geojson")
    cell = box(500100, 4999600, 500400, 4999900)
    gpd.GeoDataFrame({"id": ["testcell"]}, geometry=[cell], crs=CRS).to_file(
        roi_path, driver="GeoJSON")
    gpd.GeoDataFrame({"fid": [1], "label": ["maize"]}, geometry=[cell], crs=CRS).to_file(
        fields_path, driver="GeoJSON")
    before = {p: os.stat(p).st_mtime_ns for p in (roi_path, fields_path)}
    contents = {p: open(p, "rb").read() for p in (roi_path, fields_path)}

    build_fixture_mod.build_fixture(
        root, roi_path, fields_path, out_dir, ["B04", "B08"],
        max_bytes=30 * 1024 * 1024, dry_run=False, progress=False,
    )
    for p in (roi_path, fields_path):
        assert os.stat(p).st_mtime_ns == before[p], f"{p} was rewritten in place"
        assert open(p, "rb").read() == contents[p]


def test_build_fixture_copies_inputs_when_out_dir_differs(tmp_path):
    """The skip is scoped to the same-path case only -- a separate --out still
    gets its own roi/fields copies, byte-for-byte (`storage.transfer`, so an
    `abfss://` --out works too; A1 D4 / ADR 0003)."""
    root = _make_archive(tmp_path, n_granules=1)
    roi_path = _make_roi(tmp_path)
    fields_path = _make_roi(tmp_path, name="fields.geojson")
    out_dir = str(tmp_path / "elsewhere")

    build_fixture_mod.build_fixture(
        root, roi_path, fields_path, out_dir, ["B04", "B08"],
        max_bytes=30 * 1024 * 1024, dry_run=False, progress=False,
    )
    assert open(os.path.join(out_dir, "roi.geojson"), "rb").read() == open(roi_path, "rb").read()
    assert open(os.path.join(out_dir, "fields.geojson"), "rb").read() == open(fields_path, "rb").read()


def test_build_fixture_raises_on_missing_fields_input(tmp_path):
    root = _make_archive(tmp_path, n_granules=1)
    roi_path = _make_roi(tmp_path)
    with pytest.raises(FileNotFoundError, match="Step 0"):
        build_fixture_mod.build_fixture(
            root, roi_path, str(tmp_path / "nope.geojson"), str(tmp_path / "out"),
            ["B04"], max_bytes=30 * 1024 * 1024, dry_run=False, progress=False,
        )


def test_build_fixture_records_offset_source(tmp_path):
    """spec 42 A2: "copied from the source, not invented" is only observable
    where the source is reachable, so it must reach `_result.json`."""
    root = _make_archive(tmp_path, n_granules=2, with_offset_col=True)
    roi_path = _make_roi(tmp_path)
    fields_path = _make_roi(tmp_path, name="fields.geojson")
    summary = build_fixture_mod.build_fixture(
        root, roi_path, fields_path, str(tmp_path / "out"), ["B04"],
        max_bytes=30 * 1024 * 1024, dry_run=False, progress=False,
    )
    assert summary["offset_sources"] == {"declared": 2}
    assert summary["all_offsets_declared"] is True

    # the D1 fallback path is visibly NOT "declared"
    root2 = _make_archive(tmp_path / "b", n_granules=2, with_offset_col=False)
    summary2 = build_fixture_mod.build_fixture(
        root2, roi_path, fields_path, str(tmp_path / "out2"), ["B04"],
        max_bytes=30 * 1024 * 1024, dry_run=False, progress=False,
    )
    assert summary2["offset_sources"] == {"derived": 2}
    assert summary2["all_offsets_declared"] is False


def test_check_archive_gates_on_requested_bands(tmp_path):
    root = _make_archive(tmp_path, n_granules=1)
    roi_path = _make_roi(tmp_path)
    catalog_gdf = fs.read_parquet(os.path.join(root, "catalog.parquet"))
    roi_gdf = fs.read_geo(roi_path)

    ok = build_fixture_mod.check_archive(catalog_gdf, roi_gdf, bands=["B04", "B08", "SCL"])
    assert ok["all_bands_present"] is True and ok["missing_bands"] == []
    bad = build_fixture_mod.check_archive(catalog_gdf, roi_gdf, bands=["B04", "B12"])
    assert bad["all_bands_present"] is False and bad["missing_bands"] == ["B12"]
    # omitted -> reported as unevaluated, never silently satisfied
    assert "all_bands_present" not in build_fixture_mod.check_archive(catalog_gdf, roi_gdf)
    assert ok["offset_sources"] == {"declared": 1}
    assert ok["offset_values"] == [-1000]


def test_subsample_timestamps_spreads_across_the_span():
    """spec 42 D2's fallback must keep the SEASON, not the first N granules."""
    sel = gpd.GeoDataFrame(
        {"id": [f"g{i}" for i in range(24)]},
        geometry=[GRANULE_BOX] * 24, crs=CRS,
    )
    kept = build_fixture_mod.subsample_timestamps(sel, 12)
    assert len(kept) == 12
    assert kept["id"].iloc[0] == "g0"
    assert kept["id"].iloc[-1] == "g23"          # endpoints kept -> full span
    assert list(kept["id"]) != [f"g{i}" for i in range(12)]  # not just the head
    # no-op when already small enough
    assert len(build_fixture_mod.subsample_timestamps(sel, 50)) == 24


def test_build_fixture_raises_when_radiometry_unresolvable(tmp_path):
    # No offset column AND no baseline token in the id (MPC-style) -- refuse
    # to guess (spec 42 D1/A1).
    root = _make_archive(tmp_path, n_granules=1, with_offset_col=False, baseline_token=None)
    roi_path = _make_roi(tmp_path)
    fields_path = _make_roi(tmp_path, name="fields.geojson")
    out_dir = str(tmp_path / "out")

    with pytest.raises(ValueError, match="no radiometry declaration"):
        build_fixture_mod.build_fixture(
            root, roi_path, fields_path, out_dir, ["B04", "B08", "SCL"],
            max_bytes=30 * 1024 * 1024, dry_run=False, progress=False,
        )


# --- spec 42 D2's Apr-Sep window (run-book 43 Step 2 returned a FULL YEAR) ----


def _make_year_archive(tmp_path):
    """12 monthly granules across 2018 -- the shape the blob archive actually has
    (Step 2 on the VM: 72 granules, 2018-01-01 .. 2019-01-01), not the local
    archive's Apr-Sep-only 24."""
    root = tmp_path / "archive"
    rows = []
    for month in range(1, 13):
        gid = f"S2B_MSIL2A_2018{month:02d}15T100019_R122_T33UWP_20201014T00000{month % 10}"
        folder = root / gid
        for band, val in [("B04", 100), ("B08", 200), ("SCL", 4)]:
            _write_band(str(folder / f"{band}.tif"), val)
        rows.append({
            "id": gid, "satellite": "sentinel-2-l2a",
            "timestamp": pd.Timestamp(f"2018-{month:02d}-15", tz="UTC"),
            "s3url": "", "local_folderpath": str(folder),
            "files": "B04.tif,B08.tif,SCL.tif", "cloud_cover": 5.0,
            "geometry": GRANULE_BOX,
        })
    catalog = gpd.GeoDataFrame(rows, crs=CRS)
    catalog["offset"] = 0      # MPC pre-baseline-04.00 products declare 0, correctly
    catalog["nodata"] = 0
    fs.write_parquet(str(root / "catalog.parquet"), catalog)
    return str(root)


def test_select_granules_applies_the_date_window(tmp_path):
    root = _make_year_archive(tmp_path)
    roi_path = _make_roi(tmp_path)
    catalog_gdf = fs.read_parquet(os.path.join(root, "catalog.parquet"))
    roi_gdf = fs.read_geo(roi_path)

    assert len(build_fixture_mod.select_granules(catalog_gdf, roi_gdf)) == 12
    apr_sep = build_fixture_mod.select_granules(
        catalog_gdf, roi_gdf, "2018-04-01", "2018-09-30")
    assert len(apr_sep) == 6
    assert str(apr_sep["timestamp"].min().date()) == "2018-04-15"
    assert str(apr_sep["timestamp"].max().date()) == "2018-09-15"
    # bounds are inclusive
    assert len(build_fixture_mod.select_granules(
        catalog_gdf, roi_gdf, "2018-04-15", "2018-04-15")) == 1


def test_check_archive_reports_the_window_and_the_pre_filter_count(tmp_path):
    root = _make_year_archive(tmp_path)
    roi_path = _make_roi(tmp_path)
    catalog_gdf = fs.read_parquet(os.path.join(root, "catalog.parquet"))
    roi_gdf = fs.read_geo(roi_path)

    s = build_fixture_mod.check_archive(
        catalog_gdf, roi_gdf, bands=["B04", "B08", "SCL"],
        startdate="2018-04-01", enddate="2018-09-30")
    assert s["granules"] == 6
    assert s["granules_before_date_filter"] == 12
    assert s["date_window"] == ["2018-04-01", "2018-09-30"]
    assert s["date_span"] == ["2018-04-15", "2018-09-15"]
    # offset 0 is a REAL, correct declaration for pre-baseline-04.00 MPC products
    assert s["offset_values"] == [0]
    assert s["offset_sources"] == {"declared": 6}


def test_build_fixture_honours_the_date_window(tmp_path):
    root = _make_year_archive(tmp_path)
    roi_path = _make_roi(tmp_path)
    fields_path = _make_roi(tmp_path, name="fields.geojson")

    summary = build_fixture_mod.build_fixture(
        root, roi_path, fields_path, str(tmp_path / "out"), ["B04"],
        max_bytes=30 * 1024 * 1024, startdate="2018-04-01", enddate="2018-09-30",
        dry_run=False, progress=False,
    )
    assert summary["granules"] == 6
    assert summary["date_span"] == ["2018-04-15", "2018-09-15"]
    assert summary["offsets"] == [0]


def test_build_fixture_raises_when_the_window_selects_nothing(tmp_path):
    root = _make_year_archive(tmp_path)
    roi_path = _make_roi(tmp_path)
    fields_path = _make_roi(tmp_path, name="fields.geojson")

    with pytest.raises(ValueError, match="no granule intersects the cell within"):
        build_fixture_mod.build_fixture(
            root, roi_path, fields_path, str(tmp_path / "out"), ["B04"],
            max_bytes=30 * 1024 * 1024, startdate="2019-01-01", enddate="2019-12-31",
            dry_run=False, progress=False,
        )
