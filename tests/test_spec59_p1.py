"""Spec 59 P1 -- imagery archive layout, network-free. One group of tests per acceptance
criterion (AC numbers refer to specs/59-imagery-archive-layout.md section 5).

Synthetic items modelled on the real MPC / CDSE STAC items the spec was verified against
(demo_e2e catalog: MPC `s2:product_uri` carries the `N0212` baseline field its item id drops).
"""

from __future__ import annotations

import datetime
import os
import types

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely.geometry as sg
from rasterio.transform import from_origin

import rise_eo
from rise_eo import api, config
from rise_eo import collections as _collections
from rise_eo.catalog import processing as processing_module
from rise_eo.catalog.catalog import TileCatalog, filter_by_properties
from rise_eo.catalog.declaration import S2_L2A_DECLARATION
from rise_eo.collections import naming
from rise_eo.datacube import builder
from rise_eo.sources import cdse, mpc
from rise_eo.sources._granules import granule_folderpath, select_granules
from rise_eo.storage import fs
from rise_eo.workflows import create_datacube, runners
from rise_eo.workflows import stamp as _stamp

S2 = config.SATELLITE_S2L2A
S1 = "sentinel-1-rtc"
DT = "2018-09-18T10:00:19Z"


# --- fakes ---------------------------------------------------------------------


class _Item:
    def __init__(self, id, dt, properties, assets, geom=None):
        self.id = id
        self.datetime = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00"))
        self.geometry = sg.mapping(geom or sg.box(0, 0, 1, 1))
        self.properties = properties
        self.assets = {k: types.SimpleNamespace(href=v) for k, v in assets.items()}


def _esa(baseline: str, disc: str, dt=DT, tile="T33UWP") -> str:
    d = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00"))
    return (f"S2B_MSIL2A_{d:%Y%m%dT%H%M%S}_N{baseline.replace('.', '')}_R122_{tile}_{disc}")


def _mpc_item(baseline="02.12", disc="20201009T023142", dt=DT, tile="T33UWP", cloud=1.0):
    name = _esa(baseline, disc, dt, tile)
    mpc_id = name.replace(f"_N{baseline.replace('.', '')}", "")   # MPC's id drops the N field
    gen = f"{disc[:4]}-{disc[4:6]}-{disc[6:8]}T{disc[9:11]}:{disc[11:13]}:{disc[13:15]}.794Z"
    return _Item(mpc_id, dt, {
        "eo:cloud_cover": cloud, "s2:product_uri": name + ".SAFE",
        "s2:processing_baseline": baseline, "s2:generation_time": gen,
        "s2:mgrs_tile": tile,
    }, {"B04": f"https://mpc/{mpc_id}/B04.tif"})


def _cdse_item(baseline="05.00", disc="20230915T000622", dt=DT, tile="T33UWP", cloud=1.0):
    name = _esa(baseline, disc, dt, tile)
    safe = f"s3://eodata/Sentinel-2/MSI/L2A_N0500/2018/09/18/{name}.SAFE"
    proc = f"{disc[:4]}-{disc[4:6]}-{disc[6:8]}T{disc[9:11]}:{disc[11:13]}:{disc[13:15]}Z"
    return _Item(name, dt, {
        "eo:cloud_cover": cloud, "processing:version": baseline, "processing:datetime": proc,
    }, {
        "B04_10m": f"{safe}/GRANULE/G/IMG_DATA/R10m/T_D_B04_10m.jp2",
        "granule_metadata": f"{safe}/GRANULE/G/MTD_TL.xml",
    })


def _canon(item) -> str:
    return naming.granule_info(S2, item.id, item.properties, item.datetime).canonical_name


def _mpc_gdf(items):
    return mpc._items_to_gdf(items, collection=S2, declaration=_collections.get(S2))


def _cdse_gdf(items):
    return cdse._items_to_gdf(items, collection=S2, declaration=_collections.get(S2))


def _roi():
    return gpd.GeoDataFrame(geometry=[sg.box(0.2, 0.2, 0.5, 0.5)], crs="EPSG:4326")


# --- AC2: same ESA product from MPC and CDSE -> one folder, one row ----------------


def test_ac2_same_product_from_two_sources_is_one_folder_and_one_row(tmp_path):
    m = _mpc_item(baseline="05.00", disc="20230915T000622")
    c = _cdse_item(baseline="05.00", disc="20230915T000622")
    name = _esa("05.00", "20230915T000622")
    assert _canon(m) == _canon(c) == name
    assert m.id != c.id                                  # the providers really do differ

    [(_, mpc_dst, _)] = mpc._select_item_files(m, ["B04"], str(tmp_path))
    [(_, cdse_dst), _xml] = cdse._select_item_files(c, ["B04"], str(tmp_path))
    assert os.path.dirname(mpc_dst) == os.path.dirname(cdse_dst)
    assert os.path.dirname(mpc_dst) == str(tmp_path / S2 / "2018" / "09" / "18" / name)

    catalog = TileCatalog(str(tmp_path / S2 / "catalog.parquet"))
    gm, gc = _mpc_gdf([m]), _cdse_gdf([c])
    mpc._append_downloaded(catalog, {name: gm.iloc[0]}, [(name, mpc_dst, True)], _collections.get(S2))
    cdse._append_downloaded(catalog, {name: gc.iloc[0]}, [(name, cdse_dst, True)], _collections.get(S2))
    rows = catalog.read()
    assert list(rows["id"]) == [name]
    assert list(rows["source"]) == ["cdse,mpc"]


# --- AC3: two baselines -> two folders, two rows, one acquisition key -----------------


def test_ac3_two_baselines_are_two_granules_of_one_acquisition(tmp_path):
    old, new = _mpc_item("02.12", "20201009T023142"), _mpc_item("05.00", "20230915T000622")
    folders = {os.path.dirname(mpc._select_item_files(i, ["B04"], str(tmp_path))[0][1])
               for i in (old, new)}
    assert len(folders) == 2
    gdf = _mpc_gdf([old, new])
    assert len(set(gdf["id"])) == 2
    assert set(gdf["acquisition_key"]) == {"S2B_MSIL2A_20180918T100019_R122_T33UWP"}
    assert list(gdf["processing_version"]) == ["02.12", "05.00"]


# --- AC4: HLS files under the id's date, not STAC datetime ---------------------------


def test_ac4_hls_date_comes_from_the_id_not_the_stac_datetime(tmp_path):
    # id: 2021 day-of-year 123 = 2021-05-03. STAC datetime says the 4th (a different UTC day).
    info = naming.granule_info("hls2-s30", "HLS.S30.T33UWP.2021123T235501.v2.0", {},
                               pd.Timestamp("2021-05-04T00:02:00Z"))
    assert info.acquisition_date == datetime.date(2021, 5, 3)
    assert info.acquisition_key == "HLS.S30.T33UWP.2021123T235501"
    assert info.processing_version == "2.0"
    assert granule_folderpath(str(tmp_path), "hls2-s30", info) == str(
        tmp_path / "hls2-s30" / "2021" / "05" / "03" / "HLS.S30.T33UWP.2021123T235501.v2.0")


def test_hls_day_of_year_out_of_range_raises_instead_of_rolling_over():
    assert naming.granule_info("hls2-s30", "HLS.S30.T33UWP.2020366T100031.v2.0", {},
                               None).acquisition_date == datetime.date(2020, 12, 31)   # leap
    for doy in ("366", "000"):      # 2021 is not a leap year; there is no day 0
        with pytest.raises(ValueError, match="day of year"):
            naming.granule_info("hls2-s30", f"HLS.S30.T33UWP.2021{doy}T100031.v2.0", {}, None)


def test_a_collection_without_a_parser_gets_the_any_other_row():
    info = naming.granule_info("my-collection", "scene-1", {}, pd.Timestamp("2020-02-29T23:59Z"))
    assert (info.canonical_name, info.acquisition_key) == ("scene-1", "scene-1")
    assert info.acquisition_date == datetime.date(2020, 2, 29)
    assert info.processing_version is None
    assert not naming.publishes_version("my-collection")


# --- AC5 / AC6: the duplicate check and the selector ---------------------------------


def _frame(*rows):
    return pd.DataFrame(
        [{"id": i, "acquisition_key": k, "processing_version": v, "processing_datetime": d,
          "source": s} for i, k, v, d, s in rows])


AB = _frame(
    ("A_0212", "A", "02.12", pd.Timestamp("2020-10-09", tz="UTC"), "mpc"),
    ("A_0500", "A", "05.00", pd.Timestamp("2023-09-15", tz="UTC"), "cdse"),
    ("B_0212", "B", "02.12", pd.Timestamp("2020-10-19", tz="UTC"), "mpc"),
)


def test_ac5_processing_none_raises_listing_the_group_and_the_resolving_arguments():
    with pytest.raises(ValueError) as exc:
        processing_module.select_processing(AB, None)
    msg = str(exc.value)
    for expected in ("acquisition A", "A_0212", "A_0500", "source=mpc", "source=cdse",
                     "processing_version=02.12", "processing_datetime=2023-09-15",
                     "processing='latest'", "properties_filter"):
        assert expected in msg
    assert "acquisition B" not in msg          # a group of one is not a duplicate


def test_ac6_latest_keeps_a_at_0500_and_b_at_0212():
    sel = processing_module.select_processing(AB, "latest")
    assert set(sel.kept["id"]) == {"A_0500", "B_0212"}
    assert [s["id"] for s in sel.skipped] == ["A_0212"]


@pytest.mark.parametrize("spec", [">=04.00", "==05.00", ">=05.00,<06"])
def test_ac6_a_specifier_keeps_a_at_0500_drops_b_and_reports_it(spec, capsys):
    sel = processing_module.select_processing(AB, spec)
    assert set(sel.kept["id"]) == {"A_0500"}
    assert sel.dropped_acquisitions == ["B"]
    processing_module.print_selection(sel, prefix="[t]", processing=spec)
    assert "dropped acquisition B" in capsys.readouterr().out


def test_ac6_pep440_normalizes_zero_padded_baselines():
    keep = processing_module.select_processing(AB, "<=2.12").kept
    assert set(keep["id"]) == {"A_0212", "B_0212"}


def test_ac7_latest_raises_only_when_it_must_choose_and_cannot():
    twins = _frame(("s1_a", "S", None, None, "mpc"), ("s1_b", "S", None, None, "mpc"))
    with pytest.raises(ValueError, match=r"s1_a[\s\S]*s1_b"):
        processing_module.select_processing(twins, "latest")
    single = _frame(("s1_a", "S", None, None, "mpc"), ("s1_c", "T", None, None, "mpc"))
    assert len(processing_module.select_processing(single, "latest").kept) == 2


def test_latest_orders_version_then_datetime_and_sorts_nulls_lowest():
    rows = _frame(
        ("null", "K", None, None, "mpc"),
        ("v5_old", "K", "05.00", pd.Timestamp("2021-01-01", tz="UTC"), "mpc"),
        ("v5_new", "K", "05.00", pd.Timestamp("2022-01-01", tz="UTC"), "mpc"),
        ("v2_newest", "K", "02.12", pd.Timestamp("2030-01-01", tz="UTC"), "mpc"),
    )
    assert list(processing_module.select_processing(rows, "latest").kept["id"]) == ["v5_new"]


def test_ac5_setup_raises_on_duplicates_and_latest_resolves_them(tmp_path):
    catalog = _two_processing_catalog(tmp_path)
    kwargs = _setup_kwargs(tmp_path, catalog)
    with pytest.raises(ValueError, match="acquisition S2B_MSIL2A_20180918T100019_R122_T33UWP"):
        create_datacube.setup(**kwargs)
    create_datacube.setup(**kwargs, processing="latest")
    slice_ = fs.read_parquet(str(tmp_path / "run" / _segment(processing="latest") / "c1"
                                 / "catalog.parquet"))
    assert list(slice_["processing_version"]) == ["05.00"]
    assert pd.read_csv(tmp_path / "input.csv")["processing"].tolist() == ["latest"]


def test_ac5_the_builder_refuses_duplicates_too(tmp_path):
    """No entry point routes around D6: `build_datacube` runs the same check."""
    gdf = TileCatalog(_two_processing_catalog(tmp_path)).read()
    gdf["area_contribution"] = 100.0
    flat = builder.flatten_catalog(gdf)
    shape = gpd.GeoDataFrame({"id": ["c1"], "geometry": [sg.box(0, 0, 1, 1)]}, crs="EPSG:4326")
    common = dict(shape_gdf=shape, startdate=datetime.datetime(2018, 9, 1),
                  enddate=datetime.datetime(2018, 10, 1), bands=["B04"], mosaic_days=20,
                  export_folderpath=str(tmp_path / "cube"))
    with pytest.raises(ValueError, match="more than one processing"):
        builder.build_datacube(flat, **common)
    with pytest.raises(ValueError, match="left no rows"):
        builder.build_datacube(flat, processing=">=9", **common)


def _two_processing_catalog(tmp_path) -> str:
    """A catalog with ONE acquisition in two processings (02.12 from MPC, 05.00 from CDSE)."""
    fp = str(tmp_path / S2 / "catalog.parquet")
    cat = TileCatalog(fp)
    rows = []
    for item, src, gdf in ((_mpc_item("02.12", "20201009T023142"), "mpc", None),
                           (_cdse_item("05.00", "20230915T000622"), "cdse", None)):
        row = (_mpc_gdf if src == "mpc" else _cdse_gdf)([item]).iloc[0].to_dict()
        row.update(local_folderpath=str(tmp_path / row["id"]), files="B04.tif")
        rows.append(row)
    cat.append(rows, declaration=S2_L2A_DECLARATION)
    return fp


def _segment(processing=None):
    return create_datacube.window_folder_segment(
        pd.Timestamp("2018-09-01", tz="UTC"), pd.Timestamp("2018-10-01", tz="UTC"), 20,
        bands=["B04"], mosaic_scheme=config.MOSAIC_SCHEME, collection=S2,
        declaration=S2_L2A_DECLARATION, processing=processing)


def _setup_kwargs(tmp_path, catalog) -> dict:
    shapes = tmp_path / "shapes.geojson"
    gpd.GeoDataFrame({"id": ["c1"], "geometry": [sg.box(0, 0, 1, 1)]},
                     crs="EPSG:4326").to_file(shapes, driver="GeoJSON")
    return dict(
        catalog_filepath=catalog, timestamp_col="timestamp", shapefilepath=str(shapes),
        id_col="id", run_folderpath=str(tmp_path / "run"),
        startdate=datetime.datetime(2018, 9, 1), enddate=datetime.datetime(2018, 10, 1),
        bands=["B04"], mosaic_days=20, csv_filepath=str(tmp_path / "input.csv"), label_col=None,
    )


# --- AC7: a version specifier against S1 RTC raises at preflight -----------------------


def test_ac7_a_specifier_on_sentinel1_raises_naming_the_collection_on_download(tmp_path):
    with pytest.raises(api.PreflightError, match=r"sentinel-1-rtc.*publishes no processing"):
        api.download(
            _roi(), datetime.datetime(2018, 6, 1), datetime.datetime(2018, 7, 1), ["vv"],
            str(tmp_path), collection=S1, processing=">=05.00", max_tiles=1,
        )


def test_ac7_a_specifier_on_sentinel1_raises_on_a_build_and_latest_does_not(tmp_path):
    polys = gpd.GeoDataFrame({"fid": [1], "geometry": [sg.box(0, 0, 0.01, 0.01)]},
                             crs="EPSG:4326")
    kw = dict(
        label_polygons=polys, catalog_filepath=str(tmp_path / S1 / "catalog.parquet"),
        startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 1),
        mosaic_days=10, bands=["vv"], id_col="fid", export_folderpath=str(tmp_path / "out"),
        collection=S1,
    )
    with pytest.raises(api.PreflightError, match=r"sentinel-1-rtc.*publishes no processing"):
        api.create_training_data(**kw, processing=">=05.00")
    # "latest" passes the version preflight; it fails later, only for the missing catalog
    with pytest.raises(api.PreflightError, match="catalog_filepath does not exist"):
        api.create_training_data(**kw, processing="latest")


# --- AC8: cube identity ------------------------------------------------------------


def test_ac8_params_key_is_unchanged_when_processing_is_none_and_differs_otherwise():
    kw = dict(collection=S2, declaration=S2_L2A_DECLARATION)
    args = (["B04", "B08", "SCL"], "calendar")
    # Pinned against the pre-spec-59 implementation (computed on main before this change).
    assert create_datacube.params_key(*args, **kw) == "1a6dcdf3"
    assert create_datacube.params_key(*args, **kw, processing=None) == "1a6dcdf3"
    keys = {create_datacube.params_key(*args, **kw, processing=p)
            for p in ("latest", ">=05.00", "==05.00")}
    assert len(keys) == 3 and "1a6dcdf3" not in keys


def test_ac8_a_specifier_is_canonicalised_before_it_enters_the_path():
    canon = create_datacube._canonicalize_processing
    assert canon(">=05.00, <06") == canon("<06,>=05.00")
    assert canon(None) == ""


# --- AC9: download's default "latest", on CDSE (which never deduplicated) ---------------


def test_ac9_cdse_keeps_the_later_of_two_processings_and_prints_the_skip(
        monkeypatch, tmp_path, capsys):
    old = _cdse_item("04.00", "20220301T000000")
    new = _cdse_item("05.00", "20230915T000622")
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: [old, new])
    got = cdse.query_catalog(_roi(), datetime.datetime(2018, 1, 1), datetime.datetime(2019, 1, 1))
    assert list(got["id"]) == [_canon(new)]
    tiles = select_granules(_cdse_gdf([old, new]), processing="latest",
                            prefix="[rise_eo.cdse.download]")
    assert list(tiles["id"]) == [_canon(new)]
    out = capsys.readouterr().out
    assert f"skipped {_canon(old)}" in out and "superseded" in out


def test_ac9_mpc_selects_the_same_item_spec_33_did():
    """Spec 33: same sensing time + tile -> the newest `s2:generation_time` wins."""
    orig = _mpc_item("02.12", "20201009T023142")
    reproc = _mpc_item("05.00", "20240604T180322")
    for order in ([orig, reproc], [reproc, orig]):
        kept = processing_module.select_processing(_mpc_gdf(order), "latest").kept
        assert list(kept["id"]) == [_canon(reproc)]


# --- AC10: properties_filter on first-class columns -----------------------------------


def test_ac10_properties_filter_matches_source_set_and_processing_version_columns():
    gdf = _mpc_gdf([_mpc_item("05.00", "20230915T000622"), _mpc_item("02.12", "20201009T023142")])
    gdf["source"] = ["cdse,mpc", "mpc"]
    assert list(filter_by_properties(gdf, {"source": "cdse"})["processing_version"]) == ["05.00"]
    assert len(filter_by_properties(gdf, {"source": "mpc"})) == 2
    assert list(filter_by_properties(gdf, {"processing_version": "05.00"})["source"]) == ["cdse,mpc"]
    # reserved names never fall through to a same-named key in `properties`
    gdf["properties"] = ['{"source": "elsewhere"}', '{"source": "elsewhere"}']
    assert len(filter_by_properties(gdf, {"source": "elsewhere"})) == 0
    with pytest.raises(ValueError, match="not carried"):
        filter_by_properties(gdf, {"nope": "x"})


# --- AC11: the catalog's directory must be named for the collection -------------------


def test_ac11_a_catalog_from_another_collection_raises_naming_both(tmp_path):
    polys = gpd.GeoDataFrame({"fid": [1], "geometry": [sg.box(0, 0, 0.01, 0.01)]},
                             crs="EPSG:4326")
    with pytest.raises(api.PreflightError) as exc:
        api.create_training_data(
            label_polygons=polys, catalog_filepath=str(tmp_path / S1 / "catalog.parquet"),
            startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 1),
            mosaic_days=10, bands=["B04"], id_col="fid", export_folderpath=str(tmp_path / "o"),
            collection=S2,
        )
    assert S1 in str(exc.value) and S2 in str(exc.value)


def test_d5_create_training_data_downloads_into_the_catalogs_grandparent(monkeypatch, tmp_path):
    seen = {}

    class _Stop(Exception):
        pass

    def _fake(**kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(api, "_download_verb", _fake)
    polys = gpd.GeoDataFrame({"fid": [1], "geometry": [sg.box(0, 0, 0.01, 0.01)]},
                             crs="EPSG:4326")
    with pytest.raises(_Stop):
        api.create_training_data(
            label_polygons=polys, catalog_filepath=str(tmp_path / "root" / S2 / "catalog.parquet"),
            startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 1),
            mosaic_days=10, bands=["B04"], id_col="fid", export_folderpath=str(tmp_path / "o"),
            collection=S2, download=True, max_tiles=1,
        )
    assert seen["dst_folderpath"] == str(tmp_path / "root")


# --- AC12: stamp, then publish ---------------------------------------------------------


def _write_cog(path, value=1500):
    with rasterio.open(str(path), "w", driver="GTiff", height=2, width=2, count=1,
                       dtype="uint16", crs="EPSG:32633", transform=from_origin(0, 2, 1, 1)) as d:
        d.write(np.full((1, 2, 2), value, dtype="uint16"))


def _tags(path):
    with rasterio.open(str(path)) as s:
        return s.scales[0], s.offsets[0], s.nodata


def test_ac12_mpc_stamp_failure_leaves_no_final_file_and_a_rerun_transfers_again(
        monkeypatch, tmp_path):
    dst = tmp_path / "B04.tif"
    transfers = []
    monkeypatch.setattr(mpc.fs, "transfer",
                        lambda src, d, **kw: (transfers.append(d), _write_cog(d)))
    real = mpc.stamp_or_reencode

    def _boom(*a, **k):
        raise RuntimeError("stamp died")

    monkeypatch.setattr(mpc, "stamp_or_reencode", _boom)
    ok, reason = mpc._transfer_and_stamp_one(
        "https://x/B04.tif", str(dst), band="B04", offset=-1000,
        declaration=S2_L2A_DECLARATION, tries=1)
    assert ok is False and "stamp died" in reason
    assert not dst.exists() and not (tmp_path / "B04.tif.stage").exists()

    monkeypatch.setattr(mpc, "stamp_or_reencode", real)
    ok, reason = mpc._transfer_and_stamp_one(
        "https://x/B04.tif", str(dst), band="B04", offset=-1000,
        declaration=S2_L2A_DECLARATION)
    assert (ok, reason) == (True, "ok") and len(transfers) == 2          # transferred AGAIN
    scale, offset, nodata = _tags(dst)
    assert (scale, round(offset, 4), nodata) == (S2_L2A_DECLARATION.scale, -0.1, config.NODATA)


def test_ac12_a_kill_between_transfer_and_stamp_leaves_no_final_file(monkeypatch, tmp_path):
    dst = tmp_path / "B04.tif"
    monkeypatch.setattr(mpc.fs, "transfer", lambda src, d, **kw: _write_cog(d))

    def _killed(*a, **k):
        raise KeyboardInterrupt          # not an Exception: the retry loop must not eat it

    monkeypatch.setattr(mpc, "stamp_or_reencode", _killed)
    with pytest.raises(KeyboardInterrupt):
        mpc._transfer_and_stamp_one("https://x/B04.tif", str(dst), band="B04", offset=0,
                                    declaration=S2_L2A_DECLARATION)
    assert not dst.exists()


def test_ac12_cdse_stamp_failure_leaves_no_tif_at_all(monkeypatch, tmp_path):
    staging, dst = tmp_path / "B04.tif.src.jp2", tmp_path / "B04.tif"
    _write_cog(staging)
    monkeypatch.setattr(cdse, "stamp_or_reencode",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stamp died")))
    ok, reason, _ = cdse._convert_one(str(staging), str(dst))
    assert (ok, reason) == (False, "ConvertError")
    assert not dst.exists() and not staging.exists() and not (tmp_path / "B04.tif.stage").exists()


def test_ac12_cdse_rerun_after_a_stamp_failure_transfers_again(monkeypatch, tmp_path):
    dst = tmp_path / "B04.tif"
    transfers = []

    def _fake_transfer(src, target, **kw):
        transfers.append(target)
        _write_cog(target)

    monkeypatch.setattr(cdse.fs, "transfer", _fake_transfer)
    real = cdse.stamp_or_reencode
    monkeypatch.setattr(cdse, "stamp_or_reencode",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stamp died")))
    ok, reason, _, _ = cdse._transfer_one(
        "s3://eodata/x/B04.jp2", str(dst), {}, needs_convert=True, tries=1)
    assert ok
    ok, reason, _ = cdse._convert_one(str(dst) + ".src.jp2", str(dst))
    assert (ok, reason) == (False, "ConvertError") and not dst.exists()

    monkeypatch.setattr(cdse, "stamp_or_reencode", real)
    ok, reason, _, _ = cdse._transfer_one(
        "s3://eodata/x/B04.jp2", str(dst), {}, needs_convert=True, tries=1)
    assert ok
    ok, reason, _ = cdse._convert_one(str(dst) + ".src.jp2", str(dst))
    assert (ok, reason) == (True, "ok") and len(transfers) == 2           # transferred AGAIN
    assert _tags(dst)[2] == config.NODATA                                 # and stamped


def test_ac12_cdse_publishes_a_stamped_file(tmp_path):
    staging, dst = tmp_path / "B04.tif.src.jp2", tmp_path / "B04.tif"
    _write_cog(staging)
    ok, _, _ = cdse._convert_one(str(staging), str(dst), offset=-1000,
                                 declaration=S2_L2A_DECLARATION)
    assert ok
    scale, offset, nodata = _tags(dst)
    assert (scale, round(offset, 4), nodata) == (S2_L2A_DECLARATION.scale, -0.1, config.NODATA)


def test_ac12_stamping_a_dot_stage_name_takes_the_inplace_path(monkeypatch, tmp_path):
    from rise_eo.raster import cog

    stage = tmp_path / "B04.tif.stage"        # GDAL identifies GeoTIFF by content, not name
    _write_cog(stage)
    real, calls = cog.stamp_gdal_tags, []
    real_copy, copies = cog.rasterio.shutil.copy, []

    def _spy(*a, **k):
        calls.append(1)
        return real(*a, **k)

    def _copy_spy(*a, **k):
        copies.append(1)
        return real_copy(*a, **k)

    monkeypatch.setattr(cog, "stamp_gdal_tags", _spy)
    monkeypatch.setattr(cog.rasterio.shutil, "copy", _copy_spy)
    cog.stamp_or_reencode(str(stage), offset=-0.1, scale=0.0001, set_nodata_if_missing=0)
    assert len(calls) == 1 and copies == []    # stamped in place, no re-encode
    assert _tags(stage) == (0.0001, -0.1, 0.0)


def test_ac12_the_reencode_fallback_writes_a_cog_without_a_tif_extension(monkeypatch, tmp_path):
    from rise_eo.raster import cog

    stage = tmp_path / "B04.tif.stage"
    _write_cog(stage)
    real, calls = cog.stamp_gdal_tags, []

    def _first_fails(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("in-place stamp would break COG validity")
        return real(*a, **k)

    monkeypatch.setattr(cog, "stamp_gdal_tags", _first_fails)
    cog.stamp_or_reencode(str(stage), offset=-0.1, scale=0.0001, set_nodata_if_missing=0)
    assert len(calls) == 2                     # in-place stamp failed; re-encoded, then stamped
    assert _tags(stage) == (0.0001, -0.1, 0.0)
    with rasterio.open(str(stage)) as s:
        assert s.driver == "GTiff" and s.profile.get("compress", "").lower() == "deflate"
    assert not (tmp_path / "B04.tif.stage.reencode.part").exists()


# --- AC13: pre-59 catalogs are refused (also pinned in test_catalog.py) -----------------


def test_ac13_a_catalog_without_the_acquisition_columns_raises_naming_them(tmp_path):
    gdf = gpd.GeoDataFrame({"id": ["x"], "timestamp": [pd.Timestamp("2018-01-01", tz="UTC")],
                            "geometry": [sg.box(0, 0, 1, 1)]}, crs="EPSG:4326")
    fp = str(tmp_path / "old.parquet")
    fs.write_parquet(fp, gdf)
    with pytest.raises(ValueError, match=r"acquisition_key.*processing_version.*re-download"):
        TileCatalog(fp).read()


# --- AC14: provenance ---------------------------------------------------------------------


def test_ac14_a_cube_records_ids_versions_and_sources():
    flat = pd.DataFrame({
        "id": ["a", "a", "b"], "collection": [S2] * 3,
        "processing_version": ["02.12", "02.12", "05.00"], "source": ["mpc", "mpc", "cdse,mpc"],
    })
    assert builder._cube_provenance(flat) == {S2: {
        "ids": ["a", "b"], "processing_version": ["02.12", "05.00"], "source": ["cdse", "mpc"]}}


def test_ac14_flatten_unions_cube_provenance_without_ids_into_the_stamp(tmp_path):
    ts = [pd.Timestamp("2018-06-01", tz="UTC")]
    rows = []
    for name, prov in (("A", {S2: {"ids": ["a"], "processing_version": ["02.12"],
                                   "source": ["mpc"]}}),
                       ("B", {S2: {"ids": ["b"], "processing_version": ["05.00"],
                                   "source": ["cdse"]}})):
        folder = tmp_path / name
        folder.mkdir()
        fs.save_npy(str(folder / "datacube.npy"), np.ones((1, 2, 2, 1), dtype=np.uint16))
        fs.save_npy(str(folder / "metadata.pickle.npy"), {
            "bands": ["B04"], "timestamps": ts, "provenance": prov,
            "geotiff_metadata": {"width": 2, "height": 2, "crs": "EPSG:32633",
                                 "transform": from_origin(500000, 5000000, 10, 10)}},
            allow_pickle=True)
        rows.append({"id": name, "datacube_filepath": str(folder / "datacube.npy")})
    csv = tmp_path / "input.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    api.flatten_training_data(str(csv), str(tmp_path / "flat"), id_col="id",
                              filepath_col="datacube_filepath")

    stamp = _stamp.read_stamp(str(tmp_path / "flat" / api._FLATTEN_STAMP_NAME))
    assert stamp["provenance"] == {S2: {"processing_version": ["02.12", "05.00"],
                                        "source": ["cdse", "mpc"]}}
    identity = {k: v for k, v in stamp.items() if k not in ("_written_at", "provenance")}
    assert _stamp.matches_stamp(str(tmp_path / "flat" / api._FLATTEN_STAMP_NAME), identity)


# --- AC15 / AC16 / AC17: download(processing=) on all four paths -------------------------

A_OLD, A_NEW = _mpc_item("02.12", "20201009T023142"), _mpc_item("05.00", "20230915T000622")
B_ONLY = _mpc_item("02.12", "20201019T000000", dt="2018-09-28T10:00:19Z")


def _mpc_download(monkeypatch, tmp_path, items, **kw):
    monkeypatch.setattr(mpc, "_search_items_unsigned", lambda *a, **k: items)
    monkeypatch.setattr(mpc, "_import_pc_sign", lambda: (lambda u: u))
    fetched = []
    def _fake_transfer(src, d, **k):
        fetched.append(src)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        _write_cog(d)

    monkeypatch.setattr(mpc.fs, "transfer", _fake_transfer)
    catalog = api.download(
        _roi(), datetime.datetime(2018, 9, 1), datetime.datetime(2018, 10, 1), ["B04"],
        str(tmp_path / "archive"), source="mpc", **kw)
    return catalog, fetched


def test_ac16_a_specifier_fetches_a_at_0500_skips_b_and_says_so(monkeypatch, tmp_path, capsys):
    catalog, fetched = _mpc_download(monkeypatch, tmp_path, [A_OLD, A_NEW, B_ONLY],
                                     processing=">=05.00", max_tiles=10)
    assert fetched == [f"https://mpc/{A_NEW.id}/B04.tif"]
    assert catalog == str(tmp_path / "archive" / S2 / "catalog.parquet")   # D5
    assert catalog == rise_eo.archive_catalog_filepath(str(tmp_path / "archive"), S2)   # A1 / AC22
    assert list(TileCatalog(catalog).read()["id"]) == [_canon(A_NEW)]
    out = capsys.readouterr().out
    assert "dropped acquisition S2B_MSIL2A_20180928T100019_R122_T33UWP" in out
    assert "skipped" in out and "does not satisfy" in out


def test_ac16_max_tiles_counts_after_the_selection(monkeypatch, tmp_path):
    _mpc_download(monkeypatch, tmp_path, [A_OLD, A_NEW, B_ONLY], max_tiles=2)   # 3 found, 2 kept
    with pytest.raises(ValueError, match="exceed max_tiles"):
        _mpc_download(monkeypatch, tmp_path / "again", [A_OLD, A_NEW, B_ONLY],
                      max_tiles=1)


def test_ac16_processing_none_is_refused_naming_latest(tmp_path):
    with pytest.raises(api.PreflightError, match="latest"):
        api.download(_roi(), datetime.datetime(2018, 9, 1), datetime.datetime(2018, 10, 1),
                     ["B04"], str(tmp_path), processing=None, max_tiles=1)
    with pytest.raises(ValueError, match="latest"):      # and the source-level guard
        mpc.discover_shard_rows(_roi(), datetime.datetime(2018, 9, 1),
                                datetime.datetime(2018, 10, 1), ["B04"], str(tmp_path),
                                processing=None)


def test_ac17_the_report_names_versions_and_sources_only_when_ambiguity_was_created(
        monkeypatch, tmp_path, capsys):
    root = tmp_path
    # 1) first download of the old processing: nothing ambiguous yet
    _mpc_download(monkeypatch, root, [A_OLD], max_tiles=10)
    out = capsys.readouterr().out
    assert "[download] 1 acquisitions; 1 matched processing='latest' at mpc" in out
    assert "archive now holds" not in out
    # 2) MPC reprocesses; the second download lands the new one beside the old
    _mpc_download(monkeypatch, root, [A_NEW], max_tiles=10)
    out = capsys.readouterr().out
    assert "archive now holds >1 processing for 1 of them" in out
    assert "mpc 02.12" in out and "mpc 05.00" in out
    assert 'processing=' in out


def test_ac15_mpc_local_reaches_the_selection_via_the_verb(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(api, "_mpc_download", lambda **kw: seen.update(kw))
    api.download(_roi(), datetime.datetime(2018, 9, 1), datetime.datetime(2018, 10, 1), ["B04"],
                 str(tmp_path), source="mpc", processing="==05.00", max_tiles=1)
    assert seen["processing"] == "==05.00"


def test_ac15_cdse_local_reaches_the_selection_via_the_verb(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(api, "_cdse_download", lambda **kw: seen.update(kw))
    api.download(_roi(), datetime.datetime(2018, 9, 1), datetime.datetime(2018, 10, 1), ["B04"],
                 str(tmp_path), source="cdse", creds=object(), processing=">=05.00", max_tiles=1)
    assert seen["processing"] == ">=05.00"


def test_ac15_cdse_download_applies_it_before_the_cap(monkeypatch, tmp_path):
    old, new = _cdse_item("04.00", "20220301T000000"), _cdse_item("05.00", "20230915T000622")
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: [old, new])
    monkeypatch.setattr(cdse.fs, "transfer",
                        lambda src, d, **k: (os.makedirs(os.path.dirname(d), exist_ok=True),
                                             open(d, "wb").write(b"x")))
    cat = TileCatalog(str(tmp_path / S2 / "catalog.parquet"))
    creds = cdse.CdseCredentials(sh_client_id="i", sh_client_secret="s", s3_access_key="a",
                                 s3_secret_key="k")
    cdse.download(_roi(), datetime.datetime(2018, 9, 1), datetime.datetime(2018, 10, 1), ["B04"],
                  str(tmp_path), cat, creds, max_tiles=1, cog=False)     # 2 found, 1 kept
    assert list(cat.read()["id"]) == [_canon(new)]
    with pytest.raises(ValueError, match="latest"):
        cdse.download(_roi(), datetime.datetime(2018, 9, 1), datetime.datetime(2018, 10, 1),
                      ["B04"], str(tmp_path), cat, creds, max_tiles=1, processing=None)


def test_ac15_mpc_aml_driver_side_discovery_selects_and_the_verb_forwards(monkeypatch, tmp_path):
    monkeypatch.setattr(mpc, "_search_items_unsigned", lambda *a, **k: [A_OLD, A_NEW, B_ONLY])
    rows = mpc.discover_shard_rows(
        _roi(), datetime.datetime(2018, 9, 1), datetime.datetime(2018, 10, 1), ["B04"],
        str(tmp_path), processing=">=05.00")
    assert {r["tile_id"] for r in rows} == {_canon(A_NEW)}
    assert {r["processing_version"] for r in rows} == {"05.00"}

    seen = {}
    real_run_aml_download = runners.run_aml_download
    monkeypatch.setattr(runners, "run_aml_download", lambda **kw: seen.update(kw))
    api.download(_roi(), datetime.datetime(2018, 9, 1), datetime.datetime(2018, 10, 1), ["B04"],
                 str(tmp_path), source="mpc", processing=">=05.00", max_tiles=1, runner="aml",
                 runner_kwargs={"cluster": "c", "environment": "e:1", "root": "memory://r",
                                "identity_client_id": "i"})
    assert seen["processing"] == ">=05.00"

    got = {}
    monkeypatch.setattr(runners._mpc, "discover_shard_rows",
                        lambda *a, **kw: (got.update(kw), [])[1])
    monkeypatch.setattr(runners, "_import_aml_command", lambda: None)
    monkeypatch.setattr(runners, "_import_command_job_limits", lambda: None)
    monkeypatch.setattr(runners, "_mpc_catalog_shortfall", lambda cat, rows: rows)
    real_run_aml_download(
        "memory://roi.geojson", "2018-09-01", "2018-10-01", ["B04"], "memory://p/data",
        "memory://p/data/sentinel-2-l2a/catalog.parquet", source="mpc", cluster="c",
        environment="e:1", root="memory://p/root", identity_client_id="i", max_tiles=10,
        ml_client=object(), processing=">=05.00")
    assert got["processing"] == ">=05.00"


def test_mpc_aml_shard_csv_round_trip_keeps_processing_version_a_string(monkeypatch, tmp_path):
    """Review finding: the shard CSV round trip read "05.00" back as the float 5.0, so an AML
    download into an archive a local download already wrote failed the parquet append, and
    `properties_filter={"processing_version": "05.00"}` never matched (D8: a normalized string)."""
    from rise_eo.workflows import download as download_workflow

    root = tmp_path / "archive"
    _mpc_download(monkeypatch, tmp_path, [A_OLD], max_tiles=10)     # local: "02.12", a string
    monkeypatch.setattr(mpc, "_search_items_unsigned", lambda *a, **k: [A_NEW])
    rows = mpc.discover_shard_rows(
        _roi(), datetime.datetime(2018, 9, 1), datetime.datetime(2018, 10, 1), ["B04"],
        str(root))
    assert set(rows[0]) == set(mpc._SHARD_ROW_KEYS)       # spec 37 A1.1: producer == check
    shard_url = str(tmp_path / "shards" / "0.csv")
    with fs.open(shard_url, "w") as f:                          # as runners writes it
        pd.DataFrame(rows).to_csv(f, index=False)

    catalog = str(root / S2 / "catalog.parquet")
    download_workflow.run_shard(shard_url=shard_url, dst=str(root), catalog=catalog,
                                status_url=str(tmp_path / "_status" / "0.json"))

    gdf = TileCatalog(catalog).read()
    assert sorted(gdf["processing_version"]) == ["02.12", "05.00"]
    assert len(filter_by_properties(gdf, {"processing_version": "05.00"})) == 1


def test_ac15_cdse_aml_carries_the_specifier_on_the_node_command_line(monkeypatch):
    got = {}
    monkeypatch.setattr(runners, "_cdse_query_catalog",
                        lambda *a, **kw: (got.update(kw), pd.DataFrame(
                            {"id": ["t"], "acquisition_key": ["a"]}))[1])
    submitted = []
    monkeypatch.setattr(runners, "_import_aml_command",
                        lambda: (lambda **kw: submitted.append(kw) or types.SimpleNamespace(**kw)))
    monkeypatch.setattr(runners, "_import_command_job_limits",
                        lambda: (lambda timeout: types.SimpleNamespace(timeout=timeout)))
    monkeypatch.setattr(runners, "_aml_download_preflight", lambda *a, **k: None)
    monkeypatch.setattr(runners, "_aml_submit_and_wait",
                        lambda *a, **k: {"job_statuses": {}, "reports": {}})
    runners.run_aml_download(
        "memory://roi.geojson", "2018-09-01", "2018-10-01", ["B04"], "memory://p/data",
        "memory://p/data/sentinel-2-l2a/catalog.parquet", source="cdse", cluster="c",
        environment="e:1", root="memory://p/root", identity_client_id="i", max_tiles=10,
        vault_url="v", secret_name="s", ml_client=object(), processing=">=05.00")
    assert got["processing"] == ">=05.00"                 # the driver-side count uses it too
    assert "--processing '>=05.00'" in submitted[0]["command"]   # quoted: `>` is a redirect


def test_ac15_the_cdse_job_entrypoint_forwards_it(monkeypatch):
    from rise_eo.workflows import download as dl

    seen = {}
    monkeypatch.setattr(dl.cdse, "download", lambda *a, **kw: (seen.update(kw), cdse.DownloadResult(
        successful_count=0, total_count=0))[1])
    monkeypatch.setattr(dl.cdse.CdseCredentials, "from_json", classmethod(lambda cls, u: object()))
    dl.run_roi(roi="memory://r", startdate="2018-09-01", enddate="2018-10-01", bands=["B04"],
               dst="memory://d", catalog="memory://d/c.parquet", max_tiles=1,
               status_url="memory://s/0.json", creds_url="memory://creds", processing="==05.00")
    assert seen["processing"] == "==05.00"
    assert dl._parse_args(["--roi", "r", "--dst", "d", "--catalog", "c", "--status-url", "s",
                           "--processing", ">=05.00"]).processing == ">=05.00"


def test_ac15_create_training_data_forwards_a_specifier_and_latest_for_none(
        monkeypatch, tmp_path):
    seen = []

    class _Stop(Exception):
        pass

    def _fake(**kw):
        seen.append(kw["processing"])
        raise _Stop

    monkeypatch.setattr(api, "_download_verb", _fake)
    polys = gpd.GeoDataFrame({"fid": [1], "geometry": [sg.box(0, 0, 0.01, 0.01)]},
                             crs="EPSG:4326")
    for processing in (">=05.00", "latest", None):
        with pytest.raises(_Stop):
            api.create_training_data(
                label_polygons=polys,
                catalog_filepath=str(tmp_path / "root" / S2 / "catalog.parquet"),
                startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 1),
                mosaic_days=10, bands=["B04"], id_col="fid",
                export_folderpath=str(tmp_path / "o"), download=True, max_tiles=1,
                processing=processing)
    assert seen == [">=05.00", "latest", "latest"]


# --- D8: the STAC export carries processing:version / processing:datetime -----------------


def test_d8_stac_export_writes_the_processing_extension_fields(tmp_path):
    from rise_eo.catalog import stac

    folder = tmp_path / "g"
    folder.mkdir()
    _write_cog(folder / "B04.tif")
    row = _mpc_gdf([_mpc_item("05.00", "20230915T000622")]).iloc[0].to_dict()
    row.update(local_folderpath=str(folder), files="B04.tif")
    cat = TileCatalog(str(tmp_path / S2 / "catalog.parquet"))
    cat.append([row], declaration=S2_L2A_DECLARATION)
    items = stac.tile_catalog_to_items(cat.read(), declaration=S2_L2A_DECLARATION)
    props = items[0].properties
    assert props["processing:version"] == "05.00"
    assert props["processing:datetime"] == "2023-09-15T00:06:22Z"
    assert any("processing" in e for e in items[0].stac_extensions)


# --- Amendment A1: the archive's catalog path has one owner ---------------------

def test_ac22_archive_catalog_filepath_is_pure_and_exported():
    assert "archive_catalog_filepath" in rise_eo.__all__
    assert rise_eo.archive_catalog_filepath("/data/imagery", S2) == f"/data/imagery/{S2}/catalog.parquet"
    assert (rise_eo.archive_catalog_filepath("abfss://fs@acct.dfs.core.windows.net/imagery", S2)
            == f"abfss://fs@acct.dfs.core.windows.net/imagery/{S2}/catalog.parquet")


def test_ac23_nothing_outside_src_rebuilds_the_catalog_path_by_hand():
    import json
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # A catalog.parquet path assembled from a collection: `{...}/{COLLECTION}/catalog.parquet`,
    # a literal `sentinel-…/catalog.parquet`, or `os.path.join(…, "catalog.parquet")` /
    # `… / "catalog.parquet"` after a collection-looking argument.
    coll = r"(?:collection|COLLECTION|SATELLITE_\w+)"
    by_hand = re.compile(
        rf"""\{{[^}}]*{coll}[^}}]*\}}/catalog\.parquet"""
        r"""|sentinel-[\w-]+/catalog\.parquet"""
        rf"""|{coll}\s*[,/]\s*["']catalog\.parquet["']""")

    def code(text):  # full-line comments are prose (AC 23 exempts explanation)
        return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))

    offenders = []
    for py in (root / "demos").rglob("*.py"):
        if by_hand.search(code(py.read_text())):
            offenders.append(str(py.relative_to(root)))
    for nb in (root / "notebooks").glob("*.ipynb"):
        for cell in json.loads(nb.read_text())["cells"]:
            if cell["cell_type"] == "code" and by_hand.search(code("".join(cell["source"]))):
                offenders.append(str(nb.relative_to(root)))
    assert not offenders, f"hand-built catalog path (use rise_eo.archive_catalog_filepath): {offenders}"
