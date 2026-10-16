"""Tests for rise_eo.sources.cdse (spec 01).

Credential loading only (no network). Uses DUMMY values — never the real
secrets/cdse_credentials.json.
"""

import concurrent.futures
import concurrent.futures.process
import datetime
import json
import threading
import time
import types

import geopandas as gpd
import shapely.geometry as sg

from rise_eo import collections as _collections
from rise_eo import config
from rise_eo.sources import cdse
from rise_eo.sources.cdse import CdseCredentials


def _esa_name(id, dt, baseline):
    """A valid, per-id-unique ESA product name for a fake item (spec 59 D3 needs one)."""
    import hashlib

    h = int(hashlib.sha1(id.encode()).hexdigest(), 16)
    disc = f"20230101T{h % 10**6:06d}"
    # distinct tile per id, so two fakes at one datetime are two acquisitions
    tile = "T36" + "".join(chr(65 + (h // 26**i) % 26) for i in range(3))
    return (f"S2A_MSIL2A_{dt:%Y%m%dT%H%M%S}_N{(baseline or '05.09').replace('.', '')}"
            f"_R035_{tile}_{disc}")


def _canon(item):
    """The canonical granule name (spec 59 D3) of a fake item -- what the catalog `id` is."""
    return item.properties["s2:product_uri"].removesuffix(".SAFE")


class _FakeItem:
    """Duck-typed stand-in for a pystac `Item` (no network)."""

    def __init__(self, id, dt, geom, cloud, assets, baseline="05.09"):
        self.id = id
        self.datetime = dt
        self.geometry = sg.mapping(geom)
        self.properties = {"eo:cloud_cover": cloud,
                           "s2:product_uri": _esa_name(id, dt, baseline) + ".SAFE"}
        if baseline is not None:
            self.properties["s2:processing_baseline"] = baseline
        self.assets = {k: types.SimpleNamespace(href=v) for k, v in assets.items()}


def _fake_item(id, dt, lon, lat, cloud, safe=None, assets=None, baseline="05.09"):
    """A STAC item over box (lon,lat)-(lon+1,lat+1); default asset is one B02 href.

    `baseline` (spec 34 §1, closes #30/#10) defaults to a real >=04.00 value so
    every pre-existing call site (most of this file) exercises `_items_to_gdf`'s
    offset derivation without needing to know about it; pass `baseline=None` to
    test the "missing property" fork.
    """
    safe = safe or f"s3://eodata/{id}.SAFE"
    if assets is None:
        assets = {"B02_10m": f"{safe}/GRANULE/G/IMG_DATA/R10m/T_D_B02_10m.jp2"}
    dt = datetime.datetime.fromisoformat(dt.replace("Z", "+00:00"))
    return _FakeItem(id, dt, sg.box(lon, lat, lon + 1, lat + 1), cloud, assets, baseline=baseline)

# DUMMY = legacy JSON-key form; DUMMY_FIELDS = the dataclass field form of the same.
DUMMY = {
    "sh_clientid": "id-123",
    "sh_clientsecret": "secret-abc",
    "s3_access_key": "akia-xyz",
    "s3_secret_key": "s3secret-789",
}
DUMMY_FIELDS = {
    "sh_client_id": "id-123",
    "sh_client_secret": "secret-abc",
    "s3_access_key": "akia-xyz",
    "s3_secret_key": "s3secret-789",
}


def test_from_json_reads_legacy_keys(tmp_path):
    p = tmp_path / "cdse_credentials.json"
    p.write_text(json.dumps(DUMMY))
    creds = CdseCredentials.from_json(str(p))
    assert creds.sh_client_id == "id-123"
    assert creds.sh_client_secret == "secret-abc"
    assert creds.s3_access_key == "akia-xyz"
    assert creds.s3_secret_key == "s3secret-789"


def test_from_json_tolerates_extra_and_optional_keys(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({**DUMMY, "s3_keys_expire": "2026-08-01",
                             "note": "rotate soon", "unknown": "ignored"}))
    creds = CdseCredentials.from_json(str(p))
    assert creds.s3_keys_expire == "2026-08-01"


def test_from_env():
    env = {
        "CDSE_SH_CLIENT_ID": "id-123",
        "CDSE_SH_CLIENT_SECRET": "secret-abc",
        "CDSE_S3_ACCESS_KEY": "akia-xyz",
        "CDSE_S3_SECRET_KEY": "s3secret-789",
        "CDSE_S3_KEYS_EXPIRE": "2026-08-01",
    }
    creds = CdseCredentials.from_env(env)
    assert creds.s3_access_key == "akia-xyz"
    assert creds.s3_keys_expire == "2026-08-01"


def test_repr_masks_secrets():
    creds = CdseCredentials(**DUMMY_FIELDS)
    r = repr(creds)
    # no secret value leaks into the repr
    for v in ("id-123", "secret-abc", "akia-xyz", "s3secret-789"):
        assert v not in r
    assert "set" in r  # shows presence, not value


def test_s3_storage_options_shape():
    creds = CdseCredentials(**DUMMY_FIELDS)
    opts = creds.s3_storage_options()
    assert opts["key"] == "akia-xyz"
    assert opts["secret"] == "s3secret-789"
    assert "endpoint_url" in opts["client_kwargs"]


def test_is_expired():
    assert CdseCredentials().is_expired() is None  # unknown
    past = CdseCredentials(s3_keys_expire="2020-01-01")
    future = CdseCredentials(s3_keys_expire="2999-01-01")
    ref = datetime.date(2026, 7, 1)
    assert past.is_expired(as_of=ref) is True
    assert future.is_expired(as_of=ref) is False


# --- query_catalog pure helpers (no network) ---------------------------------


def test_items_to_gdf_parses_stac():
    items = [
        _fake_item("t1", "2018-06-30T09:57:22Z", 16.0, 48.0, 12.5),
        _fake_item("t2", "2018-07-05T09:57:22Z", 17.0, 48.0, 80.0),
    ]
    gdf = cdse._items_to_gdf(items, collection=config.SATELLITE_S2L2A,
                             declaration=_collections.get(config.SATELLITE_S2L2A))
    assert list(gdf["id"]) == [_canon(i) for i in items]
    assert list(gdf.columns) == ["id", "collection", "timestamp", "s3url",
                                 "cloud_cover", "offset", "scale", "nodata",
                                 "acquisition_key", "processing_version",
                                 "processing_datetime", "source", "properties", "geometry"]
    assert list(gdf["source"]) == ["cdse", "cdse"]
    assert list(gdf["processing_version"]) == ["05.09", "05.09"]
    assert gdf.crs.to_epsg() == 4326
    assert str(gdf["timestamp"].dt.tz) == "UTC"
    assert gdf["s3url"].iloc[0].startswith("s3://eodata/")
    assert gdf["s3url"].iloc[0].endswith(".SAFE")  # derived from an asset href
    assert gdf["collection"].iloc[0] == "sentinel-2-l2a"


def test_items_to_gdf_derives_offset_from_baseline_closes_30_10():
    """spec 34 §1, closes #30/#10: CDSE rows no longer hardcode offset=0 — the
    per-item processing baseline (the same STAC property MPC reads) decides it."""
    items = [
        _fake_item("pre", "2021-06-01T00:00:00Z", 0.0, 0.0, 5.0, baseline="02.14"),
        _fake_item("post", "2022-06-01T00:00:00Z", 0.0, 0.0, 5.0, baseline="04.00"),
    ]
    gdf = cdse._items_to_gdf(items, collection=config.SATELLITE_S2L2A,
                             declaration=_collections.get(config.SATELLITE_S2L2A))
    assert list(gdf["offset"]) == [0, -1000]
    assert list(gdf["nodata"]) == [0, 0]


def test_safe_root_from_item_derives_from_asset_href():
    it = _fake_item(
        "x", "2018-06-30T00:00:00Z", 0, 0, 1.0,
        safe="s3://eodata/Sentinel-2/MSI/L2A_N0500/2018/01/30/S2A_X.SAFE",
    )
    assert cdse._safe_root_from_item(it) == (
        "s3://eodata/Sentinel-2/MSI/L2A_N0500/2018/01/30/S2A_X.SAFE"
    )


def test_finalize_filters_cloud_and_roi():
    items = [
        _fake_item("hit", "2018-06-30T00:00:00Z", 0.0, 0.0, 10.0),   # overlaps, clear
        _fake_item("cloudy", "2018-06-30T00:00:00Z", 0.0, 0.0, 90.0),  # overlaps, cloudy
        _fake_item("far", "2018-06-30T00:00:00Z", 50.0, 50.0, 5.0),  # no overlap
    ]
    gdf = cdse._items_to_gdf(items, collection=config.SATELLITE_S2L2A,
                             declaration=_collections.get(config.SATELLITE_S2L2A))
    roi = gpd.GeoDataFrame(geometry=[sg.box(0.2, 0.2, 0.5, 0.5)], crs="EPSG:4326")
    out = cdse._finalize_catalog_gdf(gdf, roi, max_cloudcover=50.0)
    assert list(out["id"]) == [_canon(items[0])]


def test_finalize_raises_on_duplicate_ids():
    items = [
        _fake_item("dup", "2018-06-30T00:00:00Z", 0.0, 0.0, 10.0),
        _fake_item("dup", "2018-06-30T00:00:00Z", 0.0, 0.0, 10.0),
    ]
    gdf = cdse._items_to_gdf(items, collection=config.SATELLITE_S2L2A,
                             declaration=_collections.get(config.SATELLITE_S2L2A))
    roi = gpd.GeoDataFrame(geometry=[sg.box(0.2, 0.2, 0.5, 0.5)], crs="EPSG:4326")
    import pytest

    with pytest.raises(ValueError):
        cdse._finalize_catalog_gdf(gdf, roi, max_cloudcover=None)


def test_require_s3():
    import pytest

    CdseCredentials(**DUMMY_FIELDS).require_s3()  # no raise
    # SH-only creds are fine for discovery but must fail the download S3 check
    with pytest.raises(ValueError):
        CdseCredentials(sh_client_id="only-sh").require_s3()


# --- download helpers --------------------------------------------------------

SAFE = ("s3://eodata/Sentinel-2/MSI/L2A_N0500/2018/01/30/"
        "S2A_MSIL2A_20180130T080151_N0500_R035_T36PZT_20230915T000622.SAFE")


def test_select_item_files_picks_highest_res_and_xml():
    granule = f"{SAFE}/GRANULE/L2A_T36PZT_A013_20180130T080151"
    it = _fake_item(
        "S2A_T36PZT", "2018-01-30T08:00:00Z", 36.0, 11.0, 5.0, safe=SAFE,
        assets={
            # B02 only at 10m; B05 at both 20m and 60m -> keep 20m
            "B02_10m": f"{granule}/IMG_DATA/R10m/T36PZT_20180130T080151_B02_10m.jp2",
            "B05_20m": f"{granule}/IMG_DATA/R20m/T36PZT_20180130T080151_B05_20m.jp2",
            "B05_60m": f"{granule}/IMG_DATA/R60m/T36PZT_20180130T080151_B05_60m.jp2",
            # a band we didn't request
            "B03_10m": f"{granule}/IMG_DATA/R10m/T36PZT_20180130T080151_B03_10m.jp2",
            "granule_metadata": f"{granule}/MTD_TL.xml",
        },
    )

    selected = cdse._select_item_files(it, ["B02", "B05"], "/root", cog=False)
    dsts = {dst for _, dst in selected}
    # spec 59 D2/D3: the EODATA mirror is retired; same layout as MPC
    folder = f"/root/sentinel-2-l2a/2018/01/30/{_canon(it)}"
    assert dsts == {f"{folder}/B02.jp2", f"{folder}/B05.jp2", f"{folder}/MTD_TL.xml"}
    # highest-res B05 is the 20m source, not 60m
    b05_src = next(src for src, dst in selected if dst.endswith("B05.jp2"))
    assert "R20m" in b05_src and b05_src.startswith("s3://")


def test_select_item_files_cog_uses_tif_dst_but_jp2_src():
    """cog=True (default): band dst is Bxx.tif (converted on arrival), but the source
    href is still the .jp2 asset; the sidecar is unchanged."""
    granule = f"{SAFE}/GRANULE/L2A_T36PZT_A013_20180130T080151"
    it = _fake_item(
        "S2A_T36PZT", "2018-01-30T08:00:00Z", 36.0, 11.0, 5.0, safe=SAFE,
        assets={
            "B02_10m": f"{granule}/IMG_DATA/R10m/T36PZT_20180130T080151_B02_10m.jp2",
            "granule_metadata": f"{granule}/MTD_TL.xml",
        },
    )
    selected = cdse._select_item_files(it, ["B02"], "/root")  # cog defaults True
    by_dst = {dst.rsplit("/", 1)[1]: src for src, dst in selected}
    assert set(by_dst) == {"B02.tif", "MTD_TL.xml"}
    assert by_dst["B02.tif"].endswith("B02_10m.jp2")  # source stays JP2
    assert by_dst["B02.tif"].startswith("s3://")


def test_error_reason_maps_known_codes():
    assert cdse._error_reason(Exception("boom SignatureDoesNotMatch x")) == \
        "SignatureDoesNotMatch"
    assert cdse._error_reason(PermissionError("Forbidden")) == "Forbidden"
    assert cdse._error_reason(ValueError("weird")) == "ValueError"


def test_transfer_one_skips_and_reports_reason(monkeypatch, tmp_path):
    import os

    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))
    existing = tmp_path / "x.jp2"
    existing.write_bytes(b"data")
    assert cdse._transfer_one(
        "s3://eodata/a.jp2", str(existing), {}, needs_convert=False
    )[:2] == (True, "skipped")

    def boom(src, dst, **kw):
        raise PermissionError("An error occurred (Forbidden) ...")

    monkeypatch.setattr(cdse.fs, "transfer", boom)
    ok, reason, _t_s, _nbytes = cdse._transfer_one(
        "s3://eodata/b.jp2", str(tmp_path / "nope.jp2"), {}, needs_convert=False, tries=1
    )
    assert (ok, reason) == (False, "Forbidden")
    assert "Forbidden" in cdse._RETRYABLE_S3  # 403 is transient on CDSE (BUG-001)


def test_transfer_one_redownloads_zero_byte_file(monkeypatch, tmp_path):
    """A 0-byte 'touched' leftover must NOT be treated as done — it re-downloads."""
    import os

    dst = tmp_path / "z.jp2"
    dst.write_bytes(b"")  # 0-byte leftover from a prior failed transfer
    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))
    monkeypatch.setattr(cdse.fs, "size", lambda p, **k: os.path.getsize(p))
    calls = []

    def good_transfer(src, d, **kw):
        calls.append(d)
        with open(d, "wb") as f:
            f.write(b"realbytes")

    monkeypatch.setattr(cdse.fs, "transfer", good_transfer)
    assert cdse._transfer_one(
        "s3://eodata/z.jp2", str(dst), {}, needs_convert=False
    )[:2] == (True, "ok")
    assert calls == [str(dst)]  # actually re-downloaded, not skipped


def test_fmt_progress_line():
    s = cdse._fmt_progress(50, 200, ok_n=45, fail_n=5, skipped=10, elapsed_s=100.0)
    for token in ("50/200", "25%", "ok=45", "fail=5", "skip=10", "file/s", "ETA"):
        assert token in s
    assert "\r" not in s  # newline-terminated line, not a carriage-return bar


def test_download_raises_when_over_max_tiles(monkeypatch, tmp_path):
    import pytest

    from rise_eo.catalog.catalog import TileCatalog

    # 3 fake items overlapping the ROI, max_tiles=2 -> raise before any download
    items = [_fake_item(i, "2018-01-01T00:00:00Z", 0.0, 0.0, 0.0) for i in "abc"]
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: items)
    roi = gpd.GeoDataFrame(geometry=[sg.box(0.2, 0.2, 0.5, 0.5)], crs="EPSG:4326")
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    with pytest.raises(ValueError, match="exceed max_tiles"):
        cdse.download(roi, datetime.datetime(2018, 1, 1),
                      datetime.datetime(2018, 2, 1), ["B02"], str(tmp_path),
                      cat, CdseCredentials(**DUMMY_FIELDS), max_tiles=2)


def test_download_end_to_end_mocked(monkeypatch, tmp_path):
    """Full download flow with storage mocked — no network, real catalog write."""
    from rise_eo.catalog.catalog import TileCatalog

    item = _fake_item("S2A_T36PZT", "2018-01-30T08:00:00Z", 36.0, 11.0, 5.0, safe=SAFE)
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: [item])
    monkeypatch.setattr(
        cdse, "_select_item_files",
        lambda it, bands, root, cog=True, **kw: [
            (f"{cdse._safe_root_from_item(it)}/B02.jp2", str(tmp_path / "tile/B02.jp2")),
            (f"{cdse._safe_root_from_item(it)}/SCL.jp2", str(tmp_path / "tile/SCL.jp2")),
        ],
    )
    # "transfer" just creates the destination file
    def fake_transfer(src, dst, **kw):
        import os
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").close()
    monkeypatch.setattr(cdse.fs, "transfer", fake_transfer)
    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: __import__("os").path.exists(p))

    roi = gpd.GeoDataFrame(geometry=[sg.box(36.0, 11.0, 37.0, 12.0)], crs="EPSG:4326")
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    result = cdse.download(roi, datetime.datetime(2018, 1, 1),
                           datetime.datetime(2018, 2, 1), ["B02", "SCL"],
                           str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
                           max_tiles=10, cog=False)
    assert (result.successful_count, result.total_count) == (2, 2)
    assert (result.failed_count, result.skipped_count) == (0, 0)
    assert result.reason_counts == {"ok": 2}
    gdf = cat.read()
    assert len(gdf) == 1
    assert gdf["files"].iloc[0] == "B02.jp2,SCL.jp2"  # unioned + sorted
    assert result.circuit_tripped is False


def test_download_creates_missing_local_root(monkeypatch, tmp_path):
    """download() creates its local output root if absent (regression: a fresh --dst
    previously FileNotFoundError'd on the disk-usage probe / first write, because
    nothing created the root). The fix runs for any local root, so cog=False exercises
    the same code path as the reported cog=True disk-probe crash."""
    import os

    from rise_eo.catalog.catalog import TileCatalog

    item = _fake_item("S2A_T36PZT", "2018-01-30T08:00:00Z", 36.0, 11.0, 5.0, safe=SAFE)
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: [item])
    monkeypatch.setattr(
        cdse, "_select_item_files",
        lambda it, bands, root, cog=True, **kw: [
            (f"{cdse._safe_root_from_item(it)}/B02.jp2", os.path.join(root, "tile/B02.jp2")),
        ],
    )

    def fake_transfer(src, dst, **kw):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").close()
    monkeypatch.setattr(cdse.fs, "transfer", fake_transfer)
    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))

    root = tmp_path / "fresh_root"          # does NOT exist yet
    assert not root.exists()
    roi = gpd.GeoDataFrame(geometry=[sg.box(36.0, 11.0, 37.0, 12.0)], crs="EPSG:4326")
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    result = cdse.download(roi, datetime.datetime(2018, 1, 1),
                           datetime.datetime(2018, 2, 1), ["B02"],
                           str(root), cat, CdseCredentials(**DUMMY_FIELDS),
                           max_tiles=10, cog=False)
    assert root.exists()                    # download() created it
    assert result.successful_count == 1


def test_download_accumulates_timing_bytes_and_by_band(monkeypatch, tmp_path):
    """spec 23 SO-1/D11: download() records transfer/convert seconds, bytes, and per-band bytes."""
    import os

    from rise_eo.catalog.catalog import TileCatalog

    item = _fake_item("S2A_T36PZT", "2018-01-30T08:00:00Z", 36.0, 11.0, 5.0, safe=SAFE)
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: [item])
    monkeypatch.setattr(
        cdse, "_select_item_files",
        lambda it, bands, root, cog=True, **kw: [
            (f"{cdse._safe_root_from_item(it)}/B04.jp2", str(tmp_path / "tile/B04.jp2")),
            (f"{cdse._safe_root_from_item(it)}/B08.jp2", str(tmp_path / "tile/B08.jp2")),
        ],
    )
    sizes = {"B04.jp2": 400, "B08.jp2": 600}

    def fake_transfer(src, dst, **kw):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(b"x" * sizes[os.path.basename(dst)])

    monkeypatch.setattr(cdse.fs, "transfer", fake_transfer)
    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))
    monkeypatch.setattr(cdse.fs, "size", lambda p, **k: os.path.getsize(p))

    roi = gpd.GeoDataFrame(geometry=[sg.box(36.0, 11.0, 37.0, 12.0)], crs="EPSG:4326")
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    r = cdse.download(roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
                      ["B04", "B08"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
                      max_tiles=10, cog=False)  # cog=False -> no conversion timing
    assert r.bytes_downloaded == 1000
    assert r.bytes_by_band == {"B04": 400, "B08": 600}
    assert r.convert_seconds == 0.0
    assert r.transfer_seconds >= 0.0


def test_sum_results_aggregates():
    """spec 23 SO-1: download_resume passes aggregate; a later skip contributes 0 bytes."""
    a = cdse.DownloadResult(successful_count=2, total_count=2, bytes_downloaded=100,
                            transfer_seconds=1.0, convert_seconds=0.5, transfer_wall_seconds=0.4,
                            bytes_by_band={"B04": 100}, reason_counts={"ok": 2})
    b = cdse.DownloadResult(successful_count=1, total_count=1, skipped_count=1,
                            reason_counts={"skipped": 1})
    agg = cdse.sum_results([a, b])
    assert (agg.successful_count, agg.total_count, agg.skipped_count) == (3, 3, 1)
    assert agg.bytes_downloaded == 100 and agg.bytes_by_band == {"B04": 100}
    assert (agg.transfer_seconds, agg.convert_seconds) == (1.0, 0.5)
    assert agg.transfer_wall_seconds == 0.4   # wall span sums across passes
    assert agg.reason_counts == {"ok": 2, "skipped": 1}


def test_plan_download_diffs_needed_vs_present(monkeypatch, tmp_path):
    """spec 23 D13: plan_download queries STAC, diffs vs the local catalog, suggests params + eta."""
    from rise_eo.catalog.catalog import TileCatalog

    needed = gpd.GeoDataFrame({"id": ["t1", "t2", "t3"]},
                              geometry=[sg.box(0, 0, 1, 1)] * 3, crs="EPSG:4326")
    monkeypatch.setattr(cdse, "query_catalog", lambda *a, **k: needed)
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    cat.append([{"id": "t1", "collection": "s2", "timestamp": "2018-01-01T00:00:00Z",
                 "s3url": "s3://x", "local_folderpath": "x", "files": "B04.tif",
                 "cloud_cover": 1.0, "offset": 0, "nodata": 0, "geometry": sg.box(0, 0, 1, 1)}])
    plan = cdse.plan_download(
        "roi.geojson", datetime.datetime(2018, 4, 1), datetime.datetime(2018, 9, 30),
        ["B04", "B08"], catalog_filepath=str(tmp_path / "c.parquet"),
        cost_model={"transfer_mb_per_s": 5.0, "mean_bytes_by_band": {"B04": 1e6, "B08": 1e6}},
    )
    assert (plan["needed_count"], plan["present_count"], plan["missing_count"]) == (3, 1, 2)
    assert plan["missing_ids"] == ["t2", "t3"]
    assert plan["download_params"]["max_tiles"] == 3
    assert plan["estimate"]["gb"] == round(2 * 2e6 / 1e9, 2)   # 2 missing x (1MB+1MB)
    msg = cdse.format_download_plan(plan)
    assert "rise_eo.download(" in msg and "missing: 2" in msg


def test_format_download_plan_fully_present_is_not_contradictory():
    """When missing_count == 0, the message must not say "not present" nor emit a
    rise_eo.download(...) command (regression: it did both, contradicting "missing: 0")."""
    plan = {
        "needed_count": 13, "present_count": 13, "missing_count": 0, "missing_ids": [],
        "download_params": {
            "roi": "roi.geojson", "startdate": "2018-04-01T00:00:00",
            "enddate": "2018-06-01T00:00:00", "bands": ["B04"], "max_tiles": 13,
            "max_cloudcover": None, "dst_folderpath": "dst",
        },
    }
    msg = cdse.format_download_plan(plan)
    assert "fully present" in msg and "missing: 0" in msg
    assert "rise_eo.download(" not in msg
    assert "not (fully) present" not in msg


def test_circuit_breaker_trips_and_stops_early(monkeypatch, tmp_path):
    """spec 25 C4: breaker keys on consecutive **transfer** failures only, and the
    A2 pipeline has no chunk boundary -- semantics are "stops within ~max_staged of
    the trip", not an exact chunk count. `max_staged=1` forces strictly serialized
    submission (one file in flight at a time) so the trip point is deterministic."""
    from rise_eo.catalog.catalog import TileCatalog

    # 6 tiles, 1 file each; every transfer fails; breaker at 3 consecutive.
    items = [_fake_item(f"t{i}", "2018-01-01T00:00:00Z", 0.0, 0.0, 0.0) for i in range(6)]
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: items)
    monkeypatch.setattr(cdse, "_transfer_one", lambda *a, **k: (False, "Forbidden", 0.0, 0))
    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: False)

    roi = gpd.GeoDataFrame(geometry=[sg.box(-0.5, -0.5, 1.5, 1.5)], crs="EPSG:4326")
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    result = cdse.download(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B02"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, chunksize=2, max_consecutive_failures=3, max_staged=1,
    )
    assert result.circuit_tripped is True
    # streaming stop: fewer than the full work list attempted, and every attempt failed
    assert result.total_count < 6
    assert result.failed_count == result.total_count


# --- COG-on-download (spec 14) -----------------------------------------------


def _write_raster(path, width=32, height=32):
    """A tiny uint16 GeoTIFF used as a stand-in for a downloaded JP2 band."""
    import numpy as np
    import rasterio
    import rasterio.transform

    data = (np.arange(width * height).reshape(1, height, width) % 4096).astype("uint16")
    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="uint16", crs="EPSG:32637",
        transform=rasterio.transform.from_origin(0, height * 10, 10, 10),
    ) as dst:
        dst.write(data)
    return data


def test_transfer_then_convert_cog_converts_and_is_idempotent(monkeypatch, tmp_path):
    """A fetched JP2 band is converted to a COG .tif, the staging file is removed, and a
    second transfer skips the existing .tif."""
    import os

    import rasterio

    dst = str(tmp_path / "tile" / "B04.tif")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src_data = _write_raster(tmp_path / "ref.tif")

    calls = []

    def fake_transfer(src, staging, **kw):
        # storage.transfer would fetch the JP2 to the local staging sibling;
        # here we just drop a synthetic raster there (content, not extension, matters).
        calls.append(staging)
        _write_raster(staging)

    monkeypatch.setattr(cdse.fs, "transfer", fake_transfer)

    ok, reason, t_s, nbytes = cdse._transfer_one(
        "s3://eodata/x/B04.jp2", dst, {}, needs_convert=True)
    assert (ok, reason) == (True, "ok")
    ok, reason, c_s = cdse._convert_one(dst + ".src.jp2", dst)
    assert (ok, reason) == (True, "ok")
    assert nbytes > 0 and t_s >= 0 and c_s >= 0  # spec 23: transfer/convert/bytes metrics
    assert os.path.exists(dst) and not os.path.exists(dst + ".src.jp2")  # staging gone
    with rasterio.open(dst) as d:
        assert d.driver == "GTiff"  # a COG is a GeoTIFF on disk
        assert (d.read() == src_data).all()  # lossless

    # second call: final .tif present -> skip, no further transfer
    calls.clear()
    assert cdse._transfer_one(
        "s3://eodata/x/B04.jp2", dst, {}, needs_convert=True)[:2] == (True, "skipped")
    assert calls == []


def test_download_cog_accepts_remote_root_via_local_scratch(monkeypatch, tmp_path):
    """spec 34 §5 lifts the local-only guard: cog=True + a remote root_folderpath
    runs the pipeline against local scratch, then pushes the result (here, to an
    fsspec `memory://` filesystem standing in for `abfss://`) and rewrites the
    catalog's local_folderpath to the remote location."""
    import os

    from rise_eo.catalog.catalog import TileCatalog

    item = _fake_item("S2A_T36PZT", "2018-01-30T08:00:00Z", 36.0, 11.0, 5.0, safe=SAFE)
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: [item])
    monkeypatch.setattr(
        cdse, "_select_item_files",
        lambda it, bands, root, cog=True, **kw: [
            (f"{cdse._safe_root_from_item(it)}/B04.jp2", os.path.join(root, "tile", "B04.tif")),
        ],
    )
    def fake_transfer(src, target, **kw):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        _write_raster(target)

    monkeypatch.setattr(cdse.fs, "transfer", fake_transfer)

    roi = gpd.GeoDataFrame(geometry=[sg.box(36.0, 11.0, 37.0, 12.0)], crs="EPSG:4326")
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    remote_root = "memory://fsd-cdse-test/imagery"
    result = cdse.download(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B04"], remote_root, cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=True,
    )
    assert result.successful_count == 1 and result.failed_count == 0

    import fsspec

    memfs = fsspec.filesystem("memory")
    assert memfs.exists("fsd-cdse-test/imagery/tile/B04.tif")
    gdf = cat.read()
    assert gdf["local_folderpath"].iloc[0].startswith(remote_root)


def test_download_resume_loops_until_complete(monkeypatch):
    from rise_eo.sources.cdse import DownloadResult

    seq = [
        DownloadResult(1, 6, failed_count=5, circuit_tripped=True),   # bad window
        DownloadResult(4, 4, failed_count=2),                         # partial, retry
        DownloadResult(6, 6, failed_count=0),                         # complete
    ]
    calls = {"n": 0}
    seen = []

    def fake_download(*a, **k):
        r = seq[calls["n"]]
        calls["n"] += 1
        return r

    monkeypatch.setattr(cdse, "download", fake_download)
    roi = gpd.GeoDataFrame(geometry=[sg.box(0, 0, 1, 1)], crs="EPSG:4326")
    out = cdse.download_resume(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B02"], "/root", object(), CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cooldown_s=0, max_passes=5, on_pass=lambda p, r: seen.append(p),
    )
    assert len(out) == 3 and out[-1].failed_count == 0
    assert seen == [0, 1, 2]  # on_pass fired each pass


# --- spec 25: transfer/convert process-pool pipeline split -------------------


def test_transfer_one_skips_on_existing_final(monkeypatch, tmp_path):
    import os

    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))
    dst = tmp_path / "x.jp2"
    dst.write_bytes(b"data")
    assert cdse._transfer_one(
        "s3://eodata/a.jp2", str(dst), {}, needs_convert=False
    )[:2] == (True, "skipped")


def test_transfer_one_redownloads_zero_byte_leftover(monkeypatch, tmp_path):
    """A 0-byte 'touched' leftover must NOT be treated as done -- it re-transfers."""
    import os

    dst = tmp_path / "z.jp2"
    dst.write_bytes(b"")
    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))
    monkeypatch.setattr(cdse.fs, "size", lambda p, **k: os.path.getsize(p))
    calls = []

    def good_transfer(src, d, **kw):
        calls.append(d)
        with open(d, "wb") as f:
            f.write(b"realbytes")

    monkeypatch.setattr(cdse.fs, "transfer", good_transfer)
    ok, reason, t_s, nbytes = cdse._transfer_one(
        "s3://eodata/z.jp2", str(dst), {}, needs_convert=False
    )
    assert (ok, reason) == (True, "ok")
    assert nbytes == len(b"realbytes")
    assert calls == [str(dst)]


def test_transfer_one_retries_then_fails(monkeypatch, tmp_path):
    import os

    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))

    def boom(src, dst, **kw):
        raise PermissionError("An error occurred (Forbidden) ...")

    monkeypatch.setattr(cdse.fs, "transfer", boom)
    ok, reason, t_s, nbytes = cdse._transfer_one(
        "s3://eodata/b.jp2", str(tmp_path / "nope.jp2"), {}, needs_convert=False, tries=1
    )
    assert (ok, reason) == (False, "Forbidden")
    assert (t_s, nbytes) == (0.0, 0)


def test_transfer_one_needs_convert_writes_to_staging_sibling(monkeypatch, tmp_path):
    import os

    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))
    monkeypatch.setattr(cdse.fs, "size", lambda p, **k: os.path.getsize(p))
    dst = str(tmp_path / "B04.tif")
    calls = []

    def fake_transfer(src, target, **kw):
        calls.append(target)
        with open(target, "wb") as f:
            f.write(b"jp2bytes")

    monkeypatch.setattr(cdse.fs, "transfer", fake_transfer)
    ok, reason, t_s, nbytes = cdse._transfer_one(
        "s3://eodata/B04.jp2", dst, {}, needs_convert=True
    )
    assert (ok, reason) == (True, "ok")
    assert calls == [dst + ".src.jp2"]  # staged sibling, not the final .tif
    assert nbytes == len(b"jp2bytes")
    assert not os.path.exists(dst)


def test_transfer_one_sidecar_writes_straight_to_dst(monkeypatch, tmp_path):
    """cog=False / sidecar files (MTD_TL.xml) skip staging entirely."""
    import os

    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))
    monkeypatch.setattr(cdse.fs, "size", lambda p, **k: os.path.getsize(p))
    dst = str(tmp_path / "MTD_TL.xml")
    calls = []

    def fake_transfer(src, target, **kw):
        calls.append(target)
        with open(target, "wb") as f:
            f.write(b"<xml/>")

    monkeypatch.setattr(cdse.fs, "transfer", fake_transfer)
    ok, reason, t_s, nbytes = cdse._transfer_one(
        "s3://eodata/MTD_TL.xml", dst, {}, needs_convert=False
    )
    assert (ok, reason) == (True, "ok")
    assert calls == [dst]


def test_convert_one_converts_and_cleans_staging(tmp_path):
    import os

    import rasterio

    staging = str(tmp_path / "B04.tif.src.jp2")
    dst = str(tmp_path / "B04.tif")
    src_data = _write_raster(staging)

    ok, reason, c_s = cdse._convert_one(staging, dst)
    assert (ok, reason) == (True, "ok")
    assert c_s >= 0
    assert os.path.exists(dst) and not os.path.exists(staging)
    with rasterio.open(dst) as d:
        assert d.driver == "GTiff"
        assert (d.read() == src_data).all()


def test_convert_one_bad_staging_reports_convert_error_and_cleans_up(tmp_path):
    import os

    staging = tmp_path / "bad.tif.src.jp2"
    staging.write_bytes(b"not a raster")
    dst = str(tmp_path / "bad.tif")

    ok, reason, c_s = cdse._convert_one(str(staging), dst)
    assert (ok, reason) == (False, "ConvertError")
    assert not os.path.exists(staging)  # cleaned up even on failure
    assert not os.path.exists(dst)


class _SyncExecutor:
    """Synchronous stand-in for `ProcessPoolExecutor` (spec 25 test seam): runs the
    submitted callable inline on the calling thread and returns an already-done
    Future -- no subprocess spawned, immune to the spawn-vs-monkeypatch problem."""

    def submit(self, fn, *args, **kwargs):
        fut = concurrent.futures.Future()
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            fut.set_exception(e)
        else:
            fut.set_result(result)
        return fut

    def shutdown(self, wait=True):
        pass


def test_download_cog_pipeline_converts_via_injected_sync_executor(monkeypatch, tmp_path):
    """spec 25: the full download() pipeline (transfer thread pool -> convert pool)
    exercised in-process with a synchronous convert_executor -- no subprocess."""
    import os

    import rasterio

    from rise_eo.catalog.catalog import TileCatalog

    item = _fake_item("S2A_T36PZT", "2018-01-30T08:00:00Z", 36.0, 11.0, 5.0, safe=SAFE)
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: [item])
    monkeypatch.setattr(
        cdse, "_select_item_files",
        lambda it, bands, root, cog=True, **kw: [
            (f"{cdse._safe_root_from_item(it)}/B04.jp2", str(tmp_path / "tile" / "B04.tif")),
        ],
    )
    src_data = _write_raster(tmp_path / "ref.tif")

    def fake_transfer(src, target, **kw):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        _write_raster(target)

    monkeypatch.setattr(cdse.fs, "transfer", fake_transfer)
    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))
    monkeypatch.setattr(cdse.fs, "size", lambda p, **k: os.path.getsize(p))

    roi = gpd.GeoDataFrame(geometry=[sg.box(36.0, 11.0, 37.0, 12.0)], crs="EPSG:4326")
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    result = cdse.download(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B04"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=True, convert_executor=_SyncExecutor(),
    )
    dst = str(tmp_path / "tile" / "B04.tif")
    assert os.path.exists(dst) and not os.path.exists(dst + ".src.jp2")
    with rasterio.open(dst) as d:
        assert d.driver == "GTiff"
        assert (d.read() == src_data).all()
    assert result.convert_seconds > 0
    assert result.bytes_by_band.get("B04", 0) > 0
    gdf = cat.read()
    assert len(gdf) == 1 and gdf["files"].iloc[0] == "B04.tif"

    # rerun: idempotent skip
    result2 = cdse.download(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B04"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=True, convert_executor=_SyncExecutor(),
    )
    assert result2.skipped_count == 1 and result2.failed_count == 0


class _BlockingConvertExecutor:
    """Fake convert executor (spec 25 test seam): each submission blocks until
    `release_all()` is called, and tracks concurrent + peak in-flight count -- used
    to observe the `sem_staged` backpressure bound without racing on the filesystem."""

    def __init__(self):
        self.inflight = 0
        self.peak = 0
        self._lock = threading.Lock()
        self._release = threading.Event()
        self._threads = []

    def submit(self, fn, *args, **kwargs):
        with self._lock:
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)
        fut = concurrent.futures.Future()

        def _run():
            self._release.wait()
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                fut.set_exception(e)
            else:
                fut.set_result(result)
            with self._lock:
                self.inflight -= 1

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self._threads.append(t)
        return fut

    def release_all(self):
        self._release.set()

    def shutdown(self, wait=True):
        if wait:
            for t in self._threads:
                t.join(timeout=5)


def test_download_pipeline_bounds_staged_backpressure(monkeypatch, tmp_path):
    """spec 25 D5/D6: sem_staged never lets more than max_staged files be
    staged-but-unconverted at once -- observed via the fake convert executor's
    in-flight count (peak), not a filesystem race."""
    import os

    from rise_eo.catalog.catalog import TileCatalog

    n_tiles = 6
    items = [
        _fake_item(f"t{i}", "2018-01-01T00:00:00Z", float(i), 0.0, 0.0)
        for i in range(n_tiles)
    ]
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: items)

    def fake_transfer(src, target, **kw):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        _write_raster(target)

    monkeypatch.setattr(cdse.fs, "transfer", fake_transfer)
    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))
    monkeypatch.setattr(cdse.fs, "size", lambda p, **k: os.path.getsize(p))

    executor = _BlockingConvertExecutor()
    max_staged = 2

    def _releaser():
        for _ in range(500):
            if executor.peak >= max_staged:
                break
            time.sleep(0.01)
        executor.release_all()

    threading.Thread(target=_releaser, daemon=True).start()

    roi = gpd.GeoDataFrame(geometry=[sg.box(-1, -1, n_tiles + 1, 1)], crs="EPSG:4326")
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    result = cdse.download(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B02"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=True, max_staged=max_staged, convert_executor=executor,
    )
    assert result.failed_count == 0
    assert executor.peak <= max_staged


def test_download_cog_false_never_spawns_convert_pool(monkeypatch, tmp_path):
    """A cog=False run spawns zero convert processes (spec 25: lazy pool creation)."""
    import os

    from rise_eo.catalog.catalog import TileCatalog

    def _boom_factory(max_workers):
        raise AssertionError("convert pool factory must not be invoked (cog=False)")

    monkeypatch.setattr(cdse, "_make_convert_pool", _boom_factory)

    item = _fake_item("S2A_T36PZT", "2018-01-30T08:00:00Z", 36.0, 11.0, 5.0, safe=SAFE)
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: [item])

    def fake_transfer(src, dst, **kw):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        open(dst, "wb").close()

    monkeypatch.setattr(cdse.fs, "transfer", fake_transfer)
    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))

    roi = gpd.GeoDataFrame(geometry=[sg.box(36.0, 11.0, 37.0, 12.0)], crs="EPSG:4326")
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    result = cdse.download(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B02"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=False,
    )
    assert result.failed_count == 0


def test_download_all_skip_pass_never_spawns_convert_pool(monkeypatch, tmp_path):
    """An all-skip resume pass (cog=True, every final file already present) spawns
    zero convert processes (spec 25: lazy pool creation)."""
    import os

    from rise_eo.catalog.catalog import TileCatalog

    def _boom_factory(max_workers):
        raise AssertionError("convert pool factory must not be invoked (all-skip pass)")

    monkeypatch.setattr(cdse, "_make_convert_pool", _boom_factory)

    item = _fake_item("S2A_T36PZT", "2018-01-30T08:00:00Z", 36.0, 11.0, 5.0, safe=SAFE)
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: [item])
    monkeypatch.setattr(
        cdse, "_select_item_files",
        lambda it, bands, root, cog=True, **kw: [
            (f"{cdse._safe_root_from_item(it)}/B04.jp2", str(tmp_path / "tile" / "B04.tif")),
        ],
    )
    dst = tmp_path / "tile" / "B04.tif"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(b"already-there")
    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))
    monkeypatch.setattr(cdse.fs, "size", lambda p, **k: os.path.getsize(p))

    roi = gpd.GeoDataFrame(geometry=[sg.box(36.0, 11.0, 37.0, 12.0)], crs="EPSG:4326")
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    result = cdse.download(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B04"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=True,
    )
    assert result.skipped_count == 1 and result.failed_count == 0


def test_default_max_staged_disk_aware(monkeypatch, tmp_path, capsys):
    """spec 25 D5/C6: disk-aware MAX_STAGED -- roomy disk settles at `headroom`;
    tight disk shrinks toward `disk_cap`, never below `MAX_CONCURRENT_S3`, and warns
    when the result falls below `floor`."""
    procs = 8
    floor = cdse.config.MAX_CONCURRENT_S3 + procs          # 12
    headroom = cdse.config.MAX_CONCURRENT_S3 + 2 * procs   # 20

    monkeypatch.setattr(cdse.shutil, "disk_usage",
                        lambda p: types.SimpleNamespace(free=500e9))
    assert cdse._default_max_staged(str(tmp_path), procs) == headroom

    monkeypatch.setattr(cdse.shutil, "disk_usage",
                        lambda p: types.SimpleNamespace(free=2e9))  # disk_cap = 2
    staged = cdse._default_max_staged(str(tmp_path), procs)
    assert staged == cdse.config.MAX_CONCURRENT_S3   # clamped to the floor min, not 2
    assert staged < floor
    assert "disk-limited" in capsys.readouterr().out


# --- spec 25b: exception-safe callbacks (no silent hang) ---------------------


class _RaisingSubmitExecutor:
    """Fake convert executor whose `submit()` raises synchronously (as a real
    `ProcessPoolExecutor.submit` does once `BrokenProcessPool` has been detected)."""

    def submit(self, fn, *args, **kwargs):
        raise concurrent.futures.process.BrokenProcessPool("pool is broken")

    def shutdown(self, wait=True):
        pass


class _BrokenResultExecutor:
    """Fake convert executor whose `submit()` succeeds but returns a Future already
    completed with `BrokenProcessPool` -- so `_on_convert_done` fires and
    `cfut.result()` raises (simulates a worker dying *after* being handed the
    submission)."""

    def submit(self, fn, *args, **kwargs):
        fut = concurrent.futures.Future()
        fut.set_exception(concurrent.futures.process.BrokenProcessPool("worker died"))
        return fut

    def shutdown(self, wait=True):
        pass


def _download_in_thread(*args, timeout=10.0, **kwargs):
    """Run `cdse.download` on a watchdog thread and assert it doesn't hang. Returns
    the `DownloadResult` (or re-raises any exception from the worker thread)."""
    box: dict = {}

    def _run():
        try:
            box["result"] = cdse.download(*args, **kwargs)
        except Exception as e:
            box["error"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    assert not t.is_alive(), "download() hung (watchdog timeout)"
    if "error" in box:
        raise box["error"]
    return box["result"]


def _setup_multi_tile_download(monkeypatch, tmp_path, n_tiles=4):
    """Common fixture: n_tiles items, one B04 band each, fake local transfer."""
    import os

    from rise_eo.catalog.catalog import TileCatalog

    items = [
        _fake_item(f"t{i}", "2018-01-01T00:00:00Z", float(i), 0.0, 0.0)
        for i in range(n_tiles)
    ]
    monkeypatch.setattr(cdse, "_search_items", lambda *a, **k: items)
    monkeypatch.setattr(
        cdse, "_select_item_files",
        lambda it, bands, root, cog=True, **kw: [
            (f"{cdse._safe_root_from_item(it)}/B04.jp2",
             str(tmp_path / it.id / "B04.tif")),
        ],
    )

    def fake_transfer(src, target, **kw):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(b"jp2bytes")

    monkeypatch.setattr(cdse.fs, "transfer", fake_transfer)
    monkeypatch.setattr(cdse.fs, "exists", lambda p, **k: os.path.exists(p))
    monkeypatch.setattr(cdse.fs, "size", lambda p, **k: os.path.getsize(p))

    roi = gpd.GeoDataFrame(geometry=[sg.box(-1, -1, n_tiles + 1, 1)], crs="EPSG:4326")
    cat = TileCatalog(str(tmp_path / "c.parquet"))
    return roi, cat


def test_pool_submit_raises_no_hang_finalized_as_failure(monkeypatch, tmp_path):
    """spec 25b test 1: `pool.submit` raises `BrokenProcessPool` -> no hang, every
    item accounted, `pool_broken` set, `"PoolBroken"` recorded, submit loop stopped
    early."""
    roi, cat = _setup_multi_tile_download(monkeypatch, tmp_path, n_tiles=6)
    result = _download_in_thread(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B04"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=True, max_staged=2,
        convert_executor=_RaisingSubmitExecutor(),
    )
    assert result.pool_broken is True
    assert result.successful_count + result.failed_count == result.total_count
    assert result.reason_counts.get("PoolBroken", 0) > 0
    assert result.total_count < 6  # submit loop stopped early on pool_broken


def test_convert_done_result_raises_no_hang_permit_released(monkeypatch, tmp_path):
    """spec 25b test 2: `cfut.result()` raises `BrokenProcessPool` -> no hang, permit
    released (small `max_staged` would deadlock on a leak), `pool_broken` set."""
    roi, cat = _setup_multi_tile_download(monkeypatch, tmp_path, n_tiles=6)
    result = _download_in_thread(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B04"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=True, max_staged=1,
        convert_executor=_BrokenResultExecutor(),
    )
    assert result.pool_broken is True
    assert result.reason_counts.get("PoolBroken", 0) > 0
    assert result.successful_count + result.failed_count == result.total_count


def test_pool_broken_does_not_trip_transfer_breaker(monkeypatch, tmp_path):
    """spec 25b test 3: PoolBroken is breaker-neutral -- a run whose converts all fail
    via a broken pool must not trip the transfer circuit breaker."""
    roi, cat = _setup_multi_tile_download(monkeypatch, tmp_path, n_tiles=6)
    result = _download_in_thread(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B04"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=True, max_staged=2, max_consecutive_failures=2,
        convert_executor=_RaisingSubmitExecutor(),
    )
    assert result.pool_broken is True
    assert result.circuit_tripped is False


def test_catalog_flush_failure_does_not_hang_and_recovers_on_resume(monkeypatch, tmp_path):
    """spec 25b test 4: a chunk-flush `_append_downloaded` failure doesn't hang or
    lose the drain; the retried/failed rows are recovered by a subsequent pass
    (idempotent skip on already-present files, re-queue-and-retry on the flush)."""
    roi, cat = _setup_multi_tile_download(monkeypatch, tmp_path, n_tiles=6)

    calls = {"n": 0}
    real_append = cdse._append_downloaded

    def flaky_append(catalog, tile_meta, results, declaration):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full (simulated)")
        return real_append(catalog, tile_meta, results, declaration)

    monkeypatch.setattr(cdse, "_append_downloaded", flaky_append)

    result = _download_in_thread(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B04"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=False, chunksize=2,
    )
    assert result.failed_count == 0
    assert result.successful_count + result.failed_count == result.total_count
    assert result.total_count == 6

    # a subsequent normal pass writes the catalog (idempotent-skip recovery)
    result2 = cdse.download(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B04"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=False,
    )
    assert result2.skipped_count == 6
    gdf = cat.read()
    assert len(gdf) == 6


def test_sum_results_ors_pool_broken():
    """spec 25b test 6: `sum_results` ORs `pool_broken` across passes."""
    a = cdse.DownloadResult(successful_count=1, total_count=1, pool_broken=True)
    b = cdse.DownloadResult(successful_count=1, total_count=1, pool_broken=False)
    agg = cdse.sum_results([a, b])
    assert agg.pool_broken is True

    c = cdse.DownloadResult(successful_count=1, total_count=1, pool_broken=False)
    d = cdse.DownloadResult(successful_count=1, total_count=1, pool_broken=False)
    assert cdse.sum_results([c, d]).pool_broken is False


# --- spec 26: should_stop seam + _fmt_progress ETA ----------------------------


def test_should_stop_halts_submit_loop_mid_pass(monkeypatch, tmp_path):
    """spec 26 test 1: a `should_stop` that flips True once the first item is
    finalized halts the submit loop early -- no hang (watchdog), `stopped is True`,
    fewer than the full work list attempted, every attempted item accounted, no
    leaked `sem_staged` permit (a leak would deadlock the next acquire and trip the
    watchdog timeout). `max_staged=1` + a synchronous convert executor forces strictly
    serialized submission (as the spec-25 circuit-breaker test does) so the stop point
    is deterministic."""
    monkeypatch.setattr(cdse.config, "STOP_CHECK_EVERY_S", 0)  # disable the stop-check throttle
    roi, cat = _setup_multi_tile_download(monkeypatch, tmp_path, n_tiles=6)

    finalized = {"n": 0}
    real_convert_one = cdse._convert_one

    def counting_convert_one(staging, dst, **kw):
        result = real_convert_one(staging, dst, **kw)
        finalized["n"] += 1
        return result

    monkeypatch.setattr(cdse, "_convert_one", counting_convert_one)

    result = _download_in_thread(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B04"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=True, max_staged=1, convert_executor=_SyncExecutor(),
        should_stop=lambda: finalized["n"] >= 1,
    )
    assert result.stopped is True
    assert result.total_count < 6
    assert result.successful_count + result.failed_count == result.total_count


def test_should_stop_none_is_noop(monkeypatch, tmp_path):
    """spec 26 test 2: `should_stop=None` (the default) is today's behavior exactly --
    a normal run finishes all work, `stopped is False`."""
    roi, cat = _setup_multi_tile_download(monkeypatch, tmp_path, n_tiles=6)
    result = _download_in_thread(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B04"], str(tmp_path), cat, CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cog=False,
    )
    assert result.stopped is False
    assert result.total_count == 6


def test_download_resume_breaks_on_stopped_pass(monkeypatch):
    """spec 26 test 3: a `should_stop` that trips on pass 1 ends `download_resume`
    immediately -- only one pass ran, the last result has `stopped True`, no cooldown
    taken (a user stop is neither a bad-window cooldown nor a completion)."""
    from rise_eo.sources.cdse import DownloadResult

    calls = {"n": 0}

    def fake_download(*a, **k):
        calls["n"] += 1
        return DownloadResult(2, 4, failed_count=0, stopped=True)

    monkeypatch.setattr(cdse, "download", fake_download)
    slept = {"called": False}
    monkeypatch.setattr(time, "sleep", lambda s: slept.__setitem__("called", True))

    roi = gpd.GeoDataFrame(geometry=[sg.box(0, 0, 1, 1)], crs="EPSG:4326")
    out = cdse.download_resume(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B02"], "/root", object(), CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cooldown_s=5, max_passes=5, should_stop=lambda: False,
    )
    assert calls["n"] == 1
    assert len(out) == 1 and out[-1].stopped is True
    assert slept["called"] is False


def test_sum_results_ors_stopped():
    """spec 26 test 4: `sum_results` ORs `stopped` across passes."""
    a = cdse.DownloadResult(successful_count=1, total_count=1, stopped=True)
    b = cdse.DownloadResult(successful_count=1, total_count=1, stopped=False)
    agg = cdse.sum_results([a, b])
    assert agg.stopped is True

    c = cdse.DownloadResult(successful_count=1, total_count=1, stopped=False)
    d = cdse.DownloadResult(successful_count=1, total_count=1, stopped=False)
    assert cdse.sum_results([c, d]).stopped is False


def test_fmt_progress_eta_placeholder_before_first_completion():
    """spec 26 test 7: ETA is `~?` until `done > 0` (no rate to extrapolate from
    yet), a finite `~Nm` once there's a completion, and all existing fields (spec 23)
    are unchanged."""
    s0 = cdse._fmt_progress(0, 200, ok_n=0, fail_n=0, skipped=0, elapsed_s=10.0)
    assert "ETA ~?" in s0

    s1 = cdse._fmt_progress(50, 200, ok_n=45, fail_n=5, skipped=0, elapsed_s=100.0)
    assert "ETA ~?" not in s1
    assert "ETA ~" in s1
    for token in ("50/200", "25%", "ok=45", "fail=5", "file/s"):
        assert token in s1


# --- _roi_gdf reads through rise_eo.storage, not gpd.read_file(path) ---------------
# Regression: spec 37 dispatches downloads with the roi on blob. pyogrio/GDAL does
# not understand `abfss://` and raised `DataSourceError: No such file or directory`
# for a file that demonstrably existed. Same fix workflows/task.py already carries
# (spec 36 D6a). `memory://` stands in for blob here: both are fsspec-only schemes
# that GDAL cannot open, so this test fails on a revert to `gpd.read_file(roi)`.

def test_roi_gdf_reads_a_non_gdal_url_through_the_storage_seam():
    from rise_eo.storage import fs

    gdf = gpd.GeoDataFrame(
        {"id": ["a"]}, geometry=[sg.box(16.0, 48.0, 16.1, 48.1)], crs="EPSG:4326"
    )
    url = "memory://roi_seam/s2grid=test.geojson"
    with fs.open(url, "wb") as f:
        f.write(gdf.to_json().encode())

    out = cdse._roi_gdf(url)
    assert len(out) == 1
    assert out.geometry.iloc[0].bounds == (16.0, 48.0, 16.1, 48.1)


def test_roi_gdf_still_accepts_a_local_path_and_a_gdf(tmp_path):
    gdf = gpd.GeoDataFrame(geometry=[sg.box(0, 0, 1, 1)], crs="EPSG:4326")
    p = tmp_path / "roi.geojson"
    gdf.to_file(p, driver="GeoJSON")

    assert len(cdse._roi_gdf(str(p))) == 1
    assert cdse._roi_gdf(gdf) is gdf


def test_download_resume_forwards_every_option_to_download(monkeypatch):
    """A14 (#145): each resume pass hands `download` the caller's `collection`, `processing`,
    `max_cloudcover`, `cog`, `max_convert_procs`, `max_staged`, `max_concurrent_s3` and
    `chunksize` (non-default values, so a dropped one cannot hide behind a default)."""
    from rise_eo.sources.cdse import DownloadResult

    seen = []

    def fake_download(*a, **k):
        seen.append(k)
        return DownloadResult(1, 1, failed_count=0)

    monkeypatch.setattr(cdse, "download", fake_download)
    expected = dict(
        collection="c-x", processing="<=2.0", max_cloudcover=12.5, cog=False,
        max_convert_procs=3, max_staged=4, max_concurrent_s3=5, chunksize=7,
    )
    roi = gpd.GeoDataFrame(geometry=[sg.box(0, 0, 1, 1)], crs="EPSG:4326")
    cdse.download_resume(
        roi, datetime.datetime(2018, 1, 1), datetime.datetime(2018, 2, 1),
        ["B02"], "/root", object(), CdseCredentials(**DUMMY_FIELDS),
        max_tiles=10, cooldown_s=0, max_passes=1, **expected,
    )
    assert {k: seen[0][k] for k in expected} == expected
