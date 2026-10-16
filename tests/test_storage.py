"""Automated mirror of tests/manual/storage.md Section A (local backend).

Section B (S3-compatible transport) stays manual — it needs credentials and a
network endpoint. These exercise the fsspec seam end-to-end on the local backend:
the same code paths a remote backend uses, just with `file://`.
"""

import geopandas as gpd
import numpy as np
import shapely.geometry as sg

from rise_eo.storage import fs


def test_makedirs_and_exists(tmp_path):
    sub = tmp_path / "a/b/c"
    fs.makedirs(str(sub))
    assert fs.exists(str(sub))
    assert not fs.exists(str(tmp_path / "nope"))


def test_open_write_read_text(tmp_path):
    p = str(tmp_path / "hello.txt")
    with fs.open(p, "w") as f:
        f.write("hi fsd")
    with fs.open(p, "r") as f:
        assert f.read() == "hi fsd"
    assert fs.exists(p)


def test_save_load_npy_array(tmp_path):
    arr = np.arange(12).reshape(3, 4)
    p = str(tmp_path / "arr.npy")
    fs.save_npy(p, arr)
    out = fs.load_npy(p)
    assert out.shape == (3, 4)
    assert out.dtype == np.int64
    assert (out == arr).all()


def test_save_load_npy_pickled_metadata(tmp_path):
    # How datacube metadata.pickle.npy round-trips (note the [()] unwrap).
    meta = {"bands": ["B02", "B08"], "timestamps": 3}
    p = str(tmp_path / "meta.pickle.npy")
    fs.save_npy(p, meta, allow_pickle=True)
    loaded = fs.load_npy(p, allow_pickle=True)[()]
    assert loaded == meta


def test_write_read_geoparquet(tmp_path):
    # GeoParquet stores CRS as PROJJSON, so compare the EPSG code, not str(crs).
    gdf = gpd.GeoDataFrame(
        {"id": ["t1", "t2"]},
        geometry=[sg.Point(0, 0), sg.Point(1, 1)],
        crs="EPSG:4326",
    )
    p = str(tmp_path / "catalog.parquet")
    fs.write_parquet(p, gdf)
    back = fs.read_parquet(p)
    assert len(back) == 2
    assert back.crs.to_epsg() == 4326
    assert list(back["id"]) == ["t1", "t2"]
    assert back.geometry.iloc[0].wkt == "POINT (0 0)"


# --- .attrs footer preservation (spec 35 §2) ----------------------------------


def test_write_read_parquet_preserves_attrs(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"id": ["t1"]}, geometry=[sg.Point(0, 0)], crs="EPSG:4326",
    )
    gdf.attrs["fsd:declaration"] = {"reference_band": "B04"}
    p = str(tmp_path / "catalog.parquet")
    fs.write_parquet(p, gdf)
    back = fs.read_parquet(p)
    assert back.attrs["fsd:declaration"] == {"reference_band": "B04"}


def test_read_parquet_stamps_source_path(tmp_path):
    gdf = gpd.GeoDataFrame({"id": ["t1"]}, geometry=[sg.Point(0, 0)], crs="EPSG:4326")
    p = str(tmp_path / "catalog.parquet")
    fs.write_parquet(p, gdf)
    back = fs.read_parquet(p)
    assert back.attrs[fs.SOURCE_PATH_ATTRS_KEY] == p


def test_write_parquet_strips_source_path_before_writing(tmp_path):
    """spec 35 §10: fsd:source_path is read-side bookkeeping -- it must never be
    serialized into a written artifact (it would leak a local absolute path)."""
    gdf = gpd.GeoDataFrame({"id": ["t1"]}, geometry=[sg.Point(0, 0)], crs="EPSG:4326")
    p1 = str(tmp_path / "a.parquet")
    fs.write_parquet(p1, gdf)
    read_back = fs.read_parquet(p1)
    assert fs.SOURCE_PATH_ATTRS_KEY in read_back.attrs

    p2 = str(tmp_path / "b.parquet")
    fs.write_parquet(p2, read_back)  # write what we just read back
    fresh = fs.read_parquet(p2)
    # p2's own read stamps fsd:source_path = p2 (not p1, and it must not have
    # been serialized as leftover data from read_back.attrs).
    assert fresh.attrs[fs.SOURCE_PATH_ATTRS_KEY] == p2
    assert fs.peek_parquet_attrs(p2) == {}  # nothing else got written to the footer


def test_write_parquet_empty_attrs_is_the_zero_cost_fast_path(tmp_path):
    """No PANDAS_ATTRS footer key at all when attrs is empty (spec 35 §8.5) --
    proves the fast path is byte-for-byte the pre-spec-35 write."""
    import pyarrow.parquet as pq

    gdf = gpd.GeoDataFrame({"id": ["t1"]}, geometry=[sg.Point(0, 0)], crs="EPSG:4326")
    p = str(tmp_path / "catalog.parquet")
    fs.write_parquet(p, gdf)
    with open(p, "rb") as f:
        metadata = pq.read_metadata(f)
    assert fs.PANDAS_ATTRS_FOOTER_KEY not in (metadata.metadata or {})


def test_stamped_file_is_still_valid_geoparquet(tmp_path):
    """spec 35 §8.3: a stamped file reads with stock gpd.read_parquet; the `geo`
    key survives; geometry/CRS are unaffected by the footer rewrite."""
    gdf = gpd.GeoDataFrame(
        {"id": ["t1", "t2"]},
        geometry=[sg.Point(0, 0), sg.Point(1, 1)],
        crs="EPSG:4326",
    )
    gdf.attrs["fsd:declaration"] = {"reference_band": "B04"}
    p = str(tmp_path / "catalog.parquet")
    fs.write_parquet(p, gdf)

    stock_back = gpd.read_parquet(p)  # stock geopandas, not rise_eo.storage.fs
    assert stock_back.crs.to_epsg() == 4326
    assert list(stock_back["id"]) == ["t1", "t2"]

    import pyarrow.parquet as pq

    with open(p, "rb") as f:
        metadata = pq.read_metadata(f)
    assert b"geo" in (metadata.metadata or {})

    fsd_back = fs.read_parquet(p)
    assert fsd_back.crs.to_epsg() == stock_back.crs.to_epsg()
    assert list(fsd_back.geometry) == list(stock_back.geometry)


def test_peek_parquet_attrs_footer_only(tmp_path):
    gdf = gpd.GeoDataFrame({"id": ["t1"]}, geometry=[sg.Point(0, 0)], crs="EPSG:4326")
    gdf.attrs["fsd:declaration"] = {"reference_band": "B04"}
    p = str(tmp_path / "catalog.parquet")
    fs.write_parquet(p, gdf)
    assert fs.peek_parquet_attrs(p) == {"fsd:declaration": {"reference_band": "B04"}}


def test_peek_parquet_attrs_on_a_non_local_filesystem():
    """`TileCatalog.append`'s conflict check reads the stamp through
    `peek_parquet_attrs` on every append — including against an `abfss://`
    catalog (the `rise` blob ingest). Pin that the footer-only read works on a
    non-local fsspec filesystem, not just a local path."""
    import uuid

    gdf = gpd.GeoDataFrame({"id": ["t1"]}, geometry=[sg.Point(0, 0)], crs="EPSG:4326")
    gdf.attrs["fsd:declaration"] = {"reference_band": "B04"}
    p = f"memory://{uuid.uuid4()}/catalog.parquet"
    fs.write_parquet(p, gdf)
    assert fs.peek_parquet_attrs(p) == {"fsd:declaration": {"reference_band": "B04"}}


def test_ls_and_glob(tmp_path):
    import os

    fs.save_npy(str(tmp_path / "arr.npy"), np.zeros(3))
    fs.save_npy(str(tmp_path / "meta.pickle.npy"), {"x": 1}, allow_pickle=True)
    with fs.open(str(tmp_path / "hello.txt"), "w") as f:
        f.write("hi")

    names = sorted(os.path.basename(x) for x in fs.ls(str(tmp_path)))
    assert {"arr.npy", "meta.pickle.npy", "hello.txt"}.issubset(set(names))

    npys = sorted(
        os.path.basename(x) for x in fs.glob(str(tmp_path / "*.npy"))
    )
    assert npys == ["arr.npy", "meta.pickle.npy"]


def test_put_get_transfer(tmp_path):
    arr = np.arange(12).reshape(3, 4)
    p_arr = str(tmp_path / "arr.npy")
    fs.save_npy(p_arr, arr)

    p_put = str(tmp_path / "uploaded/arr_copy.npy")
    fs.put(p_arr, p_put)
    assert fs.exists(p_put)

    p_get = str(tmp_path / "downloaded/arr_back.npy")
    fs.get(p_put, p_get)
    assert np.array_equal(fs.load_npy(p_get), arr)

    p_xfer = str(tmp_path / "transferred/arr_t.npy")
    fs.transfer(p_arr, p_xfer)
    assert np.array_equal(fs.load_npy(p_xfer), arr)


def test_transfer_is_atomic_on_failure(tmp_path, monkeypatch):
    """A transfer that fails mid-copy leaves NO file at the destination (no 0-byte or
    truncated leftover) and cleans up its .part sidecar."""
    import pytest

    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 100)
    dst = tmp_path / "out" / "dst.bin"

    class _Boom:
        @staticmethod
        def copyfileobj(a, b):
            raise OSError("connection reset mid-copy")

    monkeypatch.setattr(fs, "shutil", _Boom)
    with pytest.raises(OSError):
        fs.transfer(str(src), str(dst))
    assert not dst.exists()
    assert not (tmp_path / "out" / "dst.bin.part").exists()


def test_size(tmp_path):
    p = str(tmp_path / "f.bin")
    with open(p, "wb") as f:
        f.write(b"abc")
    assert fs.size(p) == 3
    empty = str(tmp_path / "e.bin")
    open(empty, "wb").close()
    assert fs.size(empty) == 0


# --- write_text / write_bytes seam ------------------------------------------
# (A `_write_with_retry` was added here 2026-07-28 for a misdiagnosed "transient
#  adlfs race" and REVERTED the same day -- the real bug was duplicate work-unit
#  ids making threads write one blob, spec 21 D-GRID-1 / TODO #58. Its four unit
#  tests went with it; they pinned behaviour that no longer exists.)


def test_write_text_round_trip_on_a_non_local_filesystem():
    """write_text is the new seam call-site swapped into create_datacube.py:143
    (TODO #57) -- pin it works on a remote (non-local) fsspec filesystem too."""
    import uuid

    p = f"memory://{uuid.uuid4()}/geometry.geojson"
    fs.write_text(p, '{"type": "FeatureCollection", "features": []}')
    with fs.open(p, "r") as f:
        assert f.read() == '{"type": "FeatureCollection", "features": []}'


# --- read_geo: the ONE shared vector reader (TODO #47) -----------------------
#
# GDAL/pyogrio has no `abfss://` driver, so `gpd.read_file(<fsspec-only url>)` reports
# `No such file or directory` for a file that demonstrably exists. `memory://` reproduces
# that exactly -- it is an fsspec-only scheme GDAL knows nothing about -- which is why
# these pin against a non-local backend rather than tmp_path. Hit live three times:
# workflows/task.py (spec 36 D6a), sources/cdse._roi_gdf (run-book 37 Phase 1), and
# create_training_data's label polygons (the spec-40 demo, 2026-07-29).

def _geojson_on_memory_fs(n=2):
    import uuid

    gdf = gpd.GeoDataFrame(
        {"fid": list(range(n)), "crop": ["wheat", "maize"][:n],
         "geometry": [sg.box(i, 0, i + 1, 1) for i in range(n)]},
        crs="EPSG:4326",
    )
    p = f"memory://{uuid.uuid4()}/polys.geojson"
    fs.write_text(p, gdf.to_json())
    return p, gdf


def test_read_geo_reads_from_a_scheme_gdal_does_not_understand():
    p, gdf = _geojson_on_memory_fs()
    got = fs.read_geo(p)
    assert len(got) == len(gdf)
    assert list(got["fid"]) == [0, 1]


def test_gpd_read_file_still_fails_on_that_url():
    """The bug this reader exists for, pinned so nobody 'simplifies' read_geo back into
    a bare gpd.read_file. If this ever starts passing, GDAL grew fsspec support and the
    indirection could be revisited -- until then it is load-bearing."""
    import pytest

    p, _ = _geojson_on_memory_fs()
    with pytest.raises(Exception):  # noqa: B017 - pyogrio/fiona raise different types
        gpd.read_file(p)


def test_read_geo_still_reads_a_plain_local_path(tmp_path):
    """Callers must not need an is-it-remote branch."""
    gdf = gpd.GeoDataFrame({"fid": [0], "geometry": [sg.box(0, 0, 1, 1)]}, crs="EPSG:4326")
    fp = tmp_path / "local.geojson"
    gdf.to_file(fp, driver="GeoJSON")
    assert len(fs.read_geo(str(fp))) == 1


def test_the_three_todo_47_sites_accept_a_non_local_url():
    """`api._as_gdf` (create_training_data label polygons), `api`'s run_inference roi
    preflight, and `grid._as_gdf_4326` (roi_to_s2_grids) all went through
    `gpd.read_file` and broke identically on a blob-hosted input. One call each."""
    from rise_eo import api, grid

    p, _ = _geojson_on_memory_fs()

    assert len(api._as_gdf(p)) == 2                      # site 1
    assert len(grid._as_gdf_4326(p)) == 2                # site 3
    # site 2 shares _as_gdf_4326 via roi_to_s2_grids -- the path run_inference re-tiles on
    assert len(grid.roi_to_s2_grids(p, grid_size_km=5, scale_fact=1.1)) > 0


# --- rename: the atomic-publish primitive (spec 51 D2) ----------------------


def test_rename_refuses_to_nest_a_directory_into_an_existing_one(tmp_path):
    """`fs.rename` documents itself as `os.rename` locally, and spec 51 D2 leans on that
    to make a lost publish race *fail* instead of corrupting the winner. fsspec's
    `LocalFileSystem.mv` is `shutil.move`, which would move `src` INSIDE `dst` and report
    success -- pin that fsd does not inherit that."""
    import pytest

    src = tmp_path / "stage"
    src.mkdir()
    (src / "bundle.json").write_text("loser")
    dst = tmp_path / "v1"
    dst.mkdir()
    (dst / "bundle.json").write_text("winner")

    with pytest.raises(OSError):
        fs.rename(str(src), str(dst))

    assert sorted(p.name for p in dst.iterdir()) == ["bundle.json"]
    assert (dst / "bundle.json").read_text() == "winner"
    assert src.exists()  # the loser's staged copy is left intact for it to clean up


def test_rename_replaces_an_existing_file_atomically(tmp_path):
    """The other half of `os.rename` semantics, relied on by the datacube sidecar write
    (`datacube/builder.py`) and by `set_alias`: a *file* destination is replaced, not
    refused."""
    src = tmp_path / "tmp.json"
    src.write_text("new")
    dst = tmp_path / "final.json"
    dst.write_text("old")

    fs.rename(str(src), str(dst))

    assert dst.read_text() == "new"
    assert not src.exists()


# --- #80: fsspec backends ship in extras, so a missing one must name the extra ------
# fsspec resolves a backend from the URL scheme and never gets imported by fsd, so
# upstream raises "Install s3fs to access S3" -- a package name, not the extra that
# provides it. `_fs_and_path` is the single resolution point, so the mapping lives there.


def _raise_import_error(*args, **kwargs):
    raise ImportError("Install s3fs to access S3")


def test_missing_backend_names_the_extra_not_the_package(monkeypatch):
    import pytest

    monkeypatch.setattr(fs.fsspec.core, "url_to_fs", _raise_import_error)
    with pytest.raises(ImportError, match=r"fsd\[s3\]"):
        fs._fs_and_path("s3://bucket/key.tif")


def test_missing_backend_maps_abfss_to_the_azure_extra(monkeypatch):
    import pytest

    monkeypatch.setattr(fs.fsspec.core, "url_to_fs", _raise_import_error)
    with pytest.raises(ImportError, match=r"fsd\[azure\]"):
        fs._fs_and_path("abfss://c@a.dfs.core.windows.net/x.tif")


def test_an_unmapped_protocol_import_error_is_re_raised_untouched(monkeypatch):
    import pytest

    monkeypatch.setattr(fs.fsspec.core, "url_to_fs", _raise_import_error)
    with pytest.raises(ImportError, match="Install s3fs to access S3"):
        fs._fs_and_path("gs://bucket/key.tif")


# --- #80: the extras split is a gate, not a convention -----------------------------
# snakemake and s3fs are declared nowhere in src/rise_eo/ -- one is a subprocess, one is an
# fsspec backend -- so nothing at import time would notice them drifting back into core.
# The measured cost of that drift is +53 packages / +111 MB on every core install.


def _pyproject() -> dict:
    import pathlib
    import tomllib

    root = pathlib.Path(__file__).resolve().parent.parent
    return tomllib.loads((root / "pyproject.toml").read_text())


def test_snakemake_and_s3fs_are_extras_not_core():
    project = _pyproject()["project"]
    core = " ".join(project["dependencies"])
    extras = project["optional-dependencies"]

    assert "snakemake" not in core, "snakemake is the local runner only (#80)"
    assert "s3fs" not in core, "s3fs is an fsspec backend, resolved by URL scheme (#80)"
    assert any("snakemake" in d for d in extras["local"])
    assert any("s3fs" in d for d in extras["s3"])


def test_neither_extra_is_imported_anywhere_in_src():
    """The premise of #80: both are config, not code. If either gains an `import`, the
    move stops being free and this test is the place that says so."""
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parent.parent / "src" / "rise_eo"
    pattern = re.compile(r"^\s*(?:import|from)\s+(?:snakemake|s3fs)\b", re.MULTILINE)
    offenders = [str(p) for p in src.rglob("*.py") if pattern.search(p.read_text())]
    assert not offenders, f"snakemake/s3fs imported in core code: {offenders}"
