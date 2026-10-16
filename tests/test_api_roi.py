"""Spec 21 — run_inference(roi=…) preflight guards + the three merge modes.

Preflight-guard tests fail *before* tiling, so they need neither the [grid] extra nor real
imagery. The merge tests build small synthetic single-band COGs in two UTM zones.
"""

import datetime

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

import rise_eo
from rise_eo.api import _merge_outputs
from rise_eo.model import BaseModelAdapter


class _Tiny(BaseModelAdapter):
    required_bands = ["B04", "B08"]
    n_timestamps = 2
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["c"]
    feature_sequence = []

    def load(self):
        pass

    def predict(self, X):
        return X[:, 0].astype("uint8")


ROI = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")


# --- entry-point guards (before any build) -----------------------------------

def test_roi_and_cubes_mutually_exclusive(tmp_path):
    with pytest.raises(rise_eo.PreflightError, match="not both"):
        rise_eo.run_inference(
            _Tiny(), inference_datacubes=["x"], output_folderpath=str(tmp_path),
            roi=ROI, catalog_filepath="sentinel-2-l2a/c.parquet",
            startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 11),
            mosaic_days=20, bands=["B04", "B08"],
        )


def test_neither_roi_nor_cubes(tmp_path):
    with pytest.raises(rise_eo.PreflightError, match="pass roi="):
        rise_eo.run_inference(_Tiny(), output_folderpath=str(tmp_path))


def test_output_folderpath_required():
    with pytest.raises(rise_eo.PreflightError, match="output_folderpath is required"):
        rise_eo.run_inference(_Tiny(), inference_datacubes=["x"])


def test_bad_merge_value(tmp_path):
    with pytest.raises(rise_eo.PreflightError, match="merge must be"):
        rise_eo.run_inference(_Tiny(), inference_datacubes=["x"],
                          output_folderpath=str(tmp_path), merge="bogus")


def test_roi_preflight_t_mismatch(tmp_path):
    # 2018-06-01..06-11 @ 20d -> T=1, but the model wants T=2 -> refuse before tiling
    with pytest.raises(rise_eo.PreflightError, match="needs T=2"):
        rise_eo.run_inference(
            _Tiny(), output_folderpath=str(tmp_path), roi=ROI, catalog_filepath="sentinel-2-l2a/c.parquet",
            startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 6, 11),
            mosaic_days=20, bands=["B04", "B08"],
        )


def test_roi_preflight_missing_bands(tmp_path):
    with pytest.raises(rise_eo.PreflightError, match="missing model-required"):
        rise_eo.run_inference(
            _Tiny(), output_folderpath=str(tmp_path), roi=ROI, catalog_filepath="sentinel-2-l2a/c.parquet",
            startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 11),
            mosaic_days=20, bands=["B04"],                       # missing B08
        )


# --- spec 21 D-GRID-1: duplicate cell ids die in preflight, before any spend ---


def test_roi_preflight_refuses_duplicate_cell_ids(tmp_path, monkeypatch):
    """`id` is the work-unit key, so duplicates put N tasks on one export folder --
    concurrent same-blob writes (`InvalidBlockList`) on blob, a silent overwrite locally.
    `roi_to_s2_grids` prevents them at source; this pins the preflight seatbelt AND that
    it fires before the expensive steps (blob makedirs, the bundle upload, setup's N
    writes, AML dispatch). Tiling is stubbed so the test needs no [grid] extra."""
    import rise_eo.api as api
    import rise_eo.grid as _grid_mod

    dupes = gpd.GeoDataFrame(
        {"id": ["cell_a", "cell_a", "cell_b"]},
        geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2), box(2, 2, 3, 3)], crs="EPSG:4326",
    )
    monkeypatch.setattr(_grid_mod, "roi_to_s2_grids", lambda *a, **kw: dupes)

    spent = []
    monkeypatch.setattr(api.fs, "makedirs", lambda *a, **kw: spent.append("makedirs"))
    monkeypatch.setattr(api, "_ensure_bundle", lambda *a, **kw: spent.append("bundle"))

    with pytest.raises(rise_eo.PreflightError, match="only 2 distinct cell ids"):
        rise_eo.run_inference(
            _Tiny(), output_folderpath=str(tmp_path), roi=ROI, catalog_filepath="sentinel-2-l2a/c.parquet",
            startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 11),
            mosaic_days=20, bands=["B04", "B08"],
        )
    assert spent == []  # nothing was created or uploaded


def test_roi_preflight_refuses_an_roi_that_tiles_to_nothing(tmp_path, monkeypatch):
    import rise_eo.api as api
    import rise_eo.grid as _grid_mod

    empty = gpd.GeoDataFrame({"id": []}, geometry=[], crs="EPSG:4326")
    monkeypatch.setattr(_grid_mod, "roi_to_s2_grids", lambda *a, **kw: empty)
    monkeypatch.setattr(api, "_ensure_bundle",
                        lambda *a, **kw: pytest.fail("bundle staged despite a bad roi"))

    with pytest.raises(rise_eo.PreflightError, match="0 grid cells"):
        rise_eo.run_inference(
            _Tiny(), output_folderpath=str(tmp_path), roi=ROI, catalog_filepath="sentinel-2-l2a/c.parquet",
            startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 11),
            mosaic_days=20, bands=["B04", "B08"],
        )


# --- spec 47 D1/D2: a cached input.csv must match the freshly tiled grids, by id set ---


def _write_grids_and_csv(tmp_path, cached_ids, fresh_ids):
    """Mirrors what `_run_inference_roi` itself writes: `cells/input.csv` with an `id`
    column (the cached work list) and a `grids` GeoDataFrame (the freshly tiled roi)."""
    run_folderpath = tmp_path / "cells"
    run_folderpath.mkdir()
    import pandas as pd

    pd.DataFrame({"id": cached_ids}).to_csv(run_folderpath / "input.csv", index=False)
    grids = gpd.GeoDataFrame(
        {"id": fresh_ids},
        geometry=[box(i, i, i + 1, i + 1) for i in range(len(fresh_ids))],
        crs="EPSG:4326",
    )
    return str(run_folderpath / "input.csv"), grids


def test_resume_identity_same_id_set_does_not_raise(tmp_path):
    from rise_eo.api import _check_resume_identity

    csv_filepath, grids = _write_grids_and_csv(tmp_path, ["a", "b", "c"], ["c", "b", "a"])
    _check_resume_identity(csv_filepath, grids, str(tmp_path))   # no raise


def test_resume_identity_disjoint_id_set_raises_naming_folder_counts_and_sample(tmp_path):
    from rise_eo.api import _check_resume_identity

    csv_filepath, grids = _write_grids_and_csv(tmp_path, ["a", "b", "c"], ["x", "y", "z"])
    with pytest.raises(rise_eo.PreflightError) as exc_info:
        _check_resume_identity(csv_filepath, grids, str(tmp_path))
    msg = str(exc_info.value)
    assert str(tmp_path) in msg
    assert "3 cell ids" in msg                       # both counts appear
    assert any(cid in msg for cid in ("a", "b", "c", "x", "y", "z"))   # a sample of the diff
    assert "new output_folderpath" in msg
    assert "spec-46" not in msg                       # not a drift case: fully disjoint


def test_resume_identity_small_superset_names_spec46_drift(tmp_path):
    """D2: cached ⊋ fresh by a handful of ids reads as the spec-46 D4 cell-count change
    (e.g. AT_ROI 300->299), not a different roi -- the message must say so."""
    from rise_eo.api import _check_resume_identity

    csv_filepath, grids = _write_grids_and_csv(tmp_path, ["a", "b", "c"], ["a", "b"])
    with pytest.raises(rise_eo.PreflightError, match="spec-46"):
        _check_resume_identity(csv_filepath, grids, str(tmp_path))


def test_roi_resume_raises_before_setup_when_cached_ids_differ(tmp_path, monkeypatch):
    """Integration: a stale input.csv from a DIFFERENT roi must not silently win (#66) --
    the raise must happen before `_create_datacube.setup` (and therefore before any node
    ever fans out on the stale work list)."""
    import rise_eo.api as api
    import rise_eo.grid as _grid_mod

    fresh = gpd.GeoDataFrame(
        {"id": ["fresh_a", "fresh_b"]}, geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(_grid_mod, "roi_to_s2_grids", lambda *a, **kw: fresh)
    monkeypatch.setattr(api, "_ensure_bundle", lambda *a, **kw: "bundle_path")
    monkeypatch.setattr(api._create_datacube, "setup",
                        lambda *a, **kw: pytest.fail("setup ran on a stale work list"))

    _write_grids_and_csv(tmp_path, ["stale_a", "stale_b", "stale_c"], [])

    with pytest.raises(rise_eo.PreflightError, match="output_folderpath"):
        rise_eo.run_inference(
            _Tiny(), output_folderpath=str(tmp_path), roi=ROI, catalog_filepath="sentinel-2-l2a/c.parquet",
            startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 11),
            mosaic_days=20, bands=["B04", "B08"],
        )


def test_roi_resume_refusal_does_not_touch_the_run_folder(tmp_path, monkeypatch):
    """The refusal must leave the folder exactly as it found it (review, 2026-08-20): the
    check used to run AFTER `grids.geojson` was overwritten and a bundle staged, so a
    refused resume left the old run folder describing the NEW roi while its
    `cells/input.csv` still described the old one."""
    import rise_eo.api as api
    import rise_eo.grid as _grid_mod

    fresh = gpd.GeoDataFrame(
        {"id": ["fresh_a"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326",
    )
    monkeypatch.setattr(_grid_mod, "roi_to_s2_grids", lambda *a, **kw: fresh)
    monkeypatch.setattr(api, "_ensure_bundle",
                        lambda *a, **kw: pytest.fail("a bundle was staged before the refusal"))

    _write_grids_and_csv(tmp_path, ["stale_a", "stale_b"], [])
    before = sorted(p.name for p in tmp_path.iterdir())

    with pytest.raises(rise_eo.PreflightError, match="output_folderpath"):
        rise_eo.run_inference(
            _Tiny(), output_folderpath=str(tmp_path), roi=ROI, catalog_filepath="sentinel-2-l2a/c.parquet",
            startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 11),
            mosaic_days=20, bands=["B04", "B08"],
        )

    assert sorted(p.name for p in tmp_path.iterdir()) == before
    assert not (tmp_path / "grids.geojson").exists()


def test_roi_resume_same_ids_skips_setup_and_dispatches(tmp_path, monkeypatch):
    """Re-running with the SAME roi (same id set) still resumes: `setup` is skipped and
    the runner is invoked, exactly as it was before D1/D2 (AC1)."""
    import rise_eo.api as api
    import rise_eo.grid as _grid_mod
    from rise_eo.workflows import runners as _runners

    fresh = gpd.GeoDataFrame(
        {"id": ["cell_a", "cell_b"]}, geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)],
        crs="EPSG:4326",
    )
    monkeypatch.setattr(_grid_mod, "roi_to_s2_grids", lambda *a, **kw: fresh)
    monkeypatch.setattr(api, "_ensure_bundle", lambda *a, **kw: "bundle_path")
    monkeypatch.setattr(api._create_datacube, "setup",
                        lambda *a, **kw: pytest.fail("setup ran despite a matching resume"))

    class _Result:
        returncode = 0

    dispatched = []
    monkeypatch.setattr(
        _runners, "run_local_inference",
        lambda *a, **kw: (dispatched.append((a, kw)), _Result())[1],
    )
    # Everything past dispatch (collect outputs / STAC / merge) is out of scope for D1/D2 --
    # stub it so this test isolates "did the resume skip setup and reach the runner".
    monkeypatch.setattr(api, "_existing_outputs", lambda *a, **kw: ["out.tif"])
    monkeypatch.setattr(api, "_finalize_outputs", lambda *a, **kw: "DONE")

    csv_filepath, _ = _write_grids_and_csv(tmp_path, ["cell_a", "cell_b"], [])
    import pandas as pd

    pd.DataFrame({
        "id": ["cell_a", "cell_b"],
        "export_folderpath": ["a", "b"],
        "shapefilepath": ["a.geojson", "b.geojson"],
    }).to_csv(csv_filepath, index=False)

    result = rise_eo.run_inference(
        _Tiny(), output_folderpath=str(tmp_path), roi=ROI, catalog_filepath="sentinel-2-l2a/c.parquet",
        startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 11),
        mosaic_days=20, bands=["B04", "B08"],
    )
    assert len(dispatched) == 1
    assert result == "DONE"


def test_roi_passes_in_memory_footprints_not_geometry_geojson_paths(tmp_path, monkeypatch):
    """AC2 (spec 57 D2), caller side: ROI mode hands `_finalize_outputs` the footprint it
    already holds (`grids`, `.buffer(0)`-ed the way `create_datacube.setup._prepare` does),
    keyed by output path — NOT the `shapefilepath` column, which would be one blob read per
    cell. The `cog_outputs_to_items` half of AC2 is covered in `test_catalog_stac.py`; this is
    the half that proves the driver never asks for the path form at all."""
    import pandas as pd
    from shapely.geometry import Polygon

    import rise_eo.api as api
    import rise_eo.grid as _grid_mod
    from rise_eo.workflows import runners as _runners

    # slanted, not a box: `.buffer(0)` on a box is a no-op, which would hide a missing buffer.
    cell_a = Polygon([(0.0, 0.0), (1.0, 0.2), (0.9, 1.0), (0.1, 0.8)])
    cell_b = Polygon([(2.0, 2.0), (3.0, 2.3), (2.8, 3.0), (2.1, 2.7)])
    fresh = gpd.GeoDataFrame(
        {"id": ["cell_a", "cell_b"]}, geometry=[cell_a, cell_b], crs="EPSG:4326",
    )
    monkeypatch.setattr(_grid_mod, "roi_to_s2_grids", lambda *a, **kw: fresh)
    monkeypatch.setattr(api, "_ensure_bundle", lambda *a, **kw: "bundle_path")
    monkeypatch.setattr(api._create_datacube, "setup", lambda *a, **kw: None)

    class _Result:
        returncode = 0

    monkeypatch.setattr(_runners, "run_local_inference", lambda *a, **kw: _Result())
    monkeypatch.setattr(api, "_existing_outputs", lambda paths, **kw: list(paths))

    captured = {}

    def _fake_finalize(*a, **kw):
        captured.update(kw)
        return "DONE"

    monkeypatch.setattr(api, "_finalize_outputs", _fake_finalize)

    csv_filepath, _ = _write_grids_and_csv(tmp_path, ["cell_a", "cell_b"], [])
    pd.DataFrame({
        "id": ["cell_a", "cell_b"],
        "export_folderpath": ["a", "b"],
        # present in the manifest, and deliberately unreadable: reading it would be the bug.
        "shapefilepath": ["/nope/a.geojson", "/nope/b.geojson"],
    }).to_csv(csv_filepath, index=False)

    rise_eo.run_inference(
        _Tiny(), output_folderpath=str(tmp_path), roi=ROI, catalog_filepath="sentinel-2-l2a/c.parquet",
        startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 11),
        mosaic_days=20, bands=["B04", "B08"],
    )

    geometries = captured["geometries"]
    import os as _os

    assert set(geometries) == {
        _os.path.join("a", "output.tif"), _os.path.join("b", "output.tif"),
    }
    for value in geometries.values():
        assert not isinstance(value, str)          # never a geometry.geojson path
    # mapping(), not .equals(): the Item's bytes are the coordinates, so this must guard
    # coordinate-level identity (ring order included), not topological equality.
    import shapely.geometry as _sgeom

    assert _sgeom.mapping(geometries[_os.path.join("a", "output.tif")]) == \
        _sgeom.mapping(cell_a.buffer(0))
    assert _sgeom.mapping(geometries[_os.path.join("b", "output.tif")]) == \
        _sgeom.mapping(cell_b.buffer(0))


# --- merge modes -------------------------------------------------------------

def _write_cog(path, epsg, x0, y0, val, size=8, res=10, nodata=255):
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1, dtype="uint8",
        crs=CRS.from_epsg(epsg), transform=from_origin(x0, y0, res, res), nodata=nodata,
    ) as d:
        d.write(np.full((1, size, size), val, dtype="uint8"))


def test_merge_strict_refuses_mixed_crs(tmp_path):
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write_cog(a, 32636, 500000, 1300000, 1)
    _write_cog(b, 32637, 400000, 1300000, 2)
    with pytest.raises(rise_eo.PreflightError, match="multiple CRS"):
        _merge_outputs([str(a), str(b)], str(tmp_path / "m.tif"), nodata=255)


def test_merge_reproject_to_dominant_zone(tmp_path):
    # two cells in 32636 (dominant) + one in 32637 -> reproject merge into 32636
    a, b, c = tmp_path / "a.tif", tmp_path / "b.tif", tmp_path / "c.tif"
    _write_cog(a, 32636, 500000, 1300000, 1)
    _write_cog(b, 32636, 500080, 1300000, 1)
    _write_cog(c, 32637, 400000, 1300000, 2)
    dst = tmp_path / "merged.tif"
    out = _merge_outputs([str(a), str(b), str(c)], str(dst), nodata=255,
                         reproject_to_dominant=True)
    with rasterio.open(out) as s:
        assert s.crs.to_epsg() == 32636                        # dominant zone
        assert s.count == 1 and s.nodata == 255


def test_merge_reproject_area_dominant_beats_count(tmp_path):
    """spec 23 D7: the target is the max-total-AREA zone, not the most-cells zone."""
    a, b, c = tmp_path / "a.tif", tmp_path / "b.tif", tmp_path / "c.tif"
    _write_cog(a, 32636, 500000, 1300000, 1, size=4)          # two small cells (count favours 36)
    _write_cog(b, 32636, 500040, 1300000, 1, size=4)
    _write_cog(c, 32637, 400000, 1300000, 2, size=40)         # one big cell (area favours 37)
    out = _merge_outputs([str(a), str(b), str(c)], str(tmp_path / "m.tif"), nodata=255,
                         reproject_to_dominant=True)
    with rasterio.open(out) as s:
        assert s.crs.to_epsg() == 32637                        # area wins over cell count


def test_merge_strict_prints_progress_per_input(tmp_path, capsys):
    """D5: `_merge_outputs`/`_merge_mosaic` ticks per input opened -- the WAN-latency-
    bound part (every per-cell COG read over /vsiadls/, measured ~1000s on 300 cells)."""
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write_cog(a, 32636, 500000, 1300000, 1)
    _write_cog(b, 32636, 500080, 1300000, 1)
    _merge_outputs([str(a), str(b)], str(tmp_path / "m.tif"), nodata=255)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[merge] ")]
    assert any(ln.startswith("[merge] 0/2 inputs") for ln in lines)
    assert any(ln.startswith("[merge] 2/2 inputs") for ln in lines)
    assert any("inputs/s" in ln for ln in lines)


def test_merge_reproject_prints_progress_per_input(tmp_path, capsys):
    a, b, c = tmp_path / "a.tif", tmp_path / "b.tif", tmp_path / "c.tif"
    _write_cog(a, 32636, 500000, 1300000, 1)
    _write_cog(b, 32636, 500080, 1300000, 1)
    _write_cog(c, 32637, 400000, 1300000, 2)
    _merge_outputs([str(a), str(b), str(c)], str(tmp_path / "m.tif"), nodata=255,
                   reproject_to_dominant=True)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[merge] ")]
    assert any(ln.startswith("[merge] 0/3 inputs") for ln in lines)
    # The per-input WARP is the expensive phase, and it used to run in total silence AFTER
    # the header-open scan had already printed 100% (review, 2026-08-20).
    assert any(ln.startswith("[merge] 3/3 inputs reprojected") for ln in lines)
    assert any(ln.startswith("[merge] merging 3 inputs into the mosaic") for ln in lines)


def test_merge_strict_announces_the_pixel_read_phase(tmp_path, capsys):
    """`rio_merge` is where the pixels are actually read and it has no per-input hook to
    tick from, so it must at least be announced -- otherwise the 100% line above it is
    followed by unexplained silence for the bulk of the wall clock."""
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write_cog(a, 32636, 500000, 1300000, 1)
    _write_cog(b, 32636, 500080, 1300000, 1)
    _merge_outputs([str(a), str(b)], str(tmp_path / "m.tif"), nodata=255)
    out = capsys.readouterr().out
    assert "[merge] merging 2 inputs into the mosaic" in out


def test_merge_uses_one_env_for_all_inputs_not_one_per_file(tmp_path, monkeypatch):
    """Two bugs in one pin, both found on real blob data (run-book 38 Phase 4, 2026-07-28).

    1. Reads must be VSI-translated (`to_vsi`) -- the outputs may live on `abfss://`, which GDAL
       has no driver for. Bare `rasterio.open(fp)` was the 5th instance of that class here.
    2. But NOT via `rio_open` per file: it owns a `rasterio.Env` per handle, and merge holds
       every input open at once. rasterio's env stack is LIFO, so closing N of them in creation
       order tears down the root env first and the next close raises
       `EnvError: No GDAL environment exists`. ONE `rio_env` must cover the whole merge.
    """
    import rise_eo.raster as _raster
    import rise_eo.storage.azure as _az

    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write_cog(a, 32636, 500000, 1300000, 1)
    _write_cog(b, 32636, 500080, 1300000, 1)

    envs, translated, opens = [], [], []
    real_env, real_vsi, real_open = _raster.rio_env, _az.to_vsi, _raster.rio_open
    monkeypatch.setattr(_raster, "rio_env",
                        lambda paths: (envs.append(list(paths)), real_env(paths))[1])
    monkeypatch.setattr(_az, "to_vsi", lambda p: (translated.append(str(p)), real_vsi(p))[1])
    monkeypatch.setattr(_raster, "rio_open",
                        lambda *a_, **k: (opens.append(a_[0]), real_open(*a_, **k))[1])

    for reproject in (False, True):
        envs.clear()
        translated.clear()
        opens.clear()
        _merge_outputs([str(a), str(b)], str(tmp_path / f"m{reproject}.tif"), nodata=255,
                       reproject_to_dominant=reproject)
        assert len(envs) == 1, f"expected ONE env for the whole merge, got {len(envs)}"
        assert set(envs[0]) == {str(a), str(b)}, "the env must cover every input"
        assert {str(a), str(b)} <= set(translated), "a source was opened without to_vsi"
        assert opens == [], "rio_open owns an Env per handle -- unusable for an N-way merge"


def test_merge_reproject_scratch_stays_local(tmp_path):
    """The reprojection temp must go to local scratch, never `f"{fp}.reproj.tif"` next to the
    source -- a remote source would put it on the remote URL, which rasterio cannot open for
    WRITE (D5 / ADR-0001). Pinned by asserting nothing is left beside the sources."""
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write_cog(a, 32636, 500000, 1300000, 1)
    _write_cog(b, 32637, 400000, 1300000, 2)          # forces a reprojection
    _merge_outputs([str(a), str(b)], str(tmp_path / "m.tif"), nodata=255,
                   reproject_to_dominant=True)
    assert not list(tmp_path.glob("*.reproj.tif"))


def test_merge_reproject_merge_crs_override(tmp_path):
    """spec 23 D7: merge_crs forces the target CRS regardless of area/count."""
    a, b = tmp_path / "a.tif", tmp_path / "b.tif"
    _write_cog(a, 32636, 500000, 1300000, 1, size=4)
    _write_cog(b, 32637, 400000, 1300000, 2, size=40)         # bigger, would be area-dominant
    out = _merge_outputs([str(a), str(b)], str(tmp_path / "m.tif"), nodata=255,
                         reproject_to_dominant=True, merge_crs=32636)
    with rasterio.open(out) as s:
        assert s.crs.to_epsg() == 32636                        # forced target


# --- TODO #61: the collect is one listing, not one exists() per cell -------------------

def _cell_out(root, window, cell_id):
    import os

    return os.path.join(str(root), window, cell_id, "output.tif")


def test_output_key_is_scheme_independent():
    """The reason the collect can use `fs.glob` at all: a globbed hit and a caller-built
    url never string-equal on blob (adlfs drops the `abfss://` scheme), but their trailing
    `<window>/<cell>/output.tif` do match."""
    from rise_eo.api import _output_key

    url = "abfss://data@acct.dfs.core.windows.net/pfx/out/cells/20180406_20180928/165b09c/output.tif"
    globbed = "data/pfx/out/cells/20180406_20180928/165b09c/output.tif"
    local = "/tmp/out/cells/20180406_20180928/165b09c/output.tif"
    assert _output_key(url) == _output_key(globbed) == _output_key(local)
    assert _output_key(url) == "20180406_20180928/165b09c/output.tif"


def test_existing_outputs_uses_one_listing_and_keeps_order(tmp_path, monkeypatch):
    """One `fs.glob` for the whole run, zero `fs.exists` — the TODO #61 fix. Order must
    still follow the caller's candidate list, as the comprehension it replaced did."""
    from rise_eo import api as _api

    run = tmp_path / "cells"
    window = "20180406_20180928"
    for cid in ("aaa", "ccc"):                       # 'bbb' never produced an output
        d = run / window / cid
        d.mkdir(parents=True)
        (d / "output.tif").write_bytes(b"x")

    calls = {"glob": 0, "exists": 0}
    real_glob = _api.fs.glob
    monkeypatch.setattr(_api.fs, "glob",
                        lambda p, **kw: (calls.__setitem__("glob", calls["glob"] + 1),
                                         real_glob(p, **kw))[1])
    monkeypatch.setattr(_api.fs, "exists",
                        lambda p, **kw: calls.__setitem__("exists", calls["exists"] + 1))

    candidates = [_cell_out(run, window, c) for c in ("ccc", "bbb", "aaa")]
    got = _api._existing_outputs(candidates, run_folderpath=str(run))

    assert got == [_cell_out(run, window, "ccc"), _cell_out(run, window, "aaa")]
    assert calls == {"glob": 1, "exists": 0}          # ONE round trip, regardless of N


def test_existing_outputs_empty_when_nothing_matches(tmp_path):
    """A pattern that matches nothing returns empty so the caller raises its own loud
    'no per-cell outputs were produced' — never a silent success."""
    from rise_eo.api import _existing_outputs

    run = tmp_path / "cells"
    run.mkdir()
    assert _existing_outputs([_cell_out(run, "w", "id")], run_folderpath=str(run)) == []


def test_existing_outputs_prints_progress_before_and_after(tmp_path, capsys):
    """D5: collect has no per-candidate loop left to tick against (TODO #61 already
    collapsed it to one `fs.glob`), so it prints before/after in the same shape rather
    than inventing a per-item loop that no longer exists."""
    from rise_eo.api import _existing_outputs

    run = tmp_path / "cells"
    window = "20180406_20180928"
    for cid in ("aaa", "ccc"):
        d = run / window / cid
        d.mkdir(parents=True)
        (d / "output.tif").write_bytes(b"x")

    candidates = [_cell_out(run, window, c) for c in ("aaa", "bbb", "ccc")]
    _existing_outputs(candidates, run_folderpath=str(run))
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.startswith("[collect] ")]
    assert any(ln.startswith("[collect] 0/3 candidates") for ln in lines)
    # `done` is candidates PROBED (all 3), not outputs found -- one glob finishes them all,
    # so reporting the hit count as progress made a finished collect read as "2/3, 67%".
    # The hit count rides as the suffix instead (review, 2026-08-20).
    final = [ln for ln in lines if ln.startswith("[collect] 3/3 candidates")]
    assert final, lines
    assert final[-1].endswith("| 2 already have an output.tif")
    assert "eta" not in final[-1]        # nothing to extrapolate a rate from


@pytest.mark.parametrize("runner", ["local", "aml"])
def test_run_inference_roi_forwards_options_to_tiling_runner_and_finalize(
    tmp_path, monkeypatch, runner,
):
    """D10 (#145): `grid_size_km`/`scale_fact` reach the tiler; `skip_nan`,
    `predict_batch_size` and `overwrite` reach the runner (local and AML); `collection_id`
    and `dt` reach `_finalize_outputs`."""
    import pandas as pd

    import rise_eo.api as api
    import rise_eo.grid as _grid_mod
    from rise_eo.workflows import runners as _runners

    fresh = gpd.GeoDataFrame({"id": ["cell_a"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    dt = datetime.datetime(2018, 6, 1, tzinfo=datetime.timezone.utc)  # run_inference makes dt UTC
    tiled, ran, finalized = {}, {}, {}
    monkeypatch.setattr(_grid_mod, "roi_to_s2_grids",
                        lambda *a, **kw: tiled.update(kw) or fresh)
    monkeypatch.setattr(api, "_ensure_bundle", lambda *a, **kw: "bundle_path")
    monkeypatch.setattr(api, "_configure_storage", lambda *a, **kw: None)  # no global fsspec leak
    monkeypatch.setattr(api._create_datacube, "setup", lambda *a, **kw: None)

    class _Result:
        returncode = 0

    monkeypatch.setattr(_runners, "run_local_inference",
                        lambda *a, **kw: ran.update(kw) or _Result())
    monkeypatch.setattr(_runners, "run_aml_inference", lambda *a, **kw: ran.update(kw))
    monkeypatch.setattr(api, "_existing_outputs", lambda paths, **kw: list(paths))
    monkeypatch.setattr(api, "_finalize_outputs",
                        lambda *a, **kw: finalized.update(collection_id=a[4], dt=a[5]) or "DONE")

    csv_filepath, _ = _write_grids_and_csv(tmp_path, ["cell_a"], [])
    pd.DataFrame({"id": ["cell_a"], "export_folderpath": ["a"],
                  "shapefilepath": ["/nope/a.geojson"]}).to_csv(csv_filepath, index=False)

    extra = {"runner": "aml", "storage": "azure",
             "runner_kwargs": {"cluster": "c", "environment": "e:1", "root": "memory://r",
                               "identity_client_id": "id"}} if runner == "aml" else {}
    rise_eo.run_inference(
        _Tiny(), output_folderpath=str(tmp_path), roi=ROI,
        catalog_filepath="sentinel-2-l2a/c.parquet",
        startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 7, 11),
        mosaic_days=20, bands=["B04", "B08"], grid_size_km=7, scale_fact=1.3,
        skip_nan=False, predict_batch_size=16, overwrite=True,
        collection_id="my-coll", dt=dt, **extra,
    )
    assert (tiled["grid_size_km"], tiled["scale_fact"]) == (7, 1.3)
    assert (ran["skip_nan"], ran["predict_batch_size"], ran["overwrite"]) == (False, 16, True)
    assert finalized == {"collection_id": "my-coll", "dt": dt}


# --- #145 P10: alias spelling of a band is not a missing band ---

@pytest.mark.parametrize("required, bands", [(["B8A"], ["nir08"]), (["nir08"], ["B8A"])])
def test_roi_preflight_alias_spelling_is_not_a_missing_band(tmp_path, required, bands):
    class _Alias(_Tiny):
        required_bands = required

    with pytest.raises(rise_eo.PreflightError) as ei:
        rise_eo.run_inference(
            _Alias(), output_folderpath=str(tmp_path), roi=ROI,
            catalog_filepath="sentinel-2-l2a/c.parquet",
            startdate=datetime.datetime(2018, 6, 1), enddate=datetime.datetime(2018, 6, 11),
            mosaic_days=20, bands=bands,
        )
    assert str(ei.value) == (
        "preflight failed:\n"
        "  - startdate/enddate/mosaic_days give T=1 but the model needs T=2."
    )
