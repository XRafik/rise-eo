"""Spec 53 (#89) — a blob-resolved model ref cannot be loaded on the local run path.

`bundle.load` needs a local directory (`sys.path` cannot hold a URL); `run_inference`'s local
shapes (`cores=1`, the `cores>1` Snakemake fan-out, local ROI mode) never staged a non-local
resolved bundle before handing it to `load`. D1 stages once, right after preflight passes, into
`<output_folderpath>/_model` (D2); D1's amendment gates staging on `runner == "local"` so the
AML path (which stages its own copy per-node) is untouched in both directions (AC7).

`memory://` stands in for blob, per spec 52 §5 / spec 53 §8 (no network in unit tests). The
adapter below is defined in this test module (a local, non-installed module) so `bundle.save`
embeds its source under `code/` -- the same `code/`-on-`sys.path` shape #89 hit. Note this does
NOT reproduce #89's `ModuleNotFoundError` in-process: the test module is already in
`sys.modules` by the time `bundle.load` runs, so `importlib.import_module` finds the cached
module regardless of `sys.path` -- reproducing the crash itself needs a genuinely fresh
interpreter (the run-book's real-Azure repro). What these tests verify instead, and what makes
them non-vacuous (checked by commenting out each call site and confirming the assertion below
then fails): staging actually runs, exactly once per run, lands at `<output_folderpath>/_model`,
and every local/AML call site sees the path D1 says it should.
"""

from __future__ import annotations

import os
import types

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from rise_eo import api
from rise_eo.model import BaseModelAdapter, bundle, registry
from rise_eo.storage import fs
from rise_eo.workflows import infer_shard, runners

BANDS = ["B04", "B08"]
EPSG = 32637


class _StageProbeAdapter(BaseModelAdapter):
    """Trivial classifier; the only thing that matters is that its module is embedded code,
    not an installed package (see module docstring)."""

    required_bands = BANDS
    n_timestamps = 1
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["cls"]
    feature_sequence = []

    def load(self):
        pass

    def predict(self, X_chunk):
        return np.ones(X_chunk.shape[0], dtype="uint8")


def _make_bundle(tmp_path) -> str:
    return bundle.save(_StageProbeAdapter(), {}, str(tmp_path / "src_bundle"), verbose=False)


def _make_datacube_folder(folder, *, name="cell") -> str:
    """Returns `datacube.npy`'s filepath. `_resolve_inference_pairs` derives the output path
    itself (`<output_folderpath>/<dirname-of-dc>/output.tif`) -- it takes a list of datacube
    filepaths, not `(in, out)` pairs."""
    cell_dir = os.path.join(folder, name)
    os.makedirs(cell_dir, exist_ok=True)
    dc = np.zeros((1, 4, 4, len(BANDS)), dtype=np.uint16)
    dc[..., 0], dc[..., 1] = 2000, 8000   # NDVI > 0 everywhere -- no NaN pixels to skip
    dc_fp = os.path.join(cell_dir, "datacube.npy")
    fs.save_npy(dc_fp, dc)
    md = {
        "bands": BANDS,
        "timestamps": [0],
        "geotiff_metadata": {
            "width": 4, "height": 4,
            "transform": from_origin(500000, 4000000, 10, 10),
            "crs": CRS.from_epsg(EPSG),
        },
    }
    fs.save_npy(os.path.join(cell_dir, "metadata.pickle.npy"), md, allow_pickle=True)
    return dc_fp


def _expected_output(output_folderpath, dc_fp) -> str:
    stem = os.path.basename(os.path.dirname(dc_fp))
    return os.path.join(str(output_folderpath), stem, "output.tif")


def _write_output_cog(path):
    """A minimal valid COG at `path` -- stands in for a real per-cube/per-cell inference
    output when the runner itself is faked out (mirrors `test_infer_aml._write_local_cog`)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=4, width=4, count=1,
                       dtype="uint8", crs=CRS.from_epsg(EPSG),
                       transform=from_origin(500000, 4000000, 10, 10), nodata=255) as dst:
        dst.write(np.ones((1, 4, 4), dtype=np.uint8))


def _publish_to_blob_registry(tmp_path, *, alias="champion") -> tuple[str, str]:
    """Publish `_StageProbeAdapter`'s bundle to a `memory://` registry, returns
    `(registry_root, ref)`."""
    src = _make_bundle(tmp_path)
    registry_root = f"memory://{tmp_path.name}-registry"
    registry.publish(src, "probe", registry_root, alias=alias)
    return registry_root, "probe@" + alias


# --- AC1: resolves, stages, loads, runs — the exact call that raised ModuleNotFoundError -----


def test_ac1_blob_registry_ref_runs_on_the_local_cores1_path(tmp_path):
    registry_root, ref = _publish_to_blob_registry(tmp_path)
    dc_fp = _make_datacube_folder(str(tmp_path / "cubes"))
    out_folder = str(tmp_path / "out")

    result = api.run_inference(
        model=ref, registry=registry_root,
        inference_datacubes=[dc_fp],
        output_folderpath=out_folder,
    )

    out_fp = _expected_output(out_folder, dc_fp)
    assert fs.exists(out_fp)
    assert len(result.output_filepaths) == 1
    staged = os.path.join(out_folder, "_model")
    assert fs.exists(os.path.join(staged, "bundle.json"))   # D2: scratch under the run's own output


# --- AC2: fetched once per run, not once per datacube ------------------------------------


def test_ac2_bundle_fetched_once_for_a_two_cube_run(tmp_path, monkeypatch):
    registry_root, ref = _publish_to_blob_registry(tmp_path)
    dc_fp1 = _make_datacube_folder(str(tmp_path / "cubes"), name="cell_a")
    dc_fp2 = _make_datacube_folder(str(tmp_path / "cubes"), name="cell_b")

    calls = []
    real_fetch = infer_shard.fetch_bundle_to_scratch

    def spy(*a, **kw):
        calls.append(a)
        return real_fetch(*a, **kw)

    monkeypatch.setattr(infer_shard, "fetch_bundle_to_scratch", spy)

    api.run_inference(
        model=ref, registry=registry_root,
        inference_datacubes=[dc_fp1, dc_fp2],
        output_folderpath=str(tmp_path / "out"),
    )

    assert len(calls) == 1


# --- AC3: a local registry path is unchanged — no fetch, no copy, no new directory -------


def test_ac3_local_bundle_path_is_not_staged(tmp_path, monkeypatch):
    bundle_dir = _make_bundle(tmp_path)
    dc_fp = _make_datacube_folder(str(tmp_path / "cubes"))

    def _boom(*a, **kw):
        raise AssertionError("fetch_bundle_to_scratch must not be called for a local path")

    monkeypatch.setattr(infer_shard, "fetch_bundle_to_scratch", _boom)

    api.run_inference(
        model=bundle_dir, inference_datacubes=[dc_fp],
        output_folderpath=str(tmp_path / "out"),
    )

    assert not os.path.exists(str(tmp_path / "out" / "_model"))


# --- AC4: a call rejected in preflight performs no transfer -------------------------------


def test_ac4_preflight_rejection_performs_no_transfer(tmp_path, monkeypatch):
    registry_root, ref = _publish_to_blob_registry(tmp_path)
    empty_cubes = tmp_path / "no_cubes_here"
    empty_cubes.mkdir()

    def _boom(*a, **kw):
        raise AssertionError("fetch_bundle_to_scratch must not run before preflight passes")

    monkeypatch.setattr(infer_shard, "fetch_bundle_to_scratch", _boom)

    with pytest.raises(api.PreflightError, match="no inference datacubes"):
        api.run_inference(
            model=ref, registry=registry_root,
            inference_datacubes=str(empty_cubes),
            output_folderpath=str(tmp_path / "out"),
        )


# --- AC5: the staged bundle holds exactly the manifest-declared files --------------------


def test_ac5_staged_bundle_holds_exactly_the_manifest_declared_files(tmp_path):
    registry_root, ref = _publish_to_blob_registry(tmp_path)
    dc_fp = _make_datacube_folder(str(tmp_path / "cubes"))
    out_folder = str(tmp_path / "out")

    api.run_inference(
        model=ref, registry=registry_root,
        inference_datacubes=[dc_fp],
        output_folderpath=out_folder,
    )

    staged = os.path.join(out_folder, "_model")
    import json
    with open(os.path.join(staged, "bundle.json")) as f:
        manifest = json.load(f)
    from rise_eo.model import bundle as _bundle
    expected = {_bundle.BUNDLE_MANIFEST} | set(manifest.get("artifacts", {}).values()) \
        | set(_bundle.manifest_code_files(manifest))
    got = set()
    for root, _dirs, files in os.walk(staged):
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), staged)
            got.add(rel.replace(os.sep, "/"))
    assert got == expected


# --- AC6: cores>1 fan-out uses the staged local path, not the original URL ---------------


def test_ac6_cores_gt_1_fan_out_uses_the_staged_local_path(tmp_path, monkeypatch):
    registry_root, ref = _publish_to_blob_registry(tmp_path)
    dc_fp = _make_datacube_folder(str(tmp_path / "cubes"))

    captured = {}

    def _fake_run_local_infer_only(csv_fp, *, cores, bundle_path, cubes_per_task, **kw):
        captured["bundle_path"] = bundle_path
        with fs.open(csv_fp, "r") as f:
            import pandas as pd
            for out in pd.read_csv(f)["output_filepath"]:
                _write_output_cog(out)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runners, "run_local_infer_only", _fake_run_local_infer_only)

    api.run_inference(
        model=ref, registry=registry_root,
        inference_datacubes=[dc_fp],
        output_folderpath=str(tmp_path / "out"),
        cores=2,
    )

    assert not captured["bundle_path"].startswith("memory://")
    assert captured["bundle_path"] == os.path.join(str(tmp_path / "out"), "_model")


# --- AC7: the AML path is untouched, in both directions -----------------------------------


def test_ac7_roi_mode_runner_aml_performs_zero_local_fetches(tmp_path, monkeypatch):
    """A blob registry + `runner='aml'` must NOT stage on the driver -- the node stages its
    own copy (`_stage_bundle` -> `infer_shard.fetch_bundle_to_scratch`, exercised elsewhere).
    Asserted by call count, not by inspecting timings (AC7)."""
    import geopandas as gpd
    from shapely.geometry import box

    registry_root, ref = _publish_to_blob_registry(tmp_path)

    def _boom(*a, **kw):
        raise AssertionError("run_inference must not fetch a bundle to the driver for runner=aml")

    monkeypatch.setattr(infer_shard, "fetch_bundle_to_scratch", _boom)
    monkeypatch.setattr(
        "rise_eo.grid.roi_to_s2_grids",
        lambda roi, **kw: gpd.GeoDataFrame(
            {"id": ["s1"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326",
        ),
    )

    aml_calls = {}

    def _fake_run_aml_inference(input_csv, bundle_path, **kw):
        aml_calls["bundle_path"] = bundle_path
        import pandas as pd
        with fs.open(input_csv, "r") as f:
            for exp in pd.read_csv(f)["export_folderpath"]:
                _write_output_cog(f"{exp}/output.tif")
        return {"run_id": "r"}

    monkeypatch.setattr(runners, "run_aml_inference", _fake_run_aml_inference)

    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    import pandas as pd

    from rise_eo.catalog import declaration as declaration_module
    from rise_eo.catalog.declaration import S2_L2A_DECLARATION

    gdf = gpd.GeoDataFrame(
        [{"id": "T_0", "acquisition_key": "T_0", "processing_version": None, "processing_datetime": None, "source": "mpc", "satellite": "sentinel-2-l2a", "timestamp": pd.Timestamp("2018-06-01", tz="UTC"),
          "s3url": "s3://x", "local_folderpath": str(tmp_path), "files": "B04.tif,B08.tif",
          "cloud_cover": 0.0, "geometry": box(0, 0, 1, 1), "area_contribution": 100.0}],
        crs="EPSG:4326",
    )
    declaration_module.to_attrs(gdf, S2_L2A_DECLARATION)
    fs.write_parquet(str(cat), gdf)

    roi_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    api.run_inference(
        model=ref, registry=registry_root, roi=roi_gdf,
        output_folderpath=str(tmp_path / "out"),
        catalog_filepath=str(cat), startdate="2018-06-01", enddate="2018-06-11",
        mosaic_days=20, bands=BANDS, runner="aml",
        runner_kwargs={"cluster": "c", "environment": "e:1", "root": "memory://r",
                       "identity_client_id": "id"},
        storage="azure",
    )

    # the resolved (still-blob) path went straight to the AML dispatch, unstaged.
    assert aml_calls["bundle_path"].startswith(registry_root)


def test_ac7_roi_mode_runner_local_still_stages(tmp_path, monkeypatch):
    """The other direction of AC7: local ROI mode is one of the shapes that DOES stage."""
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    registry_root, ref = _publish_to_blob_registry(tmp_path)

    monkeypatch.setattr(
        "rise_eo.grid.roi_to_s2_grids",
        lambda roi, **kw: gpd.GeoDataFrame(
            {"id": ["s1"], "geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326",
        ),
    )

    local_calls = {}

    def _fake_run_local_inference(csv_fp, *, cores, bundle_path, cubes_per_task, **kw):
        local_calls["bundle_path"] = bundle_path
        with fs.open(csv_fp, "r") as f:
            for exp in pd.read_csv(f)["export_folderpath"]:
                _write_output_cog(f"{exp}/output.tif")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runners, "run_local_inference", _fake_run_local_inference)

    cat = tmp_path / "sentinel-2-l2a" / "catalog.parquet"
    from rise_eo.catalog import declaration as declaration_module
    from rise_eo.catalog.declaration import S2_L2A_DECLARATION

    gdf = gpd.GeoDataFrame(
        [{"id": "T_0", "acquisition_key": "T_0", "processing_version": None, "processing_datetime": None, "source": "mpc", "satellite": "sentinel-2-l2a", "timestamp": pd.Timestamp("2018-06-01", tz="UTC"),
          "s3url": "s3://x", "local_folderpath": str(tmp_path), "files": "B04.tif,B08.tif",
          "cloud_cover": 0.0, "geometry": box(0, 0, 1, 1), "area_contribution": 100.0}],
        crs="EPSG:4326",
    )
    declaration_module.to_attrs(gdf, S2_L2A_DECLARATION)
    fs.write_parquet(str(cat), gdf)

    roi_gdf = gpd.GeoDataFrame({"geometry": [box(0, 0, 1, 1)]}, crs="EPSG:4326")
    api.run_inference(
        model=ref, registry=registry_root, roi=roi_gdf,
        output_folderpath=str(tmp_path / "out"),
        catalog_filepath=str(cat), startdate="2018-06-01", enddate="2018-06-11",
        mosaic_days=20, bands=BANDS,
    )

    assert not local_calls["bundle_path"].startswith("memory://")
    assert local_calls["bundle_path"] == os.path.join(str(tmp_path / "out"), "_model")


# --- Opus review (2026-08-25): the staging transfer must not be silent --------------------


def test_staging_announces_destination_and_size_before_the_transfer(tmp_path, capsys):
    """Spec 47 D5 for the leg spec 53 adds. D2's rationale said the transfer "is never
    silent" because `_stage_bundle` prints -- but that is the UPLOAD leg;
    `fetch_bundle_to_scratch` prints nothing, so before this the driver went quiet between
    `[model] ... -> vN` and the first `[inference]` line for the whole download."""
    registry_root, ref = _publish_to_blob_registry(tmp_path)
    dc_fp = _make_datacube_folder(str(tmp_path / "cubes"))

    api.run_inference(
        model=ref, registry=registry_root,
        inference_datacubes=[dc_fp],
        output_folderpath=str(tmp_path / "out"),
    )

    lines = capsys.readouterr().out.splitlines()
    staged = [i for i, ln in enumerate(lines) if ln.startswith("[stage] bundle <- ")]
    assert len(staged) == 1, f"expected exactly one staging line, got {lines}"
    assert registry_root in lines[staged[0]] and "MB" in lines[staged[0]]
    inferred = [i for i, ln in enumerate(lines) if ln.startswith("[inference]")]
    assert inferred and staged[0] < inferred[0], "the size must be printed BEFORE the transfer"
