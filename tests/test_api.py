"""Tests for the high-level API façade (specs/16).

Fast + synthetic: preflight, seam guards, stubs, and the orchestration wiring of
`create_training_data` (with the heavy build/flatten monkeypatched). The real end-to-end
against downloaded tiles is a manual runbook (tests/manual/flatten.md), not a unit test.
"""

from __future__ import annotations

import datetime
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely.geometry

import fsd
from fsd import api

JAN1 = datetime.datetime(2018, 1, 1)
JAN1_NEXT = datetime.datetime(2019, 1, 1)


def _polys(tmp_path, id_col="fid", label_col="crop", n=2):
    gdf = gpd.GeoDataFrame(
        {
            id_col: list(range(n)),
            label_col: ["a", "b"][:n],
            "geometry": [shapely.geometry.box(i, 0, i + 1, 1) for i in range(n)],
        },
        crs="EPSG:4326",
    )
    return gdf


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    return str(path)


# --- compute_n_timestamps ----------------------------------------------------

def test_compute_n_timestamps_calendar_identity():
    # full 2018, 20-day windows -> ceil(365/20) = 19 (matches the spec-15 benchmark).
    assert fsd.compute_n_timestamps(JAN1, JAN1_NEXT, 20) == 19
    # exact multiple: Jan 1 -> Feb 10 is 40 days, /10 = 4 windows.
    assert fsd.compute_n_timestamps(JAN1, datetime.datetime(2018, 2, 10), 10) == 4
    # one short window.
    assert fsd.compute_n_timestamps(JAN1, datetime.datetime(2018, 1, 6), 20) == 1


# --- preflight ---------------------------------------------------------------

def test_preflight_rejects_bad_window_and_makes_nothing(tmp_path):
    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    export = tmp_path / "export"
    with pytest.raises(api.PreflightError):
        fsd.create_training_data(
            label_polygons=_polys(tmp_path), catalog_filepath=cat,
            startdate=JAN1_NEXT, enddate=JAN1,  # reversed
            mosaic_days=0, bands=[],            # bad mosaic + empty bands
            id_col="fid", label_col="crop", export_folderpath=str(export),
        )
    assert not export.exists()  # failed before any work


def test_preflight_missing_catalog(tmp_path):
    with pytest.raises(api.PreflightError, match="catalog_filepath"):
        fsd.create_training_data(
            label_polygons=_polys(tmp_path),
            catalog_filepath=str(tmp_path / "nope.parquet"),
            startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=["B04"],
            id_col="fid", label_col="crop", export_folderpath=str(tmp_path / "e"),
        )


def test_preflight_missing_columns(tmp_path):
    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    with pytest.raises(api.PreflightError, match="not in label_polygons"):
        fsd.create_training_data(
            label_polygons=_polys(tmp_path), catalog_filepath=cat,
            startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=["B04"],
            id_col="MISSING", label_col="crop", export_folderpath=str(tmp_path / "e"),
        )


# --- seam guards -------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [{"runner": "batch"}, {"storage": object()}])
def test_seam_guard_local_only(tmp_path, kwargs):
    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    with pytest.raises(api.PreflightError):
        fsd.create_training_data(
            label_polygons=_polys(tmp_path), catalog_filepath=cat,
            startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=["B04"],
            id_col="fid", label_col="crop", export_folderpath=str(tmp_path / "e"),
            **kwargs,
        )


# --- feature / aggregate wiring guards (P0.5) --------------------------------

def test_adapter_and_feature_sequence_conflict(tmp_path):
    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    with pytest.raises(api.PreflightError, match="not both"):
        fsd.create_training_data(
            label_polygons=_polys(tmp_path), catalog_filepath=cat,
            startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=["B04"],
            id_col="fid", label_col="crop", export_folderpath=str(tmp_path / "e"),
            adapter=object(), feature_sequence=[("x", {})],
        )


def test_unknown_aggregate_rejected(tmp_path):
    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    with pytest.raises(api.PreflightError, match="aggregate"):
        fsd.create_training_data(
            label_polygons=_polys(tmp_path), catalog_filepath=cat,
            startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=["B04"],
            id_col="fid", label_col="crop", export_folderpath=str(tmp_path / "e"),
            aggregate="nope",
        )


def _flattened_arrays(tmp_path, *, n_bands=2):
    """The flatten output `_apply_training_features` consumes: 6 pixels across 2 field
    ids, `(pixels, T, B)`."""
    export = tmp_path / "export"
    export.mkdir()
    ids = np.array(["fieldA", "fieldA", "fieldA", "fieldB", "fieldB", "fieldB"])
    labels = np.array(["wheat", "wheat", "wheat", "maize", "maize", "maize"])
    # fieldA's per-pixel values are 10/20/30 -> median 20; fieldB's are 100/200/300 -> 200.
    base = np.array([10.0, 20.0, 30.0, 100.0, 200.0, 300.0])
    data = np.repeat(base[:, None, None], 3, axis=1).repeat(n_bands, axis=2)  # (6, T=3, B)
    np.save(export / "data.npy", data)
    np.save(export / "ids.npy", ids)
    np.save(export / "labels.npy", labels)
    return str(export)


def test_apply_training_features_reduces_to_one_row_per_id(tmp_path):
    """The wiring both demo scripts depend on (spec 40 D1 amendment A2): `median_per_id`
    collapses per-pixel rows to one `np.nanmedian` row per labelled field BEFORE the
    feature transform, and `feature_ids`/`feature_labels` follow it. Without this, the
    demos train per-pixel on field-level labels — a field's own pixels leak across the
    train/test split, which is the whole reason the aggregate exists."""
    export = _flattened_arrays(tmp_path)
    metadata = {"bands": ["B04", "B08"]}

    api._apply_training_features(export, metadata, adapter=None,
                                 feature_sequence=[], aggregate="median_per_id")

    feats = np.load(os.path.join(export, "features.npy"))
    fids = np.load(os.path.join(export, "feature_ids.npy"))
    flabels = np.load(os.path.join(export, "feature_labels.npy"))

    assert feats.shape[0] == 2, "one row per field id, not per pixel"
    assert list(fids) == ["fieldA", "fieldB"]
    assert list(flabels) == ["wheat", "maize"]          # label by first occurrence
    assert feats[0, 0, 0] == pytest.approx(20.0)        # median(10, 20, 30)
    assert feats[1, 0, 0] == pytest.approx(200.0)       # median(100, 200, 300)

    written = np.load(os.path.join(export, "metadata.pickle.npy"), allow_pickle=True).item()
    assert written["aggregate"] == "median_per_id"      # recorded, so a reader knows the unit


def test_apply_training_features_keeps_every_pixel_when_aggregate_is_none(tmp_path):
    """The contrast that makes the test above meaningful: same input, no aggregate ->
    6 per-pixel rows and no `aggregate` stamp."""
    export = _flattened_arrays(tmp_path)
    api._apply_training_features(export, {"bands": ["B04", "B08"]}, adapter=None,
                                 feature_sequence=[], aggregate=None)

    assert np.load(os.path.join(export, "features.npy")).shape[0] == 6
    written = np.load(os.path.join(export, "metadata.pickle.npy"), allow_pickle=True).item()
    assert written["aggregate"] is None


# --- orchestration wiring (build + flatten monkeypatched) --------------------

def test_create_training_data_orchestration(tmp_path, monkeypatch):
    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    export = tmp_path / "export"
    n_px, T, bands = 7, 19, ["B04", "B08"]

    def fake_run_create_datacube(*, csv_filepath, **kw):
        # the workflow would build datacubes + write input.csv; we just write the csv.
        pd.DataFrame(
            {"datacube_filepath": ["x/datacube.npy"], "id": [0], "label": ["a"]}
        ).to_csv(csv_filepath, index=False)

    def fake_flatten(*, export_folderpath, **kw):
        from fsd.storage import fs
        fs.makedirs(export_folderpath)
        fs.save_npy(f"{export_folderpath}/data.npy", np.zeros((n_px, T, len(bands))))
        fs.save_npy(f"{export_folderpath}/ids.npy", np.zeros(n_px))
        fs.save_npy(f"{export_folderpath}/coords.npy", np.zeros((n_px, 2)))
        fs.save_npy(f"{export_folderpath}/labels.npy", np.array(["a"] * n_px))
        fs.save_npy(
            f"{export_folderpath}/metadata.pickle.npy",
            {"timestamps": list(range(T)), "bands": bands}, allow_pickle=True,
        )

    monkeypatch.setattr(api._create_datacube, "run_create_datacube", fake_run_create_datacube)
    monkeypatch.setattr(api._flatten, "flatten", fake_flatten)

    td = fsd.create_training_data(
        label_polygons=_polys(tmp_path), catalog_filepath=cat,
        startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=bands,
        id_col="fid", label_col="crop", export_folderpath=str(export),
    )
    assert isinstance(td, fsd.TrainingData)
    assert td.n_pixels == n_px
    assert td.n_timestamps == T
    assert td.bands == bands

    loaded = td.load()
    assert loaded["data"].shape == (n_px, T, len(bands))
    assert set(loaded) == {"data", "ids", "coords", "metadata", "labels"}


# --- characterisation of the shared preflight messages (#145 P10) ---------------
# Full `str(exc)` pinned: text and line order must survive the helper extraction.

def _download_kw(tmp_path, **over):
    kw = dict(
        roi=None, startdate=JAN1, enddate=JAN1_NEXT, bands=["B04"],
        dst_folderpath=str(tmp_path / "archive"), source="mpc", max_tiles=1,
    )
    kw.update(over)
    return kw


def _preflight_text(fn, *args, **kwargs) -> str:
    with pytest.raises(api.PreflightError) as ei:
        fn(*args, **kwargs)
    return str(ei.value)


def test_download_unknown_source_then_max_tiles(tmp_path):
    text = _preflight_text(fsd.download, **_download_kw(tmp_path, source="nope", max_tiles=0))
    assert text == (
        "preflight failed:\n"
        "  - source='nope' must be one of 'cdse', 'mpc'.\n"
        "  - max_tiles (0) must be >= 1."
    )


def test_download_cdse_max_tiles_then_creds(tmp_path):
    text = _preflight_text(
        fsd.download, **_download_kw(tmp_path, source="cdse", max_tiles=0, creds=None)
    )
    assert text == (
        "preflight failed:\n"
        "  - max_tiles (0) must be >= 1.\n"
        "  - creds (CdseCredentials) required for source='cdse' with runner='local'."
    )


def test_download_unserved_collection_comes_first(tmp_path):
    text = _preflight_text(
        fsd.download,
        **_download_kw(tmp_path, source="cdse", collection="sentinel-1-rtc", max_tiles=0),
    )
    assert text == (
        "preflight failed:\n"
        "  - source='cdse' does not serve collection='sentinel-1-rtc'; "
        "source='cdse' serves: ['sentinel-2-l2a'].\n"
        "  - max_tiles (0) must be >= 1.\n"
        "  - creds (CdseCredentials) required for source='cdse' with runner='local'."
    )


def test_create_training_data_wave2_max_tiles_required_then_creds(tmp_path):
    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    text = _preflight_text(
        fsd.create_training_data,
        label_polygons=_polys(tmp_path), catalog_filepath=cat,
        startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=["B04"],
        id_col="fid", label_col="crop", export_folderpath=str(tmp_path / "e"),
        download=True, source="cdse", max_tiles=None, creds=None,
    )
    assert text == (
        "preflight failed:\n"
        "  - max_tiles (>= 1) is required when download=True (got None).\n"
        "  - creds (CdseCredentials) required for source='cdse' with runner='local'."
    )


_NOT_BOTH = (
    "pass either `adapter` or `feature_sequence`, not both (ambiguous feature transform)."
)


def test_create_training_data_not_both_exact_text(tmp_path):
    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    text = _preflight_text(
        fsd.create_training_data,
        label_polygons=_polys(tmp_path), catalog_filepath=cat,
        startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=["B04"],
        id_col="fid", label_col="crop", export_folderpath=str(tmp_path / "e"),
        adapter=object(), feature_sequence=[("x", {})], runner="batch",
    )
    assert text == _NOT_BOTH   # raised alone, before the seam errors are collected


def test_flatten_training_data_not_both_exact_text(tmp_path):
    text = _preflight_text(
        fsd.flatten_training_data, str(tmp_path / "missing.csv"), str(tmp_path / "e"),
        adapter=object(), feature_sequence=[("x", {})], runner="batch",
    )
    assert text == _NOT_BOTH


class _NeedsB04:
    required_bands = ["B04"]


def test_create_training_data_aggregate_error_follows_adapter_error(tmp_path):
    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    text = _preflight_text(
        fsd.create_training_data,
        label_polygons=_polys(tmp_path), catalog_filepath=cat,
        startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=["B08"],
        id_col="fid", label_col="crop", export_folderpath=str(tmp_path / "e"),
        adapter=_NeedsB04(), aggregate="bogus", runner="batch",
    )
    assert text == (
        "preflight failed:\n"
        "  - runner='batch' not supported (valid: ['local', 'aml']).\n"
        "  - adapter.required_bands not in requested bands: ['B04']\n"
        "  - unknown aggregate 'bogus'; use None, 'median_per_id', or a callable."
    )


def test_flatten_training_data_aggregate_error_follows_seam_errors(tmp_path):
    text = _preflight_text(
        fsd.flatten_training_data, str(tmp_path / "missing.csv"), str(tmp_path / "e"),
        aggregate="bogus", runner="batch",
    )
    assert text == (
        "preflight failed:\n"
        "  - runner='batch' not supported (valid: ['local', 'aml']).\n"
        "  - unknown aggregate 'bogus'; use None, 'median_per_id', or a callable.\n"
        f"  - input_csv does not exist: {str(tmp_path / 'missing.csv')!r}"
    )


@pytest.mark.parametrize("required, bands", [(["B8A"], ["nir08"]), (["nir08"], ["B8A"])])
def test_create_training_data_alias_spelling_is_not_a_missing_band(tmp_path, required, bands):
    cat = _touch(tmp_path / "sentinel-2-l2a" / "catalog.parquet")

    class _Adapter:
        required_bands = required

    text = _preflight_text(
        fsd.create_training_data,
        label_polygons=_polys(tmp_path), catalog_filepath=cat,
        startdate=JAN1, enddate=JAN1_NEXT, mosaic_days=20, bands=bands,
        id_col="fid", label_col="crop", export_folderpath=str(tmp_path / "e"),
        adapter=_Adapter(), aggregate="bogus",
    )
    assert text == (
        "preflight failed:\n"
        "  - unknown aggregate 'bogus'; use None, 'median_per_id', or a callable."
    )
