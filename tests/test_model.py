"""Tests for the ModelAdapter contract + inference engine (spec 18). Synthetic + deterministic.

A tiny fake adapter (argmax over a NDVI feature) exercises: the F1 feature transform running
once, the F2 predict-loop chunking + NaN→nodata scatter, the F3 `(bands,H,W)` COG write with
transform/crs preserved, the STAC output catalog, band/`T` preflight, `median_per_id`, and
bundle save/load (incl. `module:attr` resolution).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pystac
import pytest
import rasterio
import shapely.geometry
from rasterio.crs import CRS
from rasterio.transform import from_origin

import rise_eo
from rise_eo.bands import modify
from rise_eo.model import bundle, engine, features
from rise_eo.model.adapter import BaseModelAdapter
from rise_eo.storage import fs

BANDS = ["B04", "B08"]           # enough to compute NDVI
EPSG = 32637


class ArgmaxNDVI(BaseModelAdapter):
    """NDVI-sign classifier: feature = NDVI, class 1 if mean-over-time NDVI > 0 else 0."""

    required_bands = BANDS
    n_timestamps = 2
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["cls"]
    feature_sequence = [
        (modify.compute_bands, dict(bands_to_compute=["NDVI"])),
        (modify.remove_bands, dict(bands_to_remove=BANDS)),
    ]

    def load(self):
        self.loaded = True

    def predict(self, X_chunk):
        return (X_chunk.mean(axis=1) > 0).astype("uint8")


class BundledArgmax(ArgmaxNDVI):
    """Same, but the decision threshold comes from a bundled artifact."""

    def load(self):
        with open(self.artifacts["thresh"]) as f:
            self.thresh = float(f.read())

    def predict(self, X_chunk):
        return (X_chunk.mean(axis=1) > self.thresh).astype("uint8")


class ModelDeterminedT(ArgmaxNDVI):
    """Leaves n_timestamps unset (0) -> deferred to the trained model / bundle (spec 22)."""
    n_timestamps = 0


def _make_datacube_folder(folder, *, bands=BANDS, T=2, H=4, W=4):
    """(T,H,W,2) datacube: top rows NDVI>0 (class 1), bottom-left NDVI<0 (class 0),
    bottom-right all-zero nodata (-> NaN feature -> output nodata 255)."""
    os.makedirs(folder, exist_ok=True)
    dc = np.zeros((T, H, W, len(bands)), dtype=np.uint16)
    dc[:, : H // 2, :, 0], dc[:, : H // 2, :, 1] = 2000, 8000        # NDVI = +0.6
    dc[:, H // 2 :, : W // 2, 0], dc[:, H // 2 :, : W // 2, 1] = 8000, 2000  # NDVI = -0.6
    fs.save_npy(os.path.join(folder, "datacube.npy"), dc)
    md = {
        "bands": bands,
        "timestamps": list(range(T)),
        "geotiff_metadata": {
            "width": W, "height": H,
            "transform": from_origin(500000, 4000000, 10, 10),
            "crs": CRS.from_epsg(EPSG),
        },
    }
    fs.save_npy(os.path.join(folder, "metadata.pickle.npy"), md, allow_pickle=True)
    return os.path.join(folder, "datacube.npy")


# --- engine: one datacube -> Output ------------------------------------------

def test_infer_datacube_scatter_and_shape(tmp_path):
    dc_fp = _make_datacube_folder(str(tmp_path / "dc"))
    datacube = fs.load_npy(dc_fp)
    adapter = ArgmaxNDVI()
    adapter.load()
    out = engine.infer_datacube(adapter, datacube, {b: i for i, b in enumerate(BANDS)})

    assert out.array.shape == (1, 4, 4)          # (bands, H, W)
    assert out.array.dtype == np.uint8
    top, bl, br = out.array[0, 0, 0], out.array[0, 3, 0], out.array[0, 3, 3]
    assert top == 1                              # NDVI > 0
    assert bl == 0                               # NDVI < 0
    assert br == 255                             # all-zero pixel -> NaN feature -> nodata


def test_run_local_skips_existing_unless_overwrite(tmp_path, monkeypatch):
    """spec 22: inference is idempotent — a rerun skips existing outputs unless overwrite=True."""
    dc = _make_datacube_folder(str(tmp_path / "dc"))
    out_fp = str(tmp_path / "out" / "o.tif")
    a = ArgmaxNDVI()
    engine.run_local(a, [(dc, out_fp)], progress=False)      # first run writes it
    assert os.path.exists(out_fp)

    calls = []
    monkeypatch.setattr(engine, "infer_datacube_to_cog",
                        lambda *args, **kw: calls.append(1))
    engine.run_local(a, [(dc, out_fp)], progress=False)      # output exists -> skip
    assert calls == []
    engine.run_local(a, [(dc, out_fp)], progress=False, overwrite=True)  # forced -> recompute
    assert calls == [1]


def test_predict_batch_size_matches_whole_tile(tmp_path):
    dc_fp = _make_datacube_folder(str(tmp_path / "dc"))
    datacube = fs.load_npy(dc_fp)
    bi = {b: i for i, b in enumerate(BANDS)}
    a = ArgmaxNDVI()
    a.load()
    whole = engine.infer_datacube(a, datacube, bi)
    chunked = engine.infer_datacube(a, datacube, bi, predict_batch_size=3)
    assert np.array_equal(whole.array, chunked.array)


# --- engine -> COG + full run_inference + STAC -------------------------------

def test_run_inference_writes_cogs_and_stac(tmp_path):
    dcs = [_make_datacube_folder(str(tmp_path / f"dc{i}")) for i in range(2)]
    out_dir = str(tmp_path / "out")
    result = rise_eo.run_inference(ArgmaxNDVI(), dcs, out_dir, progress=False)

    assert len(result.output_filepaths) == 2
    for fp in result.output_filepaths:
        with rasterio.open(fp) as src:
            assert src.count == 1 and src.crs.to_epsg() == EPSG
            assert src.nodata == 255
            assert src.read(1)[0, 0] == 1        # top pixel classified

    # STAC catalog round-trips and carries one item per output.
    cat = pystac.Catalog.from_file(result.stac_catalog_filepath)
    items = list(cat.get_items(recursive=True))
    assert len(items) == 2
    # each output gets a DISTINCT item id (the per-cube folder), not the constant "output" stem —
    # regression for the spec-26 STAC id collision (all items overwrote one <output>/output.json).
    assert len({it.id for it in items}) == 2
    assert all("proj:transform" in it.properties for it in items)


def test_run_inference_from_csv_uses_manifest_geometry(tmp_path):
    """spec 28: the pre-built input.csv path threads `shapefilepath` -> STAC gets the true
    cell polygon, not the raster bbox."""
    import json

    dc_fp = _make_datacube_folder(str(tmp_path / "dc0"))
    polygon = shapely.geometry.Polygon(
        [(14.766, 48.492), (14.789, 48.534), (14.847, 48.526), (14.825, 48.484)]
    )
    geom_fp = str(tmp_path / "dc0" / "geometry.geojson")
    with open(geom_fp, "w") as f:
        json.dump({
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "properties": {"id": "cell0"},
                "geometry": shapely.geometry.mapping(polygon),
            }],
        }, f)

    csv_fp = str(tmp_path / "input.csv")
    pd.DataFrame({
        "datacube_filepath": [dc_fp], "id": ["cell0"], "shapefilepath": [geom_fp],
    }).to_csv(csv_fp, index=False)

    out_dir = str(tmp_path / "out")
    result = rise_eo.run_inference(ArgmaxNDVI(), csv_fp, out_dir, progress=False)

    cat = pystac.Catalog.from_file(result.stac_catalog_filepath)
    items = list(cat.get_items(recursive=True))
    assert len(items) == 1
    got = shapely.geometry.shape(items[0].geometry)
    assert got.equals(polygon)
    assert list(items[0].bbox) == list(polygon.bounds)


def test_preflight_rejects_band_and_timestamp_mismatch(tmp_path):
    dc_fp = _make_datacube_folder(str(tmp_path / "dc"))

    class NeedsMore(ArgmaxNDVI):
        required_bands = ["B04", "B08", "B11"]   # B11 absent from the datacube

    with pytest.raises(rise_eo.PreflightError, match="required bands"):
        rise_eo.run_inference(NeedsMore(), [dc_fp], str(tmp_path / "o1"), progress=False)

    class WrongT(ArgmaxNDVI):
        n_timestamps = 5                          # datacube has T=2

    with pytest.raises(rise_eo.PreflightError, match="needs T=5"):
        rise_eo.run_inference(WrongT(), [dc_fp], str(tmp_path / "o2"), progress=False)


# --- F4 aggregation ----------------------------------------------------------

def test_median_per_id():
    ids = np.array([7, 7, 9])
    data = np.array([[[1.0]], [[3.0]], [[10.0]]])   # (pixels, T=1, B=1)
    labels = np.array(["a", "a", "b"])
    out_ids, out_data, out_labels = features.median_per_id(ids, data, labels)
    assert list(out_ids) == [7, 9]
    assert out_data[0, 0, 0] == 2.0                 # median(1,3)
    assert out_data[1, 0, 0] == 10.0
    assert list(out_labels) == ["a", "b"]


# --- F5 bundle: save/load + module:attr resolution ---------------------------

def test_bundle_save_load_roundtrip(tmp_path):
    thresh_fp = tmp_path / "thresh.txt"
    thresh_fp.write_text("0.0")
    bundle_dir = bundle.save(BundledArgmax(), {"thresh": str(thresh_fp)}, str(tmp_path / "bundle"))

    spec = bundle.read_spec(bundle_dir)              # model-free: no import/model-load
    assert spec["n_timestamps"] == 2
    assert spec["required_bands"] == BANDS
    assert spec["adapter"].endswith(":BundledArgmax")

    adapter = bundle.load(bundle_dir)                # resolves module:attr -> class -> instance
    assert adapter.thresh == 0.0
    assert os.path.isabs(adapter.artifacts["thresh"])


def test_bundle_load_detects_drift(tmp_path):
    bundle_dir = bundle.save(ArgmaxNDVI(), {}, str(tmp_path / "bundle"))
    # tamper: make the manifest claim a different T than the class declares.
    import json
    mfp = os.path.join(bundle_dir, "bundle.json")
    with open(mfp) as f:
        manifest = json.load(f)
    manifest["n_timestamps"] = 99
    with open(mfp, "w") as f:
        json.dump(manifest, f)
    # spec 44 D4: the check survives but its MEANING is per-origin. This adapter's source ships
    # inside the bundle, so a disagreement can only mean the bundle was edited -- which is what
    # the message now says. (The "code/bundle drift" wording is kept for installed-package
    # adapters, where version skew between image and bundle is real: see
    # tests/test_bundle_code.py::test_drift_message_is_per_origin.)
    with pytest.raises(ValueError, match="has been edited"):
        bundle.load(bundle_dir)


def test_bundle_load_allows_model_determined_n_timestamps(tmp_path):
    """A class that leaves n_timestamps unset (0) defers it to the trained model; the per-instance
    value the run set is recorded in the bundle and load() accepts it (no false 'drift'). This is
    what lets one adapter class back models trained on different T (spec 22 / demo cores>1)."""
    a = ModelDeterminedT()
    a.n_timestamps = 5                        # this model was trained on T=5
    bundle_dir = bundle.save(a, {}, str(tmp_path / "bundle"))
    assert bundle.read_spec(bundle_dir)["n_timestamps"] == 5   # bundle is authoritative
    loaded = bundle.load(bundle_dir)          # would raise "drift" (0 != 5) before the fix
    assert loaded.required_bands == BANDS


def test_resolve_ref_errors_on_bad_form():
    with pytest.raises(ValueError, match="module:attribute"):
        bundle.resolve_ref("no_colon_here")


# --- run_inference via a bundle path (model-free preflight path) --------------

def test_run_inference_from_bundle_path(tmp_path):
    thresh_fp = tmp_path / "thresh.txt"
    thresh_fp.write_text("0.0")
    bundle_dir = bundle.save(BundledArgmax(), {"thresh": str(thresh_fp)}, str(tmp_path / "bundle"))
    dc_fp = _make_datacube_folder(str(tmp_path / "dc"))
    result = rise_eo.run_inference(bundle_dir, [dc_fp], str(tmp_path / "out"), progress=False)
    with rasterio.open(result.output_filepaths[0]) as src:
        assert src.read(1)[0, 0] == 1
