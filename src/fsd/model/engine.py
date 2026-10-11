"""Inference engine: fsd owns the predict loop and output writing.

Spec: specs/18-model-adapter.md

`infer_datacube` runs one datacube through the contract — feature transform (F1) →
`datacube_to_X` → drop-NaN pixels → chunked `predict` → scatter into a nodata-filled
`(bands, H, W)` array (F2) → `to_output` (F3). `infer_datacube_to_cog` adds the COG write
(reusing `fsd.raster.cog.to_cog`, lossless + overviews) with the datacube's transform/crs.
`run_local` runs many datacubes **sequentially in-process** (there is no process pool;
`api.run_inference` fans out `cores>1` via the Snakemake infer-only runner instead). The
ROI→tiling→download front-end that produces the datacubes is `run_inference(roi=…)`.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import rasterio

from fsd.bands import modify
from fsd.datacube.files import CUBE_METADATA_NAME
from fsd.model.adapter import Output
from fsd.model.features import apply_features
from fsd.raster.cog import to_cog
from fsd.storage import fs

__all__ = ["infer_datacube", "infer_datacube_to_cog", "run_local"]


def _chunked_predict(adapter, X: np.ndarray, predict_batch_size: int | None) -> np.ndarray:
    """Call `adapter.predict` over `X` in chunks; fsd owns this loop (F2)."""
    n = X.shape[0]
    if n == 0:
        return np.empty((0,), dtype=float)
    if not predict_batch_size or predict_batch_size >= n:
        return np.asarray(adapter.predict(X))
    parts = [np.asarray(adapter.predict(X[i:i + predict_batch_size]))
             for i in range(0, n, predict_batch_size)]
    return np.concatenate(parts, axis=0)


def _scatter(preds: np.ndarray, valid: np.ndarray, n: int, nodata, dtype) -> np.ndarray:
    """Place valid-pixel predictions back into a full `(n,)`/`(n, k)` array of `nodata`."""
    if preds.ndim <= 1:
        full = np.full((n,), nodata, dtype=dtype)
        full[valid] = preds
    else:
        full = np.full((n, preds.shape[1]), nodata, dtype=dtype)
        full[valid] = preds
    return full


def infer_datacube(adapter, datacube: np.ndarray, band_indices: dict, *,
                   predict_batch_size: int | None = None, skip_nan: bool = True) -> Output:
    """Run one `(T, H, W, B)` datacube through the contract -> `Output` `(bands, H, W)`."""
    data5d = modify.expand_datacube(np.asarray(datacube, dtype=float))
    # copy band_indices: modify_bands mutates it in place, and callers reuse the dict.
    feats5d, bi = apply_features(data5d, dict(band_indices), adapter=adapter)
    feats = np.squeeze(feats5d, axis=0)                    # (T, H, W, Bf)
    _, h, w, _ = feats.shape

    X = adapter.datacube_to_X(feats, bi)                   # (H*W, n_features)
    n = X.shape[0]
    valid = ~np.isnan(X).any(axis=1) if skip_nan else np.ones(n, dtype=bool)

    preds = _chunked_predict(adapter, X[valid], predict_batch_size)
    raw_full = _scatter(preds, valid, n, adapter.output_nodata, adapter.output_dtype)
    return adapter.to_output(raw_full, (h, w))


def _write_output_cog(out: Output, transform, crs, dst_path: str) -> None:
    """Write an `Output` as a lossless COG (via `raster.cog.to_cog`) at `dst_path`.

    Writes a plain GeoTIFF sibling first, then converts. `to_cog` publishes to a remote
    `dst_path`. **The raw scratch tif and its parent dir are always node-local**
    regardless of `dst_path`: a remote dst (e.g. `abfss://.../output.tif`, the per-cell site
    an AML node writes) must NOT `os.makedirs`/`rasterio.open(mode="w")` on the
    remote URL (a forbidden remote write that scatters junk local dirs, TODO #39); it stages
    locally, and `to_cog` transfers it to blob. Mirrors `api._merge_outputs`' own local-scratch
    guard for the identical pattern.
    """
    bands, h, w = out.array.shape
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": bands,
        "dtype": out.dtype, "crs": crs, "transform": transform, "nodata": out.nodata,
    }
    if fs.is_local(dst_path):
        parent = os.path.dirname(dst_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        raw_tif = f"{dst_path}.raw.tif"
    else:
        fd, raw_tif = tempfile.mkstemp(suffix=".raw.tif", prefix="fsd-out-")
        os.close(fd)
    try:
        with rasterio.open(raw_tif, "w", **profile) as dst:
            dst.write(out.array)
        to_cog(raw_tif, dst_path)
    finally:
        if os.path.exists(raw_tif):
            os.remove(raw_tif)


def infer_datacube_to_cog(adapter, datacube_filepath: str, output_filepath: str, *,
                          predict_batch_size: int | None = None, skip_nan: bool = True) -> str:
    """Load a datacube + metadata, run inference, write the output COG. Returns its path."""
    datacube = fs.load_npy(datacube_filepath)
    metadata = fs.load_npy(
        os.path.join(os.path.dirname(datacube_filepath), CUBE_METADATA_NAME), allow_pickle=True
    )[()]
    band_indices = {b: i for i, b in enumerate(metadata["bands"])}

    out = infer_datacube(
        adapter, datacube, band_indices,
        predict_batch_size=predict_batch_size, skip_nan=skip_nan,
    )
    gt = metadata["geotiff_metadata"]
    _write_output_cog(out, gt["transform"], gt["crs"], output_filepath)
    return output_filepath


# --- fan-out over many datacubes ---------------------------------------------

# Per-process adapter cache so a bundle is loaded only once per process (in-process run_local,
# and the infer-only task's sequential group loop).
_BUNDLE_CACHE: dict[str, object] = {}


def _adapter_from_bundle_cached(bundle_path: str):
    from fsd.model import bundle as _bundle
    if bundle_path not in _BUNDLE_CACHE:
        _BUNDLE_CACHE[bundle_path] = _bundle.load(bundle_path)
    return _BUNDLE_CACHE[bundle_path]


def run_local(model, pairs: list[tuple[str, str]], *,
              predict_batch_size: int | None = None, skip_nan: bool = True,
              overwrite: bool = False, progress: bool = True) -> list[str]:
    """Infer over `pairs` of `(datacube_filepath, output_filepath)`, **sequentially in-process**.

    `model` is a live adapter **or** a bundle path (str). This is the `cores=1` / test / debug /
    small-run path; `api.run_inference` routes `cores>1` through the Snakemake **infer-only** runner
    instead — fsd has **no in-process process pool**. Existing outputs are **skipped** unless
    `overwrite`. Returns every output path.
    """
    is_bundle = isinstance(model, str)
    adapter = _adapter_from_bundle_cached(model) if is_bundle else model
    if not is_bundle:
        adapter.load()
    outs = []
    total = len(pairs)
    for i, (dc_fp, out_fp) in enumerate(pairs, 1):
        if not overwrite and fs.exists(out_fp):
            if progress:
                print(f"[inference] {i}/{total} skip (exists) -> {out_fp}", flush=True)
        else:
            infer_datacube_to_cog(
                adapter, dc_fp, out_fp,
                predict_batch_size=predict_batch_size, skip_nan=skip_nan,
            )
            if progress:
                print(f"[inference] {i}/{total} -> {out_fp}", flush=True)
        outs.append(out_fp)
    return outs
