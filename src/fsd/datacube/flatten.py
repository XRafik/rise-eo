"""Flatten datacubes -> stacked training arrays.

Spec: specs/05-flatten.md. Folds in datacube_flatten_2d.

Outputs (via fsd.storage): data.npy (pixels,t,b), coords.npy (pixels,2) [lon/lat in
EPSG:4326], ids.npy, labels.npy (optional), metadata.pickle.npy. Raises if datacubes
disagree on bands/timestamps.

Coords are reprojected from each cube's native CRS to EPSG:4326 (lon, lat) before
concatenation (TODO #16): a training set spanning two UTM zones would otherwise mix
eastings/northings from different zones in one array — same number, different place.
Lon/lat is a single common CRS so the concatenated coords are geographically comparable.
"""

from __future__ import annotations

import os

import numpy as np
import rasterio.crs
import rasterio.transform
import rasterio.warp

from fsd import config
from fsd.catalog import processing as processing_module
from fsd.datacube.files import CUBE_METADATA_NAME, TRAINING_METADATA_NAME
from fsd.storage import fs


def flatten(
    filepaths_df,                # one row per datacube
    filepath_col: str,
    id_col: str,
    export_folderpath: str,
    label_col: str | None = None,
    *,
    nodata: int = config.NODATA,
) -> None:
    """Stack per-geometry datacubes into model-ready per-pixel arrays.

    For each datacube `(t, H, W, b)`, keep pixels that are not entirely nodata
    across time+band -> `(pixels, t, b)`, concatenate across datacubes, and repeat
    `id` (and `label`) per pixel. All datacubes must share the same `bands` and
    `timestamps` (raises otherwise).
    """
    metadata_filepaths = [
        os.path.join(os.path.dirname(fp), CUBE_METADATA_NAME)
        for fp in filepaths_df[filepath_col]
    ]
    common_metadata = _check_metadata_consistency(metadata_filepaths)
    provenance = common_metadata.pop("provenance", None)

    data_list, coords_list, id_list = [], [], []
    label_list = [] if label_col is not None else None

    for _, row in filepaths_df.iterrows():
        data_2d, coords_2d = _read_datacube_2d(row[filepath_col], nodata=nodata)
        data_list.append(data_2d)
        coords_list.append(coords_2d)

        n_pixels = data_2d.shape[0]
        id_list += [row[id_col]] * n_pixels
        if label_col is not None:
            label_list += [row[label_col]] * n_pixels

    data = np.concatenate(data_list, axis=0)         # (pixels, t, b)
    coords = np.concatenate(coords_list, axis=0)     # (pixels, 2)
    common_metadata["data_shape_desc"] = ("pixel", "timestamps", "bands")
    if provenance:
        # Spec 59 D9: versions and sources only (a training set spans thousands of granules).
        common_metadata["provenance"] = provenance

    fs.makedirs(export_folderpath)
    fs.save_npy(os.path.join(export_folderpath, "data.npy"), data)
    fs.save_npy(os.path.join(export_folderpath, "coords.npy"), coords)
    fs.save_npy(os.path.join(export_folderpath, "ids.npy"), np.array(id_list))
    fs.save_npy(os.path.join(export_folderpath, TRAINING_METADATA_NAME),
                common_metadata, allow_pickle=True)
    if label_col is not None:
        fs.save_npy(os.path.join(export_folderpath, "labels.npy"),
                    np.array(label_list))


# --- helpers -----------------------------------------------------------------

def _check_metadata_consistency(metadata_filepaths, check_attributes=("bands",
                                                                       "timestamps")):
    """All datacubes must agree on `bands` and `timestamps`; return the common ones."""
    common = {attr: None for attr in check_attributes}
    provenance_blocks = []
    for mfp in metadata_filepaths:
        metadata = fs.load_npy(mfp, allow_pickle=True)[()]
        for attr in check_attributes:
            cur = metadata[attr]
            if common[attr] is None:
                common[attr] = cur
            elif cur != common[attr]:
                raise ValueError(f"Attribute {attr} are not consistent.")
        provenance_blocks.append(metadata.get("provenance"))
    merged = processing_module.merge_provenance(provenance_blocks, with_ids=False)
    if merged:
        common["provenance"] = merged
    return common


def _read_datacube_2d(filepath: str, nodata: int = 0):
    """Load a `(t, H, W, b)` datacube, keep pixels that aren't entirely nodata across
    time+band -> `(pixels, t, b)`, and return their (lon, lat) coords in EPSG:4326."""
    datacube = fs.load_npy(filepath)
    nt, _, _, nb = datacube.shape  # asserts 4-D

    # pixels whose nodata count across time(0)+band(3) is below the max => keep.
    h_idx, w_idx = np.where((datacube == nodata).sum(axis=(0, 3)) < nt * nb)
    data_2d = datacube[:, h_idx, w_idx, :].swapaxes(0, 1)  # (pixels, t, b)

    metadata = fs.load_npy(os.path.join(os.path.dirname(filepath), CUBE_METADATA_NAME),
                           allow_pickle=True)[()]
    gt = metadata["geotiff_metadata"]
    en = _easting_northing_array(width=gt["width"], height=gt["height"],
                                 transform=gt["transform"])
    coords_2d = en[h_idx, w_idx, :]                        # (pixels, 2) native CRS
    coords_2d = _to_lonlat(coords_2d, gt.get("crs"))       # -> EPSG:4326 (lon, lat)
    return data_2d, coords_2d


def _to_lonlat(coords_2d, src_crs):
    """Reproject `(x, y)` coords from the cube's native CRS to EPSG:4326 `(lon, lat)`.

    A no-op when `src_crs` is missing (can't reproject) or already EPSG:4326. Real
    datacubes always carry a CRS (the builder's `reference_profile`); the missing-CRS
    guard is only for synthetic/legacy metadata that never set one.
    """
    if src_crs is None or coords_2d.shape[0] == 0:
        return coords_2d
    dst = rasterio.crs.CRS.from_epsg(4326)
    src = rasterio.crs.CRS.from_user_input(src_crs)
    if src == dst:
        return coords_2d
    lons, lats = rasterio.warp.transform(
        src, dst, coords_2d[:, 0].tolist(), coords_2d[:, 1].tolist()
    )
    return np.column_stack([lons, lats])


def _easting_northing_array(width: int, height: int, transform):
    """Per-pixel (easting, northing) at pixel centers -> (H, W, 2)."""
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    xy_flat = np.column_stack((X.ravel(), Y.ravel()))
    eastings, northings = rasterio.transform.xy(
        transform=transform, rows=xy_flat[:, 1], cols=xy_flat[:, 0], offset="center",
    )
    en = np.array(list(zip(eastings, northings)))
    return en.reshape(*X.shape, 2)
