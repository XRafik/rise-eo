"""Raster image utilities: crop / reproject / resample / merge / load.

Spec: specs/07-raster.md. Folds in rsutils.modify_images + the raster helpers
from rsutils.utils that the data-prep path actually uses.

Convention (locked, specs/07): every in-memory op takes ``(data, profile)`` as
its first two arguments and returns ``(data, profile)``. This lets ops be chained
as a ``sequence`` of ``(func, kwargs)`` pairs (see ``modify_image_inplace``). The
in-memory datacube builder is built entirely on this convention.

NOTE (specs/10): rasterio reads rasters through GDAL's VSI layer, not fsspec, so
the file-based functions here open paths with ``rasterio.open`` directly. For a
*remote* source, fetch to local scratch via ``fsd.storage.get`` first (or pass a
GDAL ``/vsi*/`` path). Whole-file downloads stay in ``fsd.storage``; pixel reads
live here.
"""

from __future__ import annotations

import functools
import multiprocessing as mp
import os

import numpy as np
import rasterio
import rasterio.io
import rasterio.mask
import rasterio.merge
import rasterio.transform
import rasterio.warp
import tqdm

from fsd.raster import rio_open

NEAREST = rasterio.warp.Resampling.nearest

__all__ = [
    "NEAREST",
    # in-memory (data, profile) ops
    "crop",
    "reproject",
    "resample_by_ref_meta",
    "merge_inplace",
    "apply_offset",
    # sequence runners
    "modify_image_inplace",
    "modify_images_inplace",
    # file-based
    "crop_tif",
    "load_image",
    "load_images",
    "save_geotiff",
    "stack_bands",
    "save_rgb_geotiff",
    # helpers
    "driver_specific_meta_updates",
    "image_to_memfile",
    "images_to_memfiles",
    "delete_aux_xml",
]


# --- parallel-map helper -----------------------------------------------------


def _pmap(func, items, njobs, print_messages):
    """Map ``func`` over ``items``. Serial when ``njobs == 1`` (no Pool, so it
    stays usable inside tests and other already-parallel contexts); otherwise a
    ``multiprocessing.Pool``. Optional tqdm progress bar."""
    items = list(items)
    if njobs == 1:
        it = tqdm.tqdm(items) if print_messages else items
        return [func(x) for x in it]
    with mp.Pool(njobs) as p:
        mapped = p.imap(func, items)
        if print_messages:
            mapped = tqdm.tqdm(mapped, total=len(items))
        return list(mapped)


# --- in-memory (data, profile) ops -------------------------------------------


def crop(
    data: np.ndarray,
    profile: dict,
    shapes_gdf,
    nodata=None,
    all_touched: bool = False,
    crop: bool = True,
) -> tuple[np.ndarray, dict]:
    """Mask/crop an in-memory raster to ``shapes_gdf`` (reprojected to the raster
    CRS). Returns cropped ``(data, profile)``."""
    out_profile = profile.copy()
    if nodata is None:
        nodata = profile["nodata"]

    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data)
            shapes = shapes_gdf.to_crs(dataset.crs)["geometry"].to_list()
            out_data, out_transform = rasterio.mask.mask(
                dataset, shapes, crop=crop, nodata=nodata, all_touched=all_touched,
            )

    out_profile.update({
        "height": out_data.shape[1],
        "width": out_data.shape[2],
        "transform": out_transform,
        "nodata": nodata,
    })
    return out_data, out_profile


def reproject(
    data: np.ndarray,
    profile: dict,
    dst_crs,
    resampling=NEAREST,
) -> tuple[np.ndarray, dict]:
    """Reproject an in-memory raster to ``dst_crs``. No-op if already there."""
    src_crs = profile["crs"]
    if src_crs == dst_crs:
        return data, profile

    src_count = profile["count"]
    src_width = profile["width"]
    src_height = profile["height"]
    src_transform = profile["transform"]
    src_bounds = rasterio.transform.array_bounds(
        height=src_height, width=src_width, transform=src_transform
    )

    transform, width, height = rasterio.warp.calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *src_bounds
    )

    out_profile = profile.copy()
    out_profile.update({
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height,
    })
    out_profile = driver_specific_meta_updates(meta=out_profile)

    nodata = profile["nodata"]
    out_data = np.full(
        (src_count, height, width),
        fill_value=nodata if nodata is not None else 0,
        dtype=profile["dtype"],
    )

    for i in range(src_count):
        rasterio.warp.reproject(
            source=data[i],
            destination=out_data[i],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )

    return out_data, out_profile


def resample_by_ref_meta(
    data: np.ndarray,
    profile: dict,
    ref_meta: dict,
    resampling=NEAREST,
) -> tuple[np.ndarray, dict]:
    """Resample an in-memory raster onto the grid (transform/size/CRS) described
    by ``ref_meta``, keeping the source's nodata/dtype/count."""
    out_profile = ref_meta.copy()
    out_profile["nodata"] = profile["nodata"]
    out_profile["dtype"] = profile["dtype"]
    out_profile = driver_specific_meta_updates(
        meta=out_profile, driver=profile["driver"]
    )
    out_profile["count"] = profile["count"]

    nodata = out_profile["nodata"]
    out_data = np.full(
        (out_profile["count"], out_profile["height"], out_profile["width"]),
        fill_value=nodata if nodata is not None else 0,
        dtype=out_profile["dtype"],
    )

    for i in range(out_profile["count"]):
        rasterio.warp.reproject(
            source=data[i],
            destination=out_data[i],
            src_transform=profile["transform"],
            dst_transform=out_profile["transform"],
            src_nodata=profile["nodata"],
            dst_nodata=out_profile["nodata"],
            src_crs=profile["crs"],
            dst_crs=out_profile["crs"],
            resampling=resampling,
        )

    return out_data, out_profile


def apply_offset(
    data: np.ndarray, profile: dict, *, offset: int
) -> tuple[np.ndarray, dict]:
    """Shift DN by a declared additive radiometric offset:
    `clip(DN + offset, dtype_min, dtype_max)`, dtype preserved. `offset=0` is a no-op
    passthrough. Nodata (0) with a negative offset clips to 0 (stays nodata,
    order-independent of when this runs relative to masking).

    Clips to `data.dtype`'s own range (spec 58 D5.2) rather than a literal `0..65535` --
    the old hardcoded uint16 range silently mis-clipped any signed or float collection
    (e.g. Sentinel-1 RTC `float32`, whose valid range is nowhere near `0..65535`) the
    moment it declared a non-zero offset. `offset==0` still short-circuits, so this
    never touches a collection with no radiometric-offset concept.

    This is the READ-TIME apply the datacube builder uses (science needs physical
    reflectance before the median mosaic) — it never touches the on-disk
    bytes. The on-disk COG stays raw DN; the offset is metadata (GDAL tag + STAC
    `raster:bands`), applied here and, independently, by an `unscale`-aware viewer."""
    if offset == 0:
        return data, profile
    dtype_info = np.iinfo(data.dtype) if np.issubdtype(data.dtype, np.integer) \
        else np.finfo(data.dtype)
    work_dtype = np.float64 if np.issubdtype(data.dtype, np.floating) else np.int64
    out = np.clip(data.astype(work_dtype) + offset, dtype_info.min, dtype_info.max) \
        .astype(data.dtype)
    return out, profile


def merge_inplace(data_profile_list, nodata=None) -> tuple[np.ndarray, dict]:
    """Mosaic a list of in-memory rasters into one ``(data, profile)``."""
    merged_profile = data_profile_list[0][1].copy()
    memfiles = images_to_memfiles(data_profile_list)

    merged_data, merged_transform = rasterio.merge.merge(
        [memfile.open() for memfile in memfiles], nodata=nodata
    )

    merged_profile.update({
        "nodata": nodata,
        "count": merged_data.shape[0],
        "height": merged_data.shape[1],
        "width": merged_data.shape[2],
        "transform": merged_transform,
    })
    return merged_data, merged_profile


# --- sequence runners --------------------------------------------------------


def modify_image_inplace(
    data: np.ndarray,
    profile: dict,
    sequence: list,
    raise_error: bool = True,
) -> tuple[np.ndarray, dict]:
    """Apply an ordered ``[(func, kwargs), ...]`` op sequence to one in-memory
    raster. On failure with ``raise_error=False`` returns ``(None, None)``."""
    failed = False
    for func, kwargs in sequence:
        try:
            new_data, new_profile = func(data=data, profile=profile, **kwargs)
            del data, profile
            data, profile = new_data, new_profile
        except Exception:
            if raise_error:
                raise
            failed = True
            break
    if failed:
        return None, None
    return data, profile


def _modify_image_inplace_by_tuple(data_profile, sequence, raise_error):
    data, profile = data_profile
    return modify_image_inplace(
        data=data, profile=profile, sequence=sequence, raise_error=raise_error
    )


def modify_images_inplace(
    data_profile_list,
    sequence: list,
    njobs: int = 1,
    raise_error: bool = True,
    print_messages: bool = True,
):
    """Apply an op sequence over many in-memory rasters."""
    func = functools.partial(
        _modify_image_inplace_by_tuple, sequence=sequence, raise_error=raise_error
    )
    return _pmap(func, data_profile_list, njobs, print_messages)


# --- file-based --------------------------------------------------------------


def crop_tif(
    src_filepath: str,
    shapes_gdf,
    nodata=None,
    all_touched: bool = False,
) -> tuple[np.ndarray, dict]:
    """Crop straight from a file. Cheaper than load-then-crop: rasterio only
    reads the windowed region rather than the whole raster."""
    with rio_open(src_filepath) as src:
        out_meta = src.meta.copy()
        if nodata is None:
            nodata = out_meta["nodata"]
        shapes = shapes_gdf.to_crs(src.crs)["geometry"].to_list()
        out_image, out_transform = rasterio.mask.mask(
            src, shapes, crop=True, nodata=nodata, all_touched=all_touched,
        )

    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": nodata,
    })
    out_meta = driver_specific_meta_updates(meta=out_meta)
    return out_image, out_meta


def load_image(
    src_filepath: str,
    shapes_gdf=None,
    nodata=None,
    all_touched: bool = True,
    raise_error: bool = True,
) -> tuple[np.ndarray, dict]:
    """Load a raster (optionally cropping to ``shapes_gdf``) into ``(data,
    profile)``. On failure with ``raise_error=False`` returns ``(None, None)``."""
    try:
        if shapes_gdf is None:
            with rio_open(src_filepath) as src:
                return src.read(), src.meta.copy()
        return crop_tif(
            src_filepath=src_filepath,
            shapes_gdf=shapes_gdf,
            nodata=nodata,
            all_touched=all_touched,
        )
    except Exception:
        if raise_error:
            raise
        return None, None


def load_images(
    src_filepaths: list[str],
    shapes_gdf=None,
    nodata=None,
    all_touched: bool = True,
    raise_error: bool = True,
    njobs: int = 1,
    print_messages: bool = True,
):
    """Parallel load (+optional crop) -> ``list[(data, profile)]``."""
    func = functools.partial(
        load_image,
        shapes_gdf=shapes_gdf,
        nodata=nodata,
        all_touched=all_touched,
        raise_error=raise_error,
    )
    return _pmap(func, src_filepaths, njobs, print_messages)


def save_geotiff(dst_filepath: str, data: np.ndarray, profile: dict) -> None:
    """Write an in-memory ``(data, profile)`` raster to a GeoTIFF.

    ``count`` is taken from ``data`` (so this works for single- or multi-band
    arrays). Used both to persist cropped inputs and datacube outputs for visual
    inspection (QGIS).
    """
    out_profile = driver_specific_meta_updates(
        {**profile, "driver": "GTiff"}, driver="GTiff"
    )
    out_profile["count"] = data.shape[0]
    parent = os.path.dirname(dst_filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with rio_open(dst_filepath, "w", **out_profile) as dst:
        dst.write(data)
    delete_aux_xml(dst_filepath)


def stack_bands(data_profile_list) -> tuple[np.ndarray, dict]:
    """Stack single-band ``(data, profile)`` rasters sharing a grid into one
    multi-band ``(data, profile)`` (bands in list order). Grids must match."""
    datas = [data for data, _ in data_profile_list]
    profile = data_profile_list[0][1].copy()
    shapes = {d.shape[1:] for d in datas}
    if len(shapes) != 1:
        raise ValueError(f"stack_bands: rasters do not share a grid: {shapes}")
    stacked = np.concatenate(datas, axis=0)
    profile["count"] = stacked.shape[0]
    return stacked, profile


def save_rgb_geotiff(
    dst_filepath: str,
    rgb_data_profiles,
    *,
    scale_max: float | None = None,
) -> None:
    """Write three single-band ``(data, profile)`` rasters (R, G, B order, same
    grid) as a 3-band RGB GeoTIFF for visual comparison in QGIS.

    Native uint16 reflectance is kept by default (QGIS can stretch it). Pass
    ``scale_max`` to linearly map ``[0, scale_max] -> [0, 255]`` uint8 for a
    ready-to-view image (e.g. ``scale_max=3000`` for S2 visible bands).
    """
    if len(rgb_data_profiles) != 3:
        raise ValueError("save_rgb_geotiff expects exactly 3 (data, profile) bands.")
    stacked, profile = stack_bands(rgb_data_profiles)
    if scale_max is not None:
        stacked = np.clip(stacked.astype(float) / scale_max * 255.0, 0, 255).astype(
            "uint8"
        )
        profile = {**profile, "dtype": "uint8", "nodata": 0}
    save_geotiff(dst_filepath, stacked, profile)


# --- helpers -----------------------------------------------------------------


def driver_specific_meta_updates(meta: dict, driver: str = None) -> dict:
    """Apply driver-specific profile fixes (compression / quality)."""
    if driver is None:
        driver = meta["driver"]
    if driver == "GTiff":
        meta.update({"driver": "GTiff", "compress": "lzw"})
    elif driver == "JP2OpenJPEG":
        # https://github.com/rasterio/rasterio/issues/1677#issuecomment-488597072
        meta.update({"driver": "JP2OpenJPEG", "QUALITY": "100", "REVERSIBLE": "YES"})
    return meta


def image_to_memfile(data: np.ndarray, profile: dict) -> rasterio.io.MemoryFile:
    memfile = rasterio.io.MemoryFile()
    with memfile.open(**profile) as dataset:
        dataset.write(data)
    return memfile


def images_to_memfiles(data_profile_list) -> list[rasterio.io.MemoryFile]:
    return [image_to_memfile(data=d, profile=p) for d, p in data_profile_list]


def delete_aux_xml(filepath: str) -> None:
    aux_xml_filepath = filepath + ".aux.xml"
    if os.path.exists(aux_xml_filepath):
        os.remove(aux_xml_filepath)
