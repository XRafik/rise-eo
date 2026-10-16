"""Composable band transforms over 5-D arrays.

Spec: specs/06-bands.md. Folds in rsutils.modify_bands (+ the `mask_interpolate`
kernel it depended on from rsutils.utils_preprocess).

Array contract: ``(samples, timestamps, height, width, bands)`` + a ``band_indices``
dict ``{band_name: index}``. Each op is ``(bands, band_indices, **kwargs) ->
(bands, band_indices)`` so they compose via ``modify_bands``. The 5-D contract is
kept deliberately so demo-notebook code ports 1:1 (the notebooks expand 4-D
datacubes / 3-D flattened arrays to 5-D via the ``expand_*`` helpers below).
"""

from __future__ import annotations

import numba
import numpy as np

__all__ = [
    "INDEX_COMPUTE_REQUIREMENTS",
    "modify_bands",
    "mask_invalid_and_interpolate",
    "compute_bands",
    "remove_bands",
    "scale_bands",
    "expand_datacube",
    "expand_flattened",
]


# Spectral indices and the bands each one needs (carried wholesale from legacy).
INDEX_COMPUTE_REQUIREMENTS = {
    "NDVI": ["B04", "B08"],
    "NDWI_green": ["B03", "B08"],
    "NDWI_blue": ["B02", "B08"],
    "NDRE": ["B05", "B08"],
    "LSWI_B11": ["B08", "B11"],
    "LSWI_B12": ["B08", "B12"],
    "GCVI": ["B08", "B03"],
    "SAVI": ["B08", "B04"],
    "BSI": ["B02", "B04", "B08", "B11"],
    "NDTI_tillage": ["B11", "B12"],
    "PSRI": ["B04", "B02", "B06"],
}


def modify_bands(bands: np.ndarray, band_indices: dict, sequence: list):
    """Run a sequence of ``(func, kwargs)`` ops, threading ``(bands, band_indices)``."""
    for func, kwargs in sequence:
        bands, band_indices = func(bands=bands, band_indices=band_indices, **kwargs)
    return bands, band_indices


# --- temporal interpolation kernel (ported from utils_preprocess) ------------


@numba.njit(parallel=True)
def _mask_interpolate_2D(band_data_2D: np.ndarray, mask_value=np.nan):
    """Per-row linear interpolation over the time axis (last axis), filling only
    masked positions. Rows that are all-valid or all-masked are left untouched."""
    n_samples, n_timestamps = band_data_2D.shape
    out = band_data_2D.copy()
    for n_sample in numba.prange(n_samples):
        if np.isnan(mask_value):
            valid_indexes = np.where(~np.isnan(band_data_2D[n_sample]))[0]
        else:
            valid_indexes = np.where(band_data_2D[n_sample] != mask_value)[0]
        n_valid = valid_indexes.shape[0]
        if n_valid == 0 or n_valid == n_timestamps:
            continue
        fp = np.array([band_data_2D[n_sample, vi] for vi in valid_indexes])
        out[n_sample] = np.interp(
            x=np.arange(n_timestamps), xp=valid_indexes, fp=fp
        )
    return out


def _mask_interpolate(band_data: np.ndarray, mask_value=np.nan):
    """Interpolate over masked values along the last (time) axis of an N-D array."""
    if np.isnan(mask_value):
        mask_count = np.isnan(band_data).sum()
    else:
        mask_count = (band_data == mask_value).sum()
    if mask_count == 0:
        return band_data

    *n_rem, n_ts = band_data.shape
    band_data_2D = band_data.reshape((int(np.prod(n_rem)), n_ts)).astype(band_data.dtype)
    interp_2D = _mask_interpolate_2D(band_data_2D=band_data_2D, mask_value=mask_value)
    return interp_2D.reshape(band_data.shape)


def mask_invalid_and_interpolate(
    bands: np.ndarray,
    band_indices: dict,
    upper_cap: float = 10000,
    lower_cap: float = 0,
    mask_value: float = 0,
    band_indices_to_modify: list = None,
):
    """Mask out-of-range values (``>= upper_cap`` or ``<= lower_cap``) to
    ``mask_value``, then temporally interpolate the masked gaps. Operates only on
    ``band_indices_to_modify`` (all bands by default)."""
    if band_indices_to_modify is None:
        band_indices_to_modify = band_indices.keys()

    selected_indices = [
        index for band, index in band_indices.items()
        if band in band_indices_to_modify
    ]

    original_dtype = bands.dtype
    if isinstance(mask_value, float) and bands.dtype != float:
        bands = bands.astype(float)

    selected = bands[:, :, :, :, selected_indices]
    selected[np.where(selected >= upper_cap)] = mask_value
    selected[np.where(selected <= lower_cap)] = mask_value

    # Move time to the last axis for the interpolation kernel, then back.
    interp_selected = _mask_interpolate(
        selected.swapaxes(1, -1), mask_value=mask_value
    ).swapaxes(1, -1)

    bands[:, :, :, :, selected_indices] = interp_selected
    bands = bands.astype(original_dtype)
    return bands, band_indices


def compute_bands(bands: np.ndarray, band_indices: dict, bands_to_compute: list[str]):
    """Append spectral indices (NDVI, NDRE, GCVI, SAVI, ...) as new bands.

    Each computed band is concatenated on the last axis and registered in
    ``band_indices``. Raises if an index is unknown or its required source bands
    are absent.
    """
    for band in bands_to_compute:
        if band not in INDEX_COMPUTE_REQUIREMENTS:
            raise NotImplementedError(f"{band} computation not implemented.")
        for req_band in INDEX_COMPUTE_REQUIREMENTS[band]:
            if req_band not in band_indices:
                raise ValueError(f"{req_band} not present for {band} computation.")

    def b(name):
        return bands[:, :, :, :, band_indices[name]]

    for band in bands_to_compute:
        if band == "NDVI":
            red, nir = b("B04"), b("B08")
            computed = (nir - red) / (nir + red)
        elif band == "NDWI_green":
            nir, green = b("B08"), b("B03")
            computed = (green - nir) / (green + nir)
        elif band == "NDWI_blue":
            nir, blue = b("B08"), b("B02")
            computed = (blue - nir) / (blue + nir)
        elif band == "NDRE":
            nir, red_edge = b("B08"), b("B05")
            computed = (nir - red_edge) / (nir + red_edge)
        elif band == "LSWI_B11":
            nir, swir = b("B08"), b("B11")
            computed = (nir - swir) / (nir + swir)
        elif band == "LSWI_B12":
            nir, swir = b("B08"), b("B12")
            computed = (nir - swir) / (nir + swir)
        elif band == "GCVI":
            nir, green = b("B08"), b("B03")
            computed = (nir / green) - 1
        elif band == "SAVI":
            nir, red = b("B08"), b("B04")
            L = 0.48
            computed = (nir - red) * (1 + L) / (nir + red + L)
        elif band == "BSI":
            red, swir, blue, nir = b("B04"), b("B11"), b("B02"), b("B08")
            computed = ((red + swir) - (nir + blue)) / ((red + swir) + (nir + blue))
        elif band == "PSRI":
            p_678, p_500, p_750 = b("B04"), b("B02"), b("B06")
            computed = (p_678 - p_500) / p_750
        elif band == "NDTI_tillage":
            swir, swir2 = b("B11"), b("B12")
            computed = (swir - swir2) / (swir + swir2)
        else:  # pragma: no cover - guarded by the validation loop above
            raise NotImplementedError(f"{band} computation not implemented.")

        bands = np.concatenate([bands, np.expand_dims(computed, axis=-1)], axis=-1)
        band_indices[band] = max(band_indices.values()) + 1

    return bands, band_indices


def remove_bands(bands: np.ndarray, band_indices: dict, bands_to_remove: list[str]):
    """Drop bands and reindex ``band_indices`` contiguously over what remains."""
    bands_to_keep = [b for b in band_indices if b not in bands_to_remove]
    indices_to_keep = [band_indices[b] for b in bands_to_keep]
    bands = bands[:, :, :, :, indices_to_keep]
    new_band_indices = {band: i for i, band in enumerate(bands_to_keep)}
    return bands, new_band_indices


def scale_bands(
    bands: np.ndarray,
    band_indices: dict,
    bands_to_scale: list[str],
    mean: float = 0,
    std: float = 1,
):
    """In-place ``(x - mean) / std`` over the named bands."""
    indices_to_scale = [
        index for band, index in band_indices.items() if band in bands_to_scale
    ]
    bands[:, :, :, :, indices_to_scale] = (
        bands[:, :, :, :, indices_to_scale] - mean
    ) / std
    return bands, band_indices


# --- convenience: expand lower-rank arrays to the 5-D contract ----------------


def expand_datacube(datacube: np.ndarray) -> np.ndarray:
    """4-D ``(t, h, w, b)`` -> 5-D ``(1, t, h, w, b)``."""
    return np.expand_dims(datacube, axis=0)


def expand_flattened(flattened: np.ndarray) -> np.ndarray:
    """3-D ``(pixels, t, b)`` -> 5-D ``(pixels, t, 1, 1, b)``."""
    return np.expand_dims(flattened, axis=(2, 3))
