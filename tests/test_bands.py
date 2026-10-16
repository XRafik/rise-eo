"""Tests for rise_eo.bands.modify (spec 06).

5-D contract: (samples, timestamps, height, width, bands). Most tests use a
1x1x1 spatial/temporal cell so index arithmetic is hand-checkable.
"""

import numpy as np
import pytest

from rise_eo.bands import modify

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"]


def _cell(values: dict):
    """Build a (1, 1, 1, 1, nbands) array + band_indices from {band: value}."""
    band_indices = {b: i for i, b in enumerate(values)}
    arr = np.zeros((1, 1, 1, 1, len(values)), dtype=float)
    for b, v in values.items():
        arr[0, 0, 0, 0, band_indices[b]] = v
    return arr, band_indices


# --- spectral indices --------------------------------------------------------


def test_compute_ndvi():
    bands, bi = _cell({"B04": 1000.0, "B08": 3000.0})
    out, out_bi = modify.compute_bands(bands, bi, ["NDVI"])
    # (3000-1000)/(3000+1000) = 0.5
    assert out[0, 0, 0, 0, out_bi["NDVI"]] == pytest.approx(0.5)
    assert out_bi["NDVI"] == 2  # appended after the 2 source bands
    assert out.shape[-1] == 3


def test_compute_ndre_gcvi_savi():
    bands, bi = _cell({"B03": 1000.0, "B04": 1000.0, "B05": 1000.0, "B08": 3000.0})
    out, out_bi = modify.compute_bands(bands, bi, ["NDRE", "GCVI", "SAVI"])
    assert out[0, 0, 0, 0, out_bi["NDRE"]] == pytest.approx(0.5)  # (3000-1000)/4000
    assert out[0, 0, 0, 0, out_bi["GCVI"]] == pytest.approx(2.0)  # 3000/1000 - 1
    # SAVI: (nir-red)(1+L)/(nir+red+L), L=0.48
    expected_savi = (3000 - 1000) * 1.48 / (3000 + 1000 + 0.48)
    assert out[0, 0, 0, 0, out_bi["SAVI"]] == pytest.approx(expected_savi)


def test_compute_bands_missing_source_raises():
    bands, bi = _cell({"B08": 3000.0})  # NDVI needs B04 too
    with pytest.raises(ValueError):
        modify.compute_bands(bands, bi, ["NDVI"])


def test_compute_bands_unknown_index_raises():
    bands, bi = _cell({"B04": 1.0, "B08": 1.0})
    with pytest.raises(NotImplementedError):
        modify.compute_bands(bands, bi, ["NOPE"])


# --- mask + interpolate ------------------------------------------------------


def test_interpolation_fills_only_masked_positions():
    # one band, 3 timestamps, middle value masked (==0, the default mask_value)
    bands = np.zeros((1, 3, 1, 1, 1), dtype=float)
    bands[0, :, 0, 0, 0] = [2000.0, 0.0, 4000.0]
    bi = {"B02": 0}
    out, _ = modify.mask_invalid_and_interpolate(bands, bi)
    series = out[0, :, 0, 0, 0]
    assert series[0] == 2000.0  # valid -> untouched
    assert series[2] == 4000.0  # valid -> untouched
    assert series[1] == pytest.approx(3000.0)  # masked -> interpolated


def test_mask_caps_out_of_range_then_interpolates():
    # 12000 is above upper_cap (10000) -> masked -> interpolated from neighbours
    bands = np.zeros((1, 3, 1, 1, 1), dtype=float)
    bands[0, :, 0, 0, 0] = [1000.0, 12000.0, 3000.0]
    bi = {"B02": 0}
    out, _ = modify.mask_invalid_and_interpolate(bands, bi)
    assert out[0, 1, 0, 0, 0] == pytest.approx(2000.0)


def test_mask_interpolate_all_valid_unchanged():
    bands = np.zeros((1, 3, 1, 1, 1), dtype=float)
    bands[0, :, 0, 0, 0] = [1000.0, 2000.0, 3000.0]
    bi = {"B02": 0}
    out, _ = modify.mask_invalid_and_interpolate(bands, bi)
    assert np.array_equal(out[0, :, 0, 0, 0], [1000.0, 2000.0, 3000.0])


# --- remove / scale ----------------------------------------------------------


def test_remove_bands_reindexes_consistently():
    bands, bi = _cell({"B02": 1.0, "B03": 2.0, "B04": 3.0})
    out, out_bi = modify.remove_bands(bands, bi, ["B03"])
    assert out_bi == {"B02": 0, "B04": 1}
    assert out.shape[-1] == 2
    assert out[0, 0, 0, 0, out_bi["B02"]] == 1.0
    assert out[0, 0, 0, 0, out_bi["B04"]] == 3.0


def test_scale_bands():
    bands, bi = _cell({"B02": 5000.0, "B03": 2000.0})
    out, _ = modify.scale_bands(bands, bi, ["B02"], std=10000)
    assert out[0, 0, 0, 0, bi["B02"]] == pytest.approx(0.5)
    assert out[0, 0, 0, 0, bi["B03"]] == 2000.0  # untouched


# --- modify_bands sequence + expand helpers ----------------------------------


def test_modify_bands_threads_sequence():
    bands, bi = _cell({"B04": 1000.0, "B08": 3000.0})
    out, out_bi = modify.modify_bands(
        bands,
        bi,
        sequence=[
            (modify.compute_bands, dict(bands_to_compute=["NDVI"])),
            (modify.remove_bands, dict(bands_to_remove=["B04", "B08"])),
        ],
    )
    assert out_bi == {"NDVI": 0}
    assert out[0, 0, 0, 0, 0] == pytest.approx(0.5)


def test_expand_datacube():
    dc = np.zeros((4, 2, 3, 9))  # (t, h, w, b)
    assert modify.expand_datacube(dc).shape == (1, 4, 2, 3, 9)


def test_expand_flattened():
    flat = np.zeros((100, 4, 9))  # (pixels, t, b)
    assert modify.expand_flattened(flat).shape == (100, 4, 1, 1, 9)
