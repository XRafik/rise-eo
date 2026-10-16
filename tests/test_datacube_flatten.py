"""Tests for rise_eo.datacube.flatten (spec 05). Synthetic datacubes on disk."""

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

from rise_eo.datacube import flatten
from rise_eo.storage import fs

TS = [pd.Timestamp("2018-06-01", tz="UTC"), pd.Timestamp("2018-07-01", tz="UTC")]
TRANSFORM = from_origin(500000, 5000000, 10, 10)


def _save_cube(folder, arr, bands=("B04", "B08"), timestamps=TS, crs="EPSG:32633"):
    folder.mkdir(parents=True, exist_ok=True)
    fs.save_npy(str(folder / "datacube.npy"), arr)
    md = {"bands": list(bands), "timestamps": list(timestamps),
          "geotiff_metadata": {"width": arr.shape[2], "height": arr.shape[1],
                               "transform": TRANSFORM, "crs": crs}}
    fs.save_npy(str(folder / "metadata.pickle.npy"), md, allow_pickle=True)
    return str(folder / "datacube.npy")


def test_flatten_stacks_and_excludes_nodata(tmp_path):
    # A: (t=2,H=2,W=2,b=2), pixel (0,0) fully nodata -> 3 kept; B: all kept -> 4
    a = np.ones((2, 2, 2, 2), dtype=np.uint16)
    a[:, 0, 0, :] = 0
    b = np.ones((2, 2, 2, 2), dtype=np.uint16)
    fp_a = _save_cube(tmp_path / "A", a)
    fp_b = _save_cube(tmp_path / "B", b)

    df = pd.DataFrame({"fp": [fp_a, fp_b], "id": ["A", "B"], "label": [7, 9]})
    out = tmp_path / "flat"
    flatten.flatten(df, "fp", "id", str(out), label_col="label")

    data = fs.load_npy(str(out / "data.npy"))
    coords = fs.load_npy(str(out / "coords.npy"))
    ids = fs.load_npy(str(out / "ids.npy"))
    labels = fs.load_npy(str(out / "labels.npy"))
    md = fs.load_npy(str(out / "metadata.pickle.npy"), allow_pickle=True)[()]

    assert data.shape == (7, 2, 2)       # (3 + 4 pixels, t=2, b=2)
    assert coords.shape == (7, 2)
    # coords reprojected from the cube's UTM CRS to EPSG:4326 lon/lat (TODO #16):
    # transform origin (500000, 5000000) in EPSG:32633 -> ~15E / ~45N, well inside range.
    assert np.all((coords[:, 0] >= -180) & (coords[:, 0] <= 180))   # lon
    assert np.all((coords[:, 1] >= -90) & (coords[:, 1] <= 90))     # lat
    assert 10 < coords[:, 0].mean() < 20 and 44 < coords[:, 1].mean() < 46
    assert list(ids) == ["A"] * 3 + ["B"] * 4
    assert list(labels) == [7] * 3 + [9] * 4
    assert md["data_shape_desc"] == ("pixel", "timestamps", "bands")
    assert md["bands"] == ["B04", "B08"]


def test_flatten_reprojects_multizone_coords_to_common_lonlat(tmp_path):
    # TODO #16: two cubes in DIFFERENT UTM zones with the SAME easting/northing numbers
    # used to be concatenated as-is -> a 32636 easting and a 32637 easting at the same
    # value are different places. After reprojection to EPSG:4326 both are lon/lat, so
    # the west (zone 36, ~33-36E) and east (zone 37, ~36-42E) cubes land at DIFFERENT,
    # correctly-ordered longitudes despite identical native coordinates.
    arr = np.ones((2, 1, 1, 2), dtype=np.uint16)
    fp_w = _save_cube(tmp_path / "W", arr, crs="EPSG:32636")   # west of 36E
    fp_e = _save_cube(tmp_path / "E", arr, crs="EPSG:32637")   # east of 36E
    df = pd.DataFrame({"fp": [fp_w, fp_e], "id": ["W", "E"]})
    out = tmp_path / "flat"
    flatten.flatten(df, "fp", "id", str(out))

    coords = fs.load_npy(str(out / "coords.npy"))
    assert coords.shape == (2, 2)
    lon_w, lon_e = coords[0, 0], coords[1, 0]
    assert lon_w < 36 < lon_e          # same native easting -> different real longitudes
    assert np.all((coords[:, 1] >= -90) & (coords[:, 1] <= 90))


def test_flatten_no_label(tmp_path):
    fp = _save_cube(tmp_path / "A", np.ones((2, 2, 2, 2), dtype=np.uint16))
    out = tmp_path / "flat"
    flatten.flatten(pd.DataFrame({"fp": [fp], "id": ["A"]}), "fp", "id", str(out))
    assert fs.exists(str(out / "data.npy"))
    assert not fs.exists(str(out / "labels.npy"))


def test_flatten_raises_on_inconsistent_bands(tmp_path):
    fp_a = _save_cube(tmp_path / "A", np.ones((2, 2, 2, 2), dtype=np.uint16))
    fp_b = _save_cube(tmp_path / "B", np.ones((2, 2, 2, 2), dtype=np.uint16),
                      bands=("B04", "B8A"))
    df = pd.DataFrame({"fp": [fp_a, fp_b], "id": ["A", "B"]})
    with pytest.raises(ValueError, match="bands are not consistent"):
        flatten.flatten(df, "fp", "id", str(tmp_path / "flat"))
