"""Tests for rise_eo.raster.images (spec 07).

Synthetic in-memory rasters (no real .jp2 files) exercise the (data, profile)
ops, the sequence runners, the file-based round-trip, and the helpers.
"""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import rasterio.crs
import rasterio.transform
import shapely.geometry as sg

from rise_eo.raster import images, rio_open

UTM = rasterio.crs.CRS.from_epsg(32643)  # metres, so area/extent math is sane


def _read_tif(filepath):
    with rio_open(filepath) as src:
        return src.read(), src.meta.copy()


def _synthetic(width=10, height=10, west=0.0, north=100.0, res=10.0, fill=None):
    """A single-band uint16 raster at `res` m/px, top-left at (west, north)."""
    profile = {
        "driver": "GTiff",
        "dtype": "uint16",
        "nodata": 0,
        "width": width,
        "height": height,
        "count": 1,
        "crs": UTM,
        "transform": rasterio.transform.from_origin(west, north, res, res),
    }
    if fill is None:
        data = (np.arange(width * height).reshape(1, height, width) + 1).astype(
            "uint16"
        )
    else:
        data = np.full((1, height, width), fill, dtype="uint16")
    return data, profile


# --- in-memory ops -----------------------------------------------------------


def test_crop_reduces_extent_and_keeps_crs():
    data, profile = _synthetic()
    roi = gpd.GeoDataFrame(geometry=[sg.box(20, 20, 50, 50)], crs=UTM)
    out_data, out_profile = images.crop(data, profile, roi, all_touched=True)
    assert out_profile["height"] < profile["height"]
    assert out_profile["width"] < profile["width"]
    assert out_data.shape == (1, out_profile["height"], out_profile["width"])
    assert out_profile["crs"] == UTM


def test_reproject_changes_crs():
    data, profile = _synthetic()
    out_data, out_profile = images.reproject(data, profile, dst_crs="EPSG:4326")
    assert out_profile["crs"] == rasterio.crs.CRS.from_epsg(4326)
    assert out_data.shape[0] == 1


def test_reproject_same_crs_is_noop():
    data, profile = _synthetic()
    out_data, out_profile = images.reproject(data, profile, dst_crs=profile["crs"])
    assert out_data is data
    assert out_profile is profile


def test_apply_offset_zero_is_passthrough():
    data, profile = _synthetic()
    out_data, out_profile = images.apply_offset(data, profile, offset=0)
    assert out_data is data
    assert out_profile is profile


def test_apply_offset_shifts_and_clamps_uint16_no_underflow():
    data = np.array([[[1500, 500, 0]]], dtype=np.uint16)  # (1, 1, 3)
    _, profile = _synthetic()
    out_data, out_profile = images.apply_offset(data, profile, offset=-1000)
    assert out_data.dtype == np.uint16
    assert list(out_data[0, 0]) == [500, 0, 0]  # 1500-1000=500; 500-1000 clamps to 0; nodata stays 0
    assert out_profile is profile  # profile untouched


def test_is_radiometry_band_exempts_non_reflectance_bands():
    """spec 58 D5.3: `_is_reflectance` moved to `CollectionDeclaration.is_radiometry_band`
    -- declared per collection rather than a global regex."""
    from rise_eo.catalog.declaration import S2_L2A_DECLARATION

    for band in ("B01", "B04", "B08", "B8A", "B12"):
        assert S2_L2A_DECLARATION.is_radiometry_band(band) is True
    for band in ("SCL", "AOT", "WVP", "visual"):
        assert S2_L2A_DECLARATION.is_radiometry_band(band) is False


def test_resample_by_ref_meta_matches_ref_grid():
    data, profile = _synthetic(width=10, height=10, res=10.0)
    # Coarser reference covering the same extent: 5x5 @ 20 m.
    _, ref_meta = _synthetic(width=5, height=5, res=20.0)
    out_data, out_profile = images.resample_by_ref_meta(data, profile, ref_meta)
    assert out_profile["height"] == 5
    assert out_profile["width"] == 5
    assert out_data.shape == (1, 5, 5)
    # source nodata/dtype are preserved
    assert out_profile["nodata"] == profile["nodata"]
    assert out_profile["dtype"] == profile["dtype"]


def test_merge_inplace_mosaics_adjacent_tiles():
    # Two 10x10 tiles side by side (east neighbour) -> 20 wide, 10 tall.
    left = _synthetic(width=10, height=10, west=0.0, north=100.0, fill=1)
    right = _synthetic(width=10, height=10, west=100.0, north=100.0, fill=2)
    merged_data, merged_profile = images.merge_inplace([left, right], nodata=0)
    assert merged_profile["width"] == 20
    assert merged_profile["height"] == 10
    assert merged_data.shape == (1, 10, 20)
    # left half came from `left` (1s), right half from `right` (2s)
    assert (merged_data[0, :, :10] == 1).all()
    assert (merged_data[0, :, 10:] == 2).all()


# --- sequence runners --------------------------------------------------------


def _add(data, profile, amount):
    return (data + amount).astype(data.dtype), profile


def _boom(data, profile):
    raise RuntimeError("op failed")


def test_modify_image_inplace_applies_sequence():
    data, profile = _synthetic(fill=5)
    out_data, out_profile = images.modify_image_inplace(
        data, profile, sequence=[(_add, dict(amount=3)), (_add, dict(amount=1))]
    )
    assert (out_data == 9).all()
    assert out_profile is profile


def test_modify_image_inplace_swallows_error_when_not_raising():
    data, profile = _synthetic(fill=5)
    out_data, out_profile = images.modify_image_inplace(
        data, profile, sequence=[(_boom, dict())], raise_error=False
    )
    assert out_data is None and out_profile is None


def test_modify_image_inplace_raises_by_default():
    data, profile = _synthetic(fill=5)
    with pytest.raises(RuntimeError):
        images.modify_image_inplace(data, profile, sequence=[(_boom, dict())])


def test_modify_images_inplace_serial():
    items = [_synthetic(fill=1), _synthetic(fill=2)]
    out = images.modify_images_inplace(
        items, sequence=[(_add, dict(amount=10))], njobs=1, print_messages=False
    )
    assert (out[0][0] == 11).all()
    assert (out[1][0] == 12).all()


# --- file-based round-trip ---------------------------------------------------


def _write_tif(path, data, profile):
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def test_load_image_roundtrip(tmp_path):
    data, profile = _synthetic(fill=7)
    p = str(tmp_path / "img.tif")
    _write_tif(p, data, profile)
    out_data, out_profile = images.load_image(p)
    assert np.array_equal(out_data, data)
    assert out_profile["crs"] == UTM


def test_load_image_error_returns_none_when_not_raising(tmp_path):
    out = images.load_image(str(tmp_path / "missing.tif"), raise_error=False)
    assert out == (None, None)


def test_load_images_serial(tmp_path):
    paths = []
    for i in range(2):
        data, profile = _synthetic(fill=i + 1)
        p = str(tmp_path / f"img{i}.tif")
        _write_tif(p, data, profile)
        paths.append(p)
    out = images.load_images(paths, njobs=1, print_messages=False)
    assert len(out) == 2
    assert (out[0][0] == 1).all()
    assert (out[1][0] == 2).all()


# --- geotiff save / stack ----------------------------------------------------


def test_save_geotiff_roundtrip(tmp_path):
    data, profile = _synthetic(fill=42)
    dst = str(tmp_path / "out/saved.tif")
    images.save_geotiff(dst, data, profile)
    out_data, out_profile = _read_tif(dst)
    assert np.array_equal(out_data, data)
    assert out_profile["driver"] == "GTiff"
    assert out_profile["crs"] == UTM


def test_stack_bands():
    b1 = _synthetic(fill=1)
    b2 = _synthetic(fill=2)
    b3 = _synthetic(fill=3)
    stacked, profile = images.stack_bands([b1, b2, b3])
    assert stacked.shape == (3, 10, 10)
    assert profile["count"] == 3
    assert (stacked[0] == 1).all() and (stacked[2] == 3).all()


def test_stack_bands_grid_mismatch_raises():
    b1 = _synthetic(width=10, height=10, fill=1)
    b2 = _synthetic(width=5, height=5, fill=2)
    with pytest.raises(ValueError):
        images.stack_bands([b1, b2])


def test_save_rgb_geotiff_native(tmp_path):
    r, g, b = _synthetic(fill=100), _synthetic(fill=200), _synthetic(fill=300)
    dst = str(tmp_path / "rgb.tif")
    images.save_rgb_geotiff(dst, [r, g, b])
    out_data, out_profile = _read_tif(dst)
    assert out_data.shape == (3, 10, 10)
    assert out_profile["dtype"] == "uint16"
    assert (out_data[0] == 100).all()


def test_save_rgb_geotiff_scaled_to_uint8(tmp_path):
    # scale_max=3000 maps reflectance to [0,255] uint8
    r, g, b = _synthetic(fill=1500), _synthetic(fill=3000), _synthetic(fill=6000)
    dst = str(tmp_path / "rgb8.tif")
    images.save_rgb_geotiff(dst, [r, g, b], scale_max=3000)
    out_data, out_profile = _read_tif(dst)
    assert out_profile["dtype"] == "uint8"
    assert out_data[0].flat[0] == 127  # 1500/3000*255 = 127.5 -> 127
    assert out_data[1].flat[0] == 255  # 3000 -> 255
    assert out_data[2].flat[0] == 255  # clipped


def test_save_rgb_geotiff_wrong_band_count_raises():
    r, g = _synthetic(fill=1), _synthetic(fill=2)
    with pytest.raises(ValueError):
        images.save_rgb_geotiff("x.tif", [r, g])


# --- helpers -----------------------------------------------------------------


def test_driver_specific_meta_updates_gtiff():
    meta = images.driver_specific_meta_updates({"driver": "GTiff"})
    assert meta["compress"] == "lzw"


# --- COG conversion (spec 14) ------------------------------------------------


def _write_gtiff(path, width, height, dtype="uint16"):
    """A synthetic single-band GeoTIFF to feed to to_cog."""
    data = (np.arange(width * height).reshape(1, height, width) % 4096).astype(dtype)
    with rasterio.open(
        str(path), "w", driver="GTiff", height=height, width=width, count=1,
        dtype=dtype, crs=UTM,
        transform=rasterio.transform.from_origin(0, height * 10, 10, 10),
    ) as dst:
        dst.write(data)
    return data


def test_to_cog_lossless_with_overviews(tmp_path):
    from rise_eo.raster.cog import to_cog

    src = tmp_path / "src.tif"
    src_data = _write_gtiff(src, 1024, 1024)  # > blocksize so an overview is built
    dst = tmp_path / "out.tif"
    nbytes = to_cog(str(src), str(dst), overviews="AUTO", verify=True)

    assert nbytes == dst.stat().st_size
    assert not (tmp_path / "out.tif.part").exists()  # atomic: no leftover
    with rasterio.open(str(dst)) as d:
        assert d.driver == "GTiff"
        assert d.overviews(1)  # AUTO built at least one overview level
        assert (d.read() == src_data).all()  # bit-identical (lossless)
        # tiled, deflate-compressed, and NBITS=16 promotion for uint16
        assert d.profile["compress"] == "deflate"
        assert d.profile["tiled"] is True


def test_to_cog_no_overviews(tmp_path):
    from rise_eo.raster.cog import to_cog

    src = tmp_path / "src.tif"
    _write_gtiff(src, 1024, 1024)
    dst = tmp_path / "out.tif"
    to_cog(str(src), str(dst), overviews="NONE")
    with rasterio.open(str(dst)) as d:
        assert d.overviews(1) == []  # none materialized


def test_to_cog_verify_is_a_noop_on_lossless(tmp_path):
    """uint8 (SCL-like) source: no NBITS promotion, still bit-identical."""
    from rise_eo.raster.cog import to_cog

    src = tmp_path / "scl.tif"
    src_data = _write_gtiff(src, 64, 64, dtype="uint8")
    dst = tmp_path / "scl_cog.tif"
    to_cog(str(src), str(dst), overviews="NONE", verify=True)
    with rasterio.open(str(dst)) as d:
        assert (d.read() == src_data).all()


# --- ingest radiometry tags (spec 34 §1a/§1c) --------------------------------


def _write_stampable_cog(path, value, nodata=None, dtype="uint16"):
    """A small real COG to stamp — `stamp_gdal_tags` reopens in "r+", so the file
    has to be a genuine GDAL-readable raster, not a stub."""
    from rise_eo.raster.cog import to_cog

    plain = str(path) + ".plain.tif"
    with rasterio.open(
        plain, "w", driver="GTiff", height=4, width=4, count=1, dtype=dtype,
        crs=UTM, transform=rasterio.transform.from_origin(0, 40, 10, 10),
        nodata=nodata,
    ) as dst:
        dst.write(np.full((1, 4, 4), value, dtype=dtype))
    to_cog(plain, str(path), overviews="NONE")
    if nodata is not None:  # the COG driver drops nodata on some GDAL builds
        with rasterio.open(str(path), "r+", IGNORE_COG_LAYOUT_BREAK="YES") as d:
            d.nodata = nodata
    return path


@pytest.mark.parametrize("dn", [1, 500, 999, 1000])
def test_stamped_cog_preserves_low_dn_on_disk(tmp_path, dn):
    """spec 34 §1: the old `apply_boa_offset` clip(DN-1000, 0, ...) is GONE from the
    ingest/store path. A reflectance DN in (0,1000] must survive to disk untouched —
    the offset is *declared* (a tag), never baked into the pixels."""
    from rise_eo.raster.cog import stamp_gdal_tags

    p = _write_stampable_cog(tmp_path / f"b04_{dn}.tif", value=dn)
    stamp_gdal_tags(str(p), offset=-1000, scale=1 / 10000, set_nodata_if_missing=0)

    with rasterio.open(str(p)) as d:
        assert (d.read() == dn).all()  # NOT clamped to 0
        assert d.offsets[0] == -1000


def test_plain_read_of_stamped_cog_returns_raw_dn(tmp_path):
    """spec 34 §1a "no double-application": a plain `rasterio.open(...).read()` — what
    the datacube builder does — never auto-applies the GDAL SCALE/OFFSET tags. The
    builder's explicit `apply_offset` is the *only* place the offset is applied, so
    the tag and the builder cannot drift into applying it twice."""
    from rise_eo.raster.cog import stamp_gdal_tags

    p = _write_stampable_cog(tmp_path / "b04.tif", value=1500)
    stamp_gdal_tags(str(p), offset=-1000, scale=1 / 10000, set_nodata_if_missing=0)

    with rasterio.open(str(p)) as d:
        assert (d.read() == 1500).all()          # raw stamped DN, not 500
        assert d.offsets[0] == -1000             # ...while the tag still declares it
        assert d.scales[0] == pytest.approx(1 / 10000)


def test_stamp_sets_nodata_only_when_missing(tmp_path):
    """spec 34 §1c: ingest declares nodata=0 on a source that has none, but a source
    that already declares one keeps its own value."""
    from rise_eo.raster.cog import stamp_gdal_tags

    missing = _write_stampable_cog(tmp_path / "missing.tif", value=100, nodata=None)
    stamp_gdal_tags(str(missing), set_nodata_if_missing=0)
    with rasterio.open(str(missing)) as d:
        assert d.nodata == 0

    declared = _write_stampable_cog(tmp_path / "declared.tif", value=100, nodata=65535)
    stamp_gdal_tags(str(declared), set_nodata_if_missing=0)
    with rasterio.open(str(declared)) as d:
        assert d.nodata == 65535  # kept, not overwritten
