"""Cloud-Optimized GeoTIFF conversion. See specs/14-cog-on-download.md.

One canonical `to_cog(src, dst, ...)`: convert a **local** raster to a lossless COG
(DEFLATE + PREDICTOR=2, tiled), optionally with overviews. This is the single home
for the COG creation profile — the CDSE ingest path (`sources.cdse.download`) and the
spec-13 benchmark prep both go through here.

Lossless by construction: DEFLATE + PREDICTOR=2 is a reversible integer-differencing
pre-step (not a quality knob); for `uint16` sources — S2 reflectance declares NBITS=15
in a uint16 container, which PREDICTOR=2 rejects — we set `NBITS=16`, promoting only the
*declared* bit depth (pixel values are unchanged). Verified bit-identical in specs/13.

NOTE (specs/10): like all pixel I/O, conversion reads/writes **local** paths through
rasterio/GDAL (VSI), not fsspec — the documented raster-I/O exception. A remote source
must be fetched to local scratch first (`rise_eo.storage.get`).

**Remote dst:** when `dst_path` is a remote `rise_eo.storage` URL (e.g.
`abfss://...`), the COG is still produced by GDAL on **node-local scratch** (the identical
local path above, byte-for-byte), then published to `dst_path` via `storage.transfer`
(itself a write-to-`.part`-then-atomic-`mv`) — never a
remote `rasterio.open(mode="w")` (TODO #39). This is the single chokepoint that lets both
the per-cell `output.tif` (`model.engine._write_output_cog`, on an AML node) and the
merged `merged.tif` (`api._merge_outputs`, on the driver) land on blob — both callers are
unchanged; only `to_cog` learned the remote-dst branch.
"""

from __future__ import annotations

import os
import tempfile
import uuid

import numpy as np
import rasterio
import rasterio.shutil

from rise_eo import config
from rise_eo.raster import rio_open
from rise_eo.storage import fs

__all__ = ["to_cog", "cog_creation_opts", "stamp_gdal_tags", "stamp_or_reencode"]


def cog_creation_opts(
    src_path: str,
    *,
    compress: str = config.COG_COMPRESS,
    predictor: int = config.COG_PREDICTOR,
    blocksize: int = config.COG_BLOCKSIZE,
    overviews: str = config.COG_OVERVIEWS,
) -> dict:
    """COG driver creation options for `src_path`. Adds `NBITS=16` for uint16 sources
    (PREDICTOR=2 needs an 8/16/32/64-bit declared depth; S2 declares NBITS=15) — a
    lossless promotion of the *declared* depth only."""
    with rio_open(src_path) as s:
        dtype = s.dtypes[0]
    opts = dict(
        driver="COG",
        COMPRESS=compress,
        PREDICTOR=predictor,
        BLOCKSIZE=blocksize,
        OVERVIEWS=overviews,
    )
    if dtype == "uint16":
        opts["NBITS"] = 16
    return opts


def _remove_quietly(path: str) -> None:
    """Delete `path` if it exists, ignoring an `OSError`: this runs while another
    exception is propagating, and must not replace it."""
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def to_cog(
    src_path: str,
    dst_path: str,
    *,
    compress: str = config.COG_COMPRESS,
    predictor: int = config.COG_PREDICTOR,
    blocksize: int = config.COG_BLOCKSIZE,
    overviews: str = config.COG_OVERVIEWS,
    verify: bool = False,
) -> int:
    """Convert a local raster `src_path` -> Cloud-Optimized GeoTIFF `dst_path`.

    Lossless (DEFLATE + PREDICTOR, NBITS=16 for uint16). Overviews per `overviews`
    (``"AUTO"`` builds them, ``"NONE"`` skips). Returns bytes written.

    **Local `dst_path` (unchanged):** writes a sibling ``dst.part`` then ``os.replace``
    onto ``dst`` — a crash never leaves a truncated ``.tif`` that a resume would mistake
    for done.

    **Remote `dst_path`:** GDAL still writes the COG to node-local scratch
    (byte-identical to the local path above), then `storage.transfer` publishes it to
    `dst_path` (write-to-`.part`-then-atomic-`mv` on the destination filesystem) and the
    scratch file is removed. Never a remote `rasterio.open(mode="w")` (TODO #39).

    `verify=True` reads both rasters back and asserts they are bit-identical (used in
    tests / a paranoid ingest); off by default since the conversion is deterministic
    and proven lossless (specs/13).
    """
    opts = cog_creation_opts(
        src_path,
        compress=compress,
        predictor=predictor,
        blocksize=blocksize,
        overviews=overviews,
    )

    if fs.is_local(dst_path):
        parent = os.path.dirname(dst_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{dst_path}.part"
        try:
            rasterio.shutil.copy(src_path, tmp, **opts)  # GDAL picks format from driver=
            os.replace(tmp, dst_path)
        except BaseException:
            _remove_quietly(tmp)
            raise
        nbytes = os.path.getsize(dst_path)
    else:
        fd, scratch = tempfile.mkstemp(suffix=".tif", prefix=f"rise-eo-cog-{uuid.uuid4().hex}-")
        os.close(fd)
        os.remove(scratch)  # GDAL's COG driver must create the file itself
        try:
            rasterio.shutil.copy(src_path, scratch, **opts)
            fs.transfer(scratch, dst_path)  # -> dst_path.part -> atomic mv
        finally:
            if os.path.exists(scratch):
                os.remove(scratch)
        nbytes = fs.size(dst_path)

    if verify:
        with rio_open(src_path) as s, rio_open(dst_path) as d:
            if not np.array_equal(s.read(), d.read()):
                raise ValueError(f"to_cog: {dst_path} is not bit-identical to {src_path}")

    return nbytes


def stamp_gdal_tags(
    filepath: str,
    *,
    offset: float = 0.0,
    scale: float = 1.0,
    set_nodata_if_missing: float | None = None,
) -> None:
    """Declare radiometry + nodata as GDAL band-level metadata, in place.

    A header-tag edit: reopens `filepath` in update ("r+") mode and sets every band's
    `SCALE`/`OFFSET` (what `rio-tiler`/titiler's `unscale=true` reads — STAC
    `raster:bands` alone is **not** forwarded to the viewer, per titiler discussion
    #803) and, if the raster has no nodata tag yet, sets it to `set_nodata_if_missing`
    (never overwrites an already-declared nodata). Never decodes pixels:
    no radiometric loss, cheap even on a large COG.

    Plain `rasterio.open(...).read()` (what the datacube builder uses) never
    auto-applies these tags — they are inert metadata on a normal read, so this and
    the STAC `raster:bands` declaration (`rise_eo.catalog.stac`) can coexist with the
    builder's own read-time `apply_offset` without double-applying.

    `IGNORE_COG_LAYOUT_BREAK=YES`: GDAL's COG driver refuses an in-place metadata
    edit by default ("has COG layout... updating it will generally result in losing
    part of the optimizations") even though a scale/offset/nodata tag is a header-only
    change that never touches pixel data or block layout — this open option is exactly
    what lets the "cheap header edit" this function promises actually happen in place.
    """
    with rio_open(filepath, "r+", IGNORE_COG_LAYOUT_BREAK="YES") as dst:
        dst.scales = (scale,) * dst.count
        dst.offsets = (offset,) * dst.count
        if set_nodata_if_missing is not None and dst.nodata is None:
            dst.nodata = set_nodata_if_missing


def stamp_or_reencode(
    filepath: str,
    *,
    offset: float = 0.0,
    scale: float = 1.0,
    set_nodata_if_missing: float | None = None,
) -> None:
    """`stamp_gdal_tags`, falling back to a GDAL-COG-driver re-encode if the in-place
    stamp breaks COG validity (whether an in-place
    tag edit keeps a strictly-valid COG is source/GDAL-version dependent; this is the
    documented fallback, not the expected path).
    """
    try:
        stamp_gdal_tags(
            filepath, offset=offset, scale=scale,
            set_nodata_if_missing=set_nodata_if_missing,
        )
    except Exception:
        tmp = f"{filepath}.reencode.part"
        try:
            rasterio.shutil.copy(filepath, tmp, **cog_creation_opts(filepath))
            os.replace(tmp, filepath)
        finally:
            _remove_quietly(tmp)
        stamp_gdal_tags(
            filepath, offset=offset, scale=scale,
            set_nodata_if_missing=set_nodata_if_missing,
        )
