"""Spec 42 §3 / run-book 43 Steps 2-4: build the tutorial micro-fixture by
clipping one grid cell's granules out of the archive catalog.

Reads `<archive-root>/catalog.parquet`, selects granules intersecting the cell
in `--roi`, clips each granule x band to the cell (+ a resampling buffer) in its
native UTM CRS, and writes the clipped COGs + a re-catalogued `catalog.parquet` +
`NOTICE` + `README.md` under `--out`.

**Radiometry (amendment A1, spec 42 §8):** the per-granule offset is READ from
the source catalog's own declaration column, never hardcoded to one name --
it is `offset` on the current blob (MPC) archive, `boa_add_offset` on an older
MPC catalog. If a granule's row carries neither (the local CDSE `demo_e2e`
archive declares no radiometry column at all), the documented D1 fallback
re-derives the offset from the granule id's own `_N####_` baseline token
(`_N0500_` -> 05.00 -> -1000). A granule with neither a declared value nor a
derivable token is refused, not silently guessed.

Idempotent and resumable per output file (spec 42 §3/run-book 43): re-running
skips any granule x band COG that already exists and is non-empty; nothing is
ever deleted.

Not part of the rise-eo wheel; not run by the test suite (spec 42 D4) -- its output
(`tests/data/tutorial/`) is the artifact, committed alongside it.

CLI is normative (spec 42 D4, run-book 43 Steps 2-4) -- match these flags
exactly:

    # Step 2 -- check-only (read-only, no writes). --bands is optional here; pass
    # it and the "all three bands present" PASS condition is actually gated on.
    python build_fixture.py --archive-root <url>/archive --roi roi.geojson \\
        --bands B04 B08 SCL --check-only --result _result_step2.json

    # Step 3 -- dry run (counts + bytes, zero writes)
    python build_fixture.py --archive-root <url>/archive --roi roi.geojson \\
        --fields fields.geojson --out tests/data/tutorial \\
        --bands B04 B08 SCL --max-bytes 31457280 --dry-run --result _result_step3.json

    # Step 4 -- build
    python build_fixture.py --archive-root <url>/archive --roi roi.geojson \\
        --fields fields.geojson --out tests/data/tutorial \\
        --bands B04 B08 SCL --max-bytes 31457280 --result _result_step4.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.windows
from shapely.ops import unary_union

from rise_eo import config
from rise_eo.catalog.catalog import COLUMNS as CATALOG_COLUMNS
from rise_eo.catalog.declaration import S2_L2A_DECLARATION, to_attrs
from rise_eo.raster import rio_open
from rise_eo.raster.cog import to_cog
from rise_eo.raster.images import crop_tif, save_geotiff
from rise_eo.storage import fs

__all__ = [
    "select_granules", "check_archive", "estimate_bytes", "clip_granule",
    "build_fixture", "collapse_radiometry_column", "derive_offset_from_id",
    "mgrs_tile_of", "subsample_timestamps", "redact_argv", "main",
]

# MGRS tile in an S2 product id, e.g. "..._T33UWP_..." (mirrors rise_eo.catalog.stac,
# duplicated rather than imported -- this generator is a standalone tool, not
# part of the wheel, D4).
_MGRS_TILE_RE = re.compile(r"_T(\d{2}[C-X][A-Z]{2})_")
# S2 processing-baseline token, e.g. "..._N0500_..." (spec 34 / spec 40 D14).
_BASELINE_RE = re.compile(r"_N(\d{2})(\d{2})_")

# A1: never hardcode ONE declaration-column name -- checked in this order.
RADIOMETRY_COLUMNS = ("offset", "boa_add_offset")

_UINT16_BYTES = 2
# Rough DEFLATE compression factor for real S2 reflectance/SCL data, used only
# to project a dry-run byte estimate against the 30 MB cap (D2) -- not a claim
# of exactness; the real `to_cog` write is what the 30 MB check in `build`
# measures for real.
_ESTIMATED_COG_COMPRESSION = 0.5

# spec 42 D5: the "modified" form -- clipping IS modification under the EC
# legal notice on Copernicus Sentinel data. The plain form is a defect, not a
# nitpick (run-book 43 Step 4).
NOTICE_TEXT = "Contains modified Copernicus Sentinel data 2018\n"

# spec 42 D2: cell bounds + a one-pixel-plus-resampling buffer at the 10 m
# reference grid, in the granule's native UTM CRS (meters).
CELL_BUFFER_M = 30.0


# --- pure helpers (offline-testable against a synthetic mini-archive) -------


def mgrs_tile_of(granule_id: str) -> str | None:
    """The MGRS tile token (`33UWP`) parsed from a granule id, or None."""
    m = _MGRS_TILE_RE.search(str(granule_id))
    return m.group(1) if m else None


def derive_offset_from_id(granule_id: str) -> int | None:
    """D1 fallback: the additive reflectance offset implied by the granule id's
    own baseline token (`_N0500_` -> 05.00 -> -1000; -1000 for baseline >= 04.00,
    else 0). None if the id carries no baseline token -- the common case on MPC
    ids (A1), which is exactly why this is a FALLBACK, not the primary path."""
    m = _BASELINE_RE.search(str(granule_id))
    if not m:
        return None
    return -1000 if (int(m.group(1)), int(m.group(2))) >= (4, 0) else 0


def collapse_radiometry_column(catalog_gdf: gpd.GeoDataFrame) -> str | None:
    """The first of `RADIOMETRY_COLUMNS` present in `catalog_gdf`, or None if
    neither is a column (the local CDSE `demo_e2e` archive, D1's fallback
    case)."""
    for col in RADIOMETRY_COLUMNS:
        if col in catalog_gdf.columns:
            return col
    return None


def row_offset(row, radiometry_col: str | None) -> tuple[int | None, str]:
    """`(offset, source)` for one source catalog row: `"declared"` (copied
    from `radiometry_col`, A1's primary path) if present and non-null, else
    `"derived"` (D1's id-token fallback), else `(None, "missing")` -- refuse to
    guess rather than default to 0."""
    if radiometry_col is not None:
        val = row.get(radiometry_col)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return int(val), "declared"
    derived = derive_offset_from_id(row["id"])
    if derived is not None:
        return derived, "derived"
    return None, "missing"


def bands_present(catalog_gdf: gpd.GeoDataFrame) -> list[str]:
    """Union of band names found across `catalog_gdf`'s `files` column."""
    bands: set[str] = set()
    for files in catalog_gdf["files"]:
        for name in str(files).split(","):
            name = name.strip()
            if name.endswith(".tif"):
                bands.add(name[: -len(".tif")])
    return sorted(bands)


def select_granules(
    catalog_gdf: gpd.GeoDataFrame,
    roi_gdf: gpd.GeoDataFrame,
    startdate: str | None = None,
    enddate: str | None = None,
) -> gpd.GeoDataFrame:
    """spec 42 §3 step 1: granules intersecting the cell, oldest-first, within
    `[startdate, enddate]` inclusive.

    **The date window is not optional in spirit.** Spec 42 §1/D2 specify
    **Apr-Sep 2018** -- a growing season, ~9 mosaic intervals at
    `mosaic_days=20`. The generator originally had no date filter at all, which
    was invisible while the assumed source was the *local* archive (Apr-Sep
    only, 24 granules). Run-book 43 Step 2 against the **blob** archive returned
    **72 granules spanning 2018-01-01 .. 2019-01-01** -- a full year, ~3x the
    bytes, half of it winter. Passing no window here means "take whatever the
    archive holds", which is a different fixture than the spec describes.
    """
    if roi_gdf.crs is None:
        roi_gdf = roi_gdf.set_crs("EPSG:4326")
    roi_shape = unary_union(roi_gdf.to_crs(catalog_gdf.crs).geometry.values)
    sel = catalog_gdf[catalog_gdf.intersects(roi_shape)].copy()
    if startdate is not None or enddate is not None:
        # Catalog timestamps are tz-aware UTC; localize the bounds to match, or
        # the comparison raises (the dt2ts rule, spec 06).
        ts = pd.to_datetime(sel["timestamp"], utc=True)
        if startdate is not None:
            sel = sel[ts >= pd.Timestamp(startdate, tz="UTC")]
            ts = ts[sel.index]
        if enddate is not None:
            sel = sel[ts <= pd.Timestamp(enddate, tz="UTC")]
    return sel.sort_values("timestamp").reset_index(drop=True)


def check_archive(
    catalog_gdf: gpd.GeoDataFrame,
    roi_gdf: gpd.GeoDataFrame,
    bands: list[str] | None = None,
    startdate: str | None = None,
    enddate: str | None = None,
) -> dict:
    """Run-book 43 Step 2's three A1 preconditions + the structural facts --
    read-only, no writes.

    `bands` (optional) are the bands the build will request; when given, the
    summary reports whether every one of them is present in the archive, so the
    CLI can gate on run-book 43 Step 2's *stated* PASS condition "all three
    bands present" rather than merely printing it.

    `offset_values`/`offset_sources` expose the radiometry the build WOULD
    stamp, and where each value came from -- A1's "copied from the source, not
    invented" is only checkable where the source is reachable (here, on the VM),
    never offline in the test suite (see amendment A2 / spec 42 §8).
    """
    all_in_cell = select_granules(catalog_gdf, roi_gdf)
    sel = select_granules(catalog_gdf, roi_gdf, startdate, enddate)
    tiles = sorted({t for t in (mgrs_tile_of(i) for i in sel["id"]) if t})
    radiometry_col = collapse_radiometry_column(sel)
    resolved = [row_offset(row, radiometry_col) for _, row in sel.iterrows()]
    declared = [offset is not None for offset, _ in resolved]
    present = bands_present(sel)
    ts = sel["timestamp"] if len(sel) else None
    date_span = [str(ts.min().date()), str(ts.max().date())] if ts is not None and len(sel) else [None, None]
    summary = {
        "granules": int(len(sel)),
        "granules_before_date_filter": int(len(all_in_cell)),
        "date_window": [startdate, enddate],
        "mgrs_tiles": tiles,
        "single_tile": len(tiles) == 1,
        "bands": present,
        "declaration_column": radiometry_col or "derived-from-id",
        "declared_non_null": int(sum(declared)),
        "offset_values": sorted({o for o, _ in resolved if o is not None}),
        "offset_sources": _count_sources(src for _, src in resolved),
        "date_span": date_span,
    }
    if bands is not None:
        summary["requested_bands"] = list(bands)
        summary["missing_bands"] = [b for b in bands if b not in present]
        summary["all_bands_present"] = not summary["missing_bands"]
    return summary


def _count_sources(sources) -> dict:
    """`{"declared": n, "derived": m, "missing": k}` -- only non-zero keys."""
    counts: dict[str, int] = {}
    for src in sources:
        counts[src] = counts.get(src, 0) + 1
    return dict(sorted(counts.items()))


def subsample_timestamps(sel: gpd.GeoDataFrame, max_timestamps: int) -> gpd.GeoDataFrame:
    """spec 42 D2's documented fallback -- keep at most `max_timestamps`
    granules, spread EVENLY across the date span rather than taking the first N.

    Taking `sel.iloc[:12]` of 24 Apr-Sep granules would leave Apr-Jun only, which
    destroys the thing D2 says is the point of keeping all 24 ("a real seasonal
    time series with real gaps and real cloud"). Endpoints are always kept.
    """
    n = len(sel)
    if max_timestamps is None or n <= max_timestamps:
        return sel
    if max_timestamps <= 1:
        return sel.iloc[[0]].reset_index(drop=True)
    idx = sorted({round(i * (n - 1) / (max_timestamps - 1)) for i in range(max_timestamps)})
    return sel.iloc[idx].reset_index(drop=True)


def _band_filepath(row, band: str) -> str:
    return os.path.join(str(row["local_folderpath"]).rstrip("/"), f"{band}.tif")


def _buffered_cell(roi_gdf: gpd.GeoDataFrame, dst_crs, buffer_m: float = CELL_BUFFER_M) -> gpd.GeoDataFrame:
    cell = roi_gdf.to_crs(dst_crs)
    shape = unary_union(cell.geometry.values).buffer(buffer_m)
    return gpd.GeoDataFrame(geometry=[shape], crs=dst_crs)


def estimate_bytes(sel: gpd.GeoDataFrame, roi_gdf: gpd.GeoDataFrame, bands: list[str]) -> dict:
    """spec 42 D2's dry-run estimate: per-band + total projected COG bytes,
    ZERO writes. Opens each source COG's header only (GDAL metadata, no pixel
    decode) to size the buffered clip window."""
    per_band = {b: 0 for b in bands}
    for _, row in sel.iterrows():
        for band in bands:
            fp = _band_filepath(row, band)
            if not fs.exists(fp):
                continue
            with rio_open(fp) as src:
                cell = _buffered_cell(roi_gdf, src.crs)
                window = rasterio.windows.from_bounds(*cell.total_bounds, transform=src.transform)
                window = window.round_lengths().round_offsets()
            px = max(int(window.width), 0) * max(int(window.height), 0)
            per_band[band] += int(px * _UINT16_BYTES * _ESTIMATED_COG_COMPRESSION)
    total = sum(per_band.values())
    return {"per_band_bytes": per_band, "total_bytes": int(total), "granules": int(len(sel))}


def _geom_from_cog(path: str):
    """spec 42 §3 step 3: recompute the per-granule geometry from the CLIPPED
    raster's own bounds -- never copy the source footprint (it changed)."""
    with rio_open(path) as src:
        bounds, crs = src.bounds, src.crs
    import shapely.geometry

    geom = shapely.geometry.box(*bounds)
    return gpd.GeoSeries([geom], crs=crs).to_crs("EPSG:4326").iloc[0]


def clip_granule(row, bands: list[str], roi_gdf: gpd.GeoDataFrame, dst_root: str) -> dict:
    """Clip one granule's requested bands to the buffered cell, through
    `raster.cog.to_cog` (ADR 0014, the single COG chokepoint) -- byte-shaped
    exactly like a real archive. Idempotent per band file (skips one that
    already exists and is non-empty). Returns `{"granule_dir", "written":
    {band: bytes}, "missing": [band, ...]}` -- `missing` bands are ones this
    granule's source directory doesn't have (not every band file, e.g. an
    SCL-only row); they are simply excluded from the output, not an error.
    """
    granule_dir = os.path.join(dst_root, row["id"])
    written: dict[str, int] = {}
    missing: list[str] = []
    for band in bands:
        src_fp = _band_filepath(row, band)
        dst_path = os.path.join(granule_dir, f"{band}.tif")
        if not fs.exists(src_fp):
            missing.append(band)
            continue
        if fs.exists(dst_path) and fs.size(dst_path) > 0:
            written[band] = fs.size(dst_path)
            continue

        with rio_open(src_fp) as src:
            cell = _buffered_cell(roi_gdf, src.crs)
        data, profile = crop_tif(src_fp, cell, nodata=config.NODATA, all_touched=True)

        tmp_dir = tempfile.mkdtemp(prefix="rise-eo-tutorial-fixture-")
        try:
            tmp_tif = os.path.join(tmp_dir, f"{band}.tif")
            save_geotiff(tmp_tif, data, profile)
            fs.makedirs(granule_dir)
            written[band] = to_cog(tmp_tif, dst_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"granule_dir": granule_dir, "written": written, "missing": missing}


# --- orchestration (I/O-thin; the logic above is what's unit-tested) --------


def build_fixture(
    archive_root: str,
    roi_path: str,
    fields_path: str,
    out_dir: str,
    bands: list[str],
    *,
    max_bytes: int,
    startdate: str | None = None,
    enddate: str | None = None,
    max_timestamps: int | None = None,
    dry_run: bool = False,
    progress: bool = True,
) -> dict:
    """spec 42 §3 steps 1-6, called by both `--dry-run` (Step 3) and the real
    build (Step 4). Returns the summary dict the CLI turns into `_result.json`.
    """
    catalog_gdf = fs.read_parquet(f"{archive_root.rstrip('/')}/catalog.parquet")
    roi_gdf = fs.read_geo(roi_path)

    sel = select_granules(catalog_gdf, roi_gdf, startdate, enddate)
    if not len(sel):
        raise ValueError(
            f"no granule intersects the cell within [{startdate}, {enddate}] -- "
            "widen the window or check --roi."
        )
    if max_timestamps is not None:
        sel = subsample_timestamps(sel, max_timestamps)

    tiles = sorted({t for t in (mgrs_tile_of(i) for i in sel["id"]) if t})
    radiometry_col = collapse_radiometry_column(sel)

    if dry_run:
        estimate = estimate_bytes(sel, roi_gdf, bands)
        return {
            "granules": int(len(sel)),
            "mgrs_tiles": tiles,
            "bands": list(bands),
            "declaration_column": radiometry_col or "derived-from-id",
            "per_band_bytes": estimate["per_band_bytes"],
            "total_bytes": estimate["total_bytes"],
            "max_bytes": int(max_bytes),
            "under_cap": estimate["total_bytes"] < max_bytes,
        }

    fs.makedirs(out_dir)

    rows: list[dict] = []
    offset_sources: list[str] = []
    total_bytes = 0
    n = len(sel)
    t0 = time.time()
    for i, (_, row) in enumerate(sel.iterrows(), start=1):
        offset, offset_source = row_offset(row, radiometry_col)
        offset_sources.append(offset_source)
        if offset is None:
            raise ValueError(
                f"granule {row['id']!r}: no radiometry declaration "
                f"({'/'.join(RADIOMETRY_COLUMNS)}) and no derivable baseline "
                "token in its id -- refuse to guess (spec 42 D1/A1)."
            )

        clipped = clip_granule(row, bands, roi_gdf, out_dir)
        written_bands = sorted(clipped["written"])
        total_bytes += sum(clipped["written"].values())
        if not written_bands:
            raise ValueError(f"granule {row['id']!r}: none of {bands} were found/clipped.")

        # Recompute geometry from the clipped raster (reference band if
        # present, else the first band actually written) -- never the source
        # footprint (spec 42 §3 step 3).
        geom_band = config.REFERENCE_BAND if config.REFERENCE_BAND in written_bands else written_bands[0]
        geometry = _geom_from_cog(os.path.join(clipped["granule_dir"], f"{geom_band}.tif"))

        rows.append({
            "id": row["id"],
            "satellite": row.get("satellite", config.SATELLITE_S2L2A),
            "timestamp": row["timestamp"],
            "s3url": row.get("s3url", ""),
            # Whatever --out was passed as, joined with the granule id -- run-book
            # 43 always passes a repo-root-relative --out (`tests/data/tutorial`,
            # invoked from `cd fsd/`), so this lands as a portable relative path
            # in the committed fixture, not an absolute one tied to this machine.
            "local_folderpath": clipped["granule_dir"],
            "files": ",".join(f"{b}.tif" for b in written_bands),
            "cloud_cover": row.get("cloud_cover", 0.0),
            "offset": int(offset),
            "nodata": config.NODATA,
            "geometry": geometry,
        })

        if progress:
            elapsed = time.time() - t0
            rate = elapsed / i
            eta = rate * (n - i)
            print(
                f"[build_fixture] {i}/{n} {row['id']}  "
                f"offset={offset} ({offset_source})  elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    out_gdf = gpd.GeoDataFrame(rows, columns=CATALOG_COLUMNS, geometry="geometry", crs="EPSG:4326")
    to_attrs(out_gdf, S2_L2A_DECLARATION)
    fs.write_parquet(os.path.join(out_dir, "catalog.parquet"), out_gdf)

    _place_beside(fields_path, os.path.join(out_dir, "fields.geojson"))
    _place_beside(roi_path, os.path.join(out_dir, "roi.geojson"))
    fs.write_text(os.path.join(out_dir, "NOTICE"), NOTICE_TEXT)

    date_span = [str(sel["timestamp"].min().date()), str(sel["timestamp"].max().date())] if len(sel) else [None, None]
    offsets = sorted({r["offset"] for r in rows})
    sources = _count_sources(offset_sources)

    _write_readme(
        out_dir, argv=sys.argv, granule_ids=list(sel["id"]), bands=bands,
        tiles=tiles, offsets=offsets, declaration_column=radiometry_col or "derived-from-id",
        date_span=date_span, total_bytes=total_bytes,
    )

    return {
        "granules": int(len(sel)),
        "mgrs_tiles": tiles,
        "bands": list(bands),
        "declaration_column": radiometry_col or "derived-from-id",
        "total_bytes": int(total_bytes),
        "max_bytes": int(max_bytes),
        "under_cap": total_bytes <= max_bytes,
        "offsets": offsets,
        # A2: WHERE each offset came from. "declared" == copied from the source
        # catalog's own column (A1's primary path, the "copied not invented"
        # evidence); "derived" == D1's id-token fallback. The offline suite
        # cannot distinguish these -- this result JSON is the only gate that can.
        "offset_sources": sources,
        "all_offsets_declared": sources.get("declared", 0) == len(rows),
        "date_span": date_span,
    }


def _place_beside(src_path: str, dst_path: str) -> None:
    """Copy `src_path` -> `dst_path` unless they are the same object.

    Run-book 43 Step 4 passes `--roi tests/data/tutorial/roi.geojson --fields
    tests/data/tutorial/fields.geojson --out tests/data/tutorial`, i.e. the
    generator's inputs ARE its outputs. Re-serializing them there truncates the
    destination before writing, so a crash mid-write destroys Step 0's output --
    which cannot be regenerated on the VM, because Step 0 needs `shapefiles/`
    from the workspace root. So: same path -> do nothing; different path ->
    `storage.transfer`, which is atomic (.part + rename), copies bytes verbatim
    (no CRS/precision round-trip through GeoPandas) and works when `--out` is an
    `abfss://` url (A1 D4: all generator I/O through `rise_eo.storage`, ADR 0003).
    """
    if not fs.exists(src_path):
        raise FileNotFoundError(f"{src_path!r} does not exist (run-book 43 Step 0 writes it).")
    if _same_path(src_path, dst_path):
        return
    fs.transfer(src_path, dst_path)


def _same_path(a: str, b: str) -> bool:
    """Same object? Exact string first (the only safe test for a url), then
    `abspath` for plain local paths -- never `abspath` on an `abfss://` url,
    which it would mangle into a relative-looking string."""
    if a == b:
        return True
    if "://" in a or "://" in b:
        return False
    return os.path.abspath(a) == os.path.abspath(b)


def redact_argv(argv: list[str]) -> list[str]:
    """Replace the VALUE of every url-bearing flag with a placeholder.

    `tests/data/tutorial/README.md` is committed to a **public MIT repo** (spec
    42 D6), and run-book 43 Step 4 is invoked as
    `--archive-root "$AZ_ARCHIVE_ROOT/archive"` -- the shell expands that, so a
    verbatim `' '.join(sys.argv)` would publish the concrete
    `abfss://<fs>@<account>.dfs.core.windows.net/<prefix>` url. Concrete
    infrastructure identifiers live only in `AZURE_INFRA_PRIVATE.md` at the
    workspace root, never under `fsd/`.
    """
    redacted: list[str] = []
    skip_next = False
    for i, tok in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        flag, sep, value = tok.partition("=")
        if flag in _REDACTED_FLAGS:
            if sep:
                redacted.append(f"{flag}=<{flag.lstrip('-')}>")
            else:
                redacted.append(tok)
                if i + 1 < len(argv):
                    redacted.append(f"<{flag.lstrip('-')}>")
                    skip_next = True
            continue
        redacted.append(_REDACTED_URL if "://" in tok else tok)
    return redacted


# Flags whose VALUE may be a concrete infrastructure identifier.
_REDACTED_FLAGS = ("--archive-root",)
_REDACTED_URL = "<url>"


def _write_readme(out_dir, *, argv, granule_ids, bands, tiles, offsets, declaration_column,
                   date_span, total_bytes) -> None:
    argv = redact_argv(list(argv))
    lines = [
        "# Tutorial micro-fixture (spec 42)",
        "",
        NOTICE_TEXT.strip(),
        "",
        f"MGRS tile(s): {', '.join(tiles)}",
        f"Bands: {', '.join(bands)}",
        f"Granules: {len(granule_ids)}",
        f"Radiometric offset(s): {offsets} (declaration column: {declaration_column})",
        f"Date span: {date_span[0]} .. {date_span[1]}",
        f"Total size: ~{total_bytes / 1e6:.1f} MB",
        "",
        f"Generator invocation: `{' '.join(argv)}`",
        "",
        "## Source granule ids",
        *[f"- {g}" for g in granule_ids],
        "",
        "Source archive root is intentionally not recorded here (public MIT repo) --",
        "see AZURE_INFRA_PRIVATE.md at the workspace root for the concrete value.",
    ]
    fs.write_text(os.path.join(out_dir, "README.md"), "\n".join(lines) + "\n")


# --- CLI ---------------------------------------------------------------------


def _write_result(path: str, result: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python tests/data/tutorial/build_fixture.py",
        description="Spec 42 §3 / run-book 43 Steps 2-4: build the tutorial "
                     "micro-fixture by clipping one grid cell out of the archive.",
    )
    p.add_argument("--archive-root", required=True, help="archive root (contains catalog.parquet)")
    p.add_argument("--roi", required=True, help="path to roi.geojson (from Step 0)")
    p.add_argument("--fields", default=None, help="path to fields.geojson (from Step 0)")
    p.add_argument("--out", default=None, help="output dir for the fixture")
    p.add_argument("--bands", nargs="+", default=None, help="band list, e.g. B04 B08 SCL")
    p.add_argument("--max-bytes", type=int, default=None, help="hard stop, spec 42 D2 (30 MB)")
    p.add_argument("--startdate", default=None,
                   help="inclusive start of the acquisition window, YYYY-MM-DD "
                        "(spec 42 D2: Apr-Sep 2018)")
    p.add_argument("--enddate", default=None,
                   help="inclusive end of the acquisition window, YYYY-MM-DD")
    p.add_argument("--max-timestamps", type=int, default=None,
                   help="fallback cap on granule count if --dry-run reports over --max-bytes "
                        "(spec 42 D2: drop timestamps before dropping bands)")
    p.add_argument("--dry-run", action="store_true", help="counts + bytes, zero writes")
    p.add_argument("--check-only", action="store_true",
                   help="Step 2: read-only archive/precondition check, zero writes")
    p.add_argument("--result", required=True, help="path to write the _result.json")
    p.add_argument("--quiet", action="store_true", help="suppress per-granule progress lines")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.check_only:
        step = "step2_check_archive"
        try:
            catalog_gdf = fs.read_parquet(f"{args.archive_root.rstrip('/')}/catalog.parquet")
            roi_gdf = fs.read_geo(args.roi)
            summary = check_archive(catalog_gdf, roi_gdf, bands=args.bands,
                                    startdate=args.startdate, enddate=args.enddate)
        except Exception as e:  # noqa: BLE001
            _write_result(args.result, {
                "step": step, "status": "failed", "pass": False,
                "metrics": {}, "expected": {}, "error": repr(e),
            })
            raise
        print(
            f"granules intersecting cell : {summary['granules']}"
            + (f"   (of {summary['granules_before_date_filter']} before the "
               f"{summary['date_window'][0]}..{summary['date_window'][1]} window)"
               if summary["date_window"] != [None, None] else "   (NO DATE WINDOW -- see spec 42 D2)")
            + "\n"
            f"single MGRS tile           : {summary['single_tile']} {summary['mgrs_tiles']}\n"
            f"bands present              : {', '.join(summary['bands'])}"
            + (f"   (missing: {summary['missing_bands']})" if summary.get("missing_bands") else "")
            + "\n"
            f"radiometry declared        : {summary['declared_non_null']}/{summary['granules']} "
            f"rows non-null   (column: {summary['declaration_column']!r})\n"
            f"offset value(s)            : {summary['offset_values']}  sources={summary['offset_sources']}\n"
            f"date span                  : {summary['date_span'][0]} .. {summary['date_span'][1]}"
        )
        # Run-book 43 Step 2's stated PASS conditions, ALL of them -- not just
        # the ones that were convenient to compute (single tile / declaration /
        # non-empty). `--bands` is optional; when omitted the band condition
        # cannot be evaluated and is not silently treated as satisfied, it is
        # reported as unchecked.
        passed = bool(
            summary["single_tile"]
            and summary["granules"] > 0
            and summary["declared_non_null"] == summary["granules"]
            and summary["date_span"][0] is not None
            and summary.get("all_bands_present", True)
        )
        _write_result(args.result, {
            "step": step, "status": "ok" if passed else "failed", "pass": passed,
            "metrics": summary,
            "expected": {
                "single_tile": True,
                "declared_non_null_equals_granules": True,
                "all_bands_present": True if args.bands else "unchecked (--bands not given)",
                "date_span_non_empty": True,
            },
            "error": None,
        })
        return 0 if passed else 1

    missing = [n for n in ("fields", "out", "bands", "max_bytes") if getattr(args, n) is None]
    if missing:
        raise SystemExit(f"--{'/--'.join(missing)} required unless --check-only")

    step = "step3_dry_run" if args.dry_run else "step4_build"
    try:
        summary = build_fixture(
            args.archive_root, args.roi, args.fields, args.out, args.bands,
            max_bytes=args.max_bytes, max_timestamps=args.max_timestamps,
            startdate=args.startdate, enddate=args.enddate,
            dry_run=args.dry_run, progress=not args.quiet,
        )
    except Exception as e:  # noqa: BLE001 - still leave a pasteable result, then re-raise
        _write_result(args.result, {
            "step": step, "status": "failed", "pass": False,
            "metrics": {}, "expected": {}, "error": repr(e),
        })
        raise

    if args.dry_run:
        cap_msg = "under" if summary["under_cap"] else "OVER"
        print(
            f"per-band estimated bytes: {summary['per_band_bytes']}\n"
            f"projected total: {summary['total_bytes'] / 1e6:.1f} MB "
            f"({summary['granules']} granules x {len(summary['bands'])} bands) "
            f"-- {cap_msg} the {summary['max_bytes'] / 1e6:.1f} MB cap"
        )
    else:
        cap_msg = "under" if summary["under_cap"] else "OVER"
        print(
            f"built {summary['granules']} granules, {summary['total_bytes'] / 1e6:.1f} MB total "
            f"-- {cap_msg} the {summary['max_bytes'] / 1e6:.1f} MB cap\n"
            f"offset(s) {summary['offsets']} from {summary['declaration_column']!r} "
            f"-- sources={summary['offset_sources']} "
            f"(all declared by the source: {summary['all_offsets_declared']})"
        )

    passed = summary["under_cap"] and summary["granules"] > 0
    _write_result(args.result, {
        "step": step, "status": "ok" if passed else "failed", "pass": bool(passed),
        "metrics": summary,
        "expected": {"under_cap": True},
        "error": None,
    })
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
