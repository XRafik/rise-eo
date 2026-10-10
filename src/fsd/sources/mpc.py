"""MPC source: Sentinel-2 L2A discovery and download of granules that are already COG.

Spec: specs/32-mpc-source-baseline-harmonization.md

MPC serves S2 L2A assets already as COG, so a download is a byte copy via
`fsd.storage.transfer` (signed HTTPS -> local), with no jp2->COG conversion. Access is
anonymous; an optional `PC_SDK_SUBSCRIPTION_KEY` env var (read by `planetary-computer`)
raises rate limits.

⚠️ MPC serves raw, UNHARMONIZED DN and has no `raster:bands`. The per-band offset must be
derived from `s2:processing_baseline` (`_s2_radiometry.offset_for_item`) and stored as the
additive `offset` column, or every cube is off by the baseline offset.

So the copy is not *pure*: ingest then stamps scale/offset (and nodata if missing) via
`fsd.raster.cog.stamp_or_reencode`, a header edit with no pixel decode, and pushes the
result to `root_folderpath`.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
from collections.abc import Mapping, Sequence
from typing import Callable

import geopandas as gpd
import pandas as pd
import shapely

from fsd import collections as _collections
from fsd import config
from fsd.catalog import processing as processing_module
from fsd.catalog.declaration import CollectionDeclaration
from fsd.raster.cog import stamp_or_reencode
from fsd.sources._granules import (
    granule_columns,
    granule_folderpath,
    item_granule,
    report_download,
    require_valid_processing,
    select_granules,
    surviving_items,
)
from fsd.sources._s2_radiometry import offset_for_item
from fsd.sources.cdse import _finalize_catalog_gdf, _is_local_path, _roi_gdf
from fsd.storage import fs

__all__ = [
    "DownloadResult",
    "query_catalog",
    "download",
    "discover_shard_rows",
    "download_shard",
]

# The value of the catalog `source` column for rows this module writes (spec 59 D8).
SOURCE = "mpc"

# The collections this source serves (spec 58 D15). MPC also hosts hls2-s30/hls2-l30
# (see specs/58 D17/P3), but P1/P2 register no declaration for them yet, so this
# source-x-collection guard names only what fsd can actually build against today --
# extended in P3 once that collection's declaration ships. sentinel-1-rtc joins here in
# P2 (D17): MPC serves it anonymously, same as S2 L2A (D10, retracted).
SERVED_COLLECTIONS = (config.SATELLITE_S2L2A, "sentinel-1-rtc")


@dataclasses.dataclass
class DownloadResult:
    successful_count: int
    total_count: int
    skipped_count: int = 0
    failed_count: int = 0
    elapsed_s: float = 0.0
    failures: list = dataclasses.field(default_factory=list)  # (src_url, reason)


# --- catalog discovery -------------------------------------------------------


def _item_self_href(item) -> str:
    """Best-effort item self-href (informational `s3url` column); "" if unset."""
    getter = getattr(item, "get_self_href", None)
    if callable(getter):
        try:
            return getter() or ""
        except Exception:  # noqa: BLE001 - purely informational, never fatal
            return ""
    return getattr(item, "self_href", None) or ""


def _import_pc():
    """Import `planetary_computer`, naming the extra rather than the package.

    Both MPC entry points reach the package lazily, so without this a user who installed
    fsd without `[mpc]` gets a bare `ModuleNotFoundError: planetary_computer` -- a package
    name they never typed, from a call they made as `source="mpc"`. Same shape as
    `grid.roi_to_s2_grids` -> `[grid]` and `runners._require_snakemake` -> `[local]` (#80).
    """
    try:
        import planetary_computer as pc
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "source='mpc' needs the optional '[mpc]' extra: pip install 'fsd[mpc]' "
            "(brings planetary-computer, which signs MPC asset hrefs). "
            "On an AML node this means the node IMAGE was built without it."
        ) from exc
    return pc


def _search_items(
    roi_gdf: gpd.GeoDataFrame, startdate, enddate, max_cloudcover=None,
    collection: str = config.SATELLITE_S2L2A,
):
    """Query the MPC STAC API for `collection` items intersecting the ROI, signed via
    the official `planetary-computer` package."""
    import pystac_client

    pc = _import_pc()

    geom = shapely.unary_union(roi_gdf.to_crs("EPSG:4326")["geometry"])
    client = pystac_client.Client.open(config.MPC_STAC_URL, modifier=pc.sign_inplace)
    query = None
    if max_cloudcover is not None:
        query = {"eo:cloud_cover": {"lt": max_cloudcover}}
    search = client.search(
        collections=[collection],
        datetime=[startdate, enddate],
        intersects=geom,
        query=query,
        limit=200,
    )
    return list(search.items())


def _search_items_unsigned(
    roi_gdf: gpd.GeoDataFrame, startdate, enddate, max_cloudcover=None,
    collection: str = config.SATELLITE_S2L2A,
):
    """Same query as `_search_items`, but **without** the `pc.sign_inplace` modifier
: the AML fan-out's driver-side discovery must not stamp asset
    hrefs with a SAS token that can expire before a job actually runs on its node --
    signing happens **on the node**, in `download_shard`, right before the transfer."""
    import pystac_client

    geom = shapely.unary_union(roi_gdf.to_crs("EPSG:4326")["geometry"])
    client = pystac_client.Client.open(config.MPC_STAC_URL)
    query = None
    if max_cloudcover is not None:
        query = {"eo:cloud_cover": {"lt": max_cloudcover}}
    search = client.search(
        collections=[collection],
        datetime=[startdate, enddate],
        intersects=geom,
        query=query,
        limit=200,
    )
    return list(search.items())


def _items_to_gdf(
    items, *, collection: str, declaration: CollectionDeclaration,
) -> gpd.GeoDataFrame:
    """Parse MPC STAC items into a catalog GeoDataFrame. Pure — no network — so
    it is unit-testable with duck-typed fake items (`.id`, `.datetime`,
    `.geometry`, `.properties`, `.assets[*].href`)."""
    # `offset_for_item` reads S2's processing-baseline properties, which a
    # non-radiometric collection's items do not carry -- it raises for them. A
    # declaration with `radiometry_bands=()` has no offset to derive (spec 58 D17).
    needs_offset = declaration.radiometry_bands != ()
    rows = [
        {
            # `id` is the canonical granule name (spec 59 D3), not MPC's item id.
            **granule_columns(item_granule(collection, it), source=SOURCE),
            "collection": collection,
            "timestamp": it.datetime,
            "s3url": _item_self_href(it),
            "cloud_cover": it.properties.get("eo:cloud_cover"),
            "offset": offset_for_item(it) if needs_offset else 0,
            "scale": declaration.scale,
            "nodata": config.NODATA,
            "properties": json.dumps(dict(it.properties)),
            "geometry": shapely.geometry.shape(it.geometry),
        }
        for it in items
    ]
    gdf = gpd.GeoDataFrame(
        rows, columns=["id", "collection", "timestamp", "s3url", "cloud_cover",
                       "offset", "scale", "nodata", "acquisition_key", "processing_version",
                       "processing_datetime", "source", "properties", "geometry"],
        geometry="geometry", crs="EPSG:4326",
    )
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], utc=True)
    gdf["processing_datetime"] = pd.to_datetime(gdf["processing_datetime"], utc=True)
    return gdf


def query_catalog(
    roi,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    *,
    max_cloudcover: float | None = None,
    collection: str = config.SATELLITE_S2L2A,
    processing: str | None = processing_module.LATEST,
) -> gpd.GeoDataFrame:
    """Discover `collection` granules intersecting `roi` within the date range, via the
    MPC STAC API (anonymous by default).

    Returns a GeoDataFrame: id (canonical granule name), collection, timestamp, s3url,
    cloud_cover, offset, scale, nodata, acquisition_key, processing_version,
    processing_datetime, source, properties, geometry (EPSG:4326). Asserts id uniqueness.
    `processing` is applied per acquisition (spec 59 D7); default `"latest"` keeps spec 33's
    behaviour of one item per acquisition.
    """
    declaration = _collections.get(collection)
    roi_gdf = _roi_gdf(roi)
    items = _search_items(roi_gdf, startdate, enddate, max_cloudcover=max_cloudcover,
                           collection=collection)
    gdf = _items_to_gdf(items, collection=collection, declaration=declaration)
    gdf = _finalize_catalog_gdf(gdf, roi_gdf, max_cloudcover)
    return processing_module.select_processing(gdf, processing).kept


# --- granule download (byte-copy + GDAL metadata stamp) ----------------------


def _select_item_files(
    item, bands: list[str], root_folderpath: str, *,
    collection: str = config.SATELLITE_S2L2A,
    declaration: CollectionDeclaration | None = None,
) -> list[tuple[str, str, str]]:
    """Select download files from an MPC item's `assets` — MPC keys bands
    directly (`"B04"`, `"SCL"`, …), simpler than CDSE's `Bxx_YYm`. Returns
    `[(signed_href, local_dst_path, native_band), ...]`.

    `bands` may be canonical STAC EO `common_name`s (e.g. `"nir08"`) or
    already-native asset keys (e.g. `"B8A"`) — `declaration.canonical_to_native`
    normalizes either spelling to the same native key, so `bands=["B8A"]` and
    `bands=["nir08"]` select the identical asset (and, upstream, resolve to the same
    cube path).

    **Raises**, naming the band and collection, when a requested band does not exist on
    this item, rather than skipping it: a skipped band lets a cube build without it.

    Spec: 58 D8."""
    if declaration is None:
        declaration = _collections.get(collection)
    # Spec 59 D2/D3: `{root}/{collection}/YYYY/MM/DD/{canonical granule name}/`.
    dst_folder = granule_folderpath(root_folderpath, collection, item_granule(collection, item))
    selected = []
    for band in bands:
        native = declaration.canonical_to_native(band)
        asset = item.assets.get(native)
        if asset is None:
            raise ValueError(
                f"sources.mpc: requested band {band!r} (native key {native!r}) is not "
                f"available on item {item.id!r} of collection {collection!r}; available "
                f"assets: {sorted(item.assets)}."
            )
        selected.append((asset.href, os.path.join(dst_folder, f"{native}.tif"), native))
    return selected


def _failure_reason(exc: BaseException | None) -> str:
    """`"<ExceptionType>: <message>"` -- the type FIRST, because the type is the diagnosis.

    fsspec/adlfs raise `FileNotFoundError(url)`, so `str(exc)` alone is just the asset
    URL and groups nothing.
    """
    if exc is None:
        return "unknown"
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _failure_kind(reason: str) -> str:
    """The GROUPING key for a failure: the exception type, not the whole message.

    The message is the asset URL, unique per file, so grouping on the full string makes
    each failure its own group. Cutting at the first `": "` leaves the type.
    """
    head = str(reason).splitlines()[0] if reason else ""
    kind, sep, _rest = head.partition(": ")
    # Only treat the head as a type name when it looks like one -- a reason with no type
    # prefix (or a message containing an early colon) keeps its own first line.
    if sep and kind and " " not in kind:
        return kind
    return head[:120] or "unknown"


def _print_failure_summary(failures: list[tuple[str, str]], *, total: int) -> None:
    """Print WHY transfers failed, grouped by exception type -- not just how many.

    `api.download` discards the whole result, so this print is the only place the reasons
    surface. Grouped by `_failure_kind`, one example per kind: the count says how bad,
    the example says what to do.
    """
    if not failures:
        return
    import collections

    by_kind: collections.Counter = collections.Counter()
    example: dict[str, tuple[str, str]] = {}
    for src, reason in failures:
        kind = _failure_kind(reason)
        by_kind[kind] += 1
        example.setdefault(kind, (src, reason))

    print(f"[fsd.mpc.download] {len(failures)}/{total} transfers FAILED, by kind:",
          flush=True)
    for kind, n in by_kind.most_common(10):
        src, reason = example[kind]
        print(f"[fsd.mpc.download]   {n:5d} x {kind}", flush=True)
        # A rasterio/adlfs failure arrives as a kilobyte traceback; the first line is the
        # diagnosis and the url is what you retry or paste into a bug report.
        print(f"[fsd.mpc.download]           e.g. {str(reason).splitlines()[0][:200]}",
              flush=True)
        print(f"[fsd.mpc.download]           url  {src[:200]}", flush=True)
    if len(by_kind) > 10:
        print(f"[fsd.mpc.download]   ... and {len(by_kind) - 10} more kind(s)", flush=True)


def _transfer_and_stamp_one(
    src_url: str, dst_path: str, *, band: str, offset: int,
    declaration: CollectionDeclaration,
    sign: Callable[[str], str] | None = None,
    tries: int = 3, base_delay: float = 0.5,
) -> tuple[bool, str]:
    """Byte-copy one already-COG asset, then stamp the declared GDAL scale/offset
    (reflectance bands only) + nodata-if-missing tags. Returns `(ok, reason)`.

    A header edit, not a pixel decode -- though `fsd.raster.cog.stamp_or_reencode` does
    fall back to a GDAL-COG-driver re-encode if the in-place stamp would break COG validity.
    Idempotent skip on an existing non-empty `dst_path`. Stamping needs a real LOCAL file,
    so a remote `dst_path` is staged in local scratch, stamped, then pushed.

    `dst_path` is created by the LAST step and never edited afterwards (#74): a local
    destination is staged as `<dst>.stage`, stamped, and only then `os.replace`d onto
    `dst_path`. A kill or stamp error before that leaves no `dst_path`, so the `size > 0`
    skip only ever sees fully stamped files.

    `sign`, when given, is applied **inside the retry loop, immediately before each
    attempt**. An MPC SAS token lives ~45 min and a whole-archive run is longer, so signing
    at discovery fails every asset still queued when the token ages out; per-attempt
    signing also re-signs a retry after a long queue wait.
    """
    import shutil
    import tempfile
    import time

    if fs.exists(dst_path) and fs.size(dst_path) > 0:
        return True, "skipped"

    local = _is_local_path(dst_path)
    scratch_dir = None
    # `.stage`, not `.part`: `fs.transfer` already uses `<dst>.part` for its own sidecar,
    # and reusing the suffix would give `B04.tif.part.part` mid-copy (spec 59 D10).
    scratch = dst_path + ".stage"
    if not local:
        scratch_dir = tempfile.mkdtemp(prefix="fsd_mpc_")
        scratch = os.path.join(scratch_dir, os.path.basename(dst_path))

    is_reflectance = declaration.is_radiometry_band(band)
    last: Exception | None = None
    try:
        for attempt in range(tries):
            try:
                fs.transfer(sign(src_url) if sign is not None else src_url, scratch)
                stamp_or_reencode(
                    scratch,
                    # reflectance-unit offset to match the declared scale: a
                    # viewer's unscale=true computes DN*scale + offset, so the DN-space
                    # offset (-1000) must be scaled to reflectance too (-> -0.1), else
                    # unscale yields DN/10000 - 1000 ~= -1000 for every pixel (black tile).
                    offset=offset * declaration.scale if is_reflectance else 0.0,
                    scale=declaration.scale if is_reflectance else 1.0,
                    set_nodata_if_missing=config.NODATA,
                )
                if local:
                    os.replace(scratch, dst_path)   # the ONLY step that creates dst_path
                else:
                    fs.put(scratch, dst_path)
                return True, "ok"
            except Exception as e:  # noqa: BLE001 - retried below; final failure reported
                last = e
                if attempt == tries - 1:
                    break
                time.sleep(base_delay * (2**attempt))
        return False, _failure_reason(last)
    finally:
        if scratch_dir is not None:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        elif os.path.exists(scratch):
            try:
                os.remove(scratch)
            except OSError:
                pass


def _none_if_nan(value):
    """A shard-CSV round trip turns a null into NaN / ""; the catalog wants `None`."""
    if value is None or value == "" or (not isinstance(value, str) and pd.isna(value)):
        return None
    return value


def _append_downloaded(
    catalog, tile_meta: dict, results: list[tuple], declaration: CollectionDeclaration,
) -> int:
    """Group successful (tile_id, dst, ok) downloads by granule and upsert catalog
    rows. Mirrors `cdse._append_downloaded`, plus `offset`/`scale`/`nodata`/`properties`."""
    import collections

    files_by_tile = collections.defaultdict(list)
    folder_by_tile: dict[str, str] = {}
    for tile_id, dst, ok in results:
        if not ok:
            continue
        files_by_tile[tile_id].append(os.path.basename(dst))
        folder_by_tile[tile_id] = os.path.dirname(dst)

    rows = []
    for tile_id, files in files_by_tile.items():
        r = tile_meta[tile_id]
        rows.append({
            "id": tile_id,
            "collection": r["collection"],
            "timestamp": r["timestamp"],
            "s3url": r["s3url"],
            "local_folderpath": folder_by_tile[tile_id],
            "files": ",".join(sorted(files)),
            "cloud_cover": r["cloud_cover"],
            "offset": r["offset"],
            "scale": r.get("scale", declaration.scale),
            "nodata": r["nodata"],
            "acquisition_key": r.get("acquisition_key", tile_id),
            "processing_version": _none_if_nan(r.get("processing_version")),
            "processing_datetime": _none_if_nan(r.get("processing_datetime")),
            "source": SOURCE,
            "properties": r.get("properties", "{}"),
            "geometry": r["geometry"],
        })
    if rows:
        # Stamp the collection-level declaration at the one place this source appends
        # to the catalog (spec 58 D2: resolved once, driver-side, not re-derived per row).
        catalog.append(rows, declaration=declaration)
    return sum(len(f) for f in files_by_tile.values())


def download(
    roi,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    bands: list[str],
    root_folderpath: str,
    catalog,                      # fsd.catalog.catalog.TileCatalog (appended in place)
    *,
    max_tiles: int,
    max_cloudcover: float | None = None,
    progress: bool = False,
    max_concurrent: int | None = None,
    should_stop: Callable[[], bool] | None = None,
    collection: str = config.SATELLITE_S2L2A,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str = processing_module.LATEST,
) -> DownloadResult:
    """Discover matching MPC `collection` granules and download the requested band files
    to `root_folderpath`, local or remote/blob. No credentials required: MPC is anonymous.

    `processing` selects ONE processing per acquisition among what MPC offers: `"latest"`
    (default) or a PEP 440 specifier. It applies after `properties_filter` and before
    `max_tiles`; `None` raises. Every skipped granule is printed, and so is any
    acquisition this download leaves holding more than one processing in the archive.

    Assets are already COG, so this is a thread-pool transfer + stamp with no convert
    process pool. Idempotent: files already on disk are skipped. `should_stop` (optional)
    halts new submissions only, but every unit is submitted up front, so once the run
    starts it halts nothing (#157).

    `properties_filter` narrows the discovered granules by STAC property, e.g.
    `{"sat:orbit_state": "descending"}`, **before** the `max_tiles` cap, so the cap counts
    what will be transferred: a transfer is a whole-asset byte copy, and a partitioned
    collection like `sentinel-1-rtc` returns orbits a build can never use. A key no
    discovered granule carries raises, naming the keys they do carry.

    Spec: 59 D7, 58 D9.
    """
    import concurrent.futures
    import time

    declaration = _collections.get(collection)
    require_valid_processing(processing, collection)

    if _is_local_path(root_folderpath):
        fs.makedirs(root_folderpath, exist_ok=True)

    roi_gdf = _roi_gdf(roi)
    # Unsigned discovery, sign per transfer: a SAS token lives ~45 min and a whole-archive
    # download runs longer.
    sign = _import_pc_sign()
    items = _search_items_unsigned(roi_gdf, startdate, enddate, max_cloudcover=max_cloudcover,
                                    collection=collection)
    tiles = _finalize_catalog_gdf(
        _items_to_gdf(items, collection=collection, declaration=declaration),
        roi_gdf, max_cloudcover,
    )

    # Applied BEFORE the cap: `max_tiles` is a guardrail on bytes about to be moved, so it
    # must count the granules this run will actually transfer, not the ones discovery saw.
    n_discovered = len(tiles)
    tiles = select_granules(tiles, processing=processing, prefix="[fsd.mpc.download]",
                           properties_filter=properties_filter)

    if len(tiles) > max_tiles:
        narrowed = ""
        if properties_filter:
            narrowed = (f" (already narrowed from {n_discovered} by "
                        f"properties_filter={dict(properties_filter)!r})")
        raise ValueError(
            f"{len(tiles)} matched tiles exceed max_tiles={max_tiles}{narrowed}. Narrow "
            "the query or raise max_tiles. Note each tile transfers its WHOLE asset "
            "file(s), not just the ROI window -- for a partitioned collection "
            "(e.g. sentinel-1-rtc) pass properties_filter to drop the orbits the build "
            "cannot use anyway."
        )

    tile_meta = {row["id"]: row for _, row in tiles.iterrows()}
    kept_items = surviving_items(items, tile_meta, collection)

    work: list[tuple[str, str, str, str, int]] = []
    for it, tile_id in kept_items:
        offset = tile_meta[tile_id]["offset"]
        for src, dst, band in _select_item_files(
            it, bands, root_folderpath, collection=collection, declaration=declaration,
        ):
            work.append((src, dst, tile_id, band, offset))

    workers = max_concurrent if max_concurrent is not None else config.MPC_MAX_CONCURRENT
    start = time.time()
    results: list[tuple[str, str, bool]] = []
    failures: list[tuple[str, str]] = []
    skipped = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {}
        for src, dst, tid, band, offset in work:
            if should_stop is not None and should_stop():
                break
            futs[pool.submit(_transfer_and_stamp_one, src, dst, band=band, offset=offset,
                              declaration=declaration, sign=sign)] = (src, dst, tid)
        for fut in concurrent.futures.as_completed(futs):
            src, dst, tid = futs[fut]
            ok, reason = fut.result()
            if reason == "skipped":
                skipped += 1
            if not ok:
                failures.append((src, reason))
            results.append((tid, dst, ok))
            if progress:
                print(
                    f"[fsd.mpc.download] {len(results)}/{len(work)} "
                    f"ok={sum(1 for *_, o in results if o)} fail={len(failures)}",
                    flush=True,
                )

    _print_failure_summary(failures, total=len(work))
    successful = _append_downloaded(catalog, tile_meta, results, declaration)
    report_download(catalog, tiles, processing=processing, source=SOURCE)

    return DownloadResult(
        successful_count=successful,
        total_count=successful + len(failures),
        skipped_count=skipped,
        failed_count=len(failures),
        elapsed_s=time.time() - start,
        failures=failures,
    )


# --- AML fan-out: driver-side discovery + per-shard download -----------------

def discover_shard_rows(
    roi,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    bands: list[str],
    root_folderpath: str,
    *,
    max_cloudcover: float | None = None,
    collection: str = config.SATELLITE_S2L2A,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str = processing_module.LATEST,
) -> list[dict]:
    """Driver-side discovery for the AML fan-out: query MPC STAC (no bytes; unsigned, so
    no href carries a token yet) and flatten the matched items to **one row per asset**.
    `run_aml_download` partitions the rows with `shard_units` and hands each partition to
    `download_shard`, which signs on the node.

    Every row shares `collection`, so the node resolves the declaration once from the
    registry. That is allowed: this is ingest, where the declaration is only artifact
    facts, not the build's collection-variant resolution that spec 58 D13 keeps
    registry-free.

    `properties_filter` narrows the granules as in `download()`, before any row exists, so
    `max_tiles` downstream counts only what will transfer. `processing` is applied here, on
    the driver, per acquisition, so the shard CSVs carry only the chosen processing and no
    node decides.
    """
    declaration = _collections.get(collection)
    require_valid_processing(processing, collection)
    roi_gdf = _roi_gdf(roi)
    items = _search_items_unsigned(roi_gdf, startdate, enddate, max_cloudcover=max_cloudcover,
                                    collection=collection)
    tiles = _finalize_catalog_gdf(
        _items_to_gdf(items, collection=collection, declaration=declaration),
        roi_gdf, max_cloudcover,
    )
    tiles = select_granules(tiles, processing=processing, prefix="[fsd.mpc.download]",
                           properties_filter=properties_filter)
    tile_meta = {row["id"]: row for _, row in tiles.iterrows()}
    kept_items = surviving_items(items, tile_meta, collection)

    rows: list[dict] = []
    for it, tile_id in kept_items:
        meta = tile_meta[tile_id]
        for href, dst, band in _select_item_files(
            it, bands, root_folderpath, collection=collection, declaration=declaration,
        ):
            rows.append({
                "tile_id": tile_id,
                "band": band,
                "href": href,
                "dst": dst,
                "offset": meta["offset"],
                "collection": meta["collection"],
                "timestamp": meta["timestamp"].isoformat(),
                "s3url": meta["s3url"],
                "cloud_cover": meta["cloud_cover"],
                "scale": meta["scale"],
                "nodata": meta["nodata"],
                "acquisition_key": meta["acquisition_key"],
                "processing_version": _none_if_nan(meta["processing_version"]) or "",
                "processing_datetime": (
                    "" if pd.isna(meta["processing_datetime"])
                    else meta["processing_datetime"].isoformat()),
                "properties": meta["properties"],
                "geometry": meta["geometry"].wkt,
            })
    return rows


def _import_pc_sign():
    """Lazy handle to `planetary_computer.sign` -- same injection-boundary pattern as
    `workflows.runners._import_aml_command`, so `download_shard` stays substitutable in tests
    without requiring the `[mpc]` extra."""
    return _import_pc().sign


def download_shard(
    rows: list[dict],
    root_folderpath: str,
    catalog,                      # fsd.catalog.catalog.TileCatalog (appended in place)
    *,
    max_concurrent: int | None = None,
    progress: bool = False,
) -> DownloadResult:
    """Download one pre-discovered shard of MPC assets -- one of the
    N per-node jobs `run_aml_download` fans a `discover_shard_rows` work list out
    to. Each `href` is unsigned: signed here, **on the node**, right
    before the transfer, so a SAS token never sits idle between AML job submit
    and the job actually starting. Reuses the same per-asset transfer `download()`
    uses (`_transfer_and_stamp_one`); `download()` itself is untouched.
    """
    import concurrent.futures
    import time

    sign = _import_pc_sign()

    if _is_local_path(root_folderpath):
        fs.makedirs(root_folderpath, exist_ok=True)

    # Every row of one shard shares `collection` (`discover_shard_rows` writes it
    # uniformly), so the declaration is resolved once from the first row -- via the
    # registry, same judgment call as `discover_shard_rows` (this is ingest, not the
    # build-variant path D13 guards).
    declaration = _collections.get(rows[0]["collection"]) if rows else None

    workers = max_concurrent if max_concurrent is not None else config.MPC_MAX_CONCURRENT
    start = time.time()
    results: list[tuple[str, str, bool]] = []
    failures: list[tuple[str, str]] = []
    skipped = 0
    tile_meta: dict[str, dict] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {}
        for row in rows:
            tid = row["tile_id"]
            tile_meta.setdefault(tid, {
                "collection": row["collection"],
                "timestamp": row["timestamp"],
                "s3url": row["s3url"],
                "cloud_cover": row["cloud_cover"],
                "offset": row["offset"],
                "scale": row.get("scale", declaration.scale if declaration else 1.0),
                "nodata": row["nodata"],
                "acquisition_key": row.get("acquisition_key", tid),
                "processing_version": row.get("processing_version"),
                "processing_datetime": row.get("processing_datetime"),
                "properties": row.get("properties", "{}"),
                "geometry": shapely.from_wkt(row["geometry"])
                if isinstance(row["geometry"], str) else row["geometry"],
            })
            # Signed inside the worker (not here at submit): a big shard's last rows would
            # otherwise queue behind the earlier ones holding a token minted at submit time.
            fut = pool.submit(
                _transfer_and_stamp_one, row["href"], row["dst"],
                band=row["band"], offset=row["offset"], declaration=declaration, sign=sign,
            )
            futs[fut] = (row["href"], row["dst"], tid)
        for fut in concurrent.futures.as_completed(futs):
            src, dst, tid = futs[fut]
            ok, reason = fut.result()
            if reason == "skipped":
                skipped += 1
            if not ok:
                failures.append((src, reason))
            results.append((tid, dst, ok))
            if progress:
                print(
                    f"[fsd.mpc.download_shard] {len(results)}/{len(rows)} "
                    f"ok={sum(1 for *_, o in results if o)} fail={len(failures)}",
                    flush=True,
                )

    _print_failure_summary(failures, total=len(rows))
    successful = _append_downloaded(catalog, tile_meta, results, declaration)

    return DownloadResult(
        successful_count=successful,
        total_count=successful + len(failures),
        skipped_count=skipped,
        failed_count=len(failures),
        elapsed_s=time.time() - start,
        failures=failures,
    )
