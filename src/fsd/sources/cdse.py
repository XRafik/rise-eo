"""CDSE source: Sentinel-2 L2A discovery + granule download. See specs/01-sources.md.

Discovery is the **CDSE STAC API** (`pystac-client`, anonymous — no credentials):
each STAC item's `assets` already carry the per-band S3 `href`s, so we never list a
`.SAFE` over S3 (that recursive listing was the flaky path: BUG-001, ADR 0009).
The only S3-authenticated operation is the byte `transfer` of each band file, done
through the generic, provider-agnostic transport in `fsd.storage` (no direct boto3).

This module also defines the documented `download(...)` source contract (OQ-3:
function-signature, not an ABC).
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import os
import shutil
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
from fsd.storage import fs

# The collections this source serves (spec 58 D15). CDSE only serves S2 L2A -- unlike
# MPC, it has no S1/HLS product at all.
SERVED_COLLECTIONS = (config.SATELLITE_S2L2A,)

# The value of the catalog `source` column for rows this module writes (spec 59 D8).
SOURCE = "cdse"

# Environment-variable names for the cloud/Batch path (CdseCredentials.from_env).
ENV_SH_CLIENT_ID = "CDSE_SH_CLIENT_ID"
ENV_SH_CLIENT_SECRET = "CDSE_SH_CLIENT_SECRET"
ENV_S3_ACCESS_KEY = "CDSE_S3_ACCESS_KEY"
ENV_S3_SECRET_KEY = "CDSE_S3_SECRET_KEY"
ENV_S3_KEYS_EXPIRE = "CDSE_S3_KEYS_EXPIRE"

# JSON keys of the legacy `cdse_credentials.json` (kept for compatibility).
_JSON_SH_CLIENT_ID = "sh_clientid"
_JSON_SH_CLIENT_SECRET = "sh_clientsecret"
_JSON_S3_ACCESS_KEY = "s3_access_key"
_JSON_S3_SECRET_KEY = "s3_secret_key"


@dataclasses.dataclass
class CdseCredentials:
    """CDSE credentials: SH client id/secret (catalog) + S3 keys (download).

    See specs/01-sources.md. Load from a gitignored JSON file (`from_json`, local
    dev) or from environment variables (`from_env`, cloud/Batch). Secret values are
    never printed — `__repr__` masks them.
    """

    sh_client_id: str | None = None       # catalog (Sentinel Hub)
    sh_client_secret: str | None = None
    s3_access_key: str | None = None      # granule download (S3)
    s3_secret_key: str | None = None
    s3_keys_expire: str | None = None     # optional ISO date (YYYY-MM-DD), informational

    def __repr__(self) -> str:
        def m(v):
            return "set" if v else "unset"

        return (
            f"CdseCredentials(sh_client_id={m(self.sh_client_id)}, "
            f"sh_client_secret={m(self.sh_client_secret)}, "
            f"s3_access_key={m(self.s3_access_key)}, "
            f"s3_secret_key={m(self.s3_secret_key)}, "
            f"s3_keys_expire={self.s3_keys_expire!r})"
        )

    @classmethod
    def _from_dict(cls, data: dict) -> "CdseCredentials":
        return cls(
            sh_client_id=data.get(_JSON_SH_CLIENT_ID),
            sh_client_secret=data.get(_JSON_SH_CLIENT_SECRET),
            s3_access_key=data.get(_JSON_S3_ACCESS_KEY),
            s3_secret_key=data.get(_JSON_S3_SECRET_KEY),
            s3_keys_expire=data.get("s3_keys_expire"),
        )

    @classmethod
    def from_json(cls, filepath: str, **storage_options) -> "CdseCredentials":
        """Load from a JSON file using the legacy `cdse_credentials.json` keys.

        Tolerates extra keys; picks up optional `s3_keys_expire`.
        """
        with fs.open(filepath, "r", **storage_options) as f:
            data = json.load(f)
        return cls._from_dict(data)

    @classmethod
    def from_json_str(cls, s: str) -> "CdseCredentials":
        """Sibling of `from_json`: parse an in-memory JSON string in the same key format,
        instead of a file path -- the shape a Key Vault secret comes back in, having no
        filepath of its own."""
        return cls._from_dict(json.loads(s))

    @classmethod
    def from_env(cls, environ: dict | None = None) -> "CdseCredentials":
        """Load from environment variables (cloud/Batch path)."""
        e = environ if environ is not None else os.environ
        return cls(
            sh_client_id=e.get(ENV_SH_CLIENT_ID),
            sh_client_secret=e.get(ENV_SH_CLIENT_SECRET),
            s3_access_key=e.get(ENV_S3_ACCESS_KEY),
            s3_secret_key=e.get(ENV_S3_SECRET_KEY),
            s3_keys_expire=e.get(ENV_S3_KEYS_EXPIRE),
        )

    def s3_storage_options(self) -> dict:
        """`storage_options` for the CDSE S3 endpoint, for `fsd.storage` calls.

        Includes connect/read timeouts so a stalled connection (common during CDSE's
        flaky windows, BUG-001) raises instead of hanging a worker forever; botocore's
        own retries are disabled so our `_transfer_one` layer owns retry/labeling.
        """
        return {
            "key": self.s3_access_key,
            "secret": self.s3_secret_key,
            "client_kwargs": {"endpoint_url": config.CDSE_S3_ENDPOINT_URL},
            "config_kwargs": {
                "connect_timeout": config.S3_CONNECT_TIMEOUT,
                "read_timeout": config.S3_READ_TIMEOUT,
                "retries": {"max_attempts": 1},
            },
        }

    def require_s3(self) -> None:
        """Raise if the S3 keys are missing. Discovery (STAC) is anonymous, so only
        the download step needs credentials — this is the check it uses."""
        missing = [
            name for name in ("s3_access_key", "s3_secret_key")
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(f"CdseCredentials missing S3 fields: {missing}")

    def is_expired(self, as_of: datetime.date | None = None) -> bool | None:
        """Whether the S3 keys are past `s3_keys_expire`. None if unknown."""
        if not self.s3_keys_expire:
            return None
        as_of = as_of or datetime.date.today()
        return datetime.date.fromisoformat(self.s3_keys_expire) < as_of


@dataclasses.dataclass
class DownloadResult:
    successful_count: int          # files on disk after the run (new + already-present)
    total_count: int              # files attempted
    skipped_count: int = 0        # already on disk (idempotent skip)
    failed_count: int = 0         # files that failed after retries (fast-fails)
    elapsed_s: float = 0.0
    failures: list = dataclasses.field(default_factory=list)   # (src_url, reason)
    reason_counts: dict = dataclasses.field(default_factory=dict)  # {reason: count}
    circuit_tripped: bool = False  # stopped early: too many consecutive failures
    pool_broken: bool = False     # convert process pool died mid-run (segfault/OOM); resume with a fresh pool
    stopped: bool = False         # user requested a clean stop (should_stop); not a failure
    # --- timing decomposition — summed across worker threads, so
    # transfer_seconds + convert_seconds may exceed elapsed_s (they overlap). bytes_downloaded is
    # the JP2 bytes actually pulled from CDSE (basis for throughput MB/s); skips contribute 0. ---
    bytes_downloaded: int = 0            # JP2 bytes transferred this run (excludes skipped)
    transfer_seconds: float = 0.0        # summed CDSE byte-transfer wall-time (across threads)
    convert_seconds: float = 0.0         # summed local JP2->COG conversion wall-time
    # Wall-clock span the transfer phase actually occupied (earliest start .. latest end), NOT
    # summed across threads — so bytes_downloaded / transfer_wall_seconds is the *effective*
    # aggregate MB/s to compare apples-to-apples with the single-stream probe.
    transfer_wall_seconds: float = 0.0
    bytes_by_band: dict = dataclasses.field(default_factory=dict)  # {band: bytes} for extrapolation


# --- catalog discovery -------------------------------------------------------


def _roi_gdf(roi) -> gpd.GeoDataFrame:
    """Accept a GeoDataFrame or a path/url to one.

    Reads through `fsd.storage` rather than handing the path to `gpd.read_file`: a
    dispatched download has its roi on **blob**, and pyogrio/GDAL does not understand
    `abfss://` -- it reports the very misleading `No such file or directory` for a file that
    demonstrably exists. fsspec does understand it.

    Delegates to `fs.read_geo`, the one shared reader (TODO #47). This function was the
    prototype for it, and several other sites had the same bug.
    """
    if isinstance(roi, str):
        return fs.read_geo(roi)
    return roi


def _safe_root_from_item(item) -> str:
    """Derive the `.SAFE` root s3url from any of the item's S3 asset hrefs."""
    for asset in item.assets.values():
        href = asset.href
        if href.startswith("s3://") and ".SAFE/" in href:
            return href.split(".SAFE/")[0] + ".SAFE"
    raise ValueError(f"STAC item {item.id} has no s3 .SAFE asset href")


def _search_items(
    roi_gdf: gpd.GeoDataFrame, startdate, enddate, collection: str = config.SATELLITE_S2L2A,
):
    """Query the CDSE STAC API (anonymous) for `collection` items intersecting the ROI."""
    import pystac_client

    geom = shapely.unary_union(roi_gdf.to_crs("EPSG:4326")["geometry"])
    client = pystac_client.Client.open(config.CDSE_STAC_URL)
    search = client.search(
        collections=[collection],
        datetime=[startdate, enddate],
        intersects=geom,
        limit=200,   # page size; pystac-client auto-paginates
    )
    return list(search.items())


def _items_to_gdf(
    items, *, collection: str, declaration: CollectionDeclaration,
) -> gpd.GeoDataFrame:
    """Parse STAC items (pystac `Item`s) into a catalog GeoDataFrame.

    Pure — no network — so it is unit-testable with duck-typed fake items
    (`.id`, `.datetime`, `.geometry`, `.properties`, `.assets[*].href`).

    `offset` (#10, #30) is derived from the item's `processing:version` property -- CDSE's
    v1 catalogue exposes the STAC Processing extension, not MPC's `s2:` extension -- via
    `fsd.sources._s2_radiometry.offset_for_item`. `nodata` defaults to the
    S2 convention (`config.NODATA`); CDSE's own jp2->COG conversion stamps it
    (`_convert_one`), so it is never missing for a CDSE-ingested artifact.
    """
    rows = [
        {
            # `id` is the canonical granule name (spec 59 D3); for CDSE that is the ESA
            # product name the item id already is.
            **granule_columns(item_granule(collection, it), source=SOURCE),
            "collection": collection,
            "timestamp": it.datetime,
            "s3url": _safe_root_from_item(it),
            "cloud_cover": it.properties.get("eo:cloud_cover"),
            "offset": offset_for_item(it),
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


def _finalize_catalog_gdf(
    gdf: gpd.GeoDataFrame, roi_gdf: gpd.GeoDataFrame, max_cloudcover: float | None
) -> gpd.GeoDataFrame:
    """Apply the cloud filter, keep only granules intersecting the real ROI, and assert
    granule-id uniqueness."""
    if max_cloudcover is not None:
        gdf = gdf[gdf["cloud_cover"] <= max_cloudcover]

    roi_union = shapely.unary_union(roi_gdf.to_crs(gdf.crs)["geometry"])
    gdf = gdf[gdf.intersects(roi_union)].reset_index(drop=True)

    if len(gdf) and gdf["id"].value_counts().max() > 1:
        raise ValueError(
            "CDSE returned non-unique tile ids; the local folder layout assumes "
            "tile id is unique. This needs handling before proceeding."
        )
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
    """Discover `collection` granules intersecting `roi` within the date range, via the CDSE
    STAC API (anonymous — no credentials).

    Returns a GeoDataFrame: id (canonical granule name), collection, timestamp, s3url,
    cloud_cover, offset, scale, nodata, acquisition_key, processing_version,
    processing_datetime, source, properties, geometry (EPSG:4326). Asserts id uniqueness.
    `processing` (default `"latest"`, spec 59 D7) is applied per acquisition. No disk cache
    (decision).
    """
    declaration = _collections.get(collection)
    roi_gdf = _roi_gdf(roi)
    items = _search_items(roi_gdf, startdate, enddate, collection=collection)
    gdf = _items_to_gdf(items, collection=collection, declaration=declaration)
    gdf = _finalize_catalog_gdf(gdf, roi_gdf, max_cloudcover)
    return processing_module.select_processing(gdf, processing).kept


# --- granule download --------------------------------------------------------


def _select_item_files(
    item, bands: list[str], root_folderpath: str, *, cog: bool = True,
    collection: str = config.SATELLITE_S2L2A,
    declaration: CollectionDeclaration | None = None,
) -> list[tuple[str, str]]:
    """Select download files from a STAC item's `assets` (no S3 listing).

    For each requested band, pick the highest-resolution asset (`{band}_{res}`,
    smallest `_NNm`) and take its S3 `href`; add `MTD_TL.xml` from the
    `granule_metadata` asset. Returns `[(src_s3_url, local_filepath), ...]` with
    short band filenames, matching the on-disk layout. The band source href is always
    the `.jp2` asset; when `cog` the local destination is `Bxx.tif`, converted on arrival,
    else `Bxx.jp2`.

    `bands` may be canonical STAC EO `common_name`s or already-native asset keys --
    `declaration.canonical_to_native` normalizes either spelling to the same native key.
    **Raises**, naming the band and collection, when a requested band does not exist on
    this item, rather than skipping it: a skipped band lets a cube build without it.

    Spec: 58 D8.
    """
    if declaration is None:
        declaration = _collections.get(collection)
    # Spec 59 D2/D3: the EODATA mirror is retired; both sources write
    # `{root}/{collection}/YYYY/MM/DD/{canonical granule name}/`.
    dst_folder = granule_folderpath(root_folderpath, collection, item_granule(collection, item))
    band_ext = "tif" if cog else "jp2"

    selected = []
    for band in bands:
        native = declaration.canonical_to_native(band)
        # asset keys are "{BAND}_{res}"; sorted lexically 10m < 20m < 60m.
        keys = sorted(k for k in item.assets if k.split("_")[0] == native)
        if not keys:
            raise ValueError(
                f"sources.cdse: requested band {band!r} (native key {native!r}) is not "
                f"available on item {item.id!r} of collection {collection!r}; available "
                f"assets: {sorted(item.assets)}."
            )
        selected.append(
            (item.assets[keys[0]].href, os.path.join(dst_folder, f"{native}.{band_ext}"))
        )

    granule = item.assets.get("granule_metadata")
    if granule is not None:
        selected.append((granule.href, os.path.join(dst_folder, "MTD_TL.xml")))
    return selected


# CDSE S3 auth errors that are transient (permanent on real AWS): BUG-001, in BUGS.md at
# tag docs-archive-2026. Retryable ONLY because this is the CDSE-specific source: on CDSE
# these are transient (node-inconsistency roulette), whereas on real AWS they'd be permanent.
# `Forbidden`/403 and `InvalidAccessKeyId` have both been observed at scale here; retrying
# re-rolls onto a possibly-good node, so they are included.
_RETRYABLE_S3 = (
    "InvalidAccessKeyId",
    "SignatureDoesNotMatch",
    "SlowDown",
    "AccessDenied",
    "Forbidden",
    # transient connection/timeout errors (a stalled transfer that hit the timeout)
    "ReadTimeout",
    "ConnectTimeout",
    "ConnectionError",
    "EndpointConnection",
    "timed out",
)


def _is_retryable_s3(exc: Exception) -> bool:
    return any(code in str(exc) for code in _RETRYABLE_S3)


# Short reason labels for the failure report.
_S3_ERROR_CODES = _RETRYABLE_S3 + ("NoSuchKey",)


def _error_reason(exc: Exception) -> str:
    s = str(exc)
    for code in _S3_ERROR_CODES:
        if code in s:
            return code
    if isinstance(exc, PermissionError):
        return "Forbidden"  # s3fs raises bare PermissionError for 403s
    return type(exc).__name__


def _is_local_path(path: str) -> bool:
    """True if `path` resolves to the local filesystem (vs an `s3://`/`az://` URL)."""
    import fsspec.utils

    return fsspec.utils.get_protocol(path) in ("file", "local")


def _transfer_one(
    src_url: str,
    dst_path: str,
    s3opts: dict,
    *,
    needs_convert: bool,
    tries: int = 3,
    base_delay: float = 0.5,
) -> tuple[bool, str, float, int]:
    """THREAD stage. Idempotent skip on the **final** `dst_path`
    (`size > 0`, never a 0-byte "touched" leftover — that re-transfers). Otherwise
    transfers with the **fail-fast** retry loop on CDSE's transient S3 auth errors
    (BUG-001: a few quick re-rolls recover a *partial* bad window; a *sustained* one
    is not worth grinding — `download_resume` is the real recovery).

    When `needs_convert` (cog and a `.jp2` band), the byte transfer lands at the
    local staging sibling `dst_path + ".src.jp2"` (converted later by `_convert_one`,
    off this thread); otherwise (sidecar, or `cog=False`) it transfers straight to
    `dst_path`. No conversion here.

    Returns `(ok, reason, transfer_s, jp2_bytes)`; `reason` is ``"skipped"``/``"ok"``
    on success or a short error label on failure. Zeros on skip/failure.
    """
    import random
    import time

    if fs.exists(dst_path) and fs.size(dst_path) > 0:
        return True, "skipped", 0.0, 0
    target = dst_path + ".src.jp2" if needs_convert else dst_path
    last: Exception | None = None
    for attempt in range(tries):
        try:
            t0 = time.time()
            fs.transfer(src_url, target, src_options=s3opts)
            t1 = time.time()
            return True, "ok", t1 - t0, fs.size(target)
        except Exception as e:
            last = e
            if attempt == tries - 1 or not _is_retryable_s3(e):
                break
            time.sleep(min(base_delay * (2**attempt), 4.0) + random.uniform(0, 0.5))
    return False, _error_reason(last) if last else "unknown", 0.0, 0


def _convert_one(
    staging: str, dst_path: str, *, offset: int = 0,
    declaration: CollectionDeclaration | None = None,
) -> tuple[bool, str, float]:
    """PROCESS stage: `to_cog(staging, dst_path)` -- a lossless COG with overviews -- then
    stamp the declared GDAL scale/offset (radiometry bands only; `offset=0` is a no-op) and
    nodata-if-missing tags (#10, #30), then remove `staging`.

    The COG is built at `<dst>.stage`, stamped THERE, and only then `os.replace`d onto
    `dst_path` (#74). `staging` and any leftover `.stage` are removed in a `finally`: an
    unstamped `.tif` under its final name would be trusted forever by the `size > 0` skip,
    so a crash or stamp error leaves none and the next resume pass re-transfers and
    re-converts.

    Top-level and picklable, for `ProcessPoolExecutor` under spawn; it touches only local
    files and a frozen `CollectionDeclaration`. `declaration=None` falls back to the S2 L2A
    default.

    ⚠️ A failure here is a local/data fault (`"ConvertError"`), never a bad CDSE window, so
    the caller must NOT fold it into the transfer-failure circuit breaker.

    Returns `(ok, reason, convert_s)`.
    """
    import time

    from fsd.catalog.declaration import S2_L2A_DECLARATION
    from fsd.raster.cog import to_cog

    if declaration is None:
        declaration = S2_L2A_DECLARATION

    stage = dst_path + ".stage"
    try:
        t0 = time.time()
        to_cog(staging, stage)
        band = os.path.splitext(os.path.basename(dst_path))[0]
        is_reflectance = declaration.is_radiometry_band(band)
        stamp_or_reencode(
            stage,
            # reflectance-unit offset to match the declared scale: a viewer's
            # unscale=true computes DN*scale + offset, so the DN-space offset (-1000)
            # must be scaled to reflectance too (-> -0.1), else unscale yields
            # DN/10000 - 1000 ~= -1000 for every pixel (the black-tile bug).
            offset=offset * declaration.scale if is_reflectance else 0.0,
            scale=declaration.scale if is_reflectance else 1.0,
            set_nodata_if_missing=config.NODATA,
        )
        os.replace(stage, dst_path)   # the ONLY step that creates dst_path
        return True, "ok", time.time() - t0
    except Exception:
        return False, "ConvertError", 0.0
    finally:
        for leftover in (staging, stage):
            if fs.exists(leftover):
                try:
                    fs.rm(leftover)
                except Exception:
                    pass


def _append_downloaded(
    catalog, tile_meta: dict, results: list[tuple], declaration: CollectionDeclaration,
) -> int:
    """Group successful (tile_id, dst, ok) downloads by granule and upsert catalog rows
    (`catalog.append` unions `files`, so partially-downloaded granules complete on a
    later append). Returns the number of successful files."""
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
            "offset": r.get("offset", 0),
            "scale": r.get("scale", declaration.scale),
            "nodata": r.get("nodata", config.NODATA),
            "acquisition_key": r.get("acquisition_key", tile_id),
            "processing_version": r.get("processing_version"),
            "processing_datetime": r.get("processing_datetime"),
            "source": SOURCE,
            "properties": r.get("properties", "{}"),
            "geometry": r["geometry"],
        })
    if rows:
        # Stamp the collection-level declaration at the one place this source appends
        # to the catalog (spec 58 D2: resolved once, driver-side, not re-derived per row).
        catalog.append(rows, declaration=declaration)
    return sum(len(f) for f in files_by_tile.values())


def _push_scratch_to_remote(scratch_root: str, remote_root: str, catalog) -> None:
    """Spec 34 §5: after a `download()` pass completes against local scratch (because
    a remote `root_folderpath` was given), push every file under `scratch_root` to
    `remote_root` (`fs.put`, works for any fsspec backend) and rewrite the catalog's
    `local_folderpath` so it points at the remote location — the scratch dir is
    removed right after (the caller doesn't need it; the catalog is the record).

    Whole-run batch push, not per-file streaming (TODO #31 is the deferred streaming
    item this spec does not swallow) — every file already succeeded locally by the
    time this runs, so a push failure here is reported by letting the exception
    propagate (the scratch dir is left in place for a manual retry rather than removed).
    """
    import glob

    for local_fp in glob.glob(os.path.join(scratch_root, "**", "*"), recursive=True):
        if os.path.isdir(local_fp):
            continue
        rel = os.path.relpath(local_fp, scratch_root).replace(os.sep, "/")
        fs.put(local_fp, remote_root.rstrip("/") + "/" + rel)

    if fs.exists(catalog.filepath):
        gdf = catalog.read()

        def _relocate(p):
            if not str(p).startswith(scratch_root):
                return p
            rel = os.path.relpath(str(p), scratch_root).replace(os.sep, "/")
            return remote_root.rstrip("/") + "/" + rel

        gdf["local_folderpath"] = gdf["local_folderpath"].apply(_relocate)
        fs.write_parquet(catalog.filepath, gdf)

    shutil.rmtree(scratch_root, ignore_errors=True)


def _default_max_staged(root_folderpath: str, max_convert_procs: int,
                        max_concurrent_s3: int = config.MAX_CONCURRENT_S3) -> int:
    """Disk-aware `MAX_STAGED` sizing: a **safety cap** on
    staged-but-unconverted JP2s, not a throughput lever — past `floor` a bigger
    buffer gives no throughput gain (bounded-buffer queueing), so free disk only
    *shrinks* the cap, never grows it beyond the saturation target `headroom`.
    Sized **once** at `download()` start from `shutil.disk_usage(root_folderpath)`.
    """
    floor = max_concurrent_s3 + max_convert_procs
    headroom = max_concurrent_s3 + 2 * max_convert_procs
    free = shutil.disk_usage(root_folderpath).free
    disk_cap = int(free * config.STAGING_DISK_FRACTION / (config.STAGING_ITEM_GB * 1e9))
    staged = max(max_concurrent_s3, min(headroom, disk_cap))
    if staged < floor:
        print(
            f"[fsd.download] warning: disk-limited staging={staged} < {floor} "
            "(convert pool may under-saturate)",
            flush=True,
        )
    return staged


def _make_convert_pool(max_workers: int):
    """Default convert-process-pool factory. A module-level seam: tests
    monkeypatch this to assert a `cog=False` / all-skip `download()` run never spawns
    a process pool. **Spawn** start-method (GDAL-safe; `fork` + GDAL's internal
    threads can deadlock on Linux/Batch)."""
    import concurrent.futures
    import multiprocessing

    return concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers, mp_context=multiprocessing.get_context("spawn"),
    )


def _fmt_progress(done, total, ok_n, fail_n, skipped, elapsed_s) -> str:
    """A single newline-terminated progress line (log-friendly, with rate + ETA).

    ETA is derived from `done/elapsed_s`, and shown as `ETA ~?` until `done > 0` -- there is
    no rate to extrapolate from before that.
    """
    rate = done / elapsed_s if elapsed_s > 0 else 0.0
    pct = 100 * done // max(1, total)
    eta = f"~{int(((total - done) / rate) // 60)}m" if done > 0 and rate > 0 else "~?"
    return (
        f"[{int(elapsed_s // 60):3d}m{int(elapsed_s % 60):02d}s] "
        f"{done}/{total} ({pct:2d}%) ok={ok_n} fail={fail_n} skip={skipped} | "
        f"{rate:.1f} file/s | ETA {eta}"
    )


def download(
    roi,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    bands: list[str],
    root_folderpath: str,
    catalog,                      # fsd.catalog.catalog.TileCatalog (appended in place)
    creds: CdseCredentials,
    *,
    max_tiles: int,
    chunksize: int = 100,
    max_cloudcover: float | None = None,
    progress: bool = False,
    max_consecutive_failures: int | None = None,
    cog: bool = True,
    max_convert_procs: int | None = None,
    max_staged: int | None = None,
    max_concurrent_s3: int | None = None,
    convert_executor=None,
    should_stop: Callable[[], bool] | None = None,
    collection: str = config.SATELLITE_S2L2A,
    processing: str = processing_module.LATEST,
) -> DownloadResult:
    """Discover matching granules and download the requested band files (+ MTD_TL.xml) to
    `root_folderpath` through a **pipeline**: a `MAX_CONCURRENT_S3`-wide thread pool
    transfers bytes while a process pool converts fetched JP2s to COGs, chained by
    `add_done_callback` and bounded by a `max_staged` backpressure semaphore (staged but
    unconverted JP2s on disk). Idempotent (skips files already on disk); the catalog is
    upserted every `chunksize` completions so a crash doesn't lose progress; refuses if
    matched granules exceed `max_tiles`.

    `processing` selects ONE processing per acquisition among what CDSE offers: `"latest"`
    (default) or a PEP 440 specifier such as `"==05.00"`, applied before `max_tiles`.
    `None` raises. Skips are printed, and so is any acquisition this download leaves
    holding more than one processing in the archive.

    `cog` (default True) converts each fetched JP2 band to a lossless COG (`Bxx.tif`, with
    overviews) on arrival, which the datacube build reads far faster; `cog=False` keeps the
    native `.jp2`. A remote (`s3://`/`az://`) `root_folderpath` with `cog=True` stages to
    local scratch, converts there, then pushes the whole run to the remote root
    (`_push_scratch_to_remote`), a batch push rather than per-file streaming (TODO #31).

    `max_convert_procs` (default `config.MAX_CONVERT_PROCS`), `max_staged` (default
    `_default_max_staged`, disk-aware) and `convert_executor` (default a spawn-context
    `ProcessPoolExecutor`) are optional knobs; `convert_executor` is the test seam, so a
    synchronous stand-in runs the pipeline in-process. The convert pool is created
    **lazily**, on the first file that needs conversion: a `cog=False` run or an all-skip
    resume pass spawns zero processes.

    `max_consecutive_failures` is the **circuit breaker**, keyed on consecutive
    **transfer** failures only; a `_convert_one` failure is a local fault, not a CDSE
    window. When that many transfers fail back-to-back (a bad CDSE window, BUG-001), the
    submit loop stops queuing, in-flight work drains, and the pass returns with
    `circuit_tripped=True`, within roughly `max_staged` items of the trip. Pair with
    `download_resume` to retry the remainder later.

    `should_stop` (default None) is a user-stop predicate checked in the submit loop,
    throttled to once per `config.STOP_CHECK_EVERY_S`. Halts **new** submissions only:
    already-submitted work finalizes and drains, and a stopped item is never attempted, so
    it is not a failure and not counted. Sets `DownloadResult.stopped=True`.

    ⚠️ Only a converted band takes the `max_staged` semaphore, so with `cog=False` the
    whole work list is queued at once and neither a trip nor a stop halts the pass (#157).

    Spec: 01.
    """
    import collections
    import concurrent.futures
    import threading
    import time
    from functools import partial

    creds.require_s3()  # discovery (STAC) is anonymous; only download needs S3 keys
    declaration = _collections.get(collection)
    require_valid_processing(processing, collection)

    # A remote (blob) root_folderpath runs the whole existing local pipeline
    # (transfer/convert/stamp) against LOCAL scratch -- every path below stays local, so
    # nothing else in this function has to change -- and once the pass completes, the scratch
    # tree is pushed to the real remote root and the catalog rows are rewritten to point at
    # it (`_push_scratch_to_remote`). A whole-run batch push, not per-file streaming to blob
    # (TODO #31): acceptable while the scratch dir and the blob destination are both close to
    # the machine doing the work.
    import tempfile

    remote_root = None
    if not _is_local_path(root_folderpath):
        remote_root = root_folderpath
        root_folderpath = tempfile.mkdtemp(prefix="fsd_cdse_")

    # Ensure the local output root exists: the disk-usage probe in _default_max_staged
    # (and, for a first run, the per-file writes) assume it is already there. Leaf dirs
    # auto-create on write, but the root-level probe runs before any write, so a fresh
    # --dst would otherwise FileNotFoundError. Creating the root is part of download()'s
    # contract (put files under root_folderpath), not the caller's job.
    fs.makedirs(root_folderpath, exist_ok=True)

    roi_gdf = _roi_gdf(roi)
    items = _search_items(roi_gdf, startdate, enddate, collection=collection)
    tiles = _finalize_catalog_gdf(
        _items_to_gdf(items, collection=collection, declaration=declaration),
        roi_gdf, max_cloudcover,
    )
    # CDSE's own forum guidance for near-duplicate products is "use the most recent",
    # hence "latest" by default.
    tiles = select_granules(tiles, processing=processing, prefix="[fsd.cdse.download]")

    if len(tiles) > max_tiles:
        est_gb = len(tiles) * config.APPROX_GB_PER_TILE
        raise ValueError(
            f"{len(tiles)} matched tiles exceed max_tiles={max_tiles} "
            f"(~{est_gb:.0f} GB). Narrow the query or raise max_tiles."
        )

    s3opts = creds.s3_storage_options()
    tile_meta = {row["id"]: row for _, row in tiles.iterrows()}
    kept_items = surviving_items(items, tile_meta, collection)

    # Flat work list (src, dst, tile_id) built from STAC assets — no S3 listing.
    work: list[tuple[str, str, str]] = []
    for it, tile_id in kept_items:
        for src, dst in _select_item_files(it, bands, root_folderpath, cog=cog,
                                           collection=collection, declaration=declaration):
            work.append((src, dst, tile_id))

    total = len(work)
    start = time.time()

    procs = max_convert_procs if max_convert_procs is not None else config.MAX_CONVERT_PROCS
    s3_workers = max_concurrent_s3 if max_concurrent_s3 is not None else config.MAX_CONCURRENT_S3
    staged_cap = max_staged
    if staged_cap is None:
        # _default_max_staged needs a real local path; only cog=True can reach here
        # with a local root (cog=False + remote root is otherwise legal and sem_staged
        # is never acquired for it, so skip the disk probe entirely).
        staged_cap = _default_max_staged(root_folderpath, procs, s3_workers) if cog else s3_workers
    sem_staged = threading.BoundedSemaphore(staged_cap)

    lock = threading.Lock()
    all_done = threading.Event()
    pending_results: list[tuple[str, str, bool]] = []
    failures: list[tuple[str, str]] = []
    reason_counts: collections.Counter = collections.Counter()
    bytes_by_band: collections.Counter = collections.Counter()
    state = {
        "done": 0, "skipped": 0, "successful": 0,
        "consecutive": 0, "tripped": False, "pool_broken": False,
        "transfer_seconds": 0.0, "convert_seconds": 0.0, "bytes_downloaded": 0,
        "transfer_wall_start": None, "transfer_wall_end": 0.0,
        "remaining": 0, "loop_finished": False, "last_print": 0.0,
        "stopped": False, "last_stop_check": -1e18, "stop_cached": False,
    }
    convert_pool_holder: dict = {"pool": None}
    pool_create_lock = threading.Lock()
    # Serializes the actual catalog (parquet) write, which happens OUTSIDE `lock` so it does
    # not block metric updates -- but concurrent writers to the same file would otherwise
    # race and corrupt it. This lock covers the I/O only, so chunk-flushes still do not stall
    # the counter updates in `_finalize`.
    flush_lock = threading.Lock()

    def _emit():
        if progress:
            print(
                _fmt_progress(state["done"], total, reason_counts["ok"] + state["skipped"],
                              len(failures), state["skipped"], time.time() - start),
                flush=True,
            )

    def _stop() -> bool:
        # Called only from the single submit-loop thread (below), so no lock needed
        # around the throttle bookkeeping itself; only the sticky `stopped` flag
        # (read by other threads via the final result) is set under `lock`.
        if should_stop is None:
            return False
        now = time.time()
        if now - state["last_stop_check"] >= config.STOP_CHECK_EVERY_S:
            state["last_stop_check"] = now
            state["stop_cached"] = bool(should_stop())
            if state["stop_cached"] and not state["stopped"]:
                with lock:
                    state["stopped"] = True
                    in_flight = state["remaining"]
                if progress:
                    # Acknowledge the stop the moment it's seen (the run then keeps going
                    # only to drain what's already in flight — no partial files left behind).
                    print(
                        f"[fsd.download] stop requested — halting new submissions; draining "
                        f"{in_flight} in-flight transfer(s)/convert(s), then exiting…",
                        flush=True,
                    )
        return state["stop_cached"]

    def _get_convert_pool():
        with pool_create_lock:
            if convert_pool_holder["pool"] is None:
                convert_pool_holder["pool"] = (
                    convert_executor if convert_executor is not None
                    else _make_convert_pool(procs)
                )
            return convert_pool_holder["pool"]

    def _finalize(tid, src, dst, ok, reason):
        # `remaining`/`sem_staged` accounting must never sit behind a fallible call:
        # decrement `remaining` under `lock` FIRST, then flush the catalog (a parquet write,
        # which can raise) outside the lock, so a write failure cannot strand the drain.
        snapshot = None
        with lock:
            pending_results.append((tid, dst, ok))
            reason_counts[reason] += 1
            if reason == "skipped":
                state["skipped"] += 1
            if not ok:
                failures.append((src, reason))
            state["done"] += 1
            # `chunksize` sets only how often the buffer flushes to the catalog (crash
            # resilience), not executor batching.
            if len(pending_results) >= chunksize:
                snapshot = list(pending_results)
                pending_results.clear()
            if progress and time.time() - state["last_print"] >= config.PROGRESS_EVERY_S:
                state["last_print"] = time.time()
                _emit()
            state["remaining"] -= 1
            drained = state["loop_finished"] and state["remaining"] == 0
        if snapshot is not None:
            try:
                with flush_lock:
                    n = _append_downloaded(catalog, tile_meta, snapshot, declaration)
                with lock:
                    state["successful"] += n
            except Exception as e:
                print(f"[fsd.download] warning: chunk-flush failed, will retry at "
                      f"end-of-run: {e!r}", flush=True)
                with lock:
                    pending_results.extend(snapshot)
        if drained:
            all_done.set()

    def _on_convert_done(src, tid, dst, cfut):
        try:
            ok, reason, c_s = cfut.result()
        except Exception:
            ok, reason, c_s = False, "PoolBroken", 0.0
            with lock:
                state["pool_broken"] = True
        finally:
            sem_staged.release()
        with lock:
            state["convert_seconds"] += c_s
        _finalize(tid, src, dst, ok, reason)

    def _on_transfer_done(src, dst, tid, needs_convert, fut):
        t_end = time.time()  # ~transfer end; back out the start from the measured duration
        try:
            ok, reason, t_s, nbytes = fut.result()
        except Exception as e:
            ok, reason, t_s, nbytes = False, _error_reason(e), 0.0, 0
        with lock:
            state["transfer_seconds"] += t_s
            if nbytes:
                state["bytes_downloaded"] += nbytes
                bytes_by_band[os.path.splitext(os.path.basename(dst))[0]] += nbytes
                # wall span the transfer phase actually occupied (for effective MB/s, vs the
                # thread-summed transfer_seconds): earliest start .. latest end across all files.
                t_start = t_end - t_s
                if state["transfer_wall_start"] is None or t_start < state["transfer_wall_start"]:
                    state["transfer_wall_start"] = t_start
                if t_end > state["transfer_wall_end"]:
                    state["transfer_wall_end"] = t_end
            if ok:
                state["consecutive"] = 0
            else:
                state["consecutive"] += 1
                if (max_consecutive_failures is not None
                        and state["consecutive"] >= max_consecutive_failures):
                    state["tripped"] = True
        if ok and reason != "skipped" and needs_convert:
            try:
                pool = _get_convert_pool()
                offset = int(tile_meta[tid].get("offset", 0) or 0)
                cfut = pool.submit(_convert_one, dst + ".src.jp2", dst, offset=offset,
                                   declaration=declaration)
            except Exception:
                with lock:
                    state["pool_broken"] = True
                sem_staged.release()
                _finalize(tid, src, dst, False, "PoolBroken")
                return
            cfut.add_done_callback(partial(_on_convert_done, src, tid, dst))
        else:
            if needs_convert:
                sem_staged.release()
            _finalize(tid, src, dst, ok, reason)

    transfer_pool = concurrent.futures.ThreadPoolExecutor(max_workers=s3_workers)
    try:
        for src, dst, tid in work:
            if state["tripped"] or state["pool_broken"] or _stop():
                break
            needs_convert = cog and src.endswith(".jp2")
            if needs_convert:
                sem_staged.acquire()  # BLOCKS at max_staged in-flight -> backpressure
                if state["tripped"] or state["pool_broken"] or _stop():
                    sem_staged.release()
                    break
            with lock:
                state["remaining"] += 1
            fut = transfer_pool.submit(_transfer_one, src, dst, s3opts, needs_convert=needs_convert)
            fut.add_done_callback(partial(_on_transfer_done, src, dst, tid, needs_convert))

        with lock:
            state["loop_finished"] = True
            drained = state["remaining"] == 0
        if drained:
            all_done.set()
        all_done.wait()
    finally:
        transfer_pool.shutdown(wait=True)
        if convert_pool_holder["pool"] is not None:
            convert_pool_holder["pool"].shutdown(wait=True)

    if pending_results:
        try:
            state["successful"] += _append_downloaded(catalog, tile_meta, pending_results, declaration)
            pending_results.clear()
        except Exception as e:
            print(f"[fsd.download] warning: end-of-run catalog flush failed: {e!r}", flush=True)

    if remote_root is not None:
        _push_scratch_to_remote(root_folderpath, remote_root, catalog)

    report_download(catalog, tiles, processing=processing, source=SOURCE)
    _emit()  # final line

    return DownloadResult(
        successful_count=state["successful"],
        total_count=state["successful"] + len(failures),   # files actually attempted
        skipped_count=state["skipped"],
        failed_count=len(failures),
        elapsed_s=time.time() - start,
        failures=failures,
        reason_counts=dict(reason_counts),
        circuit_tripped=state["tripped"],
        pool_broken=state["pool_broken"],
        stopped=state["stopped"],
        bytes_downloaded=state["bytes_downloaded"],
        transfer_seconds=state["transfer_seconds"],
        convert_seconds=state["convert_seconds"],
        transfer_wall_seconds=(
            state["transfer_wall_end"] - state["transfer_wall_start"]
            if state["transfer_wall_start"] is not None else 0.0
        ),
        bytes_by_band=dict(bytes_by_band),
    )


def download_resume(
    roi,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    bands: list[str],
    root_folderpath: str,
    catalog,
    creds: CdseCredentials,
    *,
    max_tiles: int,
    chunksize: int = 100,
    max_cloudcover: float | None = None,
    progress: bool = False,
    max_consecutive_failures: int = 15,
    max_passes: int = 10,
    cooldown_s: float = 60.0,
    on_pass=None,
    cog: bool = True,
    max_convert_procs: int | None = None,
    max_staged: int | None = None,
    max_concurrent_s3: int | None = None,
    convert_executor=None,
    should_stop: Callable[[], bool] | None = None,
    collection: str = config.SATELLITE_S2L2A,
    processing: str = processing_module.LATEST,
) -> list[DownloadResult]:
    """Resume-loop: run `download` repeatedly until a full pass has no failures or
    `max_passes` is reached. Each pass is idempotent. On a circuit-breaker trip
    (`max_consecutive_failures`) wait `cooldown_s` then retry: *fail-fast + resume-later*
    (BUG-001). A partial window (scattered fast-fails, no trip) loops immediately.

    `on_pass(pass_index, DownloadResult)` is called after each pass (e.g. to persist
    stats), keeping file I/O out of the library. Returns the per-pass results.
    `max_convert_procs`/`max_staged`/`convert_executor` pass through to `download`.

    `should_stop` passes through to each pass; a pass returning `stopped=True` ends the loop
    at once (no cooldown, not a completion). It is also checked before each new pass.
    """
    import time

    results: list[DownloadResult] = []
    for p in range(max_passes):
        if should_stop is not None and should_stop():
            break
        r = download(
            roi, startdate, enddate, bands, root_folderpath, catalog, creds,
            max_tiles=max_tiles, chunksize=chunksize, max_cloudcover=max_cloudcover,
            progress=progress, max_consecutive_failures=max_consecutive_failures,
            cog=cog, max_convert_procs=max_convert_procs, max_staged=max_staged,
            max_concurrent_s3=max_concurrent_s3,
            convert_executor=convert_executor, should_stop=should_stop,
            collection=collection, processing=processing,
        )
        results.append(r)
        if on_pass is not None:
            on_pass(p, r)
        if r.stopped:
            break  # user stop: ends the resume loop immediately, no cooldown
        if r.failed_count == 0 and not r.circuit_tripped:
            break  # complete: a full pass attempted everything and nothing failed
        if r.circuit_tripped and cooldown_s:
            time.sleep(cooldown_s)  # bad window — back off, then resume
    return results


def sum_results(results: list[DownloadResult]) -> DownloadResult:
    """Aggregate the per-pass results of `download_resume` into one `DownloadResult`.

    Counts/bytes/seconds add; a later pass that skips an already-downloaded file contributes 0
    bytes/seconds, so the sum is the true one-time cost. `elapsed_s` is the sum of pass wall-times.
    """
    import collections

    agg = DownloadResult(successful_count=0, total_count=0)
    by_band: collections.Counter = collections.Counter()
    reasons: collections.Counter = collections.Counter()
    for r in results:
        agg.successful_count += r.successful_count
        agg.total_count += r.total_count
        agg.skipped_count += r.skipped_count
        agg.failed_count += r.failed_count
        agg.elapsed_s += r.elapsed_s
        agg.failures.extend(r.failures)
        agg.bytes_downloaded += r.bytes_downloaded
        agg.transfer_seconds += r.transfer_seconds
        agg.convert_seconds += r.convert_seconds
        agg.transfer_wall_seconds += r.transfer_wall_seconds  # per-pass spans don't overlap → sum
        agg.circuit_tripped = agg.circuit_tripped or r.circuit_tripped
        agg.pool_broken = agg.pool_broken or r.pool_broken
        agg.stopped = agg.stopped or r.stopped
        reasons.update(r.reason_counts)
        by_band.update(r.bytes_by_band)
    agg.reason_counts = dict(reasons)
    agg.bytes_by_band = dict(by_band)
    return agg


def probe_throughput(
    roi,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    bands: list[str],
    creds: CdseCredentials,
    *,
    max_cloudcover: float | None = None,
) -> tuple[float, int, float]:
    """Measure achievable CDSE **byte** throughput right now with a single-threaded fetch of ONE
    representative band file. Returns `(mb_per_s, bytes, seconds)`.

    A baseline to compare against a run's *aggregate effective* MB/s: probe≈aggregate → CDSE/link
    bound; probe≫aggregate → local contention / concurrency. Transfers the JP2 to a temp path and
    removes it (isolates network from COG-conversion; a fresh sample each call).
    """
    import tempfile
    import time

    creds.require_s3()
    roi_gdf = _roi_gdf(roi)
    items = _search_items(roi_gdf, startdate, enddate)
    tiles = _finalize_catalog_gdf(
        _items_to_gdf(items, collection=config.SATELLITE_S2L2A,
                      declaration=_collections.get(config.SATELLITE_S2L2A)),
        roi_gdf, max_cloudcover,
    )
    if not len(tiles):
        return (0.0, 0, 0.0)
    tile_ids = set(tiles["id"])
    item = next(it for it, _ in surviving_items(items, tile_ids,
                                            config.SATELLITE_S2L2A))
    band = bands[0]
    keys = sorted(k for k in item.assets if k.split("_")[0] == band)
    if not keys:
        return (0.0, 0, 0.0)
    src = item.assets[keys[0]].href
    s3opts = creds.s3_storage_options()
    tmp = os.path.join(tempfile.gettempdir(), f"fsd_probe_{item.id}_{band}.jp2")
    try:
        t0 = time.time()
        fs.transfer(src, tmp, src_options=s3opts)
        dt = time.time() - t0
        nbytes = fs.size(tmp)
    finally:
        if fs.exists(tmp):
            try:
                fs.rm(tmp)
            except Exception:
                pass
    return (nbytes / 1e6 / dt if dt > 0 else 0.0, nbytes, dt)


def plan_download(
    roi,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    bands: list[str],
    *,
    catalog_filepath: str | None = None,
    dst_folderpath: str | None = None,
    max_cloudcover: float | None = None,
    cost_model: dict | None = None,
    processing: str = processing_module.LATEST,
) -> dict:
    """Compute an actionable download plan **without downloading**.

    Queries the CDSE STAC (anonymous, no bytes) for the granules this request needs, diffs them
    against what is already in `catalog_filepath` (if given), and returns a plan dict: needed /
    present / missing granule counts + ids, the exact `fsd.download(...)` params to satisfy it
    (`max_tiles` = needed count), and — when a `cost_model` is supplied — the estimated GB + ETA
    for the missing granules. This is the CDSE (materializing-source) arm of the guardrail; a
    streamable source (MPC) would never need it (TODO #21).
    """
    needed = query_catalog(roi, startdate, enddate, max_cloudcover=max_cloudcover,
                           processing=processing)
    needed_ids = list(needed["id"])
    needed_set = set(needed_ids)

    present_ids: list[str] = []
    if catalog_filepath is not None and fs.exists(catalog_filepath):
        try:
            from fsd.catalog.catalog import TileCatalog

            existing = TileCatalog(catalog_filepath).read()
            present_ids = [i for i in existing["id"] if i in needed_set]
        except Exception:  # noqa: BLE001 - a missing/unreadable catalog just means "all missing"
            present_ids = []
    present_set = set(present_ids)
    missing_ids = [i for i in needed_ids if i not in present_set]

    plan = {
        "needed_count": len(needed_ids),
        "present_count": len(present_ids),
        "missing_count": len(missing_ids),
        "missing_ids": missing_ids,
        "download_params": {
            "roi": roi if isinstance(roi, str) else "<GeoDataFrame>",
            "startdate": startdate.isoformat() if hasattr(startdate, "isoformat") else str(startdate),
            "enddate": enddate.isoformat() if hasattr(enddate, "isoformat") else str(enddate),
            "bands": list(bands),
            "max_tiles": max(len(needed_ids), 1),
            "max_cloudcover": max_cloudcover,
            "dst_folderpath": dst_folderpath,
        },
    }
    if cost_model:
        mean_by_band = cost_model.get("mean_bytes_by_band") or {}
        per_granule = sum(mean_by_band.get(b, 0) for b in bands)
        est_bytes = plan["missing_count"] * per_granule
        mbps = cost_model.get("transfer_mb_per_s") or 0.0
        plan["estimate"] = {
            "gb": round(est_bytes / 1e9, 2),
            "download_minutes": round((est_bytes / 1e6 / mbps) / 60, 1) if mbps else None,
        }
    return plan


def format_download_plan(plan: dict) -> str:
    """Render a `plan_download` dict as a copy-pasteable message."""
    p = plan["download_params"]
    if plan["missing_count"] == 0:
        # Nothing to download — don't contradict "missing: 0" with a "not present" line
        # and a fsd.download(...) command.
        return (
            "imagery for this run is fully present in the catalog — nothing to download.\n"
            f"  needed: {plan['needed_count']} granules | present: {plan['present_count']} | "
            "missing: 0"
        )
    lines = [
        "imagery for this run is not (fully) present in the catalog.",
        f"  needed: {plan['needed_count']} granules | present: {plan['present_count']} | "
        f"missing: {plan['missing_count']}",
    ]
    est = plan.get("estimate")
    if est:
        eta = f", ~{est['download_minutes']} min" if est.get("download_minutes") else ""
        lines.append(f"  estimated: ~{est['gb']} GB{eta} (at last measured throughput)")
    band_list = ", ".join(f'"{b}"' for b in p["bands"])
    lines += [
        "  run fsd.download(",
        f'      roi={p["roi"]!r}, startdate="{p["startdate"]}", enddate="{p["enddate"]}",',
        f"      bands=[{band_list}], max_tiles={p['max_tiles']}, "
        f"max_cloudcover={p['max_cloudcover']}, dst_folderpath={p['dst_folderpath']!r})",
    ]
    return "\n".join(lines)
