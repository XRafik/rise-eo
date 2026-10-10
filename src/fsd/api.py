"""fsd high-level API — the verbs users call.

Spec: specs/16-packaging-and-api.md. A façade over the internal modules
(`sources`, `catalog`, `datacube`, `workflows`, `flatten`) that raises the scope from
implementation vocabulary ("flatten", "input.csv") to user intent ("make training data").
Besides the verbs it holds the preflight checks, the flatten identity and the raster
merge of per-cell outputs.

- `download(...)`            -> fetch S2 L2A granules + build a TileCatalog (its own verb).
- `create_training_data(...)`-> label polygons + catalog -> datacubes -> flattened arrays.
- `run_inference(...)`       -> model over pre-built cubes OR an ROI (grid cells -> per-cell
                               build+infer via the runner seam) -> COG + STAC.
- `deploy(...)`              -> register a model bundle in a registry.

`runner=`/`storage=` are the seams (ROADMAP §2.2/§2.3): `runner` is `"local"` or `"aml"`,
and `storage` is local or Azure blob, per verb (`_check_local_seams`).
Every verb runs a cheap **preflight** (ROADMAP §2.6) before any heavy work.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd

from fsd import collections as _collections
from fsd import config
from fsd import progress as _progress
from fsd.bands import modify as _modify
from fsd.catalog import processing as _processing
from fsd.catalog import stac as _stac
from fsd.catalog.catalog import TileCatalog
from fsd.catalog.catalog import filter_gdf as _filter_gdf
from fsd.datacube import flatten as _flatten
from fsd.model import bundle as _bundle
from fsd.model import engine as _engine
from fsd.model import registry as registry_mod
from fsd.model.features import apply_features as _apply_features
from fsd.model.features import resolve_aggregate as _resolve_aggregate
from fsd.model.verify_image import verify_image as _verify_image
from fsd.raster.cog import to_cog as _to_cog
from fsd.sources._granules import collection_root
from fsd.sources.cdse import SERVED_COLLECTIONS as _CDSE_SERVED_COLLECTIONS
from fsd.sources.cdse import CdseCredentials
from fsd.sources.cdse import download as _cdse_download
from fsd.sources.mpc import SERVED_COLLECTIONS as _MPC_SERVED_COLLECTIONS
from fsd.sources.mpc import download as _mpc_download
from fsd.storage import fs
from fsd.storage.azure import configure_storage as _configure_storage
from fsd.workflows import create_datacube as _create_datacube
from fsd.workflows import infer_only_task as _infer_only_task
from fsd.workflows import stamp as _stamp

__all__ = [
    "InferenceResult",
    "PreflightError",
    "TrainingData",
    "archive_catalog_filepath",
    "compute_n_timestamps",
    "create_training_data",
    "deploy",
    "download",
    "flatten_training_data",
    "run_inference",
    "verify_adapter",
]

_VALID_OVERWRITE = (False, True, "datacubes", "flatten")


# A cached work list that is a strict superset of the freshly tiled grids by no more than
# this many ids is cell-count drift from a re-tiling, not a different ROI -- so resume is
# still safe. Sized from the observed drift: AT_ROI dropped 1 cell, s2grid=476da24 dropped 8.
_RESUME_DRIFT_MAX_MISSING = 10

# How many differing ids the mismatch error quotes: a bounded sample, since a 300-cell
# mismatch printed in full is unreadable.
_RESUME_DIFF_SAMPLE = 10


class PreflightError(ValueError):
    """Raised when a cheap pre-flight check fails, before any download/build.

    Aggregates all failures so the caller fixes their config in one pass (ROADMAP §2.6).
    """


# --- helpers -----------------------------------------------------------------

def archive_catalog_filepath(dst_folderpath: str, collection: str) -> str:
    """`{dst_folderpath}/{collection}/catalog.parquet` -- the catalog of one collection.

    The archive root is the only path a caller writes; this is how to name the catalog
    inside it without rebuilding the layout by hand. Pure: no I/O, no storage configuration,
    no collection validation.

    Spec: 59.
    """
    return os.path.join(collection_root(dst_folderpath, collection), "catalog.parquet")


def compute_n_timestamps(
    startdate: datetime.datetime, enddate: datetime.datetime, mosaic_days: int
) -> int:
    """`T` for a calendar-mosaic build: ceil((enddate - startdate) / mosaic_days).

    Pure function of the caller's window — computable with no catalog and no download,
    which is what lets preflight check `T` against a model's `n_timestamps`.
    """
    start = pd.to_datetime(startdate, utc=True)
    end = pd.to_datetime(enddate, utc=True)
    total_days = (end - start) / pd.Timedelta(days=1)
    return math.ceil(total_days / mosaic_days)


_VALID_RUNNERS = ("local", "aml")


def _check_local_seams(runner: str, storage, *, storage_allowed: bool = True) -> list[str]:
    """Collect seam errors for a verb: `runner` must be `"local"` (Snakemake) or `"aml"`.

    `storage_allowed=False` refuses a non-local `storage=` loudly rather than silently
    writing local paths a remote runner cannot read. `run_inference` passes
    `roi_mode and runner == "aml"`, `verify_adapter` passes `runner == "aml"`; the other
    verbs (`download`, `create_training_data`, `flatten_training_data`, `deploy`) use the
    default `True`.
    """
    errs = []
    if runner not in _VALID_RUNNERS:
        errs.append(f"runner={runner!r} not supported (valid: {list(_VALID_RUNNERS)}).")
    if storage is not None and storage != "local":
        if not storage_allowed:
            errs.append(
                "non-local storage not supported here yet (inference/serving-on-blob is "
                "P4/P5; storage= is wired for download/create_training_data in P1)."
            )
        else:
            if isinstance(storage, str):
                backend = storage
            elif isinstance(storage, dict):
                backend = storage.get("backend")
            else:
                backend = None
            if backend != "azure":
                errs.append(f"storage backend {backend!r} not supported (only 'azure' in P1).")
    return errs


def _as_gdf(label_polygons) -> gpd.GeoDataFrame:
    """A GeoDataFrame, or a path/url to one. Read via the storage seam (`fs.read_geo`),
    never `gpd.read_file(path)` — GDAL has no `abfss://` driver and reports a blob-hosted
    file as "No such file or directory" (TODO #47)."""
    if isinstance(label_polygons, gpd.GeoDataFrame):
        return label_polygons
    return fs.read_geo(label_polygons)


def _check_window(startdate, enddate, mosaic_days, bands) -> list[str]:
    errs = []
    start = pd.to_datetime(startdate, utc=True)
    end = pd.to_datetime(enddate, utc=True)
    if not (start < end):
        errs.append(f"startdate ({start}) must be before enddate ({end}).")
    if mosaic_days < 1:
        errs.append(f"mosaic_days ({mosaic_days}) must be >= 1.")
    if not bands:
        errs.append("bands must be a non-empty list.")
    if start < end and mosaic_days >= 1 and compute_n_timestamps(start, end, mosaic_days) < 1:
        errs.append("date window yields T < 1 timestamps.")
    return errs


def _check_catalog_collection_dir(catalog_filepath: str, collection: str) -> list[str]:
    """The directory holding `catalog.parquet` is part of the contract --
    `{root}/{collection}/catalog.parquet` -- so a catalog of one collection handed to a verb
    for another is caught before any work, naming both.

    Spec: 59."""
    parent = os.path.basename(os.path.dirname(str(catalog_filepath).rstrip("/")))
    if parent == collection:
        return []
    return [
        f"catalog_filepath {catalog_filepath!r} sits in a directory named {parent!r}, but "
        f"collection={collection!r}: a catalog lives at "
        f"{{archive root}}/{collection}/catalog.parquet (spec 59 D5). Pass the catalog "
        f"`fsd.download(collection={collection!r}, ...)` returned, or fix collection=."
    ]


def _raise_preflight(errs: list[str]) -> None:
    if errs:
        raise PreflightError("preflight failed:\n  - " + "\n  - ".join(errs))


_SOURCE_SERVED_COLLECTIONS = {"cdse": _CDSE_SERVED_COLLECTIONS, "mpc": _MPC_SERVED_COLLECTIONS}


def _check_source_collection(source: str, collection: str) -> list[str]:
    """Not every (source, collection) pair is valid -- each source module declares what it
    serves; an unserved pair raises at preflight, naming what the source DOES serve, rather
    than failing deep inside discovery.

    Spec: 58."""
    served = _SOURCE_SERVED_COLLECTIONS.get(source)
    if served is not None and collection not in served:
        return [
            f"source={source!r} does not serve collection={collection!r}; "
            f"source={source!r} serves: {sorted(served)}."
        ]
    return []


def _check_cloudcover_capability(collection: str, max_cloudcover: float) -> list[str]:
    """`max_cloudcover` requires the collection to declare `supports_cloud_cover=True`
    -- passing it against a collection with no cloud-cover concept (e.g. Sentinel-1) would
    otherwise silently be a no-op filter.

    Spec: 58."""
    declaration = _collections.get(collection)
    if not declaration.supports_cloud_cover:
        return [
            f"max_cloudcover={max_cloudcover!r} was given but collection={collection!r} "
            "declares supports_cloud_cover=False (spec 58 D6) -- this collection has no "
            "cloud-cover concept, so the filter would silently do nothing."
        ]
    return []


def _check_resume_identity(csv_filepath: str, grids: gpd.GeoDataFrame, output_folderpath: str) -> None:
    """Refuse to resume when `output_folderpath`'s cached `input.csv` names a different id
    set than the freshly tiled grids.

    `input.csv` resumes by EXISTENCE, not identity, so without this a work list from a
    different roi silently wins over the fresh one (#66). ANY difference raises rather than
    repairing, because both repairs are worse: rewriting `input.csv` in place orphans every
    cell already written under the old id set, and deleting the folder is not reliable on
    blob (#50).
    """
    with fs.open(csv_filepath, "r") as f:
        cached_ids = set(pd.read_csv(f)["id"].astype(str))
    fresh_ids = set(grids["id"].astype(str))
    if cached_ids == fresh_ids:
        return
    only_cached = sorted(cached_ids - fresh_ids)
    only_fresh = sorted(fresh_ids - cached_ids)
    sample = (only_cached + only_fresh)[:_RESUME_DIFF_SAMPLE]
    msg = (
        f"output_folderpath={output_folderpath!r} already holds a work list "
        f"({len(cached_ids)} cell ids, in {os.path.join(output_folderpath, 'cells', 'input.csv')}) "
        f"that does not match the freshly tiled roi ({len(fresh_ids)} cell ids). "
        f"output_folderpath is the identity of a run (spec 47 D3) -- reusing one for a different "
        f"roi (or a different grid_size_km/scale_fact) would silently resume the OLD work list. "
        f"Sample of the differing ids: {sample}. Fix: use a new output_folderpath for this roi."
    )
    # A strict superset short by only a handful of ids is a re-tiling, not an unrelated roi.
    # Say so, or the message sends the reader hunting for a mix-up that did not happen.
    if not only_fresh and 0 < len(only_cached) <= _RESUME_DRIFT_MAX_MISSING:
        msg += (
            f" This looks like the spec-46 D4 cell-count change (2026-08-19, e.g. 300->299 "
            f"cells for AT_ROI): the cached set is a strict superset missing only "
            f"{len(only_cached)} id(s), rather than a disjoint set from an unrelated roi. If so, "
            f"the fix is the same: use a new output_folderpath."
        )
    raise PreflightError(msg)


def _artifacts_present(folder: str, names: list[str]) -> bool:
    """Every named file under `folder` exists AND is non-empty -- a half-written artifact
    (#74's class of defect: a truncated write catalogued as complete) must not read as
    "done"."""
    for name in names:
        fp = os.path.join(folder, name)
        if not fs.exists(fp) or fs.size(fp) == 0:
            return False
    return True


def _normalize_window(startdate, enddate) -> tuple[pd.Timestamp | None, pd.Timestamp | None, list[str]]:
    """Coerce the caller's dates to tz-aware UTC `Timestamp`s ONCE, at the API boundary.

    Every downstream call (window checks, sources, the AML dispatch) must forward the SAME
    typed value, because `pystac_client` widens a date-only STRING to end-of-day and a
    `datetime`/`Timestamp` not at all (pystac-client #644) -- so normalizing per-caller
    instead of here silently gives two search windows for one date.

    Returns errors instead of raising so callers aggregate them with the rest of preflight
    in one pass. `(None, None, [...])` on failure -- callers must skip any check that needs
    the parsed values when errs is non-empty. TODO #52.
    """
    errs = []
    try:
        start = pd.to_datetime(startdate, utc=True)
    except (ValueError, TypeError) as exc:
        errs.append(f"startdate={startdate!r} is not a valid date: {exc}")
        start = None
    try:
        end = pd.to_datetime(enddate, utc=True)
    except (ValueError, TypeError) as exc:
        errs.append(f"enddate={enddate!r} is not a valid date: {exc}")
        end = None
    return start, end, errs


# --- result handle -----------------------------------------------------------

@dataclass
class TrainingData:
    """Handle to a completed training-data build (paths; lazy-load arrays)."""

    export_folderpath: str      # data.npy / ids.npy / labels.npy / coords.npy / metadata.pickle.npy
    run_folderpath: str         # per-field datacubes + input.csv
    n_pixels: int
    n_timestamps: int
    bands: list[str]
    feature_bands: list[str] | None = None   # set when a feature transform was applied

    def load(self) -> dict:
        """Load the arrays into memory: data/ids/coords/metadata (+ labels if present).

        When a feature transform was applied (via `adapter=`/`feature_sequence=`), also loads
        `features`/`feature_ids`/`feature_labels` (the model-ready, possibly aggregated arrays).
        """
        out = {
            "data": fs.load_npy(os.path.join(self.export_folderpath, "data.npy")),
            "ids": fs.load_npy(os.path.join(self.export_folderpath, "ids.npy")),
            "coords": fs.load_npy(os.path.join(self.export_folderpath, "coords.npy")),
            "metadata": fs.load_npy(
                os.path.join(self.export_folderpath, "metadata.pickle.npy"), allow_pickle=True
            )[()],
        }
        labels_path = os.path.join(self.export_folderpath, "labels.npy")
        if fs.exists(labels_path):
            out["labels"] = fs.load_npy(labels_path)
        features_path = os.path.join(self.export_folderpath, "features.npy")
        if fs.exists(features_path):
            out["features"] = fs.load_npy(features_path)
            out["feature_ids"] = fs.load_npy(
                os.path.join(self.export_folderpath, "feature_ids.npy")
            )
            fl = os.path.join(self.export_folderpath, "feature_labels.npy")
            if fs.exists(fl):
                out["feature_labels"] = fs.load_npy(fl)
        return out


@dataclass
class InferenceResult:
    """Handle to a completed local inference run."""

    output_folderpath: str
    output_filepaths: list[str]
    stac_catalog_filepath: str
    merged_filepath: str | None = None
    grids_filepath: str | None = None  # ROI mode: the saved gridded-ROI GeoJSON


# --- verbs -------------------------------------------------------------------

def download(
    roi,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    bands: list[str],
    dst_folderpath: str,
    creds: CdseCredentials | None = None,
    *,
    source: str = "mpc",
    collection: str = config.SATELLITE_S2L2A,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str = _processing.LATEST,
    max_tiles: int,
    max_cloudcover: float | None = None,
    cog: bool = True,
    progress: bool = True,
    max_concurrent: int | None = None,
    storage=None,
    runner: str = "local",
    runner_kwargs: dict | None = None,
) -> str:
    """Fetch `collection` granules for the ROI/date range into the archive rooted at
    `dst_folderpath`, build/append the collection's TileCatalog, and return that catalog's
    filepath (feed it to `create_training_data`).

    **Archive layout.** `dst_folderpath` is the archive ROOT; granules land in
    `{dst_folderpath}/{collection}/YYYY/MM/DD/{canonical granule name}/` and the catalog is
    `{dst_folderpath}/{collection}/catalog.parquet` (one catalog per collection directory).
    The canonical granule name keeps every processing field and drops every provider-specific
    one, so the same granule fetched from MPC and CDSE lands in ONE folder and ONE catalog row
    (`source` = `"cdse,mpc"`), while two processings of one acquisition coexist on disk.

    **`processing`** chooses which processing of each acquisition to FETCH: `"latest"`
    (default) or a PEP 440 specifier (`"==05.00"` to match training data, `">=05.00"` to
    refuse old baselines), applied per acquisition to what the chosen `source` offers, after
    `properties_filter` and before `max_tiles`. `None` is refused. Every skipped granule is
    printed, and so is any acquisition the archive now holds in more than one processing -- a
    build over those needs its own `processing=`.

    `source` (provider) and `collection` (product) are orthogonal. `source="mpc"` (default,
    credential-free) wraps `sources.mpc.download` (Microsoft Planetary Computer, anonymous:
    `creds` is not required and `cog` is ignored, MPC assets are already COG);
    `source="cdse"` wraps `sources.cdse.download` and requires `creds`. Each source serves a
    fixed set of collections (`sources.mpc.SERVED_COLLECTIONS`/`sources.cdse.SERVED_COLLECTIONS`);
    an unserved pair raises at preflight, naming what the source DOES serve. `storage` is
    local or `"azure"`.

    `max_cloudcover` requires the collection to declare `supports_cloud_cover=True` (every
    optical collection, not Sentinel-1); passing it against a collection with no cloud-cover
    concept raises rather than silently being a no-op filter.

    `properties_filter` narrows discovered granules by STAC property, e.g.
    `{"sat:orbit_state": "descending"}`, **before** `max_tiles` is checked. A download is a
    whole-asset byte copy -- a ~5 km ROI still fetches entire ~250 km Sentinel-1 scenes --
    and a build can only ever use ONE value of a partitioned property, so fetching the others
    is pure waste. `source='cdse'` does not implement it and says so at preflight rather than
    ignoring it.

    `max_concurrent` is how many band files transfer at once (`sources.mpc.download`'s
    `max_concurrent`, `sources.cdse.download`'s `max_concurrent_s3`). `None` keeps each
    source's default -- for MPC that is `config.MPC_MAX_CONCURRENT` (4), chosen for a
    one-granule smoke run and far too low for a whole archive: 200+ granules x 4 bands over
    4 threads is latency-bound, not bandwidth-bound. Raise it (16-32 against MPC, a public
    Azure endpoint) when the run is large.

    `runner="local"` (default) downloads in-process. `runner="aml"` dispatches onto an Azure
    ML cluster colocated with blob: CDSE runs as **one** job; MPC **fans out** across N.
    `runner_kwargs` carries `cluster=`/`environment=`/`root=`/`identity_client_id=` and, for
    CDSE, exactly one of `vault_url=`+`secret_name=` (Key Vault) or `creds_url=` (blob JSON)
    -- see `workflows.runners.run_aml_download`. `creds` is ignored for `runner="aml"`: the
    job reads them on the node, so `roi` must be a url the node can also read, never an
    in-memory GeoDataFrame. CDSE only ever serves Sentinel-2 L2A, so its job needs no
    `collection`. `processing` reaches both AML shapes.

    `dst_folderpath` is the identity of this download: its `TileCatalog` is what a re-run
    diffs against to skip what is already there, so re-running with a different `roi`/
    `startdate`/`enddate`/`bands` into the same `dst_folderpath` appends into one shared
    catalog rather than starting a new one.

    Spec: 58, 59.
    """
    startdate, enddate, date_errs = _normalize_window(startdate, enddate)
    errs = _check_local_seams(runner, storage) + date_errs
    if not date_errs:
        errs += _check_window(startdate, enddate, 20, bands)
    if source not in ("cdse", "mpc"):
        errs.append(f"source={source!r} must be one of 'cdse', 'mpc'.")
    else:
        errs += _check_source_collection(source, collection)
    if max_tiles < 1:
        errs.append(f"max_tiles ({max_tiles}) must be >= 1.")
    if runner == "local" and source == "cdse" and creds is None:
        errs.append("creds (CdseCredentials) required for source='cdse' with runner='local'.")
    if max_cloudcover is not None and not errs:
        errs += _check_cloudcover_capability(collection, max_cloudcover)
    if properties_filter and source != "mpc":
        # Loud, not silently ignored: a caller who narrowed to one orbit and still got
        # every orbit's bytes would only find out from the disk bill.
        errs.append(
            f"properties_filter is not implemented for source={source!r} (only 'mpc'); "
            "drop it, or use source='mpc'."
        )
    errs += _processing.processing_errors(processing, collection=collection, allow_none=False)
    _raise_preflight(errs)

    _configure_storage(storage)
    # `dst_folderpath` is the archive root; the catalog lives in the collection's own
    # directory.
    fs.makedirs(dst_folderpath)
    catalog_filepath = archive_catalog_filepath(dst_folderpath, collection)
    fs.makedirs(os.path.dirname(catalog_filepath))

    if runner == "aml":
        from fsd.workflows import runners as _runners

        _runners.run_aml_download(
            roi=roi, startdate=startdate, enddate=enddate, bands=bands,
            dst_folderpath=dst_folderpath, catalog_filepath=catalog_filepath,
            source=source, max_tiles=max_tiles, max_cloudcover=max_cloudcover, cog=cog,
            collection=collection, properties_filter=properties_filter,
            processing=processing, **(runner_kwargs or {}),
        )
        return catalog_filepath

    catalog = TileCatalog(catalog_filepath)
    if source == "mpc":
        _mpc_download(
            roi=roi, startdate=startdate, enddate=enddate, bands=bands,
            root_folderpath=dst_folderpath, catalog=catalog,
            max_tiles=max_tiles, max_cloudcover=max_cloudcover, progress=progress,
            collection=collection, max_concurrent=max_concurrent,
            properties_filter=properties_filter, processing=processing,
        )
    else:
        _cdse_download(
            roi=roi, startdate=startdate, enddate=enddate, bands=bands,
            root_folderpath=dst_folderpath, catalog=catalog, creds=creds,
            max_tiles=max_tiles, max_cloudcover=max_cloudcover, cog=cog, progress=progress,
            collection=collection, max_concurrent_s3=max_concurrent, processing=processing,
        )
    return catalog_filepath


_download_verb = download  # internal alias: `create_training_data`'s `download` bool param
                           # shadows the module-level `download` verb within its own body.


def create_training_data(
    label_polygons,
    catalog_filepath: str,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    mosaic_days: int,
    bands: list[str],
    id_col: str,
    export_folderpath: str,
    *,
    label_col: str | None = None,
    adapter=None,
    feature_sequence=None,
    aggregate=None,
    cores: int = 1,
    source: str = "mpc",
    collection: str = config.SATELLITE_S2L2A,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str | None = None,
    download: bool = False,
    max_tiles: int | None = None,
    max_cloudcover: float | None = None,
    cog: bool = True,
    creds: CdseCredentials | None = None,
    overwrite: bool | str = False,
    runner: str = "local",
    runner_kwargs: dict | None = None,
    storage=None,
    run_folderpath: str | None = None,
) -> TrainingData:
    """Label polygons (+ imagery) -> flattened, locally-landed training arrays: the
    full-pipeline façade.

    Runs an optional download phase, `workflows.create_datacube` (one datacube per polygon,
    calendar mosaic), then `flatten_training_data`. Returns a `TrainingData` handle.

    **Skips work already done.** The download leg diffs against the catalog. The build leg
    diffs `input.csv`'s `datacube_filepath` column against what already exists: a shortfall
    of 0 submits no job. The flatten leg is skipped when `_flatten_stamp.json` already
    records the identity -- never the modification time -- of exactly this cube set + these
    run parameters (`bands`/`mosaic_days`/window/`aggregate`/feature transform). `overwrite=`
    forces past this: ``False`` (default) skips whatever is already done; ``"datacubes"``
    rebuilds the cubes (and therefore re-flattens, since the caller has explicitly asked for
    a rebuild); ``"flatten"`` keeps the cubes and redoes the flatten; ``True`` does both.
    Every skip prints one line naming what it skipped and why.

    **Download phase:** `download=False` (default) requires `catalog_filepath` to already
    exist — run `fsd.download` first, because compute never fetches from a provider
    implicitly. `download=True` first calls `fsd.download(roi=label_polygons, ...)` into
    `catalog_filepath`'s folder (`source="mpc"` default; `"cdse"` needs `creds` for
    `runner="local"`), then proceeds to build + flatten. `max_tiles` is required when
    `download=True`: a silent default here is an unbounded fetch.

    **Blob-vs-local split (`runner="aml"` only):** `export_folderpath` is always the LOCAL
    landing target for the compact array; the blob working root is `runner_kwargs["root"]`
    (catalog/cubes/`input.csv`/the raw flatten output all live there). `run_folderpath` defaults
    to a folder under that blob root for `runner="aml"`, and to `export_folderpath/run` only for
    `runner="local"`.

    **In-memory polygons:** for `runner="aml"`, an in-memory `label_polygons` GeoDataFrame is
    materialized once to a GeoJSON under `run_folderpath` (by default under the blob `root`)
    and that one URL serves as both the download ROI and the per-cell build shapefile. A
    path/URL `label_polygons` is used as-is.

    **Feature engineering:** pass an `adapter` (preferred — its `feature_sequence` is the
    *same* one used at inference, which is what keeps training and inference features from
    skewing apart) **or** a raw `feature_sequence` (adapter-less/exploratory). `aggregate` ∈
    {None, "median_per_id", callable} reduces per-pixel samples before the transform. When any
    is given, fsd writes `features.npy` (+ `feature_ids`/`feature_labels`) additively **on the
    driver, after land-local** — the raw `data.npy` is kept and cluster images stay
    general-purpose.

    `label_col` is optional: when omitted, no `labels.npy` is written and `ids.npy` is the
    join key for labels joined in later, without re-flattening.

    `runner="aml"` dispatches download + the build fan-out + the flatten reduce onto an Azure
    ML cluster; `runner_kwargs` carries `cluster=`/`environment=`/`root=`/`identity_client_id=`
    (see `workflows.runners`).

    `properties_filter` narrows the catalog to rows matching every given STAC property
    (e.g. `{"sat:orbit_state": "descending"}`) before the build -- required for
    `collection="sentinel-1-rtc"`, whose ascending/descending passes must never be medianed
    together (a build spanning both raises, enumerating what is available). A key no row in
    the catalog carries at all raises rather than silently filtering to zero rows. Omitting
    it (the default) is a no-op for a collection with no `mosaic_partition` (e.g. S2 L2A).

    `processing` resolves acquisitions the archive holds in more than one processing (two
    baselines; the same granule from two sources): `None` (default) RAISES naming each
    duplicate group and the arguments that resolve it; `"latest"` keeps the newest per
    acquisition; a PEP 440 specifier (`">=05.00"`) keeps rows satisfying it, then the newest
    (an acquisition with none is dropped and reported). It joins the cube path when not
    `None`. With `download=True` a specifier is also forwarded to the download leg, and
    `None` uses the download default `"latest"`.

    `catalog_filepath` is `{archive root}/{collection}/catalog.parquet`; its parent directory
    must be named for `collection`, and `download=True` fetches into the archive root two
    levels up.

    Spec: 49, 58, 59.
    """
    if adapter is not None and feature_sequence is not None:
        raise PreflightError(
            "pass either `adapter` or `feature_sequence`, not both (ambiguous feature transform)."
        )
    if overwrite not in _VALID_OVERWRITE:
        raise PreflightError(
            f"overwrite={overwrite!r} must be one of {list(_VALID_OVERWRITE)} (spec 49 D4)."
        )

    # Canonicalize bands to native asset keys as early as possible: every
    # downstream use -- the adapter check, the flatten identity, the cube digest -- must
    # see one spelling, or `bands=["B8A"]` and `bands=["nir08"]` would behave differently
    # despite naming the identical band.
    _declaration = _collections.get(collection)
    bands = [_declaration.canonical_to_native(b) for b in bands]

    startdate, enddate, date_errs = _normalize_window(startdate, enddate)
    errs = _check_local_seams(runner, storage) + date_errs
    if not date_errs:
        errs += _check_window(startdate, enddate, mosaic_days, bands)
    if adapter is not None:
        req = [_declaration.canonical_to_native(b)
               for b in (getattr(adapter, "required_bands", []) or [])]
        missing = [b for b in req if b not in bands]
        if missing:
            errs.append(f"adapter.required_bands not in requested bands: {missing}")
        # Deliberately no `n_timestamps` preflight: T is the caller's to choose via
        # window/mosaic_days, and an adapter is expected to fit whatever T they pick.
        # The cross-cube invariant is still enforced -- flatten raises if two cubes
        # disagree on the timestamps axis.
    try:
        _resolve_aggregate(aggregate)
    except ValueError as exc:
        errs.append(str(exc))
    errs += _processing.processing_errors(processing, collection=collection, allow_none=True)
    errs += _check_catalog_collection_dir(catalog_filepath, collection)

    gdf = None
    try:
        gdf = _as_gdf(label_polygons)
        if id_col not in gdf.columns:
            errs.append(f"column {id_col!r} not in label_polygons.")
        if label_col is not None and label_col not in gdf.columns:
            errs.append(f"column {label_col!r} not in label_polygons.")
        if len(gdf) == 0:
            errs.append("label_polygons is empty.")
        elif gdf.geometry.isna().any():
            errs.append("label_polygons has null geometries.")
    except Exception as exc:  # unreadable polygons is a preflight failure, not a crash
        errs.append(f"could not read label_polygons: {exc}")

    root = (runner_kwargs or {}).get("root")
    if runner == "aml" and not root:
        errs.append("runner_kwargs['root'] (the blob working root) is required for runner='aml'.")

    # Wave 1: raise the STRUCTURAL errors now, before any catalog access. The top-level
    # short-circuit below must be reachable without `catalog_filepath` existing at all,
    # so catalog/download preflight is deferred to wave 2 and runs only if it does not fire.
    _raise_preflight(errs)

    _configure_storage(storage)

    run_id = None
    if runner == "aml":
        # `run_id` is fresh per SUBMISSION and names only `shards/` + `_status/` under
        # `run_aml`'s own run_root -- it identifies a dispatch, which is what it should
        # identify. ARTIFACT paths must NOT carry it: a clock in an artifact path makes
        # every target missing on every call, so no skip can ever fire (#83). Hence the
        # plain, stable "train" here; per-cell granularity comes from `<params>/<id>` below.
        run_id = (runner_kwargs or {}).get("run_id") or pd.Timestamp.now(tz="UTC").strftime(
            "%Y%m%dT%H%M%SZ"
        )
        if run_folderpath is None:
            run_folderpath = f"{root.rstrip('/')}/runs/train"
    elif run_folderpath is None:
        run_folderpath = os.path.join(export_folderpath, "run")

    # The top-level short-circuit. The target is the landed arrays, and its identity is
    # computed from the REQUEST, never from `input.csv` -- so checking it costs zero
    # catalog access, zero `setup`, zero dispatch, even on a call whose `input.csv` has
    # never been written. Any `overwrite` other than `False` is a forced rebuild and must
    # never short-circuit.
    if overwrite is False:
        want_features = adapter is not None or feature_sequence is not None or aggregate is not None
        identity = _flatten_identity_from_request(
            gdf, id_col=id_col, run_folderpath=run_folderpath,
            startdate=startdate, enddate=enddate, mosaic_days=mosaic_days,
            bands=bands, collection=collection,
            mosaic_scheme=config.MOSAIC_SCHEME,
            adapter=adapter, feature_sequence=feature_sequence, aggregate=aggregate,
            properties_filter=properties_filter, processing=processing,
        )
        stamp_filepath = os.path.join(export_folderpath, _FLATTEN_STAMP_NAME)
        if _stamp.matches_stamp(stamp_filepath, identity) and _flatten_outputs_present(
            export_folderpath, label_col=label_col, want_features=want_features,
        ):
            print(f"[plan] target: {export_folderpath} arrays -> CURRENT "
                  f"(stamp matches this request)", flush=True)
            return _land_current_training_data(
                export_folderpath, run_folderpath, label_col=label_col,
                want_features=want_features,
            )

    # Announce what the walk resolved before running anything. Reached only when the
    # top-level short-circuit above did NOT fire.
    if overwrite is not False:
        stale_reason = f"overwrite={overwrite!r} forces a rebuild"
    else:
        stamp = _stamp.read_stamp(os.path.join(export_folderpath, _FLATTEN_STAMP_NAME))
        stale_reason = "no stamp" if stamp is None else "stamp does not match this request"
    print(f"[plan] target: {export_folderpath} arrays -> STALE ({stale_reason})", flush=True)
    print(f"[plan]   flatten: {len(gdf)} cubes required", flush=True)

    fs.makedirs(run_folderpath)
    fs.makedirs(export_folderpath)

    # Materialize an in-memory GeoDataFrame once, under run_folderpath -- the SAME url
    # then feeds both the download ROI and the build shapefile. Written via the storage
    # seam, never `gdf.to_file`, which needs a real local path and so would fail on a
    # blob run_folderpath (mirrors `create_datacube.setup`).
    if isinstance(label_polygons, gpd.GeoDataFrame):
        shapefilepath = os.path.join(run_folderpath, "label_polygons.geojson")
        with fs.open(shapefilepath, "w") as f:
            # `default=str` so Timestamp/datetime property columns (e.g. EuroCrops' obs
            # date) serialize -- gdf.to_json() routes through json.dumps, which (unlike
            # the GDAL GeoJSON driver the old gdf.to_file used) can't encode a Timestamp.
            f.write(gdf.to_json(default=str))
    else:
        shapefilepath = label_polygons

    # Wave 2 -- catalog/download preflight. Unreachable when the short-circuit above
    # already fired, which is the point: a satisfied re-run needs `catalog_filepath` to
    # exist no more than it needs `setup` to run.
    catalog_errs: list[str] = []
    if download:
        if source not in ("cdse", "mpc"):
            catalog_errs.append(f"source={source!r} must be one of 'cdse', 'mpc'.")
        else:
            catalog_errs += _check_source_collection(source, collection)
        if max_tiles is None or max_tiles < 1:
            catalog_errs.append(
                f"max_tiles (>= 1) is required when download=True (got {max_tiles!r})."
            )
        if runner == "local" and source == "cdse" and creds is None:
            catalog_errs.append(
                "creds (CdseCredentials) required for source='cdse' with runner='local'."
            )
        if max_cloudcover is not None and not catalog_errs:
            catalog_errs += _check_cloudcover_capability(collection, max_cloudcover)
    else:
        catalog_present = fs.exists(catalog_filepath)
        if not catalog_present:
            catalog_errs.append(
                f"catalog_filepath does not exist: {catalog_filepath} "
                "— run fsd.download first, or pass download=True (compute never fetches "
                "from a provider implicitly; spec 23 D13)."
            )
        # Catalog exists but covers NONE of the fields in-window -> hand back an actionable
        # download plan rather than an empty build. The offline `.filter` is cheap and the
        # STAC-backed plan only fires on the empty case, so this costs nothing in the
        # normal path. Only meaningful when NOT auto-downloading.
        if catalog_present and len(gdf) and not gdf.geometry.isna().any():
            try:
                covered = TileCatalog(catalog_filepath).filter(gdf, startdate, enddate)
            except Exception:  # noqa: BLE001 - a bad filter just means "skip the coverage hint"
                covered = None
            if covered is not None and len(covered) == 0:
                catalog_errs.append(_imagery_missing_message(
                    gdf, startdate, enddate, bands, catalog_filepath=catalog_filepath,
                    why="no catalog tiles intersect the label polygons in-window",
                ))
    _raise_preflight(catalog_errs)

    if download:
        # The catalog lives at `{root}/{collection}/catalog.parquet`, so the archive root is
        # its GRANDPARENT (the parent-name check ran in wave 1).
        dst_folderpath = os.path.dirname(os.path.dirname(catalog_filepath.rstrip("/"))) or "."
        _download_verb(
            roi=shapefilepath, startdate=startdate, enddate=enddate, bands=bands,
            dst_folderpath=dst_folderpath, creds=creds, source=source, collection=collection,
            properties_filter=properties_filter,
            # A specifier is forwarded; `None` (build default) fetches the download default,
            # "latest" -- one processing per acquisition, so the build then sees no ambiguity
            # it did not already have.
            processing=processing if processing is not None else _processing.LATEST,
            max_tiles=max_tiles, max_cloudcover=max_cloudcover, cog=cog,
            storage=storage, runner=runner, runner_kwargs=runner_kwargs,
        )

    # `"datacubes"` forces the flatten leg too: a caller who asked for a cube rebuild must
    # never be handed arrays flattened from the cubes being replaced.
    build_overwrite = overwrite in (True, "datacubes")
    flatten_overwrite = overwrite in (True, "flatten", "datacubes")

    print("[plan] will run: build -> flatten -> land", flush=True)

    csv_filepath = os.path.join(run_folderpath, "input.csv")
    _create_datacube.run_create_datacube(
        catalog_filepath=catalog_filepath, timestamp_col="timestamp",
        shapefilepath=shapefilepath, id_col=id_col, run_folderpath=run_folderpath,
        startdate=startdate, enddate=enddate, bands=bands,
        mosaic_days=mosaic_days,
        csv_filepath=csv_filepath, label_col=label_col, cores=cores,
        # Setup is scoped to the shortfall unless the caller forces a cube rebuild, in
        # which case setup re-reads the catalog too -- otherwise a rebuild would be fed by
        # the stale per-shape `catalog.parquet` slice it was meant to replace. Ordinary
        # re-runs and `overwrite="flatten"` both take the scoped path.
        overwrite_setup_csv=build_overwrite,
        overwrite=build_overwrite, runner=runner, runner_kwargs=runner_kwargs,
        collection=collection, properties_filter=properties_filter, processing=processing,
    )

    # Flatten phase delegates to `flatten_training_data` -- no duplicated reduce/
    # land/features logic. Reuse the SAME run_id (aml) so the flatten reduce writes to a
    # sibling `.../_flatten` prefix under the build's own run_folderpath.
    flatten_runner_kwargs = runner_kwargs
    if runner == "aml":
        flatten_runner_kwargs = dict(runner_kwargs or {})
        flatten_runner_kwargs["run_id"] = run_id

    td = flatten_training_data(
        csv_filepath, export_folderpath,
        id_col="id", label_col=("label" if label_col is not None else None),
        filepath_col="datacube_filepath",
        adapter=adapter, feature_sequence=feature_sequence, aggregate=aggregate,
        overwrite=flatten_overwrite, runner=runner, runner_kwargs=flatten_runner_kwargs,
    )

    return TrainingData(
        export_folderpath=td.export_folderpath, run_folderpath=run_folderpath,
        n_pixels=td.n_pixels, n_timestamps=td.n_timestamps, bands=td.bands,
        feature_bands=td.feature_bands,
    )


_FLATTEN_STAMP_NAME = "_flatten_stamp.json"


def _flatten_identity(input_df: pd.DataFrame, *, id_col, filepath_col, adapter, feature_sequence,
                      aggregate) -> dict:
    """Answers "were these arrays derived from exactly this request?"

    The sorted `(id, datacube_filepath)` pairs `input_df` names, plus the run parameters
    that shape the arrays (read off `input_df`'s own columns, written there by
    `create_datacube.setup`), plus the feature transform. Never a modification time.

    The feature transform is fingerprinted by qualname + kwargs, so editing a feature
    function's BODY under the same name does NOT invalidate a stamp -- rename it, or pass
    `overwrite="flatten"`.
    """
    cubes = sorted(
        [str(row[id_col]), str(row[filepath_col])] for _, row in input_df.iterrows()
    )
    params: dict = {}
    for col in ("bands", "mosaic_days", "startdate", "enddate", "collection",
               "mosaic_scheme", "properties_filter", "processing"):
        if col in input_df.columns:
            # An empty/NaN field round-trips through CSV as NaN, not "" -- normalize,
            # or this never matches `_flatten_identity_from_request`'s freshly-computed
            # "" for the same request.
            params[col] = sorted(set(input_df[col].fillna("").astype(str)))
    # An older `input.csv` has no `processing` column; that is the same request as
    # `processing=None`, which `_flatten_identity_from_request` records as [""].
    params.setdefault("processing", [""])
    params["aggregate"] = _fingerprint_aggregate(aggregate)
    params["features"] = _fingerprint_features(adapter, feature_sequence)
    identity = {"cubes": cubes, "params": params}
    # Canonicalize through a JSON round-trip: `write_stamp`/`read_stamp` compare against a
    # stamp that has already made this trip (tuples -> lists, non-JSON kwargs -> repr), so
    # comparing a freshly-computed identity that has NOT would spuriously mismatch.
    return json.loads(json.dumps(identity, default=str))


def _flatten_identity_from_request(
    gdf: gpd.GeoDataFrame, *, id_col: str, run_folderpath: str,
    startdate, enddate, mosaic_days: int, bands: list[str],
    collection: str, mosaic_scheme: str,
    adapter, feature_sequence, aggregate,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str | None = None,
) -> dict:
    """The same identity `_flatten_identity` computes, but from the REQUEST rather than
    from `input.csv`.

    That is what makes the top-level short-circuit possible: `input.csv` is `setup`'s
    OUTPUT, so an identity derived from it cannot be checked before `setup` runs. A cube's
    path is derivable from `(run_folderpath, window, id)` and nothing else, so naming the
    targets costs no catalog read, no `input.csv` read, and no `setup` call.

    It does read one small file -- `_manifest.json`, to subtract the known-empty ids. So
    the manifest and `input.csv` must agree about which ids have rows, which is what
    `create_datacube._forget_known_empty`/`_clear_known_empty` maintain; if they drift, a
    satisfied request stops matching its own stamp.

    Produces the exact same dict shape as `_flatten_identity(input_df, ...)` -- same
    `cubes` list, same `params` keys, same string forms -- so the two are comparable.
    `gdf[id_col]` is sorted, for a `cubes` order that does not depend on shapefile row
    order.
    """
    window_segment = _create_datacube.window_folder_segment(
        startdate, enddate, mosaic_days, bands=bands, mosaic_scheme=mosaic_scheme,
        collection=collection, declaration=_collections.get(collection),
        properties_filter=properties_filter, processing=processing,
    )
    # `input.csv` never gets a row for a shape `setup` found no imagery for, so
    # `_flatten_identity` -- computed FROM `input.csv` -- never names them either. Without
    # this subtraction a request containing even one such cell could never match its own
    # stamp, and would rebuild on every call.
    known_empty = _create_datacube._read_known_empty(run_folderpath, window_segment)
    ids = sorted(str(v) for v in gdf[id_col] if str(v) not in known_empty)
    cubes = sorted(
        [id_value, os.path.join(
            _create_datacube.cube_export_folderpath(run_folderpath, window_segment, id_value),
            "datacube.npy",
        )]
        for id_value in ids
    )
    params = {
        "bands": [",".join(bands)],
        "mosaic_days": [str(mosaic_days)],
        "startdate": [str(pd.to_datetime(startdate, utc=True))],
        "enddate": [str(pd.to_datetime(enddate, utc=True))],
        "collection": [collection],
        "mosaic_scheme": [mosaic_scheme],
        "properties_filter": [_create_datacube._canonicalize_properties_filter(properties_filter)],
        "processing": [_create_datacube._canonicalize_processing(processing)],
        "aggregate": _fingerprint_aggregate(aggregate),
        "features": _fingerprint_features(adapter, feature_sequence),
    }
    identity = {"cubes": cubes, "params": params}
    return json.loads(json.dumps(identity, default=str))


def _fingerprint_aggregate(aggregate):
    if aggregate is None:
        return None
    if isinstance(aggregate, str):
        return aggregate
    return _stamp.compute_callable_fingerprint(aggregate)


def _fingerprint_sequence(sequence) -> list:
    return [[_stamp.compute_callable_fingerprint(fn), kwargs] for fn, kwargs in sequence]


def _fingerprint_features(adapter, feature_sequence) -> dict | None:
    if adapter is not None:
        cls = type(adapter)
        identity: dict = {"adapter_class": f"{cls.__module__}.{cls.__qualname__}"}
        seq = getattr(adapter, "feature_sequence", None)
        if seq is not None:
            identity["feature_sequence"] = _fingerprint_sequence(seq)
        else:
            identity["features_method"] = _stamp.compute_callable_fingerprint(adapter.features)
        return identity
    if feature_sequence is not None:
        return {"feature_sequence": _fingerprint_sequence(feature_sequence)}
    return None


def _flatten_output_names(*, label_col, want_features: bool) -> list[str]:
    names = ["data.npy", "coords.npy", "ids.npy", "metadata.pickle.npy"]
    if label_col is not None:
        names.append("labels.npy")
    if want_features:
        names += ["features.npy", "feature_ids.npy"]
        if label_col is not None:
            names.append("feature_labels.npy")
    return names


def _flatten_outputs_present(export_folderpath: str, *, label_col, want_features: bool) -> bool:
    """The arrays a skip would reuse must actually be there: a matching stamp over a
    half-cleaned `export_folderpath` has to fail towards running, never towards a skip
    that then cannot load."""
    names = _flatten_output_names(label_col=label_col, want_features=want_features)
    return _artifacts_present(export_folderpath, names)


def _load_landed_arrays(export_folderpath: str):
    """`data.npy` + `metadata.pickle.npy`, loaded off `fsd.storage` -- the two files every
    landed `TrainingData` needs, shared by `flatten_training_data`'s tail and the
    top-level short-circuit."""
    data = fs.load_npy(os.path.join(export_folderpath, "data.npy"))
    metadata = fs.load_npy(
        os.path.join(export_folderpath, "metadata.pickle.npy"), allow_pickle=True
    )[()]
    return data, metadata


def _land_current_training_data(
    export_folderpath: str, run_folderpath: str, *, label_col, want_features: bool,
) -> TrainingData:
    """The target is already CURRENT: print the `[fetch]` line and return the arrays,
    without touching the catalog, `setup`, or a runner.

    `want_features` only says which files to report -- nothing here recomputes them,
    because the stamp match already proved they came from this exact request.
    """
    data, metadata = _load_landed_arrays(export_folderpath)
    names = _flatten_output_names(label_col=label_col, want_features=want_features)
    total_bytes = sum(
        fs.size(os.path.join(export_folderpath, n))
        for n in names if fs.exists(os.path.join(export_folderpath, n))
    )
    print(f"[fetch] export -> {export_folderpath} | {len(names)} files, "
          f"{total_bytes / 1e6:.1f} MB", flush=True)
    return TrainingData(
        export_folderpath=export_folderpath, run_folderpath=run_folderpath,
        n_pixels=int(data.shape[0]), n_timestamps=len(metadata["timestamps"]),
        bands=list(metadata["bands"]), feature_bands=metadata.get("feature_bands"),
    )


def flatten_training_data(
    input_csv: str,
    export_folderpath: str,
    *,
    id_col: str = "id",
    label_col: str | None = None,
    filepath_col: str = "datacube_filepath",
    nodata: int = config.NODATA,
    adapter=None,
    feature_sequence=None,
    aggregate=None,
    overwrite: bool = False,
    runner: str = "local",
    runner_kwargs: dict | None = None,
    storage=None,
) -> TrainingData:
    """Flatten already-built cubes (an `input_csv` of `datacube_filepath`s) into one training
    array, landed locally — the flatten-only sibling of `create_training_data`, for
    cubes that already exist on blob.

    `runner="local"` (default): `datacube.flatten.flatten` runs in-process (cubes stream over
    the storage seam) straight to the local `export_folderpath`. `runner="aml"`: dispatches a
    single-node cluster reduce (writes to a blob prefix under `runner_kwargs["root"]`, no
    `shard_units` fan-out — flatten concatenates ALL cubes into ONE array), then
    `storage.transfer`s the compact result down to the local `export_folderpath`. The driver
    never pulls the raw cubes itself. Both branches then apply the optional feature transform
    **on the driver only**, so cluster images stay general-purpose and emit raw arrays.

    `label_col` optional: `labels.npy` is written only when given.

    **Skip:** on completion this writes `_flatten_stamp.json` recording the
    identity of the cubes + run parameters it was derived from. `overwrite=False` (default):
    if a later call's identity matches the stamp AND every array is still present, the reduce
    is skipped entirely and the existing arrays are returned as a `TrainingData` -- otherwise
    (mismatch, missing stamp, missing/corrupt arrays) it reduces as normal. `overwrite=True`
    always reduces. The comparison never reads a modification time: a cube rebuilt
    under the same id/path is caught only if the id/path SET or the run parameters differ --
    see `_flatten_identity`.
    """
    if adapter is not None and feature_sequence is not None:
        raise PreflightError(
            "pass either `adapter` or `feature_sequence`, not both (ambiguous feature transform)."
        )

    errs = _check_local_seams(runner, storage)
    try:
        _resolve_aggregate(aggregate)
    except ValueError as exc:
        errs.append(str(exc))
    if runner == "aml":
        rk = runner_kwargs or {}
        for key in ("cluster", "environment", "root", "identity_client_id"):
            if not rk.get(key):
                errs.append(f"runner_kwargs[{key!r}] is required for runner='aml'.")
    if not fs.exists(input_csv):
        errs.append(f"input_csv does not exist: {input_csv!r}")
    _raise_preflight(errs)

    _configure_storage(storage)
    fs.makedirs(export_folderpath)

    with fs.open(input_csv, "r") as f:
        input_df = pd.read_csv(f)

    want_features = adapter is not None or feature_sequence is not None or aggregate is not None
    identity = _flatten_identity(
        input_df, id_col=id_col, filepath_col=filepath_col,
        adapter=adapter, feature_sequence=feature_sequence, aggregate=aggregate,
    )
    stamp_filepath = os.path.join(export_folderpath, _FLATTEN_STAMP_NAME)

    skip = False
    if not overwrite:
        skip = _stamp.matches_stamp(stamp_filepath, identity) and _flatten_outputs_present(
            export_folderpath, label_col=label_col, want_features=want_features,
        )

    if skip:
        print(f"[flatten] arrays match the current {len(input_df)} cubes; skipping", flush=True)
    else:
        if runner == "aml":
            from fsd.workflows import runners as _runners

            rk = dict(runner_kwargs or {})
            aml_root = rk.pop("root")
            run_id = rk.pop("run_id", None) or pd.Timestamp.now(tz="UTC").strftime(
                "%Y%m%dT%H%M%SZ"
            )
            blob_export = f"{aml_root.rstrip('/')}/runs/{run_id}/_flatten"

            _runners.run_aml_flatten(
                input_csv, blob_export, id_col=id_col, label_col=label_col,
                filepath_col=filepath_col, nodata=nodata, root=aml_root, run_id=run_id, **rk,
            )
            files = ["data.npy", "coords.npy", "ids.npy", "metadata.pickle.npy"]
            if label_col is not None:
                files.append("labels.npy")
            # force=True: this branch only runs when the flatten stamp did NOT match (a
            # genuine re-run), so stale local arrays from a prior, different identity must
            # be overwritten -- not mistaken for "already landed".
            _land_local(blob_export, export_folderpath, files, force=True)
        else:
            _flatten.flatten(
                filepaths_df=input_df, filepath_col=filepath_col, id_col=id_col,
                export_folderpath=export_folderpath, label_col=label_col, nodata=nodata,
            )

    data, metadata = _load_landed_arrays(export_folderpath)
    if not skip:
        # The stamp records which processing versions and sources the arrays were flattened
        # from (the reduce union'd them from each cube's metadata). Written after landing so
        # it can read them; a crash before this leaves no stamp, which only means
        # "re-flatten" (fail towards running).
        _stamp.write_stamp(stamp_filepath, identity, provenance=metadata.get("provenance"))

    feature_bands = metadata.get("feature_bands")
    if not skip and want_features:
        feature_bands = _apply_training_features(
            export_folderpath, metadata, adapter=adapter,
            feature_sequence=feature_sequence, aggregate=aggregate,
        )

    return TrainingData(
        export_folderpath=export_folderpath, run_folderpath=os.path.dirname(input_csv),
        n_pixels=int(data.shape[0]), n_timestamps=len(metadata["timestamps"]),
        bands=list(metadata["bands"]), feature_bands=feature_bands,
    )


def _land_local(blob_prefix: str, local_folder: str, files: list[str], *, force: bool = False) -> None:
    """Bring the compact flatten-reduce output home, one `storage.transfer` per file.

    `transfer` is single-object and atomic (`.part` + rename), so a failed copy never
    leaves a truncated `.npy` and this loop is safe to re-run.

    `force=False` (default): existence means already landed, so a retry after an interrupted
    transfer skips whatever arrived. `force=True` when the caller knows this is a genuine
    re-run -- stale local files from a PRIOR, different identity must be overwritten, not
    mistaken for "already landed".
    """
    fs.makedirs(local_folder)
    for name in files:
        dst = os.path.join(local_folder, name)
        if not force and fs.exists(dst):
            continue
        fs.transfer(os.path.join(blob_prefix, name), dst)


def _apply_training_features(export_folderpath, metadata, *, adapter, feature_sequence,
                             aggregate) -> list[str]:
    """Apply optional aggregation + the feature transform to flattened arrays.

    Writes `features.npy` (+ `feature_ids`/`feature_labels`) additively, records `feature_bands`
    + `aggregate` in metadata, and returns the feature band names. The raw `data.npy` is kept.
    """
    data = fs.load_npy(os.path.join(export_folderpath, "data.npy"))        # (pixels, T, B)
    ids = fs.load_npy(os.path.join(export_folderpath, "ids.npy"))
    labels_path = os.path.join(export_folderpath, "labels.npy")
    labels = fs.load_npy(labels_path) if fs.exists(labels_path) else None

    reducer = _resolve_aggregate(aggregate)
    if reducer is not None:
        ids, data, labels = reducer(ids, data.astype(float), labels)

    band_indices = {b: i for i, b in enumerate(metadata["bands"])}
    feats5d, feat_bi = _apply_features(
        _modify.expand_flattened(data.astype(float)), band_indices,
        adapter=adapter, feature_sequence=feature_sequence,
    )
    features = np.squeeze(feats5d, axis=(2, 3))                            # (pixels, T, Bf)
    feature_bands = [b for b, _ in sorted(feat_bi.items(), key=lambda kv: kv[1])]

    fs.save_npy(os.path.join(export_folderpath, "features.npy"), features)
    fs.save_npy(os.path.join(export_folderpath, "feature_ids.npy"), np.asarray(ids))
    if labels is not None:
        fs.save_npy(os.path.join(export_folderpath, "feature_labels.npy"), np.asarray(labels))

    agg_name = aggregate if isinstance(aggregate, str) else (
        getattr(aggregate, "__name__", "callable") if aggregate else None
    )
    metadata = dict(metadata)
    metadata["feature_bands"] = feature_bands
    metadata["aggregate"] = agg_name
    fs.save_npy(
        os.path.join(export_folderpath, "metadata.pickle.npy"), metadata, allow_pickle=True
    )
    return feature_bands


def _is_ref_shaped(model: str) -> bool:
    """Is this string shaped like a registry ref (`"name:N"` / `"name@alias"`)? A ref's two
    halves are a bare name and a bare version/alias, so **anything carrying a path separator is
    a path**, never a ref -- which is what keeps every URL out (`abfss://<fs>@<account>...`,
    which embeds "@" legitimately, `s3://`, `https://`) along with local paths that happen to
    contain "@" (`/data/rf@2026/`). A bare `rf@v1` in the cwd is genuinely indistinguishable
    from a ref and is treated as one; `./rf@v1` says "path, literally"."""
    if "/" in model or "\\" in model:
        return False
    try:
        registry_mod.parse_ref(model)
    except ValueError:
        return False
    return True


def _resolve_model_ref(model, registry, *, why):
    """`model` -> the same thing with any registry ref replaced by its version path.

    **Idempotent and shape-gated, deliberately.** Several call sites read `model` as a path
    (`_model_spec` reads `bundle.json` to preflight bands/T; `_ensure_bundle` hands a path to
    a subprocess), and they do not run in a fixed order. Being safe to call at every one of
    them is what removes the ordering constraint: a string that is not ref-shaped -- an
    already-resolved path included -- comes back untouched, so a new call site cannot
    resolve too late and fail preflight on a ref it never resolved.

    `registry=` absent + a `"name@alias"`-shaped string is a `PreflightError` naming the missing
    argument, never a silent fallback to treating it as a path. `":"` is never sniffed --
    it collides with URL schemes and Windows drive letters -- so a `"name:N"`-shaped path with
    no `registry=` passes through unchanged.

    Prints `[model] <ref> -> v<N> (verified against <env>)` here, once per verb call: this is
    the only branch where a ref actually resolves, and after it `model` is a version path,
    which is not ref-shaped, so a second call returns early above. The extra `_deploy.json`
    read is affordable only because this runs on the driver, once, before dispatch -- no
    runner or node calls it. It must never raise: a missing, empty or malformed
    `_deploy.json` degrades to the shorter line, never to a failed run.
    """
    if not isinstance(model, str):
        if registry is not None:
            raise PreflightError(
                f"{why}: registry={registry!r} was given but model is a live adapter, not a ref "
                "string ('name:version' or 'name@alias'). registry= only applies when model is a "
                "ref; pass the adapter without registry= to auto-save it, or pass a ref string."
            )
        return model
    if not _is_ref_shaped(model):
        return model
    if registry is not None:
        try:
            resolved = registry_mod.resolve(model, registry)
        except ValueError as exc:
            raise PreflightError(f"{why}: cannot resolve model={model!r} against "
                                 f"registry={registry!r} -- {exc}") from exc
        record = registry_mod.read_deploy_record(resolved.path)
        environment = (record or {}).get("environment")
        if environment:
            print(f"[model] {model} -> v{resolved.version} (verified against {environment})",
                  flush=True)
        else:
            print(f"[model] {model} -> v{resolved.version}", flush=True)
        return resolved.path
    if registry_mod.parse_ref(model)[1] == "@":
        raise PreflightError(
            f"{why}: model={model!r} looks like a registry ref ('name@alias') but no "
            "registry= was given. Pass registry=<path or URL> to resolve it, or write "
            f"the path with a directory component (e.g. './{model}') if that string is "
            "meant literally."
        )
    return model


def _model_spec(model) -> dict:
    """Read the declared spec (required_bands, n_timestamps, output_*) from a live adapter or,
    for a bundle path, from `bundle.json` alone (model-free — no import, no model load)."""
    if isinstance(model, str):
        return _bundle.read_spec(model)
    return {
        "required_bands": list(getattr(model, "required_bands", []) or []),
        "n_timestamps": int(getattr(model, "n_timestamps", 0) or 0),
        "output_dtype": getattr(model, "output_dtype", None),
        "output_nodata": getattr(model, "output_nodata", None),
        "output_band_names": list(getattr(model, "output_band_names", []) or []),
    }


def _resolve_inference_pairs(inference_datacubes, output_folderpath):
    """-> (pairs, geometries). `pairs` = [(datacube_filepath, output_filepath)]. Accepts an
    input.csv, a folder of datacube subfolders, or an explicit list of `datacube.npy` filepaths.

    `geometries` = `{output_filepath: shapefilepath}` when the source is an `input.csv`
    with a `shapefilepath` column (the manifest-driven STAC-geometry contract); `None` for the
    folder/list modes, which have no manifest to source a footprint from (STAC falls back to the
    raster bbox for those).
    """
    ids = None
    shapefilepaths = None
    if isinstance(inference_datacubes, (list, tuple)):
        dc_filepaths = [str(p) for p in inference_datacubes]
    elif isinstance(inference_datacubes, str) and inference_datacubes.endswith(".csv"):
        with fs.open(inference_datacubes, "r") as f:
            df = pd.read_csv(f)
        col = "datacube_filepath" if "datacube_filepath" in df.columns else df.columns[0]
        dc_filepaths = [str(p) for p in df[col]]
        if "id" in df.columns:
            ids = [str(i) for i in df["id"]]
        if "shapefilepath" in df.columns:
            shapefilepaths = [str(p) for p in df["shapefilepath"]]
    else:  # a folder: each subfolder holds a datacube.npy
        dc_filepaths = sorted(fs.glob(os.path.join(str(inference_datacubes), "*", "datacube.npy")))

    pairs = []
    geometries = {} if shapefilepaths is not None else None
    for i, dc in enumerate(dc_filepaths):
        stem = ids[i] if ids is not None else os.path.basename(os.path.dirname(dc))
        out = os.path.join(str(output_folderpath), stem, "output.tif")
        pairs.append((dc, out))
        if shapefilepaths is not None:
            geometries[out] = shapefilepaths[i]
    return pairs, geometries


def _merge_outputs(filepaths, dst, nodata, *, reproject_to_dominant: bool = False,
                   merge_crs=None) -> str:
    """Merge output COGs into one COG.

    `reproject_to_dominant=False` (``merge=True``) — **strict single-CRS** merge; multi-CRS
    **raises**, because the per-output COGs + STAC are the multi-zone answer. Data-faithful:
    no resampling.

    `reproject_to_dominant=True` (``merge="reproject"``) — reproject every output to one CRS
    with **nearest-neighbour**, since categorical output must not be interpolated, then
    mosaic. The target is ``merge_crs`` if given (EPSG int or CRS string), else the
    **max-total-area** CRS across cells — correct for clipped ROI-edge cells, where
    most-cells (the fallback) is not. **Lossless where a cell already matches the target**;
    reprojected only for cells changing zone. The per-cell COGs stay authoritative.
    """
    import tempfile
    import uuid as _uuid

    import rasterio

    from fsd.raster import rio_env

    # The outputs may live on `abfss://`, which GDAL has no driver for, so every READ goes
    # through the VSI seam. Bare `rasterio.open(fp)` here silently fails on blob-hosted
    # outputs -- the most-repeated mistake in this repo.
    #
    # `rio_env(filepaths)` + `rasterio.open(to_vsi(fp))`, NOT `rio_open` per file: `rio_open`
    # owns a `rasterio.Env` per handle, and merge holds every input open at once. rasterio's
    # env stack is LIFO, so closing 300 of them in creation order tears down the root env
    # first and the next close raises `EnvError: No GDAL environment exists`. One env for the
    # whole merge is both correct and cheaper -- one token fetch, not N. It is a null context
    # for local paths, so local merges are unchanged.
    with rio_env(filepaths):
        mosaic, out_transform, profile = _merge_mosaic(
            filepaths, nodata, reproject_to_dominant=reproject_to_dominant, merge_crs=merge_crs,
        )

    # The raw scratch tif must be LOCAL regardless of `dst`: a remote `dst` would otherwise
    # get "merged.tif.raw.tif" appended onto the same remote URL, which rasterio (local /
    # VSI-write only) cannot open. `to_cog` already knows how to publish a local raw file to
    # a remote `dst`; this only has to hand it a local source.
    if fs.is_local(dst):
        raw = f"{dst}.raw.tif"
    else:
        raw = os.path.join(tempfile.gettempdir(), f"fsd-merge-{_uuid.uuid4().hex}.raw.tif")
    try:
        with rasterio.open(raw, "w", **profile) as d:
            d.write(mosaic)
        _to_cog(raw, dst)
    finally:
        if os.path.exists(raw):
            os.remove(raw)
    return dst


def _merge_mosaic(filepaths, nodata, *, reproject_to_dominant: bool, merge_crs):
    """The mosaic itself. Must run inside a `rio_env(filepaths)` -- every `rasterio.open` below
    is on a VSI-translated path whose credentials live in that env."""
    import tempfile
    import uuid as _uuid

    import rasterio
    from rasterio.merge import merge as rio_merge

    from fsd.storage.azure import to_vsi

    # Tick per input, matching [setup]'s shape. The phases cost very different things, so
    # each gets its own bar rather than one that reaches 100% while the long leg is still
    # running: opening an input reads only its HEADER over /vsiadls/ (WAN latency, small),
    # while the pixels are read later inside `rio_merge` -- and, in reproject mode, inside
    # the per-input warp before it. `rio_merge` has no per-input hook to tick from, so it is
    # at least ANNOUNCED instead of being silence after a 100% line.
    tick = _progress.ticker(len(filepaths), "merge", unit="inputs")
    tick(0, force=True)

    if reproject_to_dominant:
        from rasterio.crs import CRS as _RioCRS
        from rasterio.warp import Resampling, calculate_default_transform
        from rasterio.warp import reproject as rio_reproject

        area_by_crs: dict[str, float] = {}
        for i, fp in enumerate(filepaths, 1):
            with rasterio.open(to_vsi(fp)) as s:
                key = s.crs.to_string()
                # extent area in the cell's own (metric UTM) CRS — comparable across UTM zones
                area_by_crs[key] = area_by_crs.get(key, 0.0) + (
                    abs(s.transform.a * s.transform.e) * s.width * s.height
                )
            tick(i)
        if merge_crs is not None:
            target = _RioCRS.from_user_input(merge_crs).to_string()   # user-forced target CRS
        elif any(area_by_crs.values()):
            target = max(area_by_crs, key=area_by_crs.get)            # dominant zone = max total area
        else:
            target = max(area_by_crs, key=lambda k: len(k))          # degenerate fallback

        datasets, tmps = [], []
        # The expensive per-input phase: each non-target-CRS input is fully decoded and
        # warped into local scratch. On 300 per-cell COGs over the WAN this dominates the
        # merge, and it starts right after the scan above has printed 100% -- without its
        # own ticker the run reads as finished-then-hung.
        rtick = _progress.ticker(len(filepaths), "merge", unit="inputs reprojected")
        rtick(0, force=True)
        try:
            for i, fp in enumerate(filepaths, 1):
                src = rasterio.open(to_vsi(fp))
                if src.crs.to_string() == target:
                    datasets.append(src)
                    rtick(i)
                    continue
                transform, w, h = calculate_default_transform(
                    src.crs, target, src.width, src.height, *src.bounds)
                prof = src.profile.copy()
                prof.update(driver="GTiff", crs=target, transform=transform,
                            width=w, height=h, nodata=nodata)
                # Local scratch, never `f"{fp}.reproj.tif"`: a remote `fp` would put the
                # temp on the remote URL, and rasterio cannot open a remote path for WRITE.
                # Same trap as the `raw` file below.
                tmp = os.path.join(tempfile.gettempdir(),
                                   f"fsd-reproj-{_uuid.uuid4().hex}.tif")
                tmps.append(tmp)
                with rasterio.open(tmp, "w", **prof) as d:
                    rio_reproject(
                        rasterio.band(src, 1), rasterio.band(d, 1),
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=transform, dst_crs=target,
                        src_nodata=nodata, dst_nodata=nodata,
                        resampling=Resampling.nearest,  # categorical-safe
                    )
                src.close()
                datasets.append(rasterio.open(tmp))  # local scratch -- bare open is right
                rtick(i)
            rtick(len(filepaths), force=True)
            print(f"[merge] merging {len(filepaths)} inputs into the mosaic "
                  "(reads pixels; no per-input progress)", flush=True)
            mosaic, out_transform = rio_merge(datasets, nodata=nodata)
            profile = datasets[0].profile.copy()
            profile.update(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
                           transform=out_transform, crs=target, nodata=nodata)
        finally:
            for d in datasets:
                d.close()
            for t in tmps:
                if os.path.exists(t):
                    os.remove(t)
    else:
        srcs = []
        for i, fp in enumerate(filepaths, 1):
            srcs.append(rasterio.open(to_vsi(fp)))
            tick(i)
        try:
            crs_set = {s.crs.to_string() for s in srcs}
            if len(crs_set) == 1:
                print(f"[merge] merging {len(filepaths)} inputs into the mosaic "
                      "(reads pixels; no per-input progress)", flush=True)
            if len(crs_set) > 1:
                raise PreflightError(
                    f"cannot merge outputs across multiple CRS {sorted(crs_set)}; pass "
                    'merge="reproject" for a display map (reprojects to the dominant zone, '
                    "lossy), or use the per-output COGs + STAC (fsd single-CRS-merge principle)."
                )
            mosaic, out_transform = rio_merge(srcs, nodata=nodata)
            profile = srcs[0].profile.copy()
            profile.update(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
                           transform=out_transform, nodata=nodata)
        finally:
            for s in srcs:
                s.close()

    tick(len(filepaths), force=True)
    return mosaic, out_transform, profile


def _finalize_outputs(output_filepaths, output_folderpath, spec, merge, collection_id, dt,
                      *, grids_filepath=None, merge_crs=None, geometries=None) -> InferenceResult:
    """Shared tail for both inference modes: STAC catalog + optional merge -> InferenceResult.

    `geometries`: `{output_filepath: geometry.geojson_path | shapely
    geometry}` — the true per-cell footprint, forwarded to `cog_outputs_to_items` in place of the
    raster bbox (see its docstring for the two accepted value shapes). `None` for geometry-less
    callers (see `_resolve_inference_pairs`).
    """
    items = _stac.cog_outputs_to_items(
        output_filepaths, geometries=geometries, collection_id=collection_id,
        band_names=spec.get("output_band_names") or None, dt=dt,
    )
    stac_catalog_filepath = _stac.write_stac_catalog(
        items, os.path.join(output_folderpath, "stac"),
        catalog_id="fsd-inference", collection_id=collection_id,
        description="fsd inference outputs (STAC).",
    )
    merged_filepath = None
    if merge:
        merged_filepath = _merge_outputs(
            output_filepaths, os.path.join(output_folderpath, "merged.tif"),
            nodata=spec.get("output_nodata"),
            reproject_to_dominant=(merge == "reproject"), merge_crs=merge_crs,
        )
    return InferenceResult(
        output_folderpath=output_folderpath, output_filepaths=sorted(output_filepaths),
        stac_catalog_filepath=stac_catalog_filepath, merged_filepath=merged_filepath,
        grids_filepath=grids_filepath,
    )


def run_inference(
    model,
    inference_datacubes=None,
    output_folderpath: str | None = None,
    *,
    # --- ROI mode — mutually exclusive with inference_datacubes -------------
    roi=None,
    catalog_filepath: str | None = None,
    startdate: datetime.datetime | None = None,
    enddate: datetime.datetime | None = None,
    mosaic_days: int | None = None,
    bands: list[str] | None = None,
    grid_size_km: float = 5,
    scale_fact: float = 1.1,
    collection: str = config.SATELLITE_S2L2A,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str | None = None,
    # --- shared ---
    predict_batch_size: int | None = None,
    skip_nan: bool = True,
    merge=False,
    merge_crs=None,
    cores: int | None = None,
    cubes_per_task: int | None = None,
    overwrite: bool = False,
    runner: str = "local",
    runner_kwargs: dict | None = None,
    storage=None,
    collection_id: str = "fsd-inference",
    dt=None,
    progress: bool = True,
    registry: str | None = None,
) -> InferenceResult:
    """Run a model over inference datacubes -> one COG per cube + a STAC catalog (+ optional merge).

    Two mutually-exclusive modes:

    - **pre-built cubes**: pass ``inference_datacubes`` — an ``input.csv``, a folder of
      datacube subfolders, or a list of ``datacube.npy`` filepaths. ``cores=1`` infers in-process
      (sequential); ``cores>1`` fans out via the Snakemake **infer-only** runner — fsd has no
      in-process pool; ``cubes_per_task`` groups cubes per job to amortise the bundle load.

    ``cores``/``cubes_per_task`` default to ``None`` = **auto**: for the local/pre-built paths that
    means ``1`` (sequential, one cube per job); for **ROI mode + ``runner="aml"``** it means
    *load-per-core* — the node picks ``cores = os.cpu_count()`` and groups cells so the bundle
    loads once per core rather than once per cell (TODO #25), computed on the node from the shard
    size (`workflows.infer_shard`). Pass ``cores=1`` there for the heavy-model
    *load-once-per-node* opt-out (one whole-shard group, one bundle load).
    - **ROI**: pass ``roi`` (+ ``catalog_filepath``,
      ``startdate``/``enddate``/``mosaic_days``/``bands``). fsd splits the ROI into S2 grid cells
      (``fsd.grid``), then fans out a per-cell **build-datacube + infer -> COG** task through the
      **runner seam** (Snakemake locally; Batch swaps in unchanged). Imagery is assumed
      already present in ``catalog_filepath`` — inference never touches CDSE (conserve quota).
      **``output_folderpath`` is the identity of the run**: re-running ROI mode into
      the same ``output_folderpath`` resumes the cached per-cell work list, so it must name this
      exact ``roi``/``grid_size_km``/``scale_fact`` — reusing it for a different roi raises
      ``PreflightError`` rather than silently mixing work lists.

    `model` is a live `ModelAdapter`, a **bundle path**, or a **registry ref** (``"name:N"`` /
    ``"name@alias"``) resolved against ``registry=`` — a ref given without
    ``registry=`` raises ``PreflightError`` naming the missing argument, never a silent fallback to
    treating it as a path. A bundle is required for ROI mode and for
    ``cores>1`` (both cross a subprocess) — a live adapter is auto-saved to a temp bundle. Preflight
    (before any build) asserts bands ⊇ ``required_bands`` and ``T == n_timestamps``. Inference is
    **idempotent**: existing outputs are skipped unless ``overwrite=True``. ``merge``:
    ``False`` | ``True`` (strict single-CRS) | ``"reproject"`` (cross-UTM-zone-safe merge to one
    CRS — ``merge_crs`` if given, else the max-total-area zone; lossless where a cell already
    matches the target). `runner`/`storage` are local-only for the pre-built-cubes path and for
    local ROI mode; **ROI mode + ``runner="aml"``** accepts ``storage="azure"``/an
    ``abfss://`` root and dispatches the per-cell build+infer task onto an Azure ML cluster
    instead (`runner_kwargs` carries `cluster=`/`environment=`/`root=`/`identity_client_id=`,
    see `workflows.runners.run_aml_inference`).
    """
    # Non-local storage is accepted for ROI mode + runner="aml" only; the pre-built-cubes
    # path and local ROI mode stay local.
    roi_mode = roi is not None
    # The seam check is raised on its own, BEFORE `_configure_storage`, for two reasons:
    # `_configure_storage` raises a bare `ValueError` for an unsupported backend, which
    # would replace this verb's collected `PreflightError` and discard every other error
    # with it; and it mutates process-global fsspec state, which a call that is about to be
    # rejected for using storage at all must not do.
    _raise_preflight(_check_local_seams(runner, storage,
                                        storage_allowed=(roi_mode and runner == "aml")))
    # ...and now, before the first storage access -- `_resolve_model_ref` below can resolve a
    # registry ref (a read against `registry=`), and ROI mode reads `roi` itself off storage,
    # both of which happen before this verb's own `_raise_preflight(errs)`.
    _configure_storage(storage)
    errs: list[str] = []
    if output_folderpath is None:
        errs.append("output_folderpath is required.")
    if merge not in (False, True, "reproject"):
        errs.append(f'merge must be False, True, or "reproject" (got {merge!r}).')
    # `dt` (the STAC output-item datetime) must reach `pystac.Item(datetime=...)` as a real
    # datetime object, so coerce a caller-supplied string at this boundary, not downstream.
    if dt is not None:
        try:
            dt = pd.to_datetime(dt, utc=True).to_pydatetime()
        except (ValueError, TypeError) as exc:
            errs.append(f"dt={dt!r} is not a valid date: {exc}")

    if roi_mode and inference_datacubes is not None:
        errs.append("pass either roi= or inference_datacubes=, not both.")
    if not roi_mode and inference_datacubes is None:
        errs.append("pass roi= (ROI mode) or inference_datacubes= (pre-built cubes).")

    # a ref must become a version path BEFORE `_model_spec` reads `bundle.json` off it.
    model = _resolve_model_ref(model, registry, why="run_inference")
    spec = _model_spec(model)

    if roi_mode:
        return _run_inference_roi(
            model, spec, roi, output_folderpath, errs,
            catalog_filepath=catalog_filepath, startdate=startdate, enddate=enddate,
            mosaic_days=mosaic_days, bands=bands, grid_size_km=grid_size_km,
            scale_fact=scale_fact, collection=collection, properties_filter=properties_filter,
            processing=processing,
            predict_batch_size=predict_batch_size, skip_nan=skip_nan, merge=merge,
            merge_crs=merge_crs, cores=cores, cubes_per_task=cubes_per_task, overwrite=overwrite,
            collection_id=collection_id, dt=dt, runner=runner, runner_kwargs=runner_kwargs,
            registry=registry,
        )

    # --- pre-built cubes path -------------
    required = set(spec.get("required_bands") or [])
    want_t = int(spec.get("n_timestamps") or 0)
    pairs, geometries = _resolve_inference_pairs(inference_datacubes, output_folderpath)
    if not pairs:
        errs.append(f"no inference datacubes found under {inference_datacubes!r}.")
    for dc_fp, _ in pairs:
        md_fp = os.path.join(os.path.dirname(dc_fp), "metadata.pickle.npy")
        if not fs.exists(dc_fp) or not fs.exists(md_fp):
            errs.append(f"missing datacube/metadata for {dc_fp}.")
            continue
        md = fs.load_npy(md_fp, allow_pickle=True)[()]
        missing = required - set(md["bands"])
        if missing:
            errs.append(f"{dc_fp}: datacube lacks required bands {sorted(missing)}.")
        if want_t and len(md["timestamps"]) != want_t:
            errs.append(
                f"{dc_fp}: datacube T={len(md['timestamps'])} but model needs T={want_t}."
            )
    _raise_preflight(errs)

    fs.makedirs(output_folderpath)
    # The pre-built-cubes path has no AML shape at all -- `cores==1` loads in-process here and
    # `cores>1` fans out to a LOCAL Snakemake runner -- so the driver always loads the bundle
    # on this machine and staging is unconditional. Done once per run, before either branch,
    # so `_ensure_bundle`'s own resolve (cores>1) sees an already-local path and no-ops.
    model = _stage_local_bundle(model, output_folderpath)
    # `None` = auto; the pre-built path has no node to interrogate, so auto == today's default (1).
    pb_cores = 1 if cores is None else cores
    pb_cubes_per_task = 1 if cubes_per_task is None else cubes_per_task
    if pb_cores > 1:
        # cores>1 fans out via the Snakemake infer-only runner -- there is no in-process pool
        output_filepaths = _run_prebuilt_via_runner(
            model, pairs, output_folderpath, cores=pb_cores, cubes_per_task=pb_cubes_per_task,
            overwrite=overwrite, predict_batch_size=predict_batch_size, skip_nan=skip_nan,
            registry=registry,
        )
    else:
        output_filepaths = _engine.run_local(
            model, pairs, predict_batch_size=predict_batch_size, skip_nan=skip_nan,
            overwrite=overwrite, progress=progress,
        )
    return _finalize_outputs(output_filepaths, output_folderpath, spec, merge, collection_id, dt,
                             merge_crs=merge_crs, geometries=geometries)


def _ensure_bundle(model, output_folderpath, *, why, registry=None):
    """Return a bundle path for `model`, auto-saving a live adapter (needs an importable class),
    resolving a registry ref (`"name:N"` / `"name@alias"`), or passing a bundle path through.

    Ref resolution is `_resolve_model_ref` -- called here so a direct caller of
    `_ensure_bundle` resolves too, and called again at `_model_spec`'s call sites, which read a
    bundle path FIRST; it is idempotent, so both are safe.
    """
    model = _resolve_model_ref(model, registry, why=why)
    if isinstance(model, str):
        return model
    try:
        return _bundle.save(
            model, getattr(model, "artifacts", {}) or {},
            os.path.join(output_folderpath, "_bundle"),
        )
    except Exception as exc:
        raise PreflightError(
            f"{why} needs a model bundle; auto-saving the live adapter failed ({exc}). Pass a "
            "bundle path (fsd.model.bundle.save) whose adapter class is importable by module:attr "
            "(not a __main__/interactive class)."
        ) from exc


def _stage_local_bundle(model, output_folderpath):
    """Fetch a remote bundle path to `<output_folderpath>/_model` and return the local copy.

    A bundle that is not on THIS machine cannot be `bundle.load`ed: `sys.path.insert`ing a
    URL is silently inert, because neither of CPython's two path hooks (`zipimporter`,
    `path_hook_for_FileFinder`) reads one (#89). The fetch is manifest-driven
    (`infer_shard.fetch_bundle_to_scratch`) so it needs no directory listing, and lands
    per-run rather than in a cache.

    A no-op for a path already local and for anything that is not a path at all (a live
    adapter, handled by `_ensure_bundle`'s own auto-save). Call only from a shape that
    actually loads the bundle on the driver -- this function knows "is this path local", not
    "who is about to load it", so the runner gate belongs at each call site.
    """
    if not isinstance(model, str) or fs.is_local(model):
        return model
    from fsd.workflows.infer_shard import fetch_bundle_to_scratch

    # Announce destination and total size BEFORE the transfer starts. `fetch_bundle_to_scratch`
    # is itself silent -- tolerable on an AML node, but this call is on the DRIVER, between
    # `[model] <ref> -> vN` and the first `[inference]` line, where the user is watching a
    # prompt that does nothing for as long as the download takes (13 MB has measured 627 s
    # over VPN). Costs one `bundle.json` read + one `size` per file.
    manifest = _bundle.read_spec(model)
    rels = list(manifest.get("artifacts", {}).values()) + _bundle.manifest_code_files(manifest)
    total_bytes = sum(fs.size(os.path.join(model, rel)) for rel in rels)
    print(f"[stage] bundle <- {model} | {len(rels)} files, {total_bytes / 1e6:.1f} MB",
          flush=True)
    return fetch_bundle_to_scratch(model, os.path.join(output_folderpath, "_model"))


def _run_prebuilt_via_runner(model, pairs, output_folderpath, *, cores, cubes_per_task,
                             overwrite, predict_batch_size, skip_nan, registry=None) -> list[str]:
    """Fan out pre-built-cube inference through the Snakemake infer-only runner."""
    from fsd.workflows import runners as _runners

    bundle_path = _ensure_bundle(model, output_folderpath, why="cores>1 inference", registry=registry)
    run_dir = os.path.join(output_folderpath, "_infer_run")
    fs.makedirs(run_dir)
    csv_fp = os.path.join(run_dir, "input.csv")
    df = pd.DataFrame({"datacube_filepath": [dc for dc, _ in pairs],
                       "output_filepath": [out for _, out in pairs]})
    with fs.open(csv_fp, "w") as f:
        df.to_csv(f, index=False)

    result = _runners.run_local_infer_only(
        csv_fp, cores=cores, bundle_path=bundle_path, cubes_per_task=cubes_per_task,
        overwrite=overwrite, predict_batch_size=predict_batch_size, skip_nan=skip_nan,
    )
    if result.returncode != 0:
        raise RuntimeError(f"inference runner failed (snakemake exit {result.returncode}).")
    return [out for _, out in pairs if fs.exists(out)]


def _output_key(path: str) -> str:
    """`…/<window>/<cell_id>/output.tif` -> `<window>/<cell_id>/output.tif`.

    The trailing components are the only scheme-independent part of the path, and that
    is the whole point: `fs.glob` returns the *filesystem's* path form (adlfs gives
    `container/path/…`, with no `abfss://` scheme), so a globbed hit never string-equals
    the url a caller built with `os.path.join`. Matching on the tail makes the two
    comparable under either form — and cell ids are unique within a run, so
    `<window>/<cell_id>/output.tif` is unique too.
    """
    return "/".join(str(path).rstrip("/").replace("\\", "/").split("/")[-3:])


def _existing_outputs(candidates, *, run_folderpath: str) -> list[str]:
    """Which of `candidates` (per-cell `output.tif` urls) exist — in ONE listing rather
    than one `fs.exists` per cell (TODO #61).

    Worth the indirection: at ~3 sequential blob round-trips per cell over VPN, the
    per-cell `fs.exists` comprehension this replaces cost **729 s of a 2067 s wall**.
    Order follows `candidates`, as that comprehension did.

    A pattern that matches nothing yields an empty list, and the caller already raises
    "no per-cell outputs were produced" on that — a loud failure, not a silent one.

    ⚠️ The `*/*` depth is a **contract with `create_datacube.setup`**, which builds
    `export_folderpath = run_folderpath/<window>/<id>`. The middle `*` is
    the `<window>` segment (`create_datacube.window_folder_segment`). It is a wildcard
    because the candidates are `input.csv`'s `export_folderpath` values: a resume reuses a
    cached `input.csv` without re-running `setup` (`_run_inference_roi`;
    `_check_resume_identity` compares cell ids only), so a candidate's `<window>` can
    differ from what `window_folder_segment` computes today, e.g. an `input.csv` written
    before the window key last changed. `_output_key` keeps `<window>` in the match, so
    outputs of another window in the same run folder are never counted. A change to the
    `run_folderpath/<window>/<id>` depth must change this pattern too.
    """
    # One `fs.glob` round trip, so there is no per-candidate loop left to tick against:
    # print before/after in the same `[label] done/total | elapsed` shape instead. `done`
    # counts candidates PROBED, not outputs found -- the whole leg completes in one glob, so
    # reporting hits as progress makes a finished collect read as "stuck at 0%" on a fresh
    # run. Hits ride along as the suffix. No eta: there is no intermediate rate to
    # extrapolate from.
    tick = _progress.ticker(len(candidates), "collect", unit="candidates",
                            show_rate=False, show_eta=False)
    tick(0, force=True)
    hits = {
        _output_key(h)
        for h in fs.glob(os.path.join(str(run_folderpath), "*", "*", "output.tif"))
    }
    found = [c for c in candidates if _output_key(c) in hits]
    tick(len(candidates), force=True,
         suffix=f"{len(found)} already have an output.tif")
    return found


def _imagery_missing_message(roi, startdate, enddate, bands, *, catalog_filepath, why) -> str:
    """Turn "no imagery for this request" into an actionable `fsd.download(...)` plan.

    Degrades gracefully if the STAC query itself fails — it still says, clearly, to run
    download first."""
    base = (f"imagery for this ROI/window is not present in the catalog "
            f"({catalog_filepath!r}) — run fsd.download first, then re-run. [{why}]")
    try:
        from fsd.sources import cdse as _cdse

        plan = _cdse.plan_download(
            roi, startdate, enddate, bands, catalog_filepath=catalog_filepath,
        )
        return base + "\n" + _cdse.format_download_plan(plan)
    except Exception:  # noqa: BLE001 - the plan is a nicety; never mask the real "run download"
        return base


def _run_inference_roi(
    model, spec, roi, output_folderpath, errs, *,
    catalog_filepath, startdate, enddate, mosaic_days, bands,
    grid_size_km, scale_fact, collection, properties_filter=None, processing=None,
    predict_batch_size, skip_nan, merge, merge_crs, cores, cubes_per_task, overwrite,
    collection_id, dt, runner="local", runner_kwargs=None, registry=None,
) -> InferenceResult:
    """ROI mode: preflight -> grid cells -> per-cell setup -> runner build+infer -> STAC/merge."""
    from fsd import grid as _grid
    from fsd.workflows import runners as _runners

    # --- preflight (cheap, before any build) ---
    for name, val in [("catalog_filepath", catalog_filepath), ("startdate", startdate),
                      ("enddate", enddate), ("mosaic_days", mosaic_days), ("bands", bands)]:
        if val is None:
            errs.append(f"roi mode requires {name}=.")
    errs += _processing.processing_errors(processing, collection=collection, allow_none=True)
    if catalog_filepath is not None:
        errs += _check_catalog_collection_dir(catalog_filepath, collection)
    # Normalize dates HERE, before `compute_n_timestamps` or any dispatch: an unparseable
    # date must abort on the driver in milliseconds, not after a 40-380 s node cold-start.
    if startdate is not None and enddate is not None:
        startdate, enddate, date_errs = _normalize_window(startdate, enddate)
        errs += date_errs
    # Canonicalize bands and the adapter's required_bands to native asset keys (spec 58
    # D8) before comparing -- one may be spelled canonically ("nir08"), the other
    # natively ("B8A"), and both must resolve to the same set for this check to mean
    # anything.
    _declaration = _collections.get(collection)
    if bands is not None:
        bands = [_declaration.canonical_to_native(b) for b in bands]
    required = {_declaration.canonical_to_native(b) for b in (spec.get("required_bands") or [])}
    want_t = int(spec.get("n_timestamps") or 0)
    if bands is not None:
        missing = required - set(bands)
        if missing:
            errs.append(f"bands is missing model-required {sorted(missing)}.")
    if want_t and None not in (startdate, enddate, mosaic_days):
        got_t = compute_n_timestamps(startdate, enddate, mosaic_days)
        if got_t != want_t:
            errs.append(
                f"startdate/enddate/mosaic_days give T={got_t} but the model needs T={want_t}."
            )
    roi_gdf = None
    try:
        if isinstance(roi, gpd.GeoDataFrame):
            roi_gdf = roi
        elif isinstance(roi, str):
            roi_gdf = fs.read_geo(roi)   # storage seam, not gpd.read_file (TODO #47)
    except Exception as exc:  # noqa: BLE001 - surfaced as a preflight error
        errs.append(f"could not read roi: {exc}.")
    if roi_gdf is not None and len(roi_gdf) == 0:
        errs.append("roi is empty.")

    # Split the ROI into S2 grid cells INSIDE preflight. Splitting is local, CPU-only work, so
    # doing it here rather than after `_ensure_bundle` means a bad ROI costs seconds instead
    # of a blob `makedirs`, a bundle upload (627 s for 13 MB over VPN), setup's N per-cell
    # blob writes, and an AML dispatch.
    #
    # What it catches: an ROI that yields no grid cells, and DUPLICATE CELL IDS. `id` is the
    # work-unit key -- `setup` derives `export_folderpath` from it -- so duplicates put N
    # tasks on one folder, which on blob is a concurrent same-blob write (`InvalidBlockList`)
    # and locally a silent overwrite. `roi_to_s2_grids` prevents them at source; this is the
    # seatbelt, and it fires before any spend.
    grids = None
    if not errs:
        try:
            grids = _grid.roi_to_s2_grids(
                roi_gdf if roi_gdf is not None else roi,
                grid_size_km=grid_size_km, scale_fact=scale_fact,
            )
        except ImportError:
            raise  # the [grid] extra is missing -- that message is the clearer one
        except Exception as exc:  # noqa: BLE001 - surfaced as a preflight error
            errs.append(f"could not tile roi into grid cells: {exc}.")
    if grids is not None:
        if len(grids) == 0:
            errs.append(
                f"roi tiled into 0 grid cells at grid_size_km={grid_size_km} -- the roi is "
                f"smaller than one cell, or its geometry is degenerate."
            )
        elif grids["id"].duplicated().any():
            counts = grids["id"].value_counts()
            repeated = counts[counts > 1]
            errs.append(
                f"roi tiled into {len(grids)} rows but only {grids['id'].nunique()} distinct "
                f"cell ids ({len(repeated)} repeated, worst {int(counts.iloc[0])}x). One cell "
                f"must be one work-unit; duplicates would make several tasks write the same "
                f"output folder concurrently. Is `roi=` a REGION? Passing a per-feature file "
                f"(a label/field set) tiles per (cell x feature) pair -- see spec 21 D-GRID-1."
            )
    _raise_preflight(errs)

    # Cell count IS the cluster workload and the cost -- say it before spending anything.
    print(f"[run_inference] roi -> {len(grids)} grid cells at grid_size_km={grid_size_km} "
          f"(one build+infer task each)", flush=True)

    # Check the cached work list BEFORE anything writes into output_folderpath (#66). Run it
    # any later -- after `grids.geojson` is overwritten, after a bundle is staged -- and a
    # refused resume leaves the run folder describing THIS roi while its `cells/input.csv`
    # still describes the previous one. A refusal must leave the folder as it found it.
    run_folderpath = os.path.join(output_folderpath, "cells")
    csv_filepath = os.path.join(run_folderpath, "input.csv")
    if fs.exists(csv_filepath):
        _check_resume_identity(csv_filepath, grids, output_folderpath)

    fs.makedirs(output_folderpath)

    # Runner-gated: `runner="aml"` never loads the bundle on the driver -- it stages to blob
    # and each node fetches its own copy -- so staging here would be a wasted
    # blob->local->blob round trip. `runner="local"` (both `cores=1` in-process and the
    # `cores>1` Snakemake fan-out) does load here.
    if runner == "local":
        model = _stage_local_bundle(model, output_folderpath)

    # model must be a bundle (it crosses a subprocess); auto-save a live adapter.
    bundle_path = _ensure_bundle(model, output_folderpath, why="roi mode", registry=registry)

    grids_filepath = os.path.join(output_folderpath, "grids.geojson")
    # GDAL/pyogrio has no abfss:// write driver, so a blob output_folderpath makes
    # `grids.to_file(...)` fail with "Failed to create GeoJSON datasource". Stage via the
    # storage seam instead -- mirrors `create_training_data`'s gdf staging and the seam READ
    # in `create_datacube.setup`. `to_json(default=str)` guards any non-JSON-native column;
    # grids carry only id + geometry, but the guard is cheap and the failure is not.
    with fs.open(grids_filepath, "w") as f:
        f.write(grids.to_json(default=str))

    # Per-cell setup (reuse the build workflow's setup; no labels). Skip if input.csv exists
    # so a re-run resumes -- but ONLY if that cached work list still corresponds to THIS
    # request: resume-by-existence alone lets a stale input.csv from a different roi silently
    # win (#66). That identity check ran above, before this function wrote anything.
    if not fs.exists(csv_filepath):
        try:
            _create_datacube.setup(
                catalog_filepath=catalog_filepath, timestamp_col="timestamp",
                shapefilepath=grids_filepath, id_col="id", run_folderpath=run_folderpath,
                startdate=startdate, enddate=enddate, bands=bands,
                mosaic_days=mosaic_days,
                csv_filepath=csv_filepath, label_col=None, collection=collection,
                properties_filter=properties_filter, processing=processing,
            )
        except ValueError as exc:
            raise PreflightError(_imagery_missing_message(
                roi_gdf if roi_gdf is not None else roi, startdate, enddate, bands,
                catalog_filepath=catalog_filepath, why=str(exc),
            )) from exc

    # Fan out the per-cell build+infer task via the runner seam. This is the ONLY step that
    # swaps -- tiling, setup and collect are runner-agnostic.
    if runner == "aml":
        # `cores`/`cubes_per_task` pass through as-is: `None` lets the AML node compute the
        # load-per-core default from its own core count + the shard size (`infer_shard`).
        _runners.run_aml_inference(
            csv_filepath, bundle_path, cubes_per_task=cubes_per_task, cores=cores,
            predict_batch_size=predict_batch_size, skip_nan=skip_nan, overwrite=overwrite,
            **(runner_kwargs or {}),
        )
    else:
        # local has no node to interrogate, so auto (`None`) means 1.
        result = _runners.run_local_inference(
            csv_filepath, cores=(1 if cores is None else cores), bundle_path=bundle_path,
            cubes_per_task=(1 if cubes_per_task is None else cubes_per_task),
            predict_batch_size=predict_batch_size, skip_nan=skip_nan, overwrite=overwrite,
        )
        if result.returncode != 0:
            raise RuntimeError(f"inference runner failed (snakemake exit {result.returncode}).")

    # Collect the per-cell outputs, plus each cell's true footprint for STAC geometry.
    with fs.open(csv_filepath, "r") as f:
        rows = pd.read_csv(f)
    # Pass the footprint already held in memory (`grids`, tiled by THIS driver minutes ago)
    # rather than the geometry.geojson path `create_datacube.setup` wrote it to -- that would
    # be a sequential blob read per cell, fetching back geometry the driver itself authored.
    # `.buffer(0)` mirrors `create_datacube.setup`'s own `_prepare`, so the Item this produces
    # is byte-identical to the path form's. Keyed by `id` and matched to each row's own
    # `export_folderpath`, so the pairing is correct by construction rather than by a
    # re-derived id.
    id_to_geometry = {str(gid): geom.buffer(0) for gid, geom in zip(grids["id"], grids.geometry)}
    geometries = {
        os.path.join(str(exp), "output.tif"): id_to_geometry[str(cell_id)]
        for exp, cell_id in zip(rows["export_folderpath"], rows["id"])
    }
    output_filepaths = _existing_outputs(list(geometries), run_folderpath=run_folderpath)
    if not output_filepaths:
        raise RuntimeError("no per-cell outputs were produced.")

    return _finalize_outputs(
        output_filepaths, output_folderpath, spec, merge, collection_id, dt,
        grids_filepath=grids_filepath, merge_crs=merge_crs, geometries=geometries,
    )


def _cell_coverage(grids: gpd.GeoDataFrame, catalog_gdf, startdate, enddate) -> dict[str, int]:
    """In-window catalog coverage per grid cell: the count of catalog rows `filter_gdf`
    matches to that cell alone.

    Used to pick `verify_adapter`'s deterministic default cell -- the one most likely to
    build a full cube, so a failure reads as the adapter's rather than an empty cell's.
    """
    coverage: dict[str, int] = {}
    for _, row in grids.iterrows():
        cell_gdf = gpd.GeoDataFrame({"geometry": [row.geometry]}, crs=grids.crs)
        subset = _filter_gdf(catalog_gdf, cell_gdf, startdate, enddate)
        coverage[str(row["id"])] = int(subset.shape[0])
    return coverage


_VERIFY_ADAPTER_RESULT_NAME = "_result.json"


def _finish_verify_adapter(export_folderpath: str, result: dict) -> dict:
    """Write the verdict to `_result.json` and return it.

    The verdict has to be a FILE a user or a run-book can paste back, not only a return value
    that dies with the process. Every exit from `verify_adapter` goes through here, so a
    failure verdict lands on disk exactly as a passing one does.
    """
    fs.write_text(
        os.path.join(export_folderpath, _VERIFY_ADAPTER_RESULT_NAME),
        json.dumps(result, indent=2, sort_keys=True, default=str),
    )
    if not result["pass"]:
        print(f"[verify_adapter] FAIL -- {result['error']} "
              f"See {os.path.join(export_folderpath, _VERIFY_ADAPTER_RESULT_NAME)}.",
              flush=True)
    return result


def verify_adapter(
    model,
    *,
    roi,
    catalog_filepath: str,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    mosaic_days: int,
    bands: list[str],
    export_folderpath: str,
    cell: str | None = None,
    grid_size_km: float = 5,
    scale_fact: float = 1.1,
    collection: str = config.SATELLITE_S2L2A,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str | None = None,
    predict_batch_size: int | None = None,
    skip_nan: bool = True,
    runner: str = "local",
    runner_kwargs: dict | None = None,
    storage=None,
    registry: str | None = None,
) -> dict:
    """One real grid cell's datacube, built on `runner`, landed locally, run through the
    adapter's ACTUAL inference code -- so `output.tif` can be eyeballed in QGIS before
    trusting a bundle for a many-cell fan-out.

    Where this sits in the workflow. It is NOT a substitute for either neighbour, and
    answers a different question from both:

    1. **`verify_adapter`** (here) -- is my adapter's LOGIC right? Local, minutes, iterate
       here. Checks the cube shape/`T` against the adapter's declared `n_timestamps`, the
       post-`feature_sequence` band set against `required_bands`, and whether `predict`
       returns the declared `output_dtype`/value range on REAL pixels (real SCL masking,
       real nodata, real interpolation gaps) -- none of which `adapter_smoke`'s import-only
       check can see. Says NOTHING about the image (`fsd.model.verify_image`'s job) and
       NOTHING about scale (one cell is not the fan-out) or any cell but the one it ran.
    2. `fsd.model.verify_image` -- will the IMAGE run it? One AML node, ~40-380s.
    3. `fsd.run_inference` -- the fan-out, N nodes.

    `model` is a live adapter, a bundle path, or a registry ref (``"name:N"`` / ``"name@alias"``)
    resolved against ``registry=`` -- a ref given without ``registry=`` raises
    ``PreflightError`` naming the missing argument. A live adapter is auto-saved by
    `_ensure_bundle` (the same call `run_inference` makes) to `export_folderpath/_bundle`
    -- a real, persistent bundle, **not** a temp one: `code=None` auto-detects and embeds
    the adapter's local module, so it is cluster-loadable, and its path comes back as
    `metrics["bundle_path"]`.

    It is still not a substitute for your own `bundle.save`. `_ensure_bundle` passes no
    `requirements=`, and those are what `verify_image` checks the image against at gate 2
    -- which this verb never reaches, since its inference leg runs in-process on the
    driver. Bundle deliberately for anything you intend to deploy; this one exists so
    gate 1 can run at all, and it lives inside a verify output folder that is safe to
    delete.

    `cell=`: an explicit grid-cell id uses it (`PreflightError`, naming the available ids
    bounded, if it is not in this roi); `None` (default) picks DETERMINISTICALLY -- largest
    in-window catalog coverage, tie-broken by id -- and prints which cell and why, so two
    runs over the same roi/window pick the same cell; `"random"` opts in to a random pick
    and prints the chosen id so a run worth keeping can be pinned by pasting that id back as
    `cell=`. `grids.geojson` (every cell in the roi) is ALWAYS written next to the
    output and its path printed -- open it in QGIS, pick an id, re-run with `cell=`.

    `export_folderpath` is where everything lands, LOCALLY, no hidden cache: the cube
    (`datacube.npy` + `metadata.pickle.npy`), `output.tif`, `grids.geojson`, `cell.geojson`,
    `_result.json`. No flattened/feature array is written -- the adapter is for inference,
    inference output is the grid cell's raster, and `output.tif` is what gets checked. A
    second call with the SAME roi/window/bands/mosaic_days/cell and the cube already landed
    skips the build+land entirely and goes straight to inference; the resume keys on the
    REQUEST's identity (`fsd.workflows.stamp`), never on file age. A call whose
    `export_folderpath` already holds a cube for a DIFFERENT request raises rather than
    silently reusing it.

    `runner="local"` (default) builds from a local catalog end-to-end, no network -- what the
    test suite uses. `runner="aml"` (the case that matters in practice) builds the ONE
    cell's datacube through the same per-cell unit of work `create_training_data` fans out
    (`fsd.workflows.create_datacube.run_create_datacube`), under
    `runner_kwargs["root"]/runs/<run_id>/_verify_adapter` on blob -- required, because the
    node has to be able to write the cube and a local build folder would name a driver path
    it cannot reach. The cube is then transferred DOWN into the local
    `export_folderpath`. No new build path, and the
    inference leg is a one-row call into `fsd.workflows.infer_only_task.run_infer_only`
    -- the SAME unit the cluster runs, so no branch anywhere may special-case this verb.

    Returns a `_result.json`-shaped dict: `{"step", "status", "pass", "metrics",
    "expected", "error"}`. `metrics` carries the cube shape, cube `T` vs the adapter's
    declared `n_timestamps`, the band set after `feature_sequence` vs `required_bands`,
    output dtype vs `output_dtype`, output value range, nodata fraction, and the cube/COG/
    grids paths. A `T` mismatch is reported as `pass: False` (never raised) -- only CALLER
    misuse (a bad `cell=`, an unreadable roi, a mismatched export_folderpath) raises
    `PreflightError`; every statement ABOUT the adapter comes back in the dict. This run
    says nothing about the image, nothing about scale, and nothing about any cell but the
    one it ran.
    """
    from fsd import grid as _grid

    startdate, enddate, date_errs = _normalize_window(startdate, enddate)
    # Seam errors raise on their own, before `_configure_storage` -- see the same comment in
    # `run_inference` for why. Not batched with the date errors: a seam misconfiguration is
    # fatal by itself.
    _raise_preflight(_check_local_seams(runner, storage, storage_allowed=(runner == "aml")))
    # ...and now, before the first storage access -- `_resolve_model_ref` and `fs.read_geo`
    # below both touch storage, ahead of this verb's own `_raise_preflight(errs)`.
    _configure_storage(storage)
    errs = list(date_errs)
    errs += _processing.processing_errors(processing, collection=collection, allow_none=True)
    errs += _check_catalog_collection_dir(catalog_filepath, collection)
    if not date_errs:
        errs += _check_window(startdate, enddate, mosaic_days, bands)
    if not export_folderpath:
        errs.append("export_folderpath is required.")
    if cell is not None and not isinstance(cell, str):
        errs.append(f"cell must be a string id, 'random', or None (got {cell!r}).")
    if runner == "aml" and not (runner_kwargs or {}).get("root"):
        errs.append("runner_kwargs['root'] (the blob working root) is required for runner='aml'.")

    # as in `run_inference`: resolve before `_model_spec` reads `bundle.json` off it.
    model = _resolve_model_ref(model, registry, why="verify_adapter")
    spec = _model_spec(model)
    # Canonicalize (spec 58 D8) before comparing -- see the identical comment in
    # `_run_inference_roi`.
    _declaration = _collections.get(collection)
    bands = [_declaration.canonical_to_native(b) for b in bands]
    required = {_declaration.canonical_to_native(b) for b in (spec.get("required_bands") or [])}
    missing = required - set(bands)
    if missing:
        errs.append(f"bands is missing model-required {sorted(missing)}.")
    want_t = int(spec.get("n_timestamps") or 0)

    roi_gdf = None
    try:
        if isinstance(roi, gpd.GeoDataFrame):
            roi_gdf = roi
        elif isinstance(roi, str):
            roi_gdf = fs.read_geo(roi)   # storage seam, not gpd.read_file (TODO #47)
    except Exception as exc:  # noqa: BLE001 - surfaced as a preflight error
        errs.append(f"could not read roi: {exc}.")
    if roi_gdf is not None and len(roi_gdf) == 0:
        errs.append("roi is empty.")

    grids = None
    if not errs:
        try:
            grids = _grid.roi_to_s2_grids(
                roi_gdf if roi_gdf is not None else roi,
                grid_size_km=grid_size_km, scale_fact=scale_fact,
            )
        except ImportError:
            raise  # the [grid] extra is missing -- that message is the clearer one
        except Exception as exc:  # noqa: BLE001 - surfaced as a preflight error
            errs.append(f"could not tile roi into grid cells: {exc}.")
    if grids is not None and len(grids) == 0:
        errs.append(
            f"roi tiled into 0 grid cells at grid_size_km={grid_size_km} -- the roi is "
            f"smaller than one cell, or its geometry is degenerate."
        )
    _raise_preflight(errs)

    fs.makedirs(export_folderpath)
    grids_filepath = os.path.join(export_folderpath, "grids.geojson")
    # GDAL/pyogrio has no abfss:// write driver -- stage via the storage seam (mirrors
    # `_run_inference_roi`'s own grids.geojson write).
    with fs.open(grids_filepath, "w") as f:
        f.write(grids.to_json(default=str))
    print(f"[verify_adapter] roi -> {len(grids)} grid cells; wrote {grids_filepath}",
          flush=True)

    available = sorted(grids["id"].astype(str))
    if cell == "random":
        import random as _random

        chosen_cell = _random.choice(available)
        print(f'[verify_adapter] cell="random" picked {chosen_cell!r} -- pass '
              f"cell={chosen_cell!r} to reproduce this run.", flush=True)
    elif cell is not None:
        chosen_cell = str(cell)
        if chosen_cell not in available:
            shown = available[:50]
            more = f" (+{len(available) - 50} more)" if len(available) > 50 else ""
            raise PreflightError(
                f"cell={chosen_cell!r} is not one of this roi's {len(available)} grid "
                f"cells. Available ids: {shown}{more}."
            )
    else:
        catalog_gdf = TileCatalog(catalog_filepath).read()
        coverage = _cell_coverage(grids, catalog_gdf, startdate, enddate)
        # largest in-window catalog coverage, tie-broken by (smallest) id.
        chosen_cell = min(available, key=lambda i: (-coverage.get(i, 0), i))
        print(f"[verify_adapter] cell={chosen_cell!r} picked deterministically (largest "
              f"in-window catalog coverage: {coverage.get(chosen_cell, 0)} intersecting "
              f"tiles).", flush=True)

    # --- resume by the REQUEST's identity, never mere existence -----------------------
    identity = {
        "roi": roi if isinstance(roi, str) else roi_gdf.to_json(default=str),
        "startdate": str(startdate), "enddate": str(enddate), "mosaic_days": int(mosaic_days),
        "bands": sorted(bands), "collection": collection,
        "properties_filter": _create_datacube._canonicalize_properties_filter(properties_filter),
        "grid_size_km": grid_size_km, "scale_fact": scale_fact, "cell": chosen_cell,
    }
    if processing is not None:
        # Only when used, so a stamp written before spec 59 still matches a request that
        # does not use it.
        identity["processing"] = _create_datacube._canonicalize_processing(processing)
    cube_filepath = os.path.join(export_folderpath, "datacube.npy")
    metadata_filepath = os.path.join(export_folderpath, "metadata.pickle.npy")
    stamp_filepath = os.path.join(export_folderpath, "_cube_stamp.json")

    existing_stamp = _stamp.read_stamp(stamp_filepath)
    if existing_stamp is not None:
        existing_identity = {k: v for k, v in existing_stamp.items() if not k.startswith("_")}
        if existing_identity != identity:
            raise PreflightError(
                f"export_folderpath={export_folderpath!r} already holds a cube for a "
                f"DIFFERENT request (its stamp does not match this roi/window/bands/"
                f"mosaic_days/cell) -- resuming it would silently reuse stale work (spec "
                f"47 D1's precedent: a dry run's whole purpose is to be trusted). Use a "
                f"new export_folderpath, or remove the existing one."
            )
    cube_ready = existing_stamp is not None and _artifacts_present(
        export_folderpath, ["datacube.npy", "metadata.pickle.npy"]
    )

    if cube_ready:
        print(f"[verify_adapter] cube for cell={chosen_cell!r} already landed at "
              f"{export_folderpath} and matches this request; skipping the build.",
              flush=True)
    else:
        # One-row shapefile for JUST the chosen cell -> the SAME per-cell unit of work
        # `create_training_data` fans out, through the SAME runner seam. No new build path.
        cell_row = grids.loc[grids["id"].astype(str) == chosen_cell]
        cell_filepath = os.path.join(export_folderpath, "cell.geojson")
        with fs.open(cell_filepath, "w") as f:
            f.write(cell_row.to_json(default=str))

        # The cube is built by the existing per-cell unit of work, so the BUILD has to live
        # somewhere that unit's WORKER can write. For runner="aml" that is the blob working
        # root: `create_datacube.setup` turns a local `run_folderpath` into an ABSOLUTE
        # DRIVER path and writes it into `input.csv`, so a local build folder would send the
        # node off to write the cube at a path that does not exist on it. `export_folderpath`
        # stays local throughout -- it is where the cube is LANDED, never where it is built.
        if runner == "aml":
            rk = runner_kwargs or {}
            build_run_id = rk.get("run_id") or pd.Timestamp.now(tz="UTC").strftime(
                "%Y%m%dT%H%M%SZ"
            )
            build_folderpath = (
                f"{str(rk['root']).rstrip('/')}/runs/{build_run_id}/_verify_adapter"
            )
        else:
            build_folderpath = os.path.join(export_folderpath, "_build")
        build_csv_filepath = os.path.join(build_folderpath, "input.csv")
        try:
            _create_datacube.run_create_datacube(
                catalog_filepath=catalog_filepath, timestamp_col="timestamp",
                shapefilepath=cell_filepath, id_col="id", run_folderpath=build_folderpath,
                startdate=startdate, enddate=enddate, bands=bands,
                mosaic_days=mosaic_days,
                csv_filepath=build_csv_filepath, label_col=None, cores=1,
                runner=runner, runner_kwargs=runner_kwargs, collection=collection,
                properties_filter=properties_filter, processing=processing,
            )
        except ValueError as exc:
            raise PreflightError(_imagery_missing_message(
                cell_row, startdate, enddate, bands,
                catalog_filepath=catalog_filepath, why=str(exc),
            )) from exc

        with fs.open(build_csv_filepath, "r") as f:
            build_row = pd.read_csv(f).iloc[0]
        # Landing is `storage.transfer`, exactly as `create_training_data` lands its compact
        # array -- the local cube becomes a first-class artifact.
        #
        # force=True: reaching this branch MEANS the local cube is not trusted (no stamp, or
        # the artifacts were absent/empty). With force=False the land would skip a file that
        # merely EXISTS -- say a cube left behind after the caller deleted `_cube_stamp.json`
        # to get past the "different request" refusal above -- and the `write_stamp` below
        # would then record THIS request's identity over the previous request's pixels.
        _land_local(
            os.path.dirname(str(build_row["datacube_filepath"])), export_folderpath,
            ["datacube.npy", "metadata.pickle.npy"], force=True,
        )
        _stamp.write_stamp(stamp_filepath, identity)

    # --- the inference leg IS infer_only_task.run_infer_only, unmodified ---------------
    bundle_path = _ensure_bundle(model, export_folderpath, why="verify_adapter", registry=registry)
    output_filepath = os.path.join(export_folderpath, "output.tif")
    infer_csv_filepath = os.path.join(export_folderpath, "_infer_input.csv")
    with fs.open(infer_csv_filepath, "w") as f:
        pd.DataFrame({
            "datacube_filepath": [cube_filepath], "output_filepath": [output_filepath],
        }).to_csv(f, index=False)

    written = _infer_only_task.run_infer_only(
        infer_csv_filepath, (0, 1), bundle_path,
        predict_batch_size=predict_batch_size, skip_nan=skip_nan, overwrite=True,
    )

    # --- build the verdict --------------------------------------------------------------
    metadata = fs.load_npy(metadata_filepath, allow_pickle=True)[()]
    cube_t = len(metadata["timestamps"])
    cube = fs.load_npy(cube_filepath)
    band_indices = {b: i for i, b in enumerate(metadata["bands"])}

    # The post-feature_sequence band set needs the LIVE adapter (a bundle manifest carries
    # no feature_sequence, only its declared spec) -- load it once more to introspect.
    adapter_obj = _bundle.load(bundle_path)
    feature_sequence = getattr(adapter_obj, "feature_sequence", None)
    if feature_sequence:
        _, feat_bi = _apply_features(
            np.zeros((1, cube_t, 1, 1, len(band_indices)), dtype="float32"), band_indices,
            feature_sequence=feature_sequence,
        )
        post_feature_bands = [b for b, _ in sorted(feat_bi.items(), key=lambda kv: kv[1])]
    else:
        post_feature_bands = list(metadata["bands"])

    result: dict = {
        "step": "verify_adapter",
        "status": "ok",
        "pass": False,
        "metrics": {
            "cell": chosen_cell,
            "cube_filepath": cube_filepath,
            "cube_shape": list(cube.shape),
            "cube_bands": list(metadata["bands"]),
            "cube_t": cube_t,
            "adapter_n_timestamps": want_t,
            "post_feature_sequence_bands": post_feature_bands,
            "required_bands": sorted(required),
            "output_filepath": output_filepath if written else None,
            "grids_filepath": grids_filepath,
            # The bundle gate 1 actually ran, so reusing it is a supported move rather
            # than a guess at the folder layout. Note it carries no `requirements=`
            # (`_ensure_bundle`), which gate 2 wants -- see this verb's docstring.
            "bundle_path": bundle_path,
        },
        "expected": {"n_timestamps": want_t, "required_bands": sorted(required)},
        "error": None,
    }

    if want_t and cube_t != want_t:
        result["status"] = "fail"
        result["error"] = f"cube T={cube_t} but adapter n_timestamps={want_t}."
        return _finish_verify_adapter(export_folderpath, result)

    if not written:
        result["status"] = "fail"
        result["error"] = f"no output written for cell={chosen_cell!r}."
        return _finish_verify_adapter(export_folderpath, result)

    import rasterio

    with rasterio.open(output_filepath) as src:
        arr = src.read()
        output_nodata = src.nodata
    result["metrics"].update({
        "output_dtype": str(arr.dtype),
        "output_value_min": float(np.nanmin(arr)) if arr.size else None,
        "output_value_max": float(np.nanmax(arr)) if arr.size else None,
        "output_nodata_fraction": (
            float((arr == output_nodata).sum()) / arr.size
            if output_nodata is not None and arr.size else None
        ),
    })
    output_dtype = spec.get("output_dtype")
    if output_dtype and str(arr.dtype) != str(output_dtype):
        result["status"] = "fail"
        result["error"] = (
            f"output dtype={arr.dtype} but adapter declares output_dtype={output_dtype!r}."
        )
        return _finish_verify_adapter(export_folderpath, result)

    result["pass"] = True
    print(f"[verify_adapter] pass -- open {output_filepath} and {grids_filepath} in QGIS. "
          f"This checks the ADAPTER only: nothing about the image (fsd.model.verify_image) "
          f"and nothing about scale (fsd.run_inference; one cell is not the fan-out).",
          flush=True)
    return _finish_verify_adapter(export_folderpath, result)


def _load_verified_result(verified) -> dict:
    """`verified=` is either the dict `verify_image` returned or a path/URL to the
    `_result.json` it was written to. A path that is not there, or is not JSON, is caller
    error and gets the verb's own `PreflightError` rather than a bare `FileNotFoundError`
    from the storage seam."""
    if isinstance(verified, dict):
        return verified
    try:
        with fs.open(str(verified), "r") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError) as exc:
        raise PreflightError(
            f"fsd.deploy(verified={verified!r}): cannot read that as a verify_image "
            f"_result.json -- {exc}. Pass the dict verify_image returned, the path it was "
            "written to, or verified=None to let deploy run the verification itself."
        ) from exc


def _verified_matches(result: dict, *, digest: str, environment: str) -> bool:
    """Does `result` (a `_result.json`-shaped `verify_image` output) actually speak to THIS
    deploy call -- same `environment`, and the bundle content it RECORDS having verified
    (`metrics["bundle_digest"]`, written by `verify_image` at verification time) is the
    content being deployed?

    It must be that recorded digest, never a re-digest of `metrics["bundle_path"]`: the path
    says where the bundle was, not what it held, and `bundle.save` overwrites in place.
    Verify `./demo_bundle`, retrain, re-save over it, then deploy it with the old
    `_result.json`, and a path re-digest compares the new content with itself -- a tautology
    that passes, registering "this image ran this bundle" for content the image never saw.
    Matching on the recorded digest also makes a `_result.json` portable: a colleague's result
    stays honourable on a machine where their `bundle_path` does not exist.

    A result carrying no `bundle_digest` is a mismatch, not a pass -- it cannot say what it
    verified.

    This says nothing about `pass`: a matched-but-failing result is still honoured (it flows
    on to the `pass=False` refusal below, with its own `error`). What gets refused HERE is a
    result that does not even speak to this bundle/environment pair -- refused, never
    ignored, so this must not fall back to running `verify_image` itself. The caller does
    that only when `verified=` was never given."""
    metrics = result.get("metrics") or {}
    if metrics.get("environment") != environment:
        return False
    return bool(metrics.get("bundle_digest")) and metrics["bundle_digest"] == digest


def deploy(
    bundle_path: str,
    *,
    name: str,
    registry: str,
    environment: str,
    runner: str = "aml",
    runner_kwargs: dict | None = None,
    alias: str | None = None,
    verified: str | dict | None = None,
    storage=None,
) -> str:
    """Publish a SAVED bundle into `registry` under `name`, refusing unless `environment` has
    been proven to run it. Returns a ref `run_inference(model=ref, registry=registry)`
    accepts unchanged: `"<name>:<version>"`.

    `deploy` establishes the bundle<->image pairing before it records it, one of two ways:

    * the default -- runs `fsd.model.verify_image(bundle_path, environment=environment,
      runner=runner, runner_kwargs=runner_kwargs)` itself, one real AML node; or
    * `verified=<_result.json path or dict>` -- accepts a PRIOR verification, honoured only if
      the bundle content it RECORDS having verified (`metrics["bundle_digest"]`) and its
      `environment` both match what is being deployed here. A stale or mismatched result is
      refused outright, never silently re-verified.

    Either way, `pass=False` refuses the deployment and raises with that result's own `error`.
    No version directory is created on refusal.

    `bundle_path` must be a SAVED bundle -- a live adapter is refused, naming
    `fsd.model.bundle.save`, because publishing something another machine will fetch by name is
    a deliberate act, distinct from `_ensure_bundle`'s auto-save convenience for running
    something now. The bundle's manifest must declare `requirements` and carry a `code` block;
    missing either is refused, naming the `bundle.save(...)` fix.

    `alias` (e.g. `"champion"`) publishes and repoints the alias in one call -- promotion
    is an alias reassignment, never a state transition, and multiple aliases may point at one
    version.

    Idempotent: deploying identical bundle content again returns the same version and writes
    nothing new (`fsd.model.registry.publish`) -- safe to re-run from a notebook cell.

    `storage=` reaches the same local/Azure "compute seam" gate `run_inference` and
    `verify_adapter` use, and `storage="azure"` is accepted so a blob registry can
    authenticate. It is unrelated to `registry=`, which is resolved/written directly through
    the storage seam and may itself be a URL.
    """
    errs = _check_local_seams(runner, storage)
    _raise_preflight(errs)
    # Before the first storage access -- `_bundle.read_spec` below reads `bundle_path`, and
    # `publish` writes into `registry` further down.
    _configure_storage(storage)

    if not isinstance(bundle_path, str):
        raise PreflightError(
            "fsd.deploy(...) needs a SAVED bundle path, not a live adapter (spec 51 D6) -- "
            "call fsd.model.bundle.save(adapter, artifacts, dst, requirements=[...]) first, "
            "then pass the returned path to deploy. (Auto-saving is _ensure_bundle's job, a "
            "convenience for running something now -- publishing something others will fetch "
            "by name is a deliberate act.)"
        )

    # Checked HERE, not left to `publish` at the end: a name that cannot round-trip through a
    # ref would otherwise surface only after a real verify_image node has been paid for.
    try:
        registry_mod.check_name(name)
    except ValueError as exc:
        raise PreflightError(f"fsd.deploy(name=...): {exc}") from exc

    manifest = _bundle.read_spec(bundle_path)
    if not manifest.get("requirements"):
        raise PreflightError(
            f"deploy refuses {bundle_path!r}: its manifest declares no `requirements`, so no "
            "other machine can install what it needs (spec 51 D6). Fix: re-save it with "
            "bundle.save(..., requirements=[...])."
        )
    if not _bundle.manifest_code_files(manifest):
        raise PreflightError(
            f"deploy refuses {bundle_path!r}: its manifest has no `code` block, so it does not "
            "carry its adapter -- a generic image will fail with ModuleNotFoundError (spec 51 "
            "D6). Fix: re-save it with bundle.save(..., code=...) (or the default auto-detect) "
            "so the adapter's source is embedded."
        )

    digest = registry_mod.content_digest(bundle_path)

    if verified is not None:
        result = _load_verified_result(verified)
        if not _verified_matches(result, digest=digest, environment=environment):
            _vm = result.get("metrics") or {}
            raise PreflightError(
                f"deploy(verified=...) is stale or does not match this call: it verified "
                f"bundle_digest={_vm.get('bundle_digest')!r} against "
                f"environment={_vm.get('environment')!r}, but this call deploys {digest!r} "
                f"against {environment!r} (spec 51 D5) -- a mismatched verification is "
                "refused, not ignored. A result with bundle_digest=None predates spec 51 and "
                "cannot say what it verified: re-run verify_image, or pass verified=None to "
                "let deploy run it."
            )
    else:
        result = _verify_image(
            bundle_path, environment=environment, runner=runner, runner_kwargs=runner_kwargs,
        )

    if not result.get("pass"):
        # `verify_image` splits its diagnosis in two: the top-level `error` is populated only
        # for a DRIVER-detected failure (bad code, stale wheel, no status file). When the smoke
        # job itself ran and reported a failure, `error` stays None and the diagnosis is in
        # `metrics["smoke_error"]` -- read both, or the refusal says "error: None".
        why = result.get("error") or (result.get("metrics") or {}).get("smoke_error")
        raise PreflightError(
            f"deploy refuses to register {name!r}: {environment!r} has not been proven to run "
            f"this bundle (spec 51 D5). verify_image says: {why}"
        )

    version = registry_mod.publish(bundle_path, name, registry, alias=alias)

    import fsd as _fsd  # lazy: fsd/__init__.py imports this module, so a top-level import cycles

    record = {
        "name": name,
        "version": version,
        "digest": digest,
        "environment": environment,
        "verified": result,
        "deployed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "fsd_version": _fsd.__version__,
    }
    registry_mod.write_deploy_record(name, version, record, registry)

    return f"{name}:{version}"
