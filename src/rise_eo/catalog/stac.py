"""STAC export view over the granule catalog.

Spec: specs/17-stac-catalog.md

The `TileCatalog` GeoParquet stays the working/query format; this module is an **additive**
STAC interchange view: one STAC Item per catalog row, one asset per band file. The mapping is
pure-metadata (no raster reads) by default — `proj:epsg` is derived from the MGRS tile in the
product id; per-asset `proj:shape`/`proj:transform` are opt-in (`read_proj=True`).

Serialization is a static, self-contained STAC catalog (JSON) via `pystac`, written through the
`rise_eo.storage` seam so a blob/S3 destination works later unchanged. The stac-geoparquet export
lives in `rise_eo.catalog.stac_geoparquet`.

`write_stac_catalog` takes Items from either builder here: `tile_catalog_to_items` (one Item per
granule) or `cog_outputs_to_items` (one Item per inference-output COG, for `run_inference`).
"""

from __future__ import annotations

import json
import os
import re

import pandas as pd
import pystac
import shapely.geometry
from pystac.extensions.classification import Classification, ClassificationExtension
from pystac.extensions.eo import EOExtension
from pystac.extensions.projection import ProjectionExtension
from pystac.extensions.raster import RasterBand, RasterExtension
from pystac.stac_io import DefaultStacIO

from rise_eo import progress as _progress
from rise_eo.catalog import declaration as declaration_module
from rise_eo.catalog.declaration import CollectionDeclaration
from rise_eo.storage import fs

# A fixed default, deliberately NOT `cores`: `cores` means the node's inference parallelism
# everywhere else in rise-eo, so overloading it here would mislead. Tuned for latency-bound
# round-trips -- a blob header read or a small blob write, ~1 s each over VPN -- not for core
# count. No public knob until a real run asks for one, because an unused knob is a knob that
# gets set wrong; the [collect]/[stac] segment lines are what a future retune reads.
_COLLECT_THREADS = 16

# STAC extension URIs we populate beyond eo/proj (added via their helper classes).
_GRID_EXT = "https://stac-extensions.github.io/grid/v1.1.0/schema.json"
_PROCESSING_EXT = "https://stac-extensions.github.io/processing/v1.2.0/schema.json"

# MGRS tile in an S2 product id, e.g. "..._T37PBP_..." -> zone=37, band=P, square=BP.
_MGRS_RE = re.compile(r"_T(\d{2})([C-X])([A-Z]{2})_")

_SOURCE_LINK_REL = "via"  # the source .SAFE product this row was downloaded from


class _StorageStacIO(DefaultStacIO):
    """Route pystac's JSON read/write through `rise_eo.storage` (local now, blob/S3 later)."""

    def read_text(self, source, *args, **kwargs) -> str:
        with fs.open(str(source), "r") as f:
            return f.read()

    def write_text(self, dest, txt, *args, **kwargs) -> None:
        parent = os.path.dirname(str(dest))
        if parent:
            fs.makedirs(parent)
        with fs.open(str(dest), "w") as f:
            f.write(txt)


class _ThreadedStorageStacIO(_StorageStacIO):
    """Defer `Catalog.save()`'s per-object blob writes to a thread pool.

    `Catalog.save()` walks the tree and calls `save_json` -> `write_text` once per object
    (catalog, collection, one per Item) -- for a 300-cell run that is 301 independent small
    blob writes, sequentially, at ~0.53 s each.

    The tree walk and JSON encoding stay on the driver thread (fast, pure Python, no I/O);
    only the blob write itself is handed to the pool, so the catalog **layout** -- link
    order, one Item per object -- is untouched. This changes only how the bytes reach
    storage.

    `write_text` submits and returns immediately; `wait()` (called once, after `catalog.save()`
    returns) blocks until every submitted write has finished and re-raises the first exception any
    of them hit -- a worker failure surfaces as that failure, never a silently-incomplete
    catalog. The `ThreadPoolExecutor` this is built with is itself a `with` block in the
    caller, so even an early exception out of `wait()` lets every in-flight write finish
    before it propagates (same shape as `workflows.create_datacube.setup`'s threaded loop).
    """

    def __init__(self, executor, tick):
        super().__init__()
        self._executor = executor
        self._tick = tick
        self._futures: list = []

    def write_text(self, dest, txt, *args, **kwargs) -> None:
        bound_write = super().write_text
        self._futures.append(self._executor.submit(bound_write, dest, txt, *args, **kwargs))

    def wait(self) -> None:
        import concurrent.futures

        done = 0
        for fut in concurrent.futures.as_completed(self._futures):
            fut.result()  # first exception, if any, raises here
            done += 1
            self._tick(done)


# --- mapping helpers ---------------------------------------------------------

def _parse_mgrs(item_id: str) -> tuple[str, int] | None:
    """Return (mgrs_tile, epsg) from an S2 product id, or None if not parseable.

    UTM EPSG from the latitude band: bands C..M are southern (327xx), N..X northern (326xx).
    """
    m = _MGRS_RE.search(item_id)
    if not m:
        return None
    zone, band, square = m.group(1), m.group(2), m.group(3)
    north = band >= "N"
    epsg = int(f"{'326' if north else '327'}{int(zone):02d}")
    return f"{zone}{band}{square}", epsg


def _band_role(band: str, declaration: CollectionDeclaration) -> str:
    """Spec 34 §2a / spec 58 D5 role classification for a band name: `"mask"` (the
    declared mask band, e.g. SCL/QA), `"reference"` (the declared resample-reference
    band), else `"reflectance"` for a band the declaration says carries radiometry
    (`is_radiometry_band`), else the generic `"data"` (e.g. AOT/WVP/visual — present
    in a granule's `files` but not consumed by the builder's radiometry/mask/reference
    logic)."""
    if declaration.mask_spec is not None and band == declaration.mask_spec.band:
        return "mask"
    if band == declaration.reference_band:
        return "reference"
    if declaration.is_radiometry_band(band):
        return "reflectance"
    return "data"


def _media_type_and_roles(
    filename: str, band: str | None, declaration: CollectionDeclaration,
) -> tuple[str | None, list[str]]:
    lower = filename.lower()
    role = [_band_role(band, declaration)] if band else []
    # The role classification rides ALONGSIDE "data" -- never replaces it, and never
    # duplicates it when `_band_role` already returned "data".
    data_roles = ["data", *role] if role and role[0] != "data" else ["data"]
    if lower.endswith((".tif", ".tiff")):
        return pystac.MediaType.COG, data_roles
    if lower.endswith(".jp2"):
        return pystac.MediaType.JPEG2000, data_roles
    if lower.endswith(".xml"):
        return pystac.MediaType.XML, ["metadata"]
    return None, ["data"]


def _asset_href(local_folderpath: str, filename: str) -> str:
    # Local runs: the recorded file path. (Blob/S3 hrefs arrive with the storage seam, P1.)
    return os.path.join(str(local_folderpath), filename)


def _read_proj_fields(href: str) -> dict:
    """Open a raster to read per-asset proj:shape / proj:transform (opt-in; I/O)."""
    from rise_eo.raster import rio_open

    with rio_open(href) as src:
        return {"shape": [src.height, src.width], "transform": list(src.transform)[:6]}


# --- granule catalog -> STAC items -------------------------------------------

def _set_processing(item: pystac.Item, row) -> None:
    """Set the spec 59 `processing:*` properties (and extension URI) from a row's processing columns."""
    # Spec 59 D8: the catalog's processing columns are the STAC processing extension's
    # `processing:version` / `processing:datetime` (UTC), when the provider exposed them.
    version = row.get("processing_version")
    if version is not None and not pd.isna(version):
        item.properties["processing:version"] = str(version)
    pdt = row.get("processing_datetime")
    if pdt is not None and not pd.isna(pdt):
        item.properties["processing:datetime"] = (
            pd.Timestamp(pdt).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"))
    if "processing:version" in item.properties or "processing:datetime" in item.properties:
        if _PROCESSING_EXT not in item.stac_extensions:
            item.stac_extensions.append(_PROCESSING_EXT)


def _set_grid(item: pystac.Item, item_id: str) -> None:
    """Set `proj:epsg` and `grid:code` from the MGRS tile in an S2 product id, if it has one."""
    mgrs = _parse_mgrs(item_id)
    if mgrs is not None:
        tile, epsg = mgrs
        ProjectionExtension.ext(item, add_if_missing=True).epsg = epsg
        if _GRID_EXT not in item.stac_extensions:
            item.stac_extensions.append(_GRID_EXT)
        item.properties["grid:code"] = f"MGRS-{tile}"


def _add_band_asset(
    item: pystac.Item, filename: str, row, declaration: CollectionDeclaration, read_proj: bool,
    *, nodata, offset, scale,
) -> None:
    """Add one band file of a granule row to `item` as an asset."""
    band = filename.rsplit(".", 1)[0]
    href = _asset_href(row["local_folderpath"], filename)
    media_type, roles = _media_type_and_roles(filename, band, declaration)
    asset = pystac.Asset(href=href, media_type=media_type, roles=roles, title=band)
    item.add_asset(band, asset)  # sets asset.owner=item, required before ext(add_if_missing=True)
    if "data" in roles:
        asset.extra_fields["eo:bands"] = [{"name": band}]
        if read_proj and media_type != pystac.MediaType.XML:
            asset.extra_fields.update(
                {f"proj:{k}": v for k, v in _read_proj_fields(href).items()}
            )
        if media_type in (pystac.MediaType.COG, pystac.MediaType.JPEG2000):
            is_reflectance = _band_role(band, declaration) in ("reflectance", "reference")
            RasterExtension.ext(asset, add_if_missing=True).bands = [
                RasterBand.create(
                    nodata=nodata,
                    # raster:bands offset is reflectance-unit to match the declared
                    # scale — the catalog row's `offset` is DN-unit, so scale
                    # it; `items_to_rows` divides it back out. Keeps unscale=true
                    # unit-consistent (DN*scale + offset == (DN+offset_dn)/10000).
                    offset=offset * scale if is_reflectance else 0,
                    scale=scale if is_reflectance else 1,
                )
            ]


def _granule_item(
    row, *, collection_id, read_proj: bool, declaration: CollectionDeclaration,
) -> pystac.Item:
    """Build the STAC Item for one `TileCatalog` row."""
    geom = row["geometry"]
    dt = row["timestamp"].to_pydatetime()
    coll = collection_id if collection_id is not None else row["collection"]
    row_offset = row.get("offset", 0) or 0
    row_scale = row.get("scale", declaration.scale)
    row_scale = declaration.scale if not row_scale else row_scale
    row_nodata = row.get("nodata", 0)
    row_nodata = 0 if row_nodata is None else row_nodata

    item = pystac.Item(
        id=str(row["id"]),
        geometry=shapely.geometry.mapping(geom),
        bbox=list(geom.bounds),
        datetime=dt,
        properties={},
        collection=coll,
    )

    EOExtension.ext(item, add_if_missing=True).cloud_cover = float(row["cloud_cover"])
    _set_processing(item, row)
    _set_grid(item, str(row["id"]))

    files = [f for f in str(row["files"]).split(",") if f]
    for filename in files:
        _add_band_asset(
            item, filename, row, declaration, read_proj,
            nodata=row_nodata, offset=row_offset, scale=row_scale,
        )

    if row.get("s3url"):
        item.add_link(pystac.Link(rel=_SOURCE_LINK_REL, target=str(row["s3url"])))

    return item


def tile_catalog_to_items(
    gdf, *, collection_id=None, read_proj=False, declaration: CollectionDeclaration | None = None,
) -> list[pystac.Item]:
    """Map `TileCatalog` rows (a GeoDataFrame from `.read()`) to STAC Items.

    One Item per row (one granule); one asset per band file in `files`.
    Pure-metadata unless `read_proj=True` (which opens each raster for proj:shape/transform).

    `declaration` resolves the mask/reference/radiometry-band facts used for asset role
    classification and `raster:bands` (spec 58 D5/D14) -- the explicit kwarg, else `gdf`'s
    own stamp (`rise_eo.catalog.declaration.from_attrs`), else the S2 L2A default for a
    hand-built `gdf` (mirrors `builder._resolve_declaration`, without the file-source raise:
    this is metadata export, not a build).

    Spec 34 §1a/§2a: every raster asset (a `.tif`/`.jp2` band file) gets a `raster:bands`
    entry with the row's declared `offset`/`nodata`/`scale` (`.get(...)`, defaulting to the
    resolved declaration's values for a catalog row without them) for a band the declaration
    says carries radiometry (`scale=1`/`offset=0` for mask/other bands — a no-op). This is
    the *interchange* declaration the builder/other tools read; the load-bearing one for a
    viewer is the COG's own GDAL tag (stamped at ingest,
    `rise_eo.raster.cog.stamp_gdal_tags`) — the two are written with identical values so there
    is no drift (§1a "no double-application").
    """
    if declaration is None:
        declaration = declaration_module.from_attrs(gdf) or declaration_module.S2_L2A_DECLARATION
    return [
        _granule_item(
            row, collection_id=collection_id, read_proj=read_proj, declaration=declaration,
        )
        for _, row in gdf.iterrows()
    ]


def _output_item_id(fp) -> str:
    """STAC item id for an inference-output COG.

    rise-eo writes every output as ``<cube_id>/output.tif`` (the per-cell/-cube folder *is* the id),
    so the id is the **parent directory** name — not the constant ``output`` filename stem, which
    would collide across all outputs. Falls back to the filename stem when there is no parent dir.
    """
    return (os.path.basename(os.path.dirname(str(fp)))
            or os.path.splitext(os.path.basename(str(fp)))[0])


def _read_footprint_geometry(geom_path):
    """Read the polygon + `properties.id` from a `geometry.geojson` (CRS84, one Feature),
    through the `rise_eo.storage` seam. Returns `(None, None)` if the FeatureCollection is empty."""
    with fs.open(str(geom_path), "r") as f:
        fc = json.load(f)
    features = fc.get("features") or []
    if not features:
        return None, None
    feat = features[0]
    geom = shapely.geometry.shape(feat["geometry"])
    feat_id = (feat.get("properties") or {}).get("id")
    return geom, feat_id


def _output_footprint(fp, geometries):
    """The Item footprint geometry for output COG `fp` from its `geometries` entry, or raise."""
    geom_or_path = geometries.get(str(fp), geometries.get(fp))
    if geom_or_path is None:
        raise ValueError(
            f"cog_outputs_to_items: geometries has no entry for output COG {fp!r}; "
            "the manifest-driven contract requires every output to have a footprint "
            "(pass geometries=None to fall back to the raster bbox for ALL outputs)."
        )
    if isinstance(geom_or_path, shapely.geometry.base.BaseGeometry):
        geom = geom_or_path
        if geom.is_empty:
            raise ValueError(
                f"cog_outputs_to_items: geometries entry for output COG {fp!r} is an "
                "empty geometry."
            )
    else:
        geom_path = geom_or_path
        try:
            geom, feat_id = _read_footprint_geometry(geom_path)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"cog_outputs_to_items: could not read geometry {geom_path!r} for "
                f"output COG {fp!r}: {exc}"
            ) from exc
        if geom is None or geom.is_empty:
            raise ValueError(
                f"cog_outputs_to_items: geometry.geojson at {geom_path!r} (for "
                f"{fp!r}) has no readable polygon feature."
            )
        item_id = _output_item_id(fp)
        if feat_id is not None and str(feat_id) != item_id:
            raise ValueError(
                f"cog_outputs_to_items: geometry.geojson id {feat_id!r} at "
                f"{geom_path!r} disagrees with output item id {item_id!r} for {fp!r}."
            )
    return geom


def _output_item(fp, epsg, shape, transform, geom, bbox, *, dt, collection_id, band_names):
    """Build the STAC Item for one inference-output COG from its `_describe` ingredients."""
    item = pystac.Item(
        id=_output_item_id(fp),
        geometry=shapely.geometry.mapping(geom),
        bbox=bbox,
        datetime=dt,
        properties={},
        collection=collection_id,
    )
    if epsg is not None:
        ProjectionExtension.ext(item, add_if_missing=True).epsg = epsg
    item.properties["proj:shape"] = shape
    item.properties["proj:transform"] = transform

    asset = pystac.Asset(
        href=str(fp), media_type=pystac.MediaType.COG, roles=["data"], title="output"
    )
    if band_names:
        asset.extra_fields["eo:bands"] = [{"name": b} for b in band_names]
    item.add_asset("output", asset)
    return item


def _check_unique_ids(items) -> None:
    """Raise if two output Items share an id."""
    # STAC item ids must be unique: write_stac_catalog's normalize_hrefs maps id -> <id>/<id>.json,
    # so a collision silently overwrites all-but-one item on disk AND duplicates the collection link
    # (the spec-26 bug). rise-eo writes every output as <cube_id>/output.tif, so ids come from the parent
    # dir and are unique by construction; guard loudly if a layout ever breaks that invariant.
    ids = [it.id for it in items]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(
            f"cog_outputs_to_items: item ids are not unique ({dupes}); each output must live in "
            "its own per-cube folder (<cube_id>/output.tif)."
        )


def cog_outputs_to_items(cog_filepaths, *, geometries=None, collection_id="rise-eo-inference",
                         band_names=None, dt=None) -> list[pystac.Item]:
    """Map inference-output COGs to STAC Items; used by `run_inference`.

    One Item per output COG. `proj:*` is read straight from the COG we just wrote (cheap, no
    ambiguity). `dt` is the Item datetime for all outputs (defaults to now, UTC) — outputs are
    mosaics over a window, not a single acquisition.

    `geometries`: an optional `{output_cog_filepath: value}` mapping — the
    **true S2-cell footprint**, used as the Item geometry/bbox instead of the raster bbox. `value`
    is **either**:

    - a `geometry.geojson` **path** (str), read through the `rise_eo.storage` seam — today's
      behaviour. Carries a cross-check: the file's `properties.id` must equal the item id derived
      from the output path, else raises (two independent sources that can disagree).
    - an **already-loaded** `shapely` geometry, used as-is with **no round-trip read**. Checked
      **structurally** rather than by id: the caller looked this geometry up *by* the item id
      when building the mapping, so a mismatch cannot be constructed the way it can for the path
      form. Only ROI mode, which holds `grids` in memory, uses this form.

    This is a **deterministic, manifest-driven contract, not a per-item fallback**: when
    `geometries` is given, every `fp` in `cog_filepaths` must have an entry — a missing/unreadable/
    empty one raises (a manifest that lists an output but no footprint is a real inconsistency;
    fail loud, don't silently box). `geometries=None` (the default) keeps the raster-bbox
    behavior, for geometry-less callers (unit tests; a bare list of COGs; pre-built folder/list
    inference modes with no manifest).

    The COG opens are threaded (`_COLLECT_THREADS` workers): each is I/O-bound and
    independent, and `rio_open` already owns its own `rasterio.Env` per call
    (`raster/__init__.py`) — a fresh, credentialed env per worker thread, which matters
    because rasterio's env stack is thread-local and an env entered on one thread is
    invisible to another. Item order in the returned list always
    matches `cog_filepaths`, regardless of completion order; a failure in one worker surfaces as
    that failure via `fut.result()`, not as a silently short list.
    """
    import concurrent.futures
    import datetime as _datetime

    from rasterio.warp import transform_bounds

    from rise_eo.raster import rio_open

    if dt is None:
        dt = _datetime.datetime.now(_datetime.timezone.utc)

    cog_filepaths = list(cog_filepaths)

    def _describe(fp):
        """One output COG -> its Item ingredients. Pure per-fp work (a read, never a write),
        which is what makes threading it safe."""
        with rio_open(fp) as src:
            epsg = src.crs.to_epsg() if src.crs else None
            shape = [src.height, src.width]
            transform = list(src.transform)[:6]

            if geometries is not None:
                geom = _output_footprint(fp, geometries)
                bbox = list(geom.bounds)
            else:
                bounds4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
                geom = shapely.geometry.box(*bounds4326)
                bbox = list(bounds4326)

        return epsg, shape, transform, geom, bbox

    # Results are placed BY INDEX and reassembled after, so `items` below is always in
    # `cog_filepaths` order regardless of completion order (same pattern as
    # `workflows.create_datacube.setup`'s threaded loop).
    described: list[tuple | None] = [None] * len(cog_filepaths)
    tick = _progress.ticker(len(cog_filepaths), "collect", unit="outputs")
    tick(0, force=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_COLLECT_THREADS) as pool:
        futures = {pool.submit(_describe, fp): i for i, fp in enumerate(cog_filepaths)}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            described[futures[fut]] = fut.result()
            done += 1
            tick(done)
    tick(len(cog_filepaths), force=True)

    items = [
        _output_item(
            fp, epsg, shape, transform, geom, bbox,
            dt=dt, collection_id=collection_id, band_names=band_names,
        )
        for fp, (epsg, shape, transform, geom, bbox) in zip(cog_filepaths, described)
    ]
    _check_unique_ids(items)
    return items


def cog_outputs_to_items_from_manifest(input_csv_filepath, **kwargs) -> list[pystac.Item]:
    """Convenience wrapper: build `geometries` from a `run_inference` build manifest
    (`input.csv`, columns `export_folderpath, shapefilepath, …`) and call `cog_outputs_to_items`.

    Only COGs that exist on disk are included (a manifest row's build may not have produced an
    output, e.g. a partial/resumed run). `kwargs` forwards to `cog_outputs_to_items`
    (`collection_id`, `band_names`, `dt`).
    """
    with fs.open(str(input_csv_filepath), "r") as f:
        rows = pd.read_csv(f)
    geometries = {
        os.path.join(str(exp), "output.tif"): str(sp)
        for exp, sp in zip(rows["export_folderpath"], rows["shapefilepath"])
    }
    cogs = [cog for cog in geometries if fs.exists(cog)]
    return cog_outputs_to_items(cogs, geometries=geometries, **kwargs)


def items_to_rows(items: list[pystac.Item]):
    """Inverse mapping — reconstruct the `TileCatalog` columns from Items (round-trip check)."""
    import geopandas as gpd

    rows = []
    for item in items:
        hrefs = [a.href for a in item.assets.values()]
        filenames = sorted(os.path.basename(h) for h in hrefs)
        folders = {os.path.dirname(h) for h in hrefs}
        source = next((lk.get_href() for lk in item.get_links(_SOURCE_LINK_REL)), None)
        # Spec 34: recover the declared offset/nodata from any asset's raster:bands
        # (all assets on one item share the same granule-level values, §1) — 0 if the
        # item predates the extension (no raster:bands asset at all).
        offset, nodata, scale = 0, 0, 1.0
        if RasterExtension.has_extension(item):  # item-level: hoisted out of the loop
            for asset in item.assets.values():
                bands = RasterExtension.ext(asset).bands
                if bands:
                    nodata = bands[0].nodata or 0
                    scale = bands[0].scale or 1.0
                    if bands[0].offset:
                        # raster:bands offset is reflectance-unit; the catalog
                        # `offset` column is DN-unit (the builder applies it in DN space), so
                        # divide the scale back out — else a datacube built from a re-imported
                        # catalog would be ~1000 DN high (regression of #10/#30).
                        offset = bands[0].offset / scale
        rows.append({
            "id": item.id,
            "collection": item.collection_id,
            "timestamp": pd.to_datetime(item.datetime, utc=True),
            "s3url": source,
            "local_folderpath": folders.pop() if len(folders) == 1 else ",".join(sorted(folders)),
            "files": ",".join(filenames),
            "cloud_cover": item.properties.get("eo:cloud_cover"),
            "processing_version": item.properties.get("processing:version"),
            "processing_datetime": item.properties.get("processing:datetime"),
            "offset": offset,
            "scale": scale,
            "nodata": nodata,
            "properties": json.dumps({k: v for k, v in item.properties.items()
                                       if k not in ("eo:cloud_cover", "grid:code",
                                                    "processing:version",
                                                    "processing:datetime")}),
            "geometry": shapely.geometry.shape(item.geometry),
        })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


# --- declaration mirror on the STAC Collection -------------------------------
#
# Additive, not authoritative: if the Collection and the catalog Parquet footer
# ever disagree, the footer wins. Both are written from the same
# `CollectionDeclaration` object (here), so they cannot drift in the write path.


def _stamp_collection_declaration(
    collection: pystac.Collection, decl: CollectionDeclaration,
) -> None:
    """Mirror `decl` onto `collection`: the mask band's classes as
    the STAC Classification extension's `classification:classes` on an
    `item_assets` entry (standard vocabulary, legible to any STAC-aware tool);
    the remaining fields STAC has no vocabulary for (`reference_band`,
    `mosaic_method`, `mask_keep`, `native_grid`) as `fsd:declaration`, the same
    JSON `rise_eo.catalog.declaration.to_json` writes to the Parquet footer.
    """
    collection.extra_fields[declaration_module.ATTRS_KEY] = declaration_module.to_json(decl)

    if decl.mask_spec is not None and decl.mask_spec.classes:
        # `collection.item_assets` is the top-level property (the ItemAssetsExtension
        # wrapper is deprecated in pystac >= 1.10 in favor of it).
        band = decl.mask_spec.band
        asset_def = collection.item_assets.get(band) or pystac.ItemAssetDefinition.create(
            title=band, description=None, media_type=None, roles=["data", "mask"],
        )
        # Assign into the collection FIRST -- ClassificationExtension.ext(...,
        # add_if_missing=True) needs asset_def.owner set, which only happens on
        # assignment into item_assets.
        collection.item_assets[band] = asset_def
        ClassificationExtension.ext(asset_def, add_if_missing=True).classes = [
            Classification.create(value=v, name=str(v)) for v in decl.mask_spec.classes
        ]


def collection_to_declaration(collection: pystac.Collection) -> CollectionDeclaration | None:
    """Inverse of `_stamp_collection_declaration`: read the `fsd:declaration`
    mirror back off a Collection, or `None` if it carries no stamp.
    Reads the `fsd:declaration` JSON (the footer's authoritative representation,
    §1), not the `classification:classes` mirror -- the two are written from one
    source so they cannot drift, but the JSON is the complete field set.
    """
    raw = collection.extra_fields.get(declaration_module.ATTRS_KEY)
    if raw is None:
        return None
    return declaration_module.from_json(raw)


# --- serialization -----------------------------------------------------------

def write_stac_catalog(
    items: list[pystac.Item],
    dst_folderpath: str,
    *,
    catalog_id: str = "rise-eo",
    collection_id: str | None = None,
    description: str = "rise-eo tile catalog (STAC export).",
    declaration: CollectionDeclaration | None = None,
) -> str:
    """Write a static, self-contained STAC catalog (catalog.json + collection + item JSONs).

    `declaration`, if given, is mirrored onto the Collection -- additive, never
    authoritative: the catalog Parquet footer wins on disagreement. `None` (the default)
    writes no mirror, e.g. for a hand-built item list with no catalog behind it.

    Returns the catalog.json path. Written through `rise_eo.storage`.
    """
    if not items:
        raise ValueError("write_stac_catalog: no items to write.")
    coll_id = collection_id or items[0].collection_id or "default"

    bboxes = [it.bbox for it in items if it.bbox]
    spatial = pystac.SpatialExtent([[
        min(b[0] for b in bboxes), min(b[1] for b in bboxes),
        max(b[2] for b in bboxes), max(b[3] for b in bboxes),
    ]])
    dts = [it.datetime for it in items if it.datetime]
    temporal = pystac.TemporalExtent([[min(dts), max(dts)]])

    collection = pystac.Collection(
        id=coll_id, description=description,
        extent=pystac.Extent(spatial=spatial, temporal=temporal),
    )
    collection.add_items(items)
    if declaration is not None:
        _stamp_collection_declaration(collection, declaration)

    catalog = pystac.Catalog(id=catalog_id, description=description)
    catalog.add_child(collection)

    catalog.normalize_hrefs(str(dst_folderpath))

    # catalog + collection + one Item each -- threaded the same way the collect reads are
    # (see `_ThreadedStorageStacIO`'s docstring for why the tree walk itself stays
    # single-threaded and only the blob write is deferred).
    import concurrent.futures

    n_objects = len(items) + 2  # + the collection + the catalog itself
    tick = _progress.ticker(n_objects, "stac", unit="objects")
    tick(0, force=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=_COLLECT_THREADS) as pool:
        stac_io = _ThreadedStorageStacIO(pool, tick)
        catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED, stac_io=stac_io)
        stac_io.wait()
    tick(n_objects, force=True)
    return os.path.join(str(dst_folderpath), "catalog.json")
