"""Datacube builder — general seam + S2-L2A in-memory implementation.

Spec: specs/03-datacube.md. `build_datacube` is the stable seam; an alternate
engine (e.g. rslearn) must emit the same datacube.npy + metadata.

Artifact contract (specs/00 §6):
  datacube.npy        : 4-D (timestamps|ids, height, width, bands)
  metadata.pickle.npy : {geotiff_metadata, timestamps, ids, bands,
                         data_shape_desc, geometry{shape, crs},
                         actual_start, actual_end, ...}
                         (actual_start/actual_end: this shape's own min/max acquisition
                         timestamp. The run FOLDER names the REQUESTED window, never
                         this data-derived one.)
"""

from __future__ import annotations

import datetime
import json
import os
import time
import uuid
import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager

import geopandas as gpd
import numpy as np
import shapely
from rasterio.crs import CRS

from fsd import config
from fsd.catalog import declaration as declaration_module
from fsd.catalog import processing as processing_module
from fsd.catalog.catalog import filter_by_properties
from fsd.catalog.declaration import (
    MASK_TYPE_CATEGORICAL_CLASSES,
    S2_L2A_DECLARATION,
    CollectionDeclaration,
)
from fsd.datacube import ops
from fsd.raster import images
from fsd.raster.images import apply_offset
from fsd.storage import fs

_RASTER_EXTS = (".jp2", ".tif", ".tiff")
_VALID_IF_MISSING = ("raise_error", "warn", None)
TIMINGS_FILENAME = "timings.json"
READ_LOG_FILENAME = "reads.jsonl"


@contextmanager
def _timed(store: dict, name: str):
    """Record wall-seconds for a build phase into `store` -- the benchmark seam."""
    t0 = time.perf_counter()
    yield
    store[name] = round(time.perf_counter() - t0, 4)


# --- declaration resolution ---------------------------------------------------

def _resolve_declaration(
    gdf: gpd.GeoDataFrame, declaration: CollectionDeclaration | None,
) -> CollectionDeclaration:
    """Resolution order: the explicit `declaration=` kwarg, else the
    stamp already on `gdf.attrs` (restored by `fs.read_parquet` from the Parquet
    footer, or attached by a prior call to this function), else the S2 L2A
    default -- **but only for a hand-built GeoDataFrame**.

    A `gdf` that came from a file (`fs.read_parquet` stamps
    `attrs[fs.SOURCE_PATH_ATTRS_KEY]`) and carries no declaration stamp is an
    error: it would otherwise be built as S2 by coincidence (#42). Re-stamping is a
    footer rewrite (`python -m fsd.catalog.restamp_cli`), not a re-download.
    """
    if declaration is not None:
        return declaration
    stamped = declaration_module.from_attrs(gdf)
    if stamped is not None:
        return stamped
    source_path = gdf.attrs.get(fs.SOURCE_PATH_ATTRS_KEY)
    if source_path is not None:
        raise ValueError(
            f"no CollectionDeclaration stamp found on the catalog read from "
            f"{source_path!r}. A catalog written before spec 35 (or by code that "
            "doesn't stamp one) is not silently assumed to be Sentinel-2 L2A -- "
            "re-stamp it first: `python -m fsd.catalog.restamp_cli "
            f"{source_path} --declaration sentinel-2-l2a` (spec 35 §6), or pass "
            "declaration= explicitly."
        )
    return S2_L2A_DECLARATION


def _resolve_build_declaration(
    catalog_gdf: gpd.GeoDataFrame, declaration: CollectionDeclaration | None,
) -> CollectionDeclaration:
    """`_resolve_declaration`, plus a guard: a build may request a
    **different named collection** from the one the catalog is stamped with only if
    its ARTIFACT facts are identical to the stamp's -- its BUILD POLICY fields
    (`mask_spec`, `mosaic_method`, `mosaic_partition`, `partition_policy`,
    `reference_band`) may freely differ, that is the whole point of a variant.

    A requested declaration disagreeing with the stamp on an artifact fact
    (`nodata`, `scale`, `radiometry_bands`, `band_aliases`,
    `requires_subscription_key`) is a lie about what the bytes ARE, so this raises
    instead of silently building against the wrong radiometry.

    Spec: 58 D14.
    """
    if declaration is None:
        return _resolve_declaration(catalog_gdf, None)
    stamped = declaration_module.from_attrs(catalog_gdf)
    if stamped is None:
        return declaration
    mismatched = [
        field for field in declaration_module.ARTIFACT_FACT_FIELDS
        if getattr(declaration, field) != getattr(stamped, field)
    ]
    if mismatched:
        raise ValueError(
            "build_datacube: the requested declaration disagrees with the catalog's "
            f"stamped declaration on artifact fact(s) {mismatched} -- "
            f"stamped={ {f: getattr(stamped, f) for f in mismatched} }, "
            f"requested={ {f: getattr(declaration, f) for f in mismatched} }. A build "
            "variant may change build policy (mask, reference band, mosaic "
            "method/partition) but never what the bytes ARE (spec 58 D14). Register a "
            "collection variant that keeps the same artifact facts as the stamp."
        )
    return declaration


# --- mosaic partition enforcement ---------------------------------------------

def _enforce_mosaic_partition(
    catalog_subset: gpd.GeoDataFrame, shape_gdf: gpd.GeoDataFrame,
    declared: CollectionDeclaration,
) -> None:
    """A build whose rows span more than one value of `declared.mosaic_partition`
    raises, enumerating what is available -- e.g. mixing ascending and descending
    Sentinel-1 passes into one median would blend physically incompatible backscatter
    geometries.

    No-op when the declaration has `mosaic_partition=()` (e.g. S2 L2A), or when
    `catalog_subset` carries no rows/`properties` to check (an upstream filter already
    reduced this build to nothing, a separate failure mode).

    `partition_policy="auto"` is not implemented and raises `NotImplementedError`
    rather than silently picking an orbit.

    Spec: 58 D9.
    """
    if not declared.mosaic_partition:
        return
    if declared.partition_policy == "auto":
        raise NotImplementedError(
            "build_datacube: partition_policy='auto' is declared but not implemented "
            "(spec 58 D9.2) -- a growable seam fails loudly rather than silently "
            "auto-selecting a partition. Pick one explicitly via properties_filter."
        )
    if declared.partition_policy != "raise":
        raise ValueError(
            f"build_datacube: unknown partition_policy={declared.partition_policy!r}; "
            "expected 'raise' or 'auto'."
        )
    if "properties" not in catalog_subset.columns or catalog_subset.shape[0] == 0:
        return

    by_id = catalog_subset.drop_duplicates(subset="id")
    parsed = {
        row["id"]: (json.loads(row["properties"]) if row["properties"] else {})
        for _, row in by_id.iterrows()
    }
    enforced_keys = list(declared.mosaic_partition)
    distinct = {tuple(props.get(k) for k in enforced_keys) for props in parsed.values()}
    if len(distinct) <= 1:
        return

    # Report `sat:relative_orbit` alongside the enforced key(s) when the catalog
    # carries it -- offered and reported, never enforced.
    report_keys = list(enforced_keys)
    if "sat:relative_orbit" not in report_keys and any(
        "sat:relative_orbit" in props for props in parsed.values()
    ):
        report_keys.append("sat:relative_orbit")

    shape_union = None
    if len(shape_gdf):
        shape_union = shapely.unary_union(
            shape_gdf.to_crs(catalog_subset.crs)["geometry"]
        )

    groups: dict[tuple, list] = {}
    for id_value, props in parsed.items():
        key = tuple(props.get(k) for k in report_keys)
        groups.setdefault(key, []).append(id_value)

    lines = []
    for key, ids in sorted(groups.items(), key=lambda kv: [str(v) for v in kv[0]]):
        label = ", ".join(f"{k}={v!r}" for k, v in zip(report_keys, key))
        coverage_note = ""
        if shape_union is not None and shape_union.area > 0:
            group_union = shapely.unary_union(by_id[by_id["id"].isin(ids)]["geometry"])
            coverage = min(1.0, group_union.intersection(shape_union).area / shape_union.area)
            coverage_note = f", ROI coverage {coverage * 100:.1f}%"
        lines.append(f"  ({label}): {len(ids)} acquisition(s){coverage_note}")

    raise ValueError(
        "build_datacube: rows span multiple values of "
        f"{tuple(enforced_keys)} (spec 58 D9) -- medianing across them would blend "
        "physically incompatible acquisitions. Pass properties_filter to select one, "
        "e.g. properties_filter={" + repr(enforced_keys[0]) + ": <value>}. "
        "Available:\n" + "\n".join(lines)
    )


# --- caller helper: TileCatalog rows -> band-flattened rows -------------------

def flatten_catalog(
    catalog_gdf: gpd.GeoDataFrame,
    declaration: CollectionDeclaration | None = None,
) -> gpd.GeoDataFrame:
    """Explode a filtered `TileCatalog` (one row per granule, with `area_contribution`
    from `TileCatalog.filter`) into one row per raster band file -- the band-flattened
    form `build_datacube` consumes.

    Output cols: `id, filepath, band, timestamp, geometry, area_contribution, offset,
    nodata, properties, ...`. Non-raster files (e.g. `MTD_TL.xml`) are skipped; `band` =
    filename minus ext. `properties` is the granule's source-item STAC properties
    verbatim (JSON string), duplicated across its band rows. `offset` is the granule's
    declared additive offset for a band the resolved declaration says carries radiometry
    (`CollectionDeclaration.is_radiometry_band`), else 0 -- mask/QA bands are never
    harmonized. `nodata` is the granule's declared nodata, 0 when absent; missing
    `offset`/`nodata` columns default every row to 0.

    The declaration is resolved by `_resolve_declaration` and attached as the JSON-able
    `attrs["fsd:declaration"]` -- never the dataclass itself -- which `build_datacube`
    reads instead of hardcoding S2.
    """
    declaration = _resolve_declaration(catalog_gdf, declaration)
    data = {k: [] for k in
            ("id", "filepath", "band", "timestamp", "geometry", "area_contribution",
             "offset", "nodata", "properties", "collection", "acquisition_key",
             "processing_version", "processing_datetime", "source")}
    for _, row in catalog_gdf.iterrows():
        tile_offset = row.get("offset", 0) or 0
        tile_nodata = row.get("nodata", 0)
        tile_nodata = 0 if tile_nodata is None else tile_nodata
        tile_properties = row.get("properties", "{}") or "{}"
        # Carried through band-flattening so `build_datacube` can apply the
        # per-acquisition `processing=` check and record provenance from the SAME rows
        # the build assembles from. A hand-built frame without them is its own
        # acquisition, with no version and no source (cross-processing detection off).
        tile_acq_key = row.get("acquisition_key") or row["id"]
        for file in str(row["files"]).split(","):
            band = next((file[:-len(e)] for e in _RASTER_EXTS if file.endswith(e)),
                        None)
            if band is None:
                continue
            data["id"].append(row["id"])
            data["filepath"].append(os.path.join(row["local_folderpath"], file))
            data["band"].append(band)
            data["timestamp"].append(row["timestamp"])
            data["geometry"].append(row["geometry"])
            data["area_contribution"].append(row["area_contribution"])
            data["offset"].append(tile_offset if declaration.is_radiometry_band(band) else 0)
            data["nodata"].append(tile_nodata)
            # The source item's STAC properties verbatim, carried through
            # band-flattening so `_enforce_mosaic_partition` can read `sat:orbit_state`
            # from the SAME rows the build actually assembles from -- not a re-read of
            # the pre-flatten catalog, which would risk drifting from what was built.
            data["properties"].append(tile_properties)
            data["collection"].append(row.get("collection", ""))
            data["acquisition_key"].append(tile_acq_key)
            data["processing_version"].append(row.get("processing_version"))
            data["processing_datetime"].append(row.get("processing_datetime"))
            data["source"].append(row.get("source", ""))
    flat = gpd.GeoDataFrame(data=data, crs=catalog_gdf.crs)
    declaration_module.to_attrs(flat, declaration)
    return flat


# --- the seam ----------------------------------------------------------------

def build_datacube(
    catalog_subset: gpd.GeoDataFrame,   # band-flattened granules for this shape
    shape_gdf: gpd.GeoDataFrame,        # single geometry (+ id, optional label)
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    bands: list[str],
    *,
    mosaic_days: int = config.MOSAIC_DAYS,
    reference_band: str | None = None,
    declaration: CollectionDeclaration | None = None,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str | None = None,
    export_folderpath: str,
    mosaic_scheme: str = config.MOSAIC_SCHEME,
    njobs: int = 1,
    njobs_load_images: int = 1,
    if_missing_files: str | None = "raise_error",   # raise_error | warn | None
    max_timedelta_days: int = config.MAX_TIMEDELTA_DAYS,
    write_timings: bool = False,
    write_read_log: bool = False,
) -> None:
    """Assemble one cloud-masked, time-mosaicked datacube and save it.

    Steps: missing-files check -> load+crop -> dst_crs (max mean area contribution) ->
    reference profile (merge the reference band) -> resample all to ref -> stack by
    timestamp x band -> declared op sequence (radiometry offset -> mask -> drop mask
    band -> median mosaic) -> save.

    Declaration-driven, never hardcoded (#35): the mask band and rules, the resample
    reference and the mosaic method come from a `CollectionDeclaration`, resolved by
    `_resolve_build_declaration`.

    The declared `mask_spec` is never overridden; a different mask means a different
    named collection variant (`fsd.collections.register`). The mask is skipped when
    `mask_spec` is `None` or its band is not in `bands` (so `bands=["B04"]` builds
    without SCL). An unimplemented `mask_spec.mask_type` raises `NotImplementedError`
    rather than masking approximately, as does `native_grid=True` (a growable seam must
    fail loudly).

    `reference_band`, if given, overrides the declaration's, and must be among `bands`,
    or this raises.

    `declared.mosaic_partition` is enforced before anything else runs
    (`_enforce_mosaic_partition`): rows spanning more than one value of a partitioned
    property (e.g. Sentinel-1's `sat:orbit_state`) raise, enumerating the available
    combinations -- the error is the discovery mechanism. `properties_filter` narrows
    to one; it is applied upstream (`workflows.create_datacube.setup`) and again here,
    idempotent when the rows arrived already filtered. A no-op for a declaration with
    `mosaic_partition=()` (e.g. S2 L2A).

    Each image's `offset` catalog column is applied read-time (`apply_offset`) before the
    median. The build's nodata is the first row's `nodata`, falling back to the
    declaration's `nodata` (NOT `config.NODATA`) when the column is absent.

    `startdate` must be on/before the first acquisition and `enddate` on/after the last
    (median_mosaic requirement). `mosaic_scheme="calendar"` (default) uses fixed windows
    off `startdate`, so cubes over the same startdate/enddate/mosaic_days share an
    identical `timestamps` axis; `"acquisition"` windows track the actual acquisition
    dates.

    `write_timings=True` writes a `timings.json` sidecar (per-phase seconds + counts).
    `write_read_log=True` writes a `reads.jsonl` sidecar (one row per windowed read: grid
    id, mgrs_tile, product_id, band, filepath, wall-clock epoch start/end, duration) and
    needs `njobs_load_images == 1` (no-op with a warning otherwise). The
    workflow enables them via `FSD_WRITE_TIMINGS` / `FSD_WRITE_READ_LOG` (see
    workflows.task).

    Spec: 03, 58.
    """
    declared = _resolve_build_declaration(catalog_subset, declaration)
    # The filter is applied HERE as well as upstream where the catalog is queried
    # (`workflows.create_datacube.setup`) -- applying it twice is idempotent, and a
    # caller that reaches the builder directly with an unfiltered subset (a notebook,
    # a run-book's QGIS step) can then narrow the partition in place instead of only
    # being told the rows span two. Resolved AFTER the declaration, whose stamp rides
    # `catalog_subset.attrs` and need not survive a row slice.
    catalog_subset = filter_by_properties(catalog_subset, properties_filter)
    # The same check runs here as in `create_datacube.setup`, so no entry point
    # routes around it. `processing=None` RAISES on an acquisition present in more than one
    # processing (mosaicking both would count it twice); "latest"/a specifier select one.
    # Idempotent when the rows arrived already selected.
    selection = processing_module.select_processing(catalog_subset, processing)
    processing_module.print_selection(selection, prefix="[build]", processing=processing)
    catalog_subset = selection.kept
    if processing is not None and len(catalog_subset) == 0:
        raise ValueError(
            f"build_datacube: processing={processing!r} left no rows to build from -- no "
            "acquisition has a processing that satisfies it."
        )
    _enforce_mosaic_partition(catalog_subset, shape_gdf, declared)
    if declared.native_grid:
        raise NotImplementedError(
            "build_datacube: a native single-grid source (declaration.native_grid="
            "True) is designed-for but not implemented — grid-topology generalization "
            "(the multi-tile single-CRS-collapse alternative) lands with the ERA5/"
            "CHIRPS spec (spec 34 [G2])."
        )
    if reference_band is None:
        reference_band = declared.reference_band
    if reference_band is not None and reference_band not in bands:
        raise ValueError(
            f"build_datacube: reference_band={reference_band!r} is not among the "
            f"requested bands={bands!r} (spec 58 D11). Add it to `bands`, or declare/pass "
            "a reference_band that is."
        )

    mask_spec = declared.mask_spec
    mask_active = mask_spec is not None and mask_spec.band in bands
    if mask_active and mask_spec.mask_type != MASK_TYPE_CATEGORICAL_CLASSES:
        raise NotImplementedError(
            f"build_datacube: mask_type={mask_spec.mask_type!r} is not implemented — "
            f"only {MASK_TYPE_CATEGORICAL_CLASSES!r} is (spec 34 [G3]); "
            "bitmask/threshold masks land with a later source spec."
        )
    mask_classes = list(mask_spec.classes) if mask_active else None

    if "nodata" in catalog_subset.columns and len(catalog_subset):
        nodata = int(catalog_subset["nodata"].iloc[0])
    else:
        nodata = declared.nodata
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    with _timed(timings, "missing_check"):
        _missing_files_action(
            catalog_gdf=catalog_subset, shape_gdf=shape_gdf, startdate=startdate,
            enddate=enddate, bands=bands, if_missing_files=if_missing_files,
            max_timedelta_days=max_timedelta_days,
        )

    # Load + crop each (granule, band) to the shape; adds crs/image_index, drops
    # unreadable rows. Raster pixel reads use rasterio directly -- the documented seam
    # exception. This is the read phase the benchmarks scrutinise.
    with _timed(timings, "load_images"):
        catalog_gdf, data_profile_list, reads = _load_images(
            catalog_gdf=catalog_subset, shape_gdf=shape_gdf, nodata=nodata,
            njobs=njobs_load_images, write_read_log=write_read_log,
        )
        # ⚠️ Harmonize each image's declared radiometric offset HERE -- before
        # dst_crs/reference/resample/mosaic -- so a calendar window straddling the S2
        # baseline 04.00 cutover (2022-01-25) never medians unharmonized DN together (#10).
        # Move it later and the mosaic silently mixes two radiometries.
        #
        # The cube still clips to uint16 after this, intentionally (see
        # test_build_datacube_still_clips_after_offset_pinned_behavior). The win is that
        # the on-disk COG keeps raw DN: this apply is read-time only, never written back.
        _apply_offsets(catalog_gdf, data_profile_list)

    # Collapse into a single UTM zone so rasterio.merge (single-CRS) can run.
    with _timed(timings, "dst_crs"):
        dst_crs = _get_dst_crs(catalog_gdf)

    # Reference grid = the merged reference-band (B08, 10 m) profile. Everything is
    # resampled TO this real known-10 m image, not to an abstract target grid.
    #
    # `reference_band=None` (bands are already grid-uniform, e.g. S1 RTC) has its own
    # meaning, distinct from "no band matches": use the FIRST requested band's images
    # to build the reference grid. `catalog_gdf["band"] == None` would otherwise compare
    # true for nothing, leaving `ref_indices` empty and `_get_merged_profile` failing on
    # zero images. Since every band is already grid-uniform for these collections,
    # `_get_indices_to_resample` naturally finds nothing to resample against whichever
    # band is picked here -- no separate "skip resample" branch is needed.
    with _timed(timings, "reference_profile"):
        reference_grid_band = reference_band if reference_band is not None else bands[0]
        ref_indices = catalog_gdf.loc[catalog_gdf["band"] == reference_grid_band, "image_index"]
        reference_profile = _get_merged_profile(
            indices=ref_indices, data_profile_list=data_profile_list, dst_crs=dst_crs,
            nodata=nodata, njobs=njobs,
        )

    with _timed(timings, "resample"):
        resample_indices = _get_indices_to_resample(
            indices=catalog_gdf["image_index"], data_profile_list=data_profile_list,
            reference_profile=reference_profile,
        )
        _resample_by_indices(
            indices=resample_indices, data_profile_list=data_profile_list,
            reference_profile=reference_profile, njobs=njobs,
        )

    with _timed(timings, "stack"):
        datacube, metadata = _stack_datacube(
            catalog_gdf=catalog_gdf, data_profile_list=data_profile_list, bands=bands,
            reference_profile=reference_profile, shape_gdf=shape_gdf, nodata=nodata,
        )

    with _timed(timings, "ops"):
        # Declaration-driven op-sequence assembly — replaces the
        # hardcoded S2 chain. `mask_active` (resolved above from the declaration +
        # the requested `bands`) decides whether the mask/drop pair runs at all; a
        # source with no mask (or one whose mask band wasn't requested) skips both,
        # closing #35 without a source-specific `if` here.
        sequence = []
        if mask_active:
            sequence.append((ops.apply_cloud_mask_scl,
                             dict(mask_classes=mask_classes, mask_band=mask_spec.band,
                                  mask_value=nodata)))
            if not declared.mask_keep:
                sequence.append((ops.drop_bands, dict(bands_to_drop=[mask_spec.band])))
        sequence.append((ops.median_mosaic, dict(startdate=startdate, enddate=enddate,
                                                  mosaic_days=mosaic_days,
                                                  mosaic_scheme=mosaic_scheme,
                                                  mask_value=nodata)))
        datacube, metadata = ops.run_ops(datacube, metadata, sequence=sequence)
        # The run folder is named from the REQUESTED window (startdate/enddate), never
        # each shape's actual acquisition min/max -- so that data-derived fact needs a home
        # other than the path. It is real information about the DATA rather than the
        # request, so it goes into the cube's own metadata.
        metadata["actual_start"] = catalog_subset["timestamp"].min()
        metadata["actual_end"] = catalog_subset["timestamp"].max()
        # Which granules and processing versions this cube was built from. Metadata, not
        # identity (never enters `params_key`): recorded so training and inference can be
        # compared (#101) and a stale cube diagnosed.
        metadata["provenance"] = _cube_provenance(catalog_subset)

    with _timed(timings, "save"):
        fs.makedirs(export_folderpath)
        # Metadata is a dict (geometry, per-timestamp mapping, dim names) — things that
        # don't fit in the numpy array. It's pickled via np.save (allow_pickle) rather
        # than raw pickle because a raw pickle written on macOS could not be read on
        # Ubuntu (and vice versa) — np.save's pickling proved cross-platform stable.
        # (xarray is a possible future alternative.)
        # ⚠️ Each artifact is written to a per-attempt temp path and renamed into place only
        # once fully written, so a reader never observes a partial file. metadata is
        # published FIRST and datacube.npy LAST: `datacube.npy`'s existence is the resume
        # signal (workflows.task.run_task), so by the time it appears the metadata it
        # depends on is already durable. Swap the order and a resume can find a cube whose
        # metadata is missing.
        _save_npy_atomic(os.path.join(export_folderpath, "metadata.pickle.npy"),
                          metadata, allow_pickle=True)
        _save_npy_atomic(os.path.join(export_folderpath, "datacube.npy"), datacube)

    if reads is not None:
        _write_read_log(export_folderpath, reads)
    if write_timings:
        _write_timings_sidecar(
            export_folderpath, timings, round(time.perf_counter() - t_all, 4),
            shape_gdf=shape_gdf, catalog_subset=catalog_subset, catalog_gdf=catalog_gdf,
            n_resampled=len(resample_indices), datacube=datacube, metadata=metadata,
            dst_crs=dst_crs,
        )


def _cube_provenance(catalog_subset) -> dict:
    """The provenance block for the rows a cube used (`ids` included: one cube is a
    bounded set of granules)."""
    if len(catalog_subset) == 0 or "id" not in catalog_subset.columns:
        return {}
    collection = "unknown"
    if "collection" in catalog_subset.columns and str(catalog_subset["collection"].iloc[0]):
        collection = str(catalog_subset["collection"].iloc[0])
    return processing_module.provenance(catalog_subset, collection=collection, with_ids=True)


def _save_npy_atomic(path: str, arr, allow_pickle: bool = False) -> None:
    """`fs.save_npy` a temp sidecar next to `path`, then `fs.rename` onto it -- the
    write-to-temp-then-rename publish pattern, also used by `fs.transfer`.

    A crash mid-write leaves the temp path orphaned and `path` absent; it never leaves
    `path` partially written."""
    tmp = f"{path}.tmp-{uuid.uuid4().hex}"
    fs.save_npy(tmp, arr, allow_pickle=allow_pickle)
    fs.rename(tmp, path)


def _write_timings_sidecar(export_folderpath, timings, total_seconds, *, shape_gdf,
                           catalog_subset, catalog_gdf, n_resampled, datacube, metadata,
                           dst_crs):
    """Dump per-phase timings + a few sizing counts as `timings.json`."""
    payload = {
        "id": (str(shape_gdf["id"].iloc[0]) if "id" in shape_gdf.columns else None),
        "total_seconds": total_seconds,
        "phase_seconds": timings,
        "n_band_rows": int(len(catalog_subset)),
        "n_images_loaded": int(len(catalog_gdf)),
        "n_images_resampled": int(n_resampled),
        "n_mosaic_timestamps": int(len(metadata["timestamps"])),
        "datacube_shape": list(datacube.shape),
        "dst_crs": str(dst_crs),
    }
    with fs.open(os.path.join(export_folderpath, TIMINGS_FILENAME), "w") as f:
        json.dump(payload, f, indent=2)


def _write_read_log(export_folderpath, reads):
    """Dump one JSON row per windowed read as `reads.jsonl`."""
    fs.makedirs(export_folderpath)
    with fs.open(os.path.join(export_folderpath, READ_LOG_FILENAME), "w") as f:
        for r in reads:
            f.write(json.dumps(r) + "\n")


def _mgrs_tile(product_id, filepath) -> str | None:
    """Parse the MGRS tile (`..._T36NXF_...`) from the product id or the parent
    folder name. Returns None if neither carries the `_T<MGRS tile>` marker (e.g. the
    synthetic granules in tests) — the same-file key is `filepath`, not this."""
    for s in (product_id, os.path.basename(os.path.dirname(filepath))):
        s = str(s)
        if "_T" in s:
            cand = s.split("_T", 1)[1][:5]
            if len(cand) == 5:
                return cand
    return None


# --- missing-files check -----------------------------------------------------

def _query_stats(catalog_gdf: gpd.GeoDataFrame, shape_gdf: gpd.GeoDataFrame) -> dict:
    stats = {"tile_count": 0, "area_coverage": None, "timedelta_days": None,
             "timestamp_range": None, "band_counts": {}}
    if catalog_gdf.shape[0] == 0:
        return stats

    shape_gdf = shape_gdf.to_crs(catalog_gdf.crs)
    target = shapely.unary_union(shape_gdf["geometry"])
    queried = shapely.unary_union(catalog_gdf["geometry"])
    stats["area_coverage"] = round(1 - (target - queried).area / target.area, 4)

    ts = np.array(sorted(catalog_gdf["timestamp"].tolist()))
    deltas = [td.round("D").days for td in (ts[1:] - ts[:-1])]
    stats["timedelta_days"] = dict(zip(*np.unique(deltas, return_counts=True)))
    stats["timestamp_range"] = (ts.min(), ts.max())

    stats["band_counts"] = catalog_gdf["band"].value_counts().to_dict()
    stats["tile_count"] = max(stats["band_counts"].values())
    return stats


def _check_missing(shape_gdf, catalog_gdf, startdate, enddate, bands,
                   max_timedelta_days):
    stats = _query_stats(catalog_gdf=catalog_gdf, shape_gdf=shape_gdf)
    msgs = []

    if stats["tile_count"] == 0:
        return "No tiles found.", True

    if stats["area_coverage"] < 1:
        msgs.append(f"Incomplete area coverage: {stats['area_coverage'] * 100:.2f}%")

    gaps = [td for td in stats["timedelta_days"] if td > max_timedelta_days]
    if gaps:
        msgs.append("Unusual time gaps found (days): " + ", ".join(map(str, gaps)))

    first_gap = (stats["timestamp_range"][0] - ops._dt2ts(startdate)).days
    last_gap = (ops._dt2ts(enddate) - stats["timestamp_range"][1]).days
    if first_gap > max_timedelta_days:
        msgs.append(f"First available image is {first_gap} days from startdate")
    if last_gap > max_timedelta_days:
        msgs.append(f"Last available image is {last_gap} days from enddate")

    completely = [b for b in bands if b not in stats["band_counts"]]
    partially = [b for b in bands if b in stats["band_counts"]
                 and stats["band_counts"][b] < stats["tile_count"]]
    if completely or partially:
        if completely:
            msgs.append(f"Completely missing bands: {completely}")
        if partially:
            msgs.append(f"Partially missing bands: {partially}")

    return "; ".join(msgs), False


def _missing_files_action(catalog_gdf, shape_gdf, startdate, enddate, bands,
                          if_missing_files, max_timedelta_days):
    if not any(if_missing_files is x for x in _VALID_IF_MISSING):
        raise ValueError(
            f"Invalid if_missing_files={if_missing_files}. "
            f"Must be one of {_VALID_IF_MISSING}"
        )
    msg, nothing_found = _check_missing(
        shape_gdf=shape_gdf, catalog_gdf=catalog_gdf, startdate=startdate,
        enddate=enddate, bands=bands, max_timedelta_days=max_timedelta_days,
    )
    if nothing_found:
        raise ValueError("Missing files error -- " + msg)
    if msg:
        if if_missing_files == "raise_error":
            raise ValueError("Missing files error -- " + msg)
        if if_missing_files == "warn":
            warnings.warn("Missing files warning\n" + msg, RuntimeWarning, stacklevel=2)


# --- load / dst_crs / reference / resample -----------------------------------

def _load_images(catalog_gdf, shape_gdf, nodata, njobs=1, write_read_log=False):
    """Crop every (granule, band) to the shape; return (kept rows, data_profile_list,
    reads). Adds `image_index` (position in the list) and `crs` (str, for grouping);
    drops rows that failed to read. `image_index` still indexes the full list.

    `reads` is None unless `write_read_log`: a per-read timing log built by
    reading each file serially in-process (so `njobs == 1` is required)."""
    catalog_gdf = catalog_gdf.copy()
    filepaths = catalog_gdf["filepath"].tolist()

    reads = None
    if write_read_log and njobs != 1:
        warnings.warn(
            "write_read_log requires njobs_load_images == 1; skipping read log.",
            RuntimeWarning, stacklevel=2,
        )
    if write_read_log and njobs == 1:
        data_profile_list, reads = _load_images_logged(
            filepaths=filepaths, shape_gdf=shape_gdf, nodata=nodata,
            product_ids=catalog_gdf["id"].tolist(), bands=catalog_gdf["band"].tolist(),
        )
    else:
        data_profile_list = images.load_images(
            src_filepaths=filepaths, shapes_gdf=shape_gdf,
            raise_error=False, nodata=nodata, all_touched=True, njobs=njobs,
            print_messages=False,
        )

    idx = [i if dp[0] is not None else -1 for i, dp in enumerate(data_profile_list)]
    if all(i == -1 for i in idx):
        raise ValueError("No valid images found.")

    catalog_gdf["image_index"] = idx
    catalog_gdf["crs"] = [str(p["crs"]) if p is not None else None
                          for _, p in data_profile_list]
    catalog_gdf = catalog_gdf[catalog_gdf["image_index"] != -1]
    return catalog_gdf, data_profile_list, reads


def _apply_offsets(catalog_gdf, data_profile_list) -> None:
    """Apply each row's declared `offset` to its loaded image, in place, read-time only.

    `catalog_gdf` is post-filter (kept rows only), `image_index` still indexes
    the full `data_profile_list`. Missing `offset` column (a source with no
    radiometric-offset concept) is a no-op — every offset defaults to 0."""
    if "offset" not in catalog_gdf.columns:
        return
    for idx, offset in zip(catalog_gdf["image_index"], catalog_gdf["offset"]):
        offset = int(offset) if offset else 0
        if offset:
            data_profile_list[idx] = apply_offset(*data_profile_list[idx], offset=offset)


def _load_images_logged(filepaths, shape_gdf, nodata, product_ids, bands):
    """Serial load+crop, timing each windowed read with wall-clock `time.time()`
    (comparable across processes). Returns (data_profile_list, reads)."""
    grid_id = (str(shape_gdf["id"].iloc[0]) if "id" in shape_gdf.columns else None)
    data_profile_list, reads = [], []
    for fp, pid, band in zip(filepaths, product_ids, bands):
        t0 = time.time()
        dp = images.load_image(
            src_filepath=fp, shapes_gdf=shape_gdf, nodata=nodata,
            all_touched=True, raise_error=False,
        )
        t1 = time.time()
        data_profile_list.append(dp)
        reads.append({
            "id": grid_id,
            "mgrs_tile": _mgrs_tile(pid, fp),
            "product_id": str(pid),
            "band": str(band),
            "filepath": fp,
            "start": round(t0, 6),
            "end": round(t1, 6),
            "duration": round(t1 - t0, 6),
        })
    return data_profile_list, reads


def _get_dst_crs(catalog_gdf) -> CRS:
    """The CRS with the highest mean area contribution (single-zone collapse)."""
    means = catalog_gdf.groupby("crs")["area_contribution"].mean()
    return CRS.from_string(means.sort_values(ascending=False).index[0])


def _get_merged_profile(indices, data_profile_list, dst_crs, nodata=None, njobs=1):
    """Merge the reference-band images into one profile: reproject any off-CRS ones
    to `dst_crs` first, then rasterio.merge (needs a uniform CRS)."""
    selected = [data_profile_list[i] for i in indices]

    diff = [i for i, (_, p) in enumerate(selected) if p["crs"] != dst_crs]
    if diff:
        reproj = images.modify_images_inplace(
            data_profile_list=[selected[i] for i in diff],
            sequence=[(images.reproject, dict(dst_crs=dst_crs))],
            njobs=njobs, print_messages=False,
        )
        for i, dp in zip(diff, reproj):
            selected[i] = dp

    _, merged_profile = images.merge_inplace(data_profile_list=selected, nodata=nodata)
    return merged_profile


def _get_indices_to_resample(indices, data_profile_list, reference_profile) -> list:
    out = []
    for i in indices:
        _, p = data_profile_list[i]
        if (p["crs"] != reference_profile["crs"]
                or p["height"] != reference_profile["height"]
                or p["width"] != reference_profile["width"]):
            out.append(i)
    return out


def _resample_by_indices(indices, data_profile_list, reference_profile, njobs=1):
    resampled = images.modify_images_inplace(
        data_profile_list=[data_profile_list[i] for i in indices],
        sequence=[(images.resample_by_ref_meta, dict(ref_meta=reference_profile))],
        njobs=njobs, print_messages=False,
    )
    for i, dp in zip(indices, resampled):
        data_profile_list[i] = dp
    return data_profile_list


# --- stack -------------------------------------------------------------------

def _stack_datacube(catalog_gdf, data_profile_list, bands, reference_profile,
                    shape_gdf, nodata):
    """Stack aligned images into (timestamps, H, W, bands). Every present band is
    (1, H, W) on the reference grid; a missing (ts, band) is nodata-filled to the
    same shape.

    When several granules of the SAME acquisition cover the shape (it straddles an MGRS
    tile boundary) they collide on (timestamp, band). ALL of them are merged onto the
    reference grid by nodata-fill, never one kept and the rest dropped -- keeping one
    silently discards the coverage of every other granule. Overlap tie-break: dst_crs-native
    granules win over reprojected ones, then lower image_index (deterministic first-valid-wins).
    Each band is merged independently — an S2 granule shares one valid footprint across
    bands, so a pixel resolves to the same granule for every band."""
    dst_crs = reference_profile["crs"]
    native_crs = {c for c in catalog_gdf["crs"].dropna().unique()
                  if CRS.from_string(c) == dst_crs}
    timestamps = sorted(catalog_gdf["timestamp"].unique().tolist())

    # group ALL images per (timestamp, band), each tagged (native_first, image_index)
    ts_band_indices: dict = {}
    for ts, b, idx, crs in zip(catalog_gdf["timestamp"], catalog_gdf["band"],
                               catalog_gdf["image_index"], catalog_gdf["crs"]):
        rank = (0 if crs in native_crs else 1, int(idx))
        ts_band_indices.setdefault((ts, b), []).append(rank)
    ts_id = dict(zip(catalog_gdf["timestamp"], catalog_gdf["id"]))

    ref_h, ref_w = reference_profile["height"], reference_profile["width"]
    fill_dtype = data_profile_list[int(catalog_gdf["image_index"].iloc[0])][0].dtype
    missing = np.full((1, ref_h, ref_w), nodata, dtype=fill_dtype)

    def _merge_on_ref(ranked):
        """Nodata-fill all images for one (timestamp, band) onto the reference grid."""
        out = missing.copy()
        for _, idx in sorted(ranked):          # native (0) before reprojected (1), then idx
            img = data_profile_list[idx][0]    # (1, ref_h, ref_w), already on the ref grid
            gap = out == nodata
            if not gap.any():
                break
            np.copyto(out, img, where=gap)     # first-valid-wins over the still-nodata pixels
        return out

    datacube, ids = [], []
    for ts in timestamps:
        stack = [
            _merge_on_ref(ts_band_indices[(ts, b)])
            if (ts, b) in ts_band_indices else missing
            for b in bands
        ]
        datacube.append(np.stack(stack, axis=-1))   # (1, H, W, bands)
        ids.append(ts_id[ts])
    datacube = np.concatenate(datacube, axis=0)      # (timestamps, H, W, bands)

    metadata = {
        "geotiff_metadata": reference_profile,
        "timestamps": timestamps,
        "ids": ids,
        "bands": list(bands),
        "data_shape_desc": ("timestamps|ids", "height", "width", "bands"),
        "geometry": {"shape": shape_gdf["geometry"].to_list(), "crs": shape_gdf.crs},
    }
    return datacube, metadata
