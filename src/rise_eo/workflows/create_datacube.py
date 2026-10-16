"""High-level batch entrypoint: setup work-units, then run via a runner.

Spec: specs/08-workflows.md. Preserves the demo_01 UX of run_create_datacube.

Setup reads the catalog once, then pre-slices it per shape (via `catalog.filter_gdf`)
so each parallel build job reads only its small subset — no shared-file contention. The
per-row start/end dates written to `input.csv` are the caller's requested window — the
calendar mosaic anchor — and the run-folder name is derived from that same window. This is
the shape-centric workflow TODO #15 will later optimize.
"""

from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence

import geopandas as gpd
import pandas as pd

from rise_eo import collections as _collections
from rise_eo import config
from rise_eo import progress as _progress
from rise_eo.catalog import declaration as declaration_module
from rise_eo.catalog import processing as processing_module
from rise_eo.catalog.catalog import (
    TileCatalog,
    filter_by_properties,
    filter_gdf,
    properties_filter_values,
)
from rise_eo.catalog.declaration import CollectionDeclaration
from rise_eo.datacube.files import CUBE_METADATA_NAME, CUBE_NAME
from rise_eo.storage import fs
from rise_eo.workflows import runners

COL_ID = "id"
COL_LABEL = "label"

# A unit's CONTENT identity: same id + params means the same work, safe to collapse to one
# row. `export_folderpath` is keyed by `id` ALONE (below), which is NARROWER than this tuple
# -- so two rows can differ here and still collide on folder. Catching that is the guard in
# `workflows.runners`, not this dedupe (TODO #53).
_UNIT_IDENTITY_COLS = (
    COL_ID, "startdate", "enddate", "bands", "mosaic_days", "mosaic_scheme", "collection",
    "properties_filter", "processing",
)

DECLARATION_FILENAME = "declaration.json"


def _canonicalize_processing(processing: str | None) -> str:
    """Canonical string form of a build `processing=` value: `None` -> `""`;
    `"latest"` stays; a specifier is stripped of whitespace with its clauses sorted, so
    `">=05.00, <06"` and `"<06,>=05.00"` are one selection and one cube path.

    Like `_canonicalize_properties_filter`, callers fold this into `params_key` ONLY when the
    build actually used `processing` -- `None` must contribute no component at all, or every
    existing cube path would move.

    Spec: 59 D6.
    """
    if processing is None:
        return ""
    if processing == processing_module.LATEST:
        return processing
    return ",".join(sorted(part.strip().replace(" ", "") for part in processing.split(",")))


def _canonicalize_properties_filter(
    properties_filter: Mapping[str, str | Sequence[str]] | None,
) -> str:
    """Canonical string form of a `properties_filter` selection: keys
    sorted, each value set sorted, so one selection is one stable string regardless of
    the caller's dict/list ordering. `None`/empty -> `""`.

    Callers that fold this into `params_key`'s digest add the result ONLY when
    non-empty -- an empty selection must contribute no component at all (not even an
    empty string joined in), or every existing S2 cube path would move.

    Spec: 58 D9.3.
    """
    if not properties_filter:
        return ""
    parts = []
    for key in sorted(properties_filter):
        # Same scalar/string normalization the FILTER uses, so a selection that
        # selects one set of rows is one digest -- `146` and `"146"` must not be two
        # cube paths for the same rows (`sat:relative_orbit` is an int).
        values = properties_filter_values(properties_filter[key])
        parts.append(f"{key}=" + ",".join(sorted(values)))
    return ";".join(parts)


class NoWorkUnitsError(ValueError):
    """`setup` was handed shapes but none of them had granules in range/overlap.

    A `ValueError` subclass so every existing `except ValueError` caller (notably
    `api.verify_adapter`, which turns it into an actionable `PreflightError`) keeps
    working unchanged. It exists so `build_shortfall_only` can catch THIS condition --
    an entirely out-of-coverage shortfall, the known-empty case -- without also
    swallowing `setup`'s OTHER `ValueError`, the duplicate-`id_col` guard, which is a
    loud refusal that must never be silently reinterpreted as "no imagery".
    """


def params_key(
    bands: list[str], mosaic_scheme: str, *, collection: str, declaration: CollectionDeclaration,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str | None = None,
) -> str:
    """A short digest of the params EVERY cell in a run shares -- never the set of ids.

    Folded into the `<window>` path segment so path granularity matches
    `_UNIT_IDENTITY_COLS`: two requests differing only in `bands` MUST resolve to different
    paths, or the second silently overwrites the first and the build skip reads the
    wrong-band cube as "present".

    Keyed on `collection` + the declaration digest because HLS band names (`B04`/`B08`/`B8A`)
    equal S2's, so an HLS and an S2 cube over the same cell/window would otherwise share a
    path. Any declaration change invalidates cached cube paths.

    `properties_filter` and `processing` are appended ONLY when set, so an empty one
    contributes NOTHING (not even an empty component) and a build that does not use them
    keeps its cube path, while two builds differing only in `processing` never share a cube;
    `processing=` is namespaced so it never collides with a filter.

    Same string form `setup` writes to `input.csv`, so a digest from a read-back row agrees.

    Spec: 58 D4, D9.3; 59 D6.
    """
    parts = [",".join(bands), mosaic_scheme, collection, declaration_module.digest(declaration)]
    selection = _canonicalize_properties_filter(properties_filter)
    if selection:
        parts.append(selection)
    if processing is not None:
        parts.append("processing=" + _canonicalize_processing(processing))
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def window_folder_segment(
    startdate: datetime.datetime, enddate: datetime.datetime, mosaic_days: int, *,
    bands: list[str], mosaic_scheme: str, collection: str, declaration: CollectionDeclaration,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str | None = None,
) -> str:
    """The one run-folder segment shared by every cell of a request:
    `<startdate>_<enddate>_m<mosaic_days>_<params_key>`.

    Callers that need to name an expected cube path WITHOUT running `setup` call this with
    the same arguments `setup` was given -- same inputs, same string, no catalog access.
    """
    startdate = pd.to_datetime(startdate, utc=True)
    enddate = pd.to_datetime(enddate, utc=True)
    key = params_key(bands, mosaic_scheme, collection=collection, declaration=declaration,
                     properties_filter=properties_filter, processing=processing)
    return f"{startdate.strftime('%Y%m%d')}_{enddate.strftime('%Y%m%d')}_m{mosaic_days}_{key}"


def cube_export_folderpath(run_folderpath: str, window_segment: str, id_value) -> str:
    """A cube's `export_folderpath` is derivable from `(run_folderpath, window, id)` and
    NOTHING else -- no catalog access is needed to NAME it, only to BUILD it.
    Shared by `setup`'s `_prepare` and by `api._flatten_identity_from_request`, which must
    compute the exact same string without reading `input.csv`."""
    export_folderpath = os.path.join(run_folderpath, window_segment, str(id_value))
    if fs.is_local(export_folderpath):
        # os.path.abspath is only meaningful (and safe) for a local path — on a
        # URL (e.g. abfss://...) it would corrupt the host/scheme (specs/31 §6).
        export_folderpath = os.path.abspath(export_folderpath)
    return export_folderpath


def _refuse_duplicate_ids(shapes_gdf: gpd.GeoDataFrame, id_col: str) -> None:
    """Raise `ValueError` when two shapes share an `id_col` value."""
    # `export_folderpath` is derived from `srow[id_col]`, so two shapes sharing an id are two
    # work-units writing the SAME folder -- concurrently, once `max_concurrent > 1`. On blob
    # that collides on the block-blob commit (`InvalidBlockList`) and the surviving
    # geometry.geojson is whichever shape committed last; locally it silently overwrites.
    # Either way the run is wrong, so refuse it here rather than race. The guard belongs
    # here for every caller, whatever produced the ids.
    if shapes_gdf[id_col].duplicated().any():
        counts = shapes_gdf[id_col].value_counts()
        repeated = counts[counts > 1]
        worst = ", ".join(f"{i}x{n}" for i, n in repeated.head(3).items())
        raise ValueError(
            f"shapes have duplicate '{id_col}' values: {len(repeated)} of "
            f"{shapes_gdf[id_col].nunique()} ids repeated across {len(shapes_gdf)} shapes "
            f"(worst: {worst}). Each id becomes one export folder, so duplicates make "
            f"multiple work-units write the same files concurrently. Deduplicate the shapes "
            f"(or pass an id column that is unique per shape) before calling setup()."
        )


def _append_input_rows(
    rows: list[dict], csv_filepath: str, *, mosaic_days: int, mosaic_scheme: str,
    collection: str, bands: list[str],
    properties_filter: Mapping[str, str | Sequence[str]] | None, processing: str | None,
) -> None:
    """Stamp `rows` with the request's parameters, append them after any rows already in
    `csv_filepath`, dedupe on unit identity and write the file."""
    input_df = pd.DataFrame(rows)
    input_df["added_on"] = pd.Timestamp.now(tz="UTC")
    input_df["mosaic_days"] = mosaic_days
    input_df["mosaic_scheme"] = mosaic_scheme
    input_df["collection"] = collection
    input_df["bands"] = ",".join(bands)
    input_df["properties_filter"] = _canonicalize_properties_filter(properties_filter)
    input_df["processing"] = _canonicalize_processing(processing)

    if fs.exists(csv_filepath):
        with fs.open(csv_filepath, "r") as f:
            input_df = pd.concat([pd.read_csv(f), input_df], ignore_index=True)
    input_df = _dedupe_on_unit_identity(input_df)
    with fs.open(csv_filepath, "w") as f:
        input_df.to_csv(f, index=False)


def setup(
    catalog_filepath: str,
    timestamp_col: str,
    shapefilepath: str,
    id_col: str,
    run_folderpath: str,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    bands: list[str],
    mosaic_days: int,
    csv_filepath: str,
    label_col: str | None,
    mosaic_scheme: str = config.MOSAIC_SCHEME,
    max_concurrent: int = config.SETUP_MAX_CONCURRENT,
    collection: str = config.SATELLITE_S2L2A,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str | None = None,
) -> None:
    """Per geometry: write geometry.geojson + catalog.parquet slice + input.csv row.

    Reads the catalog **once**, then reuses `catalog.filter_gdf` for each shape's date+overlap
    slice (which also persists `area_contribution`); shapes with no intersecting granules are
    skipped with a note. Prints live progress + ETA: the per-shape writes are network I/O on a
    remote run folder, so this can run for minutes and must not look like a hang.

    Shapes are prepared concurrently (`max_concurrent` threads) because that work is
    latency-bound blob I/O, not CPU; `max_concurrent=1` runs serially. `input.csv` row order
    still follows the shapefile's.

    The mosaic anchor written to each row is the caller's `startdate`/`enddate` (not the
    per-shape acquisition min/max), so every shape mosaics on the same calendar grid and the
    cubes share a `timestamps` axis that `flatten` can concatenate. The run-folder name is
    built from that same requested window + `mosaic_days`, never the per-shape acquisition
    range, so every cell of one run lands under one folder that identifies the cube contract
    it was built against; each shape's `actual_start`/`actual_end` live in its cube metadata.

    `timestamp_col` is unused (`filter_gdf`'s catalog rows already fix "timestamp") but stays
    in the caller-facing signature.

    `collection` resolves to a `CollectionDeclaration` on the driver (`rise_eo.collections.get`)
    and is written as JSON to `<run_folderpath>/<window_segment>/declaration.json`, the
    control file every shard reads. This runs uniformly, including for the built-in
    `sentinel-2-l2a`: nodes never consult the registry, so there is one code path rather than
    two where only a user's collection variant breaks, and only remotely.

    `properties_filter` narrows the catalog to rows matching every given STAC property BEFORE
    the per-shape slice. A key no row carries raises (`catalog.filter_by_properties`), naming
    the keys it does carry, rather than silently slicing to zero rows. The canonicalized
    selection is folded into `window_segment` (via `params_key`) and recorded on every
    `input.csv` row, so a differently-filtered re-run never collides on content identity.

    `processing` is applied to each shape's slice AFTER `properties_filter` and the
    date+overlap filter, grouped by `acquisition_key`: `None` (default) raises on any
    acquisition present in more than one processing, listing each group and the arguments that
    resolve it; `"latest"` / a PEP 440 specifier select one row per acquisition (a group with
    no satisfying row is dropped and reported). Joins the cube path as `properties_filter` does.

    Spec: 58 D9, D13; 59 D6.
    """
    startdate = pd.to_datetime(startdate, utc=True)
    enddate = pd.to_datetime(enddate, utc=True)
    declaration = _collections.get(collection)
    # Canonicalize bands to native asset keys as early as possible: every
    # downstream use of `bands` -- the digest, `input.csv`, the builder's catalog-band
    # match -- must see one spelling, or `bands=["B8A"]` and `bands=["nir08"]` would
    # resolve to different cube paths despite naming the identical band.
    bands = [declaration.canonical_to_native(b) for b in bands]
    shapes_gdf = fs.read_geo(shapefilepath)

    _refuse_duplicate_ids(shapes_gdf, id_col)

    # Read the catalog ONCE for the whole run, then filter it in memory per shape
    # (`filter_gdf`). `TileCatalog.filter` re-reads the file on every call, which on a
    # remote catalog costs one full download per shape (one VPN round-trip each) before a
    # single job is submitted. Same rows out, one read in.
    catalog_gdf = TileCatalog(catalog_filepath).read()
    catalog_gdf = filter_by_properties(catalog_gdf, properties_filter)

    errs = processing_module.processing_errors(
        processing, collection=collection, allow_none=True)
    if errs:
        raise ValueError("; ".join(errs))

    window_segment = window_folder_segment(startdate, enddate, mosaic_days,
                                            bands=bands, mosaic_scheme=mosaic_scheme,
                                            collection=collection, declaration=declaration,
                                            properties_filter=properties_filter,
                                            processing=processing)

    # The declaration control file -- written under the WINDOW segment, not the run-folder
    # root. One run folder holds many windows and many collections: `_UNIT_IDENTITY_COLS`
    # carries `collection` precisely so rows of different collections coexist in one
    # `input.csv`, and `_build_shortfall` dispatches every still-missing row in that file
    # regardless of which `setup` call wrote it. A single run-root `declaration.json` would
    # therefore be overwritten by the next `setup` and read back by the earlier call's nodes
    # -- a silently wrong mask/radiometry at a path that names a different collection. The
    # window segment already digests `collection` + the declaration, so it is exactly the
    # right granularity (address per unit path, never one file per run).
    #
    # Written only once the duplicate-id guard above has passed, so a refused call leaves no
    # trace.
    declaration_filepath = os.path.join(
        run_folderpath, window_segment, DECLARATION_FILENAME)
    fs.makedirs(os.path.join(run_folderpath, window_segment))
    fs.write_text(declaration_filepath,
                  json.dumps(declaration_module.to_json(declaration)))

    n_shapes = len(shapes_gdf)
    print(f"[setup] catalog read once: {len(catalog_gdf)} rows, for {n_shapes} shapes",
          flush=True)
    selections: list[processing_module.Selection] = []   # one per shape (thread-safe append)

    # The throttle + rate + ETA math lives in `rise_eo.progress` so every driver-side loop
    # shares one implementation and one output format. setup does per-shape network I/O and
    # can run for many minutes on a remote run folder, where silence is indistinguishable
    # from a hang.
    _tick = _progress.ticker(n_shapes, "setup", unit="shapes")

    def _prepare(srow) -> dict | None:
        """One shape's control files + its input.csv row. Pure per-shape work: it
        touches only this shape's own folder, and reads (never mutates) the shared
        `catalog_gdf` — which is what makes the pool below safe."""
        shape_gdf = gpd.GeoDataFrame(
            {"geometry": [srow["geometry"].buffer(0)], COL_ID: [srow[id_col]]},
            crs=shapes_gdf.crs,
        )
        if label_col is not None:
            shape_gdf[COL_LABEL] = srow[label_col]

        subset = filter_gdf(catalog_gdf, shape_gdf, startdate, enddate)
        # One processing per acquisition among the rows this cube would mosaic.
        # `None` raises on a duplicate group; the error names the shape.
        try:
            selection = processing_module.select_processing(subset, processing)
        except ValueError as exc:
            raise ValueError(f"shape id={srow[id_col]!r}: {exc}") from exc
        selections.append(selection)
        subset = selection.kept
        if subset.shape[0] == 0:
            print(f"[setup] skip id={srow[id_col]}: no tiles in range/overlap"
                  + (f" after processing={processing!r}" if processing is not None else ""),
                  flush=True)
            return None

        export_folderpath = cube_export_folderpath(run_folderpath, window_segment, srow[id_col])
        fs.makedirs(export_folderpath)
        shape_path = os.path.join(export_folderpath, "geometry.geojson")
        catalog_path = os.path.join(export_folderpath, "catalog.parquet")
        # Write via rise_eo.storage rather than gpd.to_file(path) directly, so this per-unit
        # geometry lands correctly on a remote export_folderpath too.
        # write_text (not fs.open) so a concurrent adlfs InvalidBlockList retries (TODO #57).
        fs.write_text(shape_path, shape_gdf.to_json())
        fs.write_parquet(catalog_path, subset)

        row = {
            "shapefilepath": shape_path,
            # Calendar anchor = the caller's window, not per-shape actual
            # acquisition min/max — so all shapes mosaic on the same grid.
            "startdate": startdate,
            "enddate": enddate,
            "catalog_filepath": catalog_path,
            "export_folderpath": export_folderpath,
            "datacube_filepath": os.path.join(export_folderpath, CUBE_NAME),
            "images_count": int(subset.shape[0]),
            "declaration_filepath": declaration_filepath,
            COL_ID: srow[id_col],
        }
        if label_col is not None:
            row[COL_LABEL] = srow[label_col]
        return row

    # Threads, not processes: every shape costs ~4-7 tiny blob round-trips
    # (`makedirs` + `geometry.geojson` + the `catalog.parquet` slice), so the loop is
    # latency-bound and the GIL is released for the duration of each call. Same
    # pattern `sources.mpc.download`/`download_shard` already run concurrently through
    # `rise_eo.storage` against blob. Serially this costs minutes for hundreds of shapes, before
    # a single job is submitted.
    #
    # Results are placed BY INDEX and compacted afterwards, so `input.csv` row order
    # is the shapefile's order regardless of completion order — parallelism must not
    # change the manifest. (An exception in a worker propagates out of `fut.result()`; the
    # pool's `__exit__` lets in-flight shapes finish first, so slightly more work lands
    # before it surfaces.)
    prepared: list[dict | None] = [None] * n_shapes
    _tick(0, force=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {
            pool.submit(_prepare, srow): i
            for i, (_, srow) in enumerate(shapes_gdf.iterrows())
        }
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            prepared[futures[fut]] = fut.result()
            done += 1
            _tick(done)

    rows = [r for r in prepared if r is not None]
    _tick(n_shapes, force=True)
    _report_selections(selections, processing)

    if not rows:
        raise NoWorkUnitsError("setup produced no work-units (no shape had tiles in range).")

    _append_input_rows(
        rows, csv_filepath, mosaic_days=mosaic_days, mosaic_scheme=mosaic_scheme,
        collection=collection, bands=bands, properties_filter=properties_filter,
        processing=processing,
    )


def _report_selections(selections, processing: str | None) -> None:
    """One summary line for the whole `setup` pass (a drop is reported, never
    silent) -- per-granule lines would be N shapes x M skips of noise."""
    superseded = {s["id"] for sel in selections for s in sel.skipped}
    dropped = {k for sel in selections for k in sel.dropped_acquisitions}
    if not superseded and not dropped:
        return
    print(f"[setup] processing={processing!r}: {len(superseded)} granule(s) not used "
          f"(superseded or outside the specifier); {len(dropped)} acquisition(s) dropped "
          "because no processing satisfies it"
          + (f" (e.g. {sorted(dropped)[0]})" if dropped else ""), flush=True)


def _dedupe_on_unit_identity(input_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows sharing the same content identity (`_UNIT_IDENTITY_COLS`) to one,
    keeping the newest by `added_on`.

    `setup` appends unconditionally, so without this an idempotent re-run grows `input.csv`
    by a duplicate copy of every unit each time (TODO #53). A re-run adding a genuinely new
    shape, or a changed window/params for an existing id, still adds a distinct row: this is
    a dedupe, not a "one row per id" collapse.

    Order is otherwise preserved -- `setup`'s manifest order is the shapefile's order, not
    completion order, and that is a contract other code relies on.
    """
    cols = [c for c in _UNIT_IDENTITY_COLS if c in input_df.columns]
    if not cols:
        return input_df
    # A prior run's rows round-tripped through CSV (dates/added_on came back as
    # strings); this run's freshly-prepared rows are still in-memory Timestamps. Build
    # a canonicalized identity key per row (dates normalized to a comparable ISO
    # string) rather than comparing the raw columns, so the two never fail to match on
    # type/format alone.
    date_cols = {"startdate", "enddate"} & set(cols)
    key_df = input_df[cols].copy()
    for c in date_cols:
        key_df[c] = pd.to_datetime(key_df[c], utc=True).astype(str)
    # `",".join([])` -> `""` -> an empty CSV field -> read back as NaN, not `""` (the
    # same round-trip `_row_matches_window` documents for `bands`; `properties_filter`
    # hits it too, since a request with no selection stores `""`). Without this, a
    # CSV-round-tripped NaN in a non-date column survives `key_df.astype(str)` as an
    # actual float under this pandas' string dtype (it does NOT become the literal
    # string "nan"), and `"|".join` on a row containing it raises `TypeError`.
    non_date_cols = [c for c in cols if c not in date_cols]
    key_df[non_date_cols] = key_df[non_date_cols].fillna("")
    identity_key = key_df.astype(str).agg("|".join, axis=1)

    if "added_on" not in input_df.columns:
        return input_df.loc[~identity_key.duplicated(keep="last")]
    sort_key = pd.to_datetime(input_df["added_on"], utc=True)
    order = sort_key.argsort(kind="stable")
    keep_last = ~identity_key.iloc[order].duplicated(keep="last")
    keep_index = identity_key.iloc[order].index[keep_last.to_numpy()]
    return input_df.loc[input_df.index.isin(keep_index)]  # restore manifest order


def _cube_present(datacube_filepath: str) -> bool:
    """A cube counts as present only when BOTH `datacube.npy` and its `metadata.pickle.npy`
    sibling exist and are non-empty.

    A half-written cube counted as done is the same defect as a truncated download
    catalogued as complete (#74) -- do not reintroduce it by checking one file, or by
    checking existence without size.
    """
    metadata_filepath = os.path.join(os.path.dirname(datacube_filepath), CUBE_METADATA_NAME)
    for filepath in (datacube_filepath, metadata_filepath):
        if not fs.exists(filepath) or fs.size(filepath) == 0:
            return False
    return True


_CUBE_FILES = (CUBE_NAME, CUBE_METADATA_NAME)


def _present_cube_ids_at(window_folderpath: str) -> set[str] | None:
    """The ids whose cube is fully built, from ONE recursive listing of the window folder.

    `_cube_present` costs four blob round-trips per cell (`exists` + `size`, twice), and
    both driver-side sweeps run it per cell, serially: hundreds of cells is thousands of
    sequential round-trips over the WAN with no output, which reads as a hang. A directory
    walk answers the same question in a couple of paginated requests.

    Keyed by the `<id>` path leaf, never the full path, so it does not have to reconcile
    `os.path.abspath` (local) against a backend's own path spelling (`abfss://` -> adlfs
    returns `container/...`). Returns `None` when the folder cannot be listed at all --
    including the ordinary "nothing built yet" case -- and the caller falls back to
    per-path checks, so this is a fast path, never a new source of truth.
    """
    try:
        listing = fs.find_sizes(window_folderpath)
    except Exception:  # noqa: BLE001 - unlistable (absent, or a backend without find)
        return None
    have: dict[str, set[str]] = {}
    for path, nbytes in listing.items():
        if nbytes <= 0:
            continue  # a zero-byte artifact is not present (#74)
        parts = path.replace("\\", "/").rstrip("/").split("/")
        if len(parts) < 2 or parts[-1] not in _CUBE_FILES:
            continue
        have.setdefault(parts[-2], set()).add(parts[-1])
    return {id_value for id_value, names in have.items() if len(names) == len(_CUBE_FILES)}


def _cube_present_many(
    datacube_filepaths, *, label: str, max_concurrent: int = config.SETUP_MAX_CONCURRENT,
) -> dict[str, bool]:
    """`_cube_present` for many cubes: concurrent (latency-bound blob I/O, exactly
    `setup`'s own argument for threads) and ticked, so a long sweep is never silent.
    The fallback for when `_present_cube_ids_at` cannot list the folder."""
    paths = list(datacube_filepaths)
    out: dict[str, bool] = {}
    if not paths:
        return out
    _tick = _progress.ticker(len(paths), label, unit="cubes")
    _tick(0, force=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futures = {pool.submit(_cube_present, p): p for p in paths}
        for done, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
            out[futures[fut]] = fut.result()
            _tick(done)
    _tick(len(paths), force=True)
    return out


def _presence_for_paths(datacube_filepaths, *, label: str) -> dict[str, bool]:
    """`{datacube_filepath: is the cube fully built}` for many cubes at once.

    The fast path is one recursive listing per `<window>` folder (`_present_cube_ids_at`);
    anything that cannot be listed falls back to concurrent, ticked per-path checks. Both
    driver-side sweeps -- the announced plan and the dispatch decision -- go through here,
    so neither can regress into a silent serial walk.
    """
    paths = list(datacube_filepaths)
    by_folder: dict[str, list[str]] = {}
    for path in paths:
        by_folder.setdefault(os.path.dirname(os.path.dirname(path)), []).append(path)

    out: dict[str, bool] = {}
    unresolved: list[str] = []
    for window_folderpath, group in by_folder.items():
        present_ids = _present_cube_ids_at(window_folderpath)
        if present_ids is None:
            unresolved.extend(group)
            continue
        for path in group:
            out[path] = os.path.basename(os.path.dirname(path)) in present_ids
    out.update(_cube_present_many(unresolved, label=label))
    return out


def _row_matches_window(
    row, *, bands: list[str], mosaic_days: int, startdate, enddate,
    mosaic_scheme: str, collection: str,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str | None = None,
) -> bool:
    """Does an existing `input.csv` row belong to THIS request's window/params? Same
    canonicalization `_dedupe_on_unit_identity` uses (dates -> comparable ISO strings)
    so a row that round-tripped through CSV still compares correctly against in-memory
    request values.

    `properties_filter` is compared too, canonicalized -- a row from a
    DIFFERENT partition selection over the same bands/window/collection must not be
    mistaken for a match, or a re-run scoped to one orbit could silently adopt another
    orbit's row.
    """
    want = {
        "bands": ",".join(bands),
        "mosaic_days": str(mosaic_days),
        "startdate": str(pd.to_datetime(startdate, utc=True)),
        "enddate": str(pd.to_datetime(enddate, utc=True)),
        "mosaic_scheme": mosaic_scheme,
        "collection": collection,
        "properties_filter": _canonicalize_properties_filter(properties_filter),
        "processing": _canonicalize_processing(processing),
    }
    for col, want_val in want.items():
        if col not in row.index:
            continue
        got = row[col]
        if col in ("startdate", "enddate"):
            got = str(pd.to_datetime(got, utc=True))
        else:
            # `",".join([])` -> `""` -> an empty CSV field -> read back as NaN,
            # not `""`. Without this, an empty-string field (a legitimate request)
            # round-trips to "nan" and never matches its own freshly-written request
            # value, purging every row on every call.
            got = "" if pd.isna(got) else str(got)
        if got != want_val:
            return False
    return True


def _row_matches_path(row, *, run_folderpath: str, window_segment: str) -> bool:
    """Does an existing `input.csv` row name the path THIS request derives for its id?

    ⚠️ `_row_matches_window` compares the run PARAMETERS, and that is NOT sufficient on its
    own. Change the path shape (e.g. the `_<params_key>` digest in the `<window>` segment)
    and a row written before the change still matches on every parameter while pointing at
    the OLD folder. Adopt such a row and the new addressing silently never takes effect: the
    plan announces a full rebuild, while the build leg reads the row's own stale
    `datacube_filepath`, finds the old cube present, and dispatches nothing. Worse,
    the flatten stamp then records old paths the request-derived identity can never
    reproduce, so the top-level short-circuit is dead for that request forever.

    So the path is part of what makes a row current. A row that fails this is purged and its
    id goes back into the shortfall, which regenerates the row at the right path.
    """
    if COL_ID not in row.index:
        return False
    expected = cube_export_folderpath(run_folderpath, window_segment, row[COL_ID])
    for col, want in (("export_folderpath", expected),
                      ("datacube_filepath", os.path.join(expected, CUBE_NAME))):
        if col not in row.index or pd.isna(row[col]):
            return False
        if str(row[col]) != want:
            return False
    return True


def _manifest_filepath(run_folderpath: str) -> str:
    return os.path.join(run_folderpath, "_manifest.json")


def _read_known_empty(run_folderpath: str, window_segment: str) -> set[str]:
    """The known-empty manifest, keyed to the window/params segment.

    That segment already carries the request identity, so reusing it keeps ONE identity
    granularity for both the cube path and the known-empty record -- which is what makes a
    changed window or band set clear the record automatically, since the key itself changes.
    """
    entry = _load_manifest(run_folderpath).get(window_segment)
    if not isinstance(entry, dict):
        return set()
    return {str(v) for v in entry.get("ids", [])}


def _load_manifest(run_folderpath: str) -> dict:
    path = _manifest_filepath(run_folderpath)
    if not fs.exists(path):
        return {}
    try:
        with fs.open(path, "r") as f:
            loaded = json.load(f)
    except Exception:  # noqa: BLE001 - corrupt reads as "no information"; writers replace it
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _write_known_empty(run_folderpath: str, window_segment: str, ids: set[str]) -> None:
    manifest = _load_manifest(run_folderpath)
    if ids:
        manifest[window_segment] = {"ids": sorted(ids)}
    else:
        manifest.pop(window_segment, None)
    fs.write_text(_manifest_filepath(run_folderpath),
                  json.dumps(manifest, indent=2, sort_keys=True))


def _record_known_empty(run_folderpath: str, window_segment: str, new_ids) -> None:
    if not new_ids:
        return
    existing = _read_known_empty(run_folderpath, window_segment)
    _write_known_empty(run_folderpath, window_segment,
                       existing | {str(v) for v in new_ids})


def _forget_known_empty(run_folderpath: str, window_segment: str, recovered_ids) -> None:
    """The manifest must not be write-only. An id recorded known-empty that LATER gets
    an `input.csv` row -- because the archive was re-ingested, or because a forced
    rebuild (`overwrite="datacubes"`/`True`) re-derived every shape from the catalog --
    is no longer empty, and leaving it recorded makes `api._flatten_identity_from_request`
    subtract an id that `input.csv` genuinely names. The two identities could then never
    agree again and the top-level short-circuit would be dead for that request forever:
    the exact failure the manifest exists to prevent, moved one step downstream.
    """
    recovered = {str(v) for v in recovered_ids}
    if not recovered:
        return
    existing = _read_known_empty(run_folderpath, window_segment)
    remaining = existing - recovered
    if remaining != existing:
        _write_known_empty(run_folderpath, window_segment, remaining)


def _clear_known_empty(run_folderpath: str, window_segment: str) -> None:
    """Drop this window's whole known-empty entry -- for a caller that has just
    re-derived every shape from the catalog and whose `input.csv` is therefore the
    authority. See `_forget_known_empty`."""
    if _read_known_empty(run_folderpath, window_segment):
        _write_known_empty(run_folderpath, window_segment, set())


def _current_rows(
    csv_filepath: str, *, keep: Callable[[pd.Series], bool],
) -> tuple[pd.DataFrame | None, set[str]]:
    """The rows of an existing `csv_filepath` that `keep` accepts, and their ids
    (`(None, set())` when the file does not exist)."""
    existing_df = None
    existing_ids: set[str] = set()
    if fs.exists(csv_filepath):
        with fs.open(csv_filepath, "r") as f:
            existing_df = pd.read_csv(f)
        if len(existing_df):
            keep_mask = existing_df.apply(keep, axis=1)
            existing_df = existing_df.loc[keep_mask]
        # `input.csv` rows are always written under `COL_ID` ("id") by `setup`,
        # regardless of the caller's own `id_col` name (e.g. "fid") -- the caller's
        # `shapes_gdf` is the only frame that still uses the caller's `id_col`.
        if len(existing_df) and COL_ID in existing_df.columns:
            existing_ids = set(existing_df[COL_ID].astype(str))
    return existing_df, existing_ids


def build_shortfall_only(
    *, catalog_filepath: str, timestamp_col: str, shapefilepath: str, id_col: str,
    run_folderpath: str, startdate, enddate, bands: list[str],
    mosaic_days: int, csv_filepath: str,
    label_col: str | None, mosaic_scheme: str = config.MOSAIC_SCHEME,
    max_concurrent: int = config.SETUP_MAX_CONCURRENT,
    collection: str = config.SATELLITE_S2L2A,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str | None = None,
) -> tuple[int, int, int]:
    """The build leg of the backward walk: call `setup` only for the shapes that need it.

    Every id's cube target is named from the REQUEST alone (`window_folder_segment` +
    `cube_export_folderpath`), so `setup` runs only for shapes whose cube is genuinely
    missing and not already recorded as known-empty -- never for the whole shapefile.

    Rows in an existing `csv_filepath` for a DIFFERENT window/params are dropped first: this
    function only ever GROWS `input.csv` within ONE window. Accumulating ACROSS windows is
    deliberately out of scope and blocked on #84 -- two windows of one id would collide in
    `ids.npy`.

    Prints the `[plan]` build line before any `setup` call. Returns
    `(n_present, n_missing, n_known_empty)`.
    """
    declaration = _collections.get(collection)
    bands = [declaration.canonical_to_native(b) for b in bands]
    shapes_gdf = fs.read_geo(shapefilepath)
    window_segment = window_folder_segment(
        startdate, enddate, mosaic_days, bands=bands, mosaic_scheme=mosaic_scheme,
        collection=collection, declaration=declaration, properties_filter=properties_filter,
        processing=processing,
    )
    known_empty = _read_known_empty(run_folderpath, window_segment)

    existing_df, existing_ids = _current_rows(
        csv_filepath, keep=lambda row: _row_matches_window(
            row, bands=bands, mosaic_days=mosaic_days, startdate=startdate,
            enddate=enddate, mosaic_scheme=mosaic_scheme,
            collection=collection, properties_filter=properties_filter,
            processing=processing,
        ) and _row_matches_path(
            row, run_folderpath=run_folderpath, window_segment=window_segment,
        ),
    )

    # ONE presence sweep for the whole request, up front -- a listing where the backend
    # allows it, concurrent+ticked checks otherwise. The loop below and the plan line share
    # it: a separate serial walk for each would cost four blob round-trips per cell, twice,
    # silently.
    cube_filepaths = {
        str(srow[id_col]): os.path.join(
            cube_export_folderpath(run_folderpath, window_segment, srow[id_col]),
            CUBE_NAME,
        )
        for _, srow in shapes_gdf.iterrows()
    }
    presence = _presence_for_paths(cube_filepaths.values(), label="plan")
    cube_ids = {id_value for id_value, path in cube_filepaths.items() if presence.get(path)}

    present_ids: list[str] = []
    missing_srows: list = []
    for _, srow in shapes_gdf.iterrows():
        id_value = str(srow[id_col])
        if id_value in existing_ids:
            present_ids.append(id_value)
            continue
        if id_value in cube_ids:
            # A cube with no row is not "satisfied": nothing downstream (the build leg,
            # flatten) ever looks at a cube directly, only at `input.csv` rows, and nothing
            # else calls `setup` for this id. Route it through `setup` -- idempotent, and
            # `_build_shortfall` still skips the cube itself -- so the row comes back. A
            # real cube also overrules a stale known-empty record, which is why this branch
            # is tested first.
            missing_srows.append(srow)
        elif id_value in known_empty:
            pass  # known-empty, satisfied -- never rediscovered
        else:
            missing_srows.append(srow)

    # `present_ids`/`missing_srows` above answer "does setup need to run for this id" (a
    # ROW question) -- but the announced plan must match what `_build_shortfall`
    # actually dispatches next (a CUBE question). An interrupted prior run can have a
    # row with no cube behind it yet; count that as missing for the printed line
    # without changing whether `setup` reruns for it (it doesn't need to -- the row is
    # already correct, only the runner needs to build the cube).
    cube_missing_ids = {i for i in present_ids if i not in cube_ids}
    n_total = len(shapes_gdf)
    n_missing = len(missing_srows)
    n_known_empty = n_total - len(present_ids) - n_missing
    n_missing_for_plan = n_missing + len(cube_missing_ids)
    n_present_for_plan = len(present_ids) - len(cube_missing_ids)
    print(f"[plan]   build: {n_present_for_plan} present, {n_missing_for_plan} missing, "
          f"{n_known_empty} known-empty -> will build {n_missing_for_plan}", flush=True)

    # Persist the window-scoped purge even when nothing is missing -- an existing
    # `input.csv` carrying a DIFFERENT window's rows must never leak into this window's
    # flatten (which requires every cube to share bands/timestamps).
    if existing_df is not None:
        with fs.open(csv_filepath, "w") as f:
            existing_df.to_csv(f, index=False)

    if not missing_srows:
        return len(present_ids), n_missing, n_known_empty

    shortfall_gdf = gpd.GeoDataFrame(missing_srows, crs=shapes_gdf.crs)
    with tempfile.TemporaryDirectory() as tmp:
        shortfall_shapefilepath = os.path.join(tmp, "shortfall.geojson")
        shortfall_gdf.to_file(shortfall_shapefilepath, driver="GeoJSON")
        try:
            setup(
                catalog_filepath=catalog_filepath, timestamp_col=timestamp_col,
                shapefilepath=shortfall_shapefilepath, id_col=id_col,
                run_folderpath=run_folderpath, startdate=startdate, enddate=enddate,
                bands=bands, mosaic_days=mosaic_days,
                csv_filepath=csv_filepath, label_col=label_col, mosaic_scheme=mosaic_scheme,
                max_concurrent=max_concurrent, collection=collection,
                properties_filter=properties_filter, processing=processing,
            )
        except NoWorkUnitsError:
            # `setup` raises when NONE of the shapes it was handed have granules in range.
            # That is reachable here precisely because this call is scoped to the shortfall,
            # where one out-of-coverage polygon can be the whole batch. Record the shortfall
            # as known-empty and let the caller's request converge, rather than crashing the
            # entire `create_training_data` call.
            #
            # ⚠️ Deliberately NOT `except ValueError`: `setup`'s duplicate-`id_col` guard
            # raises that too, and swallowing it would record a caller's duplicated shapes
            # as "no imagery" -- turning a loud refusal into silently missing training data.
            newly_empty = [str(srow[id_col]) for srow in missing_srows]
            _record_known_empty(run_folderpath, window_segment, newly_empty)
            print(f"[setup] shortfall of {len(missing_srows)} had no tiles in range/overlap "
                  f"-- recorded known-empty", flush=True)
            return len(present_ids), n_missing, n_known_empty

    with fs.open(csv_filepath, "r") as f:
        after_df = pd.read_csv(f)
    built_ids = set(after_df[COL_ID].astype(str)) if len(after_df) else set()
    newly_empty = [str(srow[id_col]) for srow in missing_srows
                   if str(srow[id_col]) not in built_ids]
    _record_known_empty(run_folderpath, window_segment, newly_empty)
    # ...and the converse: anything that now HAS a row is not empty any more.
    _forget_known_empty(run_folderpath, window_segment, built_ids)

    return len(present_ids), n_missing, n_known_empty


def _build_shortfall(csv_filepath: str, *, force: bool) -> tuple[str, int, int]:
    """Which `input.csv` rows still need a cube built -- the driver-side diff, one level up
    from the download diff. Returns `(dispatch_csv_filepath, n_total, n_missing)`.

    `force=True` (an `overwrite=` rebuild) treats every row as missing without touching the
    filesystem: the driver dispatches every row again, and the node actually rebuilds only
    because the caller is expected to have cleared the old artifacts first (`_force_rebuild`).

    When nothing is missing, or nothing is present yet, `dispatch_csv_filepath` IS
    `csv_filepath` -- no temp file, no extra write. Only a PARTIAL shortfall gets its own
    sibling CSV holding just the missing rows, so a run that is 95% built does not fan out
    100%.
    """
    with fs.open(csv_filepath, "r") as f:
        df = pd.read_csv(f)
    n_total = len(df)
    if force:
        return csv_filepath, n_total, n_total
    presence = _presence_for_paths(df["datacube_filepath"], label="build")
    missing_mask = ~df["datacube_filepath"].map(lambda p: presence.get(p, False))
    n_missing = int(missing_mask.sum())
    if n_missing in (0, n_total):
        return csv_filepath, n_total, n_missing
    shortfall_csv_filepath = f"{csv_filepath}.shortfall.csv"
    with fs.open(shortfall_csv_filepath, "w") as f:
        df.loc[missing_mask].to_csv(f, index=False)
    return shortfall_csv_filepath, n_total, n_missing


def _force_rebuild(csv_filepath: str) -> None:
    """Clear each row's existing cube files so a forced rebuild actually rebuilds.

    `workflows.task`'s own node-side skip (`fs.exists(datacube.npy)`) would otherwise no-op
    every row whose cube still exists. This stays identity-free -- no modification time is
    read -- and removes whatever is there, unconditionally, for exactly the rows this run
    addresses.
    """
    with fs.open(csv_filepath, "r") as f:
        df = pd.read_csv(f)
    for datacube_filepath in df["datacube_filepath"]:
        metadata_filepath = os.path.join(
            os.path.dirname(datacube_filepath), CUBE_METADATA_NAME
        )
        for filepath in (datacube_filepath, metadata_filepath):
            if fs.exists(filepath):
                fs.rm(filepath)


def run_create_datacube(
    catalog_filepath: str,
    timestamp_col: str,
    shapefilepath: str,
    id_col: str,
    run_folderpath: str,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
    bands: list[str],
    mosaic_days: int,
    csv_filepath: str,
    label_col: str | None,
    cores: int,
    *,
    mosaic_scheme: str = config.MOSAIC_SCHEME,
    dry_run: bool = False,
    unlock: bool = False,
    overwrite_setup_csv: bool = True,
    overwrite: bool = False,
    runner: str = "local",
    runner_kwargs: dict | None = None,
    collection: str = config.SATELLITE_S2L2A,
    properties_filter: Mapping[str, str | Sequence[str]] | None = None,
    processing: str | None = None,
):
    """Run setup (unless csv exists), then dispatch only the cubes that are still missing.

    `runner_kwargs` is forwarded to `runners.run_aml` when `runner="aml"`
    (e.g. `cluster=`, `environment=`, `root=`, `identity_client_id=`) -- the local runner
    takes no extra kwargs, so it is ignored for `runner="local"`.

    `overwrite=True` forces every cube in `csv_filepath` to be rebuilt (clearing existing
    artifacts first, `_force_rebuild`); `False` (default) skips per cell (`_build_shortfall`):
    a shortfall of 0 prints and returns WITHOUT submitting a single job; a partial shortfall
    dispatches only the missing rows. No modification time is read -- presence is
    `datacube.npy` + `metadata.pickle.npy`, both non-empty.

    `overwrite_setup_csv=True` (default) delete-then-regenerates the whole `input.csv` every
    call. `False` -- what `create_training_data` passes -- runs `build_shortfall_only`
    instead, calling `setup` ONLY for shapes whose cube target (named from the request, no
    catalog access) is missing and not already known-empty. Removing the flag entirely is
    blocked on #84.
    """
    bands = [_collections.get(collection).canonical_to_native(b) for b in bands]
    if overwrite_setup_csv:
        if fs.exists(csv_filepath):
            fs.rm(csv_filepath)
        setup(
            catalog_filepath=catalog_filepath, timestamp_col=timestamp_col,
            shapefilepath=shapefilepath, id_col=id_col, run_folderpath=run_folderpath,
            startdate=startdate, enddate=enddate, bands=bands,
            mosaic_days=mosaic_days,
            csv_filepath=csv_filepath, label_col=label_col, mosaic_scheme=mosaic_scheme,
            collection=collection, properties_filter=properties_filter,
            processing=processing,
        )
        # This pass just re-derived every shape straight from the catalog, so any
        # known-empty record for this window is superseded by what `input.csv` now
        # says. Clearing it here is what makes a forced rebuild the escape hatch
        # from a stale manifest, rather than a way to leave the manifest and
        # `input.csv` disagreeing.
        _clear_known_empty(
            run_folderpath,
            window_folder_segment(startdate, enddate, mosaic_days, bands=bands,
                                  mosaic_scheme=mosaic_scheme,
                                  collection=collection,
                                  declaration=_collections.get(collection),
                                  properties_filter=properties_filter,
                                  processing=processing),
        )
    else:
        build_shortfall_only(
            catalog_filepath=catalog_filepath, timestamp_col=timestamp_col,
            shapefilepath=shapefilepath, id_col=id_col, run_folderpath=run_folderpath,
            startdate=startdate, enddate=enddate, bands=bands,
            mosaic_days=mosaic_days,
            csv_filepath=csv_filepath, label_col=label_col, mosaic_scheme=mosaic_scheme,
            collection=collection, properties_filter=properties_filter,
            processing=processing,
        )

    if overwrite:
        _force_rebuild(csv_filepath)

    dispatch_csv_filepath, n_total, n_missing = _build_shortfall(csv_filepath, force=overwrite)
    if n_missing == 0:
        print(f"[build] 0 of {n_total} cubes missing; nothing to build", flush=True)
        return None
    if 0 < n_missing < n_total:
        print(f"[build] {n_missing} of {n_total} cubes missing; dispatching {n_missing}",
              flush=True)

    if runner == "local":
        return runners.run_local(dispatch_csv_filepath, cores=cores, dry_run=dry_run,
                                 unlock=unlock)
    if runner == "aml":
        return runners.run_aml(dispatch_csv_filepath, **(runner_kwargs or {}))
    raise ValueError(f"Unknown runner={runner!r}; valid values: 'local', 'aml'.")
