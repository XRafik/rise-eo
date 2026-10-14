"""TileCatalog — read/append/filter the downloaded-granule catalog.

Spec: specs/02-catalog.md

Columns: id (unique; the **canonical granule name** since spec 59 D3), acquisition_key
(identity of the physical observation, spec 59 D4), processing_version / processing_datetime
(spec 59 D8; null when the provider exposes none), source (comma-joined sorted set of the
sources that contributed files, unioned on append), collection (STAC collection id, e.g. "sentinel-2-l2a" -- spec 58 D12;
renamed from "satellite", which always held this), timestamp (UTC), s3url,
local_folderpath, files (comma-joined band filenames), cloud_cover, offset (the additive
declared radiometric offset for radiometry bands; required, 0 when a collection has no such
concept -- spec 59 A3), scale (the declared multiplicative radiometric scale; filled from the
declaration when a row has none, 1.0 with no declaration -- spec 58 D5.1, spec 59 A3; declared
metadata only, never applied to pixels), nodata (the declared nodata value; required -- spec 59
A3), properties (JSON, the source item's STAC properties
verbatim -- spec 58 D12; carries facts like `sat:orbit_state` the D9 partition guard
needs, and nothing fsd interprets directly belongs here instead of a first-class column),
geometry (EPSG:4326).

⚠️ **No read-time back-compat shim (spec 58 D12, standing policy).** A catalog written
before this schema change is not patched up here -- re-download and re-ingest, never
silently default a missing column (see `read()`).
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping, Sequence

import geopandas as gpd
import pandas as pd
import shapely

from fsd.catalog import declaration as declaration_module
from fsd.catalog.declaration import CollectionDeclaration
from fsd.storage import fs

# On-disk column order. geometry is always last for GeoParquet.
COLUMNS = [
    "id",
    "collection",
    "timestamp",
    "s3url",
    "local_folderpath",
    "files",
    "cloud_cover",
    "offset",
    "scale",
    "nodata",
    "acquisition_key",
    "processing_version",
    "processing_datetime",
    "source",
    "properties",
    "geometry",
]

CRS = "EPSG:4326"

# The columns spec 59 D8 added; a catalog lacking them predates the archive layout and is
# refused on read (older schema gaps keep their own failure at the point of use).
SPEC59_COLUMNS = ("acquisition_key", "processing_version", "processing_datetime", "source")

# Spec 59 D6: `properties_filter` names that resolve to a first-class column, never to a
# same-named key inside the `properties` JSON.
RESERVED_FILTER_COLUMNS = ("source", "processing_version")


# Column defaults for a *fresh* `TileCatalog.append`, for a column the rows do not carry.
#
# `properties` is a per-row value a source may leave out; it defaults to "{}".
# `offset` and `nodata` have NO default (spec 59 A3 / X4.1): `append` refuses a row without
# them. `scale` is filled from the declaration, not from this table (X4.2).
#
# Spec 59 D8: the sources always supply `acquisition_key`, `processing_version`,
# `processing_datetime` and `source`; a hand-built row (a test, an fsd-agnostic top-up) that
# does not is its own acquisition (`acquisition_key` = its id, set in `append`) with no
# version and no source -- the "any other" collection row of D3/D4.
_APPEND_DEFAULTS = {
    "properties": "{}",
    "processing_version": None,
    "processing_datetime": pd.NaT,
    "source": "",
}


def _union_files(*files_values: str) -> str:
    """Union comma-joined band-filename lists into one sorted, deduped string
    (also used for the comma-joined `source` set -- spec 59 D8)."""
    names: set[str] = set()
    for value in files_values:
        if value:
            names.update(part for part in str(value).split(",") if part)
    return ",".join(sorted(names))


def filter_gdf(
    gdf: gpd.GeoDataFrame,
    shapes_gdf: gpd.GeoDataFrame,
    startdate: datetime.datetime,
    enddate: datetime.datetime,
) -> gpd.GeoDataFrame:
    """Date-range (inclusive) + spatial-overlap filter over an **already-read** catalog.

    The pure half of `TileCatalog.filter`, split out so a caller filtering *many*
    shapes against *one* catalog can read the file once instead of once per shape.
    That is not a micro-optimisation on a remote catalog: `create_datacube.setup`
    over 900 shapes was 900 full downloads of the same `abfss://` parquet (~106 MiB
    of redundant transfer, ~900 round-trips) because `filter` re-read the file on
    every call.

    Does not mutate `gdf` — the returned slice is a `.copy()` before the
    `area_contribution` column is added, so one read is safely shared across calls.
    `.attrs` (the spec-35 declaration stamp) propagates to the slice through pandas'
    `__finalize__`, exactly as it did when each call re-read the file.
    """
    startdate = pd.to_datetime(startdate, utc=True)
    enddate = pd.to_datetime(enddate, utc=True)

    in_range = gdf[(gdf["timestamp"] >= startdate) & (gdf["timestamp"] <= enddate)]

    # ROI union in the catalog CRS.
    union_shape = shapely.unary_union(shapes_gdf.to_crs(gdf.crs)["geometry"])

    overlapping = in_range[in_range.intersects(union_shape)].copy()

    union_area = union_shape.area
    overlapping["area_contribution"] = overlapping["geometry"].apply(
        lambda g: g.intersection(union_shape).area / union_area * 100
    )

    return overlapping


def properties_filter_values(want) -> list[str]:
    """The requested value(s) for ONE `properties_filter` key, as a list of strings.

    A bare scalar is one value, never an iterable to be unpacked: D9 names
    `sat:orbit_state` (a string) AND `sat:relative_orbit`, which is an **integer** in
    the STAC `sat` extension — `list(146)` raises `TypeError`, and `list("descending")`
    would silently become ten one-character values.

    Comparison is on the string form of both sides, so `146` and `"146"` select the same
    rows and canonicalize to the same digest rather than filtering silently to zero (D9.1).
    """
    values = list(want) if isinstance(want, (list, tuple, set, frozenset)) else [want]
    return [str(v) for v in values]


def filter_by_properties(
    gdf: gpd.GeoDataFrame,
    properties_filter: Mapping[str, str | Sequence[str]] | None,
) -> gpd.GeoDataFrame:
    """Keep only rows whose `properties` JSON matches every key in `properties_filter`
    (spec 58 D9, part 1) -- e.g. `{"sat:orbit_state": "descending"}`. A value may be a
    single string or a sequence of acceptable values.

    A key that **no row in `gdf` carries at all** raises, naming the keys the catalog
    DOES carry -- filtering silently to zero rows is the same "user concludes their ROI
    has no coverage" failure the D9 partition guard exists to prevent, from the other
    direction. A key that some rows carry and others don't, or whose requested value(s)
    happen to match no row, is an ordinary (possibly empty) filter result, not an error.

    `None`/empty `properties_filter` is a no-op — returns `gdf` unchanged, not a copy.
    """
    if not properties_filter:
        return gdf
    parsed = [json.loads(p) if isinstance(p, str) and p else {}
              for p in gdf["properties"]]
    carried_keys: set[str] = set()
    for props in parsed:
        carried_keys.update(props)
    # `source` / `processing_version` are first-class columns (spec 59 D6): reserved, so
    # they resolve to the column even when `properties` has a same-named key, and are
    # always "carried" for the unknown-key check.
    unknown = sorted(k for k in properties_filter
                     if k not in carried_keys and k not in RESERVED_FILTER_COLUMNS)
    if unknown:
        raise ValueError(
            f"properties_filter key(s) {unknown} are not carried by any row in this "
            f"catalog; keys this catalog carries: "
            f"{sorted(carried_keys | set(RESERVED_FILTER_COLUMNS))}."
        )
    mask = []
    for i, props in enumerate(parsed):
        keep = True
        for key, want in properties_filter.items():
            wanted = properties_filter_values(want)
            if key in RESERVED_FILTER_COLUMNS:
                cell = gdf[key].iloc[i] if key in gdf.columns else None
                have = [] if cell is None or (not isinstance(cell, str) and pd.isna(cell)) \
                    else str(cell).split(",") if key == "source" else [str(cell)]
                ok = any(v in have for v in wanted)
            else:
                ok = key in props and str(props[key]) in wanted
            if not ok:
                keep = False
                break
        mask.append(keep)
    return gdf[mask]


class TileCatalog:
    def __init__(self, filepath: str, declaration: CollectionDeclaration | None = None):
        self.filepath = filepath
        # `append`'s default when its own `declaration=` kwarg is None.
        self._declaration_default = declaration

    def _existing_stamp(self) -> CollectionDeclaration | None:
        """The declaration actually stamped on the on-disk file, or `None` if the
        file doesn't exist or carries no stamp (footer-only, cheap)."""
        if not fs.exists(self.filepath):
            return None
        raw = fs.peek_parquet_attrs(self.filepath).get(declaration_module.ATTRS_KEY)
        return declaration_module.from_json(raw) if raw is not None else None

    @property
    def declaration(self) -> CollectionDeclaration | None:
        """The declaration a build against this catalog would resolve to right
        now: the on-disk stamp if the file exists, else the constructor default
."""
        if fs.exists(self.filepath):
            return self._existing_stamp()
        return self._declaration_default

    def append(self, rows: list[dict], declaration: CollectionDeclaration | None = None) -> None:
        """Upsert by id; union `files` for an existing granule (don't overwrite).

        A re-download of more bands extends the recorded `files` list rather than
        replacing it; all other columns take the newest value.

        `declaration` stamps the collection-level `CollectionDeclaration`
        on this catalog file (constructor's `declaration=` is the default when this
        kwarg is `None`). One catalog file = one collection = one declaration:
        appending a declaration that differs from the one already stamped on an
        existing catalog raises `ValueError`; appending with `declaration=None` to
        an already-stamped catalog preserves the existing stamp (an fsd-agnostic
        top-up cannot erase it).

        Spec 59 A3: every row must carry a non-empty `offset` and `nodata` (an S2 row at
        baseline >= 04.00 needs `offset_for_item`'s -1000; use 0 for a collection with no
        offset). A row without one raises `ValueError` before any file is read or written.
        `scale` is filled from the effective declaration's `scale` (the `declaration=`
        argument, else the constructor's, else the existing file's stamp; 1.0 with none) when the column is missing or a row's cell is empty.
        """
        if not rows:
            return

        effective = declaration if declaration is not None else self._declaration_default

        new = gpd.GeoDataFrame(rows, crs=CRS)
        for column in ("offset", "nodata"):
            if column not in new.columns:
                lacking = list(new["id"])
            else:
                lacking = list(new.loc[new[column].isna(), "id"])
            if lacking:
                raise ValueError(
                    f"TileCatalog.append: {column!r} is missing or empty in {len(lacking)} "
                    f"row(s), ids {lacking[:5]!r}{' ...' if len(lacking) > 5 else ''} "
                    "(spec 59 A3). Pass it explicitly: for Sentinel-2, "
                    "`offset_for_item`'s value; 0 for a collection with no offset/nodata."
                )
        # The file's footer is read only here, after the offset/nodata refusal above.
        scale_source = effective if effective is not None else self._existing_stamp()
        fill_scale = scale_source.scale if scale_source is not None else 1.0
        if "scale" not in new.columns:
            new["scale"] = fill_scale
        else:
            new["scale"] = new["scale"].astype("float64").fillna(fill_scale)
        # Normalize timestamp to tz-aware UTC for a stable on-disk dtype.
        new["timestamp"] = pd.to_datetime(new["timestamp"], utc=True)
        for column, default in _APPEND_DEFAULTS.items():
            if column not in new.columns:
                new[column] = default
        # Spec 59 D8: a hand-built row with no `acquisition_key` is its own acquisition.
        if "acquisition_key" not in new.columns:
            new["acquisition_key"] = new["id"]
        new["processing_datetime"] = pd.to_datetime(new["processing_datetime"], utc=True)

        if fs.exists(self.filepath):
            existing_stamp = self._existing_stamp()
            if effective is not None and existing_stamp is not None and effective != existing_stamp:
                raise ValueError(
                    f"TileCatalog.append: declaration conflict at {self.filepath!r} -- "
                    f"existing stamp {existing_stamp!r} != new {effective!r}. One "
                    "catalog file is one collection with one declaration; write the "
                    "conflicting rows to a different catalog file instead."
                )
            stamp = effective if effective is not None else existing_stamp
            existing = self.read()
            combined = pd.concat([existing, new], ignore_index=True)
        else:
            stamp = effective
            combined = new

        # Union `files` across rows sharing an id (oldest..newest order).
        merged_files = combined.groupby("id")["files"].agg(
            lambda s: _union_files(*s.tolist())
        )

        # `source` is a set of the sources that contributed files, unioned like `files`
        # (spec 59 D8): the same canonical name means the same processing, so the bytes
        # are interchangeable and the catalog need not say which band came from where.
        merged_sources = combined.groupby("id")["source"].agg(
            lambda s: _union_files(*s.tolist())
        )

        # Keep the last (newest) row per id for every other column...
        deduped = combined.drop_duplicates(subset="id", keep="last").set_index("id")
        # ...then overwrite `files` and `source` with the unioned values.
        deduped["files"] = merged_files
        deduped["source"] = merged_sources
        out = deduped.reset_index()

        out = gpd.GeoDataFrame(out[COLUMNS], geometry="geometry", crs=CRS)
        if stamp is not None:
            declaration_module.to_attrs(out, stamp)
        fs.write_parquet(self.filepath, out)

    def read(self) -> gpd.GeoDataFrame:
        """Return the full catalog as a GeoDataFrame.

        ⚠️ **No back-compat shim, deliberately.** A catalog predating the
        `offset`/`nodata` columns is NOT patched up here. It is disposable and must be
        re-ingested, never silently defaulted: `read()` raises `ValueError` on a
        pre-spec-59 schema (`SPEC59_COLUMNS`). It does not check `offset`/`nodata`: a hand-built
        file without them passes, and `flatten_catalog` then raises `KeyError` (spec 59 A3).
        """
        gdf = fs.read_parquet(self.filepath)
        missing = [c for c in SPEC59_COLUMNS if c not in gdf.columns]
        if missing:
            raise ValueError(
                f"TileCatalog.read: {self.filepath!r} predates the current catalog schema "
                f"(missing column(s) {missing}; spec 59 D8 / spec 58 D12). There is no "
                "compatibility shim -- re-download into the new archive layout "
                "({root}/{collection}/YYYY/MM/DD/{granule}/)."
            )
        gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], utc=True)
        return gdf

    def to_stac(self, dst_folderpath: str, **kwargs) -> str:
        """Export the catalog as a static, self-contained STAC catalog.

        Additive interchange view — the GeoParquet stays the query format. Returns the
        catalog.json path. See `fsd.catalog.stac`.
        """
        from fsd.catalog import stac

        kwargs.setdefault("declaration", self.declaration)
        items = stac.tile_catalog_to_items(self.read(), **kwargs)
        return stac.write_stac_catalog(items, dst_folderpath, declaration=self.declaration)

    def filter(
        self,
        shapes_gdf: gpd.GeoDataFrame,
        startdate: datetime.datetime,
        enddate: datetime.datetime,
    ) -> gpd.GeoDataFrame:
        """Date-range (inclusive) + spatial-overlap filter against the ROI union.

        Adds `area_contribution` (% of the ROI union each granule covers). This is
        exactly the query the datacube builder consumes.

        Reads the catalog file on **every** call. Filtering many shapes against one
        catalog should read once and call `filter_gdf` per shape instead.
        """
        return filter_gdf(self.read(), shapes_gdf, startdate, enddate)
