# Adding a new source to fsd

Spec: `specs/34-ingest-normalization-contract.md`. This is the "how do I add a source"
doc that spec 34 §2d requires — read it before implementing a new `rise_eo.sources.*`
module or wiring a new declaration into `build_datacube`. It has two halves: **ingest**
(getting bytes onto storage as a self-describing artifact) and **builder** (what
`build_datacube` reads off that artifact, and where).

There is **no `Source` ABC yet** (that formalization is TODO #11) — a source today is
just a Python module with a `query_catalog(...)` / `download(...)` pair matching the
documented CDSE/MPC contracts (`fsd/specs/01-sources.md`) plus a `CollectionDeclaration`
(below). This doc seeds what #11 will eventually codify as an interface.

## The two halves of the source contract

### 1. Ingest: `stage → normalize → put`

Every source's `download(...)` does three things, in order, for each unit of work
(one tile/granule):

1. **stage** — fetch the source's native bytes (a signed HTTPS href, an S3 object, a
   netCDF file, …) to somewhere the next step can read them.
2. **normalize** — the *only* source-specific step, and the only place a new source
   needs real code:
   - **container transform** (if the native format isn't already a COG): e.g. CDSE's
     jp2→COG (`rise_eo.raster.cog.to_cog`), a future ERA5 netCDF→COG. Skip if the source is
     already COG (MPC).
   - **radiometry/metadata declaration**: stamp the artifact's GDAL scale/offset tag +
     nodata-if-missing (`rise_eo.raster.cog.stamp_gdal_tags` / `stamp_or_reencode`) — this
     is what makes the artifact *self-describing* instead of relying on ambient config.
     `stamp_or_reencode` falls back to a GDAL-COG-driver re-encode if the in-place tag
     edit breaks COG validity (a GDAL/source-dependent edge case, not the common path).
3. **put** — write the normalized artifact to `root_folderpath` (local or blob; **into
   `{root}/{collection}/YYYY/MM/DD/{canonical granule name}/`, spec 59 D2/D3** — build the
   path with `rise_eo.sources._granules.granule_folderpath` and get the name, acquisition key
   and processing facts from `rise_eo.collections.naming.granule_info`, never from the provider
   id; **stamp, then publish** — stage as `<dst>.stage`, stamp it there, `os.replace` it onto
   `<dst>`, so a final-named file is always a stamped one, spec 59 D10;
   `rise_eo.storage.fs.transfer`/`fs.put`) and append a row to the `TileCatalog` (below),
   **stamping the source's `CollectionDeclaration` at that `catalog.append(...)` call**
   (spec 35 §4) — this is a **required** step, not optional: `TileCatalog.append(rows,
   declaration=YOUR_DECLARATION)`. Without it, the collection-level declaration is never
   persisted (spec 35 §1/§2), and every downstream build against that catalog raises
   (§5a) rather than silently guessing S2. See `sources/cdse.py`/`sources/mpc.py` for the
   two-line pattern (`catalog.append(rows, declaration=S2_L2A_DECLARATION)`).

**Radiometry is metadata, never baked into pixels** (spec 34 §1): the on-disk artifact
stays raw DN; a per-tile additive `offset` and the source's `nodata` value are declared
alongside it. Do not write a source that "corrects" pixel values on ingest — that is the
lossy mistake spec 34 fixed (`apply_boa_offset`'s old `clip(DN-1000, 0, 65535)` on the
*stored* bytes).

### 2. The catalog row (`rise_eo.catalog.catalog.TileCatalog`)

Every source's `download(...)` appends rows shaped like `rise_eo.catalog.catalog.COLUMNS`:

| Column | Meaning | Who sets it |
|---|---|---|
| `id` | the **canonical granule name** (spec 59 D3): every processing field kept, every provider-specific one dropped, so the same granule from two sources is one row | source, via `collections.naming` |
| `acquisition_key` | the canonical name minus its processing fields — the physical observation (D4); detects duplicates, never names a folder | source, via `collections.naming` |
| `processing_version` / `processing_datetime` | e.g. `"05.00"` / UTC generation time; null when the provider exposes none (D8) | source |
| `source` | comma-joined sorted set of sources that contributed files; `TileCatalog.append` unions it like `files` | source (`"mpc"`, `"cdse"`) |
| `collection` | STAC collection id | source |
| `timestamp` | acquisition time (UTC) | source |
| `s3url` | informational source href | source |
| `local_folderpath` | where the artifact(s) landed | source (`put` step) |
| `files` | comma-joined filenames in that folder | source |
| `cloud_cover` | optional QA metric | source (0 if N/A) |
| `offset` | additive radiometric offset for reflectance bands (spec 34 §1) | source; **required** — pass 0 for a source with no such concept; `TileCatalog.append` refuses a row without it (spec 59 A3) |
| `nodata` | declared nodata value | source; **required** — `append` refuses a row without it (spec 59 A3). 0 is the S2 convention; do not assume 0 for an arbitrary new source (spec 34 §1c, ClearSKY warning; see #181) |
| `geometry` | footprint, EPSG:4326 | source |

A collection with no parser in `fsd/collections/` gets the "any other" row: its item id is
the canonical name and the acquisition key, it publishes no version, and cross-processing
detection is off for it. A catalog missing the four spec 59 columns is refused on read
(`re-download`; no shim).

`offset`/`nodata` retire spec 32's bespoke `boa_add_offset` column — **there is no
back-compat shim** (spec 34 `[G4]`): `TileCatalog.read()` does not backfill a legacy
catalog missing these columns. A source that predates this schema is disposable —
re-ingest, don't migrate.

### 3. The builder contract: `CollectionDeclaration` (`rise_eo.catalog.declaration`)

`build_datacube` (`rise_eo.datacube.builder`) has **no `if source == ...`** anywhere. It
reads a `CollectionDeclaration` — the collection-level facts it needs — resolved as
(spec 35 §5): the explicit `declaration=` kwarg to `build_datacube`, else
`catalog_subset`'s own stamp (`attrs["fsd:declaration"]`, restored by `fs.read_parquet`
from the catalog Parquet's footer and set on the flattened output by
`flatten_catalog(catalog_gdf, declaration=...)`), else the S2 L2A default **for a
hand-built `catalog_gdf`** — a `catalog_gdf` that came from a file
(`fs.read_parquet`) and carries no stamp raises instead (spec 35 §5a): the artifact
must self-describe, not be guessed at. This is what step 3 above's required stamping
step exists for — see `specs/35-declaration-persistence.md` for the full design and
`rise_eo.catalog.restamp_cli` for re-stamping a catalog written before this contract.

This is the field-by-field table (spec 34 §2a) — what the builder reads, and where:

| Field | Meaning | S2 L2A default | Where it's read from |
|---|---|---|---|
| `reference_band` | the band whose grid every other band resamples onto | `"B08"` | `CollectionDeclaration.reference_band` |
| `native_grid` | `True` = one native global/regional grid; skip the multi-tile CRS-collapse | `False` | `CollectionDeclaration.native_grid` — **`True` raises `NotImplementedError`** today (spec 34 `[G2]`: the non-tiled build path ships with the ERA5/CHIRPS spec, not this one) |
| `mask_spec` | which band is the mask/QA band, how to interpret it, which values mean "masked" | `MaskSpec(band="SCL", mask_type="categorical_classes", classes=SCL_MASK_CLASSES)` | `CollectionDeclaration.mask_spec` (`None` = no mask) |
| `mask_spec.mask_type` | how to interpret the mask band | `"categorical_classes"` | **only `"categorical_classes"` is implemented** — anything else raises `NotImplementedError` (spec 34 `[G3]`) |
| `mask_keep` | keep the mask band in the output cube instead of dropping it | `False` | `CollectionDeclaration.mask_keep` |
| per-row `offset` | additive radiometric offset, applied read-time before the mosaic | S2 baseline-derived | catalog column, required: `TileCatalog.append` refuses a row without it (spec 59 A3); carried by `flatten_catalog` |
| per-row `nodata` | the missing-data sentinel | `0` | catalog column, required: `TileCatalog.append` refuses a row without it (spec 59 A3). `CollectionDeclaration.nodata` is only the fallback for a hand-built band-flattened frame passed straight to `build_datacube` |
| `mosaic_method` | how images in a window are combined | `"median"` | `CollectionDeclaration.mosaic_method` (only `"median"` is implemented; a declared-but-unimplemented value is a seam for a future op, not silently ignored — see `ops.py` if you add one) |

**The mask is opt-out per build, not per source**: if `mask_spec.band` is not in the
`bands` list you pass to `build_datacube`, the mask/drop ops are skipped entirely —
this is what lets a no-mask build (`bands=["B04"]`) work even against a declaration
that *has* a mask (spec 34 #35).

## Pointing a run at another collection is not a rename

The verbs take `collection=`, but changing it and `bands=` is not enough. Read the declaration
first (`rise_eo.collections.get(id)`). Four things break when a sentinel-2-l2a run is pointed at
`sentinel-1-rtc` (found 2026-09-12):

1. **One catalog file holds one collection.** `TileCatalog.append` raises a *declaration conflict*
   when rows of a second collection go into an existing catalog, so a new collection needs its own
   archive root.
2. **`max_cloudcover` raises** on a collection that declares `supports_cloud_cover=False`. Pass
   `None`.
3. **`properties_filter` is required** when the declaration sets `mosaic_partition` with
   `partition_policy="raise"` (S1: one `sat:orbit_state` per window).
4. **The feature sequence may stop making sense.** `rise_eo.bands.modify.compute_bands` knows only
   optical indices (NDVI, SAVI, … on B02–B12) and raises when their bands are missing. The silent
   ones are `scale_bands` and `mask_invalid_and_interpolate` given a band list: they act only on
   names they find, so S2 band names do nothing on an S1 cube and nothing raises.

In a notebook or demo, derive every collection-dependent constant from one `COLLECTION` name in a
single block, rather than editing literals in place.

## Docstring DoD

Every declaration field is documented at its definition in
`fsd/catalog/declaration.py`; the op-assembly logic in `build_datacube`'s docstring
says, for each resolved value, "reads X from Y, falls back to Z." A reviewer/newcomer
should be able to populate a new source's `CollectionDeclaration` and ingest path from
those docstrings + this doc, without reading spec 34 itself.

## Worked example: a CHIRPS-like source (single-band, no mask, native grid)

CHIRPS (daily precipitation, global native lat/lon grid) is the shape spec 34 designed
the contract *for*, but does not ship code for (that's a later, real spec — see
`fsd/specs/34-ingest-normalization-contract.md` §3's ERA5 row). Sketch of what it would
look like, to make the contract concrete:

```python
# fsd/sources/chirps.py  (illustrative — not implemented)

from rise_eo.catalog.declaration import CollectionDeclaration

CHIRPS_DECLARATION = CollectionDeclaration(
    reference_band=None,   # no resample reference: one native grid
    native_grid=True,       # -> build_datacube raises NotImplementedError today
    mask_spec=None,         # no mask/QA band at all
    nodata=-9999,           # CHIRPS's own declared nodata convention, NOT 0
    mosaic_method="median",
)


def download(roi, startdate, enddate, bands, root_folderpath, catalog, *, max_tiles, **kw):
    # stage: fetch a daily netCDF/GeoTIFF from the CHIRPS archive
    # normalize (container): netCDF -> COG (if not already a GeoTIFF), via a
    #   to_cog-style conversion (no jp2 involved, but the same "get a lossless COG"
    #   goal as CDSE)
    # normalize (radiometry): CHIRPS has no per-item offset -- stamp offset=0, scale=1,
    #   and nodata=-9999 if the source file doesn't already declare it
    # put: fs.transfer/fs.put to root_folderpath; catalog.append([{..., "offset": 0,
    #   "nodata": -9999, ...}], declaration=CHIRPS_DECLARATION)  # spec 35 §4 -- required
    ...
```

Calling `build_datacube(..., bands=["precip"], declaration=CHIRPS_DECLARATION)` on a
catalog flattened with this declaration would, today, raise `NotImplementedError`
immediately (`native_grid=True`) — a loud, documented "not yet" rather than a silent
mis-collapse through the S2-tiled multi-CRS machinery, which assumes tiles. The mask
side of the contract, though, is already exercised: a hypothetical single-band,
**tiled** no-mask source (drop `native_grid=True`) would build today, because #35's
mask-opt-out has real test coverage (`tests/test_datacube_builder.py`).

## Pointer to the future `Source` ABC (#11)

TODO #11 tracks formalizing this contract as a real `rise_eo.sources.base.Source` ABC
(`query_catalog`, `download`, a declared `CollectionDeclaration` property) instead of the
current "a module that happens to match the documented signature." That work should
start from this doc's field tables — they are the interface #11 will type-check.
