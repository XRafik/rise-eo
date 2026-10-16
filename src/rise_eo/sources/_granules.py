"""Shared by the MPC and CDSE sources: where a granule lives, and the catalog columns that
describe it.

Spec: specs/59-imagery-archive-layout.md D2/D3/D8. Both sources write the same layout,
`{root}/{collection}/YYYY/MM/DD/{canonical granule name}/`, so the same granule from either
source lands in the same folder (and the same catalog row).
"""

from __future__ import annotations

import os

from rise_eo.catalog import catalog as catalog_module
from rise_eo.catalog import processing as processing_module
from rise_eo.collections.naming import GranuleInfo, granule_info
from rise_eo.storage import fs

__all__ = [
    "item_granule", "granule_folderpath", "granule_columns", "collection_root",
    "require_valid_processing", "select_granules", "surviving_items", "report_download",
]


def item_granule(collection: str, item) -> GranuleInfo:
    """`granule_info` for a (duck-typed) STAC item."""
    return granule_info(collection, item.id, item.properties, item.datetime)


def collection_root(root_folderpath: str, collection: str) -> str:
    """`{root}/{collection}` -- the directory one catalog.parquet lives in (D2/D5)."""
    return os.path.join(root_folderpath, collection)


def granule_folderpath(root_folderpath: str, collection: str, info: GranuleInfo) -> str:
    d = info.acquisition_date
    return os.path.join(
        collection_root(root_folderpath, collection),
        f"{d.year:04d}", f"{d.month:02d}", f"{d.day:02d}", info.canonical_name,
    )


def granule_columns(info: GranuleInfo, *, source: str) -> dict:
    """The D8 catalog columns for one discovered granule."""
    return {
        "id": info.canonical_name,
        "acquisition_key": info.acquisition_key,
        "processing_version": info.processing_version,
        "processing_datetime": info.processing_datetime,
        "source": source,
    }


# --- download(processing=): the source-side half of spec 59 D7 ----------------------------


def require_valid_processing(processing, collection: str) -> None:
    """Source-level guard for `processing=` (D7): the verbs preflight it, but a direct
    `mpc.download` / `discover_shard_rows` / `cdse.download` caller must not get a silent
    no-op."""
    errs = processing_module.processing_errors(
        processing, collection=collection, allow_none=False,
    )
    if errs:
        raise ValueError("; ".join(errs))


def select_granules(granules, *, processing: str, prefix: str, properties_filter=None):
    """`properties_filter`, then D7's per-acquisition `processing` selection over one
    provider's search results -- both before the `max_tiles` cap, so the cap counts what
    will actually transfer. Every skipped granule is printed under `prefix`."""
    granules = catalog_module.filter_by_properties(granules, properties_filter)
    selection = processing_module.select_processing(granules, processing)
    processing_module.print_selection(selection, prefix=prefix, processing=processing)
    return selection.kept


def surviving_items(items, granule_ids, collection: str) -> list[tuple]:
    """`[(item, canonical_name), ...]` for the items whose canonical granule name survived
    selection (the catalog `id` is the canonical name, not the provider's item id)."""
    pairs = ((it, item_granule(collection, it).canonical_name) for it in items)
    return [(it, name) for it, name in pairs if name in granule_ids]


def report_download(catalog, granules, *, processing: str, source: str) -> None:
    """D7's post-download report: how many acquisitions this download touched, and whether
    any now holds more than one processing in the archive."""
    if len(granules) == 0 or not fs.exists(catalog.filepath):
        return
    for line in processing_module.ambiguity_lines(
        catalog.read(), set(granules["acquisition_key"]), n_matched=len(granules),
        processing=processing, source=source,
    ):
        print(line, flush=True)
