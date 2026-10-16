"""Granule naming: canonical name, acquisition key, date and processing facts per collection.

Spec: specs/59-imagery-archive-layout.md D2-D4, D8. ADR 0032.

The one entry point -- `granule_info` -- dispatches to the collection's own module
(`s2_l2a`, `s1_rtc`, `hls`). A collection with no parser (a user-registered one) gets the
"any other" row of D3/D4: its item id is the canonical name and the acquisition key, the
date is the UTC date of the item timestamp, and it publishes no processing version --
so cross-processing detection is off for it, and a version specifier against it raises.

Deliberately a dispatch table, not a field on `CollectionDeclaration`: a declaration may not
hold callables (ADR 0031). This is called only where items are discovered.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Mapping

import pandas as pd

from rise_eo.collections import hls as _hls
from rise_eo.collections import s1_rtc as _s1_rtc
from rise_eo.collections import s2_l2a as _s2_l2a

__all__ = ["GranuleInfo", "granule_info", "publishes_version", "PARSERS"]

PARSERS = {
    _s2_l2a.COLLECTION_ID: _s2_l2a,
    _s1_rtc.COLLECTION_ID: _s1_rtc,
    **{collection_id: _hls for collection_id in _hls.COLLECTION_IDS},
}


@dataclasses.dataclass(frozen=True)
class GranuleInfo:
    canonical_name: str
    acquisition_key: str
    acquisition_date: datetime.date
    processing_version: str | None
    processing_datetime: pd.Timestamp | None


def publishes_version(collection: str) -> bool:
    """Does this collection expose a processing version? `False` for S1 RTC and for a
    collection with no parser -- D6 refuses a version specifier against those."""
    parser = PARSERS.get(collection)
    return bool(parser and parser.PUBLISHES_VERSION)


def _utc_date(timestamp) -> datetime.date:
    ts = pd.Timestamp(timestamp)
    return (ts.tz_convert("UTC") if ts.tzinfo else ts).date()


def granule_info(collection: str, item_id: str, properties: Mapping, timestamp) -> GranuleInfo:
    """Everything the archive layout and the catalog need to know about one STAC item."""
    parser = PARSERS.get(collection)
    if parser is None:
        return GranuleInfo(item_id, item_id, _utc_date(timestamp), None, None)
    name = parser.canonical_name(item_id, properties)
    return GranuleInfo(
        canonical_name=name,
        acquisition_key=parser.acquisition_key(name),
        acquisition_date=parser.acquisition_date(name) or _utc_date(timestamp),
        processing_version=parser.processing_version(name, properties),
        processing_datetime=parser.processing_datetime(properties),
    )
