"""Granule naming for Harmonized Landsat Sentinel-2 (`hls2-s30`, `hls2-l30`).

Spec: specs/59-imagery-archive-layout.md D2-D4. No `CollectionDeclaration` is registered
for HLS yet (spec 58 P3); this module only fixes how its granules are named and filed, so
the layout does not have to change when it lands.

An HLS id is `HLS.S30.T33UWP.2021123T100031.v2.0`: the sensing time is `YYYYDDDThhmmss`
(**day of year**), the trailing `.v2.0` is the processing version. STAC `datetime` can
fall on a different UTC date from the id's sensing time, so the date level comes from the
id (spec 59 D2).
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Mapping

import pandas as pd

COLLECTION_IDS = ("hls2-s30", "hls2-l30")
PUBLISHES_VERSION = True

_NAME_RE = re.compile(
    r"^(?P<prefix>HLS\.[SL]30\.T\d{2}[A-Z]{3})\.(?P<year>\d{4})(?P<doy>\d{3})T(?P<time>\d{6})"
    r"\.v(?P<version>\d+\.\d+)$"
)


def _parse(name: str) -> re.Match:
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"not an HLS granule id: {name!r}")
    return m


def canonical_name(item_id: str, properties: Mapping) -> str:
    _parse(item_id)
    return item_id


def acquisition_key(name: str) -> str:
    m = _parse(name)
    return f"{m['prefix']}.{m['year']}{m['doy']}T{m['time']}"


def acquisition_date(name: str) -> datetime.date:
    m = _parse(name)
    year, doy = int(m["year"]), int(m["doy"])
    date = datetime.date(year, 1, 1) + datetime.timedelta(days=doy - 1)
    if doy < 1 or date.year != year:        # day 000, or day 366 of a non-leap year
        raise ValueError(f"HLS granule id {name!r}: day of year {m['doy']} is not in {year}")
    return date


def processing_version(name: str, properties: Mapping) -> str | None:
    return _parse(name)["version"]


def processing_datetime(properties: Mapping) -> pd.Timestamp | None:
    return None
