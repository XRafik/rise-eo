"""The `sentinel-2-l2a` collection declaration -- the only one P1 ships.

Spec: specs/58-collection-agnostic-verbs.md. Granule naming: specs/59 D3/D4/D8.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Mapping

import pandas as pd

from rise_eo.catalog.declaration import S2_L2A_DECLARATION

COLLECTION_ID = "sentinel-2-l2a"

# The declaration itself is defined in `rise_eo.catalog.declaration` (not re-defined here) to
# avoid a declaration<->collections import cycle: `builder._resolve_declaration`'s
# hand-built-GeoDataFrame fallback needs a concrete S2 default without importing the
# registry package.
DECLARATION = S2_L2A_DECLARATION


# --- granule naming (spec 59 D3/D4/D8) -----------------------------------------
#
# Plain functions, NOT fields on `CollectionDeclaration`: a declaration may not hold
# callables (ADR 0031). They are called only where items are discovered -- on the driver
# for MPC, inside the single CDSE job for CDSE -- so they are ordinary installed rise-eo code,
# never a registry lookup on a node.

PUBLISHES_VERSION = True

# ESA product name, e.g. S2B_MSIL2A_20180918T100019_N0212_R122_T33UWP_20201009T023142
_NAME_RE = re.compile(
    r"^(?P<sat>S2[ABC])_(?P<level>MSIL2A)_(?P<sense>\d{8}T\d{6})_N(?P<baseline>\d{4})_"
    r"(?P<orbit>R\d{3})_(?P<tile>T\d{2}[A-Z]{3})_(?P<disc>\d{8}T\d{6})$"
)


def _parse(name: str) -> re.Match:
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"not an ESA Sentinel-2 L2A product name: {name!r}")
    return m


def canonical_name(item_id: str, properties: Mapping) -> str:
    """The ESA product name without `.SAFE` (spec 59 D3).

    MPC's item id drops the `N0xxx` baseline field, so the name comes from
    `s2:product_uri`; CDSE's id already is the ESA name. An item that offers neither
    **raises** -- never a fallback to the provider id, which would silently break the
    cross-source equality this name exists for (spec 59 §6).
    """
    uri = properties.get("s2:product_uri")
    if uri:
        return str(uri).removesuffix(".SAFE")
    if _NAME_RE.match(item_id):
        return item_id
    raise ValueError(
        f"sentinel-2-l2a item {item_id!r} has no 's2:product_uri' and its id is not a full "
        "ESA product name (no baseline field); cannot derive the canonical granule name "
        "(spec 59 D3)."
    )


def acquisition_key(name: str) -> str:
    """`S2B_MSIL2A_20180918T100019_R122_T33UWP` -- drops `_N0212` and the discriminator."""
    m = _parse(name)
    return f"{m['sat']}_{m['level']}_{m['sense']}_{m['orbit']}_{m['tile']}"


def acquisition_date(name: str) -> datetime.date:
    return datetime.datetime.strptime(_parse(name)["sense"][:8], "%Y%m%d").date()


def processing_version(name: str, properties: Mapping) -> str | None:
    """`"05.00"`-style baseline: MPC `s2:processing_baseline`, CDSE `processing:version`,
    else the name's own `N0500` field."""
    raw = properties.get("s2:processing_baseline") or properties.get("processing:version")
    if raw:
        major, _, minor = str(raw).partition(".")
        return f"{int(major):02d}.{int(minor or 0):02d}"
    baseline = _parse(name)["baseline"]
    return f"{baseline[:2]}.{baseline[2:]}"


def processing_datetime(properties: Mapping) -> pd.Timestamp | None:
    """MPC `s2:generation_time`, CDSE `processing:datetime` (UTC); `None` when neither
    exists."""
    raw = properties.get("s2:generation_time") or properties.get("processing:datetime")
    if not raw:
        return None
    ts = pd.Timestamp(raw)
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
