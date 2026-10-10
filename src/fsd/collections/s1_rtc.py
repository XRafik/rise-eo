"""The `sentinel-1-rtc` collection declaration.

Values verified against the live MPC `sentinel-1-rtc` collection and item JSONs.

Spec: specs/58-collection-agnostic-verbs.md D17. ADR 0028 (why RTC, not GRD).
Granule naming: specs/59 D3/D4/D8.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Mapping

import pandas as pd

from fsd.catalog.declaration import CollectionDeclaration

COLLECTION_ID = "sentinel-1-rtc"

DECLARATION = CollectionDeclaration(
    # VV and VH are both 10 m -- nothing to resample (D11).
    reference_band=None,
    # Scene-based, like S2, not one native global grid.
    native_grid=False,
    # SAR has no cloud/QA band; mask + drop steps are skipped (#35).
    mask_spec=None,
    mask_keep=False,
    # Declared in every RTC `raster:bands`.
    nodata=-32768,
    mosaic_method="median",
    # Gamma naught is already calibrated linear power, not scaled DN.
    scale=1.0,
    # Empty, not None -- None means "every band carries radiometry"; no S1 band does.
    radiometry_bands=(),
    # EO common_names are optical; SAR polarizations have no canonical alias (D8 N/A).
    band_aliases=(),
    # D10, retracted: MPC dropped the RTC key requirement in 2024.
    requires_subscription_key=False,
    # Gated by D6; AC14 -- SAR has no cloud-cover concept at all.
    supports_cloud_cover=False,
    # D9: ascending/descending backscatter must never be medianed together.
    mosaic_partition=("sat:orbit_state",),
    partition_policy="raise",
)


# --- granule naming (spec 59 D3/D4/D8) -----------------------------------------

# MPC publishes no version or processing datetime for RTC (spec 59 §9), so D6's
# version specifier is refused for this collection.
PUBLISHES_VERSION = False

# S1B_IW_GRDH_1SDV_20180626T165009_20180626T165034_011547_015393_rtc
_DATE_RE = re.compile(r"_(\d{8})T\d{6}_\d{8}T\d{6}_[0-9A-Fa-f]{6}_[0-9A-Fa-f]{6}(?:_rtc)?$")


def canonical_name(item_id: str, properties: Mapping) -> str:
    return item_id


def acquisition_key(name: str) -> str:
    return name.removesuffix("_rtc")


def acquisition_date(name: str) -> datetime.date | None:
    """The sensing-start date in the name, or `None` when the name does not carry one
    (the caller then falls back to the catalog `timestamp`)."""
    m = _DATE_RE.search(name)
    return datetime.datetime.strptime(m.group(1), "%Y%m%d").date() if m else None


def processing_version(name: str, properties: Mapping) -> str | None:
    return None


def processing_datetime(properties: Mapping) -> pd.Timestamp | None:
    return None
