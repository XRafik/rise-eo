"""S2 processing-baseline -> radiometric-offset derivation, shared by CDSE + MPC.

Spec: specs/34-ingest-normalization-contract.md

The baseline property name is provider-specific — MPC's S2 STAC extension uses
`s2:processing_baseline`, while CDSE's v1 catalogue uses the generic STAC Processing
extension's `processing:version` — but the value format (`"MM.mm"`) and the semantics are
identical across both.

ESA: reflectance = (DN + offset) / QUANTIFICATION_VALUE; offset = -1000 for processing
baseline >= 04.00 (2022-01-25), else 0. (Source: the ESA S2 L2A algorithm documentation.)
"""

from __future__ import annotations

__all__ = ["baseline_tuple", "offset_for_item"]

_BASELINE_PROPS = (
    "s2:processing_baseline",  # MPC / legacy CDSE — S2 STAC extension
    "processing:version",      # CDSE STAC v1 — STAC Processing extension
)


def baseline_tuple(baseline: str) -> tuple[int, int]:
    """Parse an S2 baseline string ("04.00", "05.09", "02.14") into a
    comparable `(major, minor)` int tuple."""
    major, minor = baseline.split(".")
    return (int(major), int(minor))


def offset_for_item(item) -> int:
    """The additive reflectance-band offset for one STAC item.

    ⚠️ Keyed on **baseline**, never on acquisition date: reprocessing can stamp a >= 04.00
    baseline onto a pre-2022 acquisition, and the offset still applies.

    Resolves the baseline from the first of `_BASELINE_PROPS` present on the item, since the
    property name differs per provider while the format and semantics do not. Raises when
    none is present -- never a silent 0, because this is the correctness-critical field.
    """
    baseline = None
    for prop in _BASELINE_PROPS:
        baseline = item.properties.get(prop)
        if baseline is not None:
            break
    if baseline is None:
        raise ValueError(
            f"STAC item {item.id!r} has none of {_BASELINE_PROPS!r}; "
            "cannot derive the reflectance offset (spec 34 §3a A1)."
        )
    return -1000 if baseline_tuple(baseline) >= (4, 0) else 0
