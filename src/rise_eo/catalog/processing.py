"""Per-acquisition processing selection: the one `processing=` grammar (spec 59 D6/D7).

Spec: specs/59-imagery-archive-layout.md. ADR 0032 (the archive is lossless; duplicates
raise, they never replace).

One acquisition can be present in more than one processing (two baselines, two sources).
`select_processing` resolves that per acquisition, and it is the **only** implementation of
the ordering and the specifier grammar: `download` (D7) applies it to one provider's search
results to choose what to *fetch*; the build verbs (D6) apply it to the catalog rows a cube
would use, to choose what to *use*. Same values, same meaning; only the defaults differ.

| `processing`      | per acquisition group                                                |
|-------------------|----------------------------------------------------------------------|
| `None`            | raise if any group has more than one row                             |
| `"latest"`        | keep the latest row (`processing_version`, then `processing_datetime`) |
| a PEP 440 specifier (`">=04.00"`, `"==05.00"`) | keep rows whose version satisfies it, then as `"latest"`; a group left empty is dropped and reported |

"Latest" raises only when it must choose and cannot -- two rows that tie on both keys.
A frame is any DataFrame with `id`, `acquisition_key`, `processing_version`,
`processing_datetime`, `source`; several rows per `id` (band-flattened frames) are fine.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import pandas as pd
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from rise_eo.collections import naming

LATEST = "latest"

__all__ = [
    "LATEST", "Selection", "processing_errors", "select_processing", "print_selection",
    "ambiguity_lines", "provenance", "merge_provenance",
]


@dataclasses.dataclass
class Selection:
    """The outcome of `select_processing`: the surviving rows plus everything it skipped
    (so nothing a selection decides is silent)."""

    kept: pd.DataFrame
    skipped: list[dict] = dataclasses.field(default_factory=list)
    dropped_acquisitions: list[str] = dataclasses.field(default_factory=list)


# --- validation ---------------------------------------------------------------


def _specifier(processing: str) -> SpecifierSet:
    return SpecifierSet(processing)


def processing_errors(processing, *, collection: str, allow_none: bool) -> list[str]:
    """Preflight errors for a `processing=` value (empty list when fine).

    `allow_none=False` is `download`'s grammar (D7): `None` -- "every processing the
    provider serves" -- is refused, naming `"latest"`. A version specifier against a
    collection that publishes no version (S1 RTC) is refused, naming the collection --
    the same "don't silently no-op" rule as `max_cloudcover` against SAR.
    """
    if processing is None:
        if allow_none:
            return []
        return [
            "processing=None is not accepted on download: it would fetch every processing "
            "the provider still serves, including ones the provider itself treats as "
            f"superseded. Use processing={LATEST!r} (default) or a specifier such as "
            "'>=05.00'."
        ]
    if not isinstance(processing, str):
        return [f"processing={processing!r} must be None, {LATEST!r} or a PEP 440 "
                "specifier string such as '>=05.00'."]
    if processing == LATEST:
        return []
    try:
        _specifier(processing)
    except InvalidSpecifier:
        return [f"processing={processing!r} is not {LATEST!r} or a valid PEP 440 specifier "
                "(e.g. '>=04.00', '==05.00', '>=05.00,<06')."]
    if not naming.publishes_version(collection):
        return [f"processing={processing!r} is a version specifier, but collection "
                f"{collection!r} publishes no processing version; use processing={LATEST!r} "
                "(or None on a build) instead."]
    return []


# --- ordering -----------------------------------------------------------------


def _version(value) -> Version | None:
    if value is None or (not isinstance(value, str) and pd.isna(value)) or value == "":
        return None
    try:
        return Version(str(value))
    except InvalidVersion:
        return None


def _datetime(value) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    return ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")


_NULL_VERSION = Version("0")
_NULL_DATETIME = pd.Timestamp(0, tz="UTC")


def _order_key(row) -> tuple:
    """`processing_version` then `processing_datetime`; a null sorts below any value."""
    v = _version(row.get("processing_version"))
    d = _datetime(row.get("processing_datetime"))
    return (v is not None, v or _NULL_VERSION, d is not None, d or _NULL_DATETIME)


# --- selection ----------------------------------------------------------------


def _describe(row) -> str:
    v = row.get("processing_version")
    d = _datetime(row.get("processing_datetime"))
    return (f"{row['id']}  source={row.get('source') or '?'}  "
            f"processing_version={None if _version(v) is None else v}  "
            f"processing_datetime={None if d is None else d.strftime('%Y-%m-%dT%H:%M:%SZ')}")


def _groups(unique: pd.DataFrame) -> dict[str, list]:
    """Acquisition key -> rows (dicts). A row with no key is its own group."""
    groups: dict[str, list] = {}
    for _, row in unique.iterrows():
        key = row.get("acquisition_key")
        if key is None or (not isinstance(key, str) and pd.isna(key)) or key == "":
            key = str(row["id"])
        groups.setdefault(str(key), []).append(row)
    return groups


def select_processing(
    df: pd.DataFrame, processing: str | None, *, resolve_hint: str = "",
) -> Selection:
    """Apply `processing` per acquisition group. Raises `ValueError` for `None` over a
    duplicate group, and for `"latest"`/a specifier when the top of a group is a tie.

    `resolve_hint` is appended to the duplicate-group error: how the caller (a build verb)
    spells the arguments that would resolve it.
    """
    if len(df) == 0 or "id" not in df.columns:
        return Selection(df)
    unique = df.drop_duplicates(subset="id")
    groups = _groups(unique)

    if processing is None:
        dupes = {k: rows for k, rows in groups.items() if len(rows) > 1}
        if dupes:
            raise ValueError(_duplicates_message(dupes, resolve_hint))
        return Selection(df)

    spec = None if processing == LATEST else _specifier(processing)
    keep_ids: set = set()
    skipped: list[dict] = []
    dropped: list[str] = []
    for key, rows in groups.items():
        candidates = rows
        if spec is not None:
            candidates = []
            for row in rows:
                v = _version(row.get("processing_version"))
                if v is not None and spec.contains(v, prereleases=True):
                    candidates.append(row)
                else:
                    skipped.append({
                        "id": row["id"], "acquisition_key": key,
                        "processing_version": row.get("processing_version"),
                        "reason": f"version does not satisfy {processing!r}",
                    })
            if not candidates:
                dropped.append(key)
                continue
        ranked = sorted(candidates, key=_order_key, reverse=True)
        best = ranked[0]
        tied = [r for r in ranked if _order_key(r) == _order_key(best)]
        if len(tied) > 1:
            raise ValueError(
                f"processing={processing!r} cannot choose between {len(tied)} rows of "
                f"acquisition {key!r}: they tie on processing_version and "
                "processing_datetime (S1 RTC, for one, carries neither).\n"
                + "\n".join("  " + _describe(r) for r in tied)
            )
        keep_ids.add(best["id"])
        for row in ranked[1:]:
            skipped.append({
                "id": row["id"], "acquisition_key": key,
                "processing_version": row.get("processing_version"),
                "reason": f"superseded by {best['id']}",
            })
    kept = df[df["id"].isin(keep_ids)]
    return Selection(kept, skipped, dropped)


def _duplicates_message(dupes: dict[str, list], resolve_hint: str) -> str:
    lines = [f"{len(dupes)} acquisition(s) hold more than one processing in the rows this "
             "build would mosaic (processing=None):"]
    for key, rows in sorted(dupes.items()):
        lines.append(f"  acquisition {key}:")
        lines += ["    " + _describe(r) for r in rows]
    lines.append(
        "Resolve with processing='latest' (newest per acquisition) or a specifier such as "
        "processing='>=05.00' / '==05.00'; or narrow the rows with "
        "properties_filter={'source': 'mpc'} / {'processing_version': '05.00'}."
        + (f" {resolve_hint}" if resolve_hint else "")
    )
    return "\n".join(lines)


def print_selection(sel: Selection, *, prefix: str, processing: str | None,
                    emit: Callable[[str], None] | None = None) -> None:
    """Print every choice a selection made (spec 59 D7: nothing a download decides is
    silent). No output when it skipped nothing."""
    emit = emit or (lambda s: print(s, flush=True))
    for s in sel.skipped:
        emit(f"{prefix} skipped {s['id']} (processing_version={s['processing_version']}): "
             f"{s['reason']}")
    for key in sel.dropped_acquisitions:
        emit(f"{prefix} dropped acquisition {key}: no processing satisfies "
             f"processing={processing!r}")


# --- reporting ------------------------------------------------------------------


def ambiguity_lines(catalog_gdf: pd.DataFrame, keys, *, n_matched: int, processing: str,
                    source: str) -> list[str]:
    """D7's post-download report. Always the summary line; the second line only when at
    least one touched acquisition now holds more than one processing in the archive."""
    keys = set(keys)
    lines = [f"[download] {len(keys)} acquisitions; {n_matched} matched "
             f"processing={processing!r} at {source}"]
    if len(catalog_gdf) == 0 or "acquisition_key" not in catalog_gdf.columns:
        return lines
    touched = catalog_gdf[catalog_gdf["acquisition_key"].isin(keys)]
    ambiguous = touched.groupby("acquisition_key")["id"].nunique()
    ambiguous = ambiguous[ambiguous > 1].index
    if len(ambiguous) == 0:
        return lines
    seen: list[str] = []
    for _, row in touched[touched["acquisition_key"].isin(ambiguous)].iterrows():
        for src in str(row.get("source") or "").split(","):
            label = f"{src} {row.get('processing_version')}"
            if src and label not in seen:
                seen.append(label)
    lines.append(
        f"[download] archive now holds >1 processing for {len(ambiguous)} of them "
        f"({', '.join(sorted(seen))});\n"
        "           builds over them need processing='latest' or a specifier"
    )
    return lines


def merge_provenance(blocks, *, with_ids: bool) -> dict:
    """Union D9 provenance blocks (one per cube) into one -- the training-data stamp's
    block, which keeps versions and sources only (`with_ids=False`: a training set spans
    thousands of granules)."""
    merged: dict[str, dict[str, set]] = {}
    for block in blocks:
        for collection, b in (block or {}).items():
            cur = merged.setdefault(collection, {"processing_version": set(), "source": set(),
                                                 "ids": set()})
            for key in cur:
                cur[key] |= set(b.get(key, []))
    return {
        collection: {k: sorted(v) for k, v in cur.items() if k != "ids" or with_ids}
        for collection, cur in merged.items()
    }


def provenance(df: pd.DataFrame, *, collection: str, with_ids: bool) -> dict:
    """D9's provenance block for the rows a cube (or a training set) used."""
    if len(df) == 0:
        return {}
    unique = df.drop_duplicates(subset="id")
    versions = sorted({str(v) for v in unique.get("processing_version", []) if _version(v)})
    sources: set[str] = set()
    for value in unique.get("source", []):
        sources.update(part for part in str(value or "").split(",") if part)
    block: dict = {}
    if with_ids:
        block["ids"] = sorted(str(i) for i in unique["id"])
    block["processing_version"] = versions
    block["source"] = sorted(sources)
    return {collection: block}
