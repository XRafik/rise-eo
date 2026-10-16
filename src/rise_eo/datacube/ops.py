"""Datacube ops — pure transforms over (t, H, W, bands) + metadata.

Spec: specs/04-datacube-ops.md. Folds in the L2A-relevant subset of
core/datacube_ops.py. Each op: (datacube, metadata, **kwargs) -> (datacube, metadata),
so they compose via `run_ops`.

Dropped vs legacy: run_s2cloudless / CMK-based apply_cloud_mask (L1C-only) +
run_s2cloudless_core[_chunkwise] (s2cloudless dependency). See LIMITATIONS.md, "Legacy".
"""

from __future__ import annotations

import copy
import datetime

import numba
import numpy as np
import pandas as pd

from rise_eo import config

MOSAIC_SCHEMES = ("calendar", "acquisition")


def run_ops(datacube, metadata, sequence):
    """Run a sequence of (func, kwargs) ops, threading (datacube, metadata)."""
    for func, kwargs in sequence:
        datacube, metadata = func(datacube=datacube, metadata=metadata, **kwargs)
    return datacube, metadata


def apply_cloud_mask_scl(datacube, metadata, *, mask_classes, mask_band="SCL",
                         mask_value=0):
    """Set pixels to `mask_value` where the categorical mask band (`mask_band`,
    default `"SCL"`) has a value in `mask_classes`, across every
    non-mask band. The mask band itself is left untouched (drop it separately).

    `mask_band` generalizes this beyond S2 (`mask_type="categorical_classes"`) — any
    source whose QA band encodes
    "masked" as a small set of discrete values (not just SCL) can reuse this op
    by declaring its own `mask_band`."""
    band_indices = {band: i for i, band in enumerate(metadata["bands"])}
    if mask_band not in band_indices:
        raise ValueError(f"{mask_band} band not present in datacube")

    idx_to_modify = [i for b, i in band_indices.items() if b != mask_band]

    mask_col = datacube[:, :, :, band_indices[mask_band]]
    selected = datacube[:, :, :, idx_to_modify]
    selected[np.where(np.isin(mask_col, mask_classes))] = mask_value
    datacube[:, :, :, idx_to_modify] = selected
    return datacube, metadata


def drop_bands(datacube, metadata, *, bands_to_drop):
    """Slice out `bands_to_drop`; update `metadata['bands']`."""
    band_indices = {band: i for i, band in enumerate(metadata["bands"])}
    bands_to_keep = [b for b in band_indices if b not in bands_to_drop]
    idx_to_keep = [band_indices[b] for b in bands_to_keep]
    datacube = datacube[:, :, :, idx_to_keep]
    metadata["bands"] = bands_to_keep
    return datacube, metadata


def median_mosaic(datacube, metadata, *, startdate: datetime.datetime,
                  enddate: datetime.datetime, mosaic_days=20, mask_value=0,
                  mosaic_scheme: str | None = None):
    """Bucket timestamps into `mosaic_days` windows; per-bucket nanmedian (treating
    `mask_value` as NaN). Numba-accelerated core.

    `mosaic_scheme` (default `config.MOSAIC_SCHEME` = "calendar"):

    - **"calendar"** — fixed calendar windows `[startdate + k*mosaic_days, ...)` over
      `[startdate, enddate)`; label = the window-start boundary; **every** window is
      emitted (an empty window → an all-`mask_value` slice). The output `timestamps`
      are thus a pure function of `(startdate, enddate, mosaic_days)`, identical across
      any two datacubes built with those inputs — so `flatten` can concatenate cubes
      from different MGRS tiles/orbits/zones.

      How the window vs the revisit cadence interact:
        * `mosaic_days` > cadence  → several acquisitions per window get composited
          (`nanmedian`) into one slice — the normal temporal-mosaic case.
        * `mosaic_days` ≈ cadence  → ~one acquisition per window (median of one = itself).
        * `mosaic_days` < cadence  → each acquisition still lands alone in its window
          (unchanged, nothing lost), but the windows *between* acquisitions have no data
          and are emitted as all-`mask_value` slices. The cube degenerates to the raw
          time series laid on a fixed calendar grid, padded with empty slices — correct
          and cross-shape-consistent, but larger. Choose `mosaic_days` ≥ the revisit.
    - **"acquisition"** — legacy behavior: windows track the actual acquisition dates
      (a gap opens a new interval), labels = first acquisition per window, empty windows
      skipped. Anchor shifts ROI-to-ROI; kept for reproducing pre-spec-15 outputs.
    """
    if mosaic_scheme is None:
        mosaic_scheme = config.MOSAIC_SCHEME
    if mosaic_scheme not in MOSAIC_SCHEMES:
        raise ValueError(f"mosaic_scheme must be one of {MOSAIC_SCHEMES}")
    if mosaic_days < 1:
        return datacube, metadata

    timestamps = metadata["timestamps"]
    if mosaic_scheme == "calendar":
        labels, index_intervals = _calendar_windows(
            timestamps=timestamps, startdate=startdate, enddate=enddate,
            mosaic_days=mosaic_days,
        )
    else:  # "acquisition"
        index_intervals = _get_mosaic_ts_index_ranges(
            timestamps=timestamps, startdate=startdate, enddate=enddate,
            mosaic_days=mosaic_days,
        )
        labels = [timestamps[r[0]] for r in index_intervals]

    dtype = datacube.dtype
    datacube = datacube.astype(float)
    datacube[np.where(datacube == mask_value)] = np.nan

    mosaiced = _mosaic_from_intervals(datacube, index_intervals, mask_value)
    mosaiced = mosaiced.astype(dtype)

    md = copy.deepcopy(metadata)
    md["mosaic_index_intervals"] = index_intervals
    md["previous_timestamps"] = timestamps
    md["timestamps"] = labels
    md["data_shape_desc"] = ("timestamps", "height", "width", "bands")
    return mosaiced, md


def area_median(datacube, metadata=None, *, mask_value=0):
    """Collapse H×W to a single median pixel per timestamp (deploy helper)."""
    dtype = datacube.dtype
    _, height, width, _ = datacube.shape

    datacube = datacube.astype(float)
    datacube[np.where(datacube == mask_value)] = np.nan
    out = np.expand_dims(np.nanmedian(datacube, axis=(1, 2)), axis=(1, 2))
    out[np.isnan(out)] = mask_value
    out = out.astype(dtype)

    md = None
    if metadata is not None:
        md = copy.deepcopy(metadata)
        md["previous_height_width"] = (height, width)
    return out, md


# --- private helpers (faithful port; preserve legacy bucket behavior) ---------

def _dt2ts(dt: datetime.datetime, tz="UTC") -> pd.Timestamp:
    """Localize a naive datetime to a tz-aware pd.Timestamp (default UTC).

    The catalog stores tz-aware (UTC) acquisition timestamps, but user
    startdate/enddate arrive tz-naive. Comparison ops (== < > >=) raise when
    mixing tz-aware and tz-naive, so we attach UTC here. Attaching the tz is
    easier on a pd.Timestamp than on a datetime; hour/minute granularity is
    irrelevant since S2 revisit is ~5 days.
    """
    if dt.tzinfo is None:
        return pd.Timestamp(dt, tz=tz)
    return pd.Timestamp(dt)


def _is_sorted(seq) -> bool:
    return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))


def _checked_bounds(timestamps, startdate, enddate):
    """`(startdate, enddate)` as tz-aware `pd.Timestamp`s, after checking that `timestamps`
    is sorted and lies within them."""
    startdate = _dt2ts(startdate)
    enddate = _dt2ts(enddate)

    if not _is_sorted(timestamps):
        raise ValueError("timestamps is not sorted.")
    if startdate > timestamps[0]:
        raise ValueError("startdate must be before or on the first timestamp")
    if enddate < timestamps[-1]:
        raise ValueError("enddate must be after or on the last timestamp")
    return startdate, enddate


def _fixed_windows(startdate, enddate, mosaic_days):
    """The `(lo, hi)` pairs of `mosaic_days`-long windows tiling `[startdate, enddate)`;
    the last one may run past `enddate`."""
    windows = []
    lo = startdate
    while lo < enddate:
        windows.append((lo, lo + datetime.timedelta(days=mosaic_days)))
        lo = windows[-1][1]
    return windows


def _get_mosaic_ts_index_ranges(timestamps, startdate, enddate, mosaic_days=20):
    """Map sorted `timestamps` to (start_idx, end_idx) ranges over `mosaic_days`
    windows anchored at `startdate`. Faithful legacy port, including its quirk that a
    gap opens a new interval (advancing one bucket), so ranges track occupied buckets
    rather than every fixed window."""
    startdate, enddate = _checked_bounds(timestamps, startdate, enddate)

    mosaic_buckets = _fixed_windows(startdate, enddate, mosaic_days)

    cur = 0
    ts_indexes = [[]]
    for index, ts in enumerate(timestamps):
        lte_date = mosaic_buckets[cur][1]
        if ts <= lte_date:
            ts_indexes[cur].append(index)
        else:
            ts_indexes.append([index])
            cur += 1

    return [(min(g), max(g)) for g in ts_indexes if len(g) > 0]


def _calendar_windows(timestamps, startdate, enddate, mosaic_days=20):
    """Fixed calendar windows `[lo, lo+mosaic_days)` tiling `[startdate, enddate)`
. Returns `(labels, index_intervals)`:

    - `labels[k]` = the window-start `pd.Timestamp` (tz-aware UTC).
    - `index_intervals[k]` = `(min_i, max_i)` of the sorted `timestamps` falling in
      window k, or `None` when the window is empty.

    Windows are half-open `[lo, hi)` so a timestamp on a boundary lands in the *later*
    window; the final window is upper-inclusive so a timestamp exactly at `enddate`
    (allowed by the guard) is not dropped. Data-independent: the labels/count depend
    only on `(startdate, enddate, mosaic_days)`."""
    startdate, enddate = _checked_bounds(timestamps, startdate, enddate)

    windows = _fixed_windows(startdate, enddate, mosaic_days)

    labels, intervals = [], []
    for k, (lo, hi) in enumerate(windows):
        last = k == len(windows) - 1
        members = [i for i, ts in enumerate(timestamps)
                   if lo <= ts and (ts <= hi if last else ts < hi)]
        labels.append(lo)
        intervals.append((min(members), max(members)) if members else None)
    return labels, intervals


def _mosaic_from_intervals(datacube, index_intervals, mask_value):
    """Median-mosaic each interval into one slice, in order. A `None` interval (an
    empty calendar window) becomes an all-`mask_value` slice, so the output timestamp
    axis has one slot per interval. `datacube` already has `mask_value` set to NaN;
    an all-NaN result (fully-masked pixel or empty window) is restored to `mask_value`.
    Non-empty intervals go through the single numba core call (legacy path)."""
    n_out = len(index_intervals)
    _, height, width, n_bands = datacube.shape
    out = np.full((n_out, height, width, n_bands), np.nan)

    nonempty = [(k, iv) for k, iv in enumerate(index_intervals) if iv is not None]
    if nonempty:
        positions = np.array([k for k, _ in nonempty])
        ranges = np.array([iv for _, iv in nonempty])
        out[positions] = _median_mosaic_core(datacube, ranges)

    out[np.isnan(out)] = mask_value
    return out


@numba.njit()
def _median_mosaic_core(datacube: np.ndarray, ts_index_ranges: np.ndarray):
    n_ts, height, width, n_bands = datacube.shape
    n_mosaiced = ts_index_ranges.shape[0]
    out = np.full((n_mosaiced, height, width, n_bands), np.nan)
    for t in numba.prange(n_mosaiced):
        for h in numba.prange(height):
            for w in numba.prange(width):
                for b in numba.prange(n_bands):
                    out[t, h, w, b] = np.nanmedian(
                        datacube[ts_index_ranges[t][0]: ts_index_ranges[t][1] + 1,
                                 h, w, b]
                    )
    return out
