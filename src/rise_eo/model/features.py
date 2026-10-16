"""The single feature-transform chokepoint + per-id aggregation.

Spec: specs/18-model-adapter.md

Everything that turns raw bands into model features flows through `apply_features` — the ONE
place the transform runs, so training and inference cannot drift apart.
It works on the 5-D contract `(samples, timestamps, H, W, bands)` used by `rise_eo.bands.modify`.
"""

from __future__ import annotations

import numpy as np

from rise_eo.bands import modify

__all__ = ["apply_features", "median_per_id", "resolve_aggregate"]


def apply_features(data5d: np.ndarray, band_indices: dict, *, adapter=None,
                   feature_sequence=None):
    """Run the feature transform on a 5-D array -> `(features5d, feature_band_indices)`.

    Exactly one source of truth, in priority order:
      1. `adapter.feature_sequence` (if the adapter declares one),
      2. `adapter.features(...)` escape hatch (if `feature_sequence is None`),
      3. a raw `feature_sequence=` (adapter-less / exploratory use),
      4. identity (nothing requested).
    Passing both an `adapter` and a raw `feature_sequence` is a caller error (guarded upstream).
    """
    if adapter is not None:
        seq = getattr(adapter, "feature_sequence", None)
        if seq is not None:
            return modify.modify_bands(bands=data5d, band_indices=band_indices, sequence=seq)
        return adapter.features(data5d, band_indices)
    if feature_sequence is not None:
        return modify.modify_bands(
            bands=data5d, band_indices=band_indices, sequence=feature_sequence
        )
    return data5d, band_indices


def median_per_id(ids: np.ndarray, data: np.ndarray, labels):
    """Reduce per-pixel samples to one per `id` via `np.nanmedian` (demo_02 cell-3).

    `data` is `(pixels, T, B)`; returns `(ids, data, labels)` with one row per unique id
    (labels reduced by first-occurrence; a field's pixels share a label).
    """
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    n_ids = len(unique_ids)
    _, n_ts, n_bands = data.shape

    data_median = np.zeros((n_ids, n_ts, n_bands), dtype=data.dtype)
    labels_median = None if labels is None else np.zeros(n_ids, dtype=np.asarray(labels).dtype)
    for i in range(n_ids):
        rows = inverse == i
        data_median[i] = np.nanmedian(data[rows], axis=0)
        if labels is not None:
            labels_median[i] = np.asarray(labels)[rows][0]
    return unique_ids, data_median, labels_median


_AGGREGATES = {"median_per_id": median_per_id}


def resolve_aggregate(aggregate):
    """`None` | `"median_per_id"` | `callable(ids, data, labels)->(ids, data, labels)`."""
    if aggregate is None or callable(aggregate):
        return aggregate
    if aggregate in _AGGREGATES:
        return _AGGREGATES[aggregate]
    raise ValueError(
        f"unknown aggregate {aggregate!r}; use None, 'median_per_id', or a callable."
    )
