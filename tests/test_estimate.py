"""Offline test for the no-download ETA estimator (spec 23 §7/SO-8).

`demos/estimate.py` is not an installed package, so load it by path. Only `estimate_from_counts`
(pure math) is exercised — `estimate_run`'s live counts need STAC/grid and are not unit-tested.
"""

from __future__ import annotations

import importlib.util
import os

_ESTIMATE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demos", "estimate.py")
_spec = importlib.util.spec_from_file_location("rise_eo_demo_estimate", _ESTIMATE)
estimate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(estimate)


_COST_MODEL = {
    "transfer_mb_per_s": 5.0,
    "mean_bytes_by_band": {"B04": 10e6, "B08": 10e6, "B8A": 2.5e6, "SCL": 2.5e6},
    "convert_s_per_file": 2.0,
    "build_s_per_cube": 3.0,
    "infer_s_per_cube": 1.0,
    "t_calib": 9,
}


def test_estimate_from_counts_math():
    est = estimate.estimate_from_counts(
        granules=100, cells=50, t=9, bands=["B04", "B08", "B8A", "SCL"], cost_model=_COST_MODEL,
    )
    # bytes = 100 * (10+10+2.5+2.5)MB = 2500 MB = 2.5 GB
    assert est["gb"] == 2.5
    assert est["band_files"] == 400
    # download = 2500 MB / 5 MB/s = 500 s = 8.3 min
    assert est["download_min"] == round(500 / 60, 1)
    # convert = 400 files * 2 s = 800 s = 13.3 min
    assert est["convert_min"] == round(800 / 60, 1)
    # compute = 50 cells * (3+1) s * (9/9) = 200 s = 3.3 min
    assert est["compute_min"] == round(200 / 60, 1)
    assert est["total_min"] == round((500 + 800 + 200) / 60, 1)


def test_estimate_scales_compute_linearly_in_t():
    # 15 cells x 4 s/cube = 60 s at t=9 (1.0 min); at t=18 -> 120 s (2.0 min) — clean doubling.
    base = estimate.estimate_from_counts(granules=10, cells=15, t=9, bands=["B04"],
                                         cost_model=_COST_MODEL)
    doubled = estimate.estimate_from_counts(granules=10, cells=15, t=18, bands=["B04"],
                                            cost_model=_COST_MODEL)
    assert (base["compute_min"], doubled["compute_min"]) == (1.0, 2.0)


def test_estimate_missing_costs_read_zero():
    est = estimate.estimate_from_counts(granules=5, cells=5, t=4, bands=["B04"], cost_model={})
    assert est["gb"] == 0.0
    assert est["download_min"] is None      # no throughput -> unknown, not zero
    assert est["convert_min"] == 0.0 and est["compute_min"] == 0.0
