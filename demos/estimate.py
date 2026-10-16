"""No-download ETA estimator (spec 23 §7, D12) — the team/demo helper that answers
"how long would region X / window Y / bands Z take?" **without downloading it.**

Runtime = counts × per-unit-costs. The counts are cheap *exact* queries (CDSE STAC for granules,
`rise_eo.grid` for cells, `compute_n_timestamps` for T); the `cost_model` is calibrated by a real
run's `timings.json`. `estimate_from_counts` is the pure math (offline, unit-tested);
`estimate_run` gathers the live counts then calls it. Kept demo-level for now — may graduate to a
first-class `rise_eo.estimate` verb (D12). Caveats (see the doc §9): throughput is environment-bound,
bytes/granule vary at ROI edges, and the estimate carries a low/high band from probe-vs-effective.
"""

from __future__ import annotations


def estimate_from_counts(*, granules: int, cells: int, t: int, bands: list[str],
                         cost_model: dict) -> dict:
    """Pure math: counts × a calibrated `cost_model` -> {counts, GB, per-phase + total minutes}.

    `cost_model` keys (all optional; missing -> that phase reads 0):
      transfer_mb_per_s, mean_bytes_by_band {band: bytes}, convert_s_per_file,
      build_s_per_cube, infer_s_per_cube, t_calib (the T the per-cube costs were measured at).
    """
    mean_by_band = cost_model.get("mean_bytes_by_band") or {}
    band_files = granules * len(bands)
    total_bytes = granules * sum(mean_by_band.get(b, 0) for b in bands)

    mbps = cost_model.get("transfer_mb_per_s") or 0.0
    download_s = (total_bytes / 1e6 / mbps) if mbps else None
    convert_s = band_files * (cost_model.get("convert_s_per_file") or 0.0)

    t_calib = cost_model.get("t_calib") or t or 1
    per_cube = (cost_model.get("build_s_per_cube") or 0.0) + (cost_model.get("infer_s_per_cube") or 0.0)
    compute_s = cells * per_cube * (t / t_calib if t_calib else 1.0)

    total_s = (download_s or 0.0) + convert_s + compute_s
    return {
        "granules": granules, "band_files": band_files, "cells": cells, "t": t,
        "gb": round(total_bytes / 1e9, 2),
        "download_min": round(download_s / 60, 1) if download_s is not None else None,
        "convert_min": round(convert_s / 60, 1),
        "compute_min": round(compute_s / 60, 1),
        "total_min": round(total_s / 60, 1),
    }


def estimate_run(roi, startdate, enddate, bands, *, creds, cost_model,
                 max_cloudcover=None, mosaic_days=20, grid_size_km=5, scale_fact=1.1) -> dict:
    """Live wrapper: gather the exact counts (no download) then `estimate_from_counts`.

    granules = STAC query count; cells = `roi_to_s2_grids` count; T = `compute_n_timestamps`.
    Needs the `[grid]` extra (for `rise_eo.grid`) and network (anonymous STAC).
    """
    import rise_eo
    from rise_eo import grid
    from rise_eo.sources import cdse

    granules = len(cdse.query_catalog(roi, startdate, enddate, max_cloudcover=max_cloudcover))
    cells = len(grid.roi_to_s2_grids(roi, grid_size_km=grid_size_km, scale_fact=scale_fact))
    t = rise_eo.compute_n_timestamps(startdate, enddate, mosaic_days)
    out = estimate_from_counts(granules=granules, cells=cells, t=t, bands=bands,
                               cost_model=cost_model)
    out["region"] = roi if isinstance(roi, str) else "<GeoDataFrame>"
    return out
