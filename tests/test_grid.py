"""Tests for ROI → S2-grid tiling (spec 19). Synthetic, no network.

Skips when the optional `[grid]` extra (`s2`/`s2cell`) isn't installed — e.g. in rise-eo's lean
`.venv`; run these from the `.venv-modeldeploy` that has `.[grid]`.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
import shapely.geometry

from rise_eo import grid

pytest.importorskip("s2", reason="needs the [grid] extra (s2 + s2cell)")


def test_grid_size_to_res_5km_is_11():
    # 5 km falls in the res-11 range (3–5 km).
    assert grid.grid_size_to_res(5) == 11
    assert grid.grid_size_to_res(1000) == 3     # 840–1167 km


def _roi(minx=36.1, miny=11.4, maxx=36.9, maxy=12.0):
    return gpd.GeoDataFrame(
        geometry=[shapely.geometry.box(minx, miny, maxx, maxy)], crs="EPSG:4326"
    )


def test_roi_to_s2_grids_clipped_contained():
    roi = _roi()
    grids = grid.roi_to_s2_grids(roi, grid_size_km=5, scale_fact=1.1)

    assert isinstance(grids, gpd.GeoDataFrame)
    assert len(grids) > 0
    assert grids.crs.to_epsg() == 4326
    assert "id" in grids.columns and "geometry" in grids.columns

    roi_geom = roi.geometry.iloc[0]
    # every clipped grid intersects the ROI and (clip=True) is contained within it.
    assert grids.geometry.intersects(roi_geom).all()
    assert grids.geometry.apply(lambda g: roi_geom.buffer(1e-9).contains(g)).all()


def test_scale_fact_enlarges_unclipped_cells():
    roi = _roi()
    small = grid.roi_to_s2_grids(roi, scale_fact=1.0, clip=False)
    big = grid.roi_to_s2_grids(roi, scale_fact=1.3, clip=False)
    # same cell count (same polyfill), but scaled cells are larger on average.
    assert len(small) == len(big) > 0
    assert big.geometry.area.mean() > small.geometry.area.mean()


def test_deterministic_count():
    roi = _roi()
    a = grid.roi_to_s2_grids(roi)
    b = grid.roi_to_s2_grids(roi)
    assert len(a) == len(b)
    assert sorted(a["id"]) == sorted(b["id"])


# --- spec 21 D-GRID-1: an ROI is one region, so ids are unique ----------------


def _scattered_roi(n=40):
    """A multi-polygon ROI: n small squares scattered inside one 5 km cell's footprint --
    the shape of a field collection (e.g. AT_2018_TRAIN's 900 EuroCrops fields)."""
    step = 0.004
    boxes = [
        shapely.geometry.box(36.10 + (i % 8) * step, 11.40 + (i // 8) * step,
                             36.10 + (i % 8) * step + step / 2,
                             11.40 + (i // 8) * step + step / 2)
        for i in range(n)
    ]
    return gpd.GeoDataFrame(geometry=boxes, crs="EPSG:4326")


def test_multipolygon_roi_yields_one_row_per_cell_with_unique_ids():
    """The 2026-07-28 bug: clipping with `gpd.overlay(grids, roi_gdf)` emitted one row per
    (cell x roi-polygon) PAIR, so a 900-field ROI produced 1167 rows for 172 cells -- and
    `id` is the work-unit key, so N rows sharing an id meant N tasks writing the same
    export folder concurrently (`InvalidBlockList` on blob). Clipping to the ROI's UNION
    keeps it one row per cell however many polygons come in."""
    roi = _scattered_roi(40)
    grids = grid.roi_to_s2_grids(roi, grid_size_km=5, scale_fact=1.1)

    assert len(grids) > 0
    assert not grids["id"].duplicated().any()
    assert len(grids) == grids["id"].nunique()
    # and the cell count is a property of the REGION, not of how many polygons describe it
    assert len(grids) <= len(grid.roi_to_s2_grids(roi, grid_size_km=5, scale_fact=1.1,
                                                 clip=False))


def test_multipolygon_roi_clips_to_the_union_not_the_bounding_shape():
    """Clipping to the union must still *clip* -- the returned cells stay inside the ROI
    (this is what distinguishes it from `clip=False`), and cover the same area the ROI's
    polygons do within those cells."""
    roi = _scattered_roi(40)
    union = roi.geometry.union_all()
    grids = grid.roi_to_s2_grids(roi, grid_size_km=5, scale_fact=1.1)

    assert grids.geometry.apply(lambda g: union.buffer(1e-9).contains(g)).all()
    # nothing dropped: the clipped cells together cover the whole ROI
    assert grids.geometry.union_all().area == pytest.approx(union.area, rel=1e-6)


def test_single_polygon_roi_is_unchanged_by_the_union_clip():
    """Regression guard for the fix itself: a single-polygon ROI (the case runbook 38
    Phase 1 ran GREEN on) must tile exactly as before."""
    roi = _roi()
    grids = grid.roi_to_s2_grids(roi, grid_size_km=5, scale_fact=1.1)
    roi_geom = roi.geometry.iloc[0]

    assert not grids["id"].duplicated().any()
    assert grids.geometry.apply(lambda g: roi_geom.buffer(1e-9).contains(g)).all()
    assert grids.geometry.union_all().area == pytest.approx(roi_geom.area, rel=1e-6)


# --- spec 46 D4/D5: drop cells already covered by another cell (#69) ---------


def _single_s2_cell_polygon(res: int = 11):
    """One S2 cell's OWN (unscaled) footprint -- using it as an ROI reproduces the
    measured #69 defect: `polyfill` runs on the ROI's convex hull, so a cell's 8
    neighbours come back too, and after scale+clip they land as slivers wholly inside
    the ROI. Picks the polyfilled cell with the largest area so its own footprint sits
    away from the seed box's edge (interior, not edge-clipped)."""
    from s2 import s2

    seed = shapely.geometry.box(15.30, 48.40, 15.60, 48.60)
    cells = s2.polyfill(geo_json=shapely.geometry.mapping(seed), res=res,
                         geo_json_conformant=True, with_id=True)
    import pandas as pd
    df = pd.DataFrame(cells)
    df["geometry"] = df["geometry"].apply(shapely.geometry.Polygon)
    row = df.loc[df["geometry"].apply(lambda g: g.area).idxmax()]
    return row["id"], row["geometry"]


def test_roi_that_is_exactly_one_s2_cell_collapses_to_that_one_cell():
    """AC3 (synthetic form of the measured 476da24 case, which lives outside this repo
    -- shapefiles/s2grid=476da24.geojson): polyfilling a single cell's own footprint
    re-discovers its 8 neighbours (convex hull + intersects), all 8 of which are fully
    covered by the central, scaled+clipped cell once dedup runs -- 9 -> 1."""
    cell_id, cell_geom = _single_s2_cell_polygon()
    roi = gpd.GeoDataFrame(geometry=[cell_geom], crs="EPSG:4326")

    grids = grid.roi_to_s2_grids(roi, grid_size_km=5, scale_fact=1.1)

    assert len(grids) == 1
    assert grids["id"].iloc[0] == cell_id


def test_drop_covered_cells_preserves_the_union():
    """AC4's safety argument, exercised directly: whatever `_drop_covered_cells` drops,
    the union of what remains must equal the union of what came in (a dropped cell is a
    geometric subset of a kept one) -- built from a covering big cell plus several
    smaller cells fully inside it, mirroring the real 9-cells-1-covers-8 shape."""
    big = shapely.geometry.box(0, 0, 10, 10)
    small_a = shapely.geometry.box(1, 1, 2, 2)
    small_b = shapely.geometry.box(3, 3, 4, 4)      # shares no boundary with `big`'s edge
    small_c = shapely.geometry.box(0, 0, 1, 1)      # shares boundary with `big` (corner)
    grids = gpd.GeoDataFrame(
        {"id": ["big", "small_a", "small_b", "small_c"]},
        geometry=[big, small_a, small_b, small_c], crs="EPSG:4326",
    )
    before_union = grids.geometry.union_all()

    out = grid._drop_covered_cells(grids)

    assert list(out["id"]) == ["big"]
    assert out.geometry.union_all().area == pytest.approx(before_union.area, rel=1e-9)


def test_drop_covered_cells_tie_breaks_mutually_equal_cells_deterministically():
    """Two identical cells must collapse to exactly one, keeping the smaller id --
    without the tie-break a naive one-pass 'covered_by something else -> drop' rule
    drops BOTH, since each covers the other."""
    square = shapely.geometry.box(0, 0, 1, 1)
    grids = gpd.GeoDataFrame({"id": ["b_dup", "a_dup"]},
                              geometry=[square, shapely.geometry.box(0, 0, 1, 1)],
                              crs="EPSG:4326")

    out = grid._drop_covered_cells(grids)

    assert len(out) == 1
    assert out["id"].iloc[0] == "a_dup"

    # order-independence: same result with the rows the other way round.
    grids_swapped = gpd.GeoDataFrame({"id": ["a_dup", "b_dup"]},
                                      geometry=[square, shapely.geometry.box(0, 0, 1, 1)],
                                      crs="EPSG:4326")
    out_swapped = grid._drop_covered_cells(grids_swapped)
    assert len(out_swapped) == 1
    assert out_swapped["id"].iloc[0] == "a_dup"


def test_drop_covered_cells_keeps_non_overlapping_cells():
    """The dedup must not touch cells that are genuinely distinct (no false positives)."""
    grids = gpd.GeoDataFrame(
        {"id": ["left", "right"]},
        geometry=[shapely.geometry.box(0, 0, 1, 1), shapely.geometry.box(5, 5, 6, 6)],
        crs="EPSG:4326",
    )
    out = grid._drop_covered_cells(grids)
    assert len(out) == 2
    assert set(out["id"]) == {"left", "right"}
