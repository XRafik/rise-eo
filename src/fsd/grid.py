"""ROI → S2-geometry grid tiling.

Spec: specs/21-roi-inference-verb.md

Cover a region of interest with fixed-size S2 cells — one cell = one inference datacube = one
task in `run_inference(roi=…)`. Cells are scaled up slightly so adjacent cells overlap (no
seams at mosaic time) and clipped to the ROI so they don't spill outside it.

Needs the optional `[grid]` extra (`pip install -e ".[grid]"`: `s2` + `s2cell`) — kept out of
fsd core so the base install stays lean.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import shapely.affinity
import shapely.geometry
from shapely import STRtree
from shapely.ops import unary_union

from fsd.storage import fs

__all__ = ["roi_to_s2_grids", "grid_size_to_res", "RES_TO_KM_RANGE"]

# S2 cell edge-length range (km) per level — from s2geometry.io/resources/s2cell_statistics
# (carried wholesale from the legacy reference). Used to map a target grid size → S2 level.
RES_TO_KM_RANGE = {
    0: (7842, 7842), 1: (3921, 5004), 2: (1825, 2489), 3: (840, 1167),
    4: (432, 609), 5: (210, 298), 6: (108, 151), 7: (54, 76),
    8: (27, 38), 9: (14, 19), 10: (7, 9), 11: (3, 5),
    12: (1.699, 2), 13: (0.850, 1.185), 14: (0.425, 0.593), 15: (0.212, 0.296),
}


def grid_size_to_res(grid_size_km: float) -> int:
    """Nearest S2 level whose cell edge-length range brackets `grid_size_km` (5 km → 11)."""
    best_res, best_diff = None, None
    for res, (lo, hi) in RES_TO_KM_RANGE.items():
        for km in (lo, hi):
            diff = abs(km - grid_size_km)
            if best_diff is None or diff < best_diff:
                best_res, best_diff = res, diff
    return best_res


def _as_gdf_4326(roi) -> gpd.GeoDataFrame:
    """Accept a GeoDataFrame, a file path, or a geojson/geometry mapping → GeoDataFrame(4326)."""
    if isinstance(roi, gpd.GeoDataFrame):
        gdf = roi
    elif isinstance(roi, str):
        # Storage seam, not gpd.read_file: GDAL has no abfss:// driver and reports a
        # blob-hosted roi as "No such file or directory" (TODO #47). `run_inference`
        # splits the ROI into grid cells again in preflight, so a blob roi reaches here
        # on every P4 ROI-mode run.
        gdf = fs.read_geo(roi)
    else:  # a geojson dict / __geo_interface__ / shapely geometry
        geom = shapely.geometry.shape(roi["geometry"]) if isinstance(roi, dict) and "geometry" in roi \
            else shapely.geometry.shape(roi)
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def roi_to_s2_grids(roi, *, grid_size_km: float = 5, scale_fact: float = 1.1,
                    res: int | None = None, clip: bool = True) -> gpd.GeoDataFrame:
    """Split an ROI into overlapping S2 grid cells, clipped to the ROI.

    Steps: S2-`polyfill` the ROI's **convex hull** at the level for
    `grid_size_km` (5 km → res 11), keep cells that **intersect** the ROI, **scale** each by
    `scale_fact` (1.1 → 10 % overlap per side), then **clip** to the ROI so grids stay inside it
    (`clip=False` keeps the scaled, unclipped cells). Finally, any cell fully `covered_by`
    another cell in the result is dropped (#69): an ROI that is itself one S2 cell
    polyfills its 8 neighbours too, and after clip+scale they come back as slivers inside
    the central cell. A dropped cell is always a subset of a kept one, so the union of the
    returned cells is unchanged, and the drop count is always printed -- never silent.

    **An ROI is one region, not a list of shapes.** A multi-row `roi` is `unary_union`-ed into a
    single (multi)polygon *first*, and every step — hull, intersect, clip — works against that
    union. So the output is **one row per S2 cell, ids unique**, whether you pass 1 polygon or
    900. If you want one datacube per *shape*, that is not this function:
    pass your shapefile straight to `workflows.create_datacube` with your own `id_col`.

    `roi` is a GeoDataFrame, a file path, or a geojson mapping. Returns a GeoDataFrame with
    columns `id` (the S2 cell id) + `geometry`, in EPSG:4326 — feed it straight to
    `workflows.create_datacube` as the inference shapes (`id_col="id"`).
    """
    try:
        from s2 import s2
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "roi_to_s2_grids needs the optional '[grid]' extra: pip install -e '.[grid]' "
            "(brings s2 + s2cell)."
        ) from exc

    roi_gdf = _as_gdf_4326(roi)
    shape = unary_union(list(roi_gdf.geometry.values))
    if res is None:
        res = grid_size_to_res(grid_size_km)

    cells = s2.polyfill(
        geo_json=shapely.geometry.mapping(shape.convex_hull),
        res=res, geo_json_conformant=True, with_id=True,
    )
    df = pd.DataFrame(cells)
    df["geometry"] = df["geometry"].apply(shapely.geometry.Polygon)
    df = df[df["geometry"].apply(shape.intersects)].reset_index(drop=True)
    df["geometry"] = df["geometry"].apply(
        lambda g: shapely.affinity.scale(g, xfact=scale_fact, yfact=scale_fact)
    )
    grids = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    if clip:
        # Clip to the ROI's UNION (`shape`), never per row. `gpd.overlay` against the raw
        # `roi_gdf` emits one row per (cell x roi-polygon) pair and REPEATS cell ids. `id` is
        # the work-unit key (`create_datacube` derives `export_folderpath` from it), so N tasks
        # would write the SAME folder concurrently: `InvalidBlockList` on blob. `overlay`
        # against the union is equivalent but not faster.
        # Benchmark: `git show 55fa721`.
        grids["geometry"] = grids.geometry.intersection(shape)
        # A cell that only *touches* the ROI can intersect to a line/point; drop those
        # (and any empties) so every returned row is a real polygonal work unit.
        grids = grids[
            ~grids.geometry.is_empty
            & grids.geometry.notna()
            & grids.geom_type.isin(("Polygon", "MultiPolygon"))
        ].reset_index(drop=True)

    if grids["id"].duplicated().any():  # pragma: no cover - structurally impossible now
        dupes = grids["id"].value_counts()
        raise AssertionError(
            f"roi_to_s2_grids produced duplicate cell ids "
            f"({int((dupes > 1).sum())} of {grids['id'].nunique()} repeated, worst "
            f"{dupes.iloc[0]}x) -- one cell must be exactly one row (spec 21 D-GRID-1)."
        )

    grids = _drop_covered_cells(grids)
    return grids


#: Relative-area tolerance for `_covered`. Two cells identical or nested by construction
#: (both are `scaled_cell.intersection(shape)` against the SAME `shape`) still differ by
#: float noise (~1e-14 relative). Exact `.covered_by()` misses that, so a relative-area
#: tolerance is used.
_COVERED_TOL = 1e-9


def _covered(a: shapely.geometry.base.BaseGeometry, b: shapely.geometry.base.BaseGeometry,
             tol: float = _COVERED_TOL) -> bool:
    """`a` is covered by `b`: boundary-inclusive (unlike `contains`) and tolerant of
    GEOS floating-point noise in the exact `covered_by` predicate (see `_COVERED_TOL`).
    Equivalent to `a.covered_by(b)` up to that tolerance."""
    if not a.intersects(b):
        return False
    area_a = a.area
    if area_a == 0:
        return True
    return a.difference(b).area <= tol * area_a


def _drop_covered_cells(grids: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop any cell whose geometry is covered by another cell in the same set (#69).

    Covered-by, not `contains`/IoU: a clipped sliver *shares boundary* with the cell
    that covers it, which `contains` (boundary-exclusive) misses (see `_covered` for why
    the exact `.covered_by()` predicate itself isn't used directly). A dropped cell is
    always a geometric subset of a kept cell, so the union of the output is unchanged
    -- that's the whole safety argument.

    Two cells that mutually cover each other (i.e. are equal) would otherwise both get
    dropped by a naive one-pass check; tie-broken deterministically by keeping the
    smaller `id`.
    """
    n = len(grids)
    if n <= 1:
        print(f"[grid] {n} cells -> {n} after dropping 0 already covered", flush=True)
        return grids

    geoms = grids.geometry.to_numpy()
    ids = grids["id"].to_numpy()
    tree = STRtree(geoms)

    drop: set[int] = set()
    for i in range(n):
        candidates = [j for j in tree.query(geoms[i], predicate="intersects") if j != i]
        covering = [j for j in candidates if _covered(geoms[i], geoms[j])]
        if not covering:
            continue
        if any(not _covered(geoms[j], geoms[i]) for j in covering):
            # strictly covered by something bigger (not mutual) -> always redundant
            drop.add(i)
            continue
        # every "coverer" also covers the reverse (mutual coverage, i.e. equal) ->
        # keep exactly one of the tied group, deterministically, by smallest id
        equal_group = [i] + covering
        keep = min(equal_group, key=lambda k: ids[k])
        if i != keep:
            drop.add(i)

    if drop:
        keep_mask = [i not in drop for i in range(n)]
        grids = grids[keep_mask].reset_index(drop=True)
    print(f"[grid] {n} cells -> {len(grids)} after dropping {len(drop)} already covered",
          flush=True)
    return grids
