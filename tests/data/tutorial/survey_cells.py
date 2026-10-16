"""Spec 42 §1 / amendment A3: survey every S2 grid cell over an ROI and rank
them as tutorial-fixture candidates.

This exists so the fixture's cell choice is **reproducible evidence, not a
hand-pick**. Spec 42 §1 originally justified cell `4772924` as "the cell with
the most labelled fields"; A3 replaces that with the ranking this script
produces, because "most fields" turned out not to be the property that matters.

What it measures, per cell:

- `n_fields` / `n_crops` — how many labelled fields and distinct crops fall in it;
- `crop_area_ha` / `fill_pct` — how much of the cell the labelled fields actually
  cover (the fields are a *sample*, so fill is a few percent, not most of the cell);
- `major_share_pct` — the fraction of labelled area held by the top `--n-major`
  crops. **This is the number that decides the choice.** A 3-class collapse
  (major1 / major2 / other) is only pedagogically sound when the two majors
  dominate; where they do not, the catch-all `other` becomes the largest class,
  which teaches the wrong lesson about class balance.

Not part of the fsd wheel; not run by the test suite (spec 42 D4).

    python tests/data/tutorial/survey_cells.py \\
        --at-roi ../shapefiles/AT_ROI.geojson \\
        --fields ../shapefiles/AT_2018_TRAIN.geojson \\
        --label-col crop --top 12
"""

from __future__ import annotations

import argparse

import geopandas as gpd
import pandas as pd

from rise_eo.grid import roi_to_s2_grids

__all__ = ["survey", "main"]


def survey(
    at_roi_path: str,
    fields_path: str,
    *,
    label_col: str = "crop",
    grid_size_km: float = 5.0,
    n_major: int = 2,
) -> pd.DataFrame:
    """One row per cell that contains at least one field, ranked best-first.

    Ranked by `(n_crops, major_share_pct, crop_area_ha)` — variety first, then
    how cleanly the top-`n_major` crops dominate, then absolute labelled area.
    """
    grids = roi_to_s2_grids(at_roi_path, grid_size_km=grid_size_km)
    fields = gpd.read_file(fields_path)
    if label_col not in fields.columns:
        raise ValueError(
            f"{fields_path!r}: missing label column {label_col!r}; it has "
            f"{sorted(c for c in fields.columns if c != 'geometry')}."
        )
    if fields.crs is None:
        fields = fields.set_crs("EPSG:4326")
    fields = fields.to_crs("EPSG:4326")

    # `overlay` (not a per-cell loop): one row per (field x cell) overlap, with
    # the field clipped to the cell, so area is the area actually inside it.
    pairs = gpd.overlay(
        fields[[label_col, "geometry"]], grids[["id", "geometry"]],
        how="intersection", keep_geom_type=True,
    )
    if pairs.empty:
        return pd.DataFrame()

    area_crs = grids.estimate_utm_crs()
    pairs = pairs.to_crs(area_crs)
    pairs["area_ha"] = pairs.geometry.area / 1e4
    cell_km2 = dict(zip(grids["id"], grids.to_crs(area_crs).geometry.area / 1e6))

    rows = []
    for cell_id, sub in pairs.groupby("id"):
        by_crop = sub.groupby(label_col)["area_ha"].sum().sort_values(ascending=False)
        total = float(by_crop.sum())
        majors = list(by_crop.head(n_major).index)
        major_ha = float(by_crop.head(n_major).sum())
        rows.append({
            "id": cell_id,
            "n_fields": int(len(sub)),
            "n_crops": int(len(by_crop)),
            "crop_area_ha": round(total, 2),
            "fill_pct": round(100 * total / 100 / cell_km2[cell_id], 3),
            "major_share_pct": round(100 * major_ha / total, 1) if total else 0.0,
            "majors": ", ".join(majors),
            "other_ha": round(total - major_ha, 2),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(
        ["n_crops", "major_share_pct", "crop_area_ha"], ascending=False
    ).reset_index(drop=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python tests/data/tutorial/survey_cells.py",
        description="Rank S2 grid cells over an ROI as tutorial-fixture candidates (spec 42 A3).",
    )
    p.add_argument("--at-roi", required=True)
    p.add_argument("--fields", required=True)
    p.add_argument("--label-col", default="crop")
    p.add_argument("--grid-size-km", type=float, default=5.0)
    p.add_argument("--n-major", type=int, default=2)
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--csv", default=None, help="optional path to write the full ranking")
    args = p.parse_args(argv)

    df = survey(
        args.at_roi, args.fields, label_col=args.label_col,
        grid_size_km=args.grid_size_km, n_major=args.n_major,
    )
    if df.empty:
        print("no cell contains any field -- check --at-roi / --fields overlap.")
        return 1

    pd.set_option("display.width", 200)
    print(f"cells with >=1 field: {len(df)}")
    print(f"\n=== top {args.top}, ranked by (n_crops, major_share_pct, crop_area_ha) ===")
    print(df.head(args.top).to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nfull ranking -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
