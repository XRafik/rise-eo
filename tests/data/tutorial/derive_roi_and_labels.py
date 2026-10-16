"""Spec 42 step 0 / run-book 43 Step 0: derive the tutorial cell's ROI + labels
from the workspace shapefiles.

Laptop-only, offline, seconds. `shapefiles/` lives at the workspace root, outside
this repo, so a `git clone` on a VM cannot supply it -- this script writes two
small GeoJSONs (`roi.geojson`, `fields.geojson`) that run-book 43 Steps 2-4
(on the Azure VM) consume instead, so the VM never needs the workspace root.

CLI is normative (run-book 43 Step 0) -- match these flags exactly:

    python tests/data/tutorial/derive_roi_and_labels.py \\
        --at-roi   ../shapefiles/AT_ROI.geojson \\
        --fields   ../shapefiles/AT_2018_TRAIN.geojson \\
        --cell-id  4772924 \\
        --out      tests/data/tutorial \\
        --result   tests/outputs/p6_tutorial_fixture/_result_step0.json

Not part of the rise-eo wheel; not run by the test suite (spec 42 D4) -- its output
(`roi.geojson`, `fields.geojson`) is the artifact, committed alongside it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import geopandas as gpd

from rise_eo.grid import roi_to_s2_grids

# spec 42 D3 (as revised by amendment A3): the raw multi-class crop label
# collapses to `n_major + 1` trainable classes -- the `n_major` crops with the
# most *area inside the cell* keep their identity, everything else becomes
# "other".
#
# **The majors are DERIVED, never hardcoded.** The original D3 hardcoded
# {"maize", "hemp"} and matched it case-insensitively against the raw value.
# That silently collapsed every field to "other", because the real label values
# are HCAT compound names -- `grain_maize_corn_popcorn`, `hemp_cannabis` -- which
# never equal "maize"/"hemp". Deriving by area gets the same two crops for cell
# 4772924 (82 % of its labelled area) while surviving a different cell, a
# different ROI, or a renamed nomenclature. `survey_cells.py` ranks the
# candidates; `docs/tutorial.md` (spec 41 P7) states whatever this derives.
CROP_LABEL_OTHER = "other"
DEFAULT_LABEL_COL = "crop"
DEFAULT_N_MAJOR = 2

__all__ = [
    "CROP_LABEL_OTHER", "DEFAULT_LABEL_COL", "DEFAULT_N_MAJOR",
    "pick_major_crops", "collapse_label", "derive", "main",
]


def collapse_label(raw_crop: str, majors) -> str:
    """spec 42 D3/A3's collapse: a crop in `majors` keeps its own name,
    everything else becomes `"other"`. Matching is case-insensitive and
    whitespace-stripped, but never fuzzy -- a near-miss must fail loudly at the
    Step 0 gate rather than quietly land in `other`."""
    key = str(raw_crop).strip().lower()
    return key if key in {str(m).strip().lower() for m in majors} else CROP_LABEL_OTHER


def pick_major_crops(clipped_fields, label_col: str, n_major: int, area_crs=None) -> list[str]:
    """The `n_major` crops holding the most AREA inside the cell, largest first.

    Area, not field count: one 8 ha maize block teaches a classifier more than
    eight 0.1 ha strips of something else, and area is what the pixels in the
    datacube actually are. `clipped_fields` must already be clipped to the cell
    (`derive` does that) or this ranks area outside it too.
    """
    if area_crs is None:
        area_crs = clipped_fields.estimate_utm_crs()
    by_crop = (
        clipped_fields.to_crs(area_crs)
        .assign(_area=lambda g: g.geometry.area)
        .groupby(label_col)["_area"].sum()
        .sort_values(ascending=False)
    )
    if len(by_crop) <= n_major:
        raise ValueError(
            f"the cell holds only {len(by_crop)} distinct {label_col!r} value(s) "
            f"({list(by_crop.index)}), so a top-{n_major} + 'other' collapse "
            f"would leave 'other' empty -- pick a cell with more variety "
            "(tests/data/tutorial/survey_cells.py ranks them)."
        )
    return [str(c) for c in by_crop.head(n_major).index]


def derive(
    at_roi_path: str, fields_path: str, cell_id: str, *, grid_size_km: float = 5.0,
    label_col: str = DEFAULT_LABEL_COL, n_major: int = DEFAULT_N_MAJOR,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, list[str]]:
    """Returns `(cell_gdf, fields_gdf, majors)`: the one S2 grid cell matching
    `cell_id` (derived via `roi_to_s2_grids`, never hand-picked), its fields
    clipped to the cell with `fid` + `crop` (raw) + `label` (collapsed) columns,
    and the `n_major` crop names the collapse kept.

    `label_col` is the raw label column in `fields_path` — **`crop` for
    `AT_2018_TRAIN.geojson`**. (The original code hardcoded `EC_hcat_n`, which is
    a column of a *different* workspace file, `austria_eurocrops_sampled_…`;
    A3 corrected it and made it an argument.)

    Raises `ValueError` if `cell_id` is not among the cells `roi_to_s2_grids`
    derives from `at_roi_path` -- the documented trap (run-book 43 Step 0): the
    wrong cell id (e.g. one ~0.6 degrees east) silently selects a cell with zero
    labelled fields rather than failing, so this at least fails loudly on a
    typo'd id that doesn't exist at all.
    """
    grids = roi_to_s2_grids(at_roi_path, grid_size_km=grid_size_km)
    cell = grids[grids["id"].astype(str) == str(cell_id)]
    if len(cell) == 0:
        raise ValueError(
            f"cell id {cell_id!r} not found among the {len(grids)} cells "
            f"roi_to_s2_grids derived from {at_roi_path!r} at "
            f"grid_size_km={grid_size_km} -- check the id, don't hand-pick one."
        )
    if len(cell) > 1:  # pragma: no cover - roi_to_s2_grids guarantees unique ids
        raise AssertionError(f"cell id {cell_id!r} matched {len(cell)} rows, expected 1.")
    cell = cell.reset_index(drop=True)

    fields = gpd.read_file(fields_path)
    for col in ("fid", label_col):
        if col not in fields.columns:
            raise ValueError(
                f"{fields_path!r}: missing required column {col!r}; it has "
                f"{sorted(c for c in fields.columns if c != 'geometry')}. "
                "Pass --label-col if the label lives under a different name."
            )
    if fields.crs is None:
        fields = fields.set_crs("EPSG:4326")
    elif fields.crs.to_epsg() != 4326:
        fields = fields.to_crs("EPSG:4326")

    cell_shape = cell.geometry.iloc[0]
    clipped = fields[fields.intersects(cell_shape)].copy()
    clipped["geometry"] = clipped.geometry.intersection(cell_shape)
    clipped = clipped[~clipped.geometry.is_empty].reset_index(drop=True)
    if clipped.empty:
        raise ValueError(
            f"cell {cell_id!r} contains no field from {fields_path!r}. Most cells "
            "over an ROI hold none -- use tests/data/tutorial/survey_cells.py to "
            "pick one that does."
        )

    # Derived from area INSIDE the cell, after clipping -- never hardcoded (A3).
    majors = pick_major_crops(clipped, label_col, n_major)

    out_fields = gpd.GeoDataFrame(
        {
            "fid": clipped["fid"].values,
            "crop": clipped[label_col].values,
            "label": [collapse_label(c, majors) for c in clipped[label_col]],
        },
        geometry=clipped.geometry.values,
        crs="EPSG:4326",
    )
    return cell, out_fields, majors


def _write_result(path: str, result: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python tests/data/tutorial/derive_roi_and_labels.py",
        description="Spec 42 step 0: derive the tutorial ROI cell + its labelled "
                     "fields from the workspace shapefiles (run-book 43 Step 0).",
    )
    p.add_argument("--at-roi", required=True, help="path to AT_ROI.geojson")
    p.add_argument("--fields", required=True, help="path to AT_2018_TRAIN.geojson")
    p.add_argument("--cell-id", required=True, help="the S2 grid cell id to select")
    p.add_argument("--grid-size-km", type=float, default=5.0)
    p.add_argument("--label-col", default=DEFAULT_LABEL_COL,
                   help=f"raw label column in --fields (default {DEFAULT_LABEL_COL!r}; "
                        "AT_2018_TRAIN.geojson uses 'crop')")
    p.add_argument("--n-major", type=int, default=DEFAULT_N_MAJOR,
                   help="how many crops keep their own class; the rest become 'other' "
                        f"(default {DEFAULT_N_MAJOR} -> {DEFAULT_N_MAJOR + 1} classes)")
    p.add_argument("--out", required=True, help="output dir for roi.geojson + fields.geojson")
    p.add_argument("--result", required=True, help="path to write the _result.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        cell, fields, majors = derive(
            args.at_roi, args.fields, args.cell_id, grid_size_km=args.grid_size_km,
            label_col=args.label_col, n_major=args.n_major,
        )
    except Exception as e:  # noqa: BLE001 - still leave a pasteable result, then re-raise
        _write_result(args.result, {
            "step": "step0_derive_roi_and_labels",
            "status": "failed",
            "pass": False,
            "metrics": {},
            "expected": {},
            "error": repr(e),
        })
        raise

    os.makedirs(args.out, exist_ok=True)
    roi_path = os.path.join(args.out, "roi.geojson")
    fields_path = os.path.join(args.out, "fields.geojson")
    cell[["id", "geometry"]].to_file(roi_path, driver="GeoJSON")
    fields.to_file(fields_path, driver="GeoJSON")

    minx, miny, maxx, maxy = cell.total_bounds
    counts = fields["label"].value_counts().to_dict()
    class_str = " ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    print(
        f"cell {args.cell_id}  bounds {minx:.4f},{miny:.4f},{maxx:.4f},{maxy:.4f}  "
        f"fields {len(fields)}  classes {class_str}\n"
        f"major crops (derived by area, not hardcoded): {', '.join(majors)}"
    )

    # Run-book 43 Step 0's stated PASS conditions, actually evaluated. A
    # hardcoded `"pass": True` here would report success even when the collapse
    # produced one class -- which is exactly what happened on 2026-07-31, when
    # the hardcoded {"maize", "hemp"} met the real HCAT values
    # (`grain_maize_corn_popcorn`, `hemp_cannabis`) and every field landed in
    # "other". `_result.json` is what gets pasted back, not the logs (spec 24).
    expected_n_classes = args.n_major + 1
    n_classes = len(counts)
    passed = bool(len(fields) > 0 and n_classes == expected_n_classes)
    _write_result(args.result, {
        "step": "step0_derive_roi_and_labels",
        "status": "ok" if passed else "failed",
        "pass": passed,
        "metrics": {
            "cell_id": str(args.cell_id),
            "bounds": [round(minx, 4), round(miny, 4), round(maxx, 4), round(maxy, 4)],
            "fields": int(len(fields)),
            "n_classes": int(n_classes),
            "classes": {str(k): int(v) for k, v in counts.items()},
            "label_col": args.label_col,
            "major_crops": list(majors),
            "roi_path": roi_path,
            "fields_path": fields_path,
        },
        "expected": {"n_classes": expected_n_classes, "fields_non_empty": True},
        "error": None,
    })
    if not passed:
        print(
            f"FAIL: expected {expected_n_classes} classes over a non-empty field "
            f"set, got {n_classes} class(es) over {len(fields)} field(s) "
            f"(majors derived: {majors}). Check --label-col={args.label_col!r} is "
            "the right column and that the cell holds more than "
            f"{args.n_major} distinct crop(s).",
            file=sys.stderr,
        )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
