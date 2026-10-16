"""Regenerate an inference-output STAC catalog from its build manifest (spec 28 D4).

Re-runs just the STAC tail — **no re-inference, no downloads** — reading the existing
`input.csv` (which carries each cell's `export_folderpath`/`shapefilepath`) to rebuild the STAC
Items with the true S2-cell footprint (spec 28) instead of the old raster-bbox geometry. Safe to
run any time after `run_inference` (idempotent overwrite of `stac/`).

Run as:  python -m demos.regen_output_stac \\
             --input-csv tests/outputs/demo_e2e/model_outputs/cells/input.csv \\
             --stac-dir tests/outputs/demo_e2e/model_outputs/stac
"""

from __future__ import annotations

import argparse
import json
import os

from rise_eo.catalog import stac as _stac


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m demos.regen_output_stac",
        description="Regenerate an inference-output STAC catalog from its input.csv build "
                    "manifest (spec 28) — manifest-driven, deterministic, no re-inference.",
    )
    p.add_argument("--input-csv", required=True, help="the run_inference build manifest")
    p.add_argument("--stac-dir", required=True, help="STAC output folder (overwritten)")
    p.add_argument("--collection-id", default="fsd-inference")
    p.add_argument("--catalog-id", default="fsd-inference")
    p.add_argument("--result-json", default=None,
                   help="write the spec-24 _result.json here (default <stac-dir>/../_result.json)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    result_json = args.result_json or os.path.join(
        os.path.dirname(os.path.normpath(args.stac_dir)), "_result.json"
    )

    try:
        items = _stac.cog_outputs_to_items_from_manifest(
            args.input_csv, collection_id=args.collection_id, band_names=["crop_class"],
        )
        _stac.write_stac_catalog(
            items, args.stac_dir, catalog_id=args.catalog_id, collection_id=args.collection_id,
            description="fsd inference outputs (STAC, regenerated with true cell geometry).",
        )
        distinct_ids = len({it.id for it in items})
        non_rectangular = sum(
            1 for it in items
            if len(set(pt[0] for pt in it.geometry["coordinates"][0])) > 2
            or len(set(pt[1] for pt in it.geometry["coordinates"][0])) > 2
        )
        result = {
            "step": "regen-output-stac",
            "status": "ok",
            "pass": True,
            "metrics": {
                "items": len(items),
                "distinct_ids": distinct_ids,
                "non_rectangular_geoms": non_rectangular,
            },
            "expected": {},
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 - always leave a pasteable result
        result = {
            "step": "regen-output-stac",
            "status": "failed",
            "pass": False,
            "metrics": {},
            "expected": {},
            "error": str(e),
        }
        os.makedirs(os.path.dirname(result_json) or ".", exist_ok=True)
        with open(result_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        raise

    os.makedirs(os.path.dirname(result_json) or ".", exist_ok=True)
    with open(result_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
