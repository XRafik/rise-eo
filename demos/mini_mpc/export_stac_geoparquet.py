"""Export an existing static STAC catalog to stac-geoparquet (spec 30 Deliverable B, CLI).

Additive: reads the JSON STAC catalog `run_inference`/`regen_output_stac` already write
(`catalog.json` -> collection -> item JSONs) and writes a single `catalog.parquet` next to it via
`rise_eo.catalog.stac_geoparquet.items_to_stac_geoparquet`. Not wired into any default pipeline write
path (the full catalog-format migration is the #26 follow-on) — run this by hand when you want the
compact interchange form, e.g. as a second load path into the mini-MPC's pgSTAC (spec 30 B3,
optional/not exercised by the Tier-2 runbook, which uses ndjson per D-D).

Needs the `[serving]` extra (`stac-geoparquet`) — an isolated venv, NOT the core `.venv`:
    python3.11 -m venv .venv-serving && .venv-serving/bin/pip install -e ".[serving]"
    .venv-serving/bin/python -m demos.mini_mpc.export_stac_geoparquet \\
        --stac-dir tests/outputs/demo_e2e/model_outputs/stac

Run as:  python -m demos.mini_mpc.export_stac_geoparquet --stac-dir <stac catalog dir>
"""

from __future__ import annotations

import argparse
import json
import os

import pystac

from rise_eo.catalog import stac as _stac
from rise_eo.catalog import stac_geoparquet as _stac_geoparquet


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m demos.mini_mpc.export_stac_geoparquet",
        description="Convert a static JSON STAC catalog (catalog.json) to a single "
                    "stac-geoparquet file (catalog.parquet), written next to it.",
    )
    p.add_argument("--stac-dir", required=True,
                   help="folder holding catalog.json (as written by write_stac_catalog)")
    p.add_argument("--dst", default=None,
                   help="output .parquet path (default <stac-dir>/catalog.parquet)")
    p.add_argument("--result-json", default=None,
                   help="write the spec-24 _result.json here (default <stac-dir>/../_result.json)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    catalog_json = os.path.join(args.stac_dir, "catalog.json")
    dst_fp = args.dst or os.path.join(args.stac_dir, "catalog.parquet")
    result_json = args.result_json or os.path.join(
        os.path.dirname(os.path.normpath(args.stac_dir)), "_result.json"
    )

    try:
        catalog = pystac.Catalog.from_file(catalog_json, stac_io=_stac._StorageStacIO())
        items = list(catalog.get_items(recursive=True))
        _stac_geoparquet.items_to_stac_geoparquet(items, dst_fp)
        result = {
            "step": "export-stac-geoparquet",
            "status": "ok",
            "pass": True,
            "metrics": {"items": len(items), "dst": dst_fp},
            "expected": {},
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 - always leave a pasteable result
        result = {
            "step": "export-stac-geoparquet",
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
