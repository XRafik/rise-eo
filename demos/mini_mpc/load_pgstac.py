"""Load fsd's static STAC catalog into the mini-MPC's pgSTAC (spec 30 A2).

Reads the existing JSON STAC catalog `run_inference`/`regen_output_stac` write
(`catalog.json` -> collection -> item JSONs) through the **`rise_eo.storage`** seam, rewrites each
output COG's asset `href` from its host absolute path to the **container-visible**
`/data/<path-under-outputs-dir>` (the compose `raster` service bind-mounts `--outputs-dir` ->
`/data` — this is the one non-obvious wiring step; tiles 500 without it, spec 30 A2), emits
`collections.ndjson` + `items.ndjson`, and `pypgstac load`s both into the pgSTAC DB.

Needs `pypgstac[psycopg]` (NOT fsd's core `.venv` — a scratch venv, e.g. `.venv-mini-mpc`, per
`README.md`). Pinned API (installed `pypgstac==0.9.11`, matching the `database` image tag):
`pypgstac.db.PgstacDB(dsn=...)` + `pypgstac.load.Loader(db=...).load_collections(...)`
/ `.load_items(...)` with `insert_mode=Methods.upsert` (idempotent re-runs).

Run as:  python load_pgstac.py --stac-dir ../../tests/outputs/demo_e2e/model_outputs/stac \\
             --outputs-dir ../../tests/outputs/demo_e2e/model_outputs/cells
"""

from __future__ import annotations

import argparse
import json
import os

import pystac

DEFAULT_DSN = "postgresql://username:password@localhost:5439/postgis"


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python load_pgstac.py",
        description="Convert fsd's static STAC catalog to ndjson (rewriting output hrefs to the "
                    "/data mount) and pypgstac-load it into the mini-MPC's pgSTAC.",
    )
    p.add_argument("--stac-dir", required=True, help="folder holding catalog.json")
    p.add_argument("--outputs-dir", required=True,
                   help="the host folder bind-mounted to /data in the raster container "
                        "(compose's RISE_EO_OUTPUTS_DIR — MUST match)")
    p.add_argument("--dsn", default=DEFAULT_DSN, help="pgSTAC DSN (default: the compose db)")
    p.add_argument("--ndjson-dir", default=".",
                   help="where to write collections.ndjson / items.ndjson (default: cwd)")
    p.add_argument("--result-json", default="_result.json")
    return p.parse_args(argv)


def _rewrite_href(href: str, outputs_dir: str) -> str:
    rel = os.path.relpath(href, os.path.abspath(outputs_dir))
    if rel.startswith(".."):
        raise ValueError(
            f"load_pgstac: asset href {href!r} is not under --outputs-dir {outputs_dir!r} "
            "(so it can't be rewritten to the /data bind-mount)."
        )
    return "/data/" + rel.replace(os.sep, "/")


def _rewritten_item_dict(item: pystac.Item, outputs_dir: str) -> dict:
    d = item.to_dict(include_self_link=False)
    for asset in d.get("assets", {}).values():
        asset["href"] = _rewrite_href(asset["href"], outputs_dir)
    return d


def _write_ndjson(records: list[dict], fp: str) -> None:
    with open(fp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")


def main(argv=None) -> int:
    from pypgstac.db import PgstacDB
    from pypgstac.load import Loader, Methods

    from rise_eo.catalog import stac as _stac

    args = _parse_args(argv)
    result_json = args.result_json

    try:
        catalog_json = os.path.join(args.stac_dir, "catalog.json")
        catalog = pystac.Catalog.from_file(catalog_json, stac_io=_stac._StorageStacIO())
        items = list(catalog.get_items(recursive=True))
        collections = list(catalog.get_children())  # pystac.Collection objects
        if not items:
            raise ValueError(f"load_pgstac: no items found under {catalog_json!r}.")

        collections_ndjson = os.path.join(args.ndjson_dir, "collections.ndjson")
        items_ndjson = os.path.join(args.ndjson_dir, "items.ndjson")
        _write_ndjson([c.to_dict(include_self_link=False) for c in collections], collections_ndjson)
        _write_ndjson(
            [_rewritten_item_dict(it, args.outputs_dir) for it in items], items_ndjson
        )

        pgdb = PgstacDB(dsn=args.dsn)
        loader = Loader(db=pgdb)
        loader.load_collections(collections_ndjson, insert_mode=Methods.upsert)
        loader.load_items(items_ndjson, insert_mode=Methods.upsert)

        result = {
            "step": "load-pgstac",
            "status": "ok",
            "pass": len(collections) >= 1 and len(items) > 0,
            "metrics": {"collections": len(collections), "items": len(items)},
            "expected": {"collections": 1, "items": 300},
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 - always leave a pasteable result
        result = {
            "step": "load-pgstac",
            "status": "failed",
            "pass": False,
            "metrics": {},
            "expected": {"collections": 1, "items": 300},
            "error": str(e),
        }
        with open(result_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        raise

    with open(result_json, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
