"""Register a pgSTAC search + print the XYZ mosaic tile URL (spec 30 A3/A4).

Reuses `demos.titiler_serve.build_colormap()` (spec 29) for the discrete crop-class colormap,
`POST`s a `collections: ["rise-eo-inference"]` search to titiler-pgstac's **register** endpoint, and
prints the full XYZ template for both curl and a QGIS XYZ layer.

**Pinned API note (installed `titiler.pgstac==3.0.0`, the source of truth — spec 30 deviates from
its own draft assumption here, so this is documented per spec 29's rio-tiler precedent):** the
register/tile paths are titiler-pgstac's own `/searches/...` routes (`titiler/pgstac/main.py`,
`titiler/mosaic/factory.py` in the installed package), **not** `/mosaic/register` /
`/mosaic/{searchid}/tiles/...` — that shape is MPC's own product wrapping (naming) around the same
titiler-pgstac library, per `../../STACNOTATOR_DIGEST.md §3`. The underlying **contract is
identical** (POST a STAC search -> get a registered-mosaic id -> XYZ tiles keyed by that id); only
the path segment differs (`searches` vs `mosaic`, `id` vs `searchid` in the response body):
- `POST {raster_base}/searches/register` with the search body -> `{"id": "<search_id>", "links": [...]}`.
- `GET  {raster_base}/searches/{search_id}/tiles/WebMercatorQuad/{z}/{x}/{y}.png?assets=output&colormap=...&nodata=255&resampling=nearest`.

Needs `requests` (add to the `.venv-serving` scratch venv per `README.md`) and run from the `fsd/`
repo root (or with `demos/` on `sys.path`) so `demos.titiler_serve` imports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEMOS = os.path.dirname(_HERE)
if _DEMOS not in sys.path:
    sys.path.insert(0, _DEMOS)

from titiler_serve import build_colormap  # noqa: E402


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python demos/mini_mpc/register_and_url.py",
        description="Register a rise-eo-inference search against titiler-pgstac and print the "
                    "register->searchId->XYZ mosaic tile URL (spec 30 A3/A4).",
    )
    p.add_argument("--raster-base", default="http://127.0.0.1:8082",
                    help="the mini-MPC raster (titiler-pgstac) base URL")
    p.add_argument("--collection-id", default="rise-eo-inference")
    p.add_argument("--render-json", default=None,
                    help="optional [{code,name,color}] override for CLASS_COLORS")
    p.add_argument("--result-json", default="_result_register.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    import requests

    args = _parse_args(argv)

    try:
        cmap = build_colormap(args.render_json)
        cmap_json = json.dumps({str(k): list(v) for k, v in cmap.items()}, separators=(",", ":"))

        search_body = {"collections": [args.collection_id]}
        resp = requests.post(f"{args.raster_base}/searches/register", json=search_body, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        search_id = payload["id"]

        query = urllib.parse.urlencode(
            {"assets": "output", "colormap": cmap_json, "nodata": 255, "resampling": "nearest"}
        )
        xyz_template = (
            f"{args.raster_base}/searches/{search_id}/tiles/WebMercatorQuad/"
            "{z}/{x}/{y}.png?" + query
        )

        result = {
            "step": "register-mosaic",
            "status": "ok",
            "pass": bool(search_id),
            "metrics": {"searchid_present": bool(search_id), "search_id": search_id},
            "expected": {"searchid_present": True},
            "error": None,
            "xyz_template": xyz_template,
        }
    except Exception as e:  # noqa: BLE001 - always leave a pasteable result
        result = {
            "step": "register-mosaic",
            "status": "failed",
            "pass": False,
            "metrics": {"searchid_present": False},
            "expected": {"searchid_present": True},
            "error": str(e),
        }
        with open(args.result_json, "w") as f:
            json.dump(result, f, indent=2, default=str)
        raise

    with open(args.result_json, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(json.dumps(result, indent=2, default=str))
    print()
    print("XYZ template (curl / QGIS XYZ layer):")
    print(xyz_template)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
