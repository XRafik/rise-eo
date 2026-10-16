"""Tier-1 serving validation (spec 29): a minimal pre-styled XYZ tile server over `merged.tif`,
consumed by STACNotator's Bring-Your-Own-XYZ mode. No viewer, no HTML, no pgSTAC — one hand-rolled
route over `rio-tiler` (`../STACNOTATOR_DIGEST.md §2` mode 3).

`GET /cropmap/tiles/{z}/{x}/{y}.png` renders the categorical crop-class raster with a **discrete**
colormap (never a continuous stretch), `nodata=255` -> transparent, and **nearest** resampling
(class codes must never be interpolated) — the make-or-break trio for a categorical COG
(`demos/TITILER_LEAFLET.md §7`).

Isolated venv (NOT rise-eo core):
    python3.11 -m venv .venv-titiler && .venv-titiler/bin/pip install -e ".[titiler]"
    .venv-titiler/bin/python -m demos.titiler_serve
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# --- colormap config (D4): a display-config seam toward the STAC render extension (TODO #28) ---


def load_class_colors() -> dict[str, str]:
    """`{class_name: "#rrggbb"}` — from `e2e_austria.CLASS_COLORS` (side-effect-free import)."""
    from e2e_austria import CLASS_COLORS  # noqa: E402 (demos/ on sys.path, see above)

    return CLASS_COLORS


def _hex_to_rgba(hexcolor: str) -> tuple[int, int, int, int]:
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return (r, g, b, 255)


def build_colormap(render_json: str | None = None) -> dict[int, tuple[int, int, int, int]]:
    """Discrete `{code: (R,G,B,A)}` for codes 0..N-1, sorted by class name (deterministic —
    `sorted(le.classes_)`, matching the label-encoder order the training pipeline uses).

    `render_json` (a `[{"code": int, "name": str, "color": "#rrggbb"}, ...]` list) overrides
    `CLASS_COLORS` when present — the seam toward a model bundle supplying its own display config
    (TODO #28); falls back to `CLASS_COLORS` otherwise.
    """
    if render_json and os.path.exists(render_json):
        with open(render_json) as f:
            entries = json.load(f)
        return {int(e["code"]): _hex_to_rgba(e["color"]) for e in entries}

    class_colors = load_class_colors()
    names = sorted(class_colors)
    return {i: _hex_to_rgba(class_colors[name]) for i, name in enumerate(names)}


# --- transparent-PNG fallback for out-of-bounds tiles -------------------------

def _empty_png(tilesize: int = 256) -> bytes:
    import numpy as np
    from rio_tiler.models import ImageData

    data = np.ma.MaskedArray(
        np.zeros((1, tilesize, tilesize), dtype="uint8"),
        mask=np.ones((1, tilesize, tilesize), dtype=bool),  # fully masked -> transparent
    )
    return ImageData(data).render(img_format="PNG")


# --- app -----------------------------------------------------------------------

def create_app(merged_filepath: str, *, render_json: str | None = None):
    from fastapi import FastAPI, Response
    from fastapi.middleware.cors import CORSMiddleware
    from rio_tiler.errors import TileOutsideBounds
    from rio_tiler.io import Reader

    if not os.path.exists(merged_filepath):
        raise SystemExit(
            f"merged.tif not found at {merged_filepath!r}. Produce it first, e.g.:\n"
            "  .venv-modeldeploy/bin/python demos/e2e_austria.py --creds <cdse_credentials.json>"
        )

    cmap = build_colormap(render_json)
    nodata = 255
    empty_png = _empty_png()

    with Reader(merged_filepath) as r:
        bounds4326 = r.get_geographic_bounds("EPSG:4326")

    app = FastAPI(title="rise-eo Tier-1 pre-styled XYZ (spec 29)")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"],
    )

    xyz_template = "/cropmap/tiles/{z}/{x}/{y}.png"

    @app.get("/")
    def root():
        return {
            "xyz_template": xyz_template,
            "bounds4326": list(bounds4326),
            "class_legend": {code: rgba for code, rgba in sorted(cmap.items())},
        }

    @app.get(xyz_template)
    def tile(z: int, x: int, y: int):
        try:
            with Reader(merged_filepath) as r:
                img = r.tile(
                    x, y, z, indexes=[1], nodata=nodata, resampling_method="nearest",
                )
        except TileOutsideBounds:
            return Response(empty_png, media_type="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})
        png = img.render(img_format="PNG", colormap=cmap)
        return Response(png, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})

    app.state.xyz_template = xyz_template
    app.state.bounds4326 = bounds4326
    return app


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m demos.titiler_serve",
        description="Tier-1 serving validation (spec 29): a pre-styled XYZ over merged.tif, "
                    "for STACNotator's Bring-Your-Own-XYZ mode.",
    )
    default_merged = os.path.join(
        _HERE, "..", "tests/outputs/demo_e2e/model_outputs/merged.tif"
    )
    p.add_argument("--merged", default=default_merged, help="the pre-styled crop-map COG to serve")
    p.add_argument("--render-json", default=None,
                   help="optional [{code,name,color}] override for CLASS_COLORS")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args(argv)


def main(argv=None) -> None:
    import uvicorn

    args = _parse_args(argv)
    app = create_app(os.path.abspath(args.merged), render_json=args.render_json)

    base = f"http://{args.host}:{args.port}"
    print(f"XYZ template: {base}{app.state.xyz_template}")
    print(f"bounds (lon/lat): {list(app.state.bounds4326)}")
    print("Ctrl-C to stop.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
