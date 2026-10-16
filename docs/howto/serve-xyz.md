# How to: serve fsd output on an XYZ map viewer

> **Last verified:** 2026-07-31 @ `df98463` (spec 41 D5 tier 2 — "dated"). Re-verify after any
> change to `rise_eo.run_inference`'s STAC export (`rise_eo.catalog.stac`) or the mini-MPC serving stack in
> `demos/mini_mpc/`.

`rise_eo.run_inference` (see [`docs/tutorial.md`](../tutorial.md) §6 and
[`bundle-your-model.md`](bundle-your-model.md)) already produces standard artifacts — COGs +
a STAC catalog. **fsd builds no dashboard and no server of its own.** Serving is stock software
pointed at those artifacts; this page names two proven ways to do it and where the worked examples
live.

## Tier 1 — a single pre-styled XYZ URL (fastest, one output/merged COG)

Serve one COG (e.g. `run_inference`'s `merged_filepath`) directly as pre-styled XYZ tiles, with a
discrete colormap and nearest-neighbor resampling for categorical output. This is the simplest
integration and needs no database — good for a first look or a Bring-Your-Own-XYZ viewer slice.

```bash
python3.11 -m venv .venv-titiler && .venv-titiler/bin/pip install -e ".[titiler]"
.venv-titiler/bin/python -m demos.titiler_serve --merged <your run>/model_outputs/merged.tif
# -> XYZ template: http://127.0.0.1:8000/cropmap/tiles/{z}/{x}/{y}.png
curl -s -o /tmp/t.png -w '%{http_code} %{content_type}\n' \
  http://127.0.0.1:8000/cropmap/tiles/13/4437/2823.png        # -> 200 image/png (Austria tile)
```

Check it in QGIS (Add Layer → Add XYZ Layer, paste the template): distinct class colours,
transparent nodata, correctly placed. The same template URL works as a Bring-Your-Own-XYZ slice
in STACNotator (validated end to end, spec 29).

Two things that bite categorical rasters specifically: the colormap must stay **discrete** (a
continuous ramp smears class boundaries), and resampling must be **nearest**, never bilinear.

## Tier 2 — a real STAC API + tile server over many outputs

For serving a whole inference run (many cells, true per-cell geometry, register→searchId→XYZ like
MPC) rather than one flattened mosaic, load the STAC catalog into a **stock** pgSTAC +
stac-fastapi-pgstac + titiler-pgstac stack — the same shape MPC itself uses, so a tool built against
MPC's API (like STACNotator) treats fsd's output as "just another MPC".

The stack lives in `demos/mini_mpc/` (its `README.md` says what is borrowed and what is built
locally, and why):

```bash
cd demos/mini_mpc && cp -n .env.example .env && docker compose up --build -d && cd ../..
python3.11 -m venv .venv-serving && .venv-serving/bin/pip install -e ".[dev,serving]"
.venv-serving/bin/pip install "pypgstac[psycopg]==0.9.11" requests

.venv-serving/bin/python demos/mini_mpc/load_pgstac.py \
    --stac-dir <your run>/model_outputs/stac --outputs-dir <your run>/model_outputs/cells
# --outputs-dir must match compose's RISE_EO_OUTPUTS_DIR (the folder mounted at /data)

.venv-serving/bin/python demos/mini_mpc/register_and_url.py
# -> prints http://127.0.0.1:8082/searches/<id>/tiles/WebMercatorQuad/{z}/{x}/{y}.png?...
```

Paste the printed template into QGIS as an XYZ layer: real class colours and true (non-boxy)
per-cell footprints. Validated on the 300-item Austria run. Tear down with `docker compose down`
(keeps `./.pgdata`) or `docker compose down -v` (wipes it).

To export the STAC catalog as stac-geoparquet instead:
`.venv-serving/bin/python -m demos.mini_mpc.export_stac_geoparquet --stac-dir <your run>/model_outputs/stac`
(writes `catalog.parquet` next to `catalog.json`).

## Which tier for you

| | Tier 1 | Tier 2 |
|---|---|---|
| What it needs | one COG, a Python process | Docker, a Postgres+pgSTAC stack |
| What it serves | one pre-styled mosaic | a real STAC API over N items, per-item geometry |
| Good for | a first look, a BYO-XYZ viewer slice | production-shaped serving, "fsd looks like MPC" |

## Where to go next

- [`bundle-your-model.md`](bundle-your-model.md) — produce the COGs/STAC this page serves.
- [`run-at-scale.md`](run-at-scale.md) — produce hundreds of them instead of one.
