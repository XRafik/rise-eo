# mini-MPC — a local pgSTAC + titiler-pgstac harness

A **local, throwaway** stack that loads a rise-eo STAC catalog into a stock pgSTAC +
stac-fastapi-pgstac + titiler-pgstac and serves it through the same **register → searchId → XYZ**
flow Microsoft Planetary Computer (MPC) uses — the path STACNotator treats as a first-class fast
path (`../../STACNOTATOR_DIGEST.md §3`, workspace root, not part of this repo).

**Dataset-agnostic.** It was built for spec 30 (Tier-2 serving validation of fsd's crop-map
**inference outputs**), but nothing in the stack is specific to that dataset — any fsd static STAC
catalog loads and serves the same way. Two datasets are used in the repo:

| Dataset | STAC catalog | What it serves | Run-book |
|---|---|---|---|
| **Inference outputs** (crop map) | `tests/outputs/demo_e2e/model_outputs/stac` | single-band `output` COGs, discrete class colormap | [`docs/howto/serve-xyz.md`](../../docs/howto/serve-xyz.md) |
| **Raw imagery** (mixed-baseline S2) | `tests/outputs/spec34_mixed_baseline/stac` | `B04/B03/B02/SCL` COGs, RGB + `unscale` | run-book 34-mini-mpc-cross-baseline (tag `docs-archive-2026`) |

See `specs/30-tier2-mini-mpc-validation.md` for the design.

**No production/Azure deploy** — dev-only harness (spec 30 non-goals). No private infra values live
here (this repo is public MIT).

## What's borrowed vs. locally built

| Service | Image | Source |
|---|---|---|
| `database` (pgSTAC) | `ghcr.io/stac-utils/pgstac:v0.9.11` | **pulled as-is**, no fork — official pgSTAC (Postgres + PostGIS + schema) image. |
| `stac` (stac-fastapi-pgstac) | built from `dockerfiles/Dockerfile.stac-fastapi-pgstac` | installs the **pinned stock PyPI package** `stac-fastapi.pgstac[server]==6.3.1` on a slim Python base, runs its own unmodified entrypoint (`python -m stac_fastapi.pgstac.app`). |
| `raster` (titiler-pgstac) | built from `dockerfiles/Dockerfile.titiler-pgstac` | installs the **pinned stock PyPI package** `titiler.pgstac[psycopg-binary]==3.0.0`, run with upstream's exact `gunicorn -k uvicorn.workers.UvicornWorker titiler.pgstac.main:app` command. |

**Why not eoAPI's own `docker-compose.yml`?** eoAPI's (and titiler-pgstac's, stac-fastapi-pgstac's)
compose files all `build:` their app images **from a full monorepo checkout** — there's no
published "just `docker pull`" image for the app layer, only for `database`. Installing the **same
pinned PyPI packages** via a two-line Dockerfile that runs their own unmodified entrypoint gets the
identical stock software with a much smaller footprint and a trivially bumpable pin (spec 30 D-A).
Pins are cross-checked: `stac-fastapi.pgstac` and `titiler.pgstac` both require
`pypgstac>=0.9.11,<0.10`, matching the `pgstac:v0.9.11` DB image.

## Services / ports (bound to `127.0.0.1` by default — see `.env.example`)

- `database` — Postgres/pgSTAC, `localhost:5439`. DSN: `postgresql://username:password@localhost:5439/postgis`.
- `stac` — the STAC API, `http://localhost:8081` (`POST /search`, `/collections`, `/collections/{id}/items`).
- `raster` — titiler-pgstac, `http://localhost:8082` (`POST /searches/register`, `GET /searches/{id}/tiles/...`).
  These are titiler-pgstac's own route names; MPC's `/mosaic/register` + `searchid` are its product
  wrapping around the identical contract (see `register_and_url.py`'s docstring).

## The `/data` mount — the one non-obvious wiring step

The `raster` container reads COG pixels from a host folder bind-mounted to `/data`, set by
**`RISE_EO_OUTPUTS_DIR`** in `.env` (relative to *this* `demos/mini_mpc/` dir). `load_pgstac.py`
rewrites every asset href to `/data/<path-under-outputs-dir>`, so **`RISE_EO_OUTPUTS_DIR` and
`load_pgstac.py --outputs-dir` must resolve to the same physical folder**, or every tile 404s with
`/data/<...>: No such file or directory`. Changing `RISE_EO_OUTPUTS_DIR` requires **recreating** the
container (`docker compose up -d --force-recreate raster`) — editing `.env` alone does nothing to a
running container. Verify the live mount:

```bash
docker inspect rise-eo-mini-mpc-raster --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}'
```

## Scripts

- `load_pgstac.py` — reads a rise-eo static STAC catalog (`--stac-dir`, a `catalog.json` tree),
  rewrites each COG's asset href to `/data/...` (relative to `--outputs-dir`), emits
  `collections.ndjson` + `items.ndjson`, and `pypgstac load`s both into `database`. Dataset-agnostic.
- `register_and_url.py` — a **crop-map-specific** helper: registers a `collections=["rise-eo-inference"]`
  search and prints an XYZ URL baked with the discrete **class colormap** (`assets=output`,
  `nodata=255`). Use it **only** for the inference-output dataset. For **RGB / reflectance imagery**
  it does not apply — build the tile URL by hand (see the cookbook below); `--unscale`/`--rescale`/
  `--assets` are not among its flags.
- `export_stac_geoparquet.py` (spec 30 Deliverable B) — converts a static catalog to a single
  `catalog.parquet`. An **alternate interchange artifact**, *not* part of the Docker bring-up and
  *not* consumed by `load_pgstac.py` (which uses ndjson). Needs the `[serving]` extra
  (`stac-geoparquet`) in a separate venv. Skip it unless you specifically want the geoparquet form.

## Bring-up (general)

`load_pgstac.py`/`register_and_url.py` import `fsd`, so run them from a venv with fsd installed
(the core `.venv` is enough for `load_pgstac.py`; `register_and_url.py` also needs `requests`).

```bash
cd demos/mini_mpc
cp .env.example .env                          # then set RISE_EO_OUTPUTS_DIR (see the mount note above)
docker compose up --build -d                  # first run builds the two app images
docker compose ps -a                          # all 3 up; db healthy; raster shows :8082

cd ../..                                       # run scripts from repo root so fsd/demos import
.venv/bin/python demos/mini_mpc/load_pgstac.py \
    --stac-dir <your STAC catalog dir> \
    --outputs-dir <the folder RISE_EO_OUTPUTS_DIR points at>
```

**Worked example A — inference outputs (spec 30):** `RISE_EO_OUTPUTS_DIR=../../tests/outputs/demo_e2e/model_outputs/cells`,
`--stac-dir tests/outputs/demo_e2e/model_outputs/stac`, `--outputs-dir tests/outputs/demo_e2e/model_outputs/cells`,
then `register_and_url.py` for the crop-map XYZ URL.

**Worked example B — raw imagery (spec 34b):** `RISE_EO_OUTPUTS_DIR=../../tests/outputs/spec34_mixed_baseline`,
`--stac-dir tests/outputs/spec34_mixed_baseline/stac`, `--outputs-dir tests/outputs/spec34_mixed_baseline`,
then build an RGB `unscale` URL by hand (cookbook below). Full walk-through: run-book 34-mini-mpc-cross-baseline, at tag `docs-archive-2026`.

---

## Operations cookbook

All `curl`s assume the default local ports. `RASTER=http://localhost:8082`, `STAC=http://localhost:8081`.

### Swap in a new dataset (re-mount + reload)

```bash
# 1. point the mount at the new folder (demos/mini_mpc/.env, relative to this dir)
#    RISE_EO_OUTPUTS_DIR=../../tests/outputs/<your-dataset>
# 2. recreate the raster container so it re-mounts
cd demos/mini_mpc && docker compose up -d --force-recreate raster && cd ../..
# 3. load the new catalog (see Bring-up). Loading is additive — old collections remain
#    until you delete them (below).
```

### Register a search → get a mosaic id

A "search" is a saved STAC query; titiler mosaics over its results. Registering is **idempotent**
(the id is a hash of the body) — the same body always returns the same id.

```bash
# whole collection:
curl -s -X POST $RASTER/searches/register -H "Content-Type: application/json" \
  -d '{"collections":["sentinel-2-l2a"]}'
# -> {"id":"<hash>","links":[...]}

# filtered by time (used in runbook 34b to isolate one processing baseline):
curl -s -X POST $RASTER/searches/register -H "Content-Type: application/json" \
  -d '{"collections":["sentinel-2-l2a"],"datetime":"2021-01-01/2021-12-31"}'

# filtered by bbox (minx,miny,maxx,maxy):
curl -s -X POST $RASTER/searches/register -H "Content-Type: application/json" \
  -d '{"collections":["sentinel-2-l2a"],"bbox":[16.0,48.0,16.2,48.2]}'
```

The response `id` goes into the tile URL: `$RASTER/searches/<id>/tiles/WebMercatorQuad/{z}/{x}/{y}.png?...`.

### Build a tile URL by hand (RGB imagery)

For reflectance/RGB imagery, `register_and_url.py` does not apply — construct the query yourself:

```bash
# three single-band assets stacked as R,G,B; unscale applies each COG's GDAL scale/offset
# (spec 34: unscale=true -> physical reflectance, so rescale is in reflectance units)
$RASTER/searches/<id>/tiles/WebMercatorQuad/{z}/{x}/{y}.png?assets=B04&assets=B03&assets=B02&rescale=0,0.3&unscale=true&resampling=bilinear
```

- `unscale=true` → physical reflectance → `rescale=0,0.3`. `unscale=false` → raw DN → `rescale=0,3000`.
  Matching `rescale` to `unscale` matters — unscale changes the units by 10000×.
- Get the mosaic's real tile extent (the world-wide default bounds aren't useful for picking a tile):
  `curl -s "$RASTER/searches/<id>/WebMercatorQuad/tilejson.json?assets=B04&assets=B03&assets=B02"`

### Smoke-test one tile (before loading into QGIS)

```bash
curl -s "$RASTER/searches/<id>/tiles/WebMercatorQuad/11/1115/711.png?assets=B04&assets=B03&assets=B02&rescale=0,0.3&unscale=true&resampling=bilinear" -o /tmp/t.png
file /tmp/t.png            # want: PNG image data, ~tens of KB
cat /tmp/t.png             # if it says JSON, this prints the error (e.g. a /data 404 -> mount wrong)
```

### Inspect what's loaded (via the STAC API — no psql)

```bash
curl -s $STAC/collections | python -c "import sys,json;[print(c['id']) for c in json.load(sys.stdin)['collections']]"
curl -s "$STAC/collections/sentinel-2-l2a/items" | python -c "import sys,json;d=json.load(sys.stdin);print('items:',len(d['features']))"
```

### Delete a collection / searches / wipe the DB (psql via the container)

DB creds are the DSN above (`-U username -d postgis`). pgSTAC functions live in the `pgstac` schema.

```bash
# delete one collection and all its items:
docker compose exec database psql -U username -d postgis -c "SELECT pgstac.delete_collection('sentinel-2-l2a');"

# clear all registered searches (they're just cached queries; re-created on next register):
docker compose exec database psql -U username -d postgis -c "DELETE FROM pgstac.searches;"

# list tables if a name differs in your pgstac version:
docker compose exec database psql -U username -d postgis -c "\dt pgstac.*"

# NUKE everything (collections, items, searches, the lot) — removes the .pgdata volume:
docker compose down -v
```

After `docker compose down -v`, the next `docker compose up --build -d` starts an empty pgSTAC —
re-run `load_pgstac.py` to repopulate.

## Teardown

```bash
docker compose down        # keep ./.pgdata (fast re-run, data persists)
docker compose down -v     # also wipe the pgdata volume (empty DB next time)
```
