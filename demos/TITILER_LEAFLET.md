---
status: current
summary: COG/XYZ/titiler/STAC/MPC serving explainer; still accurate as a concepts primer, though its original "build a Leaflet dashboard" framing was superseded by spec 27 (historical) in favor of STACNotator.
---

# Serving the inference outputs in a web map — titiler + Leaflet (explainer)

> **Note (2026-07-14): this is now a CONCEPTS PRIMER, not the plan.** The original plan (build a local
> MosaicJSON Leaflet dashboard, `specs/27`) is **superseded** — rise-eo will not build a dashboard;
> **STACNotator** is the viewer, fed by a stock **pgSTAC + titiler-pgstac** stack over rise-eo's standard
> STAC + COGs (see `PROGRESS.md` LATEST + `../STACNOTATOR_DIGEST.md` + TODO #26–#29). The COG / XYZ /
> titiler / STAC / MPC explanations below are still accurate and worth reading; ignore the
> "build a Leaflet page / MosaicJSON MVP" framing.

**Audience: you, from scratch.** No prior tiling / titiler / web-map experience assumed. This doc
explains *what each piece is and why*, so the companion spec (`specs/27-titiler-leaflet-stac-verify.md`)
and the code Sonnet writes make sense. It is **explanation, not a runbook** — the actual "do this,
paste back the result" steps live in the spec's runbook.

## 0. What we're building, and why

The Austria end-to-end run (`E2E_AUSTRIA.md §8`) produced three things under
`tests/outputs/demo_e2e/model_outputs/`:

- **300 per-cell COGs** — `cells/<window>/<cell_id>/output.tif`, one crop-classification raster per
  S2 grid cell.
- **a STAC catalog** — `stac/catalog.json` → `stac/rise-eo-inference/collection.json` → 300 Item JSONs,
  each pointing at one of those COGs.
- **`merged.tif`** — the 300 cells stitched into one display mosaic (6830×6868 px, EPSG:32633).

We have **never actually looked at the STAC catalog as a catalog** — we assert in pytest that it has
300 distinct Items, but nobody has pointed a real STAC/tiling client at it and watched the crop map
appear on a slippy map. That is the goal: **stand up a tile server + a Leaflet page that reads our
STAC catalog and renders the COGs**, so we can eyeball that (a) the catalog parses, (b) the item
hrefs resolve to real COGs, (c) the geolocation is right, and (d) the class colors are right. This is
the **P5 "serving / observability" layer** (`ROADMAP.md`, `TODO #14`) — orthogonal to the pipeline,
and exactly the kind of visual validation the project already leans on (QGIS eyeballing, but in a
browser and driven by the catalog).

**This is local-only v1.** The COGs are on your laptop's disk; the item asset hrefs are absolute
*local file paths*, not URLs. Serving from Azure Blob / S3 is the storage-seam story later (§9).

---

## 1. COG — why a GeoTIFF can be read a tile at a time

A normal GeoTIFF stores its pixels in a layout you basically have to read start-to-finish. To draw a
small map window from it, a server would download the *whole* file. For a 6830×6868 raster that's
wasteful; for a 100,000² satellite scene it's a non-starter.

A **Cloud-Optimized GeoTIFF (COG)** is the same GeoTIFF data reorganized so a client can fetch *just
the bytes it needs*:

- **Internal tiling** — pixels are stored in small blocks (e.g. 512×512), not row-by-row. To draw one
  map tile you read only the blocks it overlaps.
- **Overviews** — downsampled copies (½, ¼, ⅛ …) are baked in. Zoomed out, the server reads a tiny
  overview instead of decimating full-res pixels on the fly.
- **A header up front** — the byte offsets of every block/overview are listed at the start of the
  file, so a client reads the header once and then knows exactly which byte ranges to ask for.

That last point is the magic: over HTTP this becomes a **range request** (`Range: bytes=…`) — "give me
just bytes 1,048,576–1,081,343." The server (or GDAL locally, via file seeks) reads a few kilobytes
instead of gigabytes. **This is what makes dynamic tiling possible at all.** rise-eo already writes COGs
with overviews on download (`spec 14`) and its inference outputs are COGs too — so the outputs are
*already* in the format a tile server wants. (Background: `E2E_AUSTRIA.md` Appendix A; `TODO #14`.)

---

## 2. Slippy-map tiles & the XYZ scheme — what `{z}/{x}/{y}` means

Every web map you've used (Google/OSM/etc.) is a **slippy map**: the world is cut into a pyramid of
256×256-px PNG tiles. A tile is addressed by three numbers, `z/x/y`:

- **z = zoom level.** z=0 is the whole world in *one* tile. Each zoom level doubles the grid: z=1 is
  2×2 tiles, z=2 is 4×4, … z=n is 2ⁿ × 2ⁿ tiles.
- **x, y = column, row** of the tile at that zoom, counting from the top-left.

A map library asks for exactly the tiles covering your current viewport at your current zoom, as
`.../{z}/{x}/{y}.png`. Pan or zoom → it requests different tiles. This URL shape is the **"XYZ"
tile template**, e.g. `https://host/tiles/{z}/{x}/{y}.png`.

Crucially, this grid is defined in **Web Mercator (EPSG:3857)** — the projection that makes the world a
convenient square. This matters below (§7b), because **our COGs are in UTM (EPSG:32633), not
3857.**

---

## 3. titiler — a *dynamic* tile server

Pre-rendering every `z/x/y` tile of every COG to disk (a "tile cache") is possible but rigid: you'd
regenerate on every color change and store millions of PNGs. **titiler** does it **dynamically**
instead — it's a small web server that, on each tile request, opens the COG, reads *just* the blocks/
overview for that `z/x/y` (the range-read from §1), reprojects/colors them, and returns the PNG. Nothing
is pre-baked; change a color or a COG and the next tile request just reflects it.

titiler = **[rio-tiler]** (the library that does "COG + z/x/y → a pixel array") + **[FastAPI]** (the
web framework that turns that into HTTP endpoints). For a single COG it exposes (paths as of
titiler.core 0.24):

| endpoint | what it gives you |
|---|---|
| `GET /cog/info?url=<cog>` | metadata: bounds, CRS, band count, dtype, overviews |
| `GET /cog/statistics?url=<cog>` | per-band min/max/histogram (handy to pick a rescale) |
| `GET /cog/tiles/{tms}/{z}/{x}/{y}.png?url=<cog>&…` | **the actual map tile** (the XYZ endpoint) |
| `GET /cog/{tms}/tilejson.json?url=<cog>` | a small JSON with the tile URL template + the COG's bounds/zoom, which a map lib can consume directly |
| `GET /cog/WMTSCapabilities.xml?url=<cog>` | a WMTS description (for QGIS/desktop GIS) |

`{tms}` is the **tile matrix set** — the tiling scheme; for web maps it's `WebMercatorQuad` (the XYZ /
EPSG:3857 grid from §2). `url=` is *which* COG to serve — for us, a local absolute path like
`/Users/…/output.tif` (GDAL opens local paths directly; later it becomes a `/vsicurl/…` or blob URL,
§9).

**The `/cog` endpoint serves one COG per request.** That's perfect for the single `merged.tif`
mosaic, but our catalog is **300 separate per-cell COGs**. Serving those is a *mosaic* problem —
covered next.

## 3.5 The key idea: who assembles the mosaic? (server, not browser)

The seamless, lightweight web maps you know (MPC's Sentinel-2, Google, …) feel light **not** because
they're remote — it's because the **server** assembles the mosaic, and the browser sees **one** XYZ
layer. This is the single most important idea here, so it's worth stating plainly:

- **The wrong way (heavy):** have the browser add **one Leaflet layer per COG** — 300 layer objects,
  each independently fetching tiles. The browser drowns. (This is what an early draft of our spec did;
  it's heavier than MPC, not lighter.)
- **The right way (light) — a mosaic endpoint:** the browser adds **one** XYZ layer pointed at a
  *mosaic* endpoint. For each `{z}/{x}/{y}` tile in view, the **server** figures out which COG(s) cover
  that tile, range-reads only those, mosaics + colors them, and returns one PNG. Because XYZ only ever
  requests the handful of tiles inside the current viewport, the client stays light **no matter how big
  the catalog is** (MPC does this over *millions* of scenes).

MPC's stack is [titiler-pgstac]: one mosaic endpoint, a per-tile STAC-database query to find
overlapping items, range-reads from Blob. We get the **same single-layer experience without a
database** using a **MosaicJSON** (§8) — a tiny static index of "which COG covers which tile," built
by walking our STAC catalog once. That's the design we build.

---

## 4. STAC's role here — the catalog we're verifying

**STAC (SpatioTemporal Asset Catalog)** is a JSON convention for describing geospatial datasets. rise-eo
already *queries* CDSE's STAC to discover imagery; here we *produce* one describing our inference
outputs (`src/rise_eo/catalog/stac.py`). The three levels we wrote:

- **Catalog** (`stac/catalog.json`) — the root; links to the collection.
- **Collection** (`stac/rise-eo-inference/collection.json`) — one dataset ("rise-eo inference outputs"), with a
  spatial+temporal extent and a **link to each of the 300 Items**.
- **Item** (`stac/rise-eo-inference/<cell_id>/<cell_id>.json`) — one output, with:
  - a **geometry + bbox** in EPSG:4326 (where on Earth it is),
  - `proj:code` / `proj:shape` / `proj:transform` (its native UTM grid),
  - and an **asset** named `output` whose `href` is the COG's path.

So the catalog is a browsable index: *root → collection → 300 items → 300 COGs*. "Does the STAC catalog
work?" concretely means: **can a client start at `collection.json`, walk the item links, read each
item's `output` asset href, and successfully render that COG?** That is exactly what our dashboard will
do — and it needs **no database**, because the catalog is static self-contained JSON with relative
links between the files (only the leaf asset hrefs are absolute local paths).

> **One wrinkle we already hit and fixed:** every output COG is named `output.tif`, so an earlier
> version derived the Item id from the filename stem and produced *300 links to one item*. It's now
> derived from the per-cell folder → 300 distinct Items (`E2E_AUSTRIA.md` Appendix C, `spec 26` note).
> The whole point of this dashboard is to *catch* that class of bug by eye.

---

## 5. Leaflet — the map in the browser

**[Leaflet]** is a tiny JavaScript map library. The entire "put a tiled layer on a map" idea is two
calls:

```js
const map = L.map('map').setView([48.7, 15.0], 10);          // center on Waldviertel, zoom 10
L.tileLayer('https://…/osm/{z}/{x}/{y}.png').addTo(map);      // a basemap for context
L.tileLayer('http://localhost:8000/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.png?url=/abs/merged.tif&…')
 .addTo(map);                                                  // OUR titiler layer on top
```

`L.tileLayer(template)` takes an **XYZ URL template** with `{z}/{x}/{y}` placeholders; as you pan/zoom,
Leaflet fills them in and fetches tiles. Pointing that template at **our titiler endpoint** is the
whole integration — Leaflet neither knows nor cares that the tiles are being generated on the fly from
a COG. We add:

- an **OSM basemap** underneath for geographic context,
- a **categorical legend** (class name → color), built from the same colors the crop-map figure uses
  (`demos/e2e_austria.py::CLASS_COLORS`), so the map is readable,
- optionally the 300 item **footprints** as a light GeoJSON overlay (from each Item's bbox) — that alone
  proves the catalog parses and geolocates before a single pixel loads.

**The dashboard adds exactly one raster layer** — the mosaic endpoint (§3.5/§8), pointed at a
MosaicJSON the server built by walking our STAC. Panning/zooming stays light because XYZ only requests
tiles in view. **How the STAC gets exercised:** the walk (collection.json → 300 items →
`assets.output.href`) happens **once, server-side at startup** to build the index; a broken catalog
(bad links, collided ids, missing hrefs) makes that build fail loudly or the map render wrong — which
is exactly the verification we want.

---

## 6. How the pieces fit — one combined app

At **startup**, the server walks the STAC once and builds a **MosaicJSON** (a static "which COG covers
which tile" index) from the item footprints + hrefs. Then it serves one origin:

```
browser (Leaflet page)                    one uvicorn process (demos/titiler_serve.py)
────────────────────────                  ─────────────────────────────────────────────
                                          [startup] walk stac/ → 300 items → build inference.mosaicjson
  GET  /                       ─────────▶  index → leaflet_dashboard.html
  GET  /config.json            ─────────▶  {mosaic tilejson url, colormap, legend, item_count, bounds}
  GET  /items.geojson          ─────────▶  300 footprints (from the walk) — light overlay
  GET  /mosaic/tiles/…/{z}/{x}/{y}.png ──▶  MosaicTilerFactory → per tile: look up covering COG(s) in
        ?url=inference.mosaicjson              the MosaicJSON → rio-tiler range-reads only those → PNG
  GET  /cog/info?url=…                 ──▶  TilerFactory (single-COG; for /cog/info debugging + optional
                                             merged.tif compare layer)
```

A **single FastAPI app** (one `uvicorn` command) does everything: it mounts titiler's **mosaic** router
at `/mosaic` (the main layer) and the single-COG router at `/cog` (debug/compare), builds the
MosaicJSON from the catalog, and serves the Leaflet HTML + `/items.geojson`. One origin ⇒ **no CORS
juggling, no second server, no `file://` problems** (a browser can't `fetch` a local file path). This
is the shape the spec pins.

---

## 7. The two subtleties that **will** bite (foreground these)

**(a) The outputs are categorical `uint8`, and titiler defaults to a *continuous* stretch.**
Our pixel values are **class codes 0–8** (and **255 = nodata**), not a smooth 0–255 ramp. If you serve
the COG with titiler's defaults, it linearly rescales the data across a color ramp and you get **noise**
— class 8 and class 1 blur into neighboring shades, nodata paints as a color. You **must** pass:
  - an explicit **discrete `colormap`** — a JSON dict `{class_code: color}` for **every** code 0–8,
    built from `CLASS_COLORS` (so code→color matches the crop-map figure), and
  - **`nodata=255`** so the 255 pixels render **transparent** (and request `.png`, which has an alpha
    channel), letting the basemap show through the ROI's slanted edges.
  This colormap+nodata step is the **#1 make-or-break detail**. A class map with defaults looks broken.

**(b) The COGs are UTM (EPSG:32633); XYZ tiles are Web-Mercator (EPSG:3857).**
The slippy-map grid (§2) is always 3857, but our rasters are in UTM. titiler **reprojects on the fly**
per tile — good, we don't pre-warp anything. But reprojection *resamples*, and for a **class map you
must resample with `nearest`** (the default for tiles): averaging class codes 3 and 7 into "5" would
invent a class that isn't there. **Never bilinear/cubic a categorical raster.** (Same principle as the
pipeline's reference-image resampling rule, applied to display.)

---

## 8. MosaicJSON — a single light layer, no database (what we build)

A **MosaicJSON** is a small static JSON index that maps map tiles (quadkeys) → the COGs that cover
them. It's the DB-free way to get MPC's one-layer experience (§3.5):

- **Build it by walking our STAC.** At startup the server reads `collection.json` → the 300 Items →
  each Item's `bbox` (footprint) + `assets.output.href` (COG path), and hands those to
  `cogeo-mosaic`'s `MosaicJSON.from_features` — which builds the tile→COG index **from the geometry
  alone, without opening a single COG** (the bounds are already in the STAC). So the catalog *is* the
  mosaic source, and walking it is the test.
- **Serve it as one endpoint.** titiler's mosaic factory exposes `/mosaic/tiles/{tms}/{z}/{x}/{y}.png
  ?url=inference.mosaicjson&…`. Per tile, it looks up the covering COG(s) in the index and range-reads
  only those. The browser adds **one** `L.tileLayer` — light, viewport-limited, seamless. Same
  colormap / nodata / nearest params as the single-COG case (§7).
- Our cells **don't overlap** (they tile the ROI), so there's no per-tile blending to reason about —
  each tile comes from exactly one cell's COG.

`merged.tif` is no longer the star: the mosaic **is** the seamless per-cell view, assembled live from
the catalog. We keep `merged.tif` only as an **optional compare layer** (does the catalog-driven mosaic
match the pipeline's pre-merged product?).

**The next rung (documented, not built): titiler-pgstac.** A MosaicJSON is a *precomputed* index —
rebuild it if the COGs change. [titiler-pgstac] instead serves mosaics from a **live pgSTAC Postgres
query** (what MPC runs), so the catalog is queried per-tile with no precompute — the right shape at
Azure/Batch scale (`ROADMAP` P5). We skip its local Postgres for this eyeball MVP; it's the natural
upgrade.

---

## 9. Scope & what changes later

- **Local only, v1.** COGs and the STAC live on disk; asset hrefs are absolute local paths, opened by
  GDAL directly. The dashboard is a QA/eyeball tool, not a hosted service.
- **Blob/S3 later (the storage seam).** When outputs live in Azure Blob / S3, the *only* thing that
  changes is the asset href (`/vsicurl/https://…` or a blob URL) and titiler's GDAL environment —
  titiler reads remote COGs by range-request natively. The Leaflet page is unchanged. (`TODO #17`, P1.)
- **pgSTAC at scale.** The production serving layer over a real catalog is `titiler-pgstac` (§8),
  which is where the Azure/Batch story reconnects (`ROADMAP` P5). This local demo is the on-ramp.

**Cross-links:** `E2E_AUSTRIA.md §8` (the run whose outputs we serve) · `TODO #14` (the serving layer's
north star) · `ROADMAP.md` P5 · the implementation spec `specs/27-titiler-leaflet-stac-verify.md`.

[rio-tiler]: https://cogeotiff.github.io/rio-tiler/
[FastAPI]: https://fastapi.tiangolo.com/
[Leaflet]: https://leafletjs.com/
[titiler-pgstac]: https://stac-utils.github.io/titiler-pgstac/
