# How to: download real imagery (and why a small ROI doesn't make it small)

> **Last verified:** 2026-07-31 @ `df98463` (spec 41 D5 tier 2 — "dated"). Re-verify if
> `rise_eo.download`'s granule selection, `max_tiles` preflight, or the CDSE/MPC byte-per-granule
> profile changes.

The tutorial never downloaded anything — it ran entirely on a committed fixture. This page is what
you read before you run `rise_eo.download` for real, so the byte budget doesn't surprise you mid-run.

## The one fact that makes this page necessary

**rise-eo reads whole ~110 km MGRS-tile granules — there is no windowed/partial-tile read** (see
[`ARCHITECTURE.md`](../../ARCHITECTURE.md)'s invariants and `LIMITATIONS.md`). A granule averages
**~426 MB** across four bands (B04 184 MB + B08 187 MB + B8A 51 MB + SCL 4.6 MB, measured). Your ROI
picks *which* granules intersect it — a 5 km ROI and a 100 km ROI inside the same MGRS tile pull the
**same granule**, byte for byte. **Shrinking the ROI does not shrink the download**; only shrinking
the date window or the band list does.

| Config | Granules | Download | where the number comes from |
|---|---|---|---|
| 1 MGRS tile, Apr–Sep, 4 bands | ~52 | ~18 GB | **measured** — the 2018 Austria archive, 207 granules / 4 tiles = 74 GB / 4 |
| 1 MGRS tile, 1 month, B04+B08+SCL | ~9 | ~3.4 GB | estimated, 426 MB × ¾ |
| 1 MGRS tile, 2 weeks, B04+B08+SCL | ~4 | ~1.5 GB | estimated, 426 MB × ¾ |

**Estimates run high, on purpose.** 426 MB is the sum of four band files at their measured sizes;
averaged over a whole real archive it works out nearer **~357 MB/granule** (74 GB ÷ 207), because
these are compressed rasters and compression varies with scene content. So the multiplier gives you
a ceiling and row 1 (measured over 207 real granules) is what an actual season cost. Budget with
426 MB; expect to land under it.

This is exactly why the tutorial ships a **committed micro-fixture** instead of a "small" live
download (spec 41 D11) — there is no real-download configuration that is tutorial-sized.

## Size your download before running it

Query the granule count for your ROI/window/bands through your source's own STAC search before
calling `rise_eo.download` — no bytes move for a query. `rise_eo.download`'s own preflight does the same
count internally and refuses the run if it exceeds `max_tiles` (below), so sizing first just lets
you make that decision on purpose instead of by trial and error.

```python
from rise_eo import grid

n_cells = len(grid.roi_to_s2_grids("your_roi.geojson", grid_size_km=5))
print(n_cells, "grid cells")   # cells, not granules -- see the note below
```

Grid-cell count and granule count are **different numbers** — one MGRS granule covers many 5 km
grid cells. Use your source's STAC API (CDSE OData, or `pystac_client` against MPC) filtered to
your ROI's bounding geometry, date window, and MGRS tile(s) to get the real granule count; multiply
by ~426 MB × (bands you're keeping ÷ 4) for a byte estimate.

### From the command line (CDSE only): preview, then a run you can stop

`rise_eo.sources.download_cli` is a thin CLI over CDSE's `download_resume`. `--dry-run` reads metadata
only (no band bytes) and reports what the real run would fetch. `--stop-file` lets you stop a long
run cleanly from another terminal; re-running the same command resumes it.

```bash
# preview: metadata only
.venv/bin/python -m rise_eo.sources.download_cli \
  --roi my_roi.geojson --start 2018-01-01 --end 2019-01-01 \
  --bands B04 B08 SCL --dst data/imagery --catalog data/imagery/sentinel-2-l2a/catalog.parquet \
  --max-tiles 600 --dry-run

# real run; `touch /tmp/rise-eo.stop` from another terminal stops it cleanly
.venv/bin/python -m rise_eo.sources.download_cli \
  --roi my_roi.geojson --start 2018-01-01 --end 2019-01-01 \
  --bands B04 B08 SCL --dst data/imagery --catalog data/imagery/sentinel-2-l2a/catalog.parquet \
  --max-tiles 600 --stop-file /tmp/rise-eo.stop --creds cdse_credentials.json
```

It writes `<dst>/_result.json`. The exit code is the verdict: 0 on a clean finish or a user stop,
non-zero on failures, a tripped circuit breaker or an unresolved pool break. `--help` lists the
rest (`--max-cloudcover`, `--max-passes`, `--no-cog`, …).

## `max_tiles` is a required, not optional, guardrail

```python
import rise_eo
from rise_eo.sources.cdse import CdseCredentials

catalog = rise_eo.download(
    roi="your_roi.geojson",
    startdate=..., enddate=...,
    bands=["B04", "B08", "B8A", "SCL"],
    dst_folderpath="data/imagery",   # one shared archive, reused across runs
    creds=CdseCredentials.from_env(),
    max_tiles=20,   # refuses the run if more than 20 granules would be pulled
)
```

There is no default — you must decide a number, and preflight checks the real granule count
against it before any bytes move. If it refuses, that's the guardrail working: narrow the window or
the ROI, or raise `max_tiles` deliberately having just read the table above.

## Two sources, one call shape

- **`source="cdse"`** (default) — Copernicus Data Space Ecosystem. Requires `creds`
  (`CdseCredentials.from_env()` or your own). CDSE is the reference source (spec 32) and the one
  this repo's own `demos/` benchmarks measure against.
- **`source="mpc"`** — Microsoft Planetary Computer. Anonymous — no `creds` needed, and `cog=` is
  ignored because MPC assets are already COG. This is what run-books use for cluster runs (no
  secret to provision on a compute node).

Both write the same `TileCatalog` schema; nothing downstream (`create_training_data`,
`run_inference`) cares which source built the catalog.

## Where to go next

- [`your-own-region.md`](your-own-region.md) — the rest of the pipeline over what you just
  downloaded.
- [`run-at-scale.md`](run-at-scale.md) — `runner="aml"` dispatches the download itself onto a
  cluster instead of your laptop, colocated with blob storage.
