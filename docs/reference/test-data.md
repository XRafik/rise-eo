# Reference — test data

What test data rise-eo has, where it lives, and the traps in it. Kept true in the same PR as any change
to it.

## 1. In git: the tutorial micro-fixture

`tests/data/tutorial/` — 36 real Sentinel-2 L2A granules of MGRS tile T33UWP, cropped to a small ROI
(B04, B08, SCL; 2018-04-01 → 2018-09-28; ~27.5 MB), with `fields.geojson` labels. The fast suite and
`docs/tutorial.md` run on it. Its `README.md` records how it was generated; its `NOTICE` gives the
licences.

## 2. In git: test geometries

`notebooks/shapefiles/` — read its `NOTICE` before you use one.

| File | What it is | Use it for |
|---|---|---|
| `AT_ROI.geojson` | Demo ROI over Austria, across 4 MGRS tiles (all in UTM zone 33, so one CRS) | multi-MGRS-tile runs |
| `AT_2018_TRAIN.geojson` | 900 EuroCrops fields (`fid`, `crop`), CC BY 4.0 | training data |
| `s2grid=4772924.geojson` | One grid cell inside T33UWP, inside `AT_ROI`; 43 labelled fields fall in it | anything that needs imagery **and** labels |
| `s2grid=476da24.geojson` | One grid cell inside T33UWP, 47.7 km outside `AT_ROI`; **no** labels fall in it | single-MGRS-tile imagery tests only |

**Read the bounds before you pair an ROI with labels.** A filename describes where the data came
from, not where it is. Spec 58 D18 paired a file named `austria_eurocrops_…geojson` (it lies in
Ethiopia) with an Austrian grid cell; the run-book would have produced no training data at all.
Three lines catch it:

```python
import geopandas as gpd
cell, labels = gpd.read_file("notebooks/shapefiles/s2grid=4772924.geojson"), gpd.read_file("notebooks/shapefiles/AT_2018_TRAIN.geojson")
print(cell.total_bounds, labels.to_crs(cell.crs).total_bounds)                # do they overlap at all?
print(int(labels.to_crs(cell.crs).intersects(cell.geometry.iloc[0]).sum()))   # must be > 0 (43 here)
```

## 3. Not in git: the Austria archive (real imagery)

> **Current rise-eo cannot read this archive yet.** It was downloaded on 2026-09-07, before spec 59
> changed the catalog schema. Its `catalog.parquet` lacks `acquisition_key`, `processing_version`,
> `processing_datetime` and `source`, so `TileCatalog.read` refuses it and asks for a re-download.
> Spec 59 D12 re-downloads it into the shared local archive `tests/outputs/imagery/`; that run
> has not happened yet (spec 59 P2, "the full Austria re-download"). Until then, real-imagery runs
> need that re-download first.

`tests/outputs/demo_e2e/imagery/` (gitignored). Downloaded from Microsoft Planetary Computer on
2026-09-07 by `runbooks/58-redownload-austria-mpc.md`. The table describes that download; spec 59
D12 re-fetches the same 184 granules and bands from MPC.

| | |
|---|---|
| Window | 2018-04-01 → 2018-09-30, ROI `AT_ROI.geojson` |
| MGRS tiles | T33UVP, T33UWP, T33UVQ, T33UWQ |
| Granules | 184 (552 files, 67.2 GB), `max_cloudcover=50` |
| Bands | B04, B08, SCL |
| Radiometric offset | 0: MPC serves the original 2018 processing (baseline 02.12, before ESA's 04.00 offset) |

Traps:

- **B8A is missing.** It was dropped to fit on disk. `demos/e2e_austria.py` still asks for
  `B04, B08, B8A, SCL`, so running it unchanged downloads B8A (~28 GB more). Any check that needs
  `nir08` (= B8A) needs a B8A pass first.
- **Never assume a fixed offset.** CDSE serves the 2023 reprocessing (offset −1000) for the same
  dates; derive the offset from the baseline in each catalog row's `properties` (spec 58 D12).
- **No real imagery spans two CRSs.** The four MGRS tiles share UTM zone 33. The multi-CRS set (Ethiopia,
  two UTM zones) was deleted; specs and run-books that mention it are history.

Write outputs from real-data runs under `tests/outputs/` (gitignored) and look at them in QGIS.
