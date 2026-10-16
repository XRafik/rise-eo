# Manual test — COG-on-download (spec 14)

Proves `sources.cdse.download(cog=True)` (the default) fetches Sentinel-2 L2A bands
and writes them as **lossless COGs with overviews** (`Bxx.tif`), that the catalog
records `.tif`, and that a datacube build over the COGs matches a build over the
native JP2. Needs CDSE S3 credentials (`secrets/cdse_credentials.json`) and network.

Unit tests (`tests/test_cdse.py`, `tests/test_raster.py`) cover the logic without
network; this guide confirms it on genuine tile bytes. Work top-to-bottom in one
Python session; tick a box when it passes.

---

## 0. Setup

From `fsd/` with the dev env active (`source .venv/bin/activate`):

```python
import os, datetime, tempfile
import geopandas as gpd
import rasterio, numpy as np
from rise_eo.sources import cdse
from rise_eo.sources.cdse import CdseCredentials
from rise_eo.catalog.catalog import TileCatalog

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
creds = CdseCredentials.from_json(os.path.join(os.getcwd(), "secrets/cdse_credentials.json"))
roi = gpd.read_file(os.path.join(ROOT, "shapefiles/s2grid=476da24.geojson"))  # single-tile Austria
start, end = datetime.datetime(2018, 6, 1), datetime.datetime(2018, 6, 15)
out = tempfile.mkdtemp(prefix="cog_dl_")
```

## 1. Download with COG conversion (default)

```python
cat = TileCatalog(os.path.join(out, "catalog.parquet"))
res = cdse.download(roi, start, end, ["B04", "B08", "SCL"], out, cat, creds,
                    max_tiles=5, max_cloudcover=80.0, progress=True)  # cog=True default
print(res)
```

- [ ] Completes with `failed_count == 0`.
- [ ] On disk the band files are **`B04.tif` / `B08.tif` / `SCL.tif`** (not `.jp2`), and no
      `*.src.jp2` staging files remain:
      ```python
      import glob
      print(sorted(os.path.basename(p) for p in glob.glob(f"{out}/**/*", recursive=True)
                   if os.path.isfile(p)))
      ```
- [ ] The catalog `files` column lists `.tif`:
      ```python
      print(cat.read()["files"].tolist())
      ```

## 2. Confirm they are real COGs (overviews, tiled, DEFLATE)

```python
tif = glob.glob(f"{out}/**/B04.tif", recursive=True)[0]
with rasterio.open(tif) as d:
    print("driver", d.driver, "| overviews", d.overviews(1),
          "| compress", d.profile["compress"], "| tiled", d.profile["tiled"])
```

- [ ] `overviews(1)` is non-empty (e.g. `[2, 4, 8, 16]`), `compress == "deflate"`, `tiled is True`.
- [ ] (Optional, CLI) `gdalinfo <B04.tif>` reports `LAYOUT=COG` and lists overviews / `Block=512x512`.

## 3. Lossless vs a JP2 re-download

```python
out_jp2 = tempfile.mkdtemp(prefix="jp2_dl_")
cat_jp2 = TileCatalog(os.path.join(out_jp2, "catalog.parquet"))
cdse.download(roi, start, end, ["B04"], out_jp2, cat_jp2, creds,
              max_tiles=5, max_cloudcover=80.0, cog=False)  # native JP2
jp2 = glob.glob(f"{out_jp2}/**/B04.jp2", recursive=True)[0]
with rasterio.open(jp2) as a, rasterio.open(tif) as b:
    print("bit-identical:", np.array_equal(a.read(), b.read()))
```

- [ ] Prints `bit-identical: True` (COG conversion loses no information).

## 4. Datacube parity (COG vs JP2 catalog)

Build a small datacube from the COG catalog and from the JP2 catalog over the same ROI +
window and confirm the arrays match (rasterio reads either format; the build is
format-agnostic). Use `datacube.md` as the pattern for the build call, pointing
`create_datacube.setup` at each catalog.

- [ ] The two datacubes are equal (`np.array_equal` on the `datacube.npy` arrays), proving the
      COG ingest is a drop-in for the build path.

## 5. Cleanup

```python
import shutil
shutil.rmtree(out); shutil.rmtree(out_jp2)
```

---

### Seam boundary (no network needed)

`cog=True` requires a **local** `root_folderpath`. A remote destination raises:

```python
cdse.download(roi, start, end, ["B04"], "s3://bucket/out", cat, creds, max_tiles=5)
# -> ValueError: COG-on-download (cog=True) needs a local root_folderpath in v1 ...
```

- [ ] Raises the clear seam error (use `cog=False` for a remote dst until the
      stage-local→convert→upload path lands with the Azure milestone).
