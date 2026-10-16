# Manual tests — `rise_eo.storage.fs`

Verifies the storage seam (spec `10-storage-and-scale.md`): generic fsspec I/O on
the **local** backend (Section A) and the S3-compatible **transport** (Section B,
needs credentials).

Work top-to-bottom in **one Python session** (paste each block in order; later
blocks reuse variables). Tick the box when the output matches.

---

## 0. Setup (once)

From the repo root (`fsd/`), create an environment and install in editable mode:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Confirm the package and storage module import:

```bash
python -c "import rise_eo; from rise_eo.storage import fs; print('fsd', rise_eo.__version__, '| storage import OK')"
```

- [x] Prints `fsd 0.0.1 | storage import OK`

Now start an interactive session and keep it open for the rest of Section A:

```bash
python
```

```python
import os, shutil, tempfile
import numpy as np
import geopandas as gpd
import shapely.geometry as sg
from rise_eo.storage import fs

WORK = tempfile.mkdtemp(prefix="fsd_storage_")
print("WORK =", WORK)
```

- [x] Prints a temp dir path (we clean it up in step A8)

---

## Section A — local backend

### A1. makedirs + exists

```python
sub = os.path.join(WORK, "a/b/c")
fs.makedirs(sub)
print(fs.exists(sub), fs.exists(os.path.join(WORK, "nope")))
```

- [x] `True False`

### A2. open (write/read text)

```python
p_txt = os.path.join(WORK, "hello.txt")
with fs.open(p_txt, "w") as f:
    f.write("hi fsd")
with fs.open(p_txt, "r") as f:
    print(repr(f.read()), fs.exists(p_txt))
```

- [x] `'hi fsd' True`

### A3. save_npy / load_npy (array)

```python
arr = np.arange(12).reshape(3, 4)
p_arr = os.path.join(WORK, "arr.npy")
fs.save_npy(p_arr, arr)
out = fs.load_npy(p_arr)
print(out.shape, out.dtype, bool((out == arr).all()))
```

- [x] `(3, 4) int64 True`

### A4. save_npy / load_npy (pickled metadata dict)

This is exactly how datacube `metadata.pickle.npy` round-trips (note the `[()]`).

```python
meta = {"bands": ["B02", "B08"], "timestamps": 3}
p_meta = os.path.join(WORK, "meta.pickle.npy")
fs.save_npy(p_meta, meta, allow_pickle=True)
loaded = fs.load_npy(p_meta, allow_pickle=True)[()]
print(loaded == meta, loaded)
```

- [x] `True {'bands': ['B02', 'B08'], 'timestamps': 3}`

### A5. write_parquet / read_parquet (GeoDataFrame)

This is the tile-catalog format (GeoParquet, geometry + CRS preserved).

```python
gdf = gpd.GeoDataFrame(
    {"id": ["t1", "t2"]},
    geometry=[sg.Point(0, 0), sg.Point(1, 1)],
    crs="EPSG:4326",
)
p_pq = os.path.join(WORK, "catalog.parquet")
fs.write_parquet(p_pq, gdf)
back = fs.read_parquet(p_pq)
# GeoParquet stores the CRS as PROJJSON, so str(back.crs) is the verbose JSON
# blob — compare the EPSG code instead, which is what's actually preserved.
print(len(back), str(back.crs.to_epsg()), list(back["id"]), back.geometry.iloc[0].wkt)
```

- [x] `2 4326 ['t1', 't2'] POINT (0 0)`

### A6. ls / glob

```python
print(sorted(os.path.basename(x) for x in fs.ls(WORK)))
print(sorted(os.path.basename(x) for x in fs.glob(os.path.join(WORK, "*.npy"))))
```

- [x] First line includes: `['a', 'arr.npy', 'catalog.parquet', 'hello.txt', 'meta.pickle.npy']`
- [x] Second line: `['arr.npy', 'meta.pickle.npy']`

### A7. put / get / transfer

```python
# put: upload a local file to a destination (local stands in for "remote")
p_put = os.path.join(WORK, "uploaded/arr_copy.npy")
fs.put(p_arr, p_put)

# get: download it back to a fresh local path (parent dir auto-created)
p_get = os.path.join(WORK, "downloaded/arr_back.npy")
fs.get(p_put, p_get)

# transfer: copy between filesystems (here local -> local)
p_xfer = os.path.join(WORK, "transferred/arr_t.npy")
fs.transfer(p_arr, p_xfer)

print(
    fs.exists(p_put),
    bool(np.array_equal(fs.load_npy(p_get), arr)),
    bool(np.array_equal(fs.load_npy(p_xfer), arr)),
)
```

- [x] `True True True`

### A8. teardown

```python
shutil.rmtree(WORK)
print("cleaned:", not os.path.exists(WORK))
```

- [x] `cleaned: True`

You can `exit()` the session now (Section B starts a fresh one).

---

## Section B — S3-compatible transport (needs credentials)

Proves the same `transfer` / `ls` work against a real S3 endpoint, which is what a
CDSE tile download will be. Two ways to run it:

### B1. Anonymous public S3 (no credentials)

Confirms s3fs + the seam work end-to-end without secrets. (Replace the bucket/key
with any small public object you know; AWS Open Data buckets work with `anon`.)

```python
import os, tempfile
from rise_eo.storage import fs

WORK = tempfile.mkdtemp(prefix="fsd_s3_")
opts = {"anon": True}

# list a public prefix (example shape; swap in a real public bucket/prefix)
print(fs.ls("s3://<public-bucket>/<prefix>/", **opts)[:5])

dst = os.path.join(WORK, "sample.bin")
fs.transfer("s3://<public-bucket>/<prefix>/<small-object>", dst, src_options=opts)
print("downloaded:", fs.exists(dst), os.path.getsize(dst), "bytes")
```

- [ ] `ls` returns a non-empty list
- [ ] `downloaded: True <n> bytes`

### B2. CDSE EODATA (real S3 keys)

This is the actual tile-download path. You need CDSE S3 access/secret keys and a
real `.SAFE` band object key. (After `sources/cdse` is implemented, a catalog query
gives you `s3url`s; for now paste one you obtain from the CDSE browser or legacy
code.)

```python
import os, tempfile
from rise_eo import config
from rise_eo.storage import fs

WORK = tempfile.mkdtemp(prefix="fsd_cdse_")
src_opts = {
    "key": "YOUR_S3_ACCESS_KEY",
    "secret": "YOUR_S3_SECRET_KEY",
    "client_kwargs": {"endpoint_url": config.CDSE_S3_ENDPOINT_URL},
}

# list the contents of one product (use a real EODATA prefix)
print(fs.ls("s3://eodata/Sentinel-2/MSI/L2A/.../<PRODUCT>.SAFE", **src_opts)[:10])

# transfer one band file to local disk
src = "s3://eodata/Sentinel-2/MSI/L2A/.../<PRODUCT>.SAFE/.../<BAND>.jp2"
dst = os.path.join(WORK, "band.jp2")
fs.transfer(src, dst, src_options=src_opts)
print("downloaded:", fs.exists(dst), os.path.getsize(dst), "bytes")
```

- [ ] `ls` lists the product's objects
- [ ] `downloaded: True <n> bytes` (a non-trivial size)

> Tip: never paste real keys into committed files. Use env vars
> (`os.environ["CDSE_S3_KEY"]`) or a local untracked `credentials.json`
> (already covered by `.gitignore`).

---

## Notes / known caveats

- **rasterio/GDAL reads remote rasters via VSI, not fsspec** (spec `10`). So when
  the datacube builder reads a remote `.jp2` directly, that's a GDAL `/vsis3/` path,
  not this module. For *downloading whole files* (the tile-download path), `transfer`
  / `get` here is correct.
- `transfer` is single-object in v1 (`njobs` is reserved); bulk parallelism is the
  caller's job (the CDSE source fans out many `transfer` calls).

## To extend this guide
Ask: *"augment storage.md with tests for X"* — e.g. overwrite semantics, large-file
streaming, Azure Blob (`az://`) once the `[azure]` extra is wired, or error cases
(missing file, bad credentials).
