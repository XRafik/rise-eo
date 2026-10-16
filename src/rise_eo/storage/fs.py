"""fsspec-based storage seam + first-class S3-compatible transport.

**Rule: a path that comes from a caller, a CLI flag or config may be a URL, so no other
module in rise-eo opens it directly.** Catalog, granules, datacubes and training arrays all read and
write through here, which is what makes local -> Azure Blob / S3 a config change rather
than a code change. Exceptions (ADR 0034): raster pixel reads, which go through
rasterio/GDAL VSI, and paths that are local by construction (scratch directories, the user
config file, a source tree read to package it).

Any S3-compatible store (AWS, CDSE EODATA, MinIO, …) is just an s3fs filesystem
distinguished by `endpoint_url` + keys in `storage_options`, so a granule download is
`transfer(src_s3_url, dst_url)`.

`path` may be a local path or an fsspec URL (`file://`, `s3://`, `az://`, …).
`storage_options` are backend kwargs, e.g. for CDSE S3::

    {"key": access, "secret": secret,
     "client_kwargs": {"endpoint_url": config.CDSE_S3_ENDPOINT_URL}}

Spec: specs/10-storage-and-scale.md
"""

from __future__ import annotations

import errno
import io
import json
import os
import shutil
from typing import Any

import fsspec
import fsspec.utils
import numpy as np

from rise_eo.storage.azure import to_vsi

__all__ = [
    "open",
    "exists",
    "makedirs",
    "put",
    "get",
    "ls",
    "glob",
    "save_npy",
    "load_npy",
    "read_parquet",
    "write_parquet",
    "peek_parquet_attrs",
    "rename",
    "transfer",
    "to_vsi",
    "is_local",
    "write_bytes",
    "write_text",
    "rm",
    "size",
    "modified",
    "read_geo",
    "find_sizes",
    "SOURCE_PATH_ATTRS_KEY",
]

# Writes here are deliberately not retried: see ARCHITECTURE.md §4, "the InvalidBlockList write
# retry". Retrying a deterministic id collision buries the real cause (#58).

# The upstream pandas/geopandas convention for JSON-encoding `.attrs` into the Parquet
# footer. Reusing the same key converges with a future geopandas release instead of
# forking a second convention.
PANDAS_ATTRS_FOOTER_KEY = b"PANDAS_ATTRS"

# Stamped onto `.attrs` by `read_parquet` so downstream code can tell "read from a file"
# from "hand-built in this process". Bookkeeping, not data -- `write_parquet` always
# strips it, so it never leaks an absolute local path into a written artifact.
SOURCE_PATH_ATTRS_KEY = "rise_eo:source_path"


# --- internal helpers --------------------------------------------------------


# Which extra ships each fsspec backend. A backend is never imported by rise-eo -- fsspec
# resolves it from the URL scheme -- so a missing one surfaces deep inside fsspec as
# "Install s3fs to access S3", naming a package rather than the extra that provides it.
_PROTOCOL_EXTRA = {
    "s3": "s3",
    "s3a": "s3",
    "az": "azure",
    "abfs": "azure",
    "abfss": "azure",
    "adl": "azure",
}


def _fs_and_path(url: str, storage_options: dict | None = None):
    """Resolve a URL/path to (filesystem, path-on-that-filesystem)."""
    try:
        return fsspec.core.url_to_fs(url, **(storage_options or {}))
    except ImportError as exc:
        extra = _PROTOCOL_EXTRA.get(fsspec.utils.get_protocol(url))
        if extra is None:
            raise
        raise ImportError(
            f"reading {url!r} needs the optional '[{extra}]' extra: "
            f"pip install 'rise-eo[{extra}]'."
        ) from exc


def is_local(path: str) -> bool:
    """True if `path` resolves to the local filesystem (vs. an `abfss://`/`s3://`/… URL).

    Guards code that must not apply local-path-only operations to a remote URL:
    `os.path.abspath` corrupts the scheme and host, `os.makedirs` and bare `open` cannot
    see it at all.
    """
    return fsspec.utils.get_protocol(path) in ("file", "local")


def _ensure_parent(fs, path: str) -> None:
    """Best-effort create of the parent directory (no-op on object stores)."""
    parent = path.rsplit("/", 1)[0] if "/" in path else ""
    if parent and parent != path:
        try:
            fs.makedirs(parent, exist_ok=True)
        except Exception:
            # Object stores have no real directories; ignore.
            pass


# --- generic I/O -------------------------------------------------------------


def open(path: str, mode: str = "rb", **storage_options: Any):
    """Open a file on any fsspec backend. Returns a context manager.

    >>> with open("out.txt", "w") as f: f.write("hi")
    """
    return fsspec.open(path, mode, **storage_options)


def exists(path: str, **storage_options: Any) -> bool:
    fs, p = _fs_and_path(path, storage_options)
    return fs.exists(p)


def makedirs(path: str, exist_ok: bool = True, **storage_options: Any) -> None:
    fs, p = _fs_and_path(path, storage_options)
    fs.makedirs(p, exist_ok=exist_ok)


def rm(path: str, recursive: bool = False, **storage_options: Any) -> None:
    fs, p = _fs_and_path(path, storage_options)
    fs.rm(p, recursive=recursive)


def put(local_path: str, remote_path: str, **storage_options: Any) -> None:
    """Upload a local file to a (possibly remote) destination."""
    fs, rpath = _fs_and_path(remote_path, storage_options)
    _ensure_parent(fs, rpath)
    fs.put_file(local_path, rpath)


def get(remote_path: str, local_path: str, **storage_options: Any) -> None:
    """Download a (possibly remote) file to local disk, for tools like rasterio/GDAL
    that need a real local path rather than an fsspec handle."""
    fs, rpath = _fs_and_path(remote_path, storage_options)
    parent = os.path.dirname(local_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fs.get_file(rpath, local_path)


def rename(src_path: str, dst_path: str, **storage_options: Any) -> None:
    """Move `src_path` onto `dst_path` on one fsspec filesystem, in a single `mv`.

    The atomic-publish primitive: on an HNS Azure account this is one metadata
    operation and locally it is `os.rename`, so a writer that saves to `src_path` and
    renames at the end leaves no window where a reader sees a partial artifact.

    Locally it calls `os.rename` **directly** rather than fsspec's `LocalFileSystem.mv`,
    which is `shutil.move` -- and `shutil.move` moves the source *inside* `dst_path` when
    that already exists as a directory, instead of failing. For a caller staging a
    directory and renaming it onto its final name, that silently
    turns a lost race into a corrupted destination rather than an error it can retry.
    `os.rename` gives the documented semantics: atomic replace for a file, `EEXIST`/
    `ENOTEMPTY` for a non-empty directory. Cross-device moves have no atomic form at all,
    so those fall back to fsspec's copy-and-delete.
    """
    fs, spath = _fs_and_path(src_path, storage_options)
    _, dpath = _fs_and_path(dst_path, storage_options)
    _ensure_parent(fs, dpath)
    if is_local(src_path) and is_local(dst_path):
        try:
            os.rename(spath, dpath)
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
    fs.mv(spath, dpath)


def ls(url: str, **storage_options: Any) -> list[str]:
    fs, p = _fs_and_path(url, storage_options)
    return fs.ls(p, detail=False)


def find_sizes(url: str, **storage_options: Any) -> dict[str, int]:
    """Recursively list `url`, returning `{path: size_in_bytes}` for every file under it.

    One paginated directory walk, so bulk presence checks cost a couple of requests
    rather than an `exists` + `size` round-trip per file. Raises if `url` does not exist
    — callers treating "no folder yet" as "nothing present" must handle that.
    """
    fs, p = _fs_and_path(url, storage_options)
    return {k: int(v.get("size") or 0) for k, v in fs.find(p, detail=True).items()}


def glob(pattern: str, **storage_options: Any) -> list[str]:
    fs, p = _fs_and_path(pattern, storage_options)
    return fs.glob(p)


def write_bytes(path: str, data: bytes, **storage_options: Any) -> None:
    """Write raw bytes to `path` (parent dirs created as needed)."""
    fs, p = _fs_and_path(path, storage_options)
    _ensure_parent(fs, p)
    with fs.open(p, "wb") as f:
        f.write(data)


def write_text(path: str, text: str, **storage_options: Any) -> None:
    """Write a text string to `path` as UTF-8. One call instead of an
    `fs.open(path, "w")` block, so callers don't hand-roll encoding."""
    write_bytes(path, text.encode("utf-8"), **storage_options)


def size(url: str, **storage_options: Any) -> int:
    """Byte size of a file (0 if empty). Used to distinguish a real download from a
    zero-byte "touched" leftover."""
    fs, p = _fs_and_path(url, storage_options)
    return fs.size(p)


def modified(url: str, **storage_options: Any) -> Any | None:
    """The *server's* last-modified time for a file, or `None` if the backend records
    none.

    The server's clock, not this process's — which is what makes it usable for measuring
    driver-vs-storage clock skew. `None` rather than a raise, so a caller measuring skew
    reports "unavailable" instead of silently reporting zero skew.
    """
    fs, p = _fs_and_path(url, storage_options)
    try:
        return fs.modified(p)
    except (NotImplementedError, KeyError, AttributeError):
        return None


# --- typed helpers -----------------------------------------------------------


def save_npy(
    path: str, arr, allow_pickle: bool = False, **storage_options: Any
) -> None:
    """Save a numpy array (or pickled object, if allow_pickle) to `path`.

    `path` must include the `.npy` extension (numpy does not append one when
    writing to a file object).
    """
    fs, p = _fs_and_path(path, storage_options)
    _ensure_parent(fs, p)
    with fs.open(p, "wb") as f:
        np.save(f, arr, allow_pickle=allow_pickle)


def load_npy(path: str, allow_pickle: bool = False, **storage_options: Any):
    """Load a `.npy` file. For a pickled object saved as a 0-d array, the caller
    extracts it with `[()]` (kept explicit to avoid surprising unwrapping)."""
    fs, p = _fs_and_path(path, storage_options)
    with fs.open(p, "rb") as f:
        return np.load(io.BytesIO(f.read()), allow_pickle=allow_pickle)


def read_geo(path: str, **storage_options: Any):
    """Read a vector file (GeoJSON/shapefile/…) -> GeoDataFrame, through the storage seam.

    **Never hand a path straight to `gpd.read_file`.** GDAL/pyogrio has no `abfss://`
    driver, so an fsspec-only scheme never reaches a reader and the failure surfaces as
    `DataSourceError: <abfss url>: No such file or directory` — for a file that
    demonstrably exists. fsspec understands the scheme; GDAL does not.

    Local paths pass through unchanged, so callers need no is-it-remote branch. (#47)
    """
    import geopandas as gpd

    fs, p = _fs_and_path(path, storage_options)
    with fs.open(p, "rb") as f:
        return gpd.read_file(io.BytesIO(f.read()))


def _decode_pandas_attrs(footer_metadata: dict | None) -> dict:
    """`pyarrow` footer metadata -> the restored `.attrs` dict, or `{}` if absent."""
    if not footer_metadata:
        return {}
    raw = footer_metadata.get(PANDAS_ATTRS_FOOTER_KEY)
    if raw is None:
        return {}
    return json.loads(raw.decode("utf-8"))


def read_parquet(path: str, **storage_options: Any):
    """Read a (Geo)Parquet file -> GeoDataFrame.

    Restores `.attrs` from the footer's `PANDAS_ATTRS` key if present — geopandas' own
    reader does not, unlike pandas' — and stamps `attrs[SOURCE_PATH_ATTRS_KEY] = path`.
    """
    import geopandas as gpd

    fs, p = _fs_and_path(path, storage_options)
    with fs.open(p, "rb") as f:
        raw = f.read()
    gdf = gpd.read_parquet(io.BytesIO(raw))
    import pyarrow.parquet as pq

    attrs = _decode_pandas_attrs(pq.read_metadata(io.BytesIO(raw)).metadata)
    if attrs:
        gdf.attrs.update(attrs)
    gdf.attrs[SOURCE_PATH_ATTRS_KEY] = path
    return gdf


def write_parquet(path: str, df, **storage_options: Any) -> None:
    """Write a (Geo)DataFrame to `path` as (Geo)Parquet, JSON-encoding `df.attrs` into
    the footer under `PANDAS_ATTRS`. `SOURCE_PATH_ATTRS_KEY` is always stripped first —
    it is read-side bookkeeping and would leak a local path into the artifact.

    The attrs-present path re-encodes the table through pyarrow, so row-group layout and
    compression follow pyarrow's defaults rather than `to_parquet`'s. Harmless at
    catalog size (KB–MB); don't route a large dataframe through it expecting no cost.
    Empty attrs skip the re-encode entirely.
    """
    fs, p = _fs_and_path(path, storage_options)
    _ensure_parent(fs, p)
    buf = io.BytesIO()
    attrs = {k: v for k, v in df.attrs.items() if k != SOURCE_PATH_ATTRS_KEY}
    # Strip before to_parquet, not after: geopandas >= 1.2 serializes df.attrs into the
    # footer itself, so a stamp left on `df` would leak. Shallow copy: caller's attrs stay.
    clean = df.copy(deep=False)
    clean.attrs = dict(attrs)
    clean.to_parquet(buf)

    if attrs:
        import pyarrow.parquet as pq

        buf.seek(0)
        table = pq.read_table(buf)
        metadata = dict(table.schema.metadata or {})
        metadata[PANDAS_ATTRS_FOOTER_KEY] = json.dumps(attrs).encode("utf-8")
        buf = io.BytesIO()
        pq.write_table(table.replace_schema_metadata(metadata), buf)

    with fs.open(p, "wb") as f:
        f.write(buf.getvalue())


def peek_parquet_attrs(path: str, **storage_options: Any) -> dict:
    """Read only the Parquet footer — no row group — and return the restored `.attrs`,
    or `{}`. Cheap inspection of a catalog's declaration stamp without loading it."""
    import pyarrow.parquet as pq

    fs, p = _fs_and_path(path, storage_options)
    with fs.open(p, "rb") as f:
        metadata = pq.read_metadata(f)
    return _decode_pandas_attrs(metadata.metadata)


# --- first-class S3-compatible transport -------------------------------------


def transfer(
    src_url: str,
    dst_url: str,
    *,
    src_options: dict | None = None,
    dst_options: dict | None = None,
) -> None:
    """Copy one object between fsspec filesystems (provider-agnostic).

    A granule band-file download is `transfer(s3_src, local_or_blob_dst)`. Bytes are
    streamed, so source and destination need not share a backend (e.g. CDSE S3 -> Azure
    Blob); configure each side through `src_options`/`dst_options`.

    **Atomic:** bytes go to a `.part` sidecar and are renamed onto `dst_url` only after
    the copy fully succeeds, so a failed or killed transfer never leaves a truncated file
    at the destination — which is what makes an existence check a safe "already
    downloaded" signal on resume.
    """
    src_fs, spath = _fs_and_path(src_url, src_options)
    dst_fs, dpath = _fs_and_path(dst_url, dst_options)
    _ensure_parent(dst_fs, dpath)
    tmp = f"{dpath}.part"
    try:
        with src_fs.open(spath, "rb") as fsrc, dst_fs.open(tmp, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst)
        dst_fs.mv(tmp, dpath)  # atomic on a local fs (os.rename)
    except BaseException:
        try:
            dst_fs.rm(tmp)
        except Exception:
            pass
        raise
