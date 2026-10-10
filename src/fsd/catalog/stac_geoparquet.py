"""stac-geoparquet export: a list of `pystac.Item` -> one GeoParquet file, and back.

Spec: specs/30-tier2-mini-mpc-validation.md

Uses the `stac-geoparquet` library (optional `[serving]` extra, isolated here like
`grid.py` so the core `.venv` stays lean). Not wired into any default write path (#26).
Written against `stac-geoparquet` 0.8.1 (unpinned in the extra):
`stac_geoparquet.arrow.parse_stac_items_to_parquet` and `stac_table_to_items`.

Writing stages a local tmp file: the library wants a real filesystem path (it opens
`output_path` itself), so the result is moved to the possibly remote path with `fs.put`.
Reading goes the other way through `fs.open` bytes, never a tmp file.
"""

from __future__ import annotations

import io
import os
import tempfile

import pystac

from fsd.storage import fs


def items_to_stac_geoparquet(items: list[pystac.Item], dst_filepath: str) -> str:
    """Write `items` to a single GeoParquet file at `dst_filepath` (local or fsspec URL).

    Returns `dst_filepath`. Raises `ValueError` on an empty `items` (nothing to export — same
    contract as `catalog.stac.write_stac_catalog`).
    """
    if not items:
        raise ValueError("items_to_stac_geoparquet: no items to export.")

    from stac_geoparquet.arrow import parse_stac_items_to_parquet

    with tempfile.TemporaryDirectory() as tmpdir:
        local_fp = os.path.join(tmpdir, "catalog.parquet")
        parse_stac_items_to_parquet([it.to_dict() for it in items], output_path=local_fp)
        fs.put(local_fp, str(dst_filepath))
    return str(dst_filepath)


def stac_geoparquet_to_items(src_filepath: str) -> list[pystac.Item]:
    """Read a GeoParquet file at `src_filepath` (local or fsspec URL) back to `pystac.Item`s
    (the inverse of `items_to_stac_geoparquet`, for round-trip validation)."""
    import pyarrow.parquet as pq
    from stac_geoparquet.arrow import stac_table_to_items

    with fs.open(str(src_filepath), "rb") as f:
        table = pq.read_table(io.BytesIO(f.read()))
    return [pystac.Item.from_dict(d) for d in stac_table_to_items(table)]
