"""Spec 37 Amendment A1 (#145 X3): `mpc.download_shard` refuses a shard row that lacks
one of the 16 keys `discover_shard_rows` writes, instead of filling in a default.

Offline: `_import_pc_sign` and `_transfer_and_stamp_one` are recorders.
"""

from __future__ import annotations

import fsspec
import pandas as pd
import pytest

from rise_eo.sources import mpc
from rise_eo.storage import fs
from rise_eo.workflows import download as download_cli


def _full_row(dst: str) -> dict:
    """One shard row with all 16 keys `discover_shard_rows` writes."""
    return {
        "tile_id": "T1", "band": "B04", "href": "https://mpc/B04.tif", "dst": dst,
        "offset": 0, "collection": "sentinel-2-l2a",
        "timestamp": "2018-06-01T00:00:00+00:00", "s3url": "", "cloud_cover": 0.0,
        "scale": 0.0001, "nodata": 0, "acquisition_key": "T1",
        "processing_version": "05.00", "processing_datetime": "2018-06-01T00:00:00+00:00",
        "properties": "{}", "geometry": "POINT (0 0)",
    }


class _RecordingCatalog:
    """Stands in for `TileCatalog`: records every `append` call."""

    def __init__(self):
        self.appended: list[list[dict]] = []

    def append(self, rows, declaration=None):
        self.appended.append(rows)


@pytest.fixture
def recorders(monkeypatch):
    """Stub the two network-touching calls; return the lists of their calls."""
    sign_calls: list[int] = []
    transfer_calls: list[tuple] = []

    def _fake_import_pc_sign():
        sign_calls.append(1)
        return lambda url: url + "?sig=fake"

    def _fake_transfer(src, dst, *, band, offset, declaration, sign=None):
        transfer_calls.append((src, dst))
        return True, "ok"

    monkeypatch.setattr(mpc, "_import_pc_sign", _fake_import_pc_sign)
    monkeypatch.setattr(mpc, "_transfer_and_stamp_one", _fake_transfer)
    return sign_calls, transfer_calls


def test_t1_missing_keys_raise_before_any_work(recorders, tmp_path):
    sign_calls, transfer_calls = recorders
    row = _full_row(str(tmp_path / "T1" / "B04.tif"))
    del row["scale"], row["properties"]
    catalog = _RecordingCatalog()

    with pytest.raises(ValueError) as exc:
        mpc.download_shard([row], str(tmp_path), catalog)

    msg = str(exc.value)
    assert "['properties', 'scale']" in msg
    assert "discover_shard_rows" in msg
    assert "different fsd version" in msg
    assert sign_calls == []
    assert transfer_calls == []
    assert catalog.appended == []


def test_t2_nan_processing_values_become_none(recorders, tmp_path):
    row = _full_row(str(tmp_path / "T1" / "B04.tif"))
    row["processing_version"] = float("nan")
    row["processing_datetime"] = float("nan")
    catalog = _RecordingCatalog()

    mpc.download_shard([row], str(tmp_path), catalog)

    (appended,) = catalog.appended
    assert appended[0]["processing_version"] is None
    assert appended[0]["processing_datetime"] is None


def test_t3_run_shard_fails_without_writing_status(recorders):
    root = "memory://x3_keys"
    shard_url = f"{root}/shards/0.csv"
    status_url = f"{root}/_status/0.json"
    row = _full_row(f"{root}/data/T1/B04.tif")
    del row["scale"]
    try:
        with fs.open(shard_url, "w") as f:
            pd.DataFrame([row]).to_csv(f, index=False)

        with pytest.raises(ValueError) as exc:
            download_cli.run_shard(
                shard_url=shard_url, dst=f"{root}/data",
                catalog=f"{root}/data/catalog.parquet", status_url=status_url,
            )

        msg = str(exc.value)
        assert "scale" in msg
        assert "discover_shard_rows" in msg
        assert "different fsd version" in msg
        assert not fs.exists(status_url)
    finally:
        mem = fsspec.filesystem("memory")
        if mem.exists("/x3_keys"):
            mem.rm("/x3_keys", recursive=True)
