"""Tests for spec 40 D2 deliverable 1: the four in-job stamps (`process_start_at`,
`work_start_at`, `work_end_at`, `ended_at`) that `workflows/{shard,download,infer_shard,
flatten}.py` add to the `_status/<k>.json` each already writes.

No test requires Azure: every real call each entrypoint wraps (`runners.run_local`,
`cdse.download`, `runners.run_local_inference`, `datacube.flatten.flatten`) is mocked,
mirroring the existing `test_scale_runner.py`/`test_download_aml.py`/`test_infer_aml.py`
patterns. `process_start_at` is a module-level constant stamped at import time (spec 40
D2: "before any heavy import"), so these tests only check it is present and orders
correctly relative to the other three -- they cannot observe process start itself.
"""

from __future__ import annotations

import json
import types

import pandas as pd

from rise_eo.model import BaseModelAdapter, bundle
from rise_eo.sources import cdse, mpc
from rise_eo.storage import fs
from rise_eo.workflows import download as download_cli
from rise_eo.workflows import flatten as flatten_cli
from rise_eo.workflows import infer_shard, runners, shard

_STAMP_KEYS = ("process_start_at", "work_start_at", "work_end_at", "ended_at")


def _assert_stamps_present_and_ordered(status: dict) -> None:
    for key in _STAMP_KEYS:
        assert status.get(key), f"missing stamp {key!r} in {status}"
    ts = [pd.Timestamp(status[k]) for k in _STAMP_KEYS]
    assert ts == sorted(ts)  # process_start_at <= work_start_at <= work_end_at <= ended_at


# --- shard.py -----------------------------------------------------------------------

def test_shard_status_carries_all_four_stamps(tmp_path, monkeypatch):
    shard_url = "memory://status_shard/shards/0.csv"
    rows = [{"export_folderpath": str(tmp_path / "cube1"), "shapefilepath": "g1.geojson",
             "catalog_filepath": "c1.parquet"}]
    with fs.open(shard_url, "w") as f:
        pd.DataFrame(rows).to_csv(f, index=False)

    def _fake_run_local(csv_path, *, cores, **kw):
        fs.makedirs(str(tmp_path / "cube1"))
        fs.save_npy(str(tmp_path / "cube1" / "datacube.npy"), [0])
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runners, "run_local", _fake_run_local)

    status = shard.run_shard(shard_url, cores=1)
    _assert_stamps_present_and_ordered(status)
    assert status["process_start_at"] == shard._PROCESS_START_AT
    assert "seconds" in status  # spec 40 D2: kept exactly as-is


# --- download.py: --roi (CDSE) and --shard (MPC) modes -------------------------------

def test_download_roi_mode_status_carries_all_four_stamps(monkeypatch):
    monkeypatch.setattr(download_cli.secrets, "get_secret", lambda *a, **kw: json.dumps({
        "sh_clientid": "id", "sh_clientsecret": "secret",
        "s3_access_key": "ak", "s3_secret_key": "sk",
    }))
    monkeypatch.setattr(download_cli.cdse, "download", lambda *a, **kw: cdse.DownloadResult(
        successful_count=1, total_count=1, skipped_count=0, failed_count=0,
        elapsed_s=0.5, bytes_downloaded=10,
    ))

    status_url = "memory://status_dl_roi/_status/0.json"
    status = download_cli.run_roi(
        roi="memory://status_dl_roi/roi.geojson", startdate="2018-06-01",
        enddate="2018-06-11", bands=["B04"], dst="memory://status_dl_roi/data",
        catalog="memory://status_dl_roi/data/catalog.parquet", max_tiles=10,
        status_url=status_url, vault_url="kv.vault.azure.net", secret_name="cdse-creds",
    )
    _assert_stamps_present_and_ordered(status)
    assert status["process_start_at"] == download_cli._PROCESS_START_AT
    assert status["seconds"] == 0.5


def test_download_shard_mode_status_carries_all_four_stamps(monkeypatch):
    monkeypatch.setattr(download_cli.mpc, "download_shard", lambda rows, dst, cat, **kw:
        mpc.DownloadResult(successful_count=1, total_count=1, skipped_count=0, failed_count=0))

    shard_url = "memory://status_dl_shard/shards/0.csv"
    rows = [{"tile_id": "T1", "band": "B04", "href": "https://x/B04.tif",
             "dst": "memory://status_dl_shard/data/T1/B04.tif", "offset": 0,
             "satellite": "sentinel-2-l2a", "timestamp": "2018-06-01T00:00:00Z",
             "s3url": "", "cloud_cover": 0.0, "nodata": 0, "geometry": "POINT (0 0)"}]
    with fs.open(shard_url, "w") as f:
        pd.DataFrame(rows).to_csv(f, index=False)

    status = download_cli.run_shard(
        shard_url=shard_url, dst="memory://status_dl_shard/data",
        catalog="memory://status_dl_shard/data/catalog.parquet",
        status_url="memory://status_dl_shard/_status/0.json",
    )
    _assert_stamps_present_and_ordered(status)
    assert status["process_start_at"] == download_cli._PROCESS_START_AT


# --- infer_shard.py -------------------------------------------------------------------

class _NDVIUp(BaseModelAdapter):
    required_bands = ["B04", "B08"]
    n_timestamps = 1
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["cls"]
    feature_sequence = []

    def load(self):
        pass

    def predict(self, X):
        return X


def test_infer_shard_status_carries_all_four_stamps(tmp_path, monkeypatch):
    bundle_dir = bundle.save(_NDVIUp(), {}, str(tmp_path / "bundle"))
    staged_bundle_url = "memory://status_infer/_bundle"
    runners._stage_bundle(bundle_dir, staged_bundle_url)

    shard_url = "memory://status_infer/shards/0.csv"
    rows = [{"id": "u0", "shapefilepath": "memory://cells/u0/geometry.geojson",
             "catalog_filepath": "memory://cells/u0/catalog.parquet",
             "startdate": "2018-06-01", "enddate": "2018-06-11",
             "export_folderpath": "memory://cells/u0", "mosaic_days": 20,
             "mosaic_scheme": "calendar", "declaration_filepath": "memory://unused",
             "bands": "B04,B08"}]
    with fs.open(shard_url, "w") as f:
        pd.DataFrame(rows).to_csv(f, index=False)

    monkeypatch.setattr(runners, "run_local_inference",
                        lambda *a, **kw: types.SimpleNamespace(returncode=0))

    status = infer_shard.run_infer_shard(shard_url, staged_bundle_url, cores=1)
    _assert_stamps_present_and_ordered(status)
    assert status["process_start_at"] == infer_shard._PROCESS_START_AT


# --- flatten.py -------------------------------------------------------------------

def test_flatten_status_carries_all_four_stamps(monkeypatch):
    input_csv = "memory://status_flatten/input.csv"
    with fs.open(input_csv, "w") as f:
        pd.DataFrame([{"id": "u0", "datacube_filepath": "memory://cells/u0"}]).to_csv(
            f, index=False)

    monkeypatch.setattr(flatten_cli._flatten, "flatten", lambda **kw: None)

    status_url = "memory://status_flatten/_status/0.json"
    status = flatten_cli.run(
        input_csv=input_csv, export_folderpath="memory://status_flatten/export",
        status_url=status_url,
    )
    _assert_stamps_present_and_ordered(status)
    assert status["process_start_at"] == flatten_cli._PROCESS_START_AT
    assert status["status"] == "ok"


# --- degenerate: a failed unit still writes process_start_at (D3) -------------------

def test_shard_status_stamps_present_even_when_snakemake_fails(tmp_path, monkeypatch):
    shard_url = "memory://status_shard_fail/shards/0.csv"
    rows = [{"export_folderpath": str(tmp_path / "cube_missing"),
             "shapefilepath": "g1.geojson", "catalog_filepath": "c1.parquet"}]
    with fs.open(shard_url, "w") as f:
        pd.DataFrame(rows).to_csv(f, index=False)

    monkeypatch.setattr(runners, "run_local",
                        lambda *a, **kw: types.SimpleNamespace(returncode=1))

    status = shard.run_shard(shard_url, cores=1)
    assert status["status"] == "failed"
    _assert_stamps_present_and_ordered(status)
