"""Tests for spec 37 (download on Azure ML): the CLI (D3), `mpc.download_shard`
(D2), `runners.run_aml_download` (D1/D2/D6/D7/D9), and `api.download`'s
`runner="aml"` seam.

No test requires Azure, CDSE, or MPC (spec 37 §7): the AML submission surface is
substituted exactly as spec 36's suite does it (`_FakeMLClient` + a fake
`azure.ai.ml.command`/`CommandJobLimits`), and every real download call
(`cdse.download`, `mpc.download_shard`) plus `rise_eo.secrets.get_secret` is mocked.
"""

from __future__ import annotations

import json
import types

import pandas as pd
import pytest

from rise_eo import api
from rise_eo.catalog.catalog import TileCatalog
from rise_eo.sources import cdse, mpc
from rise_eo.storage import fs
from rise_eo.workflows import download as download_cli
from rise_eo.workflows import runners


def _tiles(n: int):
    """A stand-in for `cdse.query_catalog`'s result: only its length and `acquisition_key`
    are read by `run_aml_download`."""
    import pandas as pd

    return pd.DataFrame({"id": [f"t{i}" for i in range(n)],
                         "acquisition_key": [f"a{i}" for i in range(n)]})


CREDS_JSON = json.dumps({
    "sh_clientid": "sh-id", "sh_clientsecret": "sh-secret",
    "s3_access_key": "ak", "s3_secret_key": "sk",
})


# --- test 1: workflows/download.py CLI modes (D3/D9) ------------------------

def test_download_cli_roi_mode_calls_cdse_download_and_writes_status(monkeypatch):
    calls = {}

    def _fake_get_secret(vault_url, name):
        calls["vault_url"] = vault_url
        calls["secret_name"] = name
        return CREDS_JSON

    def _fake_cdse_download(roi, startdate, enddate, bands, root_folderpath, catalog,
                             creds, *, max_tiles, max_cloudcover=None, cog=True, progress=False,
                             processing="latest"):
        calls["roi"] = roi
        calls["creds"] = creds
        calls["max_tiles"] = max_tiles
        return cdse.DownloadResult(successful_count=2, total_count=2, skipped_count=0,
                                    failed_count=0, elapsed_s=1.5, bytes_downloaded=123)

    monkeypatch.setattr(download_cli.secrets, "get_secret", _fake_get_secret)
    monkeypatch.setattr(download_cli.cdse, "download", _fake_cdse_download)

    status_url = "memory://dl_roi/_status/0.json"
    status = download_cli.run_roi(
        roi="memory://dl_roi/roi.geojson", startdate="2018-06-01", enddate="2018-06-11",
        bands=["B04"], dst="memory://dl_roi/data", catalog="memory://dl_roi/data/catalog.parquet",
        max_tiles=10, status_url=status_url, vault_url="kv-x.vault.azure.net",
        secret_name="cdse-creds",
    )

    assert calls["max_tiles"] == 10
    assert calls["creds"].s3_access_key == "ak"
    assert status["status"] == "ok"
    assert status["n_assets"] == 2
    with fs.open(status_url, "r") as f:
        assert json.load(f)["status"] == "ok"


def test_download_cli_shard_mode_calls_mpc_download_shard(monkeypatch):
    calls = {}

    def _fake_download_shard(rows, root_folderpath, catalog, **kw):
        calls["rows"] = rows
        calls["root_folderpath"] = root_folderpath
        return mpc.DownloadResult(successful_count=1, total_count=1, skipped_count=0, failed_count=0)

    monkeypatch.setattr(download_cli.mpc, "download_shard", _fake_download_shard)

    shard_url = "memory://dl_shard/shards/0.csv"
    rows = [{"tile_id": "T1", "acquisition_key": "T1", "band": "B04", "href": "https://x/B04.tif",
              "dst": "memory://dl_shard/data/T1/B04.tif", "offset": 0,
              "collection": "sentinel-2-l2a", "timestamp": "2018-06-01T00:00:00Z",
              "s3url": "", "cloud_cover": 0.0, "nodata": 0,
              "geometry": "POINT (0 0)"}]
    with fs.open(shard_url, "w") as f:
        pd.DataFrame(rows).to_csv(f, index=False)

    status = download_cli.run_shard(
        shard_url=shard_url, dst="memory://dl_shard/data",
        catalog="memory://dl_shard/data/catalog.parquet",
        status_url="memory://dl_shard/_status/0.json",
    )

    assert len(calls["rows"]) == 1
    assert calls["rows"][0]["tile_id"] == "T1"
    assert status["status"] == "ok"


# --- test 7b: D5-revised -- blob-JSON --creds-url path (CLI) ----------------

def test_download_cli_roi_mode_reads_creds_from_blob_url(monkeypatch):
    calls = {}

    def _fake_cdse_download(roi, startdate, enddate, bands, root_folderpath, catalog,
                             creds, *, max_tiles, max_cloudcover=None, cog=True, progress=False,
                             processing="latest"):
        calls["creds"] = creds
        return cdse.DownloadResult(successful_count=1, total_count=1, skipped_count=0,
                                    failed_count=0, elapsed_s=0.5, bytes_downloaded=42)

    monkeypatch.setattr(download_cli.cdse, "download", _fake_cdse_download)

    creds_url = "memory://dl_roi_blob/_secrets/cdse_credentials.json"
    with fs.open(creds_url, "w") as f:
        f.write(CREDS_JSON)

    status_url = "memory://dl_roi_blob/_status/0.json"
    status = download_cli.run_roi(
        roi="memory://dl_roi_blob/roi.geojson", startdate="2018-06-01", enddate="2018-06-11",
        bands=["B04"], dst="memory://dl_roi_blob/data",
        catalog="memory://dl_roi_blob/data/catalog.parquet",
        max_tiles=10, status_url=status_url, creds_url=creds_url,
    )

    assert calls["creds"].s3_access_key == "ak"
    assert status["status"] == "ok"


# --- test: mpc.download_shard signs on the node + reuses _transfer_and_stamp_one --

def test_mpc_download_shard_signs_on_node_and_transfers(monkeypatch, tmp_path):
    """The shard hands its transfer worker a SIGNER, and the worker applies it.

    Signing moved from submit time into the worker (2026-09-05): a big shard's last rows
    would otherwise wait in the queue holding a token minted when the whole shard was
    submitted. So this asserts the `sign` callable REACHES the worker and produces the
    signed url there -- not that `download_shard` called it eagerly, which is the thing
    that was wrong.
    """
    signed = {}

    def _fake_sign(url):
        signed["called_with"] = url
        return url + "?sig=fake"

    calls = []

    def _fake_transfer(src, dst, *, band, offset, sign=None, **kw):
        # The worker's own contract: sign immediately before the transfer.
        calls.append((sign(src) if sign else src, dst, band, offset))
        return True, "ok"

    monkeypatch.setattr(mpc, "_transfer_and_stamp_one", _fake_transfer)
    monkeypatch.setattr(mpc, "_import_pc_sign", lambda: _fake_sign)

    class _FakeCatalog:
        def __init__(self):
            self.rows = None

        def append(self, rows, declaration=None):
            self.rows = rows

    catalog = _FakeCatalog()
    rows = [{
        "tile_id": "T1", "acquisition_key": "T1", "band": "B04", "href": "https://mpc/B04.tif",
        "dst": str(tmp_path / "T1" / "B04.tif"), "offset": 0,
        "collection": "sentinel-2-l2a", "timestamp": "2018-06-01T00:00:00+00:00",
        "scale": 0.0001, "processing_version": "05.00",
        "processing_datetime": "2018-06-01T00:00:00+00:00", "properties": "{}",
        "s3url": "", "cloud_cover": 0.0, "nodata": 0, "geometry": "POINT (0 0)",
    }]

    result = mpc.download_shard(rows, str(tmp_path), catalog)

    assert signed["called_with"] == "https://mpc/B04.tif"
    assert calls[0][0] == "https://mpc/B04.tif?sig=fake"
    assert result.successful_count == 1
    assert catalog.rows[0]["id"] == "T1"


def test_mpc_download_shard_appends_to_a_real_tile_catalog(monkeypatch, tmp_path):
    """The row shape `download_shard` builds for `_append_downloaded` must survive a
    real `TileCatalog.append` -> write -> read round-trip, not just a stub capture."""
    monkeypatch.setattr(mpc, "_transfer_and_stamp_one", lambda *a, **kw: (True, "ok"))
    monkeypatch.setattr(mpc, "_import_pc_sign", lambda: (lambda url: url + "?sig=fake"))

    catalog = TileCatalog(str(tmp_path / "catalog.parquet"))
    rows = [{
        "tile_id": "T1", "acquisition_key": "T1", "band": "B04", "href": "https://mpc/B04.tif",
        "dst": str(tmp_path / "T1" / "B04.tif"), "offset": -1000,
        "collection": "sentinel-2-l2a", "timestamp": "2018-06-01T00:00:00+00:00",
        "scale": 0.0001, "processing_version": "05.00",
        "processing_datetime": "2018-06-01T00:00:00+00:00", "properties": "{}",
        "s3url": "", "cloud_cover": 0.0, "nodata": 0, "geometry": "POINT (1 2)",
    }]

    mpc.download_shard(rows, str(tmp_path), catalog)

    gdf = catalog.read()
    assert list(gdf["id"]) == ["T1"]
    assert gdf.iloc[0]["offset"] == -1000
    assert gdf.iloc[0].geometry.wkt == "POINT (1 2)"


# --- fakes: AML client + job-builder + CommandJobLimits (mirrors spec 36) ----

class _NS(types.SimpleNamespace):
    pass


@pytest.fixture
def fake_aml_command(monkeypatch):
    def _cmd(**kwargs):
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(runners, "_import_aml_command", lambda: _cmd)
    monkeypatch.setattr(
        runners, "_import_command_job_limits",
        lambda: (lambda timeout: types.SimpleNamespace(timeout=timeout)),
    )
    return _cmd


class _FakeMLClient:
    def __init__(self, job_statuses: list[str]):
        self._job_statuses = job_statuses
        self.submitted: list = []
        self.compute = _NS(get=lambda cluster: _NS(provisioning_state="Succeeded", max_instances=4))
        self.environments = _NS(get=lambda **kw: _NS())
        self.jobs = _NS(create_or_update=self._create_or_update, get=self._get)

    def _create_or_update(self, job):
        idx = len(self.submitted)
        self.submitted.append(job)
        return _NS(name=f"job-{idx}")

    def _get(self, name):
        idx = int(name.rsplit("-", 1)[1])
        return _NS(status=self._job_statuses[idx])


def _fake_get_secret(vault_url, name):
    return CREDS_JSON


def _write_status(root, run_id, k, status="ok", circuit_tripped=False):
    with fs.open(f"{root}/runs/{run_id}/_status/{k}.json", "w") as f:
        json.dump({"unit": k, "status": status, "circuit_tripped": circuit_tripped}, f)


# --- test 2/6: D1 -- CDSE submits exactly one job regardless of tile count ---

def test_run_aml_download_cdse_submits_exactly_one_job(fake_aml_command, monkeypatch):
    monkeypatch.setattr(runners, "_cdse_query_catalog", lambda *a, **kw: _tiles(37))
    ml_client = _FakeMLClient(["Completed"])

    root = "memory://aml_dl_cdse/root"
    run_id = "cdserun"
    _write_status(root, run_id, 0)

    result = runners.run_aml_download(
        "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04", "B08"],
        "memory://aml_dl_cdse/data", "memory://aml_dl_cdse/data/catalog.parquet",
        source="cdse", cluster="c", environment="fsd-env:1", root=root,
        identity_client_id="deadbeef", max_tiles=100, vault_url="kv-x.vault.azure.net",
        secret_name="cdse-creds", ml_client=ml_client, run_id=run_id,
        get_secret=_fake_get_secret,
    )

    assert len(ml_client.submitted) == 1
    assert result["n_jobs"] == 1


def test_run_aml_download_cdse_one_job_is_non_vacuous(fake_aml_command, monkeypatch):
    """Non-vacuousness (project standard): the single-job assertion isn't trivially
    true for any tile count -- prove it actually varies with tile count for a naive
    (buggy) one-job-per-tile dispatcher, so pinning it at exactly 1 job regardless of
    tile count (1 vs 37) is a real assertion, not a coincidence of the fixture."""
    for n_tiles, run_id in [(1, "mutrun_small"), (37, "mutrun_big")]:
        monkeypatch.setattr(runners, "_cdse_query_catalog", lambda *a, n=n_tiles, **kw: _tiles(n))
        ml_client = _FakeMLClient(["Completed"])
        root = f"memory://aml_dl_cdse_mut/{run_id}"
        _write_status(root, run_id, 0)

        runners.run_aml_download(
            "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
            f"{root}/data", f"{root}/data/catalog.parquet",
            source="cdse", cluster="c", environment="fsd-env:1", root=root,
            identity_client_id="deadbeef", max_tiles=100, vault_url="kv-x.vault.azure.net",
            secret_name="cdse-creds", ml_client=ml_client, run_id=run_id,
            get_secret=_fake_get_secret,
        )
        # A one-job-per-tile mutant would submit n_tiles jobs here; the real
        # dispatcher submits exactly 1 regardless -- this varies the tile count
        # specifically so that check isn't vacuously true for a fixed n_tiles=1.
        assert len(ml_client.submitted) == 1


# --- test 2/3: D1/D2 -- MPC shards a discovered asset list into N jobs -------

def _mpc_rows(n):
    return [{
        "tile_id": f"T{i}", "acquisition_key": f"T{i}", "band": "B04", "href": f"https://mpc/{i}/B04.tif",
        "dst": f"memory://x/{i}/B04.tif", "offset": 0, "collection": "sentinel-2-l2a",
        "timestamp": "2018-06-01T00:00:00+00:00", "s3url": "", "cloud_cover": 0.0,
        "nodata": 0, "geometry": "POINT (0 0)",
    } for i in range(n)]


def test_run_aml_download_mpc_shards_into_n_jobs(fake_aml_command, monkeypatch):
    rows = _mpc_rows(7)
    monkeypatch.setattr(runners._mpc, "discover_shard_rows", lambda *a, **kw: rows)
    ml_client = _FakeMLClient(["Completed", "Completed", "Completed"])

    root = "memory://aml_dl_mpc/root"
    run_id = "mpcrun"
    for k in range(3):
        _write_status(root, run_id, k)

    result = runners.run_aml_download(
        "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
        "memory://aml_dl_mpc/data", "memory://aml_dl_mpc/data/catalog.parquet",
        source="mpc", cluster="c", environment="fsd-env:1", root=root,
        identity_client_id="deadbeef", max_tiles=100, n_shards=3,
        ml_client=ml_client, run_id=run_id,
    )

    assert len(ml_client.submitted) == 3
    assert result["n_jobs"] == 3
    # partition check (spec 36 D2, reused): every asset lands in exactly one shard csv
    all_ids = set()
    for k in range(3):
        with fs.open(f"{root}/runs/{run_id}/shards/{k}.csv", "r") as f:
            shard_df = pd.read_csv(f)
        all_ids |= set(shard_df["tile_id"])
    assert all_ids == {f"T{i}" for i in range(7)}


def test_run_aml_download_mpc_partition_is_non_vacuous():
    """Non-vacuousness (project standard): a mutation that drops an asset from a
    shard fails the partition assertion above."""
    rows = _mpc_rows(7)
    shards = runners.shard_units(rows, 3)
    shards[0] = shards[0][1:]  # simulate a sharder that drops an asset
    all_ids = {r["tile_id"] for group in shards for r in group}
    assert all_ids != {r["tile_id"] for r in rows}


# --- spec 47 D8/#64: a driver-side catalog diff precedes MPC dispatch --------

def _write_mpc_catalog(path, tile_bands: dict[str, list[str]]):
    """A real `TileCatalog` with one row per tile, `files` = `<band>.tif` per band --
    the exact shape `mpc._append_downloaded` writes."""
    from shapely.geometry import Point

    from rise_eo.catalog.declaration import S2_L2A_DECLARATION

    catalog = TileCatalog(str(path))
    rows = [{
        "id": tile_id, "collection": "sentinel-2-l2a",
        "timestamp": "2018-06-01T00:00:00+00:00", "s3url": "",
        "local_folderpath": f"/data/{tile_id}",
        "files": ",".join(f"{b}.tif" for b in bands),
        "cloud_cover": 0.0, "offset": 0, "nodata": 0, "geometry": Point(0, 0),
    } for tile_id, bands in tile_bands.items()]
    catalog.append(rows, declaration=S2_L2A_DECLARATION)
    return catalog


def test_run_aml_download_mpc_shortfall_empty_returns_without_submitting(
    fake_aml_command, monkeypatch, tmp_path,
):
    """D8 (#64): every discovered asset is already catalogued -> return WITHOUT
    calling `ml_client.jobs.create_or_update` at all, same keys a dispatched run
    returns (AC7)."""
    rows = _mpc_rows(5)   # tile_id T0..T4, band B04
    monkeypatch.setattr(runners._mpc, "discover_shard_rows", lambda *a, **kw: rows)
    catalog_path = tmp_path / "catalog.parquet"
    _write_mpc_catalog(catalog_path, {f"T{i}": ["B04"] for i in range(5)})
    ml_client = _FakeMLClient(["Completed"])

    result = runners.run_aml_download(
        "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
        "memory://aml_dl_mpc_short/data", str(catalog_path),
        source="mpc", cluster="c", environment="fsd-env:1",
        root="memory://aml_dl_mpc_short/root", identity_client_id="x", max_tiles=100,
        ml_client=ml_client, run_id="shortrun",
    )

    assert ml_client.submitted == []
    assert result.keys() == {"run_id", "source", "n_jobs", "job_statuses", "reports"}
    assert result["n_jobs"] == 0


def test_run_aml_download_mpc_shortfall_partial_dispatches_only_missing(
    fake_aml_command, monkeypatch, tmp_path,
):
    """D8 (#64): a partially-present request shards only the shortfall (AC8) -- a
    95%-present request must not dispatch 100%. Mirrors the spec's own worked
    example shape (some discovered, some already catalogued)."""
    rows = _mpc_rows(10)   # T0..T9, band B04
    monkeypatch.setattr(runners._mpc, "discover_shard_rows", lambda *a, **kw: rows)
    catalog_path = tmp_path / "catalog.parquet"
    # 7 of 10 tiles already have B04; 3 (T7, T8, T9) are missing.
    _write_mpc_catalog(catalog_path, {f"T{i}": ["B04"] for i in range(7)})
    ml_client = _FakeMLClient(["Completed", "Completed", "Completed"])

    result = runners.run_aml_download(
        "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
        "memory://aml_dl_mpc_part/data", str(catalog_path),
        source="mpc", cluster="c", environment="fsd-env:1",
        root="memory://aml_dl_mpc_part/root", identity_client_id="x", max_tiles=100,
        n_shards=3, ml_client=ml_client, run_id="partrun",
    )

    assert result["n_jobs"] == 3
    all_ids = set()
    for k in range(3):
        with fs.open(f"memory://aml_dl_mpc_part/root/runs/partrun/shards/{k}.csv", "r") as f:
            shard_df = pd.read_csv(f)
        all_ids |= set(shard_df["tile_id"])
    assert all_ids == {"T7", "T8", "T9"}


def test_run_aml_download_mpc_shortfall_prints_before_and_after_counts(
    fake_aml_command, monkeypatch, tmp_path, capsys,
):
    rows = _mpc_rows(10)
    monkeypatch.setattr(runners._mpc, "discover_shard_rows", lambda *a, **kw: rows)
    catalog_path = tmp_path / "catalog.parquet"
    _write_mpc_catalog(catalog_path, {f"T{i}": ["B04"] for i in range(7)})
    ml_client = _FakeMLClient(["Completed"] * 3)

    runners.run_aml_download(
        "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
        "memory://aml_dl_mpc_msg/data", str(catalog_path),
        source="mpc", cluster="c", environment="fsd-env:1",
        root="memory://aml_dl_mpc_msg/root", identity_client_id="x", max_tiles=100,
        n_shards=3, ml_client=ml_client, run_id="msgrun",
    )
    out = capsys.readouterr().out
    assert "[download] 3 of 10 assets missing; dispatching 3" in out


def test_mpc_catalog_shortfall_absent_catalog_returns_all_rows(tmp_path):
    from rise_eo.workflows.runners import _mpc_catalog_shortfall

    rows = _mpc_rows(3)
    assert _mpc_catalog_shortfall(str(tmp_path / "nope.parquet"), rows) == rows


def test_mpc_catalog_shortfall_checks_band_not_just_tile(tmp_path):
    """A tile already in the catalog but MISSING one of the requested bands must still
    show up in the shortfall -- 'already present' is keyed on (tile_id, band), not
    tile_id alone."""
    from rise_eo.workflows.runners import _mpc_catalog_shortfall

    rows = [
        {"tile_id": "T0", "acquisition_key": "T0", "band": "B04", "dst": "x"},
        {"tile_id": "T0", "acquisition_key": "T0", "band": "B08", "dst": "y"},
    ]
    catalog_path = tmp_path / "catalog.parquet"
    _write_mpc_catalog(catalog_path, {"T0": ["B04"]})   # B08 missing for T0

    shortfall = _mpc_catalog_shortfall(str(catalog_path), rows)
    assert [r["band"] for r in shortfall] == ["B08"]


# --- test 2/5: D4/D5/D6 -- job spec carries AZURE_CLIENT_ID, timeout, KV coords,
# and never a secret value ----------------------------------------------------

def test_run_aml_download_cdse_job_carries_identity_timeout_and_kv_coords_not_secret(
    fake_aml_command, monkeypatch,
):
    monkeypatch.setattr(runners, "_cdse_query_catalog", lambda *a, **kw: _tiles(2))
    ml_client = _FakeMLClient(["Completed"])
    root = "memory://aml_dl_kv/root"
    run_id = "kvrun"
    _write_status(root, run_id, 0)

    runners.run_aml_download(
        "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
        "memory://aml_dl_kv/data", "memory://aml_dl_kv/data/catalog.parquet",
        source="cdse", cluster="c", environment="fsd-env:1", root=root,
        identity_client_id="deadbeef-guid", max_tiles=100,
        vault_url="kv-x.vault.azure.net", secret_name="cdse-creds",
        ml_client=ml_client, run_id=run_id, get_secret=_fake_get_secret,
    )

    job = ml_client.submitted[0]
    assert job.environment_variables == {"AZURE_CLIENT_ID": "deadbeef-guid"}
    assert job.limits.timeout > 0
    assert "kv-x.vault.azure.net" in job.command
    assert "cdse-creds" in job.command
    for secret_value in ("sh-secret", "sk", CREDS_JSON):
        assert secret_value not in job.command
        assert secret_value not in json.dumps(job.environment_variables)


# --- test 7b: D5-revised -- run_aml_download's blob creds_url path (dispatcher) --

def test_run_aml_download_cdse_creds_url_puts_location_not_value_in_command(
    fake_aml_command, monkeypatch,
):
    monkeypatch.setattr(runners, "_cdse_query_catalog", lambda *a, **kw: _tiles(2))
    ml_client = _FakeMLClient(["Completed"])
    root = "memory://aml_dl_blob/root"
    run_id = "blobrun"
    _write_status(root, run_id, 0)

    creds_url = "memory://aml_dl_blob/_secrets/cdse_credentials.json"
    with fs.open(creds_url, "w") as f:
        f.write(CREDS_JSON)

    runners.run_aml_download(
        "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
        "memory://aml_dl_blob/data", "memory://aml_dl_blob/data/catalog.parquet",
        source="cdse", cluster="c", environment="fsd-env:1", root=root,
        identity_client_id="deadbeef", max_tiles=100, creds_url=creds_url,
        ml_client=ml_client, run_id=run_id,
    )

    job = ml_client.submitted[0]
    assert creds_url in job.command
    assert "--vault-url" not in job.command
    assert "--secret-name" not in job.command
    for secret_value in ("sh-secret", "sk", CREDS_JSON):
        assert secret_value not in job.command
        assert secret_value not in json.dumps(job.environment_variables)


def test_aml_download_preflight_refuses_neither_cdse_creds_source():
    ml_client = _FakeMLClient(["Completed"])
    with pytest.raises(ValueError, match="exactly one CDSE creds source"):
        runners._aml_download_preflight(
            ml_client, cluster="c", environment="e:1", root="memory://pf/root6",
            source="cdse", n_assets=1, vault_url=None, secret_name=None,
            get_secret=_fake_get_secret, remaining_quota_gb=None, estimated_gb=None,
            creds_url=None,
        )


def test_aml_download_preflight_refuses_both_cdse_creds_sources():
    ml_client = _FakeMLClient(["Completed"])
    with pytest.raises(ValueError, match="exactly one CDSE creds source"):
        runners._aml_download_preflight(
            ml_client, cluster="c", environment="e:1", root="memory://pf/root7",
            source="cdse", n_assets=1, vault_url="kv", secret_name="n",
            get_secret=_fake_get_secret, remaining_quota_gb=None, estimated_gb=None,
            creds_url="memory://pf/root7/creds.json",
        )


def test_aml_download_preflight_resolves_cdse_creds_from_blob_url():
    ml_client = _FakeMLClient(["Completed"])
    creds_url = "memory://pf/root8/creds.json"
    with fs.open(creds_url, "w") as f:
        f.write(CREDS_JSON)

    warnings = runners._aml_download_preflight(
        ml_client, cluster="c", environment="e:1", root="memory://pf/root8",
        source="cdse", n_assets=1, vault_url=None, secret_name=None,
        get_secret=_fake_get_secret, remaining_quota_gb=None, estimated_gb=None,
        creds_url=creds_url,
    )
    assert warnings == []


# --- test 4: raises on Failed, and on circuit_tripped even if AML says Completed --

def test_run_aml_download_raises_when_job_reports_failed(fake_aml_command, monkeypatch):
    monkeypatch.setattr(runners._mpc, "discover_shard_rows", lambda *a, **kw: _mpc_rows(2))
    ml_client = _FakeMLClient(["Completed", "Failed"])
    root = "memory://aml_dl_fail/root"
    run_id = "failrun"
    _write_status(root, run_id, 0)
    _write_status(root, run_id, 1, status="failed")

    with pytest.raises(RuntimeError, match=r"\[1\]"):
        runners.run_aml_download(
            "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
            "memory://aml_dl_fail/data", "memory://aml_dl_fail/data/catalog.parquet",
            source="mpc", cluster="c", environment="fsd-env:1", root=root,
            identity_client_id="id", max_tiles=100, n_shards=2,
            ml_client=ml_client, run_id=run_id,
        )


def test_run_aml_download_raises_on_circuit_tripped_even_if_aml_completed(
    fake_aml_command, monkeypatch,
):
    monkeypatch.setattr(runners, "_cdse_query_catalog", lambda *a, **kw: _tiles(1))
    ml_client = _FakeMLClient(["Completed"])  # AML itself reports success
    root = "memory://aml_dl_trip/root"
    run_id = "triprun"
    _write_status(root, run_id, 0, status="ok", circuit_tripped=True)  # but the run tripped

    with pytest.raises(RuntimeError, match=r"\[0\]"):
        runners.run_aml_download(
            "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
            "memory://aml_dl_trip/data", "memory://aml_dl_trip/data/catalog.parquet",
            source="cdse", cluster="c", environment="fsd-env:1", root=root,
            identity_client_id="id", max_tiles=100, vault_url="kv-x.vault.azure.net",
            secret_name="cdse-creds", ml_client=ml_client, run_id=run_id,
            get_secret=_fake_get_secret,
        )


# --- test 5: api.download accepts runner="aml" and rejects unknown runners --

def test_api_download_runner_aml_threads_runner_kwargs(monkeypatch, tmp_path):
    calls = {}

    def _fake_run_aml_download(**kwargs):
        calls.update(kwargs)
        return {"run_id": "r", "n_jobs": 1}

    monkeypatch.setattr(runners, "run_aml_download", _fake_run_aml_download)

    dst = str(tmp_path / "data")
    catalog_filepath = api.download(
        "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"], dst,
        source="cdse", max_tiles=10, runner="aml",
        runner_kwargs={"cluster": "c", "environment": "e:1", "root": "memory://r",
                        "identity_client_id": "id", "vault_url": "kv", "secret_name": "n"},
    )

    assert calls["cluster"] == "c"
    assert calls["source"] == "cdse"
    assert catalog_filepath == str(tmp_path / "data" / "sentinel-2-l2a" / "catalog.parquet")


def test_api_download_rejects_unknown_runner():
    with pytest.raises(api.PreflightError):
        api.download(
            "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"], "memory://x",
            source="cdse", max_tiles=10, runner="batch",
        )


# --- test 6: D7 preflight refuses empty discovery / unwritable root / bad KV
# secret / expired keys, and warns on quota -----------------------------------

def test_aml_download_preflight_refuses_empty_discovery():
    ml_client = _FakeMLClient(["Completed"])
    with pytest.raises(ValueError, match="0 assets"):
        runners._aml_download_preflight(
            ml_client, cluster="c", environment="e:1", root="memory://pf/root",
            source="mpc", n_assets=0, vault_url=None, secret_name=None,
            get_secret=_fake_get_secret, remaining_quota_gb=None, estimated_gb=None,
        )


def test_aml_download_preflight_refuses_unwritable_root(monkeypatch):
    ml_client = _FakeMLClient(["Completed"])

    def _boom(root, exist_ok=True):
        raise OSError("no permission")

    monkeypatch.setattr(fs, "makedirs", _boom)
    with pytest.raises(ValueError, match="not reachable/writable"):
        runners._aml_download_preflight(
            ml_client, cluster="c", environment="e:1", root="memory://pf/root2",
            source="mpc", n_assets=1, vault_url=None, secret_name=None,
            get_secret=_fake_get_secret, remaining_quota_gb=None, estimated_gb=None,
        )


def test_aml_download_preflight_refuses_bad_kv_secret():
    ml_client = _FakeMLClient(["Completed"])

    def _bad_get_secret(vault_url, name):
        return "not json"

    with pytest.raises(ValueError, match="did not resolve/parse"):
        runners._aml_download_preflight(
            ml_client, cluster="c", environment="e:1", root="memory://pf/root3",
            source="cdse", n_assets=1, vault_url="kv", secret_name="n",
            get_secret=_bad_get_secret, remaining_quota_gb=None, estimated_gb=None,
        )


def test_aml_download_preflight_refuses_expired_cdse_keys():
    ml_client = _FakeMLClient(["Completed"])
    expired_json = json.dumps({
        "sh_clientid": "i", "sh_clientsecret": "s", "s3_access_key": "ak",
        "s3_secret_key": "sk", "s3_keys_expire": "2000-01-01",
    })

    with pytest.raises(ValueError, match="expired"):
        runners._aml_download_preflight(
            ml_client, cluster="c", environment="e:1", root="memory://pf/root4",
            source="cdse", n_assets=1, vault_url="kv", secret_name="n",
            get_secret=lambda v, n: expired_json, remaining_quota_gb=None, estimated_gb=None,
        )


def test_aml_download_preflight_warns_when_estimate_exceeds_remaining_quota():
    ml_client = _FakeMLClient(["Completed"])
    warnings = runners._aml_download_preflight(
        ml_client, cluster="c", environment="e:1", root="memory://pf/root5",
        source="cdse", n_assets=1, vault_url="kv", secret_name="n",
        get_secret=_fake_get_secret, remaining_quota_gb=10.0, estimated_gb=50.0,
    )
    assert warnings and "quota" in warnings[0]


# --- test 7: D5 -- rise_eo.secrets.get_secret is mocked; no KV network call ------

def test_secrets_get_secret_is_mockable_without_azure_extra(monkeypatch):
    from rise_eo import secrets

    def _fake(vault_url, name):
        assert vault_url == "kv-x.vault.azure.net"
        assert name == "cdse-creds"
        return CREDS_JSON

    monkeypatch.setattr(secrets, "get_secret", _fake)
    value = secrets.get_secret("kv-x.vault.azure.net", "cdse-creds")
    creds = cdse.CdseCredentials.from_json_str(value)
    assert creds.s3_access_key == "ak"
    assert creds.s3_secret_key == "sk"


def test_cdse_credentials_from_json_str_roundtrips_legacy_keys():
    creds = cdse.CdseCredentials.from_json_str(CREDS_JSON)
    assert creds.sh_client_id == "sh-id"
    assert creds.sh_client_secret == "sh-secret"
    assert creds.s3_access_key == "ak"
    assert creds.s3_secret_key == "sk"


# --- TODO #49: MPC is anonymous -- creds arguments are refused, not ignored ----
# Regression: `creds_url` was accepted and silently dropped for source="mpc" (the
# MPC branch never passed it to preflight, and the creds block is cdse-gated). A
# hand-written Phase 3 script wrapped an MPC archive run in blob_creds(), which
# would have staged the CDSE keys on blob for the whole run and never read them.

@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"creds_url": "memory://pf/x.json"}, "creds_url"),
        ({"vault_url": "https://kv.example/"}, "vault_url"),
        ({"secret_name": "cdse-creds"}, "secret_name"),
        ({"vault_url": "https://kv.example/", "secret_name": "n"}, "vault_url, secret_name"),
    ],
)
def test_aml_download_preflight_refuses_creds_for_anonymous_mpc(kwargs, expected):
    ml_client = _FakeMLClient(["Completed"])
    with pytest.raises(ValueError, match="anonymous and reads no credentials") as exc:
        runners._aml_download_preflight(
            ml_client, cluster="c", environment="e:1", root="memory://pf/root9",
            source="mpc", n_assets=5, vault_url=kwargs.get("vault_url"),
            secret_name=kwargs.get("secret_name"), get_secret=_fake_get_secret,
            remaining_quota_gb=None, estimated_gb=None, creds_url=kwargs.get("creds_url"),
        )
    assert expected in str(exc.value)


def test_aml_download_preflight_allows_mpc_without_creds():
    ml_client = _FakeMLClient(["Completed"])
    assert runners._aml_download_preflight(
        ml_client, cluster="c", environment="e:1", root="memory://pf/root10",
        source="mpc", n_assets=5, vault_url=None, secret_name=None,
        get_secret=_fake_get_secret, remaining_quota_gb=None, estimated_gb=None,
        creds_url=None,
    ) == []


def test_run_aml_download_mpc_rejects_creds_url_end_to_end(fake_aml_command, monkeypatch):
    """The dispatcher must surface it too -- preflight is only reached via run_aml_download."""
    monkeypatch.setattr(
        runners._mpc, "discover_shard_rows",
        lambda *a, **kw: [{"href": "h", "dst": "d", "band": "B04", "tile_id": "T33UWP", "acquisition_key": "T33UWP",
                            "collection": "S2A", "timestamp": "2018-06-01", "s3url": "",
                            "cloud_cover": 0.0, "offset": 0, "nodata": 0,
                            "geometry": "POINT (0 0)"}],
    )
    with pytest.raises(ValueError, match="anonymous and reads no credentials"):
        runners.run_aml_download(
            "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
            "memory://aml_mpc_creds/data", "memory://aml_mpc_creds/data/catalog.parquet",
            source="mpc", cluster="c", environment="fsd-env:1",
            root="memory://aml_mpc_creds/root", identity_client_id="deadbeef",
            max_tiles=100, creds_url="memory://aml_mpc_creds/_secrets/c.json",
            ml_client=_FakeMLClient(["Completed"]), run_id="mpccreds",
        )


# --- max_tiles is a DRIVER-side guardrail for both sources --------------------
# Regression: `max_tiles` reached CDSE only, via `--max-tiles` on the node (i.e.
# after the cluster spun up), and the MPC path dropped it entirely -- so
# `api.download(source="mpc", max_tiles=N)` raised locally (sources/mpc.py) but
# silently downloaded everything under runner="aml". The runner must not change
# what a call means (spec 36 D3).

def _rows_over_n_tiles(n: int) -> list[dict]:
    return [{"href": f"h{i}", "dst": f"d{i}", "band": "B04", "tile_id": f"T{i:05d}", "acquisition_key": f"T{i:05d}",
             "collection": "S2A", "timestamp": "2018-06-01", "s3url": "",
             "cloud_cover": 0.0, "offset": 0, "nodata": 0,
             "geometry": "POINT (0 0)"} for i in range(n)]


def test_run_aml_download_mpc_enforces_max_tiles_on_the_driver(fake_aml_command, monkeypatch):
    monkeypatch.setattr(runners._mpc, "discover_shard_rows",
                        lambda *a, **kw: _rows_over_n_tiles(3))
    with pytest.raises(ValueError, match=r"3 matched tiles exceed max_tiles=2"):
        runners.run_aml_download(
            "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
            "memory://aml_cap/data", "memory://aml_cap/data/catalog.parquet",
            source="mpc", cluster="c", environment="fsd-env:1", root="memory://aml_cap/root",
            identity_client_id="deadbeef", max_tiles=2,
            ml_client=_FakeMLClient(["Completed"]), run_id="cap",
        )


def test_run_aml_download_mpc_counts_distinct_tiles_not_assets(fake_aml_command, monkeypatch):
    """6 assets spread over 2 tiles must pass max_tiles=2 -- the cap counts MGRS
    tiles (as sources/mpc.download does), not one-row-per-asset shard rows."""
    rows = [dict(r, tile_id="T33UWP" if i < 3 else "T33UVP")
            for i, r in enumerate(_rows_over_n_tiles(6))]
    monkeypatch.setattr(runners._mpc, "discover_shard_rows", lambda *a, **kw: rows)
    ml_client = _FakeMLClient(["Completed", "Completed"])   # one per shard
    root = "memory://aml_cap_ok/root"
    for k in range(2):
        _write_status(root, "capok", k)

    runners.run_aml_download(
        "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
        "memory://aml_cap_ok/data", "memory://aml_cap_ok/data/catalog.parquet",
        source="mpc", cluster="c", environment="fsd-env:1", root=root,
        identity_client_id="deadbeef", max_tiles=2, n_shards=2,
        ml_client=ml_client, run_id="capok",
    )
    assert len(ml_client.submitted) == 2


def test_run_aml_download_cdse_enforces_max_tiles_before_dispatch(fake_aml_command, monkeypatch):
    monkeypatch.setattr(runners, "_cdse_query_catalog", lambda *a, **kw: _tiles(3))
    ml_client = _FakeMLClient(["Completed"])
    with pytest.raises(ValueError, match=r"3 matched tiles exceed max_tiles=2"):
        runners.run_aml_download(
            "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
            "memory://aml_cap_cdse/data", "memory://aml_cap_cdse/data/catalog.parquet",
            source="cdse", cluster="c", environment="fsd-env:1",
            root="memory://aml_cap_cdse/root", identity_client_id="deadbeef",
            max_tiles=2, vault_url="kv", secret_name="n", get_secret=_fake_get_secret,
            ml_client=ml_client, run_id="capcdse",
        )
    assert ml_client.submitted == []      # nothing dispatched -- the point of a driver-side guard


# --- E2 (#145): cloud-download forwarding ------------------------------------

def test_run_aml_download_cdse_command_carries_cloudcover_and_no_cog(
    fake_aml_command, monkeypatch,
):
    monkeypatch.setattr(runners, "_cdse_query_catalog", lambda *a, **kw: _tiles(2))
    ml_client = _FakeMLClient(["Completed"])
    root = "memory://aml_dl_cc/root"
    _write_status(root, "ccrun", 0)

    runners.run_aml_download(
        "memory://roi.geojson", "2018-06-01", "2018-06-11", ["B04"],
        "memory://aml_dl_cc/data", "memory://aml_dl_cc/data/catalog.parquet",
        source="cdse", cluster="c", environment="fsd-env:1", root=root,
        identity_client_id="g", max_tiles=100, vault_url="kv", secret_name="n",
        get_secret=_fake_get_secret, max_cloudcover=12.5, cog=False,
        ml_client=ml_client, run_id="ccrun",
    )

    command = ml_client.submitted[0].command
    assert "--max-cloudcover 12.5" in command
    assert "--no-cog" in command


def test_download_main_roi_mode_forwards_cloudcover_and_cog_to_run_roi(monkeypatch):
    seen = {}

    def _fake_run_roi(**kw):
        seen.update(kw)
        return {"status": "ok"}

    monkeypatch.setattr(download_cli, "run_roi", _fake_run_roi)
    argv = ["--roi", "r.geojson", "--startdate", "2018-06-01", "--enddate", "2018-06-11",
            "--bands", "B04", "--dst", "d", "--catalog", "c", "--max-tiles", "3",
            "--status-url", "s", "--creds-url", "u", "--max-cloudcover", "12.5"]
    download_cli.main(argv + ["--no-cog"])
    assert (seen["max_cloudcover"], seen["cog"]) == (12.5, False)
    download_cli.main(argv)
    assert (seen["max_cloudcover"], seen["cog"]) == (12.5, True)
