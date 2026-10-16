"""Tests for rise_eo.sources.download_cli (spec 26 §2/§7). Monkeypatched, no network."""

import json

from rise_eo.sources import cdse, download_cli

DUMMY_FIELDS = {
    "sh_client_id": "id-123",
    "sh_client_secret": "secret-abc",
    "s3_access_key": "akia-xyz",
    "s3_secret_key": "s3secret-789",
}


def test_cli_dry_run_prints_plan_and_writes_zero_byte_result(monkeypatch, tmp_path, capsys):
    """spec 26 test 5: --dry-run prints the plan, writes a status="dry-run" result-json,
    and never touches `fs.transfer` (monkeypatched to fail the test if called)."""
    plan = {
        "needed_count": 7, "present_count": 0, "missing_count": 7,
        "missing_ids": ["a"],
        "download_params": {
            "roi": "roi.geojson", "startdate": "2018-04-01", "enddate": "2018-06-01",
            "bands": ["B04"], "max_tiles": 7, "max_cloudcover": None,
            "dst_folderpath": str(tmp_path),
        },
    }
    monkeypatch.setattr(download_cli.cdse, "plan_download", lambda *a, **k: plan)

    def fail_transfer(*a, **k):
        raise AssertionError("dry-run must not transfer any bytes")

    monkeypatch.setattr(download_cli.cdse.fs, "transfer", fail_transfer)

    result_json = str(tmp_path / "_result.json")
    rc = download_cli.main([
        "--roi", "roi.geojson", "--start", "2018-04-01", "--end", "2018-06-01",
        "--bands", "B04", "--dst", str(tmp_path), "--catalog", str(tmp_path / "c.parquet"),
        "--max-tiles", "7", "--dry-run", "--result-json", result_json,
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "missing: 7" in out

    data = json.loads(open(result_json).read())
    assert data["status"] == "dry-run"
    assert data["metrics"] == {"needed": 7, "present": 0, "missing": 7}


def test_cli_real_path_wiring_stopfile_and_exit_code(monkeypatch, tmp_path):
    """spec 26 test 6: the real path wires `--stop-file` into a `should_stop` predicate
    passed to `download_resume`, writes the full result-json shape, and the exit code
    maps 0 on a clean aggregate, non-zero on `failed_count > 0`."""
    monkeypatch.setattr(
        download_cli.CdseCredentials, "from_json",
        classmethod(lambda cls, fp, **kw: cls(**DUMMY_FIELDS)),
    )
    monkeypatch.setattr(download_cli.cdse, "probe_throughput", lambda *a, **k: (12.3, 1000, 0.1))

    captured = {}

    def fake_download_resume(*a, **k):
        captured.update(k)
        return [cdse.DownloadResult(successful_count=5, total_count=5, failed_count=0)]

    monkeypatch.setattr(download_cli.cdse, "download_resume", fake_download_resume)

    stop_file = str(tmp_path / "stop")
    result_json = str(tmp_path / "_result.json")
    rc = download_cli.main([
        "--roi", "roi.geojson", "--start", "2018-04-01", "--end", "2018-06-01",
        "--bands", "B04", "--dst", str(tmp_path), "--catalog", str(tmp_path / "c.parquet"),
        "--max-tiles", "5", "--stop-file", stop_file, "--result-json", result_json,
        "--creds", str(tmp_path / "creds.json"),
    ])
    assert rc == 0

    should_stop = captured["should_stop"]
    assert should_stop() is False
    open(stop_file, "w").close()
    assert should_stop() is True

    data = json.loads(open(result_json).read())
    assert data["status"] == "ok"
    assert data["metrics"]["successful"] == 5
    assert data["metrics"]["probe_mb_per_s"] == 12.3

    def failing_download_resume(*a, **k):
        return [cdse.DownloadResult(successful_count=3, total_count=5, failed_count=2)]

    monkeypatch.setattr(download_cli.cdse, "download_resume", failing_download_resume)
    result_json2 = str(tmp_path / "_result2.json")
    rc2 = download_cli.main([
        "--roi", "roi.geojson", "--start", "2018-04-01", "--end", "2018-06-01",
        "--bands", "B04", "--dst", str(tmp_path), "--catalog", str(tmp_path / "c2.parquet"),
        "--max-tiles", "5", "--result-json", result_json2,
        "--creds", str(tmp_path / "creds.json"), "--no-probe",
    ])
    assert rc2 != 0
    data2 = json.loads(open(result_json2).read())
    assert data2["status"] == "failed"


def test_cli_requires_creds_for_real_run(monkeypatch, tmp_path):
    """No --creds / $CDSE_CREDENTIALS_JSON on a real (non-dry-run) invocation raises,
    rather than silently proceeding without S3 keys."""
    import pytest

    monkeypatch.delenv("CDSE_CREDENTIALS_JSON", raising=False)
    with pytest.raises(SystemExit):
        download_cli.main([
            "--roi", "roi.geojson", "--start", "2018-04-01", "--end", "2018-06-01",
            "--bands", "B04", "--dst", str(tmp_path), "--catalog", str(tmp_path / "c.parquet"),
            "--max-tiles", "5",
        ])


def test_cli_ok_when_resume_recovers_transient_failures(monkeypatch, tmp_path):
    """spec 26 review, finding 1: a resume that fails 2 files on pass 1 and recovers them on a
    clean pass 2 must report status="ok"/exit 0 — the completion gate is the TERMINAL pass, not
    the summed failed_count. metrics.failed reflects the terminal pass (0); metrics.failed_total
    keeps the transient count for the transfer-contention diagnostic."""
    monkeypatch.setattr(
        download_cli.CdseCredentials, "from_json",
        classmethod(lambda cls, fp, **kw: cls(**DUMMY_FIELDS)),
    )

    def recovered_resume(*a, **k):
        return [
            cdse.DownloadResult(successful_count=5, total_count=7, failed_count=2),
            cdse.DownloadResult(successful_count=2, total_count=2, failed_count=0, skipped_count=5),
        ]

    monkeypatch.setattr(download_cli.cdse, "download_resume", recovered_resume)
    result_json = str(tmp_path / "_result.json")
    rc = download_cli.main([
        "--roi", "roi.geojson", "--start", "2018-04-01", "--end", "2018-06-01",
        "--bands", "B04", "--dst", str(tmp_path), "--catalog", str(tmp_path / "c.parquet"),
        "--max-tiles", "7", "--result-json", result_json, "--no-probe",
        "--creds", str(tmp_path / "creds.json"),
    ])
    assert rc == 0
    data = json.loads(open(result_json).read())
    assert data["status"] == "ok"
    assert data["metrics"]["failed"] == 0
    assert data["metrics"]["failed_total"] == 2


def test_cli_empty_results_is_stopped(monkeypatch, tmp_path):
    """A stop-file present before the run makes download_resume return [] (it stops before pass 1);
    the CLI must label that status="stopped"/exit 0, not a false "ok"."""
    monkeypatch.setattr(
        download_cli.CdseCredentials, "from_json",
        classmethod(lambda cls, fp, **kw: cls(**DUMMY_FIELDS)),
    )
    monkeypatch.setattr(download_cli.cdse, "download_resume", lambda *a, **k: [])
    stop_file = str(tmp_path / "stop")
    open(stop_file, "w").close()
    result_json = str(tmp_path / "_result.json")
    rc = download_cli.main([
        "--roi", "roi.geojson", "--start", "2018-04-01", "--end", "2018-06-01",
        "--bands", "B04", "--dst", str(tmp_path), "--catalog", str(tmp_path / "c.parquet"),
        "--max-tiles", "7", "--stop-file", stop_file, "--result-json", result_json,
        "--no-probe", "--creds", str(tmp_path / "creds.json"),
    ])
    assert rc == 0
    assert json.loads(open(result_json).read())["status"] == "stopped"


def test_cli_expected_block_populated_and_merges_expected_json(monkeypatch, tmp_path):
    """spec 26 §4: a real run's `expected` block carries the universal success invariants,
    and `--expected-json` merges the runbook's run-specific criteria on top."""
    monkeypatch.setattr(
        download_cli.CdseCredentials, "from_json",
        classmethod(lambda cls, fp, **kw: cls(**DUMMY_FIELDS)),
    )
    monkeypatch.setattr(
        download_cli.cdse, "download_resume",
        lambda *a, **k: [cdse.DownloadResult(successful_count=5, total_count=5, failed_count=0)],
    )
    expected_json = tmp_path / "expected.json"
    expected_json.write_text(json.dumps({"missing_count_range": [10, 15], "failed": 0}))
    result_json = str(tmp_path / "_result.json")
    rc = download_cli.main([
        "--roi", "roi.geojson", "--start", "2018-04-01", "--end", "2018-06-01",
        "--bands", "B04", "--dst", str(tmp_path), "--catalog", str(tmp_path / "c.parquet"),
        "--max-tiles", "5", "--result-json", result_json, "--no-probe",
        "--creds", str(tmp_path / "creds.json"), "--expected-json", str(expected_json),
    ])
    assert rc == 0
    exp = json.loads(open(result_json).read())["expected"]
    # universal invariants present …
    assert exp["stopped"] is False and exp["circuit_tripped"] is False and exp["pool_broken"] is False
    # … plus the runbook's merged-in criterion
    assert exp["missing_count_range"] == [10, 15]


def test_cli_forwards_max_concurrent_s3_and_reports_wall_rate(monkeypatch, tmp_path):
    """--max-concurrent-s3 threads through to download_resume, and the result-json reports
    both the per-stream aggregate and the effective wall transfer rate (spec 25 measurement)."""
    monkeypatch.setattr(
        download_cli.CdseCredentials, "from_json",
        classmethod(lambda cls, fp, **kw: cls(**DUMMY_FIELDS)),
    )
    captured = {}

    def fake_resume(*a, **k):
        captured.update(k)
        # bytes=2 MB; per-stream transfer 4s (thread-summed), wall span 1s (4 streams)
        return [cdse.DownloadResult(
            successful_count=5, total_count=5, failed_count=0,
            bytes_downloaded=2_000_000, transfer_seconds=4.0, transfer_wall_seconds=1.0,
        )]

    monkeypatch.setattr(download_cli.cdse, "download_resume", fake_resume)
    result_json = str(tmp_path / "_result.json")
    rc = download_cli.main([
        "--roi", "roi.geojson", "--start", "2018-04-01", "--end", "2018-06-01",
        "--bands", "B04", "--dst", str(tmp_path), "--catalog", str(tmp_path / "c.parquet"),
        "--max-tiles", "5", "--result-json", result_json, "--no-probe",
        "--creds", str(tmp_path / "creds.json"), "--max-concurrent-s3", "2",
    ])
    assert rc == 0
    assert captured["max_concurrent_s3"] == 2
    m = json.loads(open(result_json).read())["metrics"]
    assert m["aggregate_mb_per_s"] == 0.5    # 2 MB / 4 thread-summed s (per-stream)
    assert m["wall_transfer_mb_per_s"] == 2.0  # 2 MB / 1 wall s (effective, all streams)
    assert m["transfer_wall_s"] == 1.0


def test_cli_error_populated_on_failed_status(monkeypatch, tmp_path):
    """A run that reports failures (non-exception) fills `error` with a short reason, not None."""
    monkeypatch.setattr(
        download_cli.CdseCredentials, "from_json",
        classmethod(lambda cls, fp, **kw: cls(**DUMMY_FIELDS)),
    )
    monkeypatch.setattr(
        download_cli.cdse, "download_resume",
        lambda *a, **k: [cdse.DownloadResult(successful_count=3, total_count=5, failed_count=2)],
    )
    result_json = str(tmp_path / "_result.json")
    rc = download_cli.main([
        "--roi", "roi.geojson", "--start", "2018-04-01", "--end", "2018-06-01",
        "--bands", "B04", "--dst", str(tmp_path), "--catalog", str(tmp_path / "c.parquet"),
        "--max-tiles", "5", "--result-json", result_json, "--no-probe",
        "--creds", str(tmp_path / "creds.json"),
    ])
    assert rc != 0
    data = json.loads(open(result_json).read())
    assert data["status"] == "failed"
    assert data["error"] and "terminal pass" in data["error"]


def test_cli_exception_writes_failed_result_then_reraises(monkeypatch, tmp_path):
    """spec 26 §4: if download_resume raises, the CLI still writes a status="failed" result
    (with error=repr(exc)) so the runbook flow has something to paste, then re-raises."""
    import pytest

    monkeypatch.setattr(
        download_cli.CdseCredentials, "from_json",
        classmethod(lambda cls, fp, **kw: cls(**DUMMY_FIELDS)),
    )

    def boom(*a, **k):
        raise RuntimeError("STAC endpoint unreachable")

    monkeypatch.setattr(download_cli.cdse, "download_resume", boom)
    result_json = str(tmp_path / "_result.json")
    with pytest.raises(RuntimeError, match="STAC endpoint unreachable"):
        download_cli.main([
            "--roi", "roi.geojson", "--start", "2018-04-01", "--end", "2018-06-01",
            "--bands", "B04", "--dst", str(tmp_path), "--catalog", str(tmp_path / "c.parquet"),
            "--max-tiles", "5", "--result-json", result_json, "--no-probe",
            "--creds", str(tmp_path / "creds.json"),
        ])
    data = json.loads(open(result_json).read())
    assert data["status"] == "failed"
    assert "STAC endpoint unreachable" in data["error"]


def test_cli_forwards_processing_cloudcover_cog_and_passes(monkeypatch, tmp_path):
    """A14 (#145): `--processing`, `--max-cloudcover`, `--no-cog` and `--max-passes` reach
    `download_resume` (`--no-cog` as `cog=False`)."""
    monkeypatch.setattr(
        download_cli.CdseCredentials, "from_json",
        classmethod(lambda cls, fp, **kw: cls(**DUMMY_FIELDS)),
    )
    captured = {}

    def fake_resume(*a, **k):
        captured.update(k)
        return [cdse.DownloadResult(successful_count=1, total_count=1, failed_count=0)]

    monkeypatch.setattr(download_cli.cdse, "download_resume", fake_resume)
    rc = download_cli.main([
        "--roi", "roi.geojson", "--start", "2018-04-01", "--end", "2018-06-01",
        "--bands", "B04", "--dst", str(tmp_path), "--catalog", str(tmp_path / "c.parquet"),
        "--max-tiles", "5", "--no-probe", "--creds", str(tmp_path / "creds.json"),
        "--processing", "<=2.0", "--max-cloudcover", "12.5", "--no-cog", "--max-passes", "3",
    ])
    assert rc == 0
    assert (captured["processing"], captured["max_cloudcover"], captured["cog"],
            captured["max_passes"]) == ("<=2.0", 12.5, False, 3)
