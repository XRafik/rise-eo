"""In-job entrypoint for the AML download dispatcher.

A thin CLI wrapping the existing download-to-blob path -- no pipeline logic of its
own, mirroring `rise_eo.workflows.shard`'s role for the build fan-out. Two modes, matching the
two job shapes `runners.run_aml_download` submits -- dispatch shape is per-source:

- `--roi <url> ...` -> the whole-ROI CDSE job (exactly one per run). Calls the
  unmodified `sources.cdse.download` directly, reading its S3 creds **on the
  node** from exactly one of two mutually exclusive sources:
  Key Vault (`--vault-url`/`--secret-name`, non-secret command args) or a blob
  JSON (`--creds-url`, a non-secret location). The secret value itself is never
  in the job spec.
- `--shard <url> ...` -> one of N per-shard MPC jobs. Calls the additive
  `sources.mpc.download_shard` over a pre-discovered, pre-partitioned asset-row CSV
  the driver wrote (`sources.mpc.discover_shard_rows` + `runners.shard_units`).

Both write a `_status/<k>.json` -- the same `_result.json` shape every fsd job reports in
-- built from the source call's `DownloadResult`.

Run as:
  python -m rise_eo.workflows.download --roi <url> --startdate <iso> --enddate <iso> \\
      --bands B04,B08 --dst <url> --catalog <url> --max-tiles N --vault-url <url> \\
      --secret-name <name> --status-url <url>
  python -m rise_eo.workflows.download --roi <url> --startdate <iso> --enddate <iso> \\
      --bands B04,B08 --dst <url> --catalog <url> --max-tiles N \\
      --creds-url <url> --status-url <url>
  python -m rise_eo.workflows.download --shard <url> --dst <url> --catalog <url> \\
      --status-url <url>
"""

from __future__ import annotations

import datetime as _dt

# Stamped before any heavy import, so it is process start (see workflows/shard.py).
_PROCESS_START_AT = _dt.datetime.now(_dt.timezone.utc).isoformat()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import pandas as pd  # noqa: E402

from rise_eo import secrets  # noqa: E402
from rise_eo.catalog.catalog import TileCatalog  # noqa: E402
from rise_eo.sources import cdse, mpc  # noqa: E402
from rise_eo.storage import fs  # noqa: E402


def _write_status(status_url: str, status: dict) -> None:
    with fs.open(status_url, "w") as f:
        json.dump(status, f, indent=2)


def _status_from_download_result(
    dr, *, unit: str, process_start_at: str, work_start_at: str, work_end_at: str,
) -> dict:
    """`DownloadResult` (cdse or mpc) -> the status dict.

    Both dataclasses carry the same core fields; `circuit_tripped`/`bytes_downloaded` are
    CDSE-only, so they default off/0 for an MPC shard, which has neither a circuit breaker
    nor its own transfer-byte accounting. `ended_at` is added by the caller, right before
    the `_status` write.
    """
    circuit_tripped = bool(getattr(dr, "circuit_tripped", False))
    failed = dr.failed_count > 0 or circuit_tripped
    return {
        "unit": unit,
        "status": "failed" if failed else "ok",
        "n_assets": dr.total_count,
        "n_skipped": dr.skipped_count,
        "n_failed": dr.failed_count,
        "bytes_downloaded": getattr(dr, "bytes_downloaded", 0),
        "process_start_at": process_start_at,
        "work_start_at": work_start_at,
        "work_end_at": work_end_at,
        "seconds": round(dr.elapsed_s, 3),
        "circuit_tripped": circuit_tripped,
        "error": None if not failed else f"{dr.failed_count} asset(s) failed to download",
    }


def run_roi(
    *,
    roi: str,
    startdate: str,
    enddate: str,
    bands: list[str],
    dst: str,
    catalog: str,
    max_tiles: int,
    status_url: str,
    max_cloudcover: float | None = None,
    cog: bool = True,
    vault_url: str | None = None,
    secret_name: str | None = None,
    creds_url: str | None = None,
    processing: str = "latest",
) -> dict:
    """`--roi` mode: the whole-ROI CDSE job.

    Reads S3 creds from exactly one of two mutually exclusive sources -- Key Vault
    (`vault_url`/`secret_name`) or a blob JSON (`creds_url`) -- then calls
    `sources.cdse.download` unmodified.
    """
    if creds_url:
        creds = cdse.CdseCredentials.from_json(creds_url)
    else:
        creds_json = secrets.get_secret(vault_url, secret_name)
        creds = cdse.CdseCredentials.from_json_str(creds_json)

    catalog_obj = TileCatalog(catalog)
    start = pd.Timestamp(startdate)
    end = pd.Timestamp(enddate)

    work_start_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    result = cdse.download(
        roi, start, end, bands, dst, catalog_obj, creds,
        max_tiles=max_tiles, max_cloudcover=max_cloudcover, cog=cog, progress=False,
        processing=processing,
    )
    work_end_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    status = _status_from_download_result(
        result, unit="roi", process_start_at=_PROCESS_START_AT,
        work_start_at=work_start_at, work_end_at=work_end_at,
    )
    status["ended_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    _write_status(status_url, status)
    return status


def run_shard(*, shard_url: str, dst: str, catalog: str, status_url: str) -> dict:
    """`--shard` mode: one of N per-shard MPC jobs over a pre-discovered,
    pre-partitioned asset-row CSV. No credentials needed: MPC is anonymous."""
    # Identity strings must survive the CSV round trip: a bare read_csv turns the S2
    # baseline "05.00" into the float 5.0 (spec 59 D8: a normalized string).
    with fs.open(shard_url, "r") as f:
        rows = pd.read_csv(
            f, dtype={"tile_id": str, "acquisition_key": str, "processing_version": str},
        ).to_dict("records")

    catalog_obj = TileCatalog(catalog)
    work_start_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    result = mpc.download_shard(rows, dst, catalog_obj)
    work_end_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    status = _status_from_download_result(
        result, unit=os.path.basename(shard_url), process_start_at=_PROCESS_START_AT,
        work_start_at=work_start_at, work_end_at=work_end_at,
    )
    status["ended_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    _write_status(status_url, status)
    return status


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m rise_eo.workflows.download",
        description="Run one AML-dispatched download unit (spec 37): a whole CDSE "
                     "ROI job, or one MPC shard.",
    )
    p.add_argument("--roi", help="ROI url (geopandas-readable path); --roi mode (CDSE)")
    p.add_argument("--shard", help="pre-discovered asset-row shard csv url; --shard mode (MPC)")
    p.add_argument("--startdate")
    p.add_argument("--enddate")
    p.add_argument("--bands", help="comma-separated band list")
    p.add_argument("--dst", required=True)
    p.add_argument("--catalog", required=True)
    p.add_argument("--max-tiles", type=int)
    p.add_argument("--max-cloudcover", type=float, default=None)
    p.add_argument("--no-cog", action="store_true")
    p.add_argument("--processing", default="latest",
                   help="'latest' or a PEP 440 specifier, e.g. '>=05.00' (spec 59 D7); "
                        "--roi mode (CDSE) only -- MPC selects on the driver")
    p.add_argument("--vault-url")
    p.add_argument("--secret-name")
    p.add_argument("--creds-url", help="blob JSON CDSE creds location (D5 REVISED, mutually "
                                        "exclusive with --vault-url/--secret-name)")
    p.add_argument("--status-url", required=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if bool(args.roi) == bool(args.shard):
        raise SystemExit("exactly one of --roi or --shard is required")

    if args.roi:
        status = run_roi(
            roi=args.roi, startdate=args.startdate, enddate=args.enddate,
            bands=args.bands.split(","), dst=args.dst, catalog=args.catalog,
            max_tiles=args.max_tiles, status_url=args.status_url,
            max_cloudcover=args.max_cloudcover, cog=not args.no_cog,
            vault_url=args.vault_url, secret_name=args.secret_name,
            creds_url=args.creds_url, processing=args.processing,
        )
    else:
        status = run_shard(shard_url=args.shard, dst=args.dst, catalog=args.catalog,
                            status_url=args.status_url)

    if status["status"] != "ok":
        raise SystemExit(f"download unit failed: {status}")


if __name__ == "__main__":
    main()
