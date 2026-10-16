"""Probe 03b — fsd's `/vsiadls/` read of the SAME blob object probe 03 wrote.

Closes the one number probe 03 cannot produce. Probe 03 measured rslearn reading Azure blob
through an fsspec file object (21-24 MB/s on the first VM run). That figure means nothing on
its own -- it needs fsd's own route over the same object, on the same VM, in the same minute,
to be a comparison rather than a datum.

**This runs in fsd's venv (`.venv`), not the spike venv.** The charter keeps `.venv-rslearn`
free of fsd, which is exactly why probe 03's `q4_vs_fsd` skips: no single interpreter has both.
Two probes, one object, two environments -- that is the design, not a workaround.

## Why the comparison is the whole point

`fsd` routes remote rasters through GDAL's `/vsiadls/` handler with a refreshed
`AZURE_STORAGE_ACCESS_TOKEN` (`src/fsd/raster/__init__.py:56-92`, spec 31). rslearn hands
rasterio an fsspec file object instead (`rslearn/utils/fsspec.py:157-179`) and never gives GDAL
a URL. Both work. fsd's route exists because GDAL's remote reader has optimizations -- range
caching, overview handling -- that a plain Python file object does not get. **Whether that
buys anything measurable is the open question in report section 4.1.1**, and it is the last
input to the Option B price.

## Procedure

1. Run probe 03 **with `--keep`** so the object survives.
2. Run this, in fsd's venv, pointing at the geotiff probe 03 left behind:
   `<dst-prefix>/raster_format/geotiff.tif` (`GeotiffRasterFormat.fname`,
   `raster_format.py:510`).

Usage (see `../RUNBOOK-rslearn-spike.md` Step 3c):

    python spike/probes/probe_03b_fsd_vsi_baseline.py \
        --out tests/outputs/rslearn_spike \
        --url "$AZ_ROOT/$AZ_SCRATCH_PREFIX/rslearn_spike/raster_format/geotiff.tif"

Reads only. Writes nothing to blob. Like probe 03, the result records the URL's *shape*, never
the URL -- this branch is a public MIT repo.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np

# Repeat the read so a single cold TLS handshake does not become "the" number. The first read
# is reported separately -- for a fan-out that opens each COG once, the COLD figure is the
# honest one, and the warm figure only shows what caching would buy.
N_READS = 3


def _redact(url: str) -> str:
    """The URL's shape, never its identifiers. Mirrors probe 03's `_redact`."""
    if "://" not in url:
        return "<no scheme>"
    scheme, rest = url.split("://", 1)
    n_segments = len([s for s in rest.split("/")[1:] if s])
    return f"{scheme}://<fs>@<account>/<{n_segments} path segments>"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output directory for _result.json")
    ap.add_argument(
        "--url",
        required=True,
        help="abfss:// URL of the geotiff probe 03 left behind (run it with --keep)",
    )
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    def _fail(msg: str) -> int:
        result = {
            "step": "probe_03b_fsd_vsi_baseline",
            "status": "fail",
            "pass": False,
            "metrics": {"url_shape": _redact(args.url)},
            "expected": {},
            "error": msg,
        }
        (outdir / "_result_probe03b.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return 1

    try:
        from fsd.raster import rio_open, to_vsi
    except ImportError as exc:
        return _fail(
            f"fsd not importable ({exc}). This probe runs in fsd's venv (.venv), NOT "
            ".venv-rslearn -- see the module docstring."
        )

    vsi = to_vsi(args.url)
    if vsi == args.url:
        return _fail(
            f"--url did not translate to a /vsiadls/ path, so this would not exercise the VSI "
            f"route at all. Expected an abfss:// or az:// URL; got shape {_redact(args.url)}."
        )

    # Time the AAD token acquisition SEPARATELY, because the first `rio_open` pays for it and
    # that would otherwise be charged to "read throughput". `storage_token()` fetches from a
    # module-cached `DefaultAzureCredential` whose MSAL cache refreshes internally
    # (`src/fsd/storage/azure.py:48-56`), so call 1 is a real network round-trip to AAD/IMDS
    # and calls 2..N are ~free. Without this split, fsd's cold number conflates one-time auth
    # with transfer -- and the rslearn side of the comparison does NOT pay auth in its timed
    # read at all, because probe 03's q1 resolves the filesystem first. See run-book Step 3c.
    token_timings = []
    try:
        from fsd.storage.azure import storage_token

        for _ in range(2):
            t0 = time.perf_counter()
            storage_token()
            token_timings.append(round(time.perf_counter() - t0, 4))
    except Exception as exc:  # noqa: BLE001 -- informational; the reads below are the measurement
        token_timings = [f"unavailable: {type(exc).__name__}: {exc}"]

    reads: list[dict] = []
    mb = None
    shape = None
    for i in range(N_READS):
        t0 = time.perf_counter()
        try:
            with rio_open(args.url) as src:
                array = src.read()
            secs = time.perf_counter() - t0
            mb = array.nbytes / 1e6
            shape = list(array.shape)
            reads.append(
                {
                    "read": i,
                    "seconds": round(secs, 3),
                    "mb_per_s": round(mb / max(secs, 1e-9), 1),
                }
            )
        except BaseException as exc:  # noqa: BLE001 -- a failure here is the finding
            return _fail(
                f"read {i} failed: {type(exc).__name__}: {exc} | "
                f"{traceback.format_exc().strip().splitlines()[-1]}"
            )

    cold = reads[0]
    warm = reads[1:]
    metrics = {
        "url_shape": _redact(args.url),
        "route": "/vsiadls/ + AZURE_STORAGE_ACCESS_TOKEN (fsd spec 31)",
        "array_shape": shape,
        "mb": round(mb, 1) if mb is not None else None,
        "cold_read_seconds": cold["seconds"],
        "cold_read_mb_per_s": cold["mb_per_s"],
        "warm_read_mb_per_s": [r["mb_per_s"] for r in warm],
        "median_mb_per_s": round(float(np.median([r["mb_per_s"] for r in reads])), 1),
        "aad_token_seconds": token_timings,
        "reads": reads,
        "comparison_note": (
            "Compare WARM against probe 03's q2_raster_format.read_mb_per_s, not cold. "
            "probe 03 resolves the filesystem in q1 before timing q2, so its read is measured "
            "with auth already warm; charging fsd's one-time AAD round-trip against its read "
            "would not be like for like. aad_token_seconds[0] is that one-time cost."
        ),
    }

    result = {
        "step": "probe_03b_fsd_vsi_baseline",
        "status": "ok",
        "pass": True,
        "metrics": metrics,
        "expected": {
            "note": (
                "There is no pass/fail here -- it is a baseline. Compare "
                "metrics.cold_read_mb_per_s against probe 03's "
                "q2_raster_format.read_mb_per_s, which was 23.6 MB/s on the first VM run. "
                "If they are within noise of each other, GDAL's /vsiadls/ optimizations buy "
                "nothing measurable at this object size and report section 4.1.1's open "
                "question closes in rslearn's favour. If fsd is materially faster, that is a "
                "real per-granule cost to price into Options A and B. Note this is ONE 6 MB "
                "object -- a fair test of a full COG read pattern (overviews, windowed reads, "
                "many small range requests) would need the real archive."
            )
        },
        "error": None,
    }
    (outdir / "_result_probe03b.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
