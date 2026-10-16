---
status: historical
summary: First real batch-download benchmark against the satellite_benchmark/-era CDSE pipeline; the archive it fed no longer exists (deleted 2026-07-17, CLAUDE.md).
---

# Download benchmark #1 — 1 year, Ethiopia multi-CRS ROI

First real batch-download run of `rise_eo.sources.cdse.download` (TODO #9). Purpose: see
how the STAC-discovery + S3-transfer pipeline behaves at scale and **measure the
fast-fail rate** against CDSE's flaky S3 endpoint (BUG-001).

## Outcome — COMPLETED 2026-07-02 ✅
**579/579 tiles fully downloaded** (2,316 `.jp2` + 579 `MTD_TL.xml`), balanced across
both UTM zones (289× EPSG:32637, 290× EPSG:32636), ~94 GiB in `satellite_benchmark/`.
Integrity verified: **0 zero-byte, 0 truncated, 0 leftover `.part`**. The final pass
was clean: `attempted=2895, ok=2895, failed=0` in 66 min (924 new + 1971 skips).

Getting there took several sessions across good/bad CDSE windows and surfaced four
real production bugs — each fixed with tests:

| Symptom | Root cause | Fix |
|---|---|---|
| Infinite hang (bar frozen, workers stuck) | no S3 read/connect timeout | timeouts via `config_kwargs`; timeout/conn errors retryable |
| 1,822 zero-byte + 27 truncated files | non-atomic write; lazy `open("rb")` created a 0-byte dst before the failing GET | atomic `.part`+rename; `_download_one` skips only if size>0 |
| "Frozen" progress (twice — not actually stuck) | tqdm `\r` bar doesn't animate in a redirected log | flushed newline progress lines (with ETA) |
| Killed at ~70 min, no traceback | memory pressure / jetsam (disk was 97% full → no swap) | freed disk; fresh-process resume loop + disk guard |

Everything below is the **first-pass (bad-window) measurement** that motivated the
resilience work — kept for the record.

## Configuration
| | |
|---|---|
| ROI | `shapefiles/s2grid=165bca4.geojson` (Ethiopia, straddles 36°E) |
| Window | 2018-01-01 → 2019-01-01 (1 year) |
| Bands | `B04, B08, B8A, SCL` (+ `MTD_TL.xml`) = 5 files/tile |
| Tiles matched (STAC) | **579** across 4 MGRS / 2 UTM zones (T37PBP+T37PBN=32637, T36PZU+T36PZT=32636) |
| Work items | 2,895 files |
| Destination | `satellite_benchmark/sentinel-2-l2a/` (isolated from real `satellite/`) |
| Concurrency | 4 (`config.MAX_CONCURRENT_S3`, CDSE documented quota) |
| Retry | fail-fast, 3 tries, short capped jittered backoff; `Forbidden`/403 treated retryable |
| Endpoint | OTC-pinned `eodata.ams.dataspace.copernicus.eu` |
| Date run | 2026-07-02 |

> The run was manually stopped at **96% (2776/2895 files, 49 min)**; numbers below are
> from that near-complete pass (catalog is written per chunk, so the data persisted;
> the end-of-run stats JSON did not write because of the kill).

## Headline result
**~22.5% file-level success during a sustained bad CDSE window.**

| Metric | Value |
|---|---|
| Files attempted | 2,776 / 2,895 (96%) |
| Succeeded (`ok`) | **623 (~22.5%)** |
| Failed (fast-fail) | **2,152 (~77.5%)** |
| Tiles touched (≥1 file) | 251 / 579 |
| **Tiles fully complete (5/5 files)** | **80 / 579 (13.8%)** |
| Data on disk | 21.0 GB (avg 34.4 MB/file) |
| Wall time | 49 min 09 s |
| Effective throughput | ~7.3 MB/s / ~0.94 files/s (incl. failures) |

Per-band success (files landed): `MTD_TL.xml` 135 · `B8A` 129 · `SCL` 121 · `B08` 116
· `B04` 109 — i.e. failures are spread evenly across bands (not band-specific), and
most tiles came down **partial** (135 tiles with only 1 file; only 80 with all 5).

## Failure analysis
- **Root cause = BUG-001**, not a rise-eo bug: CDSE's S3 endpoint fails per-request due to
  load-balanced node inconsistency. Live probes the same day: serial 0/3 `ok`; threaded
  a mix of `Forbidden`, `InvalidAccessKeyId`, and `ok` — the classic node roulette.
- **Failure reasons (qualitative):** dominated by `Forbidden` (403), with some
  `InvalidAccessKeyId`. (Exact per-reason counts weren't persisted this run — the
  process was killed before the stats JSON wrote. Fix: persist `reason_counts`
  incrementally; see recommendations.)
- **Retry helped but can't beat a sustained bad window.** Making `Forbidden` retryable
  moved us from an earlier **0%** pass (403 was fast-failing instantly) to ~22.5%. But
  6 retries ≈ 1 retry when the whole window is bad — confirming the report's thesis:
  **fail-fast + resume-later**, not in-run grinding.
- **ETA caveat:** tqdm showed ~22 min early (fast-fails are quick) but real time was
  ~50 min — successful 45 MB files dominate once they start landing, so early ETA is
  optimistic. Report ETA converges upward as success rate rises.

## What worked
- **STAC discovery** (anonymous, `pystac-client`): 579 tiles, unique ids, zero S3
  listing — never the failure point.
- **Idempotent, per-chunk catalog** is a real checkpoint: the killed run left a valid
  `catalog.parquet` with 251 partial tiles; a re-run skips landed files and `files`-
  unions the rest, so tiles complete across passes.
- **Multi-CRS** handled end to end (both zones present in the catalog).
- **Progress bar** (`download(progress=True)`) gives live `ok`/`fail` + ETA.

## Implications / recommendations
1. **This window was bad.** At ~22.5% success/attempt, reaching ~full coverage needs
   ~15–18 idempotent passes — wasteful. **Run during a good window** (retry the whole
   job at a different time of day) or…
2. **Circuit breaker** (BUG-001 open item): after N consecutive failures, stop the pass
   (bad window) instead of grinding, and resume later. Biggest bang for the buck.
3. **Resume-loop runner**: wrap `download` in a loop (e.g. per-month calls) that
   aggregates `DownloadResult.reason_counts` and **writes stats after each iteration**,
   so a kill never loses the breakdown, and coverage accrues pass over pass.
4. **Quota probing** (still TODO #9): once a good window is found, vary `max_workers`
   to find CDSE's real concurrency ceiling (needs the `max_workers` param).
5. **Report to CDSE** with these numbers (server-side infra; only they can fix the root
   cause).

## Reproduce
`benchmarks/download_year_ethiopia.py` (edit window/bands/root). Writes
`download_year_ethiopia_stats.json` on clean completion; the on-disk `catalog.parquet`
is the durable checkpoint.
