"""First real batch-download run (TODO #9): 1 year of the Ethiopia multi-CRS ROI,
4 bands. Exercises rise_eo.sources.cdse.download end-to-end and records fast-fail stats.

Run: .venv/bin/python benchmarks/download_year_ethiopia.py
Writes a stats JSON next to this file; the .md report is composed from it.
"""

import datetime
import json
import os
from pathlib import Path

from rise_eo.catalog.catalog import TileCatalog
from rise_eo.sources import cdse

WS = str(Path(__file__).resolve().parents[2])  # workspace root: the folder holding fsd/ and the archives
ROI = f"{WS}/shapefiles/s2grid=165bca4.geojson"
# Dedicated benchmark tree so the real `satellite/` data stays unpolluted.
ROOT = f"{WS}/satellite_benchmark/sentinel-2-l2a"
CATALOG = f"{ROOT}/catalog.parquet"            # fresh rise-eo GeoParquet catalog
BANDS = ["B04", "B08", "B8A", "SCL"]
START = datetime.datetime(2018, 1, 1)
END = datetime.datetime(2019, 1, 1)
STATS_JSON = os.path.join(os.path.dirname(__file__), "download_year_ethiopia_stats.json")


def dir_bytes(path):
    total = 0
    for r, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return total


def main():
    from rise_eo import config

    creds = cdse.CdseCredentials.from_json(f"{WS}/secrets/cdse_credentials.json")
    catalog = TileCatalog(CATALOG)

    passes = []

    def on_pass(i, r):
        # Persist cumulative stats after each pass — survives a kill (BUG-001).
        passes.append({
            "pass": i,
            "elapsed_s": round(r.elapsed_s, 1),
            "attempted": r.total_count,
            "ok": r.successful_count,
            "skipped": r.skipped_count,
            "failed": r.failed_count,
            "reason_counts": r.reason_counts,
            "circuit_tripped": r.circuit_tripped,
        })
        nbytes = dir_bytes(ROOT)
        with open(STATS_JSON, "w") as f:
            json.dump({
                "run_utc": datetime.datetime.utcnow().isoformat() + "Z",
                "roi": os.path.basename(ROI),
                "window": [START.isoformat(), END.isoformat()],
                "bands": BANDS,
                "concurrency": config.MAX_CONCURRENT_S3,
                "chunksize": 100,
                "gb_on_disk": round(nbytes / 1e9, 2),
                "catalog_rows": len(catalog.read()) if os.path.exists(CATALOG) else 0,
                "passes": passes,
            }, f, indent=2, default=str)
        print(f"[pass {i}] ok={r.successful_count} fail={r.failed_count} "
              f"tripped={r.circuit_tripped} gb={round(nbytes / 1e9, 2)}")

    results = cdse.download_resume(
        ROI, START, END, BANDS, ROOT, catalog, creds,
        max_tiles=600, chunksize=100, progress=True,
        max_consecutive_failures=15, max_passes=50, cooldown_s=60.0, on_pass=on_pass,
    )
    print("DONE passes:", len(results), "final gb:", round(dir_bytes(ROOT) / 1e9, 2))


if __name__ == "__main__":
    main()
