"""Spec 13 · Part C — combine the JP2 and COG harness runs into one team report.

Reads the two tagged `datacube_throughput_stats_<tag>.json` (from `--tag jp2` / `--tag cog`
sweeps) plus `cog_vs_jp2_storage.json` (from prep), and emits `cog_vs_jp2_report.md`:
a per-`cores` time table (JP2 vs COG), the JP2-vs-COG duration-vs-concurrency overlay
(the decode-bound test — does COG lower *and* flatten the curve?), the 3-row storage table,
and a space-vs-time verdict.

Run:  python -m benchmarks.compare_cog_jp2                 # tags jp2, cog
      python -m benchmarks.compare_cog_jp2 --jp2 jp2 --cog cog
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rise_eo.storage import fs

ROOT = str(Path(__file__).resolve().parents[2])  # workspace root: the folder holding fsd/ and the archives
BENCH = f"{ROOT}/fsd/benchmarks"
STORAGE_JSON = f"{BENCH}/cog_vs_jp2_storage.json"
REPORT = f"{BENCH}/cog_vs_jp2_report.md"
FIG_DIR = f"{BENCH}/cog_vs_jp2_figures"


def _stats_path(tag: str) -> str:
    return f"{BENCH}/datacube_throughput_stats_{tag}.json"


# --- pure comparison (unit-tested) -------------------------------------------

def compare_time(jp2_rows: list[dict], cog_rows: list[dict]) -> list[dict]:
    """Per-`cores` JP2-vs-COG time comparison. PURE.

    speedup = jp2 / cog (>1 means COG faster). Uses total wall + the `load_images`
    phase sum + mean `load_images`/grid."""
    jp = {r["cores"]: r for r in jp2_rows}
    cg = {r["cores"]: r for r in cog_rows}
    out = []
    for cores in sorted(set(jp) & set(cg)):
        j, c = jp[cores], cg[cores]
        jl = j["phase_sum"]["load_images"]
        cl = c["phase_sum"]["load_images"]
        out.append({
            "cores": cores,
            "jp2_total": j["total_seconds"], "cog_total": c["total_seconds"],
            "wall_speedup": round(j["total_seconds"] / c["total_seconds"], 2)
            if c["total_seconds"] else None,
            "jp2_load": round(jl, 1), "cog_load": round(cl, 1),
            "load_speedup": round(jl / cl, 2) if cl else None,
            "jp2_mean_load": j["mean_load_per_grid"], "cog_mean_load": c["mean_load_per_grid"],
        })
    return out


def _curve_span(row: dict):
    """(lo_conc, hi_conc, dur_lo, dur_hi, ratio) from a row's duration_curve, or None."""
    dc = row.get("duration_curve", {}).get("all")
    if not dc:
        return None
    ks = sorted(dc, key=int)
    lo, hi = dc[ks[0]]["mean_s"], dc[ks[-1]]["mean_s"]
    return int(ks[0]), int(ks[-1]), lo, hi, (round(hi / lo, 2) if lo else None)


# --- plot + report -----------------------------------------------------------

def _overlay_plot(jp2_rows, cog_rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def busiest(rows):
        r = max((x for x in rows if x.get("duration_curve")), key=lambda x: x["cores"],
                default=None)
        return r

    jr, cr = busiest(jp2_rows), busiest(cog_rows)
    if not jr or not cr:
        return None
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row, name, style in ((jr, "JP2", "o-"), (cr, "COG", "s--")):
        dc = row["duration_curve"]["all"]
        xs = sorted(dc, key=int)
        ax.plot([int(k) for k in xs], [dc[k]["mean_s"] for k in xs], style,
                label=f"{name} (cores={row['cores']})")
    ax.set_xlabel("reads in flight (concurrency)")
    ax.set_ylabel("mean read duration (s)")
    ax.set_title("JP2 vs COG — read duration vs concurrency (decode-bound test)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/duration_overlay.png", dpi=110)
    plt.close(fig)
    return f"{os.path.basename(FIG_DIR)}/duration_overlay.png"


def write_report(time_rows, jp2_rows, cog_rows, storage, meta):
    g = lambda x: f"{x / 2**30:.2f}" if x else "0.00"  # noqa: E731
    L = ["# COG vs JP2 — storage-vs-time (spec 13)\n",
         f"_generated {meta['run_utc']}_\n",
         "Same grids, window and `cores` sweep; the two runs differ **only** in tile format "
         "(native CDSE JP2 vs base COG = DEFLATE+PREDICTOR=2, tiled, no overviews). "
         "`speedup` = JP2 / COG (>1 = COG faster).\n"]

    L.append("\n## Time\n")
    L.append("| cores | JP2 wall (s) | COG wall (s) | wall × | JP2 load | COG load | load × "
             "| JP2 mean load/grid | COG mean load/grid |\n"
             "|---|---|---|---|---|---|---|---|---|\n")
    for r in time_rows:
        L.append(f"| {r['cores']} | {r['jp2_total']} | {r['cog_total']} | {r['wall_speedup']}× "
                 f"| {r['jp2_load']} | {r['cog_load']} | {r['load_speedup']}× "
                 f"| {r['jp2_mean_load']} | {r['cog_mean_load']} |\n")

    overlay = _overlay_plot(jp2_rows, cog_rows)
    if overlay:
        L.append(f"\n![duration overlay]({overlay})\n")

    L.append("\n## Storage\n")
    s = storage["storage"]
    L.append("| | JP2 (GiB) | base COG (GiB) | ratio | +overviews (GiB, est) |\n"
             "|---|---|---|---|---|\n")
    L.append(f"| total | {g(s['total_jp2'])} | {g(s['total_cog'])} | {s['cog_ratio']}× "
             f"| {g(s['total_cog_ovr'])} |\n")

    # verdict
    jspan = _curve_span(max((r for r in jp2_rows if r.get('duration_curve')),
                            key=lambda x: x['cores'], default={}))
    cspan = _curve_span(max((r for r in cog_rows if r.get('duration_curve')),
                            key=lambda x: x['cores'], default={}))
    best = max(time_rows, key=lambda r: (r["wall_speedup"] or 0), default=None)
    L.append("\n## Verdict — space vs time\n")
    if best:
        L.append(f"- **Time:** COG is up to **{best['wall_speedup']}× faster wall** "
                 f"(cores={best['cores']}); `load_images` up to **{best['load_speedup']}× "
                 f"faster**.\n")
    if jspan and cspan:
        flat = ("flatter" if cspan[4] and jspan[4] and cspan[4] < jspan[4] else "similar")
        L.append(f"- **Decode-bound test:** JP2 read duration rose {jspan[4]}× with "
                 f"concurrency vs COG {cspan[4]}× — COG's curve is **{flat}**. "
                 f"{'A flatter, lower COG curve = the JP2 wavelet *decode* was the bottleneck.' if flat == 'flatter' else 'Similar curves = contention was not decode-bound.'}\n")
    L.append(f"- **Space:** COG costs **{s['cog_ratio']}× the JP2 storage** "
             f"(+{round((s['cog_ratio'] - 1) * 100)}%); extrapolated to a year ≈ "
             f"{g(s['total_cog'] * 12 / meta.get('months', 4))} GiB vs "
             f"{g(s['total_jp2'] * 12 / meta.get('months', 4))} GiB JP2. Overviews (tiling only) "
             f"would add ~{s['overview_delta_pct']}%.\n")
    L.append("\n_The team's call: is the wall/read speedup worth the extra disk?_\n")

    with fs.open(REPORT, "w") as f:
        f.write("".join(L))
    print(f"[compare] -> {REPORT}", flush=True)


def main(argv=None):
    import datetime

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jp2", default="jp2", help="tag of the JP2 sweep")
    ap.add_argument("--cog", default="cog", help="tag of the COG sweep")
    args = ap.parse_args(argv)

    with fs.open(_stats_path(args.jp2)) as f:
        jp2 = json.load(f)
    with fs.open(_stats_path(args.cog)) as f:
        cog = json.load(f)
    with fs.open(STORAGE_JSON) as f:
        storage = json.load(f)

    time_rows = compare_time(jp2["sweep"], cog["sweep"])
    meta = {"run_utc": datetime.datetime.utcnow().isoformat() + "Z",
            "months": storage["meta"].get("months", 4)}
    write_report(time_rows, jp2["sweep"], cog["sweep"], storage, meta)


if __name__ == "__main__":
    main()
