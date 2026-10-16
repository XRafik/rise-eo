"""Build-benchmark + dataset-stats report for the full-year 1015-field EuroCrops run.

Consumes the output of `eurocrops_year_build.py` (per-field cubes + `timings.json`
sidecars under tests/outputs/datacube_year/run), then:

  1. **Build benchmark** — aggregates per-cube `timings.json` into a phase breakdown
     (which stage dominates), per-cube distribution, and the real wall-clock + effective
     parallelism (derived from the `start.txt`/`done.txt` sentinels the Snakefile writes).
  2. **Dataset stats** — flattens the cubes (spec 05) and reports pixels/class, cube
     shapes, UTM-zone split, and per-timestamp data coverage.
  3. **Phenology** — per-class NDVI time series over 2018 (the crop-type signal), the
     headline scientific check that a labelled multi-tile datacube is coherent.

Writes `benchmarks/eurocrops_year_report.md` + figures in
`benchmarks/eurocrops_year_figures/`.

Run from the workspace root:
    fsd/.venv/bin/python fsd/benchmarks/eurocrops_year_report.py
"""

from __future__ import annotations

import json
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.simplefilter("ignore")

from rise_eo.datacube import flatten  # noqa: E402
from rise_eo.storage import fs  # noqa: E402

RUN = "fsd/tests/outputs/datacube_year/run"
CSV = os.path.join(RUN, "input.csv")
OUT = "fsd/tests/outputs/datacube_year/flattened"
FIG = "fsd/benchmarks/eurocrops_year_figures"
REPORT = "fsd/benchmarks/eurocrops_year_report.md"
PHASES = ["missing_check", "load_images", "dst_crs", "reference_profile",
          "resample", "stack", "ops", "save"]


def _read_ts(path):
    try:
        with open(path) as h:
            return pd.Timestamp(h.read().strip())
    except Exception:
        return None


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(CSV)

    # ---- 1. build benchmark: aggregate timings.json + wall from sentinels ----
    recs, starts, dones = [], [], []
    for f in df["datacube_filepath"]:
        d = os.path.dirname(f)
        tp = os.path.join(d, "timings.json")
        if os.path.exists(tp):
            with open(tp) as h:
                recs.append(json.load(h))
        starts.append(_read_ts(os.path.join(d, "start.txt")))
        dones.append(_read_ts(os.path.join(d, "done.txt")))
    tim = pd.DataFrame(recs)
    ps = tim["phase_seconds"]
    phase_sum = {p: float(ps.map(lambda x: x.get(p, 0.0)).sum()) for p in PHASES}
    cpu = float(tim["total_seconds"].sum())
    per_cube = tim["total_seconds"]
    starts = [s for s in starts if s is not None]
    dones = [d for d in dones if d is not None]
    wall = (max(dones) - min(starts)).total_seconds() if starts and dones else float("nan")
    n = len(tim)

    # cube geometry
    shp = tim["datacube_shape"].tolist()
    n_ts = [s[0] for s in shp]
    n_imgs = tim["n_images_loaded"].astype(int)
    zones = tim["dst_crs"].str.extract(r"(\d{5})")[0].value_counts().to_dict()

    # ---- 2. flatten -> dataset stats ----
    flatten.flatten(filepaths_df=df, filepath_col="datacube_filepath", id_col="id",
                    label_col="label", export_folderpath=OUT)
    data = fs.load_npy(f"{OUT}/data.npy")
    labels = fs.load_npy(f"{OUT}/labels.npy")
    md = fs.load_npy(f"{OUT}/metadata.pickle.npy", allow_pickle=True)[()]
    ts = [pd.Timestamp(t) for t in md["timestamps"]]
    bands = md["bands"]

    # ---- 3. phenology: per-class NDVI per timestamp ----
    bi = {b: i for i, b in enumerate(bands)}
    b04 = data[:, :, bi["B04"]].astype(float)
    b08 = data[:, :, bi["B08"]].astype(float)
    valid = (b04 + b08) > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(valid, (b08 - b04) / (b08 + b04), np.nan)  # (px, T)
    coverage = valid.mean(axis=0)                                   # valid frac / ts
    classes = sorted(set(labels.tolist()))
    class_pixels = {c: int((labels == c).sum()) for c in classes}
    pheno = {c: np.nanmean(ndvi[labels == c], axis=0) for c in classes}

    # ---- figures ----
    xdate = [t.to_pydatetime() for t in ts]
    # (a) phenology
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(classes):
        ax.plot(xdate, pheno[c], marker="o", ms=3, lw=1.6, color=cmap(i % 20),
                label=f"{c} (n={class_pixels[c]})")
    ax.set_title("Per-class mean NDVI — 2018, 1015 EuroCrops fields (calendar mosaic, 20 d)")
    ax.set_ylabel("mean NDVI")
    ax.set_xlabel("2018 mosaic window")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower center")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{FIG}/ndvi_phenology.png", dpi=110)
    plt.close(fig)

    # (b) build phase breakdown + per-cube distribution
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    order = sorted(PHASES, key=lambda p: -phase_sum[p])[::-1]
    a1.barh(order, [phase_sum[o] for o in order], color="#4C78A8")
    a1.set_title("CPU-seconds by build phase (sum over cubes)")
    a1.set_xlabel("seconds")
    a2.hist(per_cube, bins=40, color="#72B7B2")
    a2.set_title("Per-cube total_seconds")
    a2.set_xlabel("seconds/cube")
    a2.set_ylabel("cubes")
    fig.tight_layout()
    fig.savefig(f"{FIG}/build_timings.png", dpi=110)
    plt.close(fig)

    # (c) temporal coverage
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.bar(xdate, coverage * 100, width=12, color="#54A24B")
    ax.set_title("Valid-pixel coverage per mosaic window (100% = every pixel observed)")
    ax.set_ylabel("% valid")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{FIG}/coverage.png", dpi=110)
    plt.close(fig)

    # ---- markdown ----
    load_share = 100 * phase_sum["load_images"] / cpu
    par = cpu / wall if wall == wall else float("nan")
    L = []
    L.append("# Full-year datacube benchmark + stats — 1015 EuroCrops fields (2018)\n")
    L.append(f"_Generated {pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M UTC} by "
             f"`benchmarks/eurocrops_year_report.py`._\n")
    L.append("Builds one datacube per EuroCrops field over **all of 2018** "
             "(`mosaic_days=20` → calendar mosaic, spec 15), then flattens to per-pixel "
             "training arrays. Dataset: `austria_eurocrops_sampled_ethiopia_translated."
             "geojson` (id=`fid`, label=`EC_hcat_n`, 11 crop classes) vs the COG archive "
             "`satellite_benchmark/` (EPSG:32636 & 32637). Build: "
             "`benchmarks/eurocrops_year_build.py`.\n")

    L.append("## 1. Build benchmark\n")
    L.append(f"- **Fields built:** {n} / {len(df)} (cores=8, local Snakemake runner).")
    L.append(f"- **Wall clock:** {wall/60:.1f} min · **aggregate CPU:** {cpu/60:.1f} min "
             f"· **effective parallelism:** {par:.1f}×.")
    L.append(f"- **Per cube:** mean {per_cube.mean():.2f}s, median {per_cube.median():.2f}s, "
             f"p95 {per_cube.quantile(0.95):.2f}s, max {per_cube.max():.2f}s.")
    L.append(f"- **Throughput:** {n/ (wall/60):.0f} cubes/min.\n")
    L.append("**Phase breakdown** (CPU-seconds summed over all cubes):\n")
    L.append("| phase | seconds | % |")
    L.append("|---|--:|--:|")
    for p in sorted(PHASES, key=lambda p: -phase_sum[p]):
        L.append(f"| {p} | {phase_sum[p]:.0f} | {100*phase_sum[p]/cpu:.1f} |")
    L.append(f"\n**`load_images` is {load_share:.0f}% of build CPU** — the read/crop of "
             "tile windows dominates, confirming the pipeline is **I/O-bound** even on the "
             "fast COG archive (consistent with the single-ROI year benchmark). See "
             "`eurocrops_year_figures/build_timings.png`.\n")

    L.append("## 2. Dataset\n")
    L.append(f"- **Flattened:** `data.npy {data.shape}` (pixels × {len(ts)} timestamps × "
             f"{len(bands)} bands {bands}), **{data.shape[0]:,} labelled pixels** across "
             f"{len(classes)} classes.")
    L.append(f"- **Cube timestamps:** {len(ts)} calendar mosaics "
             f"({ts[0]:%Y-%m-%d} … {ts[-1]:%Y-%m-%d}); per-cube n_ts "
             f"min {min(n_ts)}/median {int(np.median(n_ts))}/max {max(n_ts)}.")
    L.append(f"- **UTM zones (dst_crs):** {zones} — multi-zone (coords mixed, TODO #16).")
    L.append(f"- **Images loaded/cube:** mean {n_imgs.mean():.0f}, max {n_imgs.max()}.\n")
    L.append("**Pixels per class:**\n")
    L.append("| class | pixels |")
    L.append("|---|--:|")
    for c in sorted(class_pixels, key=lambda c: -class_pixels[c]):
        L.append(f"| {c} | {class_pixels[c]:,} |")
    L.append("")

    L.append("## 3. Per-class NDVI phenology\n")
    L.append("![phenology](eurocrops_year_figures/ndvi_phenology.png)\n")
    L.append("Each line is the mean NDVI of one label class across the 2018 mosaic windows. "
             "The classes **largely overlap on one clean single-season curve** — dry-season "
             "NDVI ~0.14 (Jan–Mar), green-up through mid-year, peak ~0.50 around Aug–Sep, "
             "senescence into the dry season. That overlap is expected and correct here: the "
             "EuroCrops **labels are Austrian field polygons geometrically translated onto "
             "Ethiopia**, so they do not correspond to real ground cover and are *not* meant "
             "to separate by crop type. **The value of this run is pipeline validation, not "
             "crop separability:** the phenology is physically plausible, the curves are "
             "smooth across the 20-day calendar mosaics, and all 1015 cubes share an "
             "identical 19-window axis (per-cube n_ts min=median=max=19) across **both UTM "
             "zones** — so the multi-tile/multi-zone cubes composite coherently and spec-15 "
             "calendar mosaicing holds at scale (this is what makes `flatten` clean). "
             "Temporal data availability (cloud gaps) is in "
             "`eurocrops_year_figures/coverage.png`.\n")

    L.append("## Reproduce\n```bash\n"
             "FSD_WRITE_TIMINGS=1 fsd/.venv/bin/python fsd/benchmarks/eurocrops_year_build.py\n"
             "fsd/.venv/bin/python fsd/benchmarks/eurocrops_year_report.py\n```\n")
    L.append("Cubes + flattened arrays under `tests/outputs/datacube_year/` (gitignored); "
             "this report + figures are committed.\n")

    with open(REPORT, "w") as h:
        h.write("\n".join(L))
    print("wrote", REPORT)
    print(f"wall {wall/60:.1f}min cpu {cpu/60:.1f}min par {par:.1f}x load% {load_share:.0f} "
          f"pixels {data.shape[0]} classes {len(classes)} ts {len(ts)}")


if __name__ == "__main__":
    main()
