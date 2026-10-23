"""Draw the figures of docs/findings/mpc-fast-extract-datacube-speed.md from run-book 198's export.

    .venv/bin/python benchmarks/mpc_fast_extract_figures.py tests/outputs/198_p0_export

Reads the export folder (Step 10 of runbooks/198-p0-mpc-fast-extract-parity.ipynb) and writes PNGs
to docs/findings/figures/mpc-fast-extract/, then prints the numbers the finding quotes.
"""
import json
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from rasterio.features import rasterize
from rasterio.transform import from_origin

EXPORT = Path(sys.argv[1])
FIGS = Path(__file__).resolve().parents[1] / "docs" / "findings" / "figures" / "mpc-fast-extract"
FIGS.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RISE, RISE_LIGHT = "#2a78d6", "#86b6ef"          # rise-eo: blue, its download a lighter step
MPC, MPC_LIGHT = "#eb6834", "#f5b394"            # mpc-fast-extract: orange
WITHIN, EQUAL = "#1baf7a", "#6250d6"             # agreement measures: aqua, violet
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list("oranges", ["#fde4d6", "#f5b394", "#eb6834", "#b8461b", "#6e2508"])
CRS = "EPSG:32633"
WORKLOADS = {"fields": "900 fields (training case)", "cells": "299 grid cells (inference case)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
})


def timing(name):
    """One timings/<name>.json of the export."""
    return json.loads((EXPORT / "timings" / f"{name}.json").read_text())


def save(fig, name):
    """Write one figure and close it."""
    fig.savefig(FIGS / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGS / name)


def tile_outlines():
    """MGRS tile -> (x0, y0 top, x1, y1 bottom) of its 10 m grid, in UTM 33N metres."""
    out = {}
    for tile, g in json.loads((EXPORT / "tile_grids.json").read_text()).items():
        a, _, c, _, e, f = g["transform"]
        out[tile] = (c, f, c + a * g["width"], f + e * g["height"])
    return out


def draw_tiles(ax, label=True, filled=True):
    """MGRS tile footprints: what rise-eo downloads whole."""
    for tile, (x0, y0, x1, y1) in tile_outlines().items():
        ax.add_patch(Rectangle((x0, y1), x1 - x0, y0 - y1, facecolor=RISE_LIGHT if filled else "none",
                               alpha=0.25 if filled else 1, edgecolor=RISE, lw=1.2))
        if label:
            ax.text(x0 + 2500, y0 - 2500, tile, color=RISE, fontsize=9, va="top", fontweight="bold")


def label_tiles(ax):
    """Tile names inside the visible part of each tile (call after the limits are set)."""
    (vx0, vx1), (vy0, vy1) = ax.get_xlim(), ax.get_ylim()
    for tile, (x0, y0, x1, y1) in tile_outlines().items():
        ix0, ix1, iy0, iy1 = max(x0, vx0), min(x1, vx1), max(y1, vy0), min(y0, vy1)
        if ix1 - ix0 > 4000 and iy1 - iy0 > 4000:
            ax.text(ix0 + 1200, iy1 - 1200, tile, color=RISE, fontsize=9, va="top",
                    fontweight="bold")


def km_axes(ax, title):
    """Axes in km, equal aspect, a plain title."""
    ax.set_aspect("equal")
    ax.set_title(title, loc="left")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}")
    ax.set_xlabel("easting, km (UTM 33N)")
    ax.set_ylabel("northing, km")


# --- numbers -----------------------------------------------------------------------------
download = timing("rise_download")
rows = {}
for w in WORKLOADS:
    fair, run, conv = timing(f"rise_build_fair_{w}"), timing(f"mpc_run_{w}"), timing(f"mpc_convert_{w}")
    phases = pd.read_csv(EXPORT / f"rise_phases_{w}.csv")
    rows[w] = dict(
        download=download["seconds"], build2=timing(f"rise_build_{w}")["seconds"], build4=fair["seconds"],
        run=run["seconds"], convert=conv["seconds"], gb_read=float(run["engine"]["gb"]),
        cubes=fair["cubes"], cores=fair["cores"], phases=phases,
        pixel_windows=float((phases["T"] * phases["H"] * phases["W"]).sum()),
        engine=run["engine"])

# --- figure 1: where the wall time goes ---------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.6))
bars = []
for w in WORKLOADS:
    r = rows[w]
    bars += [(f"{w}: rise-eo", [("download", r["download"], RISE_LIGHT),
                                ("build, 4 at a time", r["build4"], RISE)]),
             (f"{w}: mpc-fast-extract", [("engine run (streams blocks)", r["run"], MPC),
                                         ("cut Zarr into cubes", r["convert"], MPC_LIGHT)])]
for i, (label, segs) in enumerate(bars[::-1]):
    left = 0
    for name, sec, color in segs:
        ax.barh(i, sec / 60, left=left / 60, color=color, height=0.62, edgecolor=SURFACE, linewidth=2)
        if sec > 300:
            ax.text((left + sec / 2) / 60, i, name, ha="center", va="center", fontsize=8.5,
                    color=INK if color in (RISE_LIGHT, MPC_LIGHT) else "white")
        left += sec
    ax.text(left / 60 + 0.6, i, f"{left / 60:.1f} min", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(bars)), [b[0] for b in bars[::-1]])
ax.set_xlabel("wall time, minutes (one 2-core / 4-thread VM)")
ax.set_title("Creating the same datacubes: rise-eo vs mpc-fast-extract", loc="left", pad=22)
ax.text(0, 1.03, "mpc-fast-extract bars: dark = engine run (streams blocks from MPC), "
        "light = cutting its Zarr into rise-eo cubes", transform=ax.transAxes, fontsize=8.5, color=INK2)
for w, y in (("fields", 3), ("cells", 1)):
    r = rows[w]
    ratio = (r["download"] + r["build4"]) / (r["run"] + r["convert"])
    ax.text(52, y - 0.5, f"{w}: {ratio:.0f}× faster", fontsize=10,
            fontweight="bold", color=MPC, va="center", ha="right")
ax.set_xlim(0, 56)
save(fig, "fig1_wall_time.png")

plan = timing("plan")
n_granules = len(set(plan["fields"]["granules"]) | set(plan["cells"]["granules"]))

# --- figure 2: what each tool reads ----------------------------------------------------------
tiles = tile_outlines()
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, w in zip(axes, WORKLOADS):
    gdf = gpd.read_file(EXPORT / f"{w}.geojson").to_crs(CRS)
    draw_tiles(ax)
    blocks = set()
    for tile, (x0, y0, x1, y1) in tiles.items():
        n = math.ceil((x1 - x0) / 5120)
        for geom in gdf.geometry:
            gx0, gy0, gx1, gy1 = geom.bounds
            if gx1 < x0 or gx0 > x1 or gy1 < y1 or gy0 > y0:
                continue
            for c in range(max(0, int((gx0 - x0) // 5120)), min(n, int((gx1 - x0) // 5120) + 1)):
                for rr in range(max(0, int((y0 - gy1) // 5120)), min(n, int((y0 - gy0) // 5120) + 1)):
                    blocks.add((x0 + c * 5120, y0 - (rr + 1) * 5120))
    for bx, by in blocks:
        ax.add_patch(Rectangle((bx, by), 5120, 5120, facecolor=MPC, alpha=0.35, edgecolor="none"))
    gdf.boundary.plot(ax=ax, color=INK, linewidth=0.4)
    xs = [v for t in tiles.values() for v in (t[0], t[2])]
    ys = [v for t in tiles.values() for v in (t[1], t[3])]
    ax.set_xlim(min(xs) - 3000, max(xs) + 3000)
    ax.set_ylim(min(ys) - 3000, max(ys) + 3000)
    r = rows[w]
    km_axes(ax, WORKLOADS[w])
    ax.text(0.02, 0.02, f"rise-eo: {n_granules} whole granules over the 4 tiles = {download['archive_gb']:.0f} GB\n"
            f"mpc-fast-extract: {len(blocks)} distinct 512-px block cells (orange) "
            f"→ {r['gb_read']:.1f} GB read", transform=ax.transAxes, fontsize=8.5, color=INK,
            bbox=dict(facecolor=SURFACE, edgecolor=GRID))
fig.suptitle("What each tool reads: rise-eo downloads every tile whole (blue); mpc-fast-extract reads "
             "only the 512-px blocks the geometries touch (orange)", x=0.02, ha="left", fontsize=10.5)
save(fig, "fig2_what_is_read.png")
print("read ratio:", {w: round(download["archive_gb"] / rows[w]["gb_read"], 1) for w in WORKLOADS})

# --- figure 4: rise-eo builds the overlap of neighbouring cells more than once -----------------
cells = gpd.read_file(EXPORT / "cells.geojson").to_crs(CRS)
x0, y0, x1, y1 = cells.total_bounds
res = 50
shape = (math.ceil((y1 - y0) / res), math.ceil((x1 - x0) / res))
transform = from_origin(x0, y1, res, res)
cover = rasterize(((g.envelope, 1) for g in cells.geometry), out_shape=shape, transform=transform,
                  merge_alg=rasterio.enums.MergeAlg.add, dtype="uint8")
fig, ax = plt.subplots(figsize=(7, 6.2))
im = ax.imshow(np.ma.masked_equal(cover, 0), cmap=BLUES, vmin=1, vmax=4,
               extent=(x0, x0 + shape[1] * res, y1 - shape[0] * res, y1), interpolation="nearest")
draw_tiles(ax, label=False, filled=False)
ax.set_xlim(x0, x0 + shape[1] * res)
ax.set_ylim(y1 - shape[0] * res, y1)
label_tiles(ax)
km_axes(ax, "rise-eo builds each cell's box on its own:\noverlaps are read and built 2–4 times")
cb = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[1, 2, 3, 4])
cb.set_label("rise-eo cubes covering the pixel")
covered = cover[cover > 0]
print("cells: pixel-builds per distinct pixel:", round(covered.sum() / covered.size, 3))
save(fig, "fig4_rise_eo_rebuilds_overlaps.png")

# --- figure 3: where rise-eo's time per cube goes ----------------------------------------------
groups = {"reading imagery (load_images)": ["phase_load_images"],
          "grid alignment (reference_profile, resample)": ["phase_reference_profile", "phase_resample"],
          "mask + median, incl. numba compile (ops)": ["phase_ops"],
          "other builder steps": ["phase_missing_check", "phase_dst_crs", "phase_stack", "phase_save"]}
colors = ["#0d366b", "#86b6ef", "#2a78d6", "#c3c2b7", "#e4e3df"]
fig, ax = plt.subplots(figsize=(9, 2.8))
for i, w in enumerate(WORKLOADS):
    r = rows[w]
    per_slot = r["build4"] * r["cores"] / r["cubes"]           # wall seconds per cube, per build slot
    parts = [(name, r["phases"][cols].sum(axis=1).mean()) for name, cols in groups.items()]
    parts.append(("outside the builder (process start, imports, scheduling)",
                  per_slot - sum(v for _, v in parts)))
    left = 0
    for (name, v), color in zip(parts, colors):
        ax.barh(i, v, left=left, color=color, height=0.6, edgecolor=SURFACE, linewidth=2,
                label=name if i == 0 else None)
        if v > 0.8:
            ax.text(left + v / 2, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if color in ("#0d366b", "#2a78d6") else INK)
        left += v
    ax.text(left + 0.2, i, f"{left:.1f} s per cube", va="center", fontsize=9)
    print(w, "rise-eo per cube:", {n: round(v, 2) for n, v in parts}, "total", round(left, 2))
ax.set_yticks(range(len(WORKLOADS)), [f"{w} ({rows[w]['cubes']} cubes)" for w in WORKLOADS])
ax.set_xlabel("seconds per cube (wall time × 4 build slots ÷ cubes)")
ax.set_xlim(0, 24)
ax.set_title("Where rise-eo's time per cube goes (4 builds at a time)", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=8.5)
save(fig, "fig3_rise_eo_time_per_cube.png")

# --- figure 5: agreement --------------------------------------------------------------------
offset = json.loads((EXPORT / "compare" / "offset_2023.json").read_text())
cats, within, equal = [], [], []
for w in WORKLOADS:
    n = json.loads((EXPORT / "compare" / f"{w}.json").read_text())
    for g, label in (("single_tile", "1 tile"), ("multi_tile", "2+ tiles")):
        cats.append(f"2018 {w}\n{label} ({n[g]['cubes']})")
        within.append(n[g]["within_1"] / n[g]["both_valid"])
        equal.append(n[g]["equal"] / n[g]["both_valid"])
for band, n in offset["bands"].items():
    cats.append(f"2023 {band}\n1 tile, offset")
    within.append(n["within_1"] / n["both_valid"])
    equal.append(n["equal"] / n["both_valid"])
fig, ax = plt.subplots(figsize=(11, 4))
x = np.arange(len(cats))
for off, vals, color, name in ((-0.2, within, WITHIN, "within 1 DN"), (0.2, equal, EQUAL, "exactly equal")):
    ax.bar(x + off, np.array(vals) * 100, width=0.38, color=color, label=name, edgecolor=SURFACE)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v * 100 + 1.5, f"{v * 100:.1f}", ha="center", fontsize=7.5, color=INK)
ax.set_xticks(x, cats, fontsize=8.5)
ax.set_ylabel("% of pixels both tools have a value for")
ax.set_ylim(0, 108)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
ax.set_title("Pixel agreement: single-tile cubes match; multi-tile cubes differ by the overlap "
             "tie-break", loc="left")
save(fig, "fig5_agreement.png")

# --- figure 6: where the cubes disagree, per pixel -------------------------------------------
def block_max(a, k=4):
    """The maximum of each k × k block, so thin overlap strips survive the downsampling."""
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).max(axis=(1, 3))


fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
with rasterio.open(EXPORT / "cells_disagreement.tif") as src:
    off, valid, b = block_max(src.read(1)), block_max(src.read(4)), src.bounds
ax = axes[0]
im = ax.imshow(np.ma.masked_where(valid == 0, off), cmap=ORANGES, vmin=0, vmax=20,
               extent=(b.left, b.right, b.bottom, b.top), interpolation="nearest")
draw_tiles(ax, label=False, filled=False)
ax.set_xlim(b.left, b.right)
ax.set_ylim(b.bottom, b.top)
label_tiles(ax)
km_axes(ax, "Cells: window × band slots off by more than 1 DN, per pixel")
fig.colorbar(im, ax=ax, shrink=0.7).set_label("slots off (of 10 windows × 2 bands)")
ax = axes[1]
fields = gpd.read_file(EXPORT / "fields.geojson").to_crs(CRS)
per = pd.read_csv(EXPORT / "per_geometry_fields.csv", dtype={"id": str})
fields = fields.merge(per, left_on="field_id", right_on="id")
fields["share_off"] = fields["off"] / fields["both_valid"].clip(lower=1)
draw_tiles(ax, label=False, filled=False)
pts = fields.geometry.centroid
sc = ax.scatter(pts.x, pts.y, c=fields["share_off"] * 100, cmap=ORANGES, vmin=0, vmax=100, s=14,
                edgecolors=INK2, linewidths=0.3)
ax.set_xlim(axes[0].get_xlim())
ax.set_ylim(axes[0].get_ylim())
label_tiles(ax)
km_axes(ax, "Fields: share of pixels off by more than 1 DN, per field")
fig.colorbar(sc, ax=ax, shrink=0.7).set_label("% of the field's pixels")
fig.suptitle("Disagreement sits in the MGRS tile overlap strips (tile outlines in blue)", x=0.02,
             ha="left", fontsize=10.5)
save(fig, "fig6_disagreement_maps.png")

# --- figure 7: how large the differences are -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for ax, w in zip(axes, WORKLOADS):
    h = pd.read_csv(EXPORT / f"hist_{w}.csv").set_index("diff").loc[-100:100]
    for g, color, label in (("single_tile", WITHIN, "1 tile"), ("multi_tile", EQUAL, "2+ tiles")):
        share = h[g] / h[g].sum() * 100
        ax.plot(share.index, share.values, color=color, lw=2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("rise-eo minus mpc-fast-extract, DN (reflectance × 10,000)")
    ax.set_title(WORKLOADS[w], loc="left")
    ax.legend(frameon=False)
axes[0].set_ylabel("% of pixels (log scale)")
save(fig, "fig7_difference_histogram.png")

# --- numbers for the text ---------------------------------------------------------------------
for w in WORKLOADS:
    r = rows[w]
    mpc_total, rise_total = r["run"] + r["convert"], r["download"] + r["build4"]
    mp = r["pixel_windows"] / 1e6
    print(w, {"rise_s": round(rise_total, 1), "mpc_s": round(mpc_total, 1),
              "ratio_with_download": round(rise_total / mpc_total, 1),
              "ratio_build_only": round(r["build4"] / mpc_total, 1),
              "rise_s_per_Mpx": round(rise_total / mp, 2), "rise_build_s_per_Mpx": round(r["build4"] / mp, 2),
              "mpc_s_per_Mpx": round(mpc_total / mp, 3), "Mpx_windows": round(mp, 1),
              "cost_rise_usd": round(rise_total / 3600 * 0.35, 3), "cost_mpc_usd": round(mpc_total / 3600 * 0.35, 3),
              "gain_2_to_4_builds": round(r["build2"] / r["build4"], 2)})
    wb = pd.read_csv(EXPORT / f"window_band_{w}.csv")
    wb = wb.groupby(["group", "window"])[["both_valid", "off"]].sum()
    print(w, "off share by window:", (wb["off"] / wb["both_valid"]).round(3).unstack("window").to_dict("index"))
