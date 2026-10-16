# rise-eo demos — *benchmark harnesses*, not examples

> ⚠️ **Read this first (spec 41 D10 / ADR 0026).** Despite the folder name, **these are timing
> harnesses and their `.md` files are benchmark reports.** `e2e_austria.py` is 531 lines of which
> **12 touch `rise-eo`** (2.3 %), and `step_download` deliberately calls `cdse.probe_throughput` and
> `cdse.download_resume` **directly, bypassing `rise_eo.download`**, because it wants the single-stream
> baseline probe and the transfer-vs-convert split. Those are measurement concerns. **A newcomer
> reading this as a reference implementation learns the wrong API.**
>
> The folder is **not renamed**: that would churn references across point-in-time documents, which
> spec 41 D3 forbids. So the label is here instead.
>
> | if you want | go to |
> |---|---|
> | a minimal readable script to copy | `examples/` |
> | a guaranteed-to-succeed first run | `docs/tutorial.md` *(being built — spec 42 fixture)* |
> | "now point it at my own region" | `docs/howto/` *(being built)* |
> | **measured timings for a real run** | **you are in the right place** |

The fullest end-to-end write-up is **[`E2E_AUSTRIA.md`](E2E_AUSTRIA.md)** — running the full rise-eo pipeline
locally (**download → datacube → train → inference → COG/STAC/merged crop map**) on fresh real
Sentinel-2 data for an Austria ROI, with real timings, the download runner, and the "bring your own
model" bundling guide.

What's in this folder:

- **`e2e_austria.py`** — the driver, and a **reusable template**: point `--roi` / `--train`
  (+ `--id-col` / `--label-col`) at your own region and it runs unchanged, including cross-UTM-zone
  ROIs. See `E2E_AUSTRIA.md §4`.
- **`adapters.py`** — `DemoRF`, a complete worked `ModelAdapter` example (the model-developer-owned
  endpoints). See `E2E_AUSTRIA.md §6`.
- **`estimate.py`** — estimate the runtime/size of another region **without downloading it**
  (calibrated by a real run's `cost_model`). See `E2E_AUSTRIA.md §9`.
- **`figures/`** — committed output figures: `s2_grids.png`, `ndvi_timeseries.png`, `crop_map.png`
  (heavy artifacts live under `tests/outputs/demo_e2e/`, gitignored).

> **History.** This demo began as a Mode-A run on the Ethiopia `satellite_benchmark/` data
> (`e2e_ethiopia.py`, with EuroCrops labels *translated* onto Ethiopian imagery — pipeline-only,
> agronomically meaningless). It was renamed to `e2e_austria.py` and rebased onto **real downloaded
> Austria imagery with genuine EuroCrops labels**. The real bugs that full-ROI runs caught along the
> way (the spec-20 tile-merge coverage bug, the spec-26 STAC id collision, the multi-UTM-zone display
> merge) are recorded in **`E2E_AUSTRIA.md` Appendix C**.
