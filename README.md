# fsd

**Fetch satellite imagery, build datacubes, and run your own model over them — on your laptop or
across a cloud cluster, with the same code.**

Sentinel-2 L2A from the **Microsoft Planetary Computer** or **CDSE** → per-geometry **datacubes** →
flattened **training arrays** → **inference** → COGs + STAC. All I/O goes through an `fsspec` seam,
so local and `abfss://` are a config change rather than a rewrite, and scaling out is a `runner=`
argument.

**fsd does not train your model.** It hands you training arrays, takes back a bundled adapter, and
runs it at scale. That line is deliberate and permanent.

## Install

```bash
pip install "fsd[local,mpc,grid] @ git+https://github.com/nikhilsrajan/fsd@v0.1.0"
```

That set is what the quickstart below needs: `[local]` runs the default `runner="local"` pipeline,
`[mpc]` reads the Planetary Computer, `[grid]` tiles an ROI into grid cells.

| extra | brings | you need it for |
|---|---|---|
| `local` | `snakemake` | the default `runner="local"` pipeline — **and every AML node image** |
| `mpc` | `planetary-computer` | `source="mpc"` |
| `s3` | `s3fs` | `s3://` transport, including CDSE EODATA download |
| `grid` | `s2`, `s2cell` | ROI → grid cells (`roi=` on `run_inference`/`verify_adapter`) |
| `azure` | `adlfs`, `azure-identity`, `azure-keyvault-secrets` | `abfss://` storage, managed identity |
| `aml` | `azure-ai-ml` | `runner="aml"` — the driver side only |
| `notebooks`, `dev` | plotting + sklearn · ruff + pytest | notebooks · contributing |

The core install carries neither `snakemake` nor `s3fs`: one is only ever a subprocess, the other
an fsspec backend resolved by URL scheme, and together they cost 53 packages / 111 MB that an
AML-only or MPC-only user never touches. Ask for what you use — **fsd names the missing extra
rather than the missing package** if you don't.

Python ≥ 3.11. For development: clone, then follow [`CONTRIBUTING.md`](CONTRIBUTING.md#setup) Setup.

## 60 seconds

The Planetary Computer needs **no account and no credentials** — this runs as written.

```python
import datetime
import rise_eo

# 1. Imagery for an ROI and a window -> a GeoParquet tile catalog.
catalog = rise_eo.download(
    roi="my_roi.geojson",
    startdate=datetime.datetime(2018, 1, 1),
    enddate=datetime.datetime(2019, 1, 1),
    bands=["B04", "B08", "B8A", "SCL"],
    dst_folderpath="data/imagery",   # ONE shared archive for every run; a repeat run downloads nothing new
    source="mpc",   # no creds. For CDSE: source="cdse" + creds=CdseCredentials.from_env()
    max_tiles=20,   # required cost guardrail: refuse the run if the ROI matches more
)                   # than this many MGRS granules. Preflight checks it before any spend.

# 2. Labelled polygons + that catalog -> flattened training arrays.
training = rise_eo.create_training_data(
    label_polygons="my_labeled_fields.geojson",
    catalog_filepath=catalog,
    startdate=datetime.datetime(2018, 1, 1),
    enddate=datetime.datetime(2019, 1, 1),
    mosaic_days=20,
    bands=["B04", "B08", "B8A", "SCL"],
    id_col="fid",
    label_col="crop_type",
    export_folderpath="data/training",
)

arrays = training.load()   # {"data", "ids", "labels", "coords", "metadata"}
# arrays["data"] is (samples, timestamps, bands) -- train whatever you like on it.

# 3. Your trained model, bundled as a ModelAdapter, run over a region -> COGs + STAC.
result = rise_eo.run_inference(
    "my_bundle/",                    # the bundle is the first positional argument
    output_folderpath="data/predictions",
    roi="my_roi.geojson",            # ROI mode needs all five of the next arguments
    catalog_filepath=catalog,
    startdate=datetime.datetime(2018, 1, 1),
    enddate=datetime.datetime(2019, 1, 1),
    mosaic_days=20,
    bands=["B04", "B08", "B8A", "SCL"],
    merge=True,
)
```

`examples/eurocrops_rf.py` is the same thing end to end with a real `ModelAdapter` you can copy.

## Publish a model, then run it by name

`rise_eo.deploy` publishes a saved bundle into a registry and **refuses unless that bundle has been
proven to run on the image you name** — one real node, before anything is recorded. It returns a
ref that `run_inference` takes in place of a path:

```python
ref = rise_eo.deploy(
    "my_bundle/",                    # must be SAVED (rise_eo.save_bundle), not a live adapter
    name="crop-rf",
    registry="abfss://.../models",
    environment="fsd-infer-sklearn:3",
)                                    # -> "crop-rf:1"

rise_eo.run_inference(ref, registry="abfss://.../models")   # + the ROI arguments from step 3
```

`rise_eo.verify_adapter` does the same proof for an adapter against real imagery before you bundle it.
Both exist because the alternative is discovering a broken adapter thirty minutes into a dispatch.

## Running on a cluster

The same calls — add `runner="aml"` and point `storage=` at blob. No pipeline code changes; that is
the whole design. A 300-cell Austria ROI ran that way in **18.8 minutes across 32 nodes** (8/8
steps, 97 jobs, 213 MPC granules → 300 output COGs + STAC + a merged map).

Cloud work reads its settings from a user-level config file, written once:

```bash
fsd init      # -> ~/.config/fsd/config.toml
fsd config    # print the resolved config and where each value came from
```

> **Building an AML node image?** It must include the `local` extra. The in-job entrypoints run the
> same Snakemake orchestration a laptop does, so an image without it builds fine and then fails
> ~30 minutes into a dispatch. See [`docs/howto/build-the-images.md`](docs/howto/build-the-images.md).

## Where to go next

| you want | read |
|---|---|
| a guaranteed-to-succeed first run, offline, no credentials | [`docs/tutorial.md`](docs/tutorial.md) |
| your own region, real imagery, a cluster, your own model, or a map viewer | [`docs/howto/`](docs/howto/) |
| a readable script to copy | [`examples/`](examples/) — `eurocrops_rf.py`, a complete `ModelAdapter` |
| how the code is laid out, and what must stay true | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| where fsd is heading | [`ROADMAP.md`](ROADMAP.md) |
| what a term means | [`CONTEXT.md`](CONTEXT.md) |
| why something was decided | [`docs/adr/`](docs/adr/) |
| how fsd got this shape — the forks taken and dropped | [`docs/history.md`](docs/history.md) |
| a measured result about running at scale | [`docs/findings/`](docs/findings/) |
| the Azure environment variables | [`docs/reference/environment.md`](docs/reference/environment.md) |
| a reusable command | [`docs/howto/`](docs/howto/), or a script in [`scripts/`](scripts/) |
| design documents | [`specs/`](specs/) — start at `00-overview.md` |
| how to hand a real run to a person | [`runbooks/`](runbooks/) (copy `TEMPLATE.ipynb`) |
| open work, and in what order | `gh issue list`; the pinned "Order of work" issue |

Two benchmark reports record real runs — read them as **measurements, not instructions**; the
instructions are the tutorial and how-tos above. [`demos/E2E_AUSTRIA.md`](demos/E2E_AUSTRIA.md) is
the local pipeline with timings; [`demos/E2E_AUSTRIA_AML.md`](demos/E2E_AUSTRIA_AML.md) is the
300-cell cluster run.

## Known limits

Stated here rather than left to be discovered:

- **Sentinel-2 L2A only.** The ingest contract for a second source exists; the `Source` abstraction
  it implies does not, and the code still dispatches on a hardcoded pair of source names.
- **Cluster cost is dominated by warm-up, not work** — job admission was 36 % of the reference run.
- **`v0.1.0` is `0.y.z` on purpose.** The public API is not stable and is not claimed to be.

## Contributing

`pytest -q` and `ruff check src tests demos examples` must be clean. Tests are synthetic and
offline; anything needing credentials or a cluster is a run-book. Design lands as a spec first.
Details in [`ARCHITECTURE.md` §8](ARCHITECTURE.md#8-contributing).

## License

Code: MIT (`LICENSE`). Data derived from EuroCrops (`notebooks/shapefiles/AT_2018_TRAIN.geojson`,
`tests/data/tutorial/fields.geojson`) is CC BY 4.0; see each folder's `NOTICE` for the citation.
