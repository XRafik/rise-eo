# How to: run the same pipeline at scale on Azure ML

> **Last verified:** 2026-07-31 @ `df98463` (spec 41 D5 tier 2 — "dated"). Re-verify after any
> change to the `runner="aml"` path in `rise_eo.download`/`rise_eo.create_training_data`/`rise_eo.run_inference`,
> or after rebuilding the AML Environments.

Same calls as [`your-own-region.md`](your-own-region.md), fanned out onto an Azure ML cluster
instead of run in-process. **The same 300-cell Austria ROI ran this way in 18.8 minutes across 32
nodes** (2026-07-29 cluster demo, `demos/e2e_austria_aml.py`, run `20260729T132222Z`) — 8/8 steps,
97 jobs, 213 MPC granules, 300 output COGs + STAC + a merged map.

## This is not rise-eo's to provision

Getting to a working Azure ML workspace, a compute cluster, a UAMI with blob RBAC, and two built
Environments is a **platform-admin action**, not something this repo automates (spec 41 D2). **If
you don't have these yet, file a ticket with your platform admin — that is an acceptable first step,
not a dead end.** No Azure onboarding document exists in this public repo on purpose: the concrete
values are handed to you privately (`AZURE_INFRA_PRIVATE.md`, never public), not published here.

## Prerequisites

- VPN access and `az login` into the tenant hosting the workspace.
- An AML workspace, a compute cluster, and a UAMI with **Storage Blob Data Contributor** on the
  storage account — ask your platform admin for these by name.
- **Two AML Environments already built**: a general-purpose one (download + build + flatten) and
  an inference one (carries your model bundle). Building one is a 10–20 minute ACR build — do this
  ahead of a long unattended run, not during it. See [`build-the-images.md`](build-the-images.md) for how
  they're built.
- The five required values `rise-eo init` asks for (plus the two optional registry keys), or the
  `AZ_*` environment variables that override them. **Your storage root is not among them** — it
  is a per-run argument you pass, not config (spec 55 D1) — see
  [`docs/reference/environment.md`](../reference/environment.md). Concrete values for the `rise`
  platform live in `AZURE_INFRA_PRIVATE.md` at the workspace root, handed to you by your platform
  admin — never in this repo.
- Install with the extras the cluster path actually needs:
  ```bash
  pip install -e ".[dev,azure,aml,mpc,grid,model-example]"
  ```
  (`azure` for blob I/O, `aml` for dispatching jobs, `mpc` for anonymous MPC discovery, `grid` for
  ROI tiling, `model-example` for the training-side stack — rise-eo core stays lean and installs none
  of this by default, because rise-eo never trains a model.)

## The same three calls, one keyword different

Your blob working root is an **argument**, not config: `rise-eo init` never writes it and rise-eo never
reads it (spec 55 D1). Pick one per run and pass it — the `AZ_ROOT` export below is just this
example's way of not hard-coding a private URL into a public repo.

```python
import os

import rise_eo

# An ARGUMENT dict, not a config file. `cluster` and `identity_client_id` happen to have
# config keys you could read with `rise_eo.config.load()`; `root` deliberately does not (spec 55
# D1) -- it is a per-run destination you choose, so it is only ever passed.
runner_kwargs = {
    "cluster": os.environ["AZ_CLUSTER"],
    "environment": os.environ["AZ_ENV_NAME"],
    "root": os.environ["AZ_ROOT"],                       # per-run blob root -- abfss://..., yours to pass
    "identity_client_id": os.environ["AZ_UAMI_CLIENT_ID"],
}

catalog = rise_eo.download(
    roi="your_roi.geojson", startdate=..., enddate=..., bands=[...],
    dst_folderpath=os.environ["AZ_ROOT"] + "/imagery",   # ONE shared archive -- re-used by every run
    source="mpc",                # anonymous -- no secret to provision on the node (TODO #49)
    max_tiles=500,
    runner="aml", runner_kwargs=runner_kwargs,
)

training = rise_eo.create_training_data(
    label_polygons="your_fields.geojson", catalog_filepath=catalog,
    startdate=..., enddate=..., mosaic_days=20, bands=[...],
    id_col="fid", label_col="crop",
    export_folderpath="data/training",     # LOCAL landing target -- always local, even on AML
    runner="aml", runner_kwargs=runner_kwargs,
)

result = rise_eo.run_inference(
    "my_bundle/", output_folderpath="data/predictions",
    roi="your_roi.geojson", catalog_filepath=catalog,
    startdate=..., enddate=..., mosaic_days=20, bands=[...],
    merge="reproject",
    runner="aml", storage="azure", runner_kwargs=runner_kwargs,
)
```

**No pipeline code changed** — that's the design (`rise_eo.storage` as the seam, the runner as a
CLI-unit-of-work dispatcher). Three things are different from the local path, not more:

1. **`runner="aml"` + `runner_kwargs`** replaces `runner="local"` (the default) on every call.
2. **`export_folderpath` stays local** even on the cluster — it's always where the compact result
   lands on your machine; `runner_kwargs["root"]` is the blob **working** root (catalog, per-cell
   datacubes, `input.csv`) that the cluster nodes actually read/write.
3. **`run_inference`'s ROI mode needs `storage="azure"`** (or an `abfss://` root) in addition to
   `runner="aml"` — this is the one call where the storage seam is spelled out explicitly, because
   ROI-mode inference builds datacubes on the fly rather than reading pre-built ones.

## Why `source="mpc"` on the cluster

MPC is anonymous — no secret has to be staged on a compute node for a download job to read. rise-eo's
`run_aml_download` actively **refuses** CDSE credentials for an `runner="aml"` dispatch rather than
staging a secret on blob for something that never reads it. If your pipeline needs CDSE specifically,
that's a Key Vault–backed path (`runner_kwargs`'s `vault_url=`/`secret_name=`) — see
`workflows.runners.run_aml_download`'s docstring, not this page.

## Before a long unattended run

- **Rebuild both Environments if rise-eo changed since you last built them, and bump the version.** An
  older image runs correctly but silently omits in-job timing stamps — this voided a complete
  25-minute, 97-job run once (2026-07-29) before anyone noticed, because the pipeline's *science*
  was unaffected, only its measurement.
- **Confirm network rules, not just RBAC, before assuming an auth failure is an auth failure.**
  Project storage is deny-by-default firewalled: a VM/compute instance outside the approved subnets
  gets a 403 on every blob call regardless of identity, and that looks exactly like a credentials
  bug until you check which subnet you're on.

## Where to go next

- [`bundle-your-model.md`](bundle-your-model.md) — the bundle `run_inference` needs as its first
  argument above.
- [`serve-xyz.md`](serve-xyz.md) — what to do with 300 output COGs once the run finishes.
