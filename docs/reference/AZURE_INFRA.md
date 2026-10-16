# AZURE_INFRA — the `rise` deployment target for rise-eo scale-up

> **What this is.** A standing reference (not a spec) capturing how the RAAPID Azure
> infrastructure — repo `raapid-infra/`, built by a colleague in Terraform — is set up,
> and specifically what the **`rise`** project gives us to run `rise-eo` at scale on **Azure
> Batch**. Read this before/while writing the Azure Batch spec (the future spec 10).
> **`raapid-infra/` is READ-ONLY for us** (a hard constraint, same status as
> `fetch_satdata/` / `rsutils/` / `cdseutils/`) — at least for now. We read it to
> understand what `rise` provisions and we *consume* those resources; we never edit it. Any
> infra change (max_nodes, max_tasks_per_node, warm-cache image, adding `vms`, etc.) is a
> `main/terraform.tfvars` edit + gated `deploy.sh` run by a **platform admin** (a colleague;
> possibly the user later). We only *propose* infra changes. When we start specing, this is
> the file to re-read for the ground truth about identities, network, and gotchas.
>
> **Redaction note.** `raapid-infra/` is a **private** repo; this file is in the **public**
> `rise-eo` repo. So concrete resource **names/endpoints**, directory/group **object IDs**, the
> **VNet CIDR**, budget figures, and alert emails are **deliberately omitted** here — they
> live only in `raapid-infra/main/terraform.tfvars` and the Azure Portal. This doc keeps the
> *architecture and seam mapping* (which is the reusable value) and refers to resources by
> **role** with placeholder short-names (`st<proj>`, `kv<proj>`, `ba<proj>`, `acr<proj>`, …).
> Look up the concrete names/URLs in the tfvars or via `az` when actually wiring things.
>
> Source of truth in `raapid-infra/`: `main/terraform.tfvars` (the `rise` entry),
> `modules/project-capabilities/azure-batch/`, `modules/project-core/`,
> `docs/high-level-architecture.md`, `docs/common-ops.md`, `docs/user-guide.md`.

---

## 1. The big picture (how RAAPID/Azure is organized)

RAAPID is a shared Azure platform managed as **Terraform IaC** (`azurerm ~> 4.17`) in
`raapid-infra/`. One Azure subscription, one shared VNet, many **projects**. Projects are
*pure data*: each is an entry in the `projects` map in `main/terraform.tfvars`. Adding
capabilities to a project = editing that map + a gated `terraform apply` (`main/deploy.sh`)
run by a **platform admin**. We (rise-eo) are consumers, not admins — we don't run Terraform; we
ask for config changes and then use the resources.

Each project gets, from the **`project-core`** module (always):
- a **resource group**, a **storage account** (ADLS Gen2), a **Key Vault**, a **budget**,
  a **user-assigned managed identity** ("compute identity"), and dedicated **subnets** in
  the shared VNet, all network-isolated by an NSG.

Then **opt-in capabilities** (per project's `capabilities` list) layer on top:
`aml`, `azure-batch`, `container-apps`, `postgresql`, `functions`, `blob-storage`, `vms`.
An **ACR** (container registry) is auto-added whenever a project has `aml`,
`container-apps`, or `azure-batch`.

**Naming convention:** resource names follow the platform's
`{resource-type}-{project}-{env}-{location}` pattern (storage/registry/batch names strip
hyphens and truncate). The **concrete names are omitted here** (see the redaction note) —
resolve them from tfvars or `az` when wiring. `rise` has no `legacy_name_suffix`, so there's
no `-prod` env infix in its names.

---

## 2. The `rise` project — exactly what we get

From `main/terraform.tfvars` (the `rise` block; identifiers/budget/group-IDs redacted):

```hcl
"rise" = {
  subnet_index = <n>
  budget_amount       = <redacted>            # USD/month, with alert emails
  team_group_object_id         = "<redacted>" # Entra group object IDs — see tfvars
  project_lead_group_object_id = "<redacted>"
  capabilities = ["aml", "azure-batch"]
  capability_config = {
    aml = {
      max_compute_nodes = 2
      compute_vm_size   = "Standard_E64ds_v4"  # 64 vCPU, memory-optimized
    }
    azure-batch = {
      pool_id       = "<proj>-pool"
      vm_size       = "Standard_D64ds_v6"       # 64 vCPU, general-purpose, local NVMe temp disk
      node_priority = "dedicated"               # on-demand (not evictable spot)
      max_nodes     = 2                          # capped by compute-subnet IP budget at plan time
    }
  }
  allowed_datasets = []                          # NO access to shared Planet/Maxar datasets
}
```

### 2.1 Resources this provisions (by role — concrete names in tfvars)

`location = <loc>`, no env suffix. Referred to by role + placeholder short-name:

| Kind | Placeholder | Notes |
|---|---|---|
| Resource group | `rg<proj>` | `prevent_destroy` + delete-lock |
| **Project storage** | `st<proj>` | **ADLS Gen2** (HNS on), LRS, `data` container, firewalled |
| Key Vault | `kv<proj>` | RBAC-auth, purge-protected — **where rise-eo secrets go** |
| **Compute identity (UAMI)** | `id<proj>-compute` | the identity everything runs as |
| Compute subnet | `snet<proj>-compute` | `/24` in the shared VNet, service endpoints |
| NSG | `nsg<proj>` | deny-by-default inbound; Batch/AML mgmt ports open |
| ACR (auto) | `acr<proj>` | container registry for task images |
| **Batch account** | `ba<proj>` | BatchService mode, auto-storage = project storage |
| **Batch pool** | `<proj>-pool` | D64ds_v6, dedicated, autoscale 0→2, container-enabled |
| AML workspace | `mlw<proj>` | + workspace storage `stmlw<proj>` |
| AML compute cluster | `cluster-<proj>` | E64ds_v4, autoscale 0→2 |

### 2.2 The compute identity is the linchpin

`id<proj>-compute` (a **user-assigned managed identity**) is pre-granted, by
`project-core` + the capability modules:
- **Storage Blob Data Contributor** on `st<proj>` (read/write all project blobs)
- **Key Vault Secrets User** on `kv<proj>` (read secrets)
- **AcrPull + AcrPush** on `acr<proj>` (pull/push container images)
- **Contributor** on the Batch account `ba<proj>` (submit jobs/tasks via MSI)

The Batch **account** and every **pool node** run *as* this identity. So code on a node
that uses `DefaultAzureCredential()` (Azure SDK / `azure-identity`) automatically gets a
token for blob, Key Vault, and Batch **with no secrets baked into the image**. This is the
crux of the whole auth story.

### 2.3 Network & security posture (matters for rise-eo I/O)

- Project storage & Key Vault are **deny-by-default firewalled**: reachable only from the
  VPN IP ranges *and* the project's own subnets (via VNet **service endpoints**). Pool
  nodes live in the project compute subnet, so they're allowed. A laptop must be on the
  **VPN + `az login`** to touch storage directly.
- **Account keys are DISABLED** on project storage (`shared_access_key_enabled = false`).
  → **No connection strings / account keys exist.** All access is **Entra ID (MSI / user
  token) or user-delegation SAS.** This directly shapes how the fsspec storage seam must
  authenticate (see §4.1).
- Outbound: nodes can reach the Internet (`AllowInternetOutbound`) — so **CDSE downloads
  work** from a pool node — and Azure services.
- NSG opens the Batch node-management ports (`BatchNodeManagement` service tag) and AML
  management ports.

---

## 3. Where rise-eo fits: the two compute paths

`rise` has **both** `aml` and `azure-batch`.

> ### 3.1 ⭐ FORK RESOLVED — the scale runner targets **AML**, not Batch (2026-07-21)
>
> Measured, not assumed, by run-book `36-runner-fork-probe` (tag `docs-archive-2026`) (green 2026-07-21) against a
> **decision rule registered before the numbers were seen**:
>
> | Fact | Value | Consequence |
> |---|---|---|
> | Batch account `dedicatedCoreQuota` | **6** | The pool VM is a **64-core** SKU ⇒ Batch **cannot allocate even one node.** `lowPriorityCoreQuota` is 6 too, so there is no spot escape hatch. |
> | AML cluster `<proj>-d16` | **32 nodes × 16 vCPU = 512 cores**, `provisioningState: Succeeded` | Runs **today**. |
> | AML quota, that VM family | **6400** cluster-dedicated vCPUs (64 in use) | 512 is nowhere near the ceiling. |
> | AML cluster identity | **UserAssigned = the project compute identity** | The *same* UAMI P1 proved can reach blob (spec 31 / runbook 31). No new auth path, no new RBAC ask. |
>
> So §6.1's "Batch quota starts tiny" gotcha is **confirmed real, not hypothetical** — Batch is
> blocked behind a portal quota request (external, multi-day) before it can run at all, while AML
> is unblocked, ~4× larger, and inherits P1's proven identity. Note the AML *cluster* quota is what
> governs `AmlCompute`; the subscription-level VM-family quota is much smaller and is **not** the
> binding constraint (the probe shows a 64-core compute instance counted against the AML cluster
> quota, not the VM quota) — worth re-checking on the first real scale-up.
>
> **This does not change the shape of the design.** The unit of work is unchanged, the storage seam
> is unchanged, and the runner seam is the same seam — only which dispatcher plugs into it. Design
> lives in `specs/36-scale-runner.md`.
>
> **Batch is DROPPED, not deferred** (user, 2026-07-21 — strict YAGNI). We are **not** filing the
> quota request: AML ships the demo today, and the runner seam is already evidenced by two live
> backends (local Snakemake ↔ AML). Batch remains the more portable model on paper — a generic task
> queue that AWS Batch and k8s Jobs also fit — so if a third backend is ever wanted, this section is
> the record of why it wasn't built: **quota, not architecture.** Everything else about the pool was
> ready (see §8).

Historically (and in the roadmap's original wording) **Azure Batch was the intended target** for the
**download → datacube** fan-out. AML was framed as training/notebooks (out of rise-eo core scope) plus a
driver host; per §3.1 it is now also the scale backend.

**Azure Batch mental model** (maps 1:1 onto rise-eo's existing design):
- A **pool** = a set of identical VM nodes (the project pool, D64ds_v6) that **autoscales
  0→N** on pending-task pressure and **back to 0 when idle** (so idle cost ≈ 0).
- A **job** = a collection of **tasks**; each **task** = one command line run on a node
  (in our case, inside a container).
- A **driver** submits the job + tasks to the Batch account (via MSI), then Batch schedules
  tasks onto nodes as they scale up.

This is exactly the rise-eo **runner seam**: today the "build one datacube" unit-of-work is a
CLI (`rise_eo.workflows.task`) dispatched locally by **Snakemake**; on Azure the **same CLI**
becomes a **Batch task**, and a Batch runner replaces Snakemake as the dispatcher. Snakemake
= the *local* runner; Azure Batch = the *scale* runner. The unit-of-work doesn't change.

```
                 rise-eo today (local)                    rise-eo on rise (Azure Batch)
  unit of work   rise_eo.workflows.task CLI               SAME CLI, inside a container
  dispatcher     Snakemake (local cores)              Azure Batch job/tasks → project pool
  parallelism    N processes on 1 machine             N tasks across autoscaled nodes
  file I/O       fsspec → local disk                  fsspec → adlfs (project storage)
  secrets        secrets/ dir                          Key Vault (via MSI)
```

---

## 4. How rise-eo's seams land on `rise`

### 4.1 Storage seam (`rise_eo.storage`, fsspec) → ADLS Gen2 via MSI

- Project storage `st<proj>` is **ADLS Gen2**. fsspec talks to it through **`adlfs`**
  (`AzureBlobFileSystem` / `abfs[s]://`). URIs look like
  `abfss://data@<storage-account>.dfs.core.windows.net/<path>` (container `data`).
- **Auth = `DefaultAzureCredential`**, *not* account keys (keys are disabled). On a pool
  node this resolves to the compute identity's MSI automatically; on a laptop it's `az
  login` + VPN. This is a config change to `rise_eo.storage`, **not** new code paths — exactly
  the "storage becomes config, not code" goal. We'll need to confirm `adlfs` +
  `DefaultAzureCredential` works with our `storage.transfer` / `size` / `load_npy` calls.
- The **generic S3 transport** in rise-eo (CDSE download via `s3fs`, any `endpoint_url`) is
  untouched — CDSE stays S3; only the *destination* of a tile download becomes an `abfss://`
  URL. `storage.transfer(cdse_s3_url, abfss_dst)` is the one-line tile copy.

**Open item:** rise-eo's documented raster-read exception uses **rasterio/GDAL VSI**, not
fsspec. On Azure that means GDAL `/vsiadls/` (or `/vsiaz/`) + env-based credentials, which
is a *separate* auth path from adlfs. We must make GDAL read blobs under MSI too (likely
`AZURE_STORAGE_ACCESS_TOKEN` / `AZURE_NO_SIGN_REQUEST` config, or user-delegation SAS). This
is a real design point for the spec.

### 4.2 Runner seam → Azure Batch runner

- Add an **Azure Batch runner** alongside the Snakemake runner in `rise_eo.workflows`. It
  builds the same `input.csv` of units, then for each unit creates a Batch **task** whose
  command line is the `rise_eo.workflows.task` CLI (same args we pass today).
- Submission uses the **`azure-batch` SDK** authenticated via `DefaultAzureCredential`
  against the Batch account URL (a TF output). The compute identity has Contributor on the
  account.
- Tasks are **containerized** (the pool is `DockerCompatible`): each task runs our rise-eo
  image pulled from the project ACR.

### 4.3 Container image → ACR

- Build a rise-eo Docker image (the rise-eo package + its deps + GDAL/rasterio) and **push to the
  project ACR `acr<proj>`** (compute identity has AcrPush; a human/CI does the push).
- The pool pulls it via the compute identity (AcrPull). Optionally list it in the pool's
  `container_image_names` for a **warm cache** on scale-up (avoids first-task pull latency)
  — that's a tfvars change on the pool.

### 4.4 Secrets → Key Vault

- CDSE S3 credentials (today in `secrets/`) move to **`kv<proj>`**; rise-eo reads them at
  runtime via MSI (`azure-keyvault-secrets` + `DefaultAzureCredential`). No secrets in the
  image or in blob.

---

## 5. End-to-end: rise-eo on `rise` (the target flow)

```
  [driver]  (laptop on VPN, or AML compute, or a small VM)
     |  1. reads geometries + builds the unit list (input.csv) — as today
     |  2. az/MSI: submit Batch job to the project Batch account, one task per unit
     v
  [Batch pool]  autoscales 0 -> up to 2 x D64ds_v6, pulls rise-eo image from the project ACR
     |  each task = `python -m rise_eo.workflows.task ...` in the rise-eo container, running as the compute identity
     |    - reads CDSE creds from Key Vault (MSI)
     |    - downloads S2 tiles from CDSE S3 -> writes to abfss://data@<storage-account>... (fsspec/adlfs)
     |    - builds datacube (rasterio reads blobs via GDAL VSI), writes datacube.npy + metadata to blob
     v
  [outputs]  in project storage; flatten runs as another job/step; pool scales back to 0 (≈ $0 idle)
```

---

## 6. Gotchas & constraints (things that will bite us)

1. **Batch quota starts tiny — ✅ CONFIRMED at exactly 6 dedicated vCPUs** (measured 2026-07-21,
   run-book `36-runner-fork-probe` (tag `docs-archive-2026`); low-priority is 6 as well). A fresh Batch account can't even
   run one 64-vCPU node, and this account is still fresh. Before any real Batch run we must file a
   **per-Batch-account quota increase** in the Portal for the D-family in the region (need ≥128 vCPU
   for 2×D64). Quota is per-account — it does not carry over from other projects.
   (`common-ops.md` §quota.) **This is what sent the scale runner to AML instead — see §3.1.**
2. **`max_nodes = 2` today** — deliberately small, and validated at plan time against the
   compute subnet's IP budget. Scaling out = raise `max_nodes` in tfvars (needs subnet IP
   headroom **and** quota). A platform-admin `terraform apply`, not something we do.
3. **`max_tasks_per_node = 1` (module default).** A D64 node has 64 vCPUs; one
   fine-grained per-field task would waste 63 of them. We must either **raise
   max_tasks_per_node** (pack many tasks per node) or make each **task internally parallel**
   (one task builds many cubes across the node's cores, reusing our Snakemake-in-container
   or a process pool). **This is the key task-granularity design decision** (see §7).
4. **Keys disabled / MSI-only auth.** No account-key fallback anywhere — both adlfs *and*
   GDAL VSI must authenticate via Entra ID. Test both read paths early.
5. **Firewalled storage.** From a laptop you need VPN + `az login`; pool nodes are fine
   (they're in the subnet). Don't assume public blob access.
6. **D64ds_v6 has a local NVMe temp disk** (the `d`/`ds`). rise-eo's build is **I/O-bound**
   (load_images ≈73% of CPU); staging COG reads/writes on the node's local SSD (Batch task
   working dir) rather than round-tripping blob for scratch is likely a big lever.
7. **COG-on-download is local-dst only in rise-eo v1** (TODO #15: remote-dst COG =
   stage→convert→upload). Azure wants remote-dst; this parked TODO becomes in-scope for the
   Batch spec.
8. **No shared-dataset access** (`allowed_datasets = []`). rise brings its own S2 archive
   into project storage; we don't read the platform's Planet/Maxar containers.
9. **We don't run Terraform.** Any infra change (max_nodes, max_tasks_per_node, warm-cache
   image, adding `vms` for a driver) is a tfvars edit + gated `deploy.sh` by a platform
   admin. We propose; they apply.

---

## 7. Open design questions for the Batch spec (to work through with examples)

These are the decisions the future spec 10 must settle. Flagged here so we don't lose them:

1. **Task granularity / node packing.** One Batch task per datacube (simple, but
   `max_tasks_per_node` must go up to ~64 to fill a D64), *or* one task per node that fans
   out internally across the 64 cores (reuse Snakemake/process-pool inside the container)?
   Trade-off: scheduler overhead & retry granularity vs. node utilization. **Leaning:**
   coarse tasks that each saturate a node, given the build is I/O-bound and per-cube tasks
   are short.
2. **Where does the driver run?** Laptop-on-VPN (simplest to start), an AML compute job, or
   a small always-/on-demand VM (would need adding `vms` to rise)? The `common-ops` Batch
   example pairs `vms` + `azure-batch` for exactly this.
3. ~~**GDAL/VSI auth under MSI** for the raster-read exception — needs a spike.~~
   **✅ RESOLVED by spec 31 (proven on real Azure 2026-07-18, run-book `31-p1-datacube-on-blob` (tag `docs-archive-2026`)
   green).** `rise_eo.raster.rio_open` translates `abfss://`/`az://` → `/vsiadls/` (`storage/azure.py`)
   and opens inside a `rasterio.Env` carrying a **fresh `AZURE_STORAGE_ACCESS_TOKEN`** +
   `AZURE_STORAGE_ACCOUNT`; local paths stay a straight passthrough. The build streamed blob COGs
   this way end to end. **Residual GDAL-*writes* question RESOLVED by spec 38 D5 (2026-07-23),
   without re-spiking `rio_open`:** `raster.cog.to_cog` routes AROUND the write-to-blob gap rather
   than through it — GDAL still writes only to node-local scratch (`rio_open`'s `mode="w"` guard
   is untouched), then `storage.transfer` publishes the finished COG to blob. Inference-output COGs
   to blob are proven at the unit-test level (TODO #17 closed); the real-cluster run is pending
   (run-book `38-inference-on-aml` (tag `docs-archive-2026`)).
4. ~~**Input/output data layout in blob** — container/paths for the S2 archive, catalogs,
   datacubes, flattened arrays; how the catalog (GeoParquet) is shared to tasks.~~ **✅ RESOLVED
   by spec 36 D6** (signed off 2026-07-21): `<root>/imagery/...` (runbook 34), `<root>/runs/<run_id>/
   {input.csv, shards/<k>.csv, cubes/<cell_id>/, _status/<k>.json}` — spec 35's self-describing
   catalog is what made "how does a task learn its collection" not need answering separately.
5. **Container image** — moot for a Batch task (dropped, `AZURE_INFRA.md` §3.1); **spec 36 D5**
   answers the equivalent AML question — an AML **Environment** (conda/pip spec + rise-eo wheel),
   built and versioned by AML itself, no ACR push required.
6. ~~**Runner abstraction in `rise_eo.workflows`** — the seam so `--runner=snakemake|batch` selects
   dispatcher without touching the task CLI.~~ **✅ RESOLVED by spec 36 D3**: `runner="local"|"aml"`,
   `workflows.runners.run_aml` the only new dispatcher, `workflows/task.py` (the unit of work)
   provably unchanged (D3 invariant 1).
7. ~~**Idempotency / resume at scale** — today `done.txt` sentinels + skip-if-exists; how that
   behaves with Batch task retries and blob eventual consistency.~~ **✅ RESOLVED by spec 36 D7**:
   atomic-rename publish (temp path → `fs.rename`, §8.1's HNS-atomic-rename primitive) + the
   artifact's own existence as the resume signal, replacing `done.txt`; sentinels move to
   node-local scratch. Answered for AML task-recovery retries specifically (`AZURE_INFRA.md`
   §8.2-equivalent evidence: Azure retries independently of `maxTaskRetryCount`), which generalizes
   past Batch.
8. **Cost/observability** — **partly addressed by spec 36 D9**: each AML shard writes
   `_status/<k>.json` shaped like a spec-24 `_result.json`; the existing spec-11 `timings.json`
   sidecar already lands next to each cube via `rise_eo.storage` and needed no change. Budget-alert
   wiring itself is still open.
9. **Download, not just build, on AML — ✅ RESOLVED by spec 37** (implemented, pending review):
   the same `run_aml`/`shard_units`/D5-Environment/D4-identity/D9-telemetry machinery dispatches
   the *download*-to-blob path too (`workflows.runners.run_aml_download`), but the dispatch shape
   is **per-source**, not uniform fan-out (D1): CDSE's S3 concurrency cap is per-credential, so it
   runs as **one** job at its existing 4-wide concurrency; MPC reads straight from Azure Blob (no
   per-credential cap), so it **fans out** across N nodes near-linearly. §4.4's Key Vault plan is
   now concrete: `rise_eo.secrets.get_secret(vault_url, name)` reads the CDSE creds secret under the
   same `AZURE_CLIENT_ID` identity spec 36 D4 already sets — one identity covers blob and Key Vault,
   so this needed **no new infra grant** (the compute identity already holds `Key Vault Secrets
   User`). See `specs/37-download-on-aml.md`.
10. **Inference, not just build/download, on AML — ✅ RESOLVED by spec 38 (P4, 2026-07-23,
    pending cluster validation).** `workflows.runners.run_aml_inference` reuses `shard_units`/
    `_aml_submit_and_wait`/D4-identity verbatim (same machinery as #9), fanning out over **cells**
    (not per-source, since inference reads only the blob catalog — no CDSE, no per-credential cap,
    SO-6). **New infra unit: a *second*, model-specific AML Environment** (D4) — item 5's generic
    build Environment (`rise-eo[azure,mpc]`, no model) extended with the user's adapter package + its
    runtime deps (e.g. `scikit-learn`/`joblib`), built the same way (an `az ml environment create`
    Docker-build-context step) but as a distinct image referenced by name
    (`run_aml_inference(environment=…)`) — no per-run image build, one image serves many
    runs/bundles of that model family. Gated by a one-node adapter-import smoke (D11) before the
    N-node fan-out, so a missing dependency fails once at build/smoke time, not on every node. See
    `specs/38-inference-on-aml.md`, `docs/adr/0002-bundle-and-inference-image-decoupled.md`.

## 8. Things to confirm (not yet verified)

- ~~Exact Batch **account URL** output name and the region's Batch endpoint host.~~ **✅ CONFIRMED
  2026-07-21** (fork probe): `accountEndpoint` follows the documented
  `<batch-account>.<region>.batch.azure.com` shape, and `poolAllocationMode` is `BatchService` as
  assumed. Concrete value in the private doc.
- ~~Whether `adlfs` + `DefaultAzureCredential` covers all `rise_eo.storage` operations we use.~~
  **✅ Largely answered by spec 31 + runbooks 31/34** (green 2026-07-18/07-20): `read/write_parquet`,
  `save/load_npy`, `transfer` (`.part` + `mv`), `put`, `exists`, `makedirs` all ran against `rise`
  blob, and `FSSPEC_ABFSS_ANON` was proven to cross a subprocess boundary. ~~**Still unconfirmed:
  whether ADLS Gen2's rename is genuinely *atomic* under concurrent writers**~~ — **✅ RESOLVED
  from primary docs (2026-07-21), see §8.1 below.**
- Current **quota** actually granted on the Batch account (needs `az batch account show`) — **being
  answered by run-book `36-runner-fork-probe` (tag `docs-archive-2026`)**, together with the AML cluster names/quota and
  (the fork's real discriminator) whether the compute identity is attached to the AML clusters.
- ~~Whether the pool image (`microsoft-dsvm/ubuntu-hpc/2204`) + Docker is enough, or we need a
  custom node image.~~ **✅ CONFIRMED 2026-07-21** (fork probe): the pool runs that image with
  `nodeAgentSkuId = batch.node.ubuntu 22.04`, `containerConfiguration.type = DockerCompatible`, a
  start task already configured, and — notably — the project **ACR pre-wired as a container registry
  with the compute identity as its `identityReference`**. Batch was fully prepped for containers;
  only quota stopped it. Moot for now (§3.1), and good news whenever the quota ask lands.
- **NEW (AML), partly closed by spec-36 Phase 0 (green 2026-07-21):**
  - ~~whether an AML job must declare `identity: managed`~~ **✅ No.** A plain command job with no
    `identity:` block ran as the cluster's user-assigned identity (token `xms_mirid` = the compute
    identity). **But it only does so when the job sets `AZURE_CLIENT_ID`** — with that removed, the
    node's `DefaultAzureCredential` cannot pick among user-assigned identities and fails outright
    (proven by the run-book's negative control). See spec 36 D4.
  - which ACR the AML **workspace** builds environments into — still open, but the build log's
    password-based `docker login` plus the project ACR's `adminUserEnabled: false` points at the
    **workspace** registry rather than `acr<proj>`.

### 8.1 Atomic publish on ADLS Gen2 — RESOLVED (2026-07-21, primary docs)

Settles §7.7 (idempotency at scale), and it settles it **the same way whichever backend wins the
Batch-vs-AML fork**, so the spec-36 design can rely on it now:

- **Rename is atomic on an HNS account, and it can be made fail-if-exists.** The ADLS Gen2
  `Path - Create` REST operation performs rename via `x-ms-rename-source` and states: *"By default,
  the destination is overwritten… To fail if the destination already exists, use a conditional
  request with `If-None-Match: "*"`."* A losing racer gets `412 ConditionNotMet` (or `409
  PathAlreadyExists`), never a half-written artifact. `rise` storage is HNS-on, so this applies.
  Source: [Path - Create (ADLS Gen2 REST)](https://learn.microsoft.com/en-us/rest/api/storageservices/datalakestoragegen2/path/create).
  The HNS concept doc corroborates the atomicity ("a hierarchical namespace processes these tasks by
  updating a single entry (the parent directory)") and names the exact pattern we want — *"Tools like
  Hive and Spark often write output to temporary locations and then rename the location at the
  conclusion of the job."*
  Source: [ADLS hierarchical namespace](https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-namespace).
  ⇒ **Design consequence:** the sentinel/`done.txt` scheme can be replaced by *write-to-temp →
  atomic-rename-to-final*, with the final path's existence as the resume check. No lease, no lock,
  no eventual-consistency window to reason about.
- **Idempotent tasks are mandatory, not merely prudent.** Azure Batch retries a task when a node
  recovery operation is triggered (unhealthy node rebooted, node lost to host failure), and *"retries
  due to recovery operations are independent of and are not counted against the `maxTaskRetryCount`.
  Even if the `maxTaskRetryCount` is 0, an internal retry due to a recovery operation may occur…
  all Tasks should be idempotent."* So a re-executed unit of work must be safe **even with retries
  configured off**.
  Source: [Error handling and detection in Azure Batch](https://learn.microsoft.com/en-us/azure/batch/error-handling).

---

*Maintenance: update this when the `rise` tfvars change, when we confirm any "to confirm"
item, or when a design question is resolved (then fold it into the spec). Keep concrete
names/IDs/CIDR/budget out of this file — it's public; they belong in `raapid-infra` only.*
