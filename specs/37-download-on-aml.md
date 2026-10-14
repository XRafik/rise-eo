---
status: current
summary: Download at scale on AML: CDSE as one job, MPC fanned out to blob.
---

# Spec 37 — download on Azure ML (P2): dispatch the existing download-to-blob onto the cluster

> **Status: ✅ SIGNED OFF (user, 2026-07-22) → IMPLEMENTED + REVIEWED + MERGED to `main` (`6b845fc`,
> 2026-07-22).** **⚠️ D5 REVISED 2026-07-22 (keep-both: a blob-JSON `--creds-url` creds fallback added
> alongside KV — KV write was unavailable on the demo timeline; see the D5 REVISED note in §3, the
> re-resolved Open Q2 in §8, and the updated §4/§5/§7). → NEXT: a Sonnet@medium session implements the
> D5-revision delta against the merged code.** The original (pre-D5-revision) history follows.
>
> **Amendment A1 (2026-10-13, SIGNED OFF by the user 2026-10-14):** `download_shard` refuses a
> shard row with a missing key instead of filling in defaults (#145 X3; end of this spec).
>
> **→ (historical) NEXT: implement in a Sonnet@medium session against
> this spec** (§3 D1–D10, §4 reuse ledger, §5 deliverables, §7 tests). **Prerequisite spec 35 is already
> on `main`** (commit `f486c3c` — AML-downloaded catalogs self-describe via the ingest declaration
> stamp; no blocker). Then the user runs `runbooks/37-download-on-aml.md` Phases 0–3. Cross-validated
> before sign-off (CDSE quota, MPC
> rate limits, AML `CommandJobLimits`, Key Vault + managed identity — §9, per-source credit).
> Written by Opus@high, 2026-07-22, after spec 36
> (the AML *datacube* runner) merged to `main`. This spec is the **download sibling of spec 36**: it
> dispatches the **already-working** download-to-blob path (spec 34) onto the same AML cluster, so the
> source→blob byte-flow is cloud-colocated instead of relayed through the driver laptop.
>
> **Dispatch shape is source-dependent — the spec's headline decision (D1):** CDSE stays **one job**
> (its concurrency cap is per-credential, so fan-out cannot help); MPC **fans out across N nodes** (its
> bytes come straight from Azure Blob, and `rise` sits in the same region as MPC's data, so parallel
> nodes scale throughput near-linearly). Both reuse spec 36's dispatch machinery.
>
> **Relationship to prior specs (all prerequisites, all landed or landing):**
> - **Spec 34** (DONE) built download-to-blob itself — `stage-to-local-scratch → convert to COG (CDSE)
>   / byte-copy (MPC) → stamp offset/scale/nodata → push to blob → rewrite the catalog`. It also
>   **fixed the baseline-radiometry metadata** (TODO #30/#10). **This spec adds no download pipeline.**
> - **Spec 36** (MERGED) built the AML dispatch machinery: `runners.run_aml`, `shard_units`, the D5
>   Environment, the D4 identity env var, the D10 preflight, the D9 `_status/*.json` telemetry. **This
>   spec reuses all of it** — `shard_units` for the MPC fan-out, `n=1` for CDSE.
> - **Spec 35** (implemented) makes ingest `catalog.append` stamp the source declaration. **Land spec
>   35 on `main` before implementing this**, so an AML-downloaded catalog self-describes.

---

## 1. The problem, stated honestly

There is **no real satellite data on blob**. The archive downloaded earlier carried the pre-fix
(wrong) radiometry tag — it predates spec 34's `c2bf1f1` fix (TODO #44) — and only tiny test downloads
have run since. To validate the demo (`ROADMAP.md` §5.0) we need a real archive on blob, downloaded
with **correct** metadata, at a size that makes the datacube fan-out (spec 36) meaningful.

`api.download` already writes correct COGs to blob (spec 34), but it runs **in-process**: it has no
`runner=` seam (`api.py:216` hardcodes `_check_local_seams("local", storage)`), so every byte transits
whatever machine calls it. Run from the driver laptop over VPN, the laptop becomes the source→blob
relay — slow, and it ties up the presenter's machine for the whole transfer. Spec 34's own design note
says the scratch→push path is meant for *"where the scratch dir and the blob destination are both
close to the VM"* (`cdse.py:718`) — a cloud node — but nothing yet **puts** it on one.

**The gap is dispatch, not download.** This spec lets a download run on the cluster, colocated with
blob, driven by `runner="aml"` — and it uses the *right* parallelism for each source.

---

## 2. Scope

**In scope**
- A **download unit-of-work CLI** — `python -m fsd.workflows.download …` — thin, wrapping the existing
  download-to-blob. Two invocation modes (D3): `--roi …` (whole ROI, one job — CDSE) and `--shard …`
  (a pre-discovered asset slice — one of N MPC jobs).
- An **AML dispatcher** — `runners.run_aml_download(...)`: **CDSE →** submit one whole-ROI job;
  **MPC →** discover on the driver, `shard_units` the asset list, submit N per-shard jobs; wait,
  aggregate `_status/*.json`, raise on failure. Reuses spec 36's Environment (D5), identity (D4),
  preflight (D10), telemetry (D9), and `shard_units` (D2).
- `api.download` accepts `runner="local"|"aml"` + `runner_kwargs` (mirrors spec 36's `api.py` change).
- A **job timeout** (`CommandJobLimits(timeout=…)`) so a long transfer is not silently cancelled.
- **Credential/token delivery via Azure Key Vault**, read by the compute identity — no secret on blob
  or in the job spec (D5).
- A preflight that estimates a CDSE request against the **12 TB / 30-day** CDSE quota (reuse `plan_download`).

**Not in scope — and deliberately so**
- **CDSE fan-out.** Ruled out by measurement (D1): CDSE's concurrency ceiling is **per S3 credential**,
  so more nodes cannot buy throughput — they contend for the same 4 connections and trip the throttle.
- **Per-file streaming to blob** (TODO #31). Keep spec 34's whole-run scratch→push batch per job.
- **Reading MPC in place via `/vsicurl/` instead of materializing** (TODO #21). This spec *materializes*
  to `rise` blob on purpose (reproducibility, colocation, avoid re-signing) — streaming stays the
  opt-out, unchanged.
- **Datacube building** — spec 36's `run_aml`, run *after* this populates blob.
- **Raising the CDSE quota** (commercial credits / sponsorship) — an account action, not code.
- **Any `raapid-infra` change** — cluster, identity, environment already exist.

---

## 3. Decisions

### D1 — Dispatch shape is **per-source**: CDSE = one job; MPC = fan-out. Both by measurement.

**CDSE — one job.** Authoritative (CDSE *Quotas and Limitations*): general-user S3 access is **4
concurrent connections** at 20 MB/s each, **12 TB / rolling 30 days**; over quota it drops to **1
connection at 1 MB/s**. The 4-connection limit is a property of the **S3 credentials**, not the client
machine — so N nodes each opening `MAX_CONCURRENT_S3=4` present 4·N connections on one credential and
get throttled, not sped up. CDSE runs as **one** job at its existing 4-wide concurrency (exactly spec
10 Seam 2's original call).

**MPC — fan out.** Authoritative (MPC docs / discussion #246): MPC data is **read directly from Azure
Blob Storage**; a subscription key *"will increase the rate limit on requesting data access tokens
(SAS tokens), but won't have an effect on actual data transfer speeds, as data is accessed directly
from Azure Blob Storage."* So the transfer ceiling is **Azure Blob throughput, which scales with
parallelism** — there is no per-credential connection cap to contend for. The **only** shared limiter
is the SAS-signing endpoint, and it is amortized two ways: *"reuse a single token (up to its expiry)
for many file requests"*, and the optional `PC_SDK_SUBSCRIPTION_KEY` raises the signing rate. Two
supporting facts make MPC fan-out especially favourable here:

1. **Colocation.** MPC's public data (and its highest-rate signing) live in one Azure region, and the
   `rise` cluster is in **that same region** (per `AZURE_INFRA_PRIVATE.md` — not named here; `fsd/` is
   public). So N nodes read MPC **intra-region**: near-linear throughput, negligible egress cost.
2. **No convert.** MPC assets are already COG, so an MPC job is a pure signed byte-copy + tag-stamp
   (`mpc._transfer_and_stamp_one`) — trivially shardable, unlike CDSE's integrated transfer+convert
   pipeline.

So **`run_aml_download` chooses N by source**: CDSE forces `n_shards=1`; MPC defaults `n_shards` to the
cluster's `max_instances` (spec 36 D2), sharded round-robin over the discovered asset list.

### D2 — Two unit-of-work shapes, both over the **existing** download-to-blob code

- **CDSE (one job):** the job calls the unmodified `sources.cdse.download(roi, …, dst=abfss://…)` whole
  — its integrated transfer/convert/backpressure/circuit-breaker pipeline stays intact, staged on
  node-local scratch, pushed to blob at the end (`_push_scratch_to_remote`). **`sources/cdse.py`
  changes by zero lines.**
- **MPC (N jobs):** the driver runs discovery (STAC query — cheap, no bytes) to produce the asset work
  list, `shard_units` it, and each job downloads **its shard's assets** to blob. Because a shard is a
  pre-discovered asset slice (not an ROI), this needs a **small additive** MPC entry
  `sources.mpc.download_shard(work_rows, …)` that signs-on-node + `_transfer_and_stamp_one`s each row
  (re-using the existing per-asset transfer; the ROI-based `mpc.download` is **untouched**). Signing on
  the node (not the driver) avoids SAS-token-expiry between submit and run.

> **Amended by Amendment A1 (2026-10-13, end of this spec):** `download_shard` raises `ValueError`
> naming every missing key when a shard row lacks one of the 16 keys `discover_shard_rows` writes; it
> no longer fills in defaults.

### D3 — The CLI entrypoint: `python -m fsd.workflows.download` — thin (mirrors spec 36's `shard.py`)

`workflows/download.py` (new) carries **no pipeline logic**. Two modes:
- `--roi <url> --source cdse …` → `api.download(roi, …, dst)` (the whole CDSE job).
- `--shard <shard_csv_url> --source mpc …` → `sources.mpc.download_shard(rows, dst)` (one MPC shard).
Both resolve inputs via `fsd.storage`, and both write `<root>/runs/<run_id>/_status/<k>.json` (D9). The
job runs the CLI — no AML-specific entry-script contract, the property that made spec 36 D2 viable.

### D4 — Identity on the node: **reuse spec 36 D4 verbatim** — `AZURE_CLIENT_ID`, no storage change

Every job writes the catalog + COGs to blob (and, per D5, reads its creds/token config from blob).
This is the exact need spec 36 D4 solved and proved on the cluster (Phase 0, `mighty_seal_21kp83tsv7`):
the dispatcher sets `AZURE_CLIENT_ID = <compute identity client id>` in the job env, and
`storage/azure.py`'s bare `DefaultAzureCredential()` honours it. **`storage/azure.py` changes by zero
lines.** `identity_client_id` is a caller-supplied parameter, never hardcoded (public repo).

### D5 — Secrets come from **Azure Key Vault**, read by the compute identity — none on blob, none in the job spec

The `rise` Key Vault already exists and — decisively — **the compute identity already holds the `Key
Vault Secrets User` role on it** (`AZURE_INFRA_PRIVATE.md`), so **this needs no infra grant** (a spec-36
selling point preserved). The **same `AZURE_CLIENT_ID`** spec 36 D4 sets to authorise blob also
authorises Key Vault — `DefaultAzureCredential` selects the user-assigned identity by that client id for
*every* Azure SDK, KV included — so one env var covers storage **and** secrets, on the node and the
driver alike.

- **CDSE** S3 keys live as a KV secret (the creds JSON as the secret value). The CLI reads it with
  `SecretClient(vault_url, DefaultAzureCredential()).get_secret(name)` → a small additive
  `CdseCredentials.from_json_str(...)` (sibling of `from_json`; the ROI-based path is untouched).
- **MPC**'s optional `PC_SDK_SUBSCRIPTION_KEY` (raises the *signing* rate, D1 — not a data secret) rides
  the same KV read, or stays anonymous (lower rate).
- `vault_url` + the secret name(s) are caller-supplied `run_aml_download` parameters — the vault URL is a
  concrete `rise` identifier, so **never hardcoded in `fsd/`** (public repo), exactly like
  `identity_client_id`/`cluster`.
- **Reachability:** the KV is deny-by-default firewalled to the project subnets + VPN; the AML nodes sit
  in the compute subnet (VNet service endpoint) and the driver is on VPN, so both resolve the secret —
  nodes at run, driver at preflight.
- Rejected — **blob JSON** (this spec's earlier draft): a secret at rest in a container the run-book must
  remember to delete, with no native rotation. KV is the purpose-built store, already wired to our
  identity, so blob buys nothing and costs hygiene.
- Rejected — **job env var** (`environment_variables={…secret…}`): leaks the secret into the AML job
  spec/UI. Rejected — an **AML workspace connection/secret**: real, but AML-specific plumbing past the
  dispatcher (spec 36 D3's spirit) for no gain over the KV+identity we already have.
- New dependency: `azure-keyvault-secrets` in the `[azure]` extra (`azure-identity` is already there).

> **⚠️ D5 REVISED 2026-07-22 (keep-both: a blob-JSON creds fallback added; KV retained, unchanged).**
> Populating the KV secret proved impossible on the demo timeline: **no identity the operator can
> invoke holds a KV *write* role** (`Key Vault Secrets Officer`) on the `rise` Key Vault (`kv<proj>`,
> `AZURE_INFRA_PRIVATE.md`) — the compute
> UAMI holds only *read* (`Key Vault Secrets User`, D5 above), so it can *read* a secret but not
> *create* one. `az keyvault secret set` returned `ForbiddenByRbac` from **both** the driver laptop and
> the operator's `rise` compute VM (the VM attempt authenticated as the operator's own account, not the
> VM MSI — but even the MSI is the read-only UAMI). Getting a write role is a platform-admin action not
> available in the demo window. The operator **does** have blob *write*. So CDSE creds may now be
> delivered **either** way — caller's choice, **mutually exclusive** (preflight errs if neither or both
> is given):
> - **KV (unchanged, the preferred path wherever a write role exists):** `--vault-url`/`--secret-name`
>   → `CdseCredentials.from_json_str(get_secret(vault_url, secret_name))`.
> - **Blob JSON (new fallback, used for the demo):** `--creds-url <abfss://…/_secrets/cdse_credentials.json>`
>   → the **existing** `CdseCredentials.from_json(creds_url)`, which already reads through
>   `fsd.storage.fs.open` (`cdse.py:82`), so it supports `abfss://` with **no new read code**. The node
>   reads it with the same `AZURE_CLIENT_ID` (D4) identity that already reads the ROI and writes the COGs.
>
> **Invariant preserved:** neither path puts a secret *value* in the job spec — `creds_url` is a blob
> *location* (like the ROI/dst paths already in the command), `vault_url`/`secret_name` are non-secret
> names; the creds value is resolved at run time by the node's identity in both cases.
>
> **Trade-offs now consciously accepted for the blob path** (these were D5's original reasons to reject
> blob, so we own them explicitly): (1) a **plaintext secret at rest on blob** — mitigated by a dedicated
> `_secrets/` prefix (never the data prefix), **delete-after-run enforced in the runbook** (a Phase-3
> teardown step), and CDSE keys being time-boxed via `s3_keys_expire`; (2) **no native rotation** —
> re-upload to rotate, and the keys expire anyway; (3) **read scope = whoever can read that container** —
> an **infra check the operator must confirm**: `_secrets/` must be no more broadly readable than
> {compute identity + operator}. KV stays the more secure store and becomes the path again the moment a
> write role is available — the blob fallback exists because that role is unavailable now, not because it
> is better.

### D6 — Job timeout: set `CommandJobLimits(timeout=…)` so a long job is not silently cancelled

Authoritative (`azure.ai.ml.entities.CommandJobLimits`): `limits.timeout` (seconds) cancels the job
when reached. The dispatcher passes an explicit, generous `timeout_seconds` (for CDSE, sized from
`plan_download`'s GB estimate ÷ a conservative MB/s with a floor; for an MPC shard, from its byte
share) rather than trusting an unknown default. `command(…, limits=CommandJobLimits(timeout=…))`.

### D7 — Preflight before the spend: reuse spec 36 D10 **+ a CDSE-quota estimate**

Cheap checks that turn 20-minutes-later failures into instant ones: cluster ready; Environment
resolves; blob `root` reachable **and writable**; the KV secret resolves and parses (and CDSE keys not
`s3_keys_expired()`) — read with the driver's `az login` identity, which carries KV read for whoever
populated the vault; discovery returns ≥1 asset. **CDSE-only:** a `plan_download`/`max_tiles` estimate
of GB, warning if it plausibly exceeds the remaining **12 TB / 30-day** quota (past which every
transfer drops to 1 MB/s). `plan_download` (`cdse.py:1134`) already computes this.

### D8 — Idempotency & resume: the **catalog + final-dst skip**, with one honest limitation

`download()`/`download_shard` skip an asset whose **final dst already exists** and upsert the catalog
as they go, so a completed job/shard is safely re-runnable. **Limitation, stated plainly:** the skip
checks the *local scratch* dst, and spec 34's push is **whole-run** (scratch discarded after push). A
job that **crashes mid-run** loses its un-pushed scratch, so a re-dispatch on a fresh node
**re-downloads** the un-pushed remainder — it does not see COGs already on blob. For jobs/shards that
run to completion (the "populate blob once" path) this is a non-issue; for crash-resume it costs
re-downloaded bytes. Accept for v1 (matches spec 34's batch-push model); a blob-final-existence skip is
the small enhancement that closes it (Open Q3), composing with the per-file-streaming TODO #31. **MPC's
fan-out makes this cheap:** a crashed shard re-runs only its slice, not the whole ROI.

### D9 — Telemetry: `_status/<k>.json`, same `_result.json` shape as spec 36 D9 / spec 24

Each job writes `<root>/runs/<run_id>/_status/<k>.json` — `{unit, status, n_assets, n_skipped,
n_failed, bytes_downloaded, seconds, circuit_tripped, error}` — from the existing `DownloadResult`.
Same shape a run-book `_result.json` uses. `run_aml_download` reads them back and **raises on any
failed or circuit-tripped** job, listing which (spec 36 `run_aml`'s aggregate-and-raise, reused).

### D10 — Environment: **reuse spec 36's AML Environment** (verify the transport deps)

The job installs the same fsd wheel + `[azure]` extra spec 36 D5 built (GDAL/rasterio already present
for CDSE's COG convert). **Verify at Phase 1** (don't assume) that it also carries: `s3fs` (CDSE S3
transport, `storage.transfer`), `planetary-computer` + `pystac-client` (MPC signing/discovery), and
`azure-keyvault-secrets` (D5). Any missing one goes in the `[azure]` extra (or a new `[sources]` extra)
— a Phase-1 check, not a design unknown.

---

## 4. What this does *not* change (the reuse ledger)

| Component | Change |
|---|---|
| `sources/cdse.py` (download-to-blob) | **none** (spec 34 already did it) |
| `sources/mpc.py` | `+ download_shard(rows, …)` — **additive**; the ROI-based `download()` untouched |
| `sources/cdse.py` | `+ CdseCredentials.from_json_str(...)` — **additive** (D5, KV secret value); pipeline untouched. **D5-revised:** the blob-creds path *reuses* the existing `from_json(creds_url)` as-is (already blob-capable via `fs.open`) — no new code |
| `storage/*` (incl. `azure.py`) | **none** — D4 is an env var the dispatcher sets |
| secrets access | `+ a thin KV read` (`fsd.secrets.get_secret`, `azure-keyvault-secrets` in `[azure]`) — D5 (retained). **D5-revised:** blob-creds path adds a `--creds-url` alternative that needs no KV read |
| `datacube/`, `raster/`, `catalog/`, `bands/` | **none** |
| `workflows/runners.py` | `+ run_aml_download`; factor `_aml_submit_and_wait` shared with `run_aml`; reuse `shard_units` |
| `workflows/download.py` | new, thin CLI (`--roi` / `--shard` modes) |
| `api.py` | `download` accepts `runner="aml"` + `runner_kwargs`; `_check_local_seams` already accepts `"aml"` |
| AML Environment (spec 36 D5) | reused; verify `s3fs` / `planetary-computer` / `pystac-client` present (D10) |

---

## 5. Deliverables

| # | Deliverable |
|---|---|
| 1 | `workflows/download.py`: the in-job CLI (D3, `--roi`/`--shard`), writes `_status/<k>.json` (D9) |
| 2 | `sources/mpc.py::download_shard`: additive per-shard signed-transfer entry (D2) — existing `download()` untouched |
| 3 | `workflows/runners.py::run_aml_download`: per-source dispatch (CDSE 1 job / MPC N shards) → wait → aggregate → raise (D1/D2/D6/D7/D9); factor `_aml_submit_and_wait` shared with `run_aml` |
| 4 | `api.py`: `download(runner="local"\|"aml", runner_kwargs=…)` end-to-end |
| 5 | Secret/token delivery (D5, **keep-both** per the D5-revised note): **either** Key Vault (`fsd.secrets.get_secret` + `CdseCredentials.from_json_str`; `azure-keyvault-secrets` in `[azure]`) **or** a blob-JSON `--creds-url` (reuses the existing `CdseCredentials.from_json`), mutually exclusive; **no secret *value* in the job spec** either way. (The blob path *does* place a plaintext creds file at rest on blob — accepted, mitigated per the D5-revised trade-offs; delete-after-run in the runbook.) |
| 6 | Preflight (D7) + job timeout (D6) |
| 7 | Docs: `AZURE_INFRA.md` (download-on-AML, per-source shape); `LIMITATIONS.md` (D8 crash-resume); `RECIPES.md`; `CHANGES.md`; **fix the stale `cdse.py:676` docstring** (says remote+cog "raises… deferred"; the code has staged→pushed since spec 34) |
| 8 | Run-book `37-download-on-aml.md`: Phases 0–3 (§6), user-run |

---

## 6. Validation — phased (Claude never runs these; each is a run-book, user pastes back `_result.json`)

- **Phase 0 — identity + KV secret read.** Reuses spec 36 Phase 0 (green). Adds: a node reads the CDSE
  creds secret from Key Vault via `SecretClient(vault_url, DefaultAzureCredential())` (same
  `AZURE_CLIENT_ID` identity) and `require_s3()` succeeds — proving D5, and that the compute identity's
  `Key Vault Secrets User` role reaches the vault from the compute subnet, before any download runs.
- **Phase 1 — one tile to blob, per source.** CDSE `--roi` single job + MPC single shard, a 1-tile
  window, `dst=abfss://…`. Proves the CLI wiring, the Environment (incl. `s3fs`/`planetary-computer`,
  D10), the stage→(convert)→push, and a **correct radiometry tag** on the blob COG. Compare the catalog
  + one COG's GDAL tags against a local download of the same tile.
- **Phase 2 — MPC fan-out + speedup.** N shards over a multi-tile MPC ROI; confirm a true partition (no
  asset lost/duplicated across shards), and **measure wall-clock vs `n_shards=1`** to evidence the D1
  fan-out claim (expect near-linear until the blob/signing knee). This is the phase that answers "does
  MPC fan-out actually make it faster."
- **Phase 3 — the real archive.** The ROI/window you actually need on blob — CDSE one job and/or MPC
  fanned out. **This is the deliverable that unblocks spec 36's datacube fan-out on real data** and
  retires the TODO #44 pre-fix artifacts.

---

## 7. Tests (fast, synthetic, deterministic — no Azure, no CDSE, no MPC, no network)

Reuse spec 36's injection pattern: `_FakeMLClient` + the `fake_aml_command` fixture.

1. `workflows/download.py` `--roi` mode calls `api.download` with the expected kwargs and writes a D9
   `_status/*.json`; `--shard` mode calls `mpc.download_shard` with the shard rows — both with the real
   download mocked (no network).
2. `run_aml_download(source="cdse")` submits **exactly one** job regardless of tile count (D1);
   `run_aml_download(source="mpc")` shards a discovered asset list with `shard_units` and submits **N**
   jobs — asserted against the fake submission. Both carry `AZURE_CLIENT_ID` (D4) and `limits.timeout`
   (D6), the KV `vault_url`/secret-name (non-secret) in the command args, and **no secret value** in
   `environment_variables` (D5).
3. MPC sharding is a **partition** (reuse spec 36's `shard_units` tests) — no asset lost/duplicated;
   `K>N` degrades to `N` non-empty shards.
4. `run_aml_download` raises when a mocked job reports Failed, **and** when `_status/*.json` reports
   `circuit_tripped: true` even if AML says Completed (D9).
5. `api.download` accepts `runner="aml"` (threads `runner_kwargs`) and rejects unknown runners.
6. Preflight (D7) refuses: empty discovery; unwritable root; a KV secret that does not resolve/parse or
   whose CDSE keys are `s3_keys_expired()` (KV read mocked); and **warns** when the CDSE GB estimate
   exceeds an injected remaining-quota threshold.
7. D5: `fsd.secrets.get_secret` is mocked (a fake `SecretClient`); the CLI parses the returned value via
   `CdseCredentials.from_json_str` — no secret in argv, no KV network call in the test.
7b. D5-revised (blob path): the CLI's `--creds-url` mode reads a `memory://…/cdse_credentials.json` via
   `CdseCredentials.from_json` and calls `cdse.download` (real download mocked) — no KV, no network.
   Dispatcher: CDSE with `creds_url` puts `--creds-url` (a location, not a secret) in the command and
   still leaks **no creds value**; preflight resolves/expiry-checks the blob creds and **errs when
   neither or both** of {`creds_url`, (`vault_url`,`secret_name`)} is supplied.
8. **Non-vacuousness** (project standard): a mutation making CDSE submit a 2nd job fails test 2's
   single-job assertion; a mutation dropping an MPC asset fails test 3's partition assertion.

---

## 8. Open questions (resolve during implementation; do not block sign-off)

1. **MPC shard granularity** — round-robin over **assets** (finest, best load-balance) vs over
   **items/MGRS-tiles** (coarser, fewer signing calls, keeps a tile's bands on one node). Lean:
   per-asset round-robin via `shard_units` (matches spec 36); confirm at Phase 2.
2. ~~**Secret/token delivery** (D5)~~ ✅ **RESOLVED (user, 2026-07-22): Azure Key Vault**, read by the
   compute identity (which already holds `Key Vault Secrets User`). Replaces the earlier blob-JSON draft.
   **↳ RE-RESOLVED (user, 2026-07-22, later): keep BOTH — KV *and* a blob-JSON `--creds-url` fallback**
   (see the **D5 REVISED** note in §3). KV write turned out unavailable on the demo timeline
   (`ForbiddenByRbac` from laptop + VM), so blob is the path used now; KV stays wired for when a write
   role lands. The earlier "replaces the blob-JSON draft" line is thus superseded — blob is back, as a
   deliberate, mitigated fallback alongside KV rather than the sole mechanism.
3. **Crash-resume re-download** (D8) — accept for v1, or add a blob-final-existence skip so a fresh-node
   resume sees COGs already pushed? Lean: accept + open a TODO (composes with TODO #31 streaming).
4. **`n_shards` / `timeout_seconds` defaults** — `max_instances` for MPC, and timeout from the GB
   estimate; oversubscription (more MPC shards than nodes, for straggler smoothing) is a Phase-2
   measurement, not a design guess.
5. **Subscription key** — whether to require `PC_SDK_SUBSCRIPTION_KEY` for the fan-out (higher signing
   rate) or start anonymous and add it only if Phase 2 shows the signing endpoint is the knee.

---

## 9. Best-practice alignment / sources

Per `CLAUDE.md`'s standing practice: what each source actually contributed.

- **[CDSE — Quotas and Limitations](https://documentation.dataspace.copernicus.eu/Quotas.html)** — the
  fact behind **D1 (CDSE=1 job)** and **D7**: *"Number of concurrent connections limit: 4"*,
  *"Bandwidth limit per connection (MB/s): 20"*, *"Monthly transfer limit (TB): 12"*, and *"After
  reaching this monthly transfer limit, the maximum bandwidth drops to 1MB/s and the number of
  concurrent connections drops to 1."* The cap being **per credential** is why CDSE fan-out cannot help.
- **[MPC — STAC API Data/Rate Limits (discussion #246)](https://github.com/microsoft/PlanetaryComputer/discussions/246)**
  and **[Where to get PC_SDK_SUBSCRIPTION_KEY (#77)](https://github.com/microsoft/PlanetaryComputer/discussions/77)**
  — the fact behind **D1 (MPC fans out)**: a subscription key *"will increase the rate limit on
  requesting data access tokens, but won't have an effect on actual data transfer speeds, as data is
  accessed directly from Azure Blob Storage"*; highest rates come from *"an API key… and requests from
  within the Azure West Europe region"*; and *"reuse a single token (up to its expiry) for many file
  requests."* So MPC throughput scales with parallelism (Blob egress, not a per-credential cap), the
  signing endpoint is the only shared limiter, and it is amortized by token reuse + the key.
- **[Planetary Computer SAS — Get Token REST API](https://learn.microsoft.com/en-us/rest/api/planetarycomputer/data-plane/sas/get-token?view=rest-planetarycomputer-data-plane-2025-04-30-preview)**
  — corroborated that data access is via short-lived SAS tokens on the blob URL (hence sign-on-node in
  D2 to avoid expiry between submit and run).
- **[Key Vault secrets + managed identity (`SecretClient` / `DefaultAzureCredential`)](https://learn.microsoft.com/en-us/azure/key-vault/secrets/quick-create-net)**
  and **[user-assigned identity selection via `AZURE_CLIENT_ID`](https://github.com/Azure/Azure-Functions/issues/2100)**
  — basis for **D5**: `SecretClient(vault_url, DefaultAzureCredential()).get_secret(name)`, and a
  user-assigned identity is selected by setting `AZURE_CLIENT_ID` — the **same** env var spec 36 D4
  already sets for blob, so one identity/credential covers storage and Key Vault. The load-bearing
  local fact is that **the compute UAMI already holds `Key Vault Secrets User` on the `rise` vault**
  (`AZURE_INFRA_PRIVATE.md`) — so D5 is zero-infra-ask, and the vault is firewalled-reachable from the
  compute subnet + VPN.
- **[`azure.ai.ml.entities.CommandJobLimits`](https://learn.microsoft.com/en-us/python/api/azure-ai-ml/azure.ai.ml.entities.commandjoblimits?view=azure-python)**
  — basis for **D6**: `limits=CommandJobLimits(timeout=<seconds>)`, the max run duration after which the
  job is cancelled — why a long transfer needs an explicit timeout.
- **[Command job YAML schema (v2)](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command?view=azureml-api-2)**
  — confirmed a single command job runs one arbitrary CLI on one instance with `limits.timeout`, the
  same submission shape spec 36 validated — no parallel-job entry-script contract (D2/D3).
- **Internal (why this spec is small):** spec 34 built + runbook-validated download-to-blob with correct
  radiometry (`cdse.py::_push_scratch_to_remote`, `mpc.py::_transfer_and_stamp_one`); spec 36 built the
  dispatch machinery reused here (`run_aml`, `shard_units`, `_aml_preflight`, `_import_aml_command`, D5
  Environment, D4 `AZURE_CLIENT_ID`); spec 35 stamps the declaration at ingest. `config.MAX_CONCURRENT_S3
  = 4` encodes the CDSE cap; `config.MPC_MAX_CONCURRENT = 4` is a conservative default, **not** an
  MPC-imposed cap (its own comment: *"kept small and hotspot-friendly"*) — which is why raising MPC
  parallelism is safe. The `rise`-in-MPC's-region colocation is from `AZURE_INFRA_PRIVATE.md` (not named
  here — `fsd/` is public).

*Searched (standing spec-cross-validation permission): CDSE S3 concurrent-connection / rate / monthly
transfer limits and over-quota throttling; Microsoft Planetary Computer SAS-token signing rate limits,
subscription-key effect on transfer vs signing, and West-Europe colocation; Azure ML v2 command-job
`CommandJobLimits.timeout`; `azure-keyvault-secrets` `SecretClient` with a user-assigned managed
identity via `AZURE_CLIENT_ID`.*

---

## Amendment A1 (2026-10-13) — `download_shard` refuses a shard row with a missing key (#145 X3)

**Status:** SIGNED OFF (user, 2026-10-14), before any X3 code was written. At sign-off the
user cut three tests and the gate-4 negative case as redundant (How to verify). Decided by the user on
2026-10-09 (question Q3f of the refactor audit, #145 Decisions comment 6081711552: "Drop. A
driver/image version mismatch fails with a clear `KeyError`") and 2026-10-13 (the four X3 questions:
drop the defaults in both layers; one check up front that raises `ValueError`; record it in this spec;
ship X3 alone, before G3). The exception type supersedes Q3f's `KeyError` (A1.6). Amends D2. Line
numbers below are at `main` a4b5f3e, before the X3 code.

**Problem.**
- The driver writes one shard CSV per MPC job (`workflows/runners.py:1247-1249`). Every row comes from
  `discover_shard_rows` (`sources/mpc.py:631-650`), which writes 16 keys. The node reads the CSV
  (`workflows/download.py:137-141`) and passes the rows to `download_shard` (`:144`).
- `download_shard` fills in five of those keys when a row lacks them (`mpc.py:707`, `:709-712`):
  `scale` (the declaration's), `acquisition_key` (the granule name), `processing_version` and
  `processing_datetime` (`None`), and `properties` (`"{}"`).
- Today's driver always writes all 16 keys, and `run_id` is new for every dispatch
  (`runners.py:1152`), so no node reads a shard CSV left over from an older run. A key is therefore
  missing only when the driver runs an older fsd than the node's image, or when a caller builds the
  rows by hand (`download_shard` is public: `mpc.py:55`).
- In the first case the defaults are silently wrong. A driver from before spec 59 writes no
  `acquisition_key` or `processing_*`. The node then catalogs each granule with its own name as the
  acquisition key and no processing, so two processings of one acquisition no longer share a key
  (spec 59 D4; the D6 build check and the D7 report both group by that key). The job reports
  success; nothing points back at the download.
- `_append_downloaded` holds three more defaults (`mpc.py:436`, `:438`, `:442`: `scale`,
  `acquisition_key`, `properties`). Both callers always pass those keys: `download()` passes
  `_items_to_gdf` rows (`mpc.py:181-191`), and `download_shard` passes its `tile_meta`, built with
  every key. These defaults never fire.
- The `else 1.0` in `declaration.scale if declaration else 1.0` (`:707`) is dead too: `declaration` is
  `None` only when `rows` is empty (`:688`), and then the loop does not run.

**Decision.**
- **A1.1** `download_shard` checks its rows first, before `_import_pc_sign` (`:679`), before
  `makedirs` (`:682`) and before any transfer. It collects every key, of the 16 below, that at least
  one row lacks. If there is any, it raises `ValueError`, and nothing is signed, downloaded or
  appended. The message:
  - names the missing keys, sorted;
  - says `discover_shard_rows` writes them, so the shard CSV was most likely written by a driver
    running a different fsd version than this node's image;
  - says how to fix it: run both from one fsd version, by updating the driver's fsd or rebuilding the
    AML environment image (`docs/howto/build-the-images.md`).

  The 16 keys, as one private module-level tuple in `sources/mpc.py`: `tile_id`, `band`, `href`,
  `dst`, `offset`, `collection`, `timestamp`, `s3url`, `cloud_cover`, `scale`, `nodata`,
  `acquisition_key`, `processing_version`, `processing_datetime`, `properties`, `geometry`.
- **A1.2** `download_shard`'s `tile_meta` reads every key with `row[...]`. The five defaults and the
  `else 1.0` go.
- **A1.3** `_append_downloaded` reads `scale`, `acquisition_key` and `properties` with `r[...]`;
  `processing_version` / `processing_datetime` too, inside `_none_if_nan`, as the grep line under
  How to verify requires (clarified 2026-10-14, during implementation; no behaviour change).
- **A1.4** Unchanged: `_none_if_nan` on `processing_*` (`:439-440`). A key that is present with an
  empty value is not missing: the CSV round trip turns `""` into NaN, and `_none_if_nan` turns NaN into
  `None`. The check reads keys, never values.
- **A1.5** Unchanged: an empty `rows` passes the check and returns an empty result, as now.
- **A1.6** `ValueError`, not `KeyError`: Python defines `KeyError` as a failed mapping lookup and
  `ValueError` as an argument of the right type with an inappropriate value (Prior art). Here the
  argument is a list of dicts whose shape is wrong. It is a check, not a lookup.
- **A1.7** Where the user sees it: the node job fails; the driver's `_aml_submit_and_wait` raises
  `RuntimeError: job(s)/shard(s) failed: ...` (`runners.py:649`), as for any node failure, and the
  message itself is in that job's log. The node writes no `_status/<k>.json`, because `run_shard`
  raises before `_write_status` (`download.py:144-152`). Unchanged by X3.
- **A1.8** In the same PR: one sentence on the check in `download_shard`'s docstring; the pointer at
  D2.

**Prior art.**
- D8 of spec 59: "No shim, no migration tool (spec 58 D12's standing policy): a pre-59 catalog is not
  read. It raises naming the missing columns". A1.1 applies the same rule to shard rows.
- #145 X1 (spec 52's 2026-10-13 amendment) and X2 (spec 59 Amendment A2): a pre-spec tolerance is
  dropped and fails loudly instead.
- [Python 3 — Built-in Exceptions](https://docs.python.org/3/library/exceptions.html), the basis for
  A1.6: `KeyError` is "Raised when a mapping (dictionary) key is not found in the set of existing
  keys"; `ValueError` is "Raised when an operation or function receives an argument that has the right
  type but an inappropriate value, and the situation is not described by a more precise exception".

**How to verify.** Each new test is input → expected outcome.
- **T1** One row with every key except `scale` and `properties` → `download_shard` raises `ValueError`;
  the message contains `'properties'`, `'scale'`, `discover_shard_rows` and `different fsd version`;
  the sign stub was never called; the catalog received no append. Fails on the old code, which
  downloads.
- **T2** One row with all 16 keys, `processing_version` and `processing_datetime` both NaN → no raise;
  the appended catalog row has `None` for both. Guards A1.4 (a check that treats NaN as missing);
  passes on the old code too.
- **T3** `workflows.download.run_shard` on a `memory://` shard CSV with the `scale` column dropped →
  `ValueError` with the T1 wording, and no `_status` JSON written. Fails on the old code.
- `grep -nE '\.get\("(scale|acquisition_key|processing_version|processing_datetime|properties)"' src/fsd/sources/mpc.py`
  prints nothing.
- Already covered, so no new test: real `discover_shard_rows` output passes the check, through a CSV
  and `run_shard` (`tests/test_spec59_p1.py:749`); A1.3's removed defaults (the grep line).
- **Gate 4** (the check runs on the AML node): a run-book in `runbooks/`, happy path only. One MPC
  shard, one tile, B04 → the job completes; the catalog row has a non-empty `acquisition_key` and
  `processing_version`. The refusal itself is proved by T3 through the node's own entry point; on AML
  it would add only that an exception reaches the job log.

**Risks.**
- A caller of the public `download_shard` that builds rows by hand without these keys now raises. Fix:
  add the keys. In this repo, two test fixtures (`tests/test_download_aml.py:174-179`, `:197-201`) lack
  `scale`, `processing_version`, `processing_datetime` and `properties`; the X3 PR adds them.

**Out of scope.**
- The opposite skew, a driver newer than the image: the old node ignores columns it does not know, so
  nothing fails. Catching it would need a version stamp in the shard CSV.
- The CDSE path: one whole-ROI job, no shard rows.
- `_row_matches_window` / `_dedupe_on_unit_identity` skipping missing columns (logged on #145 by X2).
- G3's rewrites of `download` / `download_shard` (#145 A2, A8); they move the simplified code later.
