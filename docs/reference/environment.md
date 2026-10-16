# Reference — environment variables

Every `AZ_*` variable the run-books use: what it means, **where the value comes from**, and a
command that tells you whether the one you set is right.

> **Continuously-true document** (spec 41 D3) — maintained, and mechanically checked:
> `tests/test_docs.py::test_az_vars_are_documented` fails the suite if `rise_eo.config`'s keys and
> this table drift apart. **Adding or renaming one requires editing `src/rise_eo/config.py` and this
> file in the same change** (spec 54 D6).

## How to use it

A user-level file holds what is **durable** — stable for you across runs and mostly across
projects — outside every project tree, so it can never be committed by accident (spec 54):

```bash
fsd init                         # interactive; writes ~/.config/fsd/config.toml
fsd init --blank                 # or: write it empty and fill it in yourself (spec 55 D4)
```

| in the file | why |
|---|---|
| `subscription_id`, `resource_group`, `workspace`, `cluster`, `uami_client_id` | **required** — the platform coordinates. A platform admin hands all five over together and every AML call needs them |
| `model_registry`, `image_registry` | **optional** — where the registries live. Named rather than chosen per run, and models and images both outlive the runs that made them (spec 55 D2) |

```python
import rise_eo
cfg = rise_eo.config.load()          # explicit -- fsd's own code never reads this on its own
```

**`root` is deliberately NOT in the file** (spec 55 D1). It names a *per-run destination* (cubes, outputs) chosen
by whoever runs the job, not a durable address — the imagery archive is the one shared exception (spec 59 D12), a single root every run reuses — so every fsd verb takes it as an argument
(`dst_folderpath=`, `output_folderpath=`, `runner_kwargs["root"]`). Nothing writes `AZ_ROOT` —
`fsd init` does not prompt for it, `fsd config` does not print it, and `--from-env-file` parses
and drops it. **It is not config.** The callers that need a root — the two tracked notebooks and
`demos/e2e_austria_aml.py` — read it from the environment themselves, because a public repo may
not hold a literal storage URL (spec 55 D3):

```bash
export AZ_ROOT=abfss://…         # read by the NOTEBOOK/demo script, never by fsd
```

**The bare `AZ_*` names override the file** for everything that *is* config (spec 54 D4) — they
sit *above* it in precedence:

```bash
export AZ_RG=my-resource-group   # overrides config.toml's [azure].resource_group for this shell
```

`fsd config` prints the resolved value of each key and which of the three sources it came from --
useful the moment a stale shell export silently outranks the file.

Concrete values for the `rise` platform are in **`AZURE_INFRA_PRIVATE.md` at the workspace root**
(uncommitted, never pushed). This repo is public MIT: **no real name, id, URL or CIDR ever lands in
it.**

**None of these are read by `src/rise_eo/`.** Every one is operator-facing — consumed by `rise_eo.config`,
the run-book shell, the `az` CLI, or `demos/e2e_austria_aml.py`. fsd's library code takes storage
locations as *arguments*; it never reads the environment or `config.toml` on its own (spec 54 D3).
(The one near-match you may find in `src/` is `_AZ_RE`, a compiled regex in `storage/azure.py:29`
— not a variable.)

## Verify everything at once

```bash
az account show --query '{sub:id, user:user.name}' -o table
az ml workspace show -n "$AZ_ML_WORKSPACE" -g "$AZ_RG" --query name -o tsv
az storage fs directory exists -f "$AZ_FS" --account-name "$AZ_ACCOUNT" \
  --auth-mode login -n "${AZ_ROOT#*windows.net/}" --query exists
```

**The single most valuable check** — an unset variable expands to an empty string and silently
builds a *wrong path* rather than failing:

```bash
for v in $(grep -oE '`AZ_[A-Z0-9_]+`' docs/reference/environment.md | tr -d '`' | sort -u); do
  [ -n "${!v}" ] || echo "EMPTY: $v"
done
```

⚠️ **`export VAR="$(az …)"` assigns the error text on failure**, so the variable looks set and is
garbage. Every `$(az …)` row below is worth echoing once before you rely on it.

## 1. Subscription & resource group

| variable | meaning | value comes from | verify |
|---|---|---|---|
| `AZ_SUBSCRIPTION_ID` | the subscription everything lives in | decode ring | `az account show --query id -o tsv` |
| `AZ_RG` | resource group holding the AML workspace + storage | decode ring | `az group show -n "$AZ_RG" --query name -o tsv` |
| `AZ_LOC` | region | decode ring | `az group show -n "$AZ_RG" --query location -o tsv` |

## 2. Storage

| variable | meaning | value comes from | verify |
|---|---|---|---|
| `AZ_ACCOUNT` | ADLS Gen2 storage account | decode ring | `az storage account show -n "$AZ_ACCOUNT" -g "$AZ_RG" --query isHnsEnabled` → must be `true` |
| `AZ_FS` | filesystem / container in that account | decode ring | `az storage fs show -n "$AZ_FS" --account-name "$AZ_ACCOUNT" --auth-mode login --query name` |
| `AZ_PREFIX` | your path prefix inside the container | your choice (a username works) | — |
| `AZ_SCRATCH_PREFIX` | scratch prefix, no leading/trailing slash | your choice | — |
| `AZ_ROOT` | **derived** root URL for a run's artifacts. **Not config** (spec 55 D1) — fsd never reads it; a notebook or a shell does | `abfss://$AZ_FS@$AZ_ACCOUNT.dfs.core.windows.net/…` | `python -c "import rise_eo.storage as s; print(s.fs.exists('$AZ_ROOT'))"` |
| `AZ_MODEL_REGISTRY` | where `rise_eo.model.registry` publishes. **Config** (optional), overrides `[azure].model_registry` | a durable path, NOT under a run root | `python -c "import rise_eo.storage as s; print(s.fs.exists('$AZ_MODEL_REGISTRY'))"` |
| `AZ_IMAGE_REGISTRY` | where `rise_eo.image.registry` publishes image definitions (spec 56). **Config** (optional) | a durable path, NOT under a run root | `python -c "import rise_eo.storage as s; print(s.fs.exists('$AZ_IMAGE_REGISTRY'))"` |

## 3. The archive and its catalog — ⚠️ four spellings of one idea

`AZ_ARCHIVE` · `AZ_ARCHIVE_ROOT` · `AZ_ARCHIVE_PATH` · `AZ_ARCHIVE_CATALOG` (+ `AZ_CATALOG`,
`AZ_CATALOG_URL`) all point into the same place. They accreted because each run-book was written in
a different week.

**They are deliberately NOT unified.** Run-books are point-in-time documents (spec 41 D3) and are
never edited after the fact — the same rule that forced the GitHub issue numbers to align rather
than rewriting 473 references. So: **set whichever the run-book you are running names**, and use
this table to see they are the same thing.

| variable | form | equals |
|---|---|---|
| `AZ_ARCHIVE_ROOT` | **canonical** — full `abfss://` root | `$AZ_ROOT` |
| `AZ_ARCHIVE` | full `abfss://`, root + `/archive` | `$AZ_ARCHIVE_ROOT/archive` |
| `AZ_ARCHIVE_PATH` | **container-relative**, no scheme — `az storage` CLI wants this form | `${AZ_ARCHIVE#*windows.net/}` |
| `AZ_ARCHIVE_CATALOG` | the catalog file | `$AZ_ARCHIVE_ROOT/archive/catalog.parquet` |
| `AZ_CATALOG` | alias | `$AZ_ARCHIVE/catalog.parquet` |
| `AZ_CATALOG_URL` | alias, full `abfss://` | `$AZ_ARCHIVE_CATALOG` |

**Verify (the one that matters — a wrong catalog path is a whole wasted run):**

```bash
python -c "
import rise_eo.storage as s, geopandas as gpd
print('exists:', s.fs.exists('$AZ_ARCHIVE_CATALOG'))
print('rows  :', len(gpd.read_parquet('$AZ_ARCHIVE_CATALOG')))"
```

Run-book 37 writes `$AZ_ROOT/archive/catalog.parquet`, which 36 and 38 read — **not** the `mpc/`
prefix from run-book 34.

## 4. Azure ML

| variable | meaning | value comes from | verify |
|---|---|---|---|
| `AZ_ML_WORKSPACE` | AML workspace | decode ring | `az ml workspace show -n "$AZ_ML_WORKSPACE" -g "$AZ_RG" --query name -o tsv` |
| `AZ_CLUSTER` | the d16 compute cluster | decode ring | `az ml compute show -n "$AZ_CLUSTER" -g "$AZ_RG" -w "$AZ_ML_WORKSPACE" --query '{state:provisioningState,max:scaleSettings.maxNodeCount}'` |
| `AZ_ACR` | container registry backing the environments | decode ring | `az acr show -n "$AZ_ACR" --query loginServer -o tsv` |
| `AZ_ENV_NAME` | general-purpose environment (download/build/flatten) | fixed: `fsd-aml-env` | — |
| `AZ_INFER_ENV_NAME` | inference environment — fsd + the adapter's **deps** only (since spec 44 it carries no adapter source; bundles do) | fixed: `fsd-infer-env` | — |
| `AZ_ENV_VERSION` | **query, never guess** | `az ml environment list … --query '[0].version'` | `echo "$AZ_ENV_VERSION"` — a number, not an error string |
| `AZ_INFER_ENV_VERSION` | same, for the inference env | same | `echo "$AZ_INFER_ENV_VERSION"` |
| `AZ_INFER_BUILD_CONTEXT` | folder holding the fsd wheel the inference image is built from; pass it as `rise_eo.model.verify_image(build_context=...)` to refuse an image built from a pre-spec-44 wheel | e.g. `notebooks/demo_model` | `ls "$AZ_INFER_BUILD_CONTEXT"/fsd-*.whl` |
| `AZ_ENV_NAME_VERSION` | `name:version`, read by `demos/e2e_austria_aml.py` | derived | `echo "$AZ_ENV_NAME_VERSION"` |

⚠️ **Rebuild the environment after any `src/rise_eo/` change**, then re-query the version. A stale
version silently runs old code on the nodes — the failure looks like a logic bug, not a config one.

## 5. Identity & secrets

| variable | meaning | value comes from | verify |
|---|---|---|---|
| `AZ_UAMI_NAME` | user-assigned managed identity the compute runs as | decode ring | `az identity show -g "$AZ_RG" -n "$AZ_UAMI_NAME" --query name -o tsv` |
| `AZ_UAMI_CLIENT_ID` | its client id | `az identity show … --query clientId -o tsv` | must be a GUID |
| `AZ_VAULT_URL` | Key Vault URL, if you have write access | decode ring | `az keyvault show --id "$AZ_VAULT_URL" --query name -o tsv` |
| `AZ_CDSE_SECRET_NAME` | CDSE credentials secret in that vault | decode ring | `az keyvault secret show --vault-name … -n "$AZ_CDSE_SECRET_NAME" --query name` |
| `AZ_CREDS_URL` | blob copy of the CDSE credentials the nodes read | derived from `AZ_ROOT` | `python -c "import rise_eo.storage as s; print(s.fs.exists('$AZ_CREDS_URL'))"` |
| `AZ_LOCAL_CREDS_JSON` | your local `cdse_credentials.json` | your machine | `test -f "$AZ_LOCAL_CREDS_JSON" && echo ok` |

**Nothing here is a secret value** — these are *names and locations* of secrets. Never put a
credential in `config.toml` or an `AZ_*` export.

## 6. Run inputs

| variable | meaning | value comes from |
|---|---|---|
| `AZ_ROI_REAL_LOCAL` | its local twin under `../shapefiles/` | your machine |
| `AZ_START` / `AZ_END` | window, `YYYY-MM-DD` | your choice |
| `AZ_BANDS` | comma-separated band list | `B02,B03,B04,B08,B8A,SCL` |
| `AZ_MAX_TILES` | cap on MGRS tiles | a real cap (issue #49); the Austria archive is 576 |

⚠️ **An ROI is one *region*, not a label set.** Passing a 900-polygon label file as `roi=` produced
1167 rows for 172 cells and killed a 300-cell cluster run twice (issue #58, spec 21 D-GRID-1).
`run_inference` now prints `roi -> N grid cells` in preflight — **N is the cluster workload; read it
before you spend.**

## 7. Fan-out & run control

| variable | default | meaning |
|---|---|---|
| `AZ_N_SHARDS` | `8` | fan-out width. **The right value differs by verb** — see [`docs/findings/workload-regimes.md`](../findings/workload-regimes.md): inference is work-bound and rewards 16-way fan-out; training is overhead-bound and wants 1–2 nodes with a large `cubes_per_task`. |
| `AZ_LOCAL_CORES` | `4` | cores for the local comparison leg |
| `AZ_MERGE_MODE` | `strict` | `strict` \| `reproject` (lossy display merge across UTM zones) |
| `AZ_COMPARE_CELLS` | `3` | cells sampled for the local-vs-cluster comparison |
| `AZ_ON_VM` | unset | set to any value to force "the driver is in Azure"; otherwise auto-detected via `/var/lib/waagent` |

## Related

- `rise_eo.config` (`src/rise_eo/config.py`) — the schema and loader for the seven keys `fsd init` writes
  (five required + two optional registries; `root` is not one of them — spec 55 D1)
- `AZURE_INFRA_PRIVATE.md` (workspace root, uncommitted) — the concrete values
- [`AZURE_INFRA.md`](AZURE_INFRA.md) — the scrubbed public description of the platform
- the run-books that used these variables are at tag `docs-archive-2026` (spec 102 P3c)
