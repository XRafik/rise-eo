# How to: build the two AML node images

> **Last verified:** 2026-08-27 @ spec 56. Re-verify after any change to
> `rise_eo.image`/`rise_eo.aml`, to the extras in `pyproject.toml`, or to the base image.

Everything on an Azure ML cluster runs inside a container. fsd needs **two**, and neither of them
is per-model. Since spec 56 the recipe for each is an `rise_eo.image.ImageDefinition` -- declared in
Python, resolved, digested, and handed to `rise_eo.aml.ensure_environment`, which checks the image
registry before building anything. No Dockerfile lives in this repo any more; `render_dockerfile()`
generates one at build time.

Do this **before** [`run-at-scale.md`](run-at-scale.md). That page assumes both images exist.

## The two images

| image | used by | what it adds |
|---|---|---|
| **`fsd-aml-env`** | download shards, datacube builds, `create_training_data`'s flatten | fsd + `[azure,mpc]` |
| **`fsd-infer-sklearn`** | the `run_inference` fan-out, `verify_image`'s smoke job | the same, **plus** `scikit-learn` + `joblib` |

The inference image is generic **per dependency family**, never per model. Since spec 44 the
adapter's *source* travels inside the bundle, so the image copies no adapter and sets no
`PYTHONPATH`. Every sklearn model you ever bundle runs on this one image. A torch model needs an
`fsd-infer-torch` -- `BASE.derive(name="fsd-infer-torch", extra_pip=("torch",))` -- built
identically with the last dependency swapped.

## Prerequisites

- `az login` into the tenant hosting the workspace, plus the `ml` extension (`az extension add -n ml`).
- An AML workspace and resource group -- a platform-admin action, not something this repo
  provisions. Ask for them by name; see [`run-at-scale.md`](run-at-scale.md).
- A local fsd checkout with its venv (`pip install -e ".[dev,aml,mpc,azure,grid]"`) -- or, for a
  consumer with no checkout, fsd installed from a git ref (`pip install "fsd[aml] @ git+https://github.com/nikhilsrajan/fsd@<ref>"`).
- Your Azure coordinates in `~/.config/fsd/config.toml`, **including `image_registry`** (spec 55
  D2 -- the storage-seam path definitions are published to):
  ```bash
  fsd init      # interactive; fill in every value, including image_registry
  ```
  Every variable is documented in [`../reference/environment.md`](../reference/environment.md).

## Declare the definitions

```python
from rise_eo.image import ImageDefinition

# fsd developer, from a checkout: fsd="path:..." -- the ONE case that builds a wheel and hashes
# its content (spec 56 D5). A consumer with no checkout uses a pinned git ref instead:
#   fsd="git+https://github.com/nikhilsrajan/fsd@<40-char sha or branch>"
BASE = ImageDefinition(
    name="fsd-aml-env",
    fsd="path:/path/to/your/fsd/checkout",
    extras=("local", "azure", "mpc"),
)
INFER = BASE.derive(name="fsd-infer-sklearn", extra_pip=("scikit-learn", "joblib"))
```

> **`local` is not optional here (#80).** `snakemake` lives in the `[local]` extra, and the
> in-job entrypoints (`rise_eo.workflows.shard`, `rise_eo.workflows.infer_shard`) call back into
> `runners.run_local` — a node runs the same Snakemake orchestration a laptop does. Leave it
> out and the build succeeds, then the dispatch fails ~30 min in. If you built images before
> 2026-09-04, rebuild them with `local` added.

`ImageDefinition` is a frozen dataclass -- it renders a Dockerfile (`render_dockerfile()`) and a
build context (`write_context(dir)`); it never builds one and never touches the network by
itself. `name` never affects what gets built: it is where the image is registered, not what it
contains (D2) -- `BASE.derive(name=...)` alone changes nothing about the image.

**Bringing your own Dockerfile?** `ImageDefinition(name=..., build_context="./my-context")` takes
a directory you own; fsd digests its file contents (never the path) and builds it unchanged, no
`fsd=`/`base=`/`extras=` needed.

## Ensure the environments

```python
import rise_eo
import rise_eo.aml

cfg = rise_eo.config.load()

result = rise_eo.aml.ensure_environment(
    BASE, registry=cfg.image_registry,
    resource_group=cfg.resource_group, workspace=cfg.workspace,
)
print(result.ref, "reused" if result.reused else "just built")
```

**`alias=` defaults to `None`: a run moves no registry pointer.** Before spec 102 the default was
`"current"`, so any teammate's run repointed the shared `current` alias. Now only a call that asks
moves one. Shared aliases (`current`, `champion`, `demo-*`) are the maintainer's to move, from
`main`; a teammate working against the shared registry passes a personal one, `alias="dev-<you>"`,
or none at all and uses `result.ref` / `result.registry_ref` directly. The maintainer promotes with
the same call plus `alias="current"`; that works on a reuse as well as on a build.

In order (D4):

1. resolve the definition -- every moving reference fixed: a git ref pinned to a 40-char sha, a
   `path:` fsd hashed to its built wheel's content digest, the base image tag resolved to
   `@sha256:...` when the registry can answer;
2. digest the resolved definition and look it up in `registry=` by digest;
3. on a hit, confirm the AML asset it names **still exists** (`az ml environment show`) -- a
   registry entry whose asset was deleted is stale, and `ensure_environment` will not hand back a
   version that fails at job submission;
4. on a miss (no entry, or a deleted asset): render the context, `az ml environment create`,
   capture the version AML assigned;
5. publish the (possibly new) definition to `registry=` and, only if you passed `alias=`, point that
   alias at it.

**`ensure_environment` never waits for the build.** An AML v2 image build is an ACR task run, not
an AML job -- `result.build_url`, when set, is the Studio page; **you** watch it for
`Build status: Succeeded` before submitting anything against the environment. Nothing here reports
that programmatically (see Step 4's warning history, kept below).

**Re-running the same call is cheap and safe**, unlike the old `az ml environment create` flow --
a digest hit with a live asset builds nothing. Pass `force=True` to rebuild anyway (a base image
moved under a tag you did not pin).

## Wait for the ACR build ⚠️

**This is the step people skip, and it is the expensive one to skip.** `az ml environment create`
registers the *asset* and returns immediately; the image builds asynchronously in ACR (~10-20 min
each, occasionally flaky). The v2 `Environment` object carries **no build state whatsoever**, so
`ml_client.environments.get(name, version)` succeeds against a half-built image and every fsd
preflight goes green regardless. Submit early and jobs do not fail -- they sit in *Preparing*
until ACR is done, and that wait lands inside `job_admission_seconds`.

**This gate is manual, and there is no way around that.** Nothing reports it programmatically: the
`Environment` object has no build state; the build is an **ACR task run, not an AML job**, so it
does **not** appear in `az ml job list` (a poll loop built on that query printed `0/0 builds
terminal` for ten minutes while both builds completed normally in Studio, 2026-08-20); `az ml
environment show` proves the **asset** is registered, never that the image is built.

So open `result.build_url` (or `az ml environment show -n <name> -v <version> ...` and build the
Studio URL from the `WSID`/tenant it needs -- `rise_eo.aml.environment.build_link` does exactly that)
and read it. Wait for **Build status: Succeeded** on *both* images before running anything against
them.

## Use them

```python
runner_kwargs = dict(environment=result.ref, ...)

from rise_eo.model.verify_image import verify_image

vres = verify_image(
    bundle_dir, environment=result.ref, runner="aml", runner_kwargs=runner_kwargs,
    image_ref=result.registry_ref, registry=cfg.image_registry,   # spec 56 D8
)
assert vres["pass"], vres
```

**`result.ref` and `result.registry_ref` are different numbers and are not interchangeable.**
AML versions *assets*; the image registry versions *definitions*, in its own integer sequence —
`fsd-aml-env:5` in AML can be `fsd-aml-env:1` in the registry. `environment=` wants AML's
(`.ref`, `.version`); `image_ref=` wants the registry's (`.registry_ref`, `.registry_version`).
Swapping them fails as a missing `v<N>` directory, not as a type error, so name the field.

`verify_image`'s wheel-staleness gate (spec 47) now has two equivalent forms (D8):
`build_context=` for a checked-out build (unchanged since spec 47), and `image_ref=` +
`registry=` for anything built through `ensure_environment` -- it resolves the registry entry and
checks its resolved `fsd` reference the same way `build_context` checks a wheel. Pass whichever
matches how the image was built; `build_context` wins if both are given.

## Prefer to run it than read it?

[`notebooks/build_images.ipynb`](../../notebooks/build_images.ipynb) is this page as a
runnable notebook, and is **the one notebook this repo tracks**. It reads every Azure coordinate
(including `image_registry`) from `~/.config/fsd/config.toml` via `rise_eo.config.load()` rather than
carrying them, and `tests/test_notebooks.py` fails the build if it ever gains a saved output or a
hardcoded identifier -- that guard is why it can be public at all. Clear its outputs
(*Kernel -> Restart & Clear All Outputs*) before committing.

## Troubleshooting

**`ensure_environment` built when I expected a reuse.** Either the definition really changed (for
`fsd="path:..."`, an uncommitted edit changes the wheel's content digest -- there is no separate
git-dirty check, D5), or the registry entry's AML asset no longer exists and `ensure_environment`
correctly refused to hand back a dead reference.

**`az` returned something that is not a version.** `az ml` tries to auto-upgrade itself and cannot
on an AML compute instance: the extension there lives system-wide at `/opt/az/extensions/ml`,
owned by root, while you run as `azureuser`. It fails with `Permission denied` and leaves the
extension **half-deleted**, after which every `az ml` command breaks. Run these steps from your
laptop, or reinstall the extension with `sudo`.

**`verify_image` returns `wheel_has_spec44: False` / `image_ref_has_spec44: False`.** The image
was built from a pre-spec-44 fsd. Rebuild from a current checkout or a current git ref.

**A build failed.** Studio -> Environments -> the version -> build log. Calling
`ensure_environment` again retries the same digest and gets you the same version back --
fix the underlying problem (an unreachable base image, a broken `pip install`) and pass
`force=True` for a fresh attempt.
