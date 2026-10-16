"""`rise_eo.model.verify_image` — does an inference IMAGE actually run THIS bundle?

Spec: specs/45-bundle-transparency-and-image-verification.md (#67)

An inference image is generic per *dependency family* (sklearn/xgboost/torch/keras), never per
model, so "does this image run this bundle?" is a question worth asking as a function call.

1. driver-side first, free: manifest is v2, a `code` block exists, `check_requirements` against
   the declared list, and — when `build_context` is given — the wheel-staleness gate that
   refuses a too-old image in ~2 s instead of paying for a 40-380 s cold start;
2. stage the bundle exactly as `run_aml_inference` does (`runners._stage_bundle`);
3. submit **one node** running `python -m rise_eo.workflows.adapter_smoke`;
4. always read `_status/*.json` back, and treat a *missing* status file as its own diagnosis:
   the job died before the entrypoint ran, so it is the image or node auth, not the adapter.

⚠️ Three properties are non-negotiable, each of them a defect already paid for once: it must
run as a JOB, never on the driver, because a local run passes trivially; the returned dict is
`_result.json`-shaped so a run-book can paste it straight back; and it is meant to be called at
the step it protects, immediately before `run_inference`, not hoisted into an upfront gate that
hardcodes paths a later step creates.
"""

from __future__ import annotations

import glob
import json
import os
import urllib.request
import uuid
import zipfile

from rise_eo.image import registry as _image_registry
from rise_eo.model import bundle as _bundle
from rise_eo.model import registry as _registry
from rise_eo.storage import fs
from rise_eo.storage.azure import configure_storage as _configure_storage

__all__ = ["verify_image"]


def _default_fetch_rise_eo_source(rise_eo_ref: str, relpath: str) -> str:
    """`relpath` of the rise-eo source tree named by `rise_eo_ref` (a resolved `git+...@<sha>`).

    Read via GitHub's raw-content host, since rise-eo is public there and this avoids a local
    clone for a one-file check. Raises on anything unexpected; the caller turns that into a
    `pass=False` result rather than a crash -- the gate must keep failing, never become
    unable to report."""
    url_part, _, sha = rise_eo_ref.partition("@")
    repo_url = url_part[len("git+"):].removesuffix(".git")
    if not repo_url.startswith("https://github.com/"):
        # Said plainly rather than mis-fetched: `removeprefix` on a non-GitHub URL is a no-op,
        # so the raw URL would come out as `raw.githubusercontent.com/https://gitlab.com/...`
        # and 404 -- a confusing pass=False instead of "this gate only reads GitHub".
        raise ValueError(
            f"verify_image cannot check {rise_eo_ref!r}: the image_ref= gate reads rise-eo's source "
            "from raw.githubusercontent.com, so it only understands a "
            "'git+https://github.com/...' reference. Pass build_context= instead."
        )
    repo_path = repo_url.removeprefix("https://github.com/")
    raw_url = f"https://raw.githubusercontent.com/{repo_path}/{sha}/{relpath}"
    with urllib.request.urlopen(raw_url, timeout=10) as resp:  # noqa: S310 - fixed https scheme
        return resp.read().decode()


def _image_ref_has_spec44(rise_eo_ref: str, *, fetch=_default_fetch_rise_eo_source) -> bool:
    """The `image_ref=` equivalent of `_wheel_has_spec44`: does the rise-eo this image was built
    from already carry `manifest_code_files`?

    A `wheel:<digest>` reference carries no fetchable source, only its content digest, so it
    is trusted rather than checked -- a developer's own `path:` build was always their
    responsibility.

    ⚠️ **A non-`git+` reference is likewise trusted, and that is a HOLE in this gate.** A
    PyPI spec (`rise_eo==0.2.0`) names a version this function has no cheap way to read
    `bundle.py` out of, so it passes unchecked. Harmless while rise-eo is not on PyPI (#82), and
    the resolved reference is recorded in `metrics["image_ref_rise_eo"]` either way -- but it
    must be closed, by reading the sdist's `bundle.py`, before a PyPI-installed image is
    verified this way."""
    if rise_eo_ref.startswith("wheel:"):
        return True
    if not rise_eo_ref.startswith("git+"):
        return True
    src = fetch(rise_eo_ref, "src/rise_eo/model/bundle.py")
    return "def manifest_code_files" in src


def _find_wheel(build_context: str) -> str:
    """A caller who passes `build_context` has asserted the folder holds the wheel the image was
    built from. An absent wheel is a statement about the CALL, not the image, so
    this raises rather than returning a verdict -- it must run before `verify_image`'s `try`."""
    wheels = sorted(glob.glob(os.path.join(build_context, "rise_eo-*.whl")))
    if not wheels:
        raise ValueError(f"build_context={build_context!r} contains no rise_eo-*.whl.")
    return wheels[-1]


def _wheel_has_spec44(wheel: str) -> bool:
    """The wheel-staleness gate: does the rise-eo wheel this image was built from already carry
    `manifest_code_files`?

    A wheel too old to know about embedded code never downloads `code/` and never touches
    `sys.path`, so the node raises `ModuleNotFoundError` however good the bundle is -- and it
    cannot self-diagnose, because an old rise-eo has none of the code that would report it. That
    is why this is checked here, on the driver.

    Unlike an absent wheel, a stale one IS a statement about the image, so it stays inside the
    `try` and returns a verdict rather than raising."""
    with zipfile.ZipFile(wheel) as zf:
        src = zf.read("rise_eo/model/bundle.py").decode()
    return "def manifest_code_files" in src


def verify_image(
    bundle_path: str,
    *,
    environment: str,
    runner: str = "aml",
    runner_kwargs: dict | None = None,
    build_context: str | None = None,
    image_ref: str | None = None,
    registry: str | None = None,
    storage=None,
    _fetch_rise_eo_source=_default_fetch_rise_eo_source,
) -> dict:
    """Does `environment` actually run `bundle_path`? Submits one real node and reports.

    `bundle_path` is a local bundle folder or an already-staged URL -- `storage="azure"`
    authenticates adlfs for that read (#86), the same `storage=` kwarg
    `run_inference`/`verify_adapter`/`deploy` take. `environment` is the inference
    Environment reference to verify (e.g. `"rise-eo-infer-sklearn:3"`).

    `runner` must be `"aml"` -- `runner="local"` **raises** rather than returning a pass,
    because "verified locally" is the exact false positive this helper exists to prevent: the
    driver's venv already has the adapter's source on `sys.path` and its dependencies
    installed, so a local run tells you nothing about the image.

    `runner_kwargs` carries the AML dispatch parameters, the same names `run_aml_inference` takes:
    required `cluster`, `root`, `identity_client_id`; optional `ml_client` (inject one for
    tests, otherwise a real `MLClient` is built from `subscription_id`/
    `resource_group_name`/`workspace_name`), `poll_interval_seconds` (default 30), and `run_id`
    (an opaque tag under `root` that namespaces the staged bundle + status file for this call --
    defaults to a fresh uuid; pass one explicitly to make the status URL predictable).

    `build_context`, if given, is the folder holding the rise-eo wheel the image was built from --
    enables the wheel-staleness gate (see `_wheel_has_spec44`). `image_ref`/`registry`
    are the alternative: an image built by `rise_eo.aml.ensure_environment` names no
    checkout folder, so instead `image_ref` (e.g. `"rise-eo-infer-sklearn:4"`) is resolved through
    the image `registry` and its resolved `rise-eo` reference is checked the same way a wheel is.
    `build_context` wins if both are given; neither is required.

    Returns a `_result.json`-shaped dict: `{"step", "status", "pass", "metrics",
    "expected", "error"}`. `metrics["bundle_digest"]` records the content digest of what was
    verified, which is what `rise_eo.deploy(verified=...)` matches against. Every
    driver-detectable failure (no `code` block, a stale wheel, a partial stage, a missing
    node status file) sets `pass=False` with a populated `error`.
    `verify_image` raises only on caller misuse it cannot report as a verification result --
    a non-`"aml"` `runner`, or `runner_kwargs` missing `cluster`/`root`/`identity_client_id`.
    """
    if runner != "aml":
        raise ValueError(
            f"verify_image(runner={runner!r}) is not a verification -- only 'aml' actually runs "
            "the bundle inside the target image on a real node. A local run passes trivially "
            "because the driver already has the adapter's source on sys.path and its "
            "dependencies installed (ADR 0002), which is the exact false positive this helper "
            "exists to prevent."
        )
    kwargs = dict(runner_kwargs or {})
    missing_kwargs = [k for k in ("cluster", "root", "identity_client_id") if not kwargs.get(k)]
    if missing_kwargs:
        raise ValueError(
            f"verify_image(runner='aml') requires runner_kwargs{sorted(missing_kwargs)!r}."
        )
    # Before the first storage access -- `_bundle.read_spec(bundle_path)` below.
    _configure_storage(storage)
    # A caller who passes build_context has asserted the folder holds the wheel, so an absent
    # wheel is caller misuse: it must raise here, before the try, never become pass=False.
    wheel = _find_wheel(build_context) if build_context else None
    # build_context wins if both are given, so the checkout path behaves unchanged. A caller
    # who passes image_ref without registry, or a ref the registry cannot resolve, is misuse
    # -- same treatment as an absent wheel.
    image_ref_rise_eo = None
    prerename_image = False
    if image_ref and not build_context:
        if not registry:
            raise ValueError("verify_image(image_ref=...) requires registry=.")
        definition = _image_registry.resolve(image_ref, registry).definition
        prerename_image = "rise_eo" not in definition and "fsd" in definition
        image_ref_rise_eo = definition.get("rise_eo")

    result: dict = {
        "step": "verify_image",
        "status": "ok",
        "pass": False,
        "metrics": {},
        "expected": {"bundle_version": _bundle.BUNDLE_VERSION, "code_block_present": True,
                     "smoke_status": "ok"},
        "error": None,
    }
    try:
        # --- driver-side, free -----------------------------------------------------------
        manifest = _bundle.read_spec(bundle_path)   # model-free: no import, no model load
        version = manifest.get("fsd_bundle_version")
        code = manifest.get("code")
        result["metrics"].update({
            "bundle_path": str(bundle_path),
            # WHAT was verified, not WHERE it was. `deploy(verified=...)`
            # honours a prior result only if this digest matches the bundle being deployed;
            # re-digesting `bundle_path` at deploy time instead would prove nothing, because
            # `bundle.save` overwrites in place -- the same path can hold
            # different content on different days, and a same-path comparison is a tautology.
            # Recording it here also makes a `_result.json` portable between machines.
            "bundle_digest": _registry.content_digest(bundle_path),
            "bundle_version": version,
            "adapter_ref": manifest.get("adapter"),
            "code_block_present": bool(code),
            "code_files": (code or {}).get("files"),
            "requirements": manifest.get("requirements"),
            "environment": environment,
        })
        if not code:
            raise ValueError(
                f"{bundle_path} is a version-{version} bundle with no `code` block, so it does "
                "not carry its adapter -- on a generic image this WILL fail with "
                "ModuleNotFoundError. Re-save it with rise_eo.model.bundle.save so the adapter's "
                "source is embedded."
            )
        # Informational, not fatal: the driver's venv need not mirror the image. The real
        # check is `check_requirements` running INSIDE the smoke job, below.
        result["metrics"]["requirement_problems_here"] = _bundle.check_requirements(
            manifest.get("requirements"))

        if wheel:
            fresh = _wheel_has_spec44(wheel)
            result["metrics"]["build_context_wheel"] = wheel
            result["metrics"]["wheel_has_spec44"] = fresh
            if not fresh:
                raise ValueError(
                    f"{wheel} predates spec 44 -- it has no `manifest_code_files`, so the "
                    "image's fetch_bundle_to_scratch will not download code/ and bundle.load "
                    "will not put it on sys.path. Rebuild the wheel and the image, then re-run "
                    "verify_image."
                )

        if prerename_image:
            raise ValueError(
                f"{image_ref} was built before the package was renamed to rise-eo, so it cannot run "
                "this code. Rebuild it (docs/howto/build-the-images.md), then re-run verify_image."
            )

        if image_ref_rise_eo is not None:
            fresh = _image_ref_has_spec44(image_ref_rise_eo, fetch=_fetch_rise_eo_source)
            result["metrics"]["image_ref"] = image_ref
            result["metrics"]["image_ref_rise_eo"] = image_ref_rise_eo
            result["metrics"]["image_ref_has_spec44"] = fresh
            if not fresh:
                raise ValueError(
                    f"{image_ref} was built from {image_ref_rise_eo!r}, which predates spec 44 -- "
                    "it has no `manifest_code_files`, so the image's fetch_bundle_to_scratch "
                    "will not download code/ and bundle.load will not put it on sys.path. "
                    "Rebuild the image from a current rise-eo, then re-run verify_image."
                )

        # --- stage + submit ----------------------------------------------------------------
        from rise_eo.workflows import (
            runners,  # lazy: keeps rise_eo.model import-light (no azure at import time)
        )

        ml_client = kwargs.get("ml_client")
        if ml_client is None:
            from azure.ai.ml import MLClient
            from azure.identity import DefaultAzureCredential

            ml_client = MLClient(
                DefaultAzureCredential(), kwargs.get("subscription_id"),
                kwargs.get("resource_group_name"), kwargs.get("workspace_name"),
            )

        root = str(kwargs["root"]).rstrip("/")
        # A deterministic tag can be supplied via runner_kwargs["run_id"] -- lets a caller (or a
        # test, injecting a fake ml_client) predict status_url ahead of the call; otherwise a
        # fresh one per call, same as run_aml_inference's own run_id.
        run_tag = str(kwargs.get("run_id") or uuid.uuid4().hex[:8])
        staged = runners._stage_bundle(bundle_path, f"{root}/_verify_image/{run_tag}/bundle")
        result["metrics"]["staged_bundle_url"] = staged

        # Prove the code files actually landed -- the transport half of the change.
        staged_code = _bundle.manifest_code_files(manifest)
        landed = [rel for rel in staged_code if fs.exists(f"{staged}/{rel}")]
        result["metrics"]["code_files_staged"] = f"{len(landed)}/{len(staged_code)}"
        if len(landed) != len(staged_code):
            raise ValueError(
                f"only {len(landed)}/{len(staged_code)} code files reached {staged}: "
                f"missing {[r for r in staged_code if r not in landed]}"
            )

        status_url = f"{root}/_verify_image/{run_tag}/_status/smoke.json"
        aml_command = runners._import_aml_command()
        job = aml_command(
            command=f"python -m rise_eo.workflows.adapter_smoke {staged} --status-url {status_url}",
            environment=environment,
            compute=kwargs["cluster"],
            environment_variables={"AZURE_CLIENT_ID": kwargs["identity_client_id"]},
            display_name=f"rise-eo-verify-image-{run_tag}",
            experiment_name="rise-eo-verify-image",
        )
        result["metrics"]["status_url"] = status_url
        submit_error = None
        try:
            runners._aml_submit_and_wait(
                ml_client, {"smoke": job}, root, f"verify-image-{run_tag}",
                poll_interval_seconds=kwargs.get("poll_interval_seconds", 30),
            )
        except Exception as exc:  # noqa: BLE001 - the useful error lives on status_url, not here
            submit_error = str(exc)

        if fs.exists(status_url):
            with fs.open(status_url, "r") as f:
                smoke = json.load(f)
            result["metrics"]["smoke_status"] = smoke.get("status")
            result["metrics"]["smoke_error"] = smoke.get("error")
        else:
            # No status file => the job died BEFORE the entrypoint ran (bad image, rise-eo not
            # installed, blob auth). adapter_smoke always writes its status before exiting, so
            # its absence is itself the diagnosis.
            result["metrics"]["smoke_status"] = "no status file written"
            result["metrics"]["smoke_error"] = (
                "the job failed before `python -m rise_eo.workflows.adapter_smoke` could write its "
                "status. That points at the IMAGE or node auth, not the adapter: rise-eo not "
                "installed in the environment, a broken entrypoint, or the node failing to reach "
                "blob (check the compute identity's RBAC)."
            )
        if submit_error:
            result["metrics"]["driver_error"] = submit_error

        result["pass"] = (
            result["metrics"]["smoke_status"] == "ok" and version == _bundle.BUNDLE_VERSION
        )
        result["status"] = "ok" if result["pass"] else "fail"
    except Exception as exc:  # noqa: BLE001 - always a shaped result, never a bare traceback
        result["status"] = "fail"
        result["pass"] = False
        result["error"] = str(exc)
    return result
