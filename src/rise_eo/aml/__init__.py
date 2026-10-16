"""`rise_eo.aml` -- the AML builder for `rise_eo.image.ImageDefinition`.

Spec: specs/56-image-definitions-and-registry.md

`rise_eo.image` is backend-agnostic by construction; this module is not. It is the only
place that shells out to `az`, and it is check-then-build:

1. resolve + digest the definition (`rise_eo.image.digest`),
2. look it up in the registry by digest,
3. confirm the AML asset the registry points at still exists (a deleted asset is stale),
4. on a miss: render the context and `az ml environment create`,
5. publish the (possibly new) definition, record the AML asset it became, and set the
   alias (only when `alias=` is passed, on a reuse as well as on a build).

`ensure_environment` never waits for the build to finish: an AML v2 image build is an
ACR task run, not an AML job, so it returns the version and the Studio URL immediately.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from typing import Callable, NamedTuple

from rise_eo.aml import environment as env_mod
from rise_eo.image import digest as digest_mod
from rise_eo.image import registry as img_registry
from rise_eo.image.definition import ImageDefinition
from rise_eo.storage.azure import configure_storage as _configure_storage

__all__ = ["EnsureResult", "ensure_environment"]


class EnsureResult(NamedTuple):
    """**Two version numbers, deliberately both present.** `version`
    /`ref` are AML's -- what `az` assigned, the string every `environment=` argument wants.
    `registry_version`/`registry_ref` are the image registry's -- an independent integer
    sequence over *definitions*, and the only thing `rise_eo.model.verify_image(image_ref=...,
    registry=...)` can resolve. They are almost never equal, and passing one where the other
    belongs fails as a missing version directory rather than as a type error, so a caller
    should always name the field rather than reach for "the version"."""

    name: str
    version: str
    ref: str
    digest: str
    reused: bool
    build_url: str | None
    registry_version: int
    registry_ref: str


def _provenance(resolved: dict) -> dict:
    """`image.json`'s OCI-annotation-named fields, derived from the resolved
    definition rather than invented."""
    prov: dict = {"org.opencontainers.image.created": datetime.now(timezone.utc).isoformat()}
    fsd_ref = resolved.get("fsd")
    if fsd_ref and fsd_ref.startswith("git+"):
        url_part, sep, rev = fsd_ref.partition("@")
        prov["org.opencontainers.image.source"] = url_part[len("git+"):]
        if sep:
            prov["org.opencontainers.image.revision"] = rev
    base = resolved.get("base", "")
    if "@sha256:" in base:
        base_name, _, base_digest = base.partition("@")
        prov["org.opencontainers.image.base.name"] = base_name
        prov["org.opencontainers.image.base.digest"] = base_digest
    elif base:
        prov["org.opencontainers.image.base.name"] = base
    return prov


def ensure_environment(
    defn: ImageDefinition,
    *,
    registry: str,
    resource_group: str,
    workspace: str,
    force: bool = False,
    alias: str | None = None,
    storage: str | dict | None = None,
    storage_options: dict | None = None,
    resolve_base_digest: Callable[[str], str | None] | None = None,
    resolve_git_ref: Callable[[str, str], str] | None = None,
    _find_by_digest: Callable = img_registry.find_by_digest,
    _resolve: Callable = img_registry.resolve,
    _publish: Callable = img_registry.publish,
    _set_alias: Callable = img_registry.set_alias,
    _write_aml_record: Callable = img_registry.write_aml_record,
    _environment_exists: Callable = env_mod.environment_exists,
    _create_environment: Callable = env_mod.create_environment,
    _build_link: Callable = env_mod.build_link,
) -> EnsureResult:
    """Digest `defn`, reuse a matching registered environment if one still exists in AML,
    otherwise build and publish a new one. `force=True` rebuilds regardless of a digest hit
    (a base image moved under a tag you did not pin -- flytekit's `force_push()`).

    `alias=None` (the default) moves no registry pointer: not on a build, not on a reuse. Pass
    `alias="current"` (or any name) to point it at the resulting version, whether that was built
    or reused -- shared aliases such as `current` are the maintainer's to move (spec 102 D12).

    `storage="azure"` forbids the anonymous fallback for an `abfss://` registry, exactly as
    `deploy`/`run_inference`/`verify_image` do (this was the one
    public verb reaching the storage seam without configuring it). It is a **hardening, not a
    fix for a broken path** -- `adlfs` defaults to `anon=None`, which tries a credential
    first, so a registry read with `az login` in place already worked. What it buys is the
    failure mode: with `anon=False` a credential problem
    raises, instead of silently degrading to an anonymous read that returns nothing and looks
    exactly like "this definition is not registered yet" -- which would rebuild a 10-20 minute
    image on every call.

    The `_find_by_digest`/`_resolve`/`_publish`/`_set_alias`/`_write_aml_record`/`_environment_exists`/
    `_create_environment`/`_build_link` parameters are the seam tests stub -- override them to avoid a
    real registry or a real `az` call; production code never passes them.
    """
    # Before the first storage access -- `_find_by_digest` below.
    _configure_storage(storage)
    # One temp directory spans resolve AND build on purpose: for `fsd="path:..."` the digest
    # is of a wheel this function builds, and the image must be built from THAT wheel, not a
    # second one built moments later. For every other `fsd` form
    # nothing is written here until the build branch runs.
    with tempfile.TemporaryDirectory() as tmp:
        return _ensure(
            defn, tmp, registry=registry, resource_group=resource_group, workspace=workspace,
            force=force, alias=alias, storage_options=storage_options,
            resolve_base_digest=resolve_base_digest, resolve_git_ref=resolve_git_ref,
            _find_by_digest=_find_by_digest, _resolve=_resolve, _publish=_publish,
            _set_alias=_set_alias, _write_aml_record=_write_aml_record, _environment_exists=_environment_exists,
            _create_environment=_create_environment, _build_link=_build_link,
        )


def _ensure(
    defn: ImageDefinition,
    tmp: str,
    *,
    registry: str,
    resource_group: str,
    workspace: str,
    force: bool,
    alias: str | None,
    storage_options: dict | None,
    resolve_base_digest,
    resolve_git_ref,
    _find_by_digest: Callable,
    _resolve: Callable,
    _publish: Callable,
    _set_alias: Callable,
    _write_aml_record: Callable,
    _environment_exists: Callable,
    _create_environment: Callable,
    _build_link: Callable,
) -> EnsureResult:
    """`ensure_environment`'s body, with `tmp` (the shared wheel + build context directory)
    made explicit. Split out only so the `TemporaryDirectory` has one obvious lifetime."""
    resolved = digest_mod.resolve(
        defn, resolve_base_digest=resolve_base_digest, resolve_git_ref=resolve_git_ref,
        wheel_dir=tmp,
    )
    d = digest_mod.digest(resolved)

    if not force:
        hit = _find_by_digest(defn.name, registry, d, storage_options=storage_options)
        if hit is not None:
            record = _resolve(f"{defn.name}:{hit}", registry, storage_options=storage_options)
            aml = record.aml or {}
            aml_name, aml_version = aml.get("name", defn.name), aml.get("version")
            if aml_version is not None and _environment_exists(
                aml_name, aml_version, resource_group=resource_group, workspace=workspace,
            ):
                url = _build_link(
                    aml_name, aml_version, resource_group=resource_group, workspace=workspace,
                )
                # `publish` is the usual place an alias moves, and a reuse never reaches it:
                # an explicit ask must still work (promoting a teammate's build to `current`).
                if alias:
                    _set_alias(defn.name, alias, hit, registry, storage_options=storage_options)
                return EnsureResult(
                    aml_name, aml_version, f"{aml_name}:{aml_version}", d, True, url,
                    hit, f"{defn.name}:{hit}",
                )
            # the registry entry's asset is gone (or never had one) -- fall through to build

    # `write_context` reuses the wheel `resolve` already built into `tmp` -- same file, same
    # digest as the one the registry is about to record.
    context_dir = defn.build_context if defn.build_context is not None else defn.write_context(tmp)
    version = _create_environment(
        defn.name, context_dir, resource_group=resource_group, workspace=workspace,
    )

    aml_record = {"name": defn.name, "version": version, "workspace": workspace}
    published_version = _publish(
        defn.name, registry, resolved, d,
        aml=aml_record, provenance=_provenance(resolved), alias=alias,
        storage_options=storage_options,
    )
    # `publish` is idempotent by digest, so a rebuild of an UNCHANGED definition (a deleted
    # asset, or force=True) returns the version that already exists and writes no image.json
    # -- leaving the registry pointing at the AML version we just replaced. Without this the
    # next call finds that asset missing again and rebuilds forever.
    # `_aml.json` is a mutable sidecar, so it cannot disturb the content digest.
    _write_aml_record(
        defn.name, published_version, aml_record, registry, storage_options=storage_options,
    )
    url = _build_link(defn.name, version, resource_group=resource_group, workspace=workspace)
    return EnsureResult(
        defn.name, version, f"{defn.name}:{version}", d, False, url,
        published_version, f"{defn.name}:{published_version}",
    )
