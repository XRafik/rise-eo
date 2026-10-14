"""The image definition registry, mirroring `fsd.model.registry`'s layout:

    <registry>/
      <name>/
        _aliases.json          {"current": 7}
        v7/
          image.json           the resolved definition + the digest + the AML asset it became
          _complete.json       written last; the all-or-nothing marker
          _aml.json            [optional] the AML asset it currently is, if it was rebuilt
        v8/ ...

Built on `fsd.registry._core` -- version allocation, `_aliases.json`,
`_complete.json` and the collision retry are the same code `fsd.model.registry` runs on.

**The registry path is an argument, always**: every function here takes `registry=`;
nothing under `src/fsd/` reads it from config or an environment variable.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import NamedTuple

from fsd.image import digest as digest_mod
from fsd.registry import _core as core
from fsd.storage import fs

__all__ = [
    "Resolved", "Status", "find_by_digest", "publish", "read_aml_record", "resolve",
    "set_alias", "status", "write_aml_record",
]

DEFINITION_FILE = "image.json"
AML_FILE = "_aml.json"


class Resolved(NamedTuple):
    name: str
    version: int
    path: str
    digest: str
    definition: dict
    aml: dict | None


class Status(NamedTuple):
    state: str  # "registered" | "unregistered"
    digest: str
    registered: int | None
    reason: str


def _content_digest(path: str, storage_options: dict) -> str:
    """Re-digest what is actually at `path`: read its `image.json`'s `definition` field and
    recompute `digest.digest()` over it -- proves the landed bytes are the resolved
    definition they claim to be (the same role `fsd.model.registry.content_digest` plays,
    re-reading content rather than trusting a stored value)."""
    with fs.open(os.path.join(path, DEFINITION_FILE), "r", **storage_options) as f:
        record = json.load(f)
    return digest_mod.digest(record["definition"])


def find_by_digest(
    name: str, registry: str, digest: str, *, storage_options: dict | None = None,
) -> int | None:
    """The version whose content digest is `digest`, or `None`. Reads each version's
    `_complete.json` (small: `{"digest": ...}`), not its full `image.json` -- the same cost
    shape as `fsd.model.registry.publish`'s idempotency scan."""
    opts = storage_options or {}
    root = core.name_root(registry, name)
    for v in core.list_versions(root, opts):
        marker_path = os.path.join(core.version_path(registry, name, v), core.COMPLETE_FILE)
        try:
            with fs.open(marker_path, "r", **opts) as f:
                marker = json.load(f)
        except FileNotFoundError:
            continue
        if marker.get("digest") == digest:
            return v
    return None


def publish(
    name: str,
    registry: str,
    resolved: dict,
    digest: str,
    *,
    aml: dict | None = None,
    provenance: dict | None = None,
    alias: str | None = None,
    storage_options: dict | None = None,
) -> int:
    """Publish a resolved definition to `<registry>/<name>/`. Returns the version integer.

    Idempotent by digest, mirroring `fsd.model.registry.publish`: a version whose
    content digest already equals `digest` is returned and nothing is written. `aml` is
    the `{"name", "version", "workspace"}` the definition became;
    `provenance` carries the `org.opencontainers.image.*` fields, merged in verbatim.
    """
    core.check_name(name)
    opts = storage_options or {}
    root = core.name_root(registry, name)

    existing = find_by_digest(name, registry, digest, storage_options=opts)
    if existing is not None:
        if alias:
            core.set_alias(name, alias, existing, registry, opts)
        return existing

    record: dict = {"digest": digest, "definition": resolved}
    if aml is not None:
        record["aml"] = aml
    if provenance:
        record.update(provenance)
    body = json.dumps(record, indent=2, sort_keys=True).encode("utf-8")

    version = core.write_new_version(
        root, [(DEFINITION_FILE, body)], digest, opts,
        content_digest_fn=_content_digest,
    )
    if alias:
        core.set_alias(name, alias, version, registry, opts)
    return version


def set_alias(
    name: str, alias: str, version: int, registry: str, *, storage_options: dict | None = None,
) -> None:
    """Repoint `alias` at an already-published `version`. Never touches a version directory.
    `publish(alias=...)` does this on every publish; this is the same move without a publish,
    for a caller that found its definition already registered."""
    core.set_alias(name, alias, version, registry, storage_options or {})


def resolve(ref: str, registry: str, *, storage_options: dict | None = None) -> Resolved:
    """Turn `'fsd-aml-env:7'` or `'fsd-aml-env@current'` into its published record.

    `.aml` is the **current** AML asset for this definition: `_aml.json` if one is there,
    otherwise the `aml` block frozen into `image.json` when the version was first published.
    The sidecar wins because the definition is immutable and the asset it maps to is not --
    an asset deleted and rebuilt keeps the same digest, so it lands on the same version with
    a new AML version number (`write_aml_record`). See that function for why this is a
    separate file rather than an edit to `image.json`.
    """
    opts = storage_options or {}
    name, version, path = core.resolve_ref(ref, registry, opts)
    with fs.open(os.path.join(path, DEFINITION_FILE), "r", **opts) as f:
        record = json.load(f)
    aml = read_aml_record(name, version, registry, storage_options=opts) or record.get("aml")
    return Resolved(name, version, path, record["digest"], record["definition"], aml)


def read_aml_record(
    name: str, version: int, registry: str, *, storage_options: dict | None = None,
) -> dict | None:
    """`_aml.json` for one version, or `None` if it was never rewritten."""
    opts = storage_options or {}
    path = os.path.join(core.version_path(registry, name, version), AML_FILE)
    try:
        with fs.open(path, "r", **opts) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def write_aml_record(
    name: str, version: int, record: dict, registry: str, *, storage_options: dict | None = None,
) -> None:
    """Repoint a published definition at the AML asset it currently is (`_aml.json`).

    **Why a sidecar and not an edit to `image.json`:** a definition
    version is immutable and content-addressed -- `publish` is idempotent by digest, so a
    rebuild of the *same* definition (a deleted asset, or `force=True`) can never
    allocate a new version to record the new AML version in. Without this file the registry
    keeps pointing at the deleted asset, `ensure_environment` finds it missing on every
    subsequent call, and rebuilds a 10-20 minute image forever. Staged and renamed like
    `_aliases.json`, and outside `image.json`, so it never touches the content the digest is
    computed over -- exactly the role `_deploy.json` plays in `fsd.model.registry`.
    """
    opts = storage_options or {}
    vpath = core.version_path(registry, name, version)
    if not fs.exists(vpath, **opts):
        raise ValueError(f"cannot write AML record: {vpath!r} does not exist")
    stage = os.path.join(vpath, f"{core.STAGING_PREFIX}{uuid.uuid4().hex}.json")
    target = os.path.join(vpath, AML_FILE)
    try:
        fs.write_text(stage, json.dumps(record, indent=2, sort_keys=True), **opts)
        fs.rename(stage, target, **opts)
    except BaseException:
        core.discard(stage, opts)
        raise


def status(
    name: str, registry: str, digest: str, *, storage_options: dict | None = None,
) -> Status:
    """Do I need to build? A value, not a print: the caller (a notebook, `ensure_environment`)
    decides what to do with it."""
    version = find_by_digest(name, registry, digest, storage_options=storage_options)
    if version is None:
        return Status("unregistered", digest, None, f"no entry for {name} with this digest")
    return Status("registered", digest, version, f"{name}:{version} already holds this digest")
