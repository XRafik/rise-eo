"""The model registry: a name for a bundle, on the storage seam.

    <registry>/
      <name>/
        _aliases.json          {"champion": 3, "current": 4}
        v1/  bundle.json, code/, artifacts..., _complete.json   [+ _deploy.json, step 2]
        v2/  ...

Everything reads and writes through `rise_eo.storage.fs`, so a local registry and a blob
registry are the same code. Versions are immutable; `_aliases.json` is the only mutable
pointer. `publish` is idempotent by content digest. A version is published **in
place** -- no staging prefix, no directory rename -- and re-digested before
`v<N>/_complete.json` is written last, which is the all-or-nothing moment a directory
rename used to provide. `migrate` relocates a registry and re-digests every marked version,
so a ref to a marked version resolves identically against a copy of it.

**Concurrency, stated rather than implied.** There is no lock. What IS
guaranteed: a reader never sees a half-written `_aliases.json` (staged and renamed), and
`publish` never returns a version whose bytes are not the ones it was given -- it
re-digests what landed before marking it complete, and retries at `v<N+1>` on a genuine
collision. What is NOT guaranteed: two concurrent `set_alias` calls can lose an update,
two concurrent `publish` calls can leave a gap in the version sequence, and two
simultaneous publishers writing the same `v<N>` can interleave their files -- a publisher
whose re-digest then fails leaves the directory unmarked, so the damage is a stranded
directory rather than a bad model being served. See
`rise_eo.registry._core.write_new_version`.

`resolve`/`publish` are the interface; the storage-seam layout is v1's only backend, but
a second one can be added without either signature changing.

Spec: specs/51-deploy-model-registry.md, specs/52-registry-on-blob.md
"""

from __future__ import annotations

import json
import os
import re
from typing import NamedTuple

from rise_eo.model import bundle as bundle_mod
from rise_eo.registry import _core as core
from rise_eo.registry._files import STAGING_PREFIX, write_json_staged
from rise_eo.registry._files import digest_files as _digest_of
from rise_eo.storage import fs

__all__ = [
    "Resolved",
    "check_name",
    "content_digest",
    "migrate",
    "parse_ref",
    "publish",
    "read_deploy_record",
    "resolve",
    "set_alias",
    "version_path",
    "write_deploy_record",
]

ALIASES_FILE = core.ALIASES_FILE
DEPLOY_FILE = "_deploy.json"
_COMPLETE_FILE = core.COMPLETE_FILE
_STAGING_PREFIX = STAGING_PREFIX
_discard = core.discard

_REF_RE = re.compile(r"^(?P<name>[^:@]+)(?P<sep>[:@])(?P<value>.+)$")
_VERSION_ALIAS_RE = re.compile(r"^v(\d+)$")


class Resolved(NamedTuple):
    name: str
    version: int
    path: str


# --- ref parsing --------------------------------------------------------------


def parse_ref(ref: str) -> tuple[str, str, str]:
    """`'crop-rf:3'` -> `('crop-rf', ':', '3')`; `'crop-rf@champion'` -> `('crop-rf', '@',
    'champion')`. A bare name (no `:` or `@`) raises, naming both forms -- a default that
    silently picked one would be the failure mode aliases exist to make visible."""
    m = _REF_RE.match(ref)
    if not m:
        raise ValueError(
            f"model ref {ref!r} is ambiguous: use 'name:version' (e.g. 'crop-rf:3') or "
            "'name@alias' (e.g. 'crop-rf@champion')."
        )
    return m.group("name"), m.group("sep"), m.group("value")


def check_name(name: str) -> None:
    """Refuse a model name that cannot round-trip through a ref -- `deploy` returns a ref
    `run_inference` must accept unchanged.

    `deploy` returns `f"{name}:{version}"`, and the reader of that string is `api._is_ref_shaped`
    -> `parse_ref`. A name carrying `/` or `\\` makes the ref look like a *path*, so it is never
    resolved and dies as `FileNotFoundError: crop/rf:1/bundle.json`; a name carrying `:` or `@`
    re-splits at the wrong place (`'crop:rf:1'` -> name `'crop'`, version `'rf:1'`). Both would
    publish successfully and hand back a ref nothing can resolve -- so they are refused here,
    where the name is chosen, rather than surfacing at the first run.

    A leading `.` is refused for a different reason: `_list_names` skips dot-entries (that is
    how `.staging-*` stays invisible), so such a name would be silently dropped by `migrate`
    -- a "move" that quietly loses a model is worse than a refused name."""
    if not name or not name.strip():
        raise ValueError("model name must be a non-empty string.")
    bad = [c for c in ("/", "\\", ":", "@") if c in name]
    if bad:
        raise ValueError(
            f"model name {name!r} contains {bad!r}, which a ref cannot carry: deploy returns "
            f"'{name}:<version>' and that string would parse as a path (/, \\) or split at the "
            "wrong separator (:, @), so nothing could resolve it. Use letters, digits, '-' or "
            "'_' (e.g. 'crop-rf')."
        )
    if name.startswith("."):
        raise ValueError(
            f"model name {name!r} starts with '.': the registry treats dot-entries as internal "
            "(staging files), so this name would be invisible to migrate. Pick another."
        )


version_path = core.version_path


def resolve(ref: str, registry: str, *, storage_options: dict | None = None) -> Resolved:
    """Turn a ref into the version it names. Costs one `_aliases.json` read for a named
    alias, and nothing for a version pin (`name:N` or the `name@vN` shorthand) -- pinning
    a version never lists or reads anything, which keeps N nodes each resolving a
    model off the hot path."""
    name, sep, value = parse_ref(ref)
    if sep == ":":
        if not value.isdigit():
            raise ValueError(f"model ref {ref!r}: version must be an integer, got {value!r}")
        version = int(value)
    else:
        m = _VERSION_ALIAS_RE.match(value)
        if m:
            version = int(m.group(1))
        else:
            aliases = core.read_aliases(registry, name, storage_options or {})
            if value not in aliases:
                raise ValueError(
                    f"alias {value!r} not found for {name!r} in {registry!r} "
                    f"(known aliases: {sorted(aliases)})"
                )
            version = aliases[value]
    return Resolved(name, version, version_path(registry, name, version))


def set_alias(
    name: str, alias: str, version: int, registry: str, *, storage_options: dict | None = None,
) -> None:
    """Repoint `alias` at `version`. Never touches a version directory.

    Refuses an alias shaped `v<digits>`: that spelling is reserved for the `name@vN`
    version-pin shorthand, so an alias with that name could never be reached by
    `resolve` -- it would silently pin the literal version number instead.

    The new `_aliases.json` is staged and renamed rather than written in place, because
    `resolve` reads that one file and a fan-out resolving `@champion` while someone
    promotes must never read a half-written one. Two *concurrent* `set_alias` calls can
    still lose an update -- the registry has no lock -- but no reader ever
    sees a torn file.
    """
    core.set_alias(name, alias, version, registry, storage_options or {})


# --- content digest ------------------------------------------------------------


def _bundle_content_rels(manifest: dict) -> list[str]:
    """The file set the content digest is computed over: the manifest itself, every artifact,
    every embedded code file -- exactly what `bundle.load` needs, and nothing a caller
    happened to drop beside it."""
    rels = {bundle_mod.BUNDLE_MANIFEST}
    rels.update(manifest.get("artifacts", {}).values())
    rels.update(bundle_mod.manifest_code_files(manifest))
    return sorted(rels)


def _read_bundle_content(base_path: str, storage_options: dict | None) -> list[tuple[str, bytes]]:
    """`(relative path, bytes)` for every manifest-declared file under `base_path` --
    a fresh `bundle.save` output and an already-published version directory have the same
    shape, so this reads either."""
    opts = storage_options or {}
    with fs.open(os.path.join(base_path, bundle_mod.BUNDLE_MANIFEST), "r", **opts) as f:
        manifest = json.load(f)
    rels = _bundle_content_rels(manifest)
    return [
        (rel, _read_bytes(os.path.join(base_path, rel), opts))
        for rel in rels
    ]


def _read_bytes(path: str, storage_options: dict) -> bytes:
    with fs.open(path, "rb", **storage_options) as f:
        return f.read()


def content_digest(bundle_path: str, *, storage_options: dict | None = None) -> str:
    """SHA-256 over a bundle's manifest-declared files (sorted relative path + bytes).

    Two bundles with this equal are the same content. Used by `publish` for idempotency and
    by `migrate` to prove a copy arrived intact."""
    return _digest_of(_read_bundle_content(bundle_path, storage_options))


# --- the deploy record -----------------------------------------------------------


def read_deploy_record(version_dir: str, *, storage_options: dict | None = None) -> dict | None:
    """`version_dir`'s `_deploy.json` as a dict, or `None` if there is no such file (a version
    published before step 2, or by a caller that never called `deploy`) -- and also `None` for
    an empty or malformed file, since a print-line degrading to "no record" is the correct
    failure mode: a ref that resolved must never fail to run because its bookkeeping record
    was unreadable.

    **Never raises**, and the catch is deliberately wide enough to mean it. `ValueError` covers
    both `json.JSONDecodeError` *and* `UnicodeDecodeError` -- a truncated or byte-corrupted
    record decodes as neither JSON nor UTF-8, and catching only `JSONDecodeError` let that one
    escape and kill a run whose model had already resolved. `OSError` covers absent (its
    `FileNotFoundError` subclass), unreadable, and a backend that is briefly unhappy; degrading
    to "no record" is right for all of them, because the record is bookkeeping, never the
    thing being run. A non-dict payload (a bare list, a bare string) is `None` too."""
    opts = storage_options or {}
    path = os.path.join(version_dir, DEPLOY_FILE)
    try:
        with fs.open(path, "r", **opts) as f:
            record = json.load(f)
    except (ValueError, OSError):
        return None
    return record if isinstance(record, dict) else None


def _read_deploy_digest(version_dir: str, storage_options: dict) -> str | None:
    """The `digest` field of `version_dir`'s `_deploy.json`, or `None` if there is no such
    file (a version published before step 2, or by a caller that never called `deploy`).
    `publish`'s idempotency loop reads this first (a small metadata read) and falls back to
    recomputing the content digest only when it is absent."""
    record = read_deploy_record(version_dir, storage_options=storage_options)
    return record.get("digest") if record is not None else None


def write_deploy_record(
    name: str, version: int, record: dict, registry: str, *, storage_options: dict | None = None,
) -> None:
    """Write `_deploy.json` beside `bundle.json` in `<registry>/<name>/v<version>/` --
    the binding between this version, the image that was proven to run it, and the
    verification result. Staged and renamed like `_aliases.json` (`set_alias`), so a reader
    never observes a half-written record; writing it does not touch the version's
    manifest-declared content, so it never changes the content digest."""
    opts = storage_options or {}
    vpath = version_path(registry, name, version)
    if not fs.exists(vpath, **opts):
        raise ValueError(f"cannot write deploy record: {vpath!r} does not exist")
    write_json_staged(vpath, DEPLOY_FILE, record, opts)


# --- publish --------------------------------------------------------------------


def _list_names(registry: str, storage_options: dict) -> list[str]:
    if not fs.exists(registry, **storage_options):
        return []
    names = []
    for entry in fs.ls(registry, **storage_options):
        leaf = entry.rstrip("/").rsplit("/", 1)[-1]
        if leaf.startswith("."):
            continue
        names.append(leaf)
    return sorted(names)


def _write_new_version(
    name_root: str, files: list[tuple[str, bytes]], digest: str, storage_options: dict,
) -> int:
    """See `rise_eo.registry._core.write_new_version`; this passes the bundle content digest."""
    return core.write_new_version(
        name_root, files, digest, storage_options,
        content_digest_fn=lambda path, opts: content_digest(path, storage_options=opts),
    )


def publish(
    bundle_path: str,
    name: str,
    registry: str,
    *,
    alias: str | None = None,
    storage_options: dict | None = None,
    bundle_storage_options: dict | None = None,
) -> int:
    """Publish the bundle at `bundle_path` to `<registry>/<name>/`. Returns the version
    integer.

    Idempotent by content digest: if a version with identical content already
    exists, it is returned and nothing is written -- `publish` is safe to call again
    from a re-run notebook cell. Otherwise the next integer version is allocated and
    published in place, marked complete only once its bytes are re-digested and confirmed.

    `storage_options` reaches the registry; `bundle_storage_options` reaches
    `bundle_path` (only needed when the bundle itself is not local).

    Cost: the idempotency check reads each existing version's stored digest from its
    `_deploy.json` (a small metadata read) and falls back to recomputing the content
    digest only for a version that has none -- e.g. one published before step 2 landed
    `_deploy.json`, or by a direct `publish` call that was never `deploy`ed. `publish`
    itself never writes `_deploy.json` -- that is `deploy`'s job; this is just where it reads
    one if a prior `deploy` left it behind.
    """
    check_name(name)
    files = _read_bundle_content(bundle_path, bundle_storage_options)
    digest = _digest_of(files)

    opts = storage_options or {}
    name_root = core.name_root(registry, name)
    for v in core.list_versions(name_root, opts):
        vpath = version_path(registry, name, v)
        existing_digest = _read_deploy_digest(vpath, opts)
        if existing_digest is None:
            existing_digest = content_digest(vpath, storage_options=storage_options)
        if existing_digest == digest:
            if alias:
                set_alias(name, alias, v, registry, storage_options=storage_options)
            return v

    version = _write_new_version(name_root, files, digest, opts)
    if alias:
        set_alias(name, alias, version, registry, storage_options=storage_options)
    return version


# --- migrate ---------------------------------------------------------------------


def migrate(
    src_registry: str,
    dst_registry: str,
    *,
    src_storage_options: dict | None = None,
    dst_storage_options: dict | None = None,
) -> None:
    """Copy a registry tree onto a new root, re-digesting every marked version.

    Not a schema rewrite: nothing the registry writes names its own location, so a ref that
    resolved against `src_registry` resolves identically against `dst_registry` once callers
    pass the new `registry=`. That invariant is what makes a move "a copy plus a changed
    argument" -- do not write an absolute registry path into any registry file. A version
    whose recomputed digest disagrees with what was copied is refused, not silently accepted.
    An unmarked version is not copied, but `_aliases.json` is, so a pin or alias naming one
    resolves against `src_registry` and dangles against `dst_registry`.
    """
    src_opts = src_storage_options or {}
    dst_opts = dst_storage_options or {}
    for name in _list_names(src_registry, src_opts):
        name_src = core.name_root(src_registry, name)
        name_dst = core.name_root(dst_registry, name)

        aliases_src = os.path.join(name_src, ALIASES_FILE)
        if fs.exists(aliases_src, **src_opts):
            with fs.open(aliases_src, "r", **src_opts) as f:
                raw = f.read()
            fs.write_text(os.path.join(name_dst, ALIASES_FILE), raw, **dst_opts)

        for version in core.list_versions(name_src, src_opts):
            _migrate_version(
                os.path.join(name_src, f"v{version}"),
                os.path.join(name_dst, f"v{version}"),
                src_opts, dst_opts,
            )


def _migrate_version(vsrc: str, vdst: str, src_opts: dict, dst_opts: dict) -> None:
    files = _read_bundle_content(vsrc, src_opts)
    before = _digest_of(files)
    for rel, data in files:
        fs.write_bytes(os.path.join(vdst, rel), data, **dst_opts)
    after = content_digest(vdst, storage_options=dst_opts)
    if after != before:
        raise ValueError(
            f"migrate: {vsrc} -> {vdst} digest mismatch after copy (source {before}, "
            f"copied {after}) -- refusing a possibly-corrupted copy."
        )
    # Marked last, same as `_write_new_version`: migrate has already written and
    # re-digest-confirmed above, and `core.list_versions` shows only marked versions.
    fs.write_text(
        os.path.join(vdst, _COMPLETE_FILE),
        json.dumps({"digest": after}, indent=2, sort_keys=True),
        **dst_opts,
    )
    # `_deploy.json` is not manifest-declared content, so it never affects the digest above
    # -- but it is the durable record that a `deploy` happened. Dropping it on a relocation
    # would silently lose every version's binding record.
    deploy_src = os.path.join(vsrc, DEPLOY_FILE)
    if fs.exists(deploy_src, **src_opts):
        with fs.open(deploy_src, "r", **src_opts) as f:
            raw = f.read()
        fs.write_text(os.path.join(vdst, DEPLOY_FILE), raw, **dst_opts)
