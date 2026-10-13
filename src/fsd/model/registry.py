"""The model registry: a name for a bundle, on the storage seam.

    <registry>/
      <name>/
        _aliases.json          {"champion": 3, "current": 4}
        v1/  bundle.json, code/, artifacts..., _complete.json   [+ _deploy.json, step 2]
        v2/  ...

Everything reads and writes through `fsd.storage.fs`, so a local registry and a blob
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
directory rather than a bad model being served. See `_write_new_version`.

`resolve`/`publish` are the interface; the storage-seam layout is v1's only backend, but
a second one can be added without either signature changing.

Spec: specs/51-deploy-model-registry.md, specs/52-registry-on-blob.md
"""

from __future__ import annotations

import json
import os
import re
from typing import NamedTuple

from fsd.model import bundle as bundle_mod
from fsd.registry._files import STAGING_PREFIX, write_json_staged
from fsd.registry._files import digest_files as _digest_of
from fsd.registry._files import discard as _discard
from fsd.storage import fs

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

ALIASES_FILE = "_aliases.json"
DEPLOY_FILE = "_deploy.json"
_COMPLETE_FILE = "_complete.json"
_STAGING_PREFIX = STAGING_PREFIX
_MAX_PUBLISH_ATTEMPTS = 16

_REF_RE = re.compile(r"^(?P<name>[^:@]+)(?P<sep>[:@])(?P<value>.+)$")
_VERSION_ALIAS_RE = re.compile(r"^v(\d+)$")
_VERSION_DIR_RE = re.compile(r"^v(\d+)$")


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


def version_path(registry: str, name: str, version: int) -> str:
    return os.path.join(registry, name, f"v{version}")


def _name_root(registry: str, name: str) -> str:
    return os.path.join(registry, name)


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
            aliases = _read_aliases(registry, name, storage_options)
            if value not in aliases:
                raise ValueError(
                    f"alias {value!r} not found for {name!r} in {registry!r} "
                    f"(known aliases: {sorted(aliases)})"
                )
            version = aliases[value]
    return Resolved(name, version, version_path(registry, name, version))


def _read_aliases(registry: str, name: str, storage_options: dict | None) -> dict:
    opts = storage_options or {}
    path = os.path.join(_name_root(registry, name), ALIASES_FILE)
    try:
        with fs.open(path, "r", **opts) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


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
    if _VERSION_ALIAS_RE.match(alias):
        raise ValueError(
            f"alias {alias!r} is shadowed by the 'name@v<N>' version-pin shorthand and "
            "could never be resolved -- pick a different alias name."
        )
    opts = storage_options or {}
    name_root = _name_root(registry, name)
    target = os.path.join(name_root, f"v{version}")
    if not fs.exists(target, **opts):
        raise ValueError(f"cannot set alias {alias!r} -> v{version}: {target!r} does not exist")

    path = os.path.join(name_root, ALIASES_FILE)
    try:
        with fs.open(path, "r", **opts) as f:
            aliases = json.load(f)
    except FileNotFoundError:
        aliases = {}
    aliases[alias] = version
    write_json_staged(name_root, ALIASES_FILE, aliases, opts)


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


def _is_version_complete(version_dir: str, storage_options: dict) -> bool:
    """A version counts once `v<N>/_complete.json` exists, and not before."""
    return fs.exists(os.path.join(version_dir, _COMPLETE_FILE), **storage_options)


def _list_versions(name_root: str, storage_options: dict) -> list[int]:
    if not fs.exists(name_root, **storage_options):
        return []
    versions = []
    for entry in fs.ls(name_root, **storage_options):
        leaf = entry.rstrip("/").rsplit("/", 1)[-1]
        m = _VERSION_DIR_RE.match(leaf)
        if m and _is_version_complete(os.path.join(name_root, leaf), storage_options):
            versions.append(int(m.group(1)))
    return sorted(versions)


def _holds_content(path: str, digest: str, storage_options: dict) -> bool:
    """True if `path` is a bundle directory whose content is exactly `digest`. Anything
    unreadable is "not ours", never an error: this only runs on the retry path, where the
    single question is whether to claim `path` or move on to the next version."""
    try:
        return content_digest(path, storage_options=storage_options) == digest
    except Exception:
        return False


def _write_new_version(
    name_root: str, files: list[tuple[str, bytes]], digest: str, storage_options: dict,
) -> int:
    """Write `files` directly under the next free `v<N>` and mark it complete last,
    returning the version that actually holds them. No staging prefix,
    no directory rename -- `storage.fs.rename` is unchanged by this and is not used here.

    Three steps land in order: (1) write every file straight into `v<N>/`; (2) re-digest
    what actually landed and confirm it equals the caller's digest -- publishing is only
    complete once the bytes are provably the caller's, the same proof `migrate` uses to
    accept a copy; (3) write `v<N>/_complete.json` **last**, carrying that digest.
    A single-object write is atomic on every backend fsd targets, so step 3 is the
    all-or-nothing moment the directory rename used to provide.

    A target that already exists and is complete is a genuine collision only if it does not
    hold the caller's digest, in which case the loop retries at `v<N+1>` -- idempotency,
    reached without writing anything. A target that exists but is *not* complete (an
    interrupted publish) is nobody's, so it is **cleared** and rewritten; clearing is what
    keeps the previous attempt's undeclared leftovers out of the version the re-digest is
    about to bless. A target whose freshly landed bytes do not match the caller's digest --
    two publishers interleaving files at the same `v<N>` -- is left unmarked and the loop
    moves on rather than claiming it.

    A target without `_complete.json` is invisible to `_list_versions`, whatever files it
    holds, and is cleared by the next `publish` that allocates its number.

    Bounded at `_MAX_PUBLISH_ATTEMPTS`: the loop retries only genuine version
    collisions, so an unbounded run here would mean something is structurally wrong, not a
    transient race.
    """
    version = max(_list_versions(name_root, storage_options), default=0) + 1
    for _ in range(_MAX_PUBLISH_ATTEMPTS):
        target = os.path.join(name_root, f"v{version}")
        if _is_version_complete(target, storage_options):
            if _holds_content(target, digest, storage_options):
                return version
            version += 1
            continue

        if fs.exists(target, **storage_options):
            # An incomplete target holds a *previous* attempt's leftovers, and the re-digest
            # below cannot see them: `content_digest` covers
            # only manifest-declared files, so an artifact or a `code/*.py` that THIS
            # bundle does not declare survives into the version and is then marked
            # complete. `bundle.load` puts a version's `code/` on `sys.path`, so a stale
            # module left there is importable by the adapter that lands next. The old
            # stage-then-rename got a clean directory for free -- a fresh staging prefix
            # every attempt -- but publishing in place has to ask for one. Best-effort, like
            # every other `_discard`: a blob backend where recursive rm is unreliable (#50) is
            # no worse off than if this call did not exist.
            _discard(target, storage_options)
        try:
            for rel, data in files:
                fs.write_bytes(os.path.join(target, rel), data, **storage_options)
            landed_digest = content_digest(target, storage_options=storage_options)
        except OSError as exc:
            raise OSError(
                f"publish: writing v{version} under {name_root!r} failed: {exc}"
            ) from exc

        if landed_digest != digest:
            # Someone else's bytes are interleaved with ours at this target.
            # Leaving it unmarked is correct -- it stays invisible -- but claiming it
            # as ours would be wrong, so move on rather than retry the same target.
            version += 1
            continue

        fs.write_text(
            os.path.join(target, _COMPLETE_FILE),
            json.dumps({"digest": digest}, indent=2, sort_keys=True),
            **storage_options,
        )
        return version

    raise RuntimeError(
        f"publish: exhausted {_MAX_PUBLISH_ATTEMPTS} attempts allocating a version under "
        f"{name_root!r} -- repeated version collisions."
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
    name_root = _name_root(registry, name)
    for v in _list_versions(name_root, opts):
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
        name_src = _name_root(src_registry, name)
        name_dst = _name_root(dst_registry, name)

        aliases_src = os.path.join(name_src, ALIASES_FILE)
        if fs.exists(aliases_src, **src_opts):
            with fs.open(aliases_src, "r", **src_opts) as f:
                raw = f.read()
            fs.write_text(os.path.join(name_dst, ALIASES_FILE), raw, **dst_opts)

        for version in _list_versions(name_src, src_opts):
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
    # re-digest-confirmed above, and `_list_versions` shows only marked versions.
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
