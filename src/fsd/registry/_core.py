"""The registry mechanics `fsd.model.registry` and `fsd.image.registry` share: version
allocation, `_aliases.json`, `_complete.json`, the publish loop with its collision retry.

Every function here is generic over the file set (`list[tuple[rel_path, bytes]]`) and the
digest of what landed -- no bundle-specific or image-specific content is assumed.
`content_digest_fn` is the one thing a caller supplies: "digest the directory at this path",
because what counts as a version's content differs per registry (a bundle's manifest-declared
files; an image definition's `image.json`). The user-facing parts -- ref parsing, name checks,
their messages -- stay in each registry.

The concurrency guarantees (what a reader and `publish` can rely on, and what is not
guaranteed) are stated in `fsd.model.registry`'s module docstring; `write_new_version` below
carries the publish-loop proof they rest on.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable

from fsd.registry._files import STAGING_PREFIX as STAGING_PREFIX
from fsd.registry._files import discard as discard
from fsd.registry._files import write_json_staged
from fsd.storage import fs

ALIASES_FILE = "_aliases.json"
COMPLETE_FILE = "_complete.json"
MAX_PUBLISH_ATTEMPTS = 16

VERSION_ALIAS_RE = re.compile(r"^v(\d+)$")
VERSION_DIR_RE = re.compile(r"^v(\d+)$")

ContentDigestFn = Callable[[str, dict], str]


def version_path(registry: str, name: str, version: int) -> str:
    return os.path.join(registry, name, f"v{version}")


def name_root(registry: str, name: str) -> str:
    return os.path.join(registry, name)


def is_version_complete(version_dir: str, storage_options: dict) -> bool:
    """A version counts once `v<N>/_complete.json` exists, and not before."""
    return fs.exists(os.path.join(version_dir, COMPLETE_FILE), **storage_options)


def list_versions(root: str, storage_options: dict) -> list[int]:
    if not fs.exists(root, **storage_options):
        return []
    versions = []
    for entry in fs.ls(root, **storage_options):
        leaf = entry.rstrip("/").rsplit("/", 1)[-1]
        m = VERSION_DIR_RE.match(leaf)
        if m and is_version_complete(os.path.join(root, leaf), storage_options):
            versions.append(int(m.group(1)))
    return sorted(versions)


def _holds_content(
    path: str, digest: str, storage_options: dict, content_digest_fn: ContentDigestFn,
) -> bool:
    """True if `path` holds content that is exactly `digest`. Anything unreadable is "not
    ours", never an error: this only runs on the retry path, where the single question is
    whether to claim `path` or move on to the next version."""
    try:
        return content_digest_fn(path, storage_options) == digest
    except Exception:
        return False


def write_new_version(
    root: str,
    files: list[tuple[str, bytes]],
    digest: str,
    storage_options: dict,
    *,
    content_digest_fn: ContentDigestFn,
) -> int:
    """Write `files` directly under the next free `v<N>` of `root` and mark it complete
    last, returning the version that actually holds them, in the caller's order. No staging
    prefix, no directory rename -- `storage.fs.rename` is not used here.

    `content_digest_fn(path, storage_options)` digests what a version directory holds,
    over the same file set the caller's `digest` covers (a bundle's manifest-declared files;
    an image definition's `image.json`).

    Three steps land in order: (1) write every file straight into `v<N>/`; (2) re-digest
    what actually landed and confirm it equals the caller's digest -- publishing is only
    complete once the bytes are provably the caller's, the same proof `migrate` uses to
    accept a copy; (3) write `v<N>/_complete.json` **last**, carrying that digest.
    A single-object write is atomic on every backend fsd targets, so step 3 is the
    all-or-nothing moment a directory rename would otherwise provide.

    A target that already exists and is complete is a genuine collision only if it does not
    hold the caller's digest, in which case the loop retries at `v<N+1>` -- idempotency,
    reached without writing anything. A target that exists but is *not* complete (an
    interrupted publish) is nobody's, so it is **cleared** and rewritten; clearing is what
    keeps the previous attempt's undeclared leftovers out of the version the re-digest is
    about to bless. A target whose freshly landed bytes do not match the caller's digest --
    two publishers interleaving files at the same `v<N>` -- is left unmarked and the loop
    moves on rather than claiming it.

    A target without `_complete.json` is invisible to `list_versions`, whatever files it
    holds, and is cleared by the next publish that allocates its number.

    Bounded at `MAX_PUBLISH_ATTEMPTS`: the loop retries only genuine version
    collisions, so an unbounded run here would mean something is structurally wrong, not a
    transient race.
    """
    version = max(list_versions(root, storage_options), default=0) + 1

    for _ in range(MAX_PUBLISH_ATTEMPTS):
        target = os.path.join(root, f"v{version}")
        if is_version_complete(target, storage_options):
            if _holds_content(target, digest, storage_options, content_digest_fn):
                return version
            version += 1
            continue

        if fs.exists(target, **storage_options):
            # An incomplete target holds a *previous* attempt's leftovers, and the re-digest
            # below cannot see them: a digest over only the declared files (for a bundle,
            # `content_digest` covers manifest-declared files only) misses an artifact or a
            # `code/*.py` that THIS bundle does not declare, so it survives into the version
            # and is then marked complete. `bundle.load` puts a version's `code/` on
            # `sys.path`, so a stale module left there is importable by the adapter that
            # lands next. The old stage-then-rename got a clean directory for free -- a
            # fresh staging prefix every attempt -- but publishing in place has to ask for
            # one. Best-effort, like every other `discard`: a blob backend where recursive
            # rm is unreliable (#50) is no worse off than if this call did not exist.
            discard(target, storage_options)
        try:
            for rel, data in files:
                fs.write_bytes(os.path.join(target, rel), data, **storage_options)
            landed_digest = content_digest_fn(target, storage_options)
        except OSError as exc:
            raise OSError(
                f"publish: writing v{version} under {root!r} failed: {exc}"
            ) from exc

        if landed_digest != digest:
            # Someone else's bytes are interleaved with ours at this target.
            # Leaving it unmarked is correct -- it stays invisible -- but claiming it
            # as ours would be wrong, so move on rather than retry the same target.
            version += 1
            continue

        fs.write_text(
            os.path.join(target, COMPLETE_FILE),
            json.dumps({"digest": digest}, indent=2, sort_keys=True),
            **storage_options,
        )
        return version

    raise RuntimeError(
        f"publish: exhausted {MAX_PUBLISH_ATTEMPTS} attempts allocating a version under "
        f"{root!r} -- repeated version collisions."
    )


def read_aliases(registry: str, name: str, storage_options: dict) -> dict:
    path = os.path.join(name_root(registry, name), ALIASES_FILE)
    try:
        with fs.open(path, "r", **storage_options) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


_REF_RE = re.compile(r"^(?P<name>[^:@]+)(?P<sep>[:@])(?P<value>.+)$")


def parse_ref(ref: str) -> tuple[str, str, str]:
    """`'fsd-aml-env:7'` -> `('fsd-aml-env', ':', '7')`; `'fsd-aml-env@current'` ->
    `('fsd-aml-env', '@', 'current')` (mirrors `fsd.model.registry.parse_ref`)."""
    m = _REF_RE.match(ref)
    if not m:
        raise ValueError(
            f"ref {ref!r} is ambiguous: use 'name:version' (e.g. 'fsd-aml-env:7') or "
            "'name@alias' (e.g. 'fsd-aml-env@current')."
        )
    return m.group("name"), m.group("sep"), m.group("value")


def check_name(name: str) -> None:
    """Refuse a name that cannot round-trip through a ref (mirrors
    `fsd.model.registry.check_name` -- same reasoning, generalized)."""
    if not name or not name.strip():
        raise ValueError("registry entry name must be a non-empty string.")
    bad = [c for c in ("/", "\\", ":", "@") if c in name]
    if bad:
        raise ValueError(
            f"name {name!r} contains {bad!r}, which a ref cannot carry: use letters, "
            "digits, '-' or '_'."
        )
    if name.startswith("."):
        raise ValueError(f"name {name!r} starts with '.': dot-entries are internal.")


def resolve_ref(ref: str, registry: str, storage_options: dict) -> tuple[str, int, str]:
    """Turn a ref into `(name, version, path)`. A version pin (`name:N` / `name@vN`) costs
    no reads; a named alias costs one `_aliases.json` read (mirrors
    `fsd.model.registry.resolve`)."""
    name, sep, value = parse_ref(ref)
    if sep == ":":
        if not value.isdigit():
            raise ValueError(f"ref {ref!r}: version must be an integer, got {value!r}")
        version = int(value)
    else:
        m = VERSION_ALIAS_RE.match(value)
        if m:
            version = int(m.group(1))
        else:
            aliases = read_aliases(registry, name, storage_options)
            if value not in aliases:
                raise ValueError(
                    f"alias {value!r} not found for {name!r} in {registry!r} "
                    f"(known aliases: {sorted(aliases)})"
                )
            version = aliases[value]
    return name, version, version_path(registry, name, version)


def set_alias(
    name: str, alias: str, version: int, registry: str, storage_options: dict,
) -> None:
    """Repoint `alias` at `version`, staged and renamed so a reader never sees a
    half-written `_aliases.json` (mirrors `fsd.model.registry.set_alias`)."""
    if VERSION_ALIAS_RE.match(alias):
        raise ValueError(
            f"alias {alias!r} is shadowed by the 'name@v<N>' version-pin shorthand and "
            "could never be resolved -- pick a different alias name."
        )
    root = name_root(registry, name)
    target = os.path.join(root, f"v{version}")
    if not fs.exists(target, **storage_options):
        raise ValueError(f"cannot set alias {alias!r} -> v{version}: {target!r} does not exist")

    aliases = read_aliases(registry, name, storage_options)
    aliases[alias] = version
    write_json_staged(root, ALIASES_FILE, aliases, storage_options)
