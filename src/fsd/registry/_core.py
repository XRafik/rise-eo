"""Generic registry mechanics: version allocation, `_aliases.json`, `_complete.json`, the
collision retry.

`fsd.model.registry` proved this shape first, and its guarantees are documented
in its own module docstring -- read that before touching this file. **This module is a
parameterized copy of that logic, not an import of it**:
`fsd/model/registry.py`'s tests monkeypatch its own module-level `_list_versions` /
`_write_new_version` and call them directly, which only works if those functions stay defined
in `fsd/model/registry.py`'s own namespace -- a real "move + re-export" would leave those
patches inert (the moved functions would close over *this* module's globals, not the caller's).
So the model registry is untouched, and this module exists purely for a second caller
(`fsd.image.registry`) that has no such tests to preserve.

Every function here is generic over the file set (`list[tuple[rel_path, bytes]]`) and the
digest of what landed -- there is no bundle-specific, or image-specific, content assumed.
`content_digest_fn` is the one thing a caller supplies: "digest the directory at this path",
because what counts as a version's content differs per registry (a bundle's manifest-declared
files; an image definition's `image.json`).
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
    """A version counts once `v<N>/_complete.json` exists, and not before; the model
    registry (`fsd.model.registry`) uses the same rule."""
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
    manifest_rel: str | None = None,
) -> int:
    """Write `files` under the next free `v<N>` and mark it complete last, returning the
    version that actually holds them. Mirrors `fsd.model.registry._write_new_version`,
    generalized over the content digest, except that it sorts `files` by path and writes
    `manifest_rel` (image.json for the image registry), when given, last among them.
    """
    version = max(list_versions(root, storage_options), default=0) + 1
    if manifest_rel is not None:
        ordered_files = sorted(files, key=lambda item: (item[0] == manifest_rel, item[0]))
    else:
        ordered_files = sorted(files, key=lambda item: item[0])

    for _ in range(MAX_PUBLISH_ATTEMPTS):
        target = os.path.join(root, f"v{version}")
        if is_version_complete(target, storage_options):
            if _holds_content(target, digest, storage_options, content_digest_fn):
                return version
            version += 1
            continue

        if fs.exists(target, **storage_options):
            discard(target, storage_options)
        try:
            for rel, data in ordered_files:
                fs.write_bytes(os.path.join(target, rel), data, **storage_options)
            landed_digest = content_digest_fn(target, storage_options)
        except OSError as exc:
            raise OSError(
                f"publish: writing v{version} under {root!r} failed: {exc}"
            ) from exc

        if landed_digest != digest:
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
