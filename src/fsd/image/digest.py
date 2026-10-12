"""Resolving and digesting an `ImageDefinition`.

Two functions: `resolve()` turns every moving reference into a fixed one -- a `git+...@main`
becomes `git+...@<40-char sha>`, a `path:` `fsd` becomes a wheel content digest, an
unpinned base tag becomes `@sha256:...` when it can be resolved and stays a tag (recorded
unresolved) when it can't -- and `digest()` hashes the result. `resolve()`'s output is
exactly `image.json`'s `definition` field.

**The exclusion list is a named constant** (`_DIGEST_EXCLUDE`), not an accident of what the
dataclass happens to carry: `name` never enters the payload, because renaming an image does
not change its contents (the same rule as flytekit's `parameters_to_exclude`). `build_context`
is excluded from the payload as a path and replaced by `build_context_digest` -- a digest of
its file *contents*, never its location (flytekit nullifies `registry_config` for the same
reason: "a path that does not affect the image must not affect the key").
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
import zipfile
from typing import Callable

from fsd.image.definition import ImageDefinition, _build_wheel
from fsd.registry._files import digest_files

__all__ = ["digest", "resolve", "wheel_digest"]

_DIGEST_EXCLUDE = frozenset({"name"})
# A full sha is the canonical pin. An ABBREVIATED one (>=7 hex, e.g. `@9a00f2b`) is already
# immutable and pip installs it fine, but `git ls-remote` CANNOT expand it: it matches REF
# PATTERNS, not object ids, and returns exit 0 with empty stdout for a sha. So an abbreviation
# is kept verbatim rather than sent to a lookup that would silently succeed-with-nothing. The
# cost is that two abbreviations of one commit digest differently -- a rebuild for nothing,
# which is the cheaper of the two failures.
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def wheel_digest(path: str) -> str:
    """A content hash that ignores zip metadata.

    `pip wheel` stamps timestamps, so two wheels built from identical source are never
    byte-identical -- hashing the file would report "changed" on every run. Hashing the
    sorted (member name, CRC) pairs compares what is actually INSIDE the wheel.
    """
    with zipfile.ZipFile(path) as zf:
        payload = "\n".join(
            f"{i.filename}:{i.CRC}" for i in sorted(zf.infolist(), key=lambda i: i.filename)
        )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _digest_directory(path: str) -> str:
    """SHA-256 over every file under `path` (sorted relative path + bytes) -- the
    `build_context=` escape hatch's digest, content not path."""
    files: list[tuple[str, bytes]] = []
    for root, _dirs, filenames in os.walk(path):
        for fn in filenames:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, path)
            with open(full, "rb") as f:
                files.append((rel, f.read()))
    return digest_files(files)


def _git_ls_remote_sha(repo_url: str, ref: str) -> str:
    result = subprocess.run(
        ["git", "ls-remote", repo_url, ref], capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git ls-remote {repo_url} {ref} failed: {result.stderr.strip()[:300]}"
        )
    if not result.stdout.strip():
        raise RuntimeError(
            f"git ls-remote {repo_url} {ref}: no such branch or tag. `ls-remote` matches ref "
            "names only -- pass a branch, a tag, or a commit sha (>=7 hex, kept as given)."
        )
    return result.stdout.split()[0]


def _resolve_fsd(
    fsd: str, resolve_git_ref: Callable[[str, str], str], wheel_dir: str | None = None,
) -> str:
    if fsd.startswith("path:"):
        src = fsd[len("path:"):]
        if wheel_dir is not None:
            # Build into the caller's directory so the wheel this digest describes is the
            # SAME file that goes into the build context. Building twice (once here, once in
            # `write_context`) both doubles the slow path and opens a window where an edit
            # between the two makes the registry record a digest the image does not have,
            # which is the expensive failure.
            return f"wheel:{wheel_digest(_build_wheel(src, wheel_dir))}"
        with tempfile.TemporaryDirectory() as tmp:
            wheel = _build_wheel(src, tmp)
            return f"wheel:{wheel_digest(wheel)}"
    if fsd.startswith("git+"):
        url_part, sep, ref = fsd.partition("@")
        if not sep:
            raise ValueError(f"fsd={fsd!r}: a git+ reference needs '@<ref>'.")
        if _GIT_SHA_RE.match(ref):
            return fsd
        repo_url = url_part[len("git+"):]
        sha = resolve_git_ref(repo_url, ref)
        return f"git+{repo_url}@{sha}"
    return fsd  # a plain pip spec (e.g. PyPI), left as declared


def _default_resolve_base_digest(base: str) -> str | None:
    """`HEAD /v2/{repo}/manifests/{tag}` against the base's own registry (the Docker
    Registry HTTP API v2, which `mcr.microsoft.com` serves unauthenticated for public
    images) -- `Docker-Content-Digest` is the tag's current digest. Anything going wrong
    (offline, private registry, unexpected host shape) returns `None` rather than raising:
    an unresolvable base is a warning (`base_resolved: false`), never a hard failure."""
    try:
        host, repo_tag = base.split("/", 1)
        repo, tag = repo_tag.rsplit(":", 1)
        url = f"https://{host}/v2/{repo}/manifests/{tag}"
        req = urllib.request.Request(url, method="HEAD", headers={
            # A multi-arch tag resolves to an index/manifest-list, not a manifest; omitting
            # those two media types is how a HEAD against such a tag comes back 404/406 and
            # silently degrades to `base_resolved: false`.
            "Accept": "application/vnd.docker.distribution.manifest.list.v2+json,"
                      "application/vnd.oci.image.index.v1+json,"
                      "application/vnd.docker.distribution.manifest.v2+json,"
                      "application/vnd.oci.image.manifest.v1+json",
        })
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 - fixed https scheme
            return resp.headers.get("Docker-Content-Digest")
    except (ValueError, OSError, urllib.error.URLError):
        return None


def _resolve_base(base: str, resolve_base_digest: Callable[[str], str | None]) -> tuple[str, bool]:
    if "@sha256:" in base:
        return base, True
    digest_value = resolve_base_digest(base)
    if digest_value is None:
        return base, False
    repo, _, _tag = base.rpartition(":")
    return f"{repo}@{digest_value}", True


def resolve(
    defn: ImageDefinition,
    *,
    resolve_base_digest: Callable[[str], str | None] | None = None,
    resolve_git_ref: Callable[[str, str], str] | None = None,
    wheel_dir: str | None = None,
) -> dict:
    """Turn `defn` into the resolved-definition dict `image.json` stores: every
    moving reference fixed, `name` and `build_context` (a path) excluded, `build_context`
    replaced by its content digest when set.

    `resolve_base_digest`/`resolve_git_ref` default to the real network calls; a test
    injects fakes so nothing here needs a live registry or a live git remote.

    `wheel_dir`, for a `path:` `fsd` only, is where the wheel this digest is computed from is
    left, so the caller can hand that exact file to `write_context` instead of building a
    second one (`ImageDefinition.write_context` reuses an `fsd-*.whl` already sitting there).
    """
    resolve_base_digest = resolve_base_digest or _default_resolve_base_digest
    resolve_git_ref = resolve_git_ref or _git_ls_remote_sha

    if defn.build_context is not None:
        return {"build_context_digest": _digest_directory(defn.build_context)}

    if not defn.fsd:
        raise ValueError(
            "ImageDefinition.fsd must be set to resolve() (or pass build_context=)."
        )

    payload = {
        k: v for k, v in dataclasses.asdict(defn).items()
        if k not in _DIGEST_EXCLUDE and k != "build_context"
    }
    payload["fsd"] = _resolve_fsd(defn.fsd, resolve_git_ref, wheel_dir)
    base_ref, base_resolved = _resolve_base(defn.base, resolve_base_digest)
    payload["base"] = base_ref
    payload["base_resolved"] = base_resolved
    return payload


def digest(resolved: dict) -> str:
    """SHA-256 over the resolved-definition dict's canonical JSON (sorted keys, no
    whitespace) -- no `id()`, no dict-iteration order, no absolute paths."""
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"
