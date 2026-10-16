"""File-level helpers shared by `rise_eo.model.registry` and `rise_eo.registry._core`: the
staged JSON write, the best-effort discard of a staging leftover, and the content digest
(also used by `rise_eo.image.digest`). Stdlib and `rise_eo.storage.fs` only, so any of them can
import it without a cycle.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import uuid

from rise_eo.storage import fs

STAGING_PREFIX = ".staging-"


def discard(path: str, storage_options: dict) -> None:
    """Best-effort removal of an abandoned staging tree. A leftover is harmless -- it is
    never visible as a version (`list_versions` matches `v<N>` only) -- so a
    failure to clean up must not mask the error that caused it."""
    with contextlib.suppress(Exception):
        fs.rm(path, recursive=True, **storage_options)


def write_json_staged(directory: str, filename: str, obj, storage_options: dict) -> None:
    """Write `obj` as JSON to `<directory>/<filename>`, staged beside it and then renamed, so a
    reader never sees a half-written file."""
    stage = os.path.join(directory, f"{STAGING_PREFIX}{uuid.uuid4().hex}.json")
    try:
        fs.write_text(stage, json.dumps(obj, indent=2, sort_keys=True), **storage_options)
        fs.rename(stage, os.path.join(directory, filename), **storage_options)
    except BaseException:
        discard(stage, storage_options)
        raise


def digest_files(files: list[tuple[str, bytes]]) -> str:
    """SHA-256 over `(relative path, bytes)` pairs, sorted by path, each fed as path, NUL, bytes."""
    # The byte order is the contract: changing it changes every registered version's digest.
    h = hashlib.sha256()
    for rel, data in sorted(files, key=lambda item: item[0]):
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(data)
    return f"sha256:{h.hexdigest()}"
