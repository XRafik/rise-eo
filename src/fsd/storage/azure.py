"""Azure ADLS Gen2 compute seam — `to_vsi`, the GDAL VSI token, and `storage=` config.

Deliberately not a storage-options registry, and no credential object is passed around:
adlfs auto-resolves `DefaultAzureCredential` from `account_name` + `anon=False`, and
fsspec's per-protocol config (`fsspec.config.conf` / `FSSPEC_{PROTOCOL}_*`) already does
what a registry would.

This module supplies only the two things fsspec/adlfs cannot: the `abfss://` ->
`/vsiadls/` translation GDAL needs (adlfs is not on GDAL's pixel-read path), and a token
for GDAL's Azure VSI handler, which unlike adlfs never refreshes a token it is given.

Spec: specs/31-p1-azure-storage-seam.md
"""

from __future__ import annotations

import os
import re
import threading
from typing import Any

import fsspec

__all__ = ["to_vsi", "account_from_url", "storage_token", "configure_storage"]

# abfss://<filesystem>@<account>.dfs.core.windows.net/<path>
_ABFSS_RE = re.compile(r"^abfss://([^@/]+)@([^./]+)\.dfs\.core\.windows\.net/(.*)$")
# az://<filesystem>/<path> (alias; account comes from ambient config, not the URL)
_AZ_RE = re.compile(r"^az://([^/]+)/(.*)$")

_credential_lock = threading.Lock()
_credential = None


def _get_credential():
    """The single module-cached `DefaultAzureCredential`. Reuse is deliberate: it shares
    the token cache and avoids Entra 429 throttling."""
    global _credential
    if _credential is None:
        with _credential_lock:
            if _credential is None:  # pragma: no branch - re-check under the lock
                from azure.identity import DefaultAzureCredential

                _credential = DefaultAzureCredential()
    return _credential


def storage_token() -> str:
    """A fresh Storage-scoped bearer token from the module-cached credential.

    Call this before *every* GDAL VSI open: `AZURE_STORAGE_ACCESS_TOKEN` is static and GDAL
    never refreshes it, so refresh is ours to own. Cheap, because `get_token` caches and
    auto-refreshes internally — re-fetching per open beats a hand-rolled expiry margin.
    """
    return _get_credential().get_token("https://storage.azure.com/.default").token


def to_vsi(url: str) -> str:
    """Translate a canonical blob URL to a GDAL VSI path. Local paths pass through unchanged.

    - `abfss://<fs>@<account>.dfs.core.windows.net/<path>` -> `/vsiadls/<fs>/<path>`
    - `az://<fs>/<path>` (alias)                            -> `/vsiadls/<fs>/<path>`
    - anything else (a local path, `file://`, ...)           -> returned unchanged

    A string that declares itself `abfss://`/`az://` but doesn't match the expected shape
    raises rather than being silently treated as local.
    """
    if url.startswith("abfss://"):
        m = _ABFSS_RE.match(url)
        if not m:
            raise ValueError(f"malformed abfss:// URL (expected fs@account.dfs.core.windows.net/path): {url!r}")
        filesystem, _account, path = m.groups()
        return f"/vsiadls/{filesystem}/{path}"
    if url.startswith("az://"):
        m = _AZ_RE.match(url)
        if not m:
            raise ValueError(f"malformed az:// URL (expected fs/path): {url!r}")
        filesystem, path = m.groups()
        return f"/vsiadls/{filesystem}/{path}"
    return url


def account_from_url(url: str) -> str | None:
    """The storage account name from a fully-qualified `abfss://` URL host, or `None` for any
    other URL shape (including the `az://` alias, whose account comes from ambient config)."""
    m = _ABFSS_RE.match(url)
    return m.group(2) if m else None


def storage_backend(storage: Any) -> str | None:
    """The backend name a `storage=` kwarg names: a string is itself, a dict gives its
    `"backend"` key, anything else is `None`."""
    if isinstance(storage, str):
        return storage
    if isinstance(storage, dict):
        return storage.get("backend")
    return None


def configure_storage(storage: Any) -> None:
    """Apply a verb's `storage=` kwarg. `None`/`"local"` are no-ops; `"azure"` (or
    `{"backend": "azure", ...}`) enables authenticated adlfs; anything else raises.

    Sets **both** `os.environ` and `fsspec.config.conf`, and both are required. fsspec
    populates `conf` from `FSSPEC_*` env at *import* time, so mutating `os.environ`
    afterwards does not update this process's already-imported `conf`; conversely
    `os.environ` is what a subprocess child inherits and re-reads on its own import,
    which is how the setting crosses that boundary with no live credential object.

    `FSSPEC_ABFSS_ANON` is the only key needed: fsspec's `apply_config` keys on
    `AzureBlobFileSystem.protocol`, covering `abfs`/`az`/`abfss` at once, and the account
    comes from the URL host rather than from config.
    """
    if storage is None or storage == "local":
        return
    backend = storage_backend(storage)
    if backend != "azure":
        raise ValueError(f"storage backend {backend!r} not supported (only 'azure' in P1).")
    os.environ["FSSPEC_ABFSS_ANON"] = "false"
    fsspec.config.conf.setdefault("abfss", {})["anon"] = False
