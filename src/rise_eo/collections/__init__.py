"""Public collection registry -- `rise_eo.collections.register()` / `.get()`.

Spec: specs/58-collection-agnostic-verbs.md D2, D13. ADR 0030, 0031.

Per-collection facts (bands, mask, radiometry, grid) live here, keyed by STAC collection
id, rather than hardcoded into a source module. Built-in collections register themselves
at import time (`s2_l2a`, `s1_rtc`); a caller's own variant calls `register()` directly -- no entry
points, no packaging, no image rebuild.

**The registry is consulted ONLY on the driver.** A `collection=` string is resolved to a
`CollectionDeclaration` here, in-process, before any work is dispatched; the resolved
declaration then travels as JSON in a control file under the run folder
(`rise_eo.workflows.create_datacube.setup` writes it). A node reads that file -- it never
imports this module to resolve a collection string, because this registry is a plain
in-process dict that does not exist inside a fresh AML job container (ADR 0031).
"""

from __future__ import annotations

from rise_eo.catalog.declaration import CollectionDeclaration
from rise_eo.collections import s1_rtc as _s1_rtc
from rise_eo.collections import s2_l2a as _s2_l2a

__all__ = ["register", "get", "known", "REGISTRY"]

REGISTRY: dict[str, CollectionDeclaration] = {}


def register(collection_id: str, declaration: CollectionDeclaration, *, force: bool = False) -> None:
    """Register `declaration` under `collection_id`.

    Refuses to silently overwrite an existing registration unless `force=True` --
    registering a variant under a name already taken is almost always a typo, not an
    intended override. A declaration may not contain callables (ADR 0031): it must be
    representable as plain JSON (`rise_eo.catalog.declaration.to_json`).
    """
    if collection_id in REGISTRY and not force:
        raise ValueError(
            f"rise_eo.collections.register: {collection_id!r} is already registered; pass "
            "force=True to overwrite it."
        )
    REGISTRY[collection_id] = declaration


def get(collection_id: str) -> CollectionDeclaration:
    """The registered declaration for `collection_id`, or raise naming what IS registered."""
    try:
        return REGISTRY[collection_id]
    except KeyError:
        raise ValueError(
            f"rise_eo.collections.get: unknown collection {collection_id!r}; known: "
            f"{sorted(REGISTRY)}."
        ) from None


def known() -> list[str]:
    """Every registered collection id, sorted."""
    return sorted(REGISTRY)


register(_s2_l2a.COLLECTION_ID, _s2_l2a.DECLARATION)
register(_s1_rtc.COLLECTION_ID, _s1_rtc.DECLARATION)
