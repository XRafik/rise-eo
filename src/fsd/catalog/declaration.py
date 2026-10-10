"""Collection -> builder declaration contract.

Spec: specs/34-ingest-normalization-contract.md, specs/58-collection-agnostic-verbs.md

`build_datacube` is a generic engine: no `if collection == "s2"` anywhere. It reads what it
needs from a `CollectionDeclaration`, carried on the flattened catalog as the JSON-able
`GeoDataFrame.attrs[ATTRS_KEY]` (set by `flatten_catalog`, restored from the Parquet footer
by `fsd.storage.fs.read_parquet`; never the dataclass itself) or passed explicitly. A
collection that needs different mask/reference/mosaic behavior registers its own
declaration (`fsd.collections.register`); see `docs/adding-a-source.md`.

Per-row values that vary (the radiometric `offset`, the per-row `nodata`) live in catalog
columns (`fsd.catalog.catalog.COLUMNS`); the declaration's `nodata` is only the fallback
when the column is absent.

Two kinds of field:

- **Artifact facts** (describe the bytes; stamped at ingest, never vary per build):
  `nodata`, `scale`, `radiometry_bands`, `band_aliases`, `requires_subscription_key`.
- **Build policy** (bytes -> cube; may vary per build via a different named collection
  variant): `mask_spec`, `mosaic_method`, `mosaic_partition`, `partition_policy`,
  `reference_band`.
"""

from __future__ import annotations

import dataclasses

from fsd import config

__all__ = [
    "MASK_TYPE_CATEGORICAL_CLASSES",
    "MaskSpec",
    "CollectionDeclaration",
    "ARTIFACT_FACT_FIELDS",
    "FSD_DECLARATION_VERSION",
    "ATTRS_KEY",
    "to_json",
    "from_json",
    "to_attrs",
    "from_attrs",
    "digest",
]

# Persistence. `ATTRS_KEY` is the key under which the plain-dict JSON form of a
# `CollectionDeclaration` lives inside `GeoDataFrame.attrs` -- never the dataclass itself,
# which would not survive the round-trip. `fsd.storage.fs` serializes that whole `.attrs`
# dict to the Parquet footer's `PANDAS_ATTRS` key.
#
# The version is bumped when fields are added: without it, an older fsd reading a newer
# footer would report "unknown field" instead of "written by a newer fsd -- upgrade".
FSD_DECLARATION_VERSION = 2
ATTRS_KEY = "fsd:declaration"

_DECLARATION_FIELDS = (
    "reference_band", "native_grid", "mask_spec", "mask_keep", "nodata",
    "mosaic_method", "scale", "radiometry_bands", "band_aliases",
    "requires_subscription_key", "supports_cloud_cover", "mosaic_partition",
    "partition_policy",
)
_MASK_SPEC_FIELDS = ("band", "mask_type", "classes", "bits")

# The `CollectionDeclaration` fields that are facts about the stored artifact.
ARTIFACT_FACT_FIELDS = (
    "nodata", "scale", "radiometry_bands", "band_aliases", "requires_subscription_key",
)

# The only implemented `MaskSpec.mask_type`: mask wherever the
# mask band's pixel value is one of `classes` (covers S2 SCL). `bitmask`
# (Landsat/HLS QA) and `threshold` (continuous cloud-probability) are the
# named-but-unimplemented seam for a later source -- see `CollectionDeclaration`.
MASK_TYPE_CATEGORICAL_CLASSES = "categorical_classes"


@dataclasses.dataclass(frozen=True)
class MaskSpec:
    """Which band to read, how to interpret it, and which values mean "masked".

    `band` — the mask band's name (e.g. ``"SCL"``). `mask_type` — one of the
    growable `MASK_TYPE_*` constants; only `categorical_classes` is
    implemented (`builder.build_datacube` raises `NotImplementedError` for
    any other value — a loud, documented gap, not a silent wrong mask).
    `classes` — for `categorical_classes`, the pixel values that mean
    "masked" (e.g. S2 SCL cloud/shadow/nodata classes). `bits` — for
    `bitmask`, the bit POSITIONS that mean "masked" when ANY is set (e.g. HLS
    Fmask cloud/shadow/snow); a separate field from `classes`, never an
    overload of it -- conflating pixel VALUES with bit POSITIONS yields a
    silently wrong mask. `mask_type="bitmask"` is named but not implemented
    (`builder.build_datacube` raises `NotImplementedError`).
    """

    band: str
    mask_type: str = MASK_TYPE_CATEGORICAL_CLASSES
    classes: tuple[int, ...] = ()
    bits: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class CollectionDeclaration:
    """What `build_datacube` needs to know about a collection, read once per build
    instead of hardcoded.

    `reference_band` — the band whose grid (10 m B08 for S2) every other band is resampled
    onto. `None` means no resample step; the first requested band gives the merge
    geometry. Independent of `native_grid`: S2 needs a reference (10/20/60 m bands), HLS and
    Sentinel-1 RTC do not.

    `native_grid` — `True` means one native global/regional grid, skipping the
    multi-MGRS-tile single-CRS collapse. Designed-for, not implemented: `build_datacube`
    raises `NotImplementedError`.

    `mask_spec` — `None` means no mask (e.g. CHIRPS). A `MaskSpec` whose `band` is not in
    the build's requested `bands` also means no masking. Never overridden by a verb
    parameter: a different mask is a different named collection variant
    (`fsd.collections.register`).

    `mask_keep` — `False` drops the mask band after masking; `True` keeps it in the cube.

    `nodata` — fallback when the catalog rows carry no `nodata` column (older or hand-built
    catalogs); a per-row value wins.

    `mosaic_method` — only "median" is implemented (`fsd.datacube.ops.median_mosaic`).

    `scale` — the declared multiplicative radiometric scale (e.g. `1/10000` for S2),
    alongside the per-row `offset` column. Declared, never applied by the build (ADR 0011):
    pixels stay raw DN. Default `1.0`.

    `radiometry_bands` — which bands carry `offset`/`scale`; `None` means every band, a
    tuple names exactly which (mask/QA bands never do). See `is_radiometry_band`.

    `band_aliases` — canonical STAC EO `common_name` (`"red"`, `"nir"`, `"nir08"`, ...) ->
    this collection's native asset key (S2 L2A `"nir08"` -> `"B8A"`). fsd declares the
    mapping itself rather than trusting a provider's `common_name` (MPC's HLS assignments
    are inconsistent).

    `requires_subscription_key` — whether the collection needs `PC_SDK_SUBSCRIPTION_KEY`
    (or the AML Key Vault equivalent) although MPC is otherwise anonymous. A collection
    capability, not a source fact.

    `supports_cloud_cover` — whether the collection has an `eo:cloud_cover` discovery
    filter. With `False` (e.g. Sentinel-1), `max_cloudcover` is refused with a
    `PreflightError` naming the collection rather than silently ignored.

    `mosaic_partition` / `partition_policy` — catalog property keys that must hold a
    SINGLE value within one build, and what to do when they don't (`"raise"`; `"auto"` is
    declared but raises `NotImplementedError`). Empty tuple = no enforcement.

    Spec: 58.
    """

    reference_band: str | None = None
    native_grid: bool = False
    mask_spec: MaskSpec | None = None
    mask_keep: bool = False
    nodata: int = 0
    mosaic_method: str = "median"
    scale: float = 1.0
    radiometry_bands: tuple[str, ...] | None = None
    # A tuple of (canonical, native) pairs, not a dict -- a frozen dataclass with a dict
    # field is unhashable (`hash()` raises `TypeError: unhashable type: 'dict'`), and
    # `CollectionDeclaration` must stay hashable/frozen like every other field here.
    # `canonical_to_native` converts to a dict at lookup time.
    band_aliases: tuple[tuple[str, str], ...] = ()
    requires_subscription_key: bool = False
    supports_cloud_cover: bool = False
    mosaic_partition: tuple[str, ...] = ()
    partition_policy: str = "raise"

    def is_radiometry_band(self, band: str) -> bool:
        """Whether `band` carries this collection's declared radiometric
        `offset`/`scale`. `radiometry_bands=None` means every band does."""
        return self.radiometry_bands is None or band in self.radiometry_bands

    def canonical_to_native(self, band: str) -> str:
        """Resolve `band` (a canonical STAC EO `common_name`, or already a
        native asset key) to this collection's native asset key. A name not
        in `band_aliases` is returned unchanged -- it is assumed to already
        be native."""
        return dict(self.band_aliases).get(band, band)


# Kept here (rather than only in `fsd.collections.s2_l2a`) for existing imports and because
# `builder._resolve_declaration`'s hand-built-GeoDataFrame fallback needs a concrete
# default without importing the registry package (a declaration<->collections import
# cycle: `fsd.collections` imports this module).
S2_L2A_DECLARATION = CollectionDeclaration(
    reference_band=config.REFERENCE_BAND,
    native_grid=False,
    mask_spec=MaskSpec(
        band="SCL",
        mask_type=MASK_TYPE_CATEGORICAL_CLASSES,
        classes=tuple(config.SCL_MASK_CLASSES),
    ),
    mask_keep=False,
    nodata=config.NODATA,
    mosaic_method="median",
    scale=config.S2_REFLECTANCE_SCALE,
    # Every S2 L2A band except the mask (SCL). NOT `None` ("all bands"): that would
    # radiometrically offset SCL too (SCL is a classification, not a DN).
    radiometry_bands=tuple(b for b in config.S2L2A_ALL_BANDS if b != "SCL"),
    band_aliases=(("red", "B04"), ("nir", "B08"), ("nir08", "B8A")),
    requires_subscription_key=False,
    supports_cloud_cover=True,
    mosaic_partition=(),
    partition_policy="raise",
)


# --- serialization (pure functions, no I/O) -----------------------------------
#
# The dataclass never goes in `.attrs` directly: a dataclass there crashes once a
# JSON-encoding writer touches it.


def _mask_spec_to_json(mask_spec: MaskSpec | None) -> dict | None:
    if mask_spec is None:
        return None
    return {
        "band": mask_spec.band,
        "mask_type": mask_spec.mask_type,
        "classes": list(mask_spec.classes),
        "bits": list(mask_spec.bits),
    }


def _mask_spec_from_json(raw: dict | None) -> MaskSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            f"declaration.mask_spec must be a JSON object or null, got "
            f"{type(raw).__name__}: {raw!r}."
        )
    unknown = set(raw) - set(_MASK_SPEC_FIELDS)
    if unknown:
        raise ValueError(
            f"declaration.mask_spec has unknown field(s) {sorted(unknown)}; "
            f"known fields are {_MASK_SPEC_FIELDS}."
        )
    if "band" not in raw:
        raise ValueError("declaration.mask_spec is missing required field 'band'.")
    kwargs: dict = {"band": raw["band"]}
    if "mask_type" in raw:
        kwargs["mask_type"] = raw["mask_type"]
    if "classes" in raw:
        kwargs["classes"] = tuple(raw["classes"])
    if "bits" in raw:
        kwargs["bits"] = tuple(raw["bits"])
    return MaskSpec(**kwargs)


def to_json(decl: CollectionDeclaration) -> dict:
    """`CollectionDeclaration` -> a plain JSON-able dict, field-for-field.

    Tuples (`MaskSpec.classes`/`.bits`, `radiometry_bands`, `mosaic_partition`) become
    JSON arrays/null; `from_json` rehydrates them back into tuples, keeping the
    dataclass frozen/hashable.
    """
    return {
        "fsd_declaration_version": FSD_DECLARATION_VERSION,
        "reference_band": decl.reference_band,
        "native_grid": decl.native_grid,
        "mask_spec": _mask_spec_to_json(decl.mask_spec),
        "mask_keep": decl.mask_keep,
        "nodata": decl.nodata,
        "mosaic_method": decl.mosaic_method,
        "scale": decl.scale,
        "radiometry_bands": (
            list(decl.radiometry_bands) if decl.radiometry_bands is not None else None
        ),
        "band_aliases": dict(decl.band_aliases),  # JSON wire form: still a dict/object
        "requires_subscription_key": decl.requires_subscription_key,
        "supports_cloud_cover": decl.supports_cloud_cover,
        "mosaic_partition": list(decl.mosaic_partition),
        "partition_policy": decl.partition_policy,
    }


def from_json(raw: dict) -> CollectionDeclaration:
    """Inverse of `to_json`. Raises on a version newer than this fsd supports, an
    unknown field at a known version, or a `mask_spec` object missing `band`
    -- fail loudly rather than half-understand a declaration. A missing *optional* field
    takes the dataclass default, which is what keeps a v1 footer (or a future v3)
    readable."""
    if not isinstance(raw, dict):
        raise ValueError(
            f"declaration JSON must be a JSON object, got {type(raw).__name__}: {raw!r}."
        )
    version = raw.get("fsd_declaration_version")
    if version is None:
        raise ValueError(
            "declaration JSON is missing required field 'fsd_declaration_version'."
        )
    if version > FSD_DECLARATION_VERSION:
        raise ValueError(
            f"declaration JSON was written by a newer fsd (version {version}) than "
            f"this one supports (version {FSD_DECLARATION_VERSION}); upgrade fsd "
            "before reading this catalog."
        )
    unknown = set(raw) - {"fsd_declaration_version", *_DECLARATION_FIELDS}
    if unknown:
        raise ValueError(
            f"declaration JSON (version {version}) has unknown field(s) "
            f"{sorted(unknown)}; known fields are {_DECLARATION_FIELDS}."
        )

    kwargs: dict = {}
    if "reference_band" in raw:
        kwargs["reference_band"] = raw["reference_band"]
    if "native_grid" in raw:
        kwargs["native_grid"] = raw["native_grid"]
    if "mask_spec" in raw:
        kwargs["mask_spec"] = _mask_spec_from_json(raw["mask_spec"])
    if "mask_keep" in raw:
        kwargs["mask_keep"] = raw["mask_keep"]
    if "nodata" in raw:
        kwargs["nodata"] = raw["nodata"]
    if "mosaic_method" in raw:
        kwargs["mosaic_method"] = raw["mosaic_method"]
    if "scale" in raw:
        kwargs["scale"] = raw["scale"]
    if "radiometry_bands" in raw:
        rb = raw["radiometry_bands"]
        kwargs["radiometry_bands"] = tuple(rb) if rb is not None else None
    if "band_aliases" in raw:
        # Dict insertion order == JSON object key order (Python `dict`/`json` both
        # preserve it), so this exactly inverts `to_json`'s `dict(decl.band_aliases)`
        # without needing to sort -- a `CollectionDeclaration` compares equal to its own
        # JSON round-trip.
        kwargs["band_aliases"] = tuple(raw["band_aliases"].items())
    if "requires_subscription_key" in raw:
        kwargs["requires_subscription_key"] = raw["requires_subscription_key"]
    if "supports_cloud_cover" in raw:
        kwargs["supports_cloud_cover"] = raw["supports_cloud_cover"]
    if "mosaic_partition" in raw:
        kwargs["mosaic_partition"] = tuple(raw["mosaic_partition"])
    if "partition_policy" in raw:
        kwargs["partition_policy"] = raw["partition_policy"]
    return CollectionDeclaration(**kwargs)


def to_attrs(gdf, decl: CollectionDeclaration) -> None:
    """Stamp `decl` onto `gdf.attrs[ATTRS_KEY]` as a plain JSON-able dict, never the
    dataclass itself. Mutates `gdf.attrs` in place."""
    gdf.attrs[ATTRS_KEY] = to_json(decl)


def from_attrs(gdf) -> CollectionDeclaration | None:
    """Read the stamped declaration back off `gdf.attrs[ATTRS_KEY]`, or `None`
    if `gdf` carries no stamp."""
    raw = gdf.attrs.get(ATTRS_KEY)
    if raw is None:
        return None
    return from_json(raw)


def digest(decl: CollectionDeclaration) -> str:
    """A short, stable hash of `decl`'s full JSON form -- used by
    `fsd.workflows.create_datacube.params_key` so any collection-level change
    (mask classes, nodata, reference band, ...) correctly invalidates cached cube
    paths, which a single S2-shaped mask-classes verb parameter never could."""
    import hashlib
    import json

    raw = json.dumps(to_json(decl), sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:8]
