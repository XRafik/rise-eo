"""Tests for rise_eo.catalog.declaration's serialization (spec 35 §2a/§3).

Pure functions, no I/O: `to_json`/`from_json` (dataclass <-> JSON dict) and
`to_attrs`/`from_attrs` (JSON dict <-> `GeoDataFrame.attrs`).
"""

import json
from unittest import mock

import geopandas as gpd
import pytest
import shapely

from rise_eo.catalog import declaration as declaration_module
from rise_eo.catalog.declaration import (
    S2_L2A_DECLARATION,
    CollectionDeclaration,
    MaskSpec,
)


def test_to_json_from_json_round_trip_exact():
    decl = CollectionDeclaration(
        reference_band="B04",
        mask_spec=MaskSpec(band="QA", classes=(1, 2, 3)),
        mask_keep=True,
        nodata=7,
        mosaic_method="median",
    )
    assert declaration_module.from_json(declaration_module.to_json(decl)) == decl
    # JSON-level round trip too (spec 35 §3): to_json(from_json(x)) == x.
    raw = declaration_module.to_json(decl)
    assert declaration_module.to_json(declaration_module.from_json(raw)) == raw


def test_to_json_from_json_round_trip_s2_default():
    assert declaration_module.from_json(declaration_module.to_json(S2_L2A_DECLARATION)) == S2_L2A_DECLARATION


def test_mask_spec_classes_rehydrate_as_tuple_not_list():
    decl = CollectionDeclaration(mask_spec=MaskSpec(band="SCL", classes=(0, 1, 3)))
    raw = declaration_module.to_json(decl)
    assert raw["mask_spec"]["classes"] == [0, 1, 3]  # JSON array on the wire
    back = declaration_module.from_json(raw)
    assert isinstance(back.mask_spec.classes, tuple)
    assert back.mask_spec.classes == (0, 1, 3)
    # frozen/hashable (a list would break this).
    hash(back)


def test_no_mask_source_mask_spec_null_round_trips_as_none():
    decl = CollectionDeclaration(reference_band=None, mask_spec=None)
    raw = declaration_module.to_json(decl)
    assert raw["mask_spec"] is None
    back = declaration_module.from_json(raw)
    assert back.mask_spec is None
    assert back == decl


def test_to_json_is_plain_json_able_dict():
    """spec 35 §2a: everything to_json produces must survive json.dumps -- no
    dataclass leaks into the wire format."""
    raw = declaration_module.to_json(S2_L2A_DECLARATION)
    json.dumps(raw)  # must not raise


def test_from_json_missing_version_raises():
    raw = declaration_module.to_json(S2_L2A_DECLARATION)
    del raw["fsd_declaration_version"]
    with pytest.raises(ValueError, match="fsd_declaration_version"):
        declaration_module.from_json(raw)


def test_from_json_newer_version_raises():
    raw = declaration_module.to_json(S2_L2A_DECLARATION)
    raw["fsd_declaration_version"] = declaration_module.RISE_EO_DECLARATION_VERSION + 1
    with pytest.raises(ValueError, match="newer rise-eo"):
        declaration_module.from_json(raw)


def test_from_json_unknown_field_raises():
    raw = declaration_module.to_json(S2_L2A_DECLARATION)
    raw["not_a_real_field"] = 1
    with pytest.raises(ValueError, match="unknown field"):
        declaration_module.from_json(raw)


def test_from_json_non_object_raises_clearly():
    """A hand-edited/corrupt footer must fail with a message about the shape, not
    an incidental TypeError from iterating an int (spec 35 §3's fail-loudly rule)."""
    with pytest.raises(ValueError, match="must be a JSON object"):
        declaration_module.from_json([1, 2, 3])

    raw = declaration_module.to_json(S2_L2A_DECLARATION)
    raw["mask_spec"] = 5
    with pytest.raises(ValueError, match="mask_spec must be a JSON object"):
        declaration_module.from_json(raw)


def test_from_json_missing_optional_field_takes_dataclass_default():
    raw = declaration_module.to_json(S2_L2A_DECLARATION)
    del raw["mask_keep"]
    back = declaration_module.from_json(raw)
    assert back.mask_keep == CollectionDeclaration.__dataclass_fields__["mask_keep"].default


def test_to_attrs_from_attrs_round_trip():
    gdf = gpd.GeoDataFrame({"id": ["a"]}, geometry=[shapely.box(0, 0, 1, 1)], crs="EPSG:4326")
    decl = CollectionDeclaration(reference_band="B04")
    declaration_module.to_attrs(gdf, decl)
    assert declaration_module.from_attrs(gdf) == decl


def test_from_attrs_no_stamp_returns_none():
    gdf = gpd.GeoDataFrame({"id": ["a"]}, geometry=[shapely.box(0, 0, 1, 1)], crs="EPSG:4326")
    assert declaration_module.from_attrs(gdf) is None


def test_to_attrs_never_puts_the_dataclass_in_attrs():
    """spec 35 §2a's future-geopandas guard: the on-attrs value is a dict, and
    it (and everything rise-eo puts in .attrs) must be JSON-able."""
    gdf = gpd.GeoDataFrame({"id": ["a"]}, geometry=[shapely.box(0, 0, 1, 1)], crs="EPSG:4326")
    declaration_module.to_attrs(gdf, S2_L2A_DECLARATION)
    assert not isinstance(gdf.attrs[declaration_module.ATTRS_KEY], CollectionDeclaration)
    for value in gdf.attrs.values():
        json.dumps(value)  # must not raise


# --- spec 58 AC10: the v1 <-> v2 footer contract ---------------------------------------

# A verbatim v1 footer, as a pre-spec-58 rise-eo wrote it: version 1, only the six v1 fields,
# and a `mask_spec` with no `bits`. Frozen as a literal on purpose -- deriving it from
# today's `to_json` would silently track any future field change and stop being the v1
# shape it exists to pin.
_V1_FOOTER = {
    "fsd_declaration_version": 1,
    "reference_band": "B08",
    "native_grid": False,
    "mask_spec": {"band": "SCL", "mask_type": "categorical_classes",
                  "classes": [0, 1, 3, 7, 8, 9, 10]},
    "mask_keep": False,
    "nodata": 0,
    "mosaic_method": "median",
}

_V1_FIELDS = ("reference_band", "native_grid", "mask_spec", "mask_keep", "nodata",
              "mosaic_method")


def test_from_json_parses_a_v1_footer():
    """Spec 58 AC10 (first half): a catalog stamped by a pre-spec-58 rise-eo still reads.
    Every v2-only field takes its dataclass default, which for S2 is exactly what the
    v1-era code meant -- `scale=1.0`, `radiometry_bands=None`, no aliases."""
    decl = declaration_module.from_json(_V1_FOOTER)
    assert decl.reference_band == "B08"
    assert decl.mask_spec == MaskSpec(band="SCL", classes=(0, 1, 3, 7, 8, 9, 10))
    assert decl.mask_spec.bits == ()
    assert decl.scale == CollectionDeclaration.__dataclass_fields__["scale"].default
    assert decl.radiometry_bands is None
    assert decl.band_aliases == ()


def test_v2_footer_read_by_v1_era_code_raises_version_mismatch_not_unknown_field():
    """Spec 58 AC10 (second half): the version check must run BEFORE the unknown-field
    check, or a v1-era rise-eo meeting a v2 footer reports "unknown field ['scale', ...]" --
    which reads as a corrupt catalog -- instead of "upgrade rise-eo" (spec 35 §5a).
    Simulated by shrinking this module's known-version/known-field sets back to v1's."""
    v2_footer = declaration_module.to_json(S2_L2A_DECLARATION)
    assert v2_footer["fsd_declaration_version"] == 2

    with mock.patch.object(declaration_module, "RISE_EO_DECLARATION_VERSION", 1):
        with mock.patch.object(declaration_module, "_DECLARATION_FIELDS", _V1_FIELDS):
            with pytest.raises(ValueError, match="newer rise-eo") as exc:
                declaration_module.from_json(v2_footer)
    assert "unknown field" not in str(exc.value)
