"""Golden files for the two on-disk format versions (spec 102 D11, AC7).

`FSD_DECLARATION_VERSION` (collection declaration, in every catalog's Parquet footer) and
`BUNDLE_VERSION` (`bundle.json`) each have `tests/data/formats/<kind>.v<N>.json`.

* **Test A** writes a *fixed synthetic* example with today's code and compares it to the golden
  file for the current version. The example is pinned here, not read from S2 defaults, so a changed
  S2 default is not mistaken for a format change. Nothing in either manifest is volatile (no
  timestamps, digests, fsd version or paths), so no normalization is needed.
* **Test B** loads every supported version from its golden file. The bundle's list is
  `SUPPORTED_BUNDLE_VERSIONS`; the declaration has no such list (`from_json` accepts any version
  <= current), so its equivalent is `range(1, FSD_DECLARATION_VERSION + 1)`.

A second bundle golden, `bundle.v<N>.minimal.json`, pins the writer's other branches: `code_origin`
"installed" (saved with `code=False`), a `sequence` feature and no `requirements`.

To change a format: bump the constant, add `<kind>.v<N+1>.json` (copy the failing test's "got"
output), keep the old files. A bundle bump also needs `bundle.v<N+1>.minimal.json`.

Provenance: `declaration.v1.json` was produced by running the v1 `to_json` from git history
(`2398c13^`) on `SourceDeclaration(reference_band="B08", mask_spec=MaskSpec(band="SCL",
classes=(0, 1, 3, 8, 9, 10)), mask_keep=False, nodata=0, mosaic_method="median")` (input picked by
hand, not S2's). `bundle.v1.json` is `bundle.v2.json` minus `code`/`code_origin`/`requirements`, with
the version set to 1; checked against the v1 writer at `ca3833b^`, whose manifest is exactly
`fsd_bundle_version, adapter, artifacts, feature` + the five spec fields.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

from rise_eo.catalog import declaration
from rise_eo.catalog.declaration import CollectionDeclaration, MaskSpec
from rise_eo.model import bundle

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "data", "formats")


def _golden_path(kind: str, version: int, variant: str = "") -> str:
    return os.path.join(GOLDEN_DIR, f"{kind}.v{version}{variant}.json")


def _read_golden(kind: str, version: int, variant: str = "") -> dict:
    with open(_golden_path(kind, version, variant)) as f:
        return json.load(f)


def _assert_matches_golden(
    kind: str, const: str, version: int, got: dict, variant: str = ""
) -> None:
    path = _golden_path(kind, version, variant)
    rel = os.path.relpath(path, os.path.dirname(GOLDEN_DIR))
    assert os.path.exists(path), (
        f"{const} is {version} but {rel} does not exist. Add it with this content:\n"
        f"{json.dumps(got, indent=2)}"
    )
    with open(path) as f:
        want = json.load(f)
    assert got == want, (
        f"The {kind} format written by today's code differs from {rel}. If the change is "
        f"intended, bump {const} to {version + 1} and add {kind}.v{version + 1}{variant}.json with:\n"
        f"{json.dumps(got, indent=2)}"
    )


# --- declaration -----------------------------------------------------------------------------

# Every field set explicitly, to a value that differs from its default (so `from_json` dropping
# any field fails `decl == _DECLARATION`).
_DECLARATION = CollectionDeclaration(
    reference_band="B08",
    native_grid=True,
    mask_spec=MaskSpec(band="SCL", mask_type="bitmask", classes=(3, 8), bits=(1, 2)),
    mask_keep=True,
    nodata=7,
    mosaic_method="mean",
    scale=0.5,
    radiometry_bands=("B04", "B08"),
    band_aliases=(("red", "B04"), ("nir", "B08")),
    requires_subscription_key=True,
    supports_cloud_cover=True,
    mosaic_partition=("sat:orbit_state",),
    partition_policy="auto",
)


def test_declaration_format_matches_golden():
    version = declaration.FSD_DECLARATION_VERSION
    _assert_matches_golden(
        "declaration", "FSD_DECLARATION_VERSION", version, declaration.to_json(_DECLARATION)
    )


@pytest.mark.parametrize("version", range(1, declaration.FSD_DECLARATION_VERSION + 1))
def test_declaration_golden_files_still_load(version):
    decl = declaration.from_json(_read_golden("declaration", version))
    if version == 1:
        # v1 footers predate bits/scale/radiometry_bands/band_aliases/...; those take today's defaults.
        assert decl == CollectionDeclaration(
            reference_band="B08",
            mask_spec=MaskSpec(band="SCL", classes=(0, 1, 3, 8, 9, 10)),
            mask_keep=False, nodata=0, mosaic_method="median",
        )
    else:
        assert decl == _DECLARATION


# --- bundle ----------------------------------------------------------------------------------

_ADAPTER_SRC = '''
from rise_eo.model.adapter import BaseModelAdapter


class GoldenAdapter(BaseModelAdapter):
    required_bands = ["B04", "B08"]
    n_timestamps = 3
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["klass"]
    feature_sequence = None

    def load(self):
        self.loaded = True

    def predict(self, X):
        return X
'''


@pytest.fixture
def golden_adapter(tmp_path, monkeypatch):
    src_dir = tmp_path / "srcroot"
    src_dir.mkdir()
    (src_dir / "golden_adapter.py").write_text(_ADAPTER_SRC)
    monkeypatch.syspath_prepend(str(src_dir))
    import golden_adapter as mod

    yield mod.GoldenAdapter
    sys.modules.pop("golden_adapter", None)


def test_bundle_format_matches_golden(tmp_path, golden_adapter):
    model = tmp_path / "model.bin"
    model.write_bytes(b"x")
    bdir = bundle.save(
        golden_adapter(), {"model": str(model)}, str(tmp_path / "b"),
        requirements=["scikit-learn>=1.3"], verbose=False,
    )
    _assert_matches_golden(
        "bundle", "BUNDLE_VERSION", bundle.BUNDLE_VERSION, bundle.read_spec(bdir)
    )


_MINIMAL_ADAPTER_SRC = '''
from rise_eo.model.adapter import BaseModelAdapter


def ndvi(data, profile):
    return data, profile


class GoldenMinimalAdapter(BaseModelAdapter):
    required_bands = ["B04", "B08"]
    n_timestamps = 3
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["klass"]
    feature_sequence = [(ndvi, {})]

    def load(self):
        self.loaded = True

    def predict(self, X):
        return X
'''


@pytest.fixture
def golden_minimal_adapter(tmp_path, monkeypatch):
    src_dir = tmp_path / "minimal_srcroot"
    src_dir.mkdir()
    (src_dir / "golden_minimal_adapter.py").write_text(_MINIMAL_ADAPTER_SRC)
    monkeypatch.syspath_prepend(str(src_dir))
    import golden_minimal_adapter as mod

    yield mod.GoldenMinimalAdapter
    sys.modules.pop("golden_minimal_adapter", None)


def test_minimal_bundle_format_matches_golden(tmp_path, golden_minimal_adapter):
    model = tmp_path / "model.bin"
    model.write_bytes(b"x")
    bdir = bundle.save(
        golden_minimal_adapter(), {"model": str(model)}, str(tmp_path / "b"),
        code=False, verbose=False,
    )
    _assert_matches_golden(
        "bundle", "BUNDLE_VERSION", bundle.BUNDLE_VERSION, bundle.read_spec(bdir), ".minimal"
    )


@pytest.mark.parametrize("version", bundle.SUPPORTED_BUNDLE_VERSIONS)
def test_minimal_bundle_golden_still_loads(tmp_path, golden_minimal_adapter, version):
    if not os.path.exists(_golden_path("bundle", version, ".minimal")):
        pytest.skip(f"no minimal golden for bundle v{version}")
    manifest = _read_golden("bundle", version, ".minimal")
    bdir = tmp_path / "b"
    bdir.mkdir()
    (bdir / "bundle.json").write_text(json.dumps(manifest))
    (bdir / "model.bin").write_bytes(b"x")
    adapter = bundle.load(str(bdir))
    assert type(adapter).__name__ == "GoldenMinimalAdapter"
    assert adapter.loaded


def test_every_bundle_golden_is_a_supported_version():
    """Dropping a version from `SUPPORTED_BUNDLE_VERSIONS` must mean deleting its golden file."""
    assert bundle.BUNDLE_VERSION in bundle.SUPPORTED_BUNDLE_VERSIONS
    on_disk = {
        int(f.split(".v")[1].split(".")[0]) for f in os.listdir(GOLDEN_DIR) if f.startswith("bundle.v")
    }
    assert on_disk == set(bundle.SUPPORTED_BUNDLE_VERSIONS)


@pytest.mark.parametrize("version", bundle.SUPPORTED_BUNDLE_VERSIONS)
def test_bundle_golden_files_still_load(tmp_path, monkeypatch, golden_adapter, version):
    manifest = _read_golden("bundle", version)
    bdir = tmp_path / "b"
    bdir.mkdir()
    (bdir / "bundle.json").write_text(json.dumps(manifest))
    (bdir / "model.bin").write_bytes(b"x")
    if manifest.get("code"):  # v2 bundled: ship the adapter source, as `save` would
        code_dir = bdir / manifest["code"]["root"]
        code_dir.mkdir()
        (code_dir / "golden_adapter.py").write_text(_ADAPTER_SRC)
        sys.modules.pop("golden_adapter", None)  # force a fresh import from code/, not the fixture's copy
        src_dir = str(tmp_path / "srcroot")
        monkeypatch.setattr(sys, "path", [p for p in sys.path if p != src_dir])
    adapter = bundle.load(str(bdir))
    if manifest.get("code"):
        assert os.path.abspath(sys.modules["golden_adapter"].__file__).startswith(str(bdir))
    assert type(adapter).__name__ == "GoldenAdapter"
    assert adapter.loaded
