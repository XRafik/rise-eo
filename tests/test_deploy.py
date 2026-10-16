"""Spec 51 §9 step 2 — `rise_eo.deploy` (D5/D6/D7).

One test group per acceptance criterion this step owns (handoff-spec51-step2-deploy.md):
AC1, AC2, AC3, AC5, AC7, AC8, AC9, AC10 (the record half), AC13. D6's live-adapter refusal
is included too (small, and the natural place to check it against the real verb).

A bundle deployable end-to-end needs BOTH a `code` block and `requirements` (D6), so fixtures
here write a real importable adapter module (mirrors `tests/test_bundle_transparency.py`)
rather than `test_registry.py`'s `code=False` shortcut, which is deliberately undeployable.

AC7's "runs verify_image itself" path is exercised with the same fake `MLClient` +
`azure.ai.ml.command` injection seam `tests/test_bundle_transparency.py`/`test_infer_aml.py`
use (spec 36 D3 invariant 3): no network, no Azure SDK import. Every other AC uses
`verified=<dict>` (D5's other path) instead, which needs no AML machinery at all -- that is
what makes AC7 the one test that actually proves the enforcement gate calls `verify_image`.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import types

import pytest

from rise_eo import api
from rise_eo.model import bundle, registry
from rise_eo.storage import fs
from rise_eo.workflows import runners

ADAPTER_SRC = '''
from rise_eo.model.adapter import BaseModelAdapter


class TinyAdapter(BaseModelAdapter):
    required_bands = ["B04", "B08"]
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["klass"]

    def load(self):
        self.loaded = True

    def predict(self, X):
        return X
'''


@pytest.fixture
def importable(tmp_path, monkeypatch):
    def _make(rel: str, src: str = ADAPTER_SRC, root: str | None = None) -> str:
        root_dir = tmp_path / (root or "srcroot")
        target = root_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(src))
        monkeypatch.syspath_prepend(str(root_dir))
        return str(root_dir)

    yield _make


def _purge(*names: str) -> None:
    for n in names:
        sys.modules.pop(n, None)


def _deployable_bundle(tmp_path, importable, dst="b", requirements=("packaging",), modname="tdmod") -> str:
    """A bundle with both a `code` block and `requirements` -- the only shape `deploy` accepts."""
    importable(f"{modname}.py")
    mod = sys.modules.get(modname) or __import__(modname)
    try:
        return bundle.save(
            mod.TinyAdapter(), {}, str(tmp_path / dst),
            requirements=list(requirements) if requirements else None, verbose=False,
        )
    finally:
        _purge(modname)


def _matching_verified(bundle_path: str, environment: str) -> dict:
    """A `verified=` dict that D5 will accept for THIS bundle/environment -- no AML call.

    `bundle_digest` is what D5 matches on (`verify_image` records it at verification time);
    `bundle_path` is evidence only, and deploy must never re-digest it -- see
    `test_deploy_refuses_a_verified_result_whose_bundle_was_overwritten_in_place`."""
    return {
        "step": "verify_image", "status": "ok", "pass": True,
        "metrics": {
            "bundle_path": bundle_path,
            "bundle_digest": registry.content_digest(bundle_path),
            "environment": environment,
        },
        "expected": {}, "error": None,
    }


# --- AC1: deploy publishes v1 and the ref works through the public verb ------------------


def test_deploy_publishes_v1_and_run_inference_accepts_the_ref(tmp_path, importable):
    bpath = _deployable_bundle(tmp_path, importable)
    registry_root = str(tmp_path / "registry")

    ref = api.deploy(
        bpath, name="crop-rf", registry=registry_root, environment="rise-eo-infer-sklearn:6",
        verified=_matching_verified(bpath, "rise-eo-infer-sklearn:6"),
    )

    assert ref == "crop-rf:1"
    resolved = registry.resolve(ref, registry_root)
    assert resolved.version == 1
    assert fs.exists(os.path.join(resolved.path, bundle.BUNDLE_MANIFEST))

    # AC6's own proof pattern: reaching a LATER preflight error is what shows the ref resolved
    # (an unresolved ref dies at `_model_spec`'s `bundle.json` read, not here).
    cubes = tmp_path / "cubes"
    cubes.mkdir()
    with pytest.raises(api.PreflightError, match="no inference datacubes"):
        api.run_inference(
            model=ref, registry=registry_root,
            inference_datacubes=str(cubes), output_folderpath=str(tmp_path / "out"),
        )


# --- spec 52 AC8: spec 51 AC12 holds for real, against a URL registry -------------------


def test_deploy_set_alias_resolve_run_inference_against_a_url_registry(tmp_path, importable):
    """spec 52 AC8: the full `deploy` -> `set_alias` -> `resolve` -> `run_inference` chain
    behaves identically against a local registry path and a `memory://` one. This is spec 51
    AC12 actually exercised end-to-end (not just `publish`/`resolve`, see
    `test_registry.py::test_publish_resolve_round_trip_against_a_url_registry`) -- blocked by
    #88 until spec 52 D1 removed the directory rename `_write_new_version` used to hang on
    against any non-local backend."""
    bpath = _deployable_bundle(tmp_path, importable)
    registry_root = "memory://ac8-registry"

    ref = api.deploy(
        bpath, name="crop-rf", registry=registry_root, environment="rise-eo-infer-sklearn:6",
        alias="champion", verified=_matching_verified(bpath, "rise-eo-infer-sklearn:6"),
    )
    assert ref == "crop-rf:1"

    registry.set_alias("crop-rf", "staging", 1, registry_root)
    resolved = registry.resolve("crop-rf@staging", registry_root)
    assert resolved.version == 1
    assert fs.exists(os.path.join(resolved.path, bundle.BUNDLE_MANIFEST))

    # same proof pattern as AC1's local test: a LATER preflight error shows the ref resolved.
    cubes = tmp_path / "cubes"
    cubes.mkdir()
    with pytest.raises(api.PreflightError, match="no inference datacubes"):
        api.run_inference(
            model="crop-rf@champion", registry=registry_root,
            inference_datacubes=str(cubes), output_folderpath=str(tmp_path / "out"),
        )


# --- AC2: deploying identical content twice is idempotent -------------------------------


def test_deploy_identical_content_twice_is_idempotent(tmp_path, importable):
    bpath = _deployable_bundle(tmp_path, importable)
    registry_root = str(tmp_path / "registry")
    verified = _matching_verified(bpath, "env:1")

    ref1 = api.deploy(bpath, name="crop-rf", registry=registry_root, environment="env:1",
                      verified=verified)
    before = fs.ls(os.path.join(registry_root, "crop-rf"))

    ref2 = api.deploy(bpath, name="crop-rf", registry=registry_root, environment="env:1",
                      verified=verified)
    after = fs.ls(os.path.join(registry_root, "crop-rf"))

    assert ref2 == ref1
    assert len(after) == len(before)


# --- AC3: deploying changed content creates v2; v1 is untouched -------------------------


def test_deploy_changed_content_creates_v2_and_leaves_v1_untouched(tmp_path, importable):
    registry_root = str(tmp_path / "registry")
    b1 = _deployable_bundle(tmp_path, importable, dst="b1", modname="tdmod1")
    ref1 = api.deploy(b1, name="crop-rf", registry=registry_root, environment="env:1",
                      verified=_matching_verified(b1, "env:1"))
    assert ref1 == "crop-rf:1"

    b2 = _deployable_bundle(tmp_path, importable, dst="b2", requirements=["packaging", "numpy"],
                            modname="tdmod2")
    ref2 = api.deploy(b2, name="crop-rf", registry=registry_root, environment="env:1",
                      verified=_matching_verified(b2, "env:1"))
    assert ref2 == "crop-rf:2"

    v1_path = registry.version_path(registry_root, "crop-rf", 1)
    with fs.open(os.path.join(v1_path, bundle.BUNDLE_MANIFEST), "r") as f:
        v1_manifest_in_registry = json.load(f)
    with open(os.path.join(b1, bundle.BUNDLE_MANIFEST)) as f:
        v1_manifest_source = json.load(f)
    assert v1_manifest_in_registry == v1_manifest_source


# --- AC5: alias is written on deploy; redeploying with the same alias just repoints -----


def test_deploy_alias_repoints_without_touching_either_version(tmp_path, importable):
    registry_root = str(tmp_path / "registry")
    b1 = _deployable_bundle(tmp_path, importable, dst="b1", modname="tdmod3")
    api.deploy(b1, name="crop-rf", registry=registry_root, environment="env:1", alias="champion",
              verified=_matching_verified(b1, "env:1"))
    v1_manifest_path = os.path.join(registry.version_path(registry_root, "crop-rf", 1),
                                    bundle.BUNDLE_MANIFEST)
    with fs.open(v1_manifest_path, "r") as f:
        v1_before = f.read()

    b2 = _deployable_bundle(tmp_path, importable, dst="b2", requirements=["packaging", "numpy"],
                            modname="tdmod4")
    api.deploy(b2, name="crop-rf", registry=registry_root, environment="env:1", alias="champion",
              verified=_matching_verified(b2, "env:1"))

    assert registry.resolve("crop-rf@champion", registry_root).version == 2
    with fs.open(v1_manifest_path, "r") as f:
        v1_after = f.read()
    assert v1_after == v1_before


# --- AC7: deploy runs verify_image itself and refuses on pass=False ---------------------


class _NS(types.SimpleNamespace):
    pass


class _FakeMLClient:
    def __init__(self, job_statuses: dict[str, str]):
        self._job_statuses = job_statuses
        self.submitted: list = []
        self.compute = _NS(get=lambda cluster: _NS(provisioning_state="Succeeded", max_instances=4))
        self.environments = _NS(get=lambda **kw: _NS())
        self.jobs = _NS(create_or_update=self._create_or_update, get=self._get)

    def _create_or_update(self, job):
        idx = len(self.submitted)
        name = f"job-{idx}"
        self.submitted.append((name, job))
        return _NS(name=name)

    def _get(self, name):
        idx = int(name.rsplit("-", 1)[1])
        return _NS(status=list(self._job_statuses.values())[idx])


@pytest.fixture
def fake_aml_command(monkeypatch):
    def _cmd(**kwargs):
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(runners, "_import_aml_command", lambda: _cmd)
    return _cmd


def _status_url(root: str, run_id: str) -> str:
    return f"{root}/_verify_image/{run_id}/_status/smoke.json"


def test_deploy_runs_verify_image_and_publishes_on_pass(tmp_path, importable, fake_aml_command):
    bpath = _deployable_bundle(tmp_path, importable, modname="tdmod5")
    registry_root = str(tmp_path / "registry")
    aml_root = "memory://deploy_ok/root"
    ml_client = _FakeMLClient({"smoke": "Completed"})
    with fs.open(_status_url(aml_root, "runok"), "w") as f:
        json.dump({"status": "ok", "error": None}, f)

    ref = api.deploy(
        bpath, name="crop-rf", registry=registry_root, environment="rise-eo-infer-sklearn:3",
        runner_kwargs=dict(cluster="c", root=aml_root, identity_client_id="x",
                           ml_client=ml_client, run_id="runok"),
    )

    assert ref == "crop-rf:1"
    v1_path = registry.version_path(registry_root, "crop-rf", 1)
    with fs.open(os.path.join(v1_path, registry.DEPLOY_FILE), "r") as f:
        record = json.load(f)
    assert record["verified"]["pass"] is True


def test_deploy_refuses_on_verify_image_failure_and_creates_no_version(
    tmp_path, importable, fake_aml_command,
):
    bpath = _deployable_bundle(tmp_path, importable, modname="tdmod6")
    registry_root = str(tmp_path / "registry")
    aml_root = "memory://deploy_fail/root"
    ml_client = _FakeMLClient({"smoke": "Completed"})
    with fs.open(_status_url(aml_root, "runfail"), "w") as f:
        json.dump({"status": "fail", "error": "predict() raised"}, f)

    # verify_image only populates the top-level `error` on a driver-detected failure (bad
    # code, stale wheel, no status file) -- a smoke job that itself ran and reported
    # pass=False leaves `error=None` and the diagnosis in `metrics["smoke_error"]` instead.
    # D5's "surfaces that result's own error" therefore has to read BOTH, or the refusal
    # says "error: None" and tells the operator nothing about why their model failed.
    with pytest.raises(api.PreflightError, match=r"has not been proven.*predict\(\) raised"):
        api.deploy(
            bpath, name="crop-rf", registry=registry_root, environment="rise-eo-infer-sklearn:3",
            runner_kwargs=dict(cluster="c", root=aml_root, identity_client_id="x",
                               ml_client=ml_client, run_id="runfail"),
        )

    assert not fs.exists(os.path.join(registry_root, "crop-rf"))


# --- AC8: verified= only skips re-verification when digest AND environment both match ---


def test_deploy_accepts_a_matching_verified_result_without_running_verify_image(
    tmp_path, importable, monkeypatch,
):
    bpath = _deployable_bundle(tmp_path, importable, modname="tdmod7")
    registry_root = str(tmp_path / "registry")

    def _forbidden(*a, **kw):
        raise AssertionError("verify_image was called despite a matching verified= result")

    monkeypatch.setattr(api, "_verify_image", _forbidden)

    ref = api.deploy(
        bpath, name="crop-rf", registry=registry_root, environment="env:1",
        verified=_matching_verified(bpath, "env:1"),
    )
    assert ref == "crop-rf:1"


def test_deploy_refuses_a_verified_result_from_a_different_environment(tmp_path, importable):
    bpath = _deployable_bundle(tmp_path, importable, modname="tdmod8")
    registry_root = str(tmp_path / "registry")

    with pytest.raises(api.PreflightError, match="stale or does not match"):
        api.deploy(
            bpath, name="crop-rf", registry=registry_root, environment="env:1",
            verified=_matching_verified(bpath, "env:OTHER"),
        )
    assert not fs.exists(os.path.join(registry_root, "crop-rf"))


def test_deploy_refuses_a_verified_result_for_different_bundle_content(tmp_path, importable):
    registry_root = str(tmp_path / "registry")
    b1 = _deployable_bundle(tmp_path, importable, dst="b1", modname="tdmod9")
    b2 = _deployable_bundle(tmp_path, importable, dst="b2", requirements=["packaging", "numpy"],
                            modname="tdmod10")

    with pytest.raises(api.PreflightError, match="stale or does not match"):
        api.deploy(
            b1, name="crop-rf", registry=registry_root, environment="env:1",
            verified=_matching_verified(b2, "env:1"),  # verified a DIFFERENT bundle's content
        )
    assert not fs.exists(os.path.join(registry_root, "crop-rf"))


def test_deploy_refuses_a_verified_result_whose_bundle_was_overwritten_in_place(
    tmp_path, importable,
):
    """The regression this whole check exists for. `bundle.save` overwrites in place (spec 51
    §1 H1), so verify -> retrain -> re-save -> deploy leaves a `_result.json` that names the
    RIGHT path holding the WRONG content. Matching on `metrics["bundle_path"]` re-digested at
    deploy time compares the new content with itself and passes, registering "this image ran
    this bundle" for content the image never saw. Matching on the RECORDED `bundle_digest`
    catches it."""
    importable("tdstale.py")
    import tdstale
    bpath = str(tmp_path / "demo_bundle")
    try:
        bundle.save(tdstale.TinyAdapter(), {}, bpath, requirements=["packaging"], verbose=False)
        verified_old = _matching_verified(bpath, "env:6")          # verified content A
        bundle.save(tdstale.TinyAdapter(), {}, bpath,              # retrain: same path, content B
                    requirements=["packaging", "numpy"], verbose=False)
    finally:
        _purge("tdstale")
    assert registry.content_digest(bpath) != verified_old["metrics"]["bundle_digest"]

    registry_root = str(tmp_path / "registry")
    with pytest.raises(api.PreflightError, match="stale or does not match"):
        api.deploy(bpath, name="crop-rf", registry=registry_root, environment="env:6",
                   verified=verified_old)
    assert not fs.exists(os.path.join(registry_root, "crop-rf"))


def test_deploy_reads_verified_from_a_result_json_path(tmp_path, importable):
    """`verified=` takes the dict OR the path it was written to; a path that is not there is
    the verb's own PreflightError, not a bare FileNotFoundError from the storage seam."""
    bpath = _deployable_bundle(tmp_path, importable, modname="tdmodpath")
    result_json = tmp_path / "_result.json"
    with open(result_json, "w") as f:
        json.dump(_matching_verified(bpath, "env:1"), f)
    registry_root = str(tmp_path / "registry")

    assert api.deploy(bpath, name="crop-rf", registry=registry_root, environment="env:1",
                      verified=str(result_json)) == "crop-rf:1"

    with pytest.raises(api.PreflightError, match="cannot read that as a verify_image"):
        api.deploy(bpath, name="crop-rf", registry=registry_root, environment="env:1",
                   verified=str(tmp_path / "nope.json"))


def test_deploy_refuses_a_verified_result_that_records_no_bundle_digest(tmp_path, importable):
    """A result that cannot say WHAT it verified is a mismatch, not a pass."""
    bpath = _deployable_bundle(tmp_path, importable, modname="tdmod14")
    no_digest = _matching_verified(bpath, "env:1")
    del no_digest["metrics"]["bundle_digest"]
    registry_root = str(tmp_path / "registry")

    with pytest.raises(api.PreflightError, match="bundle_digest=None predates spec 51"):
        api.deploy(bpath, name="crop-rf", registry=registry_root, environment="env:1",
                   verified=no_digest)
    assert not fs.exists(os.path.join(registry_root, "crop-rf"))


def test_deploy_surfaces_a_matched_but_failing_verified_results_own_error(tmp_path, importable):
    """A `verified=` result that DOES match this bundle/environment but failed must still be
    honoured (D5) -- refused with ITS error, not folded into the generic mismatch message."""
    bpath = _deployable_bundle(tmp_path, importable, modname="tdmod11")
    registry_root = str(tmp_path / "registry")
    failing = {
        "step": "verify_image", "status": "fail", "pass": False,
        "metrics": {
            "bundle_path": bpath,
            "bundle_digest": registry.content_digest(bpath),
            "environment": "env:1",
        },
        "expected": {}, "error": "ModuleNotFoundError: no module named tinyadapter",
    }

    with pytest.raises(api.PreflightError, match="ModuleNotFoundError"):
        api.deploy(bpath, name="crop-rf", registry=registry_root, environment="env:1",
                  verified=failing)
    assert not fs.exists(os.path.join(registry_root, "crop-rf"))


# --- AC9: refuses a bundle with no requirements / no code block -------------------------


def test_deploy_refuses_a_bundle_with_no_requirements(tmp_path, importable):
    importable("tdreqmod.py")
    import tdreqmod
    try:
        bpath = bundle.save(tdreqmod.TinyAdapter(), {}, str(tmp_path / "b"), verbose=False)
    finally:
        _purge("tdreqmod")
    registry_root = str(tmp_path / "registry")

    with pytest.raises(api.PreflightError, match=r"bundle\.save\(\.\.\., requirements="):
        api.deploy(bpath, name="crop-rf", registry=registry_root, environment="env:1",
                  verified=_matching_verified(bpath, "env:1"))
    assert not fs.exists(os.path.join(registry_root, "crop-rf"))


def test_deploy_refuses_a_bundle_with_no_code_block(tmp_path):
    joblib = pytest.importorskip("joblib")
    bpath = bundle.save(joblib.Parallel, {}, str(tmp_path / "b"), code=False,
                        requirements=["packaging"], verbose=False)
    registry_root = str(tmp_path / "registry")

    with pytest.raises(api.PreflightError, match=r"bundle\.save\(\.\.\., code="):
        api.deploy(bpath, name="crop-rf", registry=registry_root, environment="env:1",
                  verified=_matching_verified(bpath, "env:1"))
    assert not fs.exists(os.path.join(registry_root, "crop-rf"))


# --- AC1: a name that cannot round-trip through a ref is refused BEFORE verification -----


@pytest.mark.parametrize("name", ["crop:rf", "crop@rf", "crop/rf", ".hidden", ""])
def test_deploy_refuses_a_name_a_ref_cannot_carry(tmp_path, importable, name, monkeypatch):
    """AC1 says the returned ref is one `run_inference` accepts unchanged. `crop/rf:1` is
    path-shaped so `_is_ref_shaped` never resolves it; `crop:rf:1` re-splits at the wrong
    separator. Both would publish fine and hand back an unusable ref -- refuse at the name.
    And refuse before `verify_image`, so a bad name costs no AML node."""
    bpath = _deployable_bundle(tmp_path, importable, modname="tdmodname")

    def _forbidden(*a, **kw):
        raise AssertionError("verify_image was called despite an unusable name")

    monkeypatch.setattr(api, "_verify_image", _forbidden)
    registry_root = str(tmp_path / "registry")

    with pytest.raises(api.PreflightError, match="rise_eo.deploy"):
        api.deploy(bpath, name=name, registry=registry_root, environment="env:1")
    assert not fs.exists(registry_root)


def test_a_deployed_ref_is_ref_shaped_so_run_inference_resolves_it(tmp_path, importable):
    """The positive half: the string `deploy` returns really does pass the D4 shape gate."""
    bpath = _deployable_bundle(tmp_path, importable, modname="tdmodshape")
    registry_root = str(tmp_path / "registry")
    ref = api.deploy(bpath, name="crop_rf-2026", registry=registry_root, environment="env:1",
                     verified=_matching_verified(bpath, "env:1"))
    assert api._is_ref_shaped(ref)
    assert registry.parse_ref(ref)[0] == "crop_rf-2026"


# --- D6: a live adapter is refused, naming rise_eo.model.bundle.save ------------------------


def test_deploy_refuses_a_live_adapter(tmp_path, importable):
    importable("tdlivemod.py")
    import tdlivemod
    try:
        with pytest.raises(api.PreflightError, match="rise_eo.model.bundle.save"):
            api.deploy(tdlivemod.TinyAdapter(), name="crop-rf", registry=str(tmp_path / "registry"),
                      environment="env:1")
    finally:
        _purge("tdlivemod")


# --- AC14 second half: deploy refuses verify_adapter's REAL auto-saved bundle -----------


def test_deploy_refuses_verify_adapters_real_auto_saved_bundle(tmp_path, importable, monkeypatch):
    """AC14 second half: D6's refusal must hold against `verify_adapter`'s real, on-disk
    auto-saved bundle -- reached through `result["metrics"]["bundle_path"]`, not a hand-built
    folder standing in for it (`_ensure_bundle` passes no `requirements=`, so this bundle
    genuinely lacks them)."""
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely.geometry

    import rise_eo
    import rise_eo.grid as grid_mod
    from rise_eo.workflows import create_datacube as _create_datacube
    from rise_eo.workflows import infer_only_task as _infer_only_task

    modname = "tdac14adapter"
    importable(f"{modname}.py", src="""
from rise_eo.model.adapter import BaseModelAdapter


class TinyAdapter(BaseModelAdapter):
    required_bands = ["B04", "B08"]
    n_timestamps = 0
    output_dtype = "uint8"
    output_nodata = 255
    output_band_names = ["klass"]

    def features(self, data5d, band_indices):
        return data5d, band_indices

    def predict(self, X_chunk):
        return np.zeros(X_chunk.shape[0], dtype="uint8")
""")
    mod = __import__(modname)

    roi = gpd.GeoDataFrame({"geometry": [shapely.geometry.box(0, 0, 1, 1)]}, crs="EPSG:4326")
    grids = gpd.GeoDataFrame(
        {"id": ["cell_a"]}, geometry=[shapely.geometry.box(0, 0, 1, 1)], crs="EPSG:4326",
    )
    monkeypatch.setattr(grid_mod, "roi_to_s2_grids", lambda *a, **kw: grids.copy())

    catalog_gdf = gpd.GeoDataFrame(
        {"geometry": [shapely.geometry.box(0, 0, 1, 1)]}, crs="EPSG:4326",
    )
    catalog_gdf["timestamp"] = pd.to_datetime(["2018-01-01"], utc=True)
    cat = str(tmp_path / "sentinel-2-l2a" / "catalog.parquet")
    fs.write_parquet(cat, catalog_gdf)

    def fake_run_create_datacube(*, csv_filepath, **kw):
        cube_dir = os.path.join(os.path.dirname(csv_filepath), "cellbuild")
        fs.makedirs(cube_dir)
        fs.save_npy(os.path.join(cube_dir, "datacube.npy"),
                    np.zeros((3, 2, 2, 2), dtype="float32"))
        fs.save_npy(os.path.join(cube_dir, "metadata.pickle.npy"),
                    {"timestamps": list(range(3)), "bands": ["B04", "B08"]}, allow_pickle=True)
        pd.DataFrame(
            {"datacube_filepath": [os.path.join(cube_dir, "datacube.npy")]}
        ).to_csv(csv_filepath, index=False)

    def fake_run_infer_only(input_csv, rows, bundle_path, **kw):
        import rasterio
        from rasterio.transform import from_origin

        df = pd.read_csv(input_csv)
        out_fp = str(df["output_filepath"].iloc[0])
        os.makedirs(os.path.dirname(out_fp), exist_ok=True)
        profile = {
            "driver": "GTiff", "height": 2, "width": 2, "count": 1, "dtype": "uint8",
            "crs": "EPSG:4326", "transform": from_origin(0, 1, 0.5, 0.5), "nodata": 255,
        }
        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(np.zeros((1, 2, 2), dtype="uint8"))
        return [out_fp]

    monkeypatch.setattr(_create_datacube, "run_create_datacube", fake_run_create_datacube)
    monkeypatch.setattr(_infer_only_task, "run_infer_only", fake_run_infer_only)

    export = str(tmp_path / "export")
    try:
        result = rise_eo.verify_adapter(
            mod.TinyAdapter(), roi=roi, catalog_filepath=cat,
            startdate="2018-01-01", enddate="2018-03-02", mosaic_days=20,
            bands=["B04", "B08"], export_folderpath=export, cell="cell_a",
        )
    finally:
        _purge(modname)

    assert result["pass"] is True
    bundle_path = result["metrics"]["bundle_path"]
    assert bundle_path == os.path.join(export, "_bundle")
    assert os.path.exists(os.path.join(bundle_path, "bundle.json"))

    with pytest.raises(api.PreflightError, match="requirements"):
        api.deploy(bundle_path, name="crop-rf", registry=str(tmp_path / "registry"),
                  environment="env:1")


# --- AC10 (record half): _deploy.json records name/version/digest/environment/verified --


def test_deploy_writes_the_deploy_record(tmp_path, importable):
    bpath = _deployable_bundle(tmp_path, importable, modname="tdmod12")
    registry_root = str(tmp_path / "registry")
    verified = _matching_verified(bpath, "rise-eo-infer-sklearn:6")

    ref = api.deploy(bpath, name="crop-rf", registry=registry_root,
                     environment="rise-eo-infer-sklearn:6", verified=verified)

    v1_path = registry.version_path(registry_root, "crop-rf", 1)
    with fs.open(os.path.join(v1_path, registry.DEPLOY_FILE), "r") as f:
        record = json.load(f)

    assert record["name"] == "crop-rf"
    assert record["version"] == 1
    assert record["digest"] == registry.content_digest(bpath)
    assert record["environment"] == "rise-eo-infer-sklearn:6"
    assert record["verified"] == verified
    assert "deployed_at" in record and "fsd_version" in record
    assert ref == "crop-rf:1"


# --- AC13a: nothing deploy writes contains the registry root, an absolute path, or a URL ---


def test_deploy_writes_no_file_containing_the_registry_root(tmp_path, importable):
    bpath = _deployable_bundle(tmp_path, importable, modname="tdmod13")
    registry_root = str(tmp_path / "registry")

    api.deploy(bpath, name="crop-rf", registry=registry_root, environment="env:1",
              alias="champion", verified=_matching_verified(bpath, "env:1"))

    root_str = registry_root
    for path, _size in fs.find_sizes(registry_root).items():
        if not (path.endswith(".json") or os.path.basename(path).startswith("_")):
            continue
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", errors="replace")
        assert root_str not in text
