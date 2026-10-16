"""Spec 56 D8 / AC7 -- `verify_image`'s `image_ref=`/`registry=` path.

Reuses the `importable` / `_purge` / `_FakeMLClient` / `fake_aml_command` conventions from
`tests/test_bundle_transparency.py` (spec 45), which this feature sits beside without
touching -- those tests, and spec 47's, run unmodified (see that file).
"""

from __future__ import annotations

import json
import sys
import textwrap
import types

import pytest

from rise_eo.image import registry as image_registry
from rise_eo.model import bundle
from rise_eo.model.verify_image import verify_image
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


class _NS(types.SimpleNamespace):
    pass


class _FakeMLClient:
    """Mirrors `tests/test_bundle_transparency.py`'s fixture of the same name."""

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


@pytest.fixture
def fake_aml_command(monkeypatch):
    def _cmd(**kwargs):
        import types
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(runners, "_import_aml_command", lambda: _cmd)
    return _cmd


def _status_url(root: str, run_id: str) -> str:
    return f"{root}/_verify_image/{run_id}/_status/smoke.json"


def _publish_definition(tmp_path, name, rise_eo_ref):
    registry_root = str(tmp_path / "image_registry")
    resolved = {
        "base": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04@sha256:" + "a" * 64,
        "rise_eo": rise_eo_ref, "extras": ["azure"], "extra_pip": [], "base_resolved": True,
    }
    from rise_eo.image import digest as digest_mod
    d = digest_mod.digest(resolved)
    image_registry.publish(name, registry_root, resolved, d)
    return registry_root


def test_verify_image_ref_current_rise_eo_passes(tmp_path, importable, fake_aml_command):
    importable("vref_mod.py")
    import vref_mod
    try:
        bdir = bundle.save(vref_mod.TinyAdapter(), {}, str(tmp_path / "b"), verbose=False)
    finally:
        _purge("vref_mod")

    rise_eo_ref = "git+https://github.com/nikhilsrajan/fsd@" + "c" * 40
    registry_root = _publish_definition(tmp_path, "rise-eo-infer-sklearn", rise_eo_ref)

    root = "memory://vimgref_ok/root"
    ml_client = _FakeMLClient({"smoke": "Completed"})
    with fs.open(_status_url(root, "runok"), "w") as f:
        json.dump({"status": "ok", "error": None}, f)

    def _fetch(rise_eo_ref_arg, relpath):
        assert rise_eo_ref_arg == rise_eo_ref
        return "def manifest_code_files(): pass\n"

    result = verify_image(
        bdir, environment="rise-eo-infer-sklearn:1", runner="aml",
        runner_kwargs=dict(cluster="c", root=root, identity_client_id="x",
                           ml_client=ml_client, run_id="runok"),
        image_ref="rise-eo-infer-sklearn:1", registry=registry_root,
        _fetch_rise_eo_source=_fetch,
    )

    assert result["pass"] is True
    assert result["metrics"]["image_ref_has_spec44"] is True


def test_verify_image_ref_stale_rise_eo_refuses_before_submission(tmp_path, importable, fake_aml_command):
    importable("vrefstale_mod.py")
    import vrefstale_mod
    try:
        bdir = bundle.save(vrefstale_mod.TinyAdapter(), {}, str(tmp_path / "b"), verbose=False)
    finally:
        _purge("vrefstale_mod")

    rise_eo_ref = "git+https://github.com/nikhilsrajan/fsd@" + "d" * 40
    registry_root = _publish_definition(tmp_path, "rise-eo-infer-sklearn", rise_eo_ref)

    ml_client = _FakeMLClient({"smoke": "Completed"})

    def _fetch(rise_eo_ref_arg, relpath):
        return "# a pre-spec-44 bundle.py, no manifest_code_files\n"

    result = verify_image(
        bdir, environment="rise-eo-infer-sklearn:1", runner="aml",
        runner_kwargs=dict(cluster="c", root="memory://vimgref_stale/root",
                           identity_client_id="x", ml_client=ml_client),
        image_ref="rise-eo-infer-sklearn:1", registry=registry_root,
        _fetch_rise_eo_source=_fetch,
    )

    assert result["pass"] is False
    assert "predates spec 44" in result["error"]
    assert ml_client.submitted == []


def test_verify_image_ref_prerename_image_raises_before_submission(tmp_path, importable, fake_aml_command):
    importable("vrefold_mod.py")
    import vrefold_mod
    try:
        bdir = bundle.save(vrefold_mod.TinyAdapter(), {}, str(tmp_path / "b"), verbose=False)
    finally:
        _purge("vrefold_mod")

    registry_root = str(tmp_path / "image_registry")
    resolved = {
        "base": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04@sha256:" + "a" * 64,
        "fsd": "git+https://github.com/nikhilsrajan/fsd@" + "e" * 40,
        "extras": ["azure"], "extra_pip": [], "base_resolved": True,
    }
    from rise_eo.image import digest as digest_mod
    image_registry.publish("rise-eo-infer-sklearn", registry_root, resolved, digest_mod.digest(resolved))

    ml_client = _FakeMLClient({"smoke": "Completed"})
    with pytest.raises(ValueError, match="built before the package was renamed"):
        verify_image(
            bdir, environment="rise-eo-infer-sklearn:1", runner="aml",
            runner_kwargs=dict(cluster="c", root="memory://vimgref_old/root",
                               identity_client_id="x", ml_client=ml_client),
            image_ref="rise-eo-infer-sklearn:1", registry=registry_root,
        )
    assert ml_client.submitted == []


def test_verify_image_ref_without_registry_raises(tmp_path, importable, fake_aml_command):
    importable("vrefnoreg_mod.py")
    import vrefnoreg_mod
    try:
        bdir = bundle.save(vrefnoreg_mod.TinyAdapter(), {}, str(tmp_path / "b"), verbose=False)
    finally:
        _purge("vrefnoreg_mod")

    ml_client = _FakeMLClient({"smoke": "Completed"})
    with pytest.raises(ValueError, match="requires registry="):
        verify_image(
            bdir, environment="rise-eo-infer-sklearn:1", runner="aml",
            runner_kwargs=dict(cluster="c", root="memory://vimgref_noreg/root",
                               identity_client_id="x", ml_client=ml_client),
            image_ref="rise-eo-infer-sklearn:1",
        )
    assert ml_client.submitted == []


def test_build_context_wins_over_image_ref(tmp_path, importable, fake_aml_command):
    """D8: `build_context` wins if both are given -- the checkout path is unchanged."""
    import zipfile

    importable("vrefboth_mod.py")
    import vrefboth_mod
    try:
        bdir = bundle.save(vrefboth_mod.TinyAdapter(), {}, str(tmp_path / "b"), verbose=False)
    finally:
        _purge("vrefboth_mod")

    build_ctx = tmp_path / "ctx"
    build_ctx.mkdir()
    wheel = build_ctx / "rise_eo-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("rise_eo/model/bundle.py", "def manifest_code_files(): pass\n")

    def _explode(rise_eo_ref_arg, relpath):
        raise AssertionError("image_ref must not be consulted when build_context is given")

    ml_client = _FakeMLClient({"smoke": "Completed"})
    root = "memory://vimgref_both/root"
    with fs.open(_status_url(root, "runboth"), "w") as f:
        json.dump({"status": "ok", "error": None}, f)

    result = verify_image(
        bdir, environment="rise-eo-infer-sklearn:1", runner="aml",
        runner_kwargs=dict(cluster="c", root=root, identity_client_id="x",
                           ml_client=ml_client, run_id="runboth"),
        build_context=str(build_ctx),
        image_ref="rise-eo-infer-sklearn:1", registry="memory://unused-registry",
        _fetch_rise_eo_source=_explode,
    )

    assert result["pass"] is True
    assert "image_ref_has_spec44" not in result["metrics"]
