"""Spec 45 — bundle transparency, bundle validation, and image verification.

One test group per acceptance criterion in `specs/45-bundle-transparency-and-image-verification.md`
§4. Adapter modules are written to disk under `tmp_path` and imported by manipulating `sys.path`
(mirrors `tests/test_bundle_code.py`), so `save`'s auto-detection sees a real, importable module —
never `__main__`. `verify_image`'s AML path is exercised with the same fake `MLClient` +
`azure.ai.ml.command` injection seam `tests/test_infer_aml.py` uses (spec 36 D3 invariant 3): no
network, no Azure SDK import, in these tests.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import types

import pytest

from rise_eo.model import bundle, registry
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


@pytest.fixture
def importable(tmp_path, monkeypatch):
    """Write a module into a directory, put it on `sys.path`, clean `sys.modules` after."""
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


# --- AC1: save() reports what it embedded, by default; verbose=False silences it -----------


def test_save_reports_root_files_adapter_and_requirements(tmp_path, importable, capsys):
    importable("d1_mod.py")
    import d1_mod

    try:
        bdir = bundle.save(d1_mod.TinyAdapter(), {}, str(tmp_path / "b"),
                           requirements=["scikit-learn>=1.5"])
        out = capsys.readouterr().out

        assert "[bundle] code root" in out
        assert "d1_mod.py" in out
        assert "d1_mod:TinyAdapter" in out
        assert "scikit-learn>=1.5" in out
        assert isinstance(bdir, str)              # return type is unchanged (spec 45 D1)
    finally:
        _purge("d1_mod")


def test_save_verbose_false_prints_nothing(tmp_path, importable, capsys):
    importable("d1silent_mod.py")
    import d1silent_mod

    try:
        bdir = bundle.save(d1silent_mod.TinyAdapter(), {}, str(tmp_path / "b"), verbose=False)
        out = capsys.readouterr().out

        assert out == ""
        assert isinstance(bdir, str)
    finally:
        _purge("d1silent_mod")


def test_save_report_names_no_code_when_adapter_is_installed(tmp_path, capsys):
    joblib = pytest.importorskip("joblib")
    bdir = bundle.save(joblib.Parallel, {}, str(tmp_path / "b"), code=False)
    out = capsys.readouterr().out

    assert "no code embedded" in out
    assert isinstance(bdir, str)


# --- AC2: adapter not at the top of code/ -> refuse, before copying anything ---------------


def test_two_trees_in_code_kwarg_raises_naming_the_pulled_up_file(tmp_path, importable):
    root = importable("my_adapter.py", root="demo_model")
    other_root = importable("helper.py", src="V = 1\n", root="other_tree")
    import my_adapter

    try:
        with pytest.raises(ValueError, match=r"would not be importable at the top of code/"):
            bundle.save(
                my_adapter.TinyAdapter(), {}, str(tmp_path / "b"),
                code=[os.path.join(root, "my_adapter.py"), os.path.join(other_root, "helper.py")],
            )
        # D2: refuses before any CODE is copied (the bundle dir itself, and any plain
        # artifacts, are created earlier in save() -- only code/ must stay absent).
        assert not os.path.exists(os.path.join(str(tmp_path / "b"), "code"))
    finally:
        _purge("my_adapter")


def test_package_adapter_at_top_of_code_is_accepted(tmp_path, importable):
    """Regression guard: the normal package-adapter case (D2's own example,
    `my_pkg.adapters` -> `my_pkg/adapters.py`) must NOT be refused."""
    importable("d2pkg/__init__.py", src="")
    importable(
        "d2pkg/adapters.py",
        src='''
        from rise_eo.model.adapter import BaseModelAdapter


        class D2PkgAdapter(BaseModelAdapter):
            required_bands = ["B04"]
            output_dtype = "uint8"
            output_nodata = 255
            output_band_names = ["k"]

            def load(self):
                pass
        ''',
    )
    from d2pkg.adapters import D2PkgAdapter

    try:
        bdir = bundle.save(D2PkgAdapter(), {}, str(tmp_path / "b"), verbose=False)
        assert bundle.read_spec(bdir)["code"]["files"] == [
            "d2pkg/__init__.py", "d2pkg/adapters.py",
        ]
    finally:
        _purge("d2pkg", "d2pkg.adapters")


# --- AC3/AC4: an unembedded sibling import raises, naming it + the fix; a dependency doesn't ---


def test_unembedded_sibling_import_raises_naming_it_and_the_fix(tmp_path, importable):
    importable("d3_mod.py", src='''
        from helper import V
        from rise_eo.model.adapter import BaseModelAdapter


        class D3Adapter(BaseModelAdapter):
            required_bands = ["B04"]
            output_dtype = "uint8"
            output_nodata = 255
            output_band_names = ["k"]

            def load(self):
                pass
    ''')
    importable("helper.py", src="V = 1\n", root="srcroot")  # same tree as d3_mod.py
    import d3_mod

    try:
        with pytest.raises(ValueError, match=r"helper.*not embedded") as exc_info:
            bundle.save(d3_mod.D3Adapter(), {}, str(tmp_path / "b"), code=None)
        assert "Fix: code=" in str(exc_info.value)
        assert not os.path.exists(os.path.join(str(tmp_path / "b"), "code"))
    finally:
        _purge("d3_mod")


def test_dependency_import_does_not_raise(tmp_path, importable):
    """A real installed distribution (packaging, already a fsd dependency) must be left
    alone -- it's declared via `requirements=`, never embedded."""
    importable("d3dep_mod.py", src='''
        from packaging.requirements import Requirement
        from rise_eo.model.adapter import BaseModelAdapter


        class D3DepAdapter(BaseModelAdapter):
            required_bands = ["B04"]
            output_dtype = "uint8"
            output_nodata = 255
            output_band_names = ["k"]

            def load(self):
                pass
    ''')
    import d3dep_mod

    try:
        bdir = bundle.save(d3dep_mod.D3DepAdapter(), {}, str(tmp_path / "b"), verbose=False)
        assert bundle.read_spec(bdir)["code"]["files"] == ["d3dep_mod.py"]
    finally:
        _purge("d3dep_mod")


# --- AC4 (second half): a transitive sibling chain is detected -----------------------------


def test_transitive_sibling_chain_is_detected(tmp_path, importable):
    """a imports b (embedded), b imports c (NOT embedded) -> refuse naming c, even though
    the adapter file itself (a) never imports c directly."""
    root = importable("d3chain_mod.py", src='''
        from b_sibling import BVAL
        from rise_eo.model.adapter import BaseModelAdapter


        class ChainAdapter(BaseModelAdapter):
            required_bands = ["B04"]
            output_dtype = "uint8"
            output_nodata = 255
            output_band_names = ["k"]

            def load(self):
                pass
    ''')
    importable("b_sibling.py", src="from c_sibling import CVAL\nBVAL = CVAL\n")
    importable("c_sibling.py", src="CVAL = 1\n")
    import d3chain_mod

    try:
        with pytest.raises(ValueError, match=r"c_sibling"):
            bundle.save(
                d3chain_mod.ChainAdapter(), {}, str(tmp_path / "b"),
                code=[os.path.join(root, "d3chain_mod.py"), os.path.join(root, "b_sibling.py")],
            )
    finally:
        _purge("d3chain_mod")


# --- AC5/AC6: verify_image -- pass/fail against a fake AML client, runner='local' raises ---


class _NS(types.SimpleNamespace):
    pass


class _FakeMLClient:
    """Mirrors tests/test_infer_aml.py's fixture -- job execution is never actually run; the
    test writes whatever `_status/*.json` the real node would have written."""

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


def test_verify_image_passes_against_a_known_good_bundle(tmp_path, importable, fake_aml_command):
    importable("vok_mod.py")
    import vok_mod

    try:
        bdir = bundle.save(vok_mod.TinyAdapter(), {}, str(tmp_path / "b"), verbose=False)
    finally:
        _purge("vok_mod")

    root = "memory://vimg_ok/root"
    ml_client = _FakeMLClient({"smoke": "Completed"})
    with fs.open(_status_url(root, "runok"), "w") as f:
        json.dump({"status": "ok", "error": None}, f)

    result = verify_image(
        bdir, environment="fsd-infer-sklearn:3", runner="aml",
        runner_kwargs=dict(cluster="c", root=root, identity_client_id="x",
                           ml_client=ml_client, run_id="runok"),
    )

    assert result["pass"] is True
    assert result["status"] == "ok"
    assert result["metrics"]["smoke_status"] == "ok"
    assert result["metrics"]["code_files_staged"].split("/")[0] == \
        result["metrics"]["code_files_staged"].split("/")[1]  # N/N landed
    # WHAT was verified, not just where (spec 51 D5): `deploy(verified=...)` matches on this
    # digest, because `bundle_path` alone cannot distinguish a bundle from its replacement.
    assert result["metrics"]["bundle_digest"] == registry.content_digest(bdir)


def test_verify_image_fails_with_populated_error_when_bundle_has_no_code(
    tmp_path, fake_aml_command,
):
    """`code=False` -> no `code` block -> verify_image must catch this BEFORE submitting
    anything (the driver-side-first ordering, D4 step 1)."""
    joblib = pytest.importorskip("joblib")
    bdir = bundle.save(joblib.Parallel, {}, str(tmp_path / "b"), code=False, verbose=False)

    root = "memory://vimg_nocode/root"
    ml_client = _FakeMLClient({"smoke": "Completed"})

    result = verify_image(
        bdir, environment="fsd-infer-sklearn:3", runner="aml",
        runner_kwargs=dict(cluster="c", root=root, identity_client_id="x", ml_client=ml_client),
    )

    assert result["pass"] is False
    assert result["status"] == "fail"
    assert result["error"] and "no `code` block" in result["error"]
    assert ml_client.submitted == []                    # never got to submission


def test_verify_image_stale_wheel_detected_before_submission(tmp_path, importable, fake_aml_command):
    """The build_context wheel-staleness gate must fire BEFORE any job is submitted --
    that's the whole point (spec 45 D4 step 1: refuse in ~2s instead of a cold start)."""
    import zipfile

    importable("vstale_mod.py")
    import vstale_mod

    try:
        bdir = bundle.save(vstale_mod.TinyAdapter(), {}, str(tmp_path / "b"), verbose=False)
    finally:
        _purge("vstale_mod")

    build_ctx = tmp_path / "ctx"
    build_ctx.mkdir()
    stale_wheel = build_ctx / "fsd-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(stale_wheel, "w") as zf:
        zf.writestr("fsd/model/bundle.py", "# a pre-spec-44 bundle.py, no manifest_code_files\n")

    ml_client = _FakeMLClient({"smoke": "Completed"})
    result = verify_image(
        bdir, environment="fsd-infer-sklearn:3", runner="aml",
        runner_kwargs=dict(cluster="c", root="memory://vimg_stale/root", identity_client_id="x",
                           ml_client=ml_client),
        build_context=str(build_ctx),
    )

    assert result["pass"] is False
    assert "predates spec 44" in result["error"]
    assert ml_client.submitted == []


def test_verify_image_absent_wheel_raises_not_a_verdict(tmp_path, importable, fake_aml_command):
    """Spec 47 D10/D11: a build_context with no fsd-*.whl is caller misuse, not a statement about
    the image -- it must raise ValueError and submit nothing, never come back as pass=False."""
    importable("vabsent_mod.py")
    import vabsent_mod

    try:
        bdir = bundle.save(vabsent_mod.TinyAdapter(), {}, str(tmp_path / "b"), verbose=False)
    finally:
        _purge("vabsent_mod")

    build_ctx = tmp_path / "ctx_empty"
    build_ctx.mkdir()

    ml_client = _FakeMLClient({"smoke": "Completed"})
    with pytest.raises(ValueError, match="contains no fsd-\\*.whl"):
        verify_image(
            bdir, environment="fsd-infer-sklearn:3", runner="aml",
            runner_kwargs=dict(cluster="c", root="memory://vimg_absent/root", identity_client_id="x",
                               ml_client=ml_client),
            build_context=str(build_ctx),
        )

    assert ml_client.submitted == []


def test_verify_image_missing_status_file_is_its_own_diagnosis(tmp_path, importable, fake_aml_command):
    """The job "completed" per AML but never wrote a status file -- the entrypoint never
    ran (image/auth problem), and that must itself become the diagnosis, not a KeyError."""
    importable("vmissing_mod.py")
    import vmissing_mod

    try:
        bdir = bundle.save(vmissing_mod.TinyAdapter(), {}, str(tmp_path / "b"), verbose=False)
    finally:
        _purge("vmissing_mod")

    ml_client = _FakeMLClient({"smoke": "Completed"})
    result = verify_image(
        bdir, environment="fsd-infer-sklearn:3", runner="aml",
        runner_kwargs=dict(cluster="c", root="memory://vimg_missing/root", identity_client_id="x",
                           ml_client=ml_client, run_id="runmissing"),
    )

    assert result["pass"] is False
    assert result["metrics"]["smoke_status"] == "no status file written"


def test_verify_image_runner_local_raises(tmp_path):
    """D5: 'verified locally' is the false positive this helper exists to prevent."""
    with pytest.raises(ValueError, match="not a verification"):
        verify_image("some/bundle", environment="fsd-infer-sklearn:3", runner="local")


def test_verify_image_missing_runner_kwargs_raises(tmp_path):
    with pytest.raises(ValueError, match="runner_kwargs"):
        verify_image(
            "some/bundle", environment="fsd-infer-sklearn:3", runner="aml", runner_kwargs={},
        )
