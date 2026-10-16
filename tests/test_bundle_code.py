"""Spec 44 phase 1 — bundle-carried adapter source.

One test per acceptance criterion in `specs/44-bundle-carried-adapter-code.md` §4. The load-bearing
one is `test_load_works_when_the_module_is_importable_from_nowhere_else`: it is the proof that the
per-adapter Docker image (spec 38 D4) is no longer needed.

Everything here is synthetic and offline. Adapter modules are written into `tmp_path` and imported
by manipulating `sys.path`, then removed again — so no test leaves a module cached in `sys.modules`
for the next one (the very failure D2's guard exists to catch).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

from rise_eo.model import bundle

# A minimal adapter module, written to disk so it has a real, importable source file.
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
    """Write a module into a directory, put it on `sys.path`, and clean `sys.modules` after."""
    created: list[str] = []

    def _make(rel: str, src: str = ADAPTER_SRC, root: str | None = None) -> str:
        root_dir = tmp_path / (root or "srcroot")
        target = root_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(src))
        monkeypatch.syspath_prepend(str(root_dir))
        return str(root_dir)

    yield _make

    for name in created:
        sys.modules.pop(name, None)


def _purge(*names: str) -> None:
    for n in names:
        sys.modules.pop(n, None)


# --- criterion 1: a plain local module is embedded, and `save`'s signature is unchanged ------


def test_local_module_is_embedded(tmp_path, importable):
    importable("tiny_mod.py")
    import tiny_mod

    try:
        bdir = bundle.save(tiny_mod.TinyAdapter(), {}, str(tmp_path / "b"))
        manifest = bundle.read_spec(bdir)

        assert manifest["fsd_bundle_version"] == 2
        assert manifest["adapter"] == "tiny_mod:TinyAdapter"
        assert manifest["code_origin"] == "bundled"
        assert manifest["code"] == {"root": "code", "files": ["tiny_mod.py"]}
        assert os.path.exists(os.path.join(bdir, "code", "tiny_mod.py"))
    finally:
        _purge("tiny_mod")


def test_save_call_signature_is_unchanged(tmp_path, importable):
    """The user's bundling cell must not have to change (spec 44 §2) — positional args only."""
    importable("cell17_mod.py")
    import cell17_mod

    try:
        art = tmp_path / "w.txt"
        art.write_text("x")
        bdir = bundle.save(cell17_mod.TinyAdapter(), {"model": str(art)}, str(tmp_path / "b"))
        assert bundle.read_spec(bdir)["adapter"].endswith(":TinyAdapter")
        assert bundle.read_spec(bdir)["artifacts"] == {"model": "w.txt"}
    finally:
        _purge("cell17_mod")


# --- criterion 2: a package keeps its layout, and intra-package imports still resolve --------


def test_package_adapter_preserves_layout_and_intra_package_imports(tmp_path, importable):
    """The MLflow `code_paths` defect fsd fixes: flattening would break `from .helpers import`."""
    importable("mypkg/__init__.py", src="")
    importable("mypkg/helpers.py", src="BANDS = ['B04', 'B08']\n")
    importable(
        "mypkg/adapters.py",
        src='''
        from rise_eo.model.adapter import BaseModelAdapter
        from mypkg.helpers import BANDS


        class PkgAdapter(BaseModelAdapter):
            required_bands = BANDS
            output_dtype = "uint8"
            output_nodata = 255
            output_band_names = ["k"]

            def load(self):
                self.loaded = True
        ''',
    )
    from mypkg.adapters import PkgAdapter

    try:
        bdir = bundle.save(PkgAdapter(), {}, str(tmp_path / "b"))
        files = bundle.read_spec(bdir)["code"]["files"]

        # Layout preserved: nested paths, NOT flattened to basenames.
        assert files == ["mypkg/__init__.py", "mypkg/adapters.py", "mypkg/helpers.py"]
        assert os.path.exists(os.path.join(bdir, "code", "mypkg", "helpers.py"))
    finally:
        _purge("mypkg", "mypkg.adapters", "mypkg.helpers")


# --- criterion 3: an installed adapter is left alone (spec 38 D4's world still works) --------


def test_installed_adapter_is_not_embedded(tmp_path):
    """`joblib` stands in for any pip-installed adapter package: origin `installed`, no code."""
    joblib = pytest.importorskip("joblib")

    origin, detail = bundle.classify_adapter_source(joblib.Parallel)
    assert origin == "installed"
    assert detail is None


def test_fsd_own_classes_are_never_embedded():
    """An adapter subclassing fsd's base must not drag src/rise_eo/ into the bundle."""
    from rise_eo.model.adapter import BaseModelAdapter

    assert bundle.classify_adapter_source(BaseModelAdapter)[0] == "installed"


# --- criterion 4: `__main__` / notebook classes are refused, with the fix in the message -----


def test_main_module_adapter_is_refused():
    class Inline:  # defined in the test module, but we force the __main__ case
        pass

    Inline.__module__ = "__main__"

    with pytest.raises(ValueError, match="notebook"):
        bundle.adapter_code_files(Inline)


def test_nonexistent_source_is_refused(tmp_path, importable):
    """A notebook cell reports source at /tmp/ipykernel_*/1234.py, which does not exist."""
    importable("ghost_mod.py")
    import ghost_mod

    try:
        os.remove(os.path.join(os.path.dirname(ghost_mod.__file__), "ghost_mod.py"))
        origin, detail = bundle.classify_adapter_source(ghost_mod.TinyAdapter)
        assert origin == "unresolvable"
        assert "does not exist" in detail
    finally:
        _purge("ghost_mod")


# --- criterion 5: THE test — load with the module importable from nowhere else ---------------


def test_load_works_when_the_module_is_importable_from_nowhere_else(tmp_path, importable):
    """The D4-image-replacement proof.

    Build a bundle, then load it in a **fresh subprocess** whose `sys.path` does not contain the
    adapter's original directory and whose `sys.modules` is empty. If this passes, the adapter
    reached the interpreter purely by riding inside the bundle — which is exactly what a cluster
    node does, and exactly what used to require baking the .py into a Docker image.
    """
    importable("lonely_mod.py")
    import lonely_mod

    try:
        bdir = bundle.save(lonely_mod.TinyAdapter(), {}, str(tmp_path / "b"))
    finally:
        _purge("lonely_mod")

    script = textwrap.dedent(f"""
        import sys
        from rise_eo.model import bundle
        assert "lonely_mod" not in sys.modules
        try:
            import lonely_mod
        except ImportError:
            pass
        else:
            raise SystemExit("FAIL: lonely_mod was importable from the environment")
        adapter = bundle.load({str(bdir)!r})
        assert type(adapter).__name__ == "TinyAdapter"
        assert adapter.loaded is True
        assert adapter.required_bands == ["B04", "B08"]
        print("OK")
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in sys.path if p and "srcroot" not in p and os.path.isdir(p)
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, cwd=str(tmp_path), check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    assert "OK" in proc.stdout


# --- criterion 6 (+ amendment A1): the collision guard, both halves --------------------------


def test_collision_guard_raises_on_different_source(tmp_path, importable):
    """mlflow#12377's silent wrong model: same module name, different implementation."""
    importable("clash_mod.py")
    import clash_mod

    try:
        bdir = bundle.save(clash_mod.TinyAdapter(), {}, str(tmp_path / "b"))
        # Tamper with the bundle's copy so it differs from the imported module.
        copy = os.path.join(bdir, "code", "clash_mod.py")
        with open(copy, "a") as f:
            f.write("\n# a different implementation\n")

        with pytest.raises(ValueError, match="DIFFERENT source"):
            bundle.load(bdir)
    finally:
        _purge("clash_mod")


def test_collision_guard_allows_byte_identical_source(tmp_path, importable):
    """Amendment A1: save-then-load in ONE process is normal (api._ensure_bundle does it).

    The imported module and the bundle's copy always sit at different paths there, so a
    path-comparison guard would be a false-positive machine. Content comparison must allow it.
    """
    importable("same_mod.py")
    import same_mod

    try:
        bdir = bundle.save(same_mod.TinyAdapter(), {}, str(tmp_path / "b"))
        assert os.path.exists(os.path.join(bdir, "code", "same_mod.py"))
        adapter = bundle.load(bdir)          # must NOT raise
        assert type(adapter).__name__ == "TinyAdapter"
    finally:
        _purge("same_mod")


def test_collision_guard_checks_ancestor_packages_not_just_the_leaf(tmp_path, importable):
    """A stale PARENT package is the subtler half of the same failure.

    `sys.modules["mypkg"]` keeps its own `__path__`, so `import mypkg.adapters` resolves through
    the already-imported parent and never consults the bundle's copy — the bundle's code is
    silently ignored even though the leaf module was never imported.
    """
    importable("ancpkg/__init__.py", src="")
    importable(
        "ancpkg/adapters.py",
        src="""
        from rise_eo.model.adapter import BaseModelAdapter


        class AncAdapter(BaseModelAdapter):
            required_bands = ["B04"]
            output_dtype = "uint8"
            output_nodata = 255
            output_band_names = ["k"]

            def load(self):
                pass
        """,
    )
    from ancpkg.adapters import AncAdapter

    try:
        bdir = bundle.save(AncAdapter(), {}, str(tmp_path / "b"))
        # Only the PARENT stays imported, and it now differs from the bundle's copy.
        _purge("ancpkg.adapters")
        with open(os.path.join(bdir, "code", "ancpkg", "__init__.py"), "w") as f:
            f.write("# a different package init\n")

        with pytest.raises(ValueError, match=r"'ancpkg' is already imported"):
            bundle.load(bdir)
    finally:
        _purge("ancpkg", "ancpkg.adapters")


# --- criterion 7: idempotence -----------------------------------------------------------------


def test_repeated_loads_add_no_duplicate_syspath_entries(tmp_path, importable):
    importable("idem_mod.py")
    import idem_mod

    try:
        bdir = bundle.save(idem_mod.TinyAdapter(), {}, str(tmp_path / "b"))
        code_root = os.path.join(bdir, "code")
        before = list(sys.path)
        bundle.load(bdir)
        bundle.load(bdir)
        bundle.load(bdir)
        assert sys.path.count(code_root) == 1
    finally:
        _purge("idem_mod")
        sys.path[:] = before


# --- criterion 8: version-1 bundles still load ------------------------------------------------


def test_version_1_bundle_still_loads(tmp_path, importable):
    """A v1 bundle has no `code` block — indistinguishable from a v2 installed-package bundle.
    Both mean "resolve the ref from the environment", i.e. exactly today's behavior."""
    importable("v1_mod.py")
    import v1_mod

    try:
        bdir = bundle.save(v1_mod.TinyAdapter(), {}, str(tmp_path / "b"))
        mfp = os.path.join(bdir, "bundle.json")
        with open(mfp) as f:
            manifest = json.load(f)
        manifest["fsd_bundle_version"] = 1          # rewrite as a legacy bundle
        manifest.pop("code")
        manifest.pop("code_origin")
        with open(mfp, "w") as f:
            json.dump(manifest, f)

        adapter = bundle.load(bdir)                  # resolves from sys.path, as it always did
        assert type(adapter).__name__ == "TinyAdapter"
    finally:
        _purge("v1_mod")


def test_unknown_bundle_version_is_refused(tmp_path, importable):
    importable("future_mod.py")
    import future_mod

    try:
        bdir = bundle.save(future_mod.TinyAdapter(), {}, str(tmp_path / "b"))
        mfp = os.path.join(bdir, "bundle.json")
        with open(mfp) as f:
            manifest = json.load(f)
        manifest["fsd_bundle_version"] = 99
        with open(mfp, "w") as f:
            json.dump(manifest, f)

        with pytest.raises(ValueError, match="unsupported fsd_bundle_version"):
            bundle.load(bdir)
    finally:
        _purge("future_mod")


# --- criterion 9: the size cap, and the escapes -----------------------------------------------


def test_oversized_code_is_refused_and_names_the_override(tmp_path, importable, monkeypatch):
    importable("bigpkg/__init__.py", src="")
    importable(
        "bigpkg/adapters.py",
        src='''
        from rise_eo.model.adapter import BaseModelAdapter


        class BigAdapter(BaseModelAdapter):
            required_bands = ["B04"]
            output_dtype = "uint8"
            output_nodata = 255
            output_band_names = ["k"]

            def load(self):
                pass
        ''',
    )
    from bigpkg.adapters import BigAdapter

    try:
        monkeypatch.setattr(bundle, "MAX_CODE_FILES", 1)
        with pytest.raises(ValueError, match=r"code=\["):
            bundle.adapter_code_files(BigAdapter())
    finally:
        _purge("bigpkg", "bigpkg.adapters")


def test_code_false_keeps_the_installed_package_behavior(tmp_path, importable):
    importable("optout_mod.py")
    import optout_mod

    try:
        bdir = bundle.save(optout_mod.TinyAdapter(), {}, str(tmp_path / "b"), code=False)
        manifest = bundle.read_spec(bdir)
        assert manifest["code_origin"] == "installed"
        assert "code" not in manifest
        assert not os.path.exists(os.path.join(bdir, "code"))
    finally:
        _purge("optout_mod")


def test_explicit_code_list_is_honoured(tmp_path, importable):
    root = importable("explicit_mod.py")
    extra = os.path.join(root, "sidecar.py")
    with open(extra, "w") as f:
        f.write("VALUE = 1\n")
    import explicit_mod

    try:
        bdir = bundle.save(
            explicit_mod.TinyAdapter(), {}, str(tmp_path / "b"),
            code=[os.path.join(root, "explicit_mod.py"), extra],
        )
        assert bundle.read_spec(bdir)["code"]["files"] == ["explicit_mod.py", "sidecar.py"]
    finally:
        _purge("explicit_mod")


# --- criterion 10: both manifest-driven transports carry the code, and nothing else ----------


def test_manifest_code_files_enumerates_without_listing_directories(tmp_path, importable):
    importable("transport_mod.py")
    import transport_mod

    try:
        art = tmp_path / "w.txt"
        art.write_text("x")
        bdir = bundle.save(transport_mod.TinyAdapter(), {"model": str(art)}, str(tmp_path / "b"))
        manifest = bundle.read_spec(bdir)

        assert bundle.manifest_code_files(manifest) == ["code/transport_mod.py"]
        # A v1/installed manifest enumerates nothing — the transports stay unchanged for it.
        assert bundle.manifest_code_files({"artifacts": {}}) == []
    finally:
        _purge("transport_mod")


def test_stage_and_fetch_transfer_every_code_file(tmp_path, importable):
    """`_stage_bundle` (driver -> blob) and `fetch_bundle_to_scratch` (blob -> node scratch) both
    read the same manifest. Exercised over local paths, which is what `rise_eo.storage` gives us
    without credentials — the seam is identical for abfss://."""
    from rise_eo.workflows import runners
    from rise_eo.workflows.infer_shard import fetch_bundle_to_scratch

    importable("moved_mod.py")
    import moved_mod

    try:
        art = tmp_path / "w.txt"
        art.write_text("x")
        bdir = bundle.save(moved_mod.TinyAdapter(), {"model": str(art)}, str(tmp_path / "b"))

        staged = runners._stage_bundle(bdir, str(tmp_path / "staged"))
        assert os.path.exists(os.path.join(staged, "code", "moved_mod.py"))
        assert os.path.exists(os.path.join(staged, "w.txt"))

        scratch = fetch_bundle_to_scratch(staged, str(tmp_path / "scratch"))
        assert os.path.exists(os.path.join(scratch, "code", "moved_mod.py"))
        assert sorted(os.listdir(scratch)) == ["bundle.json", "code", "w.txt"]
    finally:
        _purge("moved_mod")


# --- criterion 11: declared requirements are checked, not installed --------------------------


def test_check_requirements_reports_missing_and_mismatched():
    assert bundle.check_requirements([]) == []
    assert bundle.check_requirements(None) == []
    assert bundle.check_requirements(["packaging"]) == []          # certainly installed

    (missing,) = bundle.check_requirements(["definitely-not-a-real-package-xyz"])
    assert "not installed" in missing

    (bad,) = bundle.check_requirements(["packaging<0.1"])
    assert "installed packaging==" in bad


def test_requirements_are_recorded_in_the_manifest(tmp_path, importable):
    importable("req_mod.py")
    import req_mod

    try:
        bdir = bundle.save(
            req_mod.TinyAdapter(), {}, str(tmp_path / "b"), requirements=["scikit-learn>=1.5"],
        )
        assert bundle.read_spec(bdir)["requirements"] == ["scikit-learn>=1.5"]
    finally:
        _purge("req_mod")


def test_smoke_job_names_the_dependency_instead_of_a_traceback(tmp_path, importable):
    """D5: a missing dep must fail once, on the smoke node, as a NAMED dependency."""
    from rise_eo.workflows import adapter_smoke

    importable("smoke_mod.py")
    import smoke_mod

    try:
        bdir = bundle.save(
            smoke_mod.TinyAdapter(), {}, str(tmp_path / "b"),
            requirements=["definitely-not-a-real-package-xyz"],
        )
        status = adapter_smoke.run_smoke(bdir, str(tmp_path / "status.json"))

        assert status["status"] == "failed"
        assert "definitely-not-a-real-package-xyz" in status["error"]
        assert "rebuild" in status["error"]
    finally:
        _purge("smoke_mod")


def test_smoke_job_passes_for_a_bundled_adapter(tmp_path, importable):
    from rise_eo.workflows import adapter_smoke

    importable("smokeok_mod.py")
    import smokeok_mod

    try:
        bdir = bundle.save(smokeok_mod.TinyAdapter(), {}, str(tmp_path / "b"))
        status = adapter_smoke.run_smoke(bdir, str(tmp_path / "status.json"))
        assert status["status"] == "ok", status
    finally:
        _purge("smokeok_mod")


# --- D4: the drift check's meaning is per-origin ----------------------------------------------


def test_drift_message_is_per_origin(tmp_path, importable):
    """Bundled -> "the bundle has been edited". Installed -> the original "code/bundle drift"
    wording, where version skew between the image's pip package and the bundle is real."""
    importable("drift_mod.py")
    import drift_mod

    try:
        bdir = bundle.save(drift_mod.TinyAdapter(), {}, str(tmp_path / "b"))
        mfp = os.path.join(bdir, "bundle.json")

        with open(mfp) as f:
            manifest = json.load(f)
        manifest["required_bands"] = ["B02"]          # disagree with the class
        with open(mfp, "w") as f:
            json.dump(manifest, f)
        with pytest.raises(ValueError, match="has been edited"):
            bundle.load(bdir)

        # Same tamper, but presented as an installed-package bundle.
        manifest.pop("code")
        manifest["code_origin"] = "installed"
        with open(mfp, "w") as f:
            json.dump(manifest, f)
        with pytest.raises(ValueError, match="code/bundle drift"):
            bundle.load(bdir)
    finally:
        _purge("drift_mod")
