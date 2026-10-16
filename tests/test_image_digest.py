"""Spec 56 §9 step 2 -- `rise_eo.image.digest`: AC2 (digest stability) and AC3 (real resolution)."""

from __future__ import annotations

from rise_eo.image import ImageDefinition
from rise_eo.image import digest as digest_mod


def _fake_git_ref(repo_url: str, ref: str) -> str:
    assert ref == "main"
    return "a" * 40


def _fake_base_digest(base: str) -> str | None:
    return "sha256:" + "b" * 64


def _make_pkg(tmp_path, name="fsd", extra_line=""):
    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "pyproject.toml").write_text(f"""
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "{name}"
version = "0.0.1"
""")
    (pkg / name).mkdir()
    (pkg / name / "__init__.py").write_text(f"VALUE = 1{extra_line}\n")
    return str(pkg)


# --- AC2: digest stability ---------------------------------------------------------------


def test_digest_is_deterministic_across_calls():
    defn = ImageDefinition(
        name="fsd-aml-env", fsd="git+https://github.com/nikhilsrajan/fsd@main",
        extras=("azure", "mpc"),
    )
    r1 = digest_mod.resolve(defn, resolve_base_digest=_fake_base_digest, resolve_git_ref=_fake_git_ref)
    r2 = digest_mod.resolve(defn, resolve_base_digest=_fake_base_digest, resolve_git_ref=_fake_git_ref)

    assert digest_mod.digest(r1) == digest_mod.digest(r2)


def test_digest_ignores_name():
    a = ImageDefinition(name="a", fsd="git+https://github.com/nikhilsrajan/fsd@main", extras=("azure",))
    b = ImageDefinition(name="b", fsd="git+https://github.com/nikhilsrajan/fsd@main", extras=("azure",))

    ra = digest_mod.resolve(a, resolve_base_digest=_fake_base_digest, resolve_git_ref=_fake_git_ref)
    rb = digest_mod.resolve(b, resolve_base_digest=_fake_base_digest, resolve_git_ref=_fake_git_ref)

    assert digest_mod.digest(ra) == digest_mod.digest(rb)
    assert "name" not in ra


def test_digest_differs_on_any_resolved_field():
    a = ImageDefinition(name="x", fsd="git+https://github.com/nikhilsrajan/fsd@main", extras=("azure",))
    b = ImageDefinition(name="x", fsd="git+https://github.com/nikhilsrajan/fsd@main", extras=("azure", "mpc"))

    ra = digest_mod.resolve(a, resolve_base_digest=_fake_base_digest, resolve_git_ref=_fake_git_ref)
    rb = digest_mod.resolve(b, resolve_base_digest=_fake_base_digest, resolve_git_ref=_fake_git_ref)

    assert digest_mod.digest(ra) != digest_mod.digest(rb)


# --- AC3: resolution is real --------------------------------------------------------------


def test_resolve_git_ref_resolves_to_a_40_char_sha():
    defn = ImageDefinition(name="x", fsd="git+https://github.com/nikhilsrajan/fsd@main")
    resolved = digest_mod.resolve(defn, resolve_base_digest=_fake_base_digest, resolve_git_ref=_fake_git_ref)

    assert resolved["fsd"] == f"git+https://github.com/nikhilsrajan/fsd@{'a' * 40}"


def test_resolve_git_ref_already_a_sha_is_left_unchanged():
    sha = "b" * 40
    defn = ImageDefinition(name="x", fsd=f"git+https://github.com/nikhilsrajan/fsd@{sha}")

    def _explode(*a, **k):
        raise AssertionError("should not call ls-remote for an already-pinned sha")

    resolved = digest_mod.resolve(defn, resolve_base_digest=_fake_base_digest, resolve_git_ref=_explode)
    assert resolved["fsd"] == defn.fsd


def test_resolve_path_fsd_equals_wheel_digest_of_the_built_wheel(tmp_path):
    src = _make_pkg(tmp_path)
    defn = ImageDefinition(name="x", fsd=f"path:{src}")

    resolved = digest_mod.resolve(defn, resolve_base_digest=_fake_base_digest, resolve_git_ref=_fake_git_ref)

    assert resolved["fsd"].startswith("wheel:")

    import tempfile
    with tempfile.TemporaryDirectory() as dest:
        from rise_eo.image.definition import _build_wheel
        wheel = _build_wheel(src, dest)
        want = digest_mod.wheel_digest(wheel)
    assert resolved["fsd"] == f"wheel:{want}"


def test_resolve_unresolvable_base_sets_base_resolved_false_and_still_digests():
    defn = ImageDefinition(name="x", fsd="git+https://github.com/nikhilsrajan/fsd@main")

    resolved = digest_mod.resolve(
        defn, resolve_base_digest=lambda base: None, resolve_git_ref=_fake_git_ref,
    )

    assert resolved["base_resolved"] is False
    assert resolved["base"] == defn.base
    d = digest_mod.digest(resolved)
    assert d.startswith("sha256:")


def test_resolve_already_pinned_base_is_recorded_resolved():
    defn = ImageDefinition(
        name="x", fsd="git+https://github.com/nikhilsrajan/fsd@main",
        base="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04@sha256:" + "c" * 64,
    )

    def _explode(*a, **k):
        raise AssertionError("should not query the registry for an already-pinned base")

    resolved = digest_mod.resolve(defn, resolve_base_digest=_explode, resolve_git_ref=_fake_git_ref)
    assert resolved["base_resolved"] is True
    assert resolved["base"] == defn.base


def test_resolve_build_context_digests_file_contents_not_paths(tmp_path):
    ctx1 = tmp_path / "ctx1"
    ctx1.mkdir()
    (ctx1 / "Dockerfile").write_text("FROM scratch\n")

    ctx2 = tmp_path / "ctx2"
    ctx2.mkdir()
    (ctx2 / "Dockerfile").write_text("FROM scratch\n")

    d1 = ImageDefinition(name="a", build_context=str(ctx1))
    d2 = ImageDefinition(name="b", build_context=str(ctx2))

    r1 = digest_mod.resolve(d1)
    r2 = digest_mod.resolve(d2)
    assert digest_mod.digest(r1) == digest_mod.digest(r2)

    (ctx2 / "Dockerfile").write_text("FROM scratch\nRUN echo hi\n")
    r2b = digest_mod.resolve(d2)
    assert digest_mod.digest(r1) != digest_mod.digest(r2b)


# --- Opus review, 2026-08-27 ------------------------------------------------------


def test_an_abbreviated_sha_is_kept_verbatim_not_sent_to_ls_remote():
    """Spec 56 D1's own example pins `@9a00f2b`. `git ls-remote` matches ref NAMES, not object
    ids -- it exits 0 with empty output for a sha -- so asking it would raise instead of
    resolving. An abbreviated sha is already immutable, so it is kept as declared."""
    defn = ImageDefinition(name="x", fsd="git+https://github.com/nikhilsrajan/fsd@9a00f2b")

    def _explode(*a, **k):
        raise AssertionError("ls-remote cannot expand an abbreviated sha; do not call it")

    resolved = digest_mod.resolve(
        defn, resolve_base_digest=_fake_base_digest, resolve_git_ref=_explode,
    )
    assert resolved["fsd"] == defn.fsd


def test_resolve_with_wheel_dir_leaves_the_wheel_it_hashed(tmp_path):
    """The digest and the image must come from ONE wheel: `ensure_environment` hands `resolve`
    the same directory it later builds the context in, and `write_context` reuses what is
    there rather than building a second, differently-timestamped wheel."""
    import glob

    src = _make_pkg(tmp_path)
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    defn = ImageDefinition(name="x", fsd=f"path:{src}")

    resolved = digest_mod.resolve(
        defn, resolve_base_digest=_fake_base_digest, resolve_git_ref=_fake_git_ref,
        wheel_dir=str(ctx),
    )
    built = glob.glob(str(ctx / "fsd-*.whl"))
    assert len(built) == 1
    assert resolved["fsd"] == f"wheel:{digest_mod.wheel_digest(built[0])}"

    defn.write_context(str(ctx))
    assert glob.glob(str(ctx / "fsd-*.whl")) == built  # not rebuilt
