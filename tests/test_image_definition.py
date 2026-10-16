"""Spec 56 §9 step 1 -- `rise_eo.image.definition.ImageDefinition`: declare, derive, render."""

from __future__ import annotations

import os

from rise_eo.image import ImageDefinition


def test_derive_overrides_only_the_given_fields():
    base = ImageDefinition(
        name="fsd-aml-env", fsd="git+https://github.com/nikhilsrajan/fsd@9a00f2b",
        extras=("azure", "mpc"),
    )
    infer = base.derive(name="fsd-infer-sklearn", extra_pip=("scikit-learn", "joblib"))

    assert infer.name == "fsd-infer-sklearn"
    assert infer.extra_pip == ("scikit-learn", "joblib")
    assert infer.fsd == base.fsd
    assert infer.extras == base.extras
    assert infer.base == base.base


def test_definition_is_hashable_and_comparable():
    a = ImageDefinition(name="x", fsd="git+https://example.com/fsd@main")
    b = ImageDefinition(name="x", fsd="git+https://example.com/fsd@main")
    c = ImageDefinition(name="y", fsd="git+https://example.com/fsd@main")

    assert a == b
    assert hash(a) == hash(b)
    assert a != c


def test_render_dockerfile_git_ref_has_no_wheel_copy():
    defn = ImageDefinition(
        name="fsd-aml-env", fsd="git+https://github.com/nikhilsrajan/fsd@9a00f2b",
        extras=("azure", "mpc"),
    )
    text = defn.render_dockerfile()

    assert "FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest" in text
    assert "COPY fsd-*.whl" not in text
    assert 'fsd[azure,mpc] @ git+https://github.com/nikhilsrajan/fsd@9a00f2b' in text


def test_render_dockerfile_path_fsd_copies_a_wheel():
    defn = ImageDefinition(name="dev", fsd="path:/some/checkout", extras=("azure",))
    text = defn.render_dockerfile()

    assert "COPY fsd-*.whl /tmp/" in text
    assert "$(ls /tmp/fsd-*.whl)[azure]" in text


def test_render_dockerfile_includes_extra_pip():
    defn = ImageDefinition(
        name="fsd-infer-sklearn", fsd="git+https://example.com/fsd@main",
        extras=("azure", "mpc"), extra_pip=("scikit-learn", "joblib"),
    )
    text = defn.render_dockerfile()

    assert "scikit-learn joblib" in text


def test_render_dockerfile_raises_when_build_context_is_set():
    defn = ImageDefinition(name="mine", build_context="/some/dir")
    try:
        defn.render_dockerfile()
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_write_context_writes_a_dockerfile(tmp_path):
    defn = ImageDefinition(name="x", fsd="git+https://example.com/fsd@main")
    out = defn.write_context(str(tmp_path / "ctx"))

    assert out == str(tmp_path / "ctx")
    assert os.path.exists(os.path.join(out, "Dockerfile"))
