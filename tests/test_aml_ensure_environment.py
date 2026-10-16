"""Spec 56 §9 step 5 -- `rise_eo.aml.ensure_environment`: AC4 (check-then-build), builder
stubbed throughout so no test reaches Azure (AC8)."""

from __future__ import annotations

import os

import fsspec
import pytest

from rise_eo.aml import ensure_environment
from rise_eo.image import ImageDefinition


@pytest.fixture()
def _clean_fsspec_conf():
    """`configure_storage` mutates `fsspec.config.conf` and `os.environ` PROCESS-GLOBALLY, so
    any test that lets `storage="azure"` through has to put them back or it silently fails
    `tests/test_azure_seam.py::test_configure_storage_*_is_noop` later in the same session
    (Opus review, 2026-08-27 -- caught exactly this way). Mirrors that module's own fixture."""
    before = {k: dict(v) for k, v in fsspec.config.conf.items()}
    before_env = os.environ.get("FSSPEC_ABFSS_ANON")
    yield
    fsspec.config.conf.clear()
    fsspec.config.conf.update(before)
    if before_env is None:
        os.environ.pop("FSSPEC_ABFSS_ANON", None)
    else:
        os.environ["FSSPEC_ABFSS_ANON"] = before_env

DEFN = ImageDefinition(
    name="fsd-aml-env", fsd="git+https://github.com/nikhilsrajan/fsd@main",
    extras=("azure", "mpc"),
)


def _fake_git_ref(repo_url, ref):
    return "a" * 40


def _fake_base_digest(base):
    return None  # unresolved base, deterministic and network-free


def _kwargs(**overrides):
    kwargs = dict(
        resolve_base_digest=_fake_base_digest, resolve_git_ref=_fake_git_ref,
        resource_group="rg", workspace="ws",
    )
    kwargs.update(overrides)
    return kwargs


def test_no_registry_entry_builds_and_publishes(tmp_path):
    registry_root = str(tmp_path / "registry")
    calls = {"create": 0}

    def _create(name, context_dir, *, resource_group, workspace):
        calls["create"] += 1
        return "1"

    result = ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=_create,
        _build_link=lambda *a, **k: "https://ml.azure.com/x",
        **_kwargs(),
    )

    assert calls["create"] == 1
    assert result.reused is False
    assert result.version == "1"
    assert result.ref == "fsd-aml-env:1"


def test_registry_hit_with_live_asset_builds_nothing(tmp_path):
    registry_root = str(tmp_path / "registry")

    def _create(*a, **k):
        raise AssertionError("should not build: the registry entry's asset is live")

    # first call registers v1
    ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=lambda name, ctx, **k: "5",
        _build_link=lambda *a, **k: "https://ml.azure.com/x",
        **_kwargs(),
    )

    result = ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=_create,
        _environment_exists=lambda *a, **k: True,
        _build_link=lambda *a, **k: "https://ml.azure.com/x",
        **_kwargs(),
    )

    assert result.reused is True
    assert result.version == "5"
    assert result.ref == "fsd-aml-env:5"


def test_registry_hit_with_deleted_asset_builds(tmp_path):
    registry_root = str(tmp_path / "registry")
    ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=lambda name, ctx, **k: "5",
        _build_link=lambda *a, **k: "https://ml.azure.com/x",
        **_kwargs(),
    )

    calls = {"create": 0}

    def _create(name, ctx, **k):
        calls["create"] += 1
        return "6"

    result = ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=_create,
        _environment_exists=lambda *a, **k: False,
        _build_link=lambda *a, **k: "https://ml.azure.com/x",
        **_kwargs(),
    )

    assert calls["create"] == 1
    assert result.reused is False
    assert result.version == "6"


def test_force_rebuilds_even_on_a_hit(tmp_path):
    registry_root = str(tmp_path / "registry")
    ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=lambda name, ctx, **k: "5",
        _build_link=lambda *a, **k: "https://ml.azure.com/x",
        **_kwargs(),
    )

    calls = {"create": 0}

    def _create(name, ctx, **k):
        calls["create"] += 1
        return "7"

    result = ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=_create,
        _environment_exists=lambda *a, **k: True,
        _build_link=lambda *a, **k: "https://ml.azure.com/x",
        force=True,
        **_kwargs(),
    )

    assert calls["create"] == 1
    assert result.reused is False
    assert result.version == "7"


# --- Opus review, 2026-08-27: the registry must follow the AML asset ---------------


# `:1` rather than `@current`: `ensure_environment` sets no alias by default (spec 102 D12), and
# every caller below publishes only the unchanged `DEFN`, so v1 is the only version there is.
def _aml_of(registry_root, ref="fsd-aml-env:1"):
    from rise_eo.image import registry as ireg
    return ireg.resolve(ref, registry_root).aml


def test_rebuilding_a_deleted_asset_repoints_the_registry(tmp_path):
    """`publish` is idempotent by digest, so a rebuild of an UNCHANGED definition allocates no
    new version -- without `_aml.json` the registry keeps naming the asset that was deleted,
    `ensure_environment` finds it missing again, and rebuilds a 10-20 minute image on every
    single call. Regression for exactly that loop."""
    registry_root = str(tmp_path / "registry")
    ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=lambda name, ctx, **k: "5",
        _build_link=lambda *a, **k: "https://ml.azure.com/x",
        **_kwargs(),
    )
    assert _aml_of(registry_root)["version"] == "5"

    # v5's asset is gone -> rebuild produces v6, and the registry must now say v6
    ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=lambda name, ctx, **k: "6",
        _environment_exists=lambda *a, **k: False,
        _build_link=lambda *a, **k: "https://ml.azure.com/x",
        **_kwargs(),
    )
    assert _aml_of(registry_root)["version"] == "6"

    # and the next call reuses v6 instead of rebuilding a third time
    def _explode(*a, **k):
        raise AssertionError("rebuild loop: the registry still points at the deleted asset")

    third = ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=_explode,
        _environment_exists=lambda name, ver, **k: ver != "5",
        _build_link=lambda *a, **k: "https://ml.azure.com/x",
        **_kwargs(),
    )
    assert third.reused is True
    assert third.version == "6"


def test_force_rebuild_also_repoints_the_registry(tmp_path):
    """Same mechanism via the other route into a rebuild-without-a-new-version (D4's `force`)."""
    registry_root = str(tmp_path / "registry")
    ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=lambda name, ctx, **k: "5",
        _build_link=lambda *a, **k: "https://ml.azure.com/x", **_kwargs(),
    )
    ensure_environment(
        DEFN, registry=registry_root, force=True,
        _create_environment=lambda name, ctx, **k: "7",
        _environment_exists=lambda *a, **k: True,
        _build_link=lambda *a, **k: "https://ml.azure.com/x", **_kwargs(),
    )
    assert _aml_of(registry_root)["version"] == "7"


def test_the_aml_sidecar_does_not_disturb_the_content_digest(tmp_path):
    """`_aml.json` sits beside `image.json`, never inside it -- so a repoint must leave the
    version resolvable by the same digest it was published under (AC5's immutability)."""
    from rise_eo.image import registry as ireg

    registry_root = str(tmp_path / "registry")
    first = ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=lambda name, ctx, **k: "5",
        _build_link=lambda *a, **k: "https://ml.azure.com/x", **_kwargs(),
    )
    ensure_environment(
        DEFN, registry=registry_root, force=True,
        _create_environment=lambda name, ctx, **k: "9",
        _build_link=lambda *a, **k: "https://ml.azure.com/x", **_kwargs(),
    )
    assert ireg.find_by_digest("fsd-aml-env", registry_root, first.digest) == 1
    assert ireg.resolve("fsd-aml-env:1", registry_root).digest == first.digest


def test_the_aml_version_and_the_registry_version_are_reported_separately(tmp_path):
    """AML numbers ASSETS, the registry numbers DEFINITIONS, and they diverge immediately:
    `verify_image(image_ref=..., registry=...)` resolves the registry's number, `environment=`
    wants AML's. Handing over the wrong one fails as a missing `v<N>` directory, so both are
    named fields rather than one ambiguous `version`."""
    from rise_eo.image import registry as ireg

    registry_root = str(tmp_path / "registry")
    r = ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=lambda name, ctx, **k: "5",   # AML says 5; the registry says 1
        _build_link=lambda *a, **k: "https://ml.azure.com/x", **_kwargs(),
    )
    assert (r.version, r.ref) == ("5", "fsd-aml-env:5")
    assert (r.registry_version, r.registry_ref) == (1, "fsd-aml-env:1")
    assert ireg.resolve(r.registry_ref, registry_root).digest == r.digest

    reused = ensure_environment(
        DEFN, registry=registry_root,
        _create_environment=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no build")),
        _environment_exists=lambda *a, **k: True,
        _build_link=lambda *a, **k: "https://ml.azure.com/x", **_kwargs(),
    )
    assert reused.reused is True
    assert (reused.version, reused.registry_version) == ("5", 1)


def test_storage_is_configured_before_the_first_registry_read(
    tmp_path, monkeypatch, _clean_fsspec_conf,
):
    """`ensure_environment` reaches the storage seam (`find_by_digest`) and must configure it
    first, like every other public verb (spec 52 D4). `adlfs`'s own default (`anon=None`) tries
    a credential, so the authenticated path already worked; what `storage="azure"` adds is
    `anon=False`, which makes a credential problem RAISE instead of degrading to an anonymous
    read that returns nothing and is indistinguishable from "not registered yet" -- i.e. a
    rebuild of a 10-20 minute image on every call (Opus review, 2026-08-27)."""
    import rise_eo.aml as aml_mod

    order = []
    real_configure = aml_mod._configure_storage
    monkeypatch.setattr(
        aml_mod, "_configure_storage",
        lambda storage: (order.append(("configure", storage)), real_configure(storage))[0],
    )

    def _spy_find(*a, **k):
        order.append(("find", None))
        return None

    ensure_environment(
        DEFN, registry=str(tmp_path / "registry"), storage="azure",
        _find_by_digest=_spy_find,
        _create_environment=lambda name, ctx, **k: "1",
        _build_link=lambda *a, **k: "https://ml.azure.com/x",
        **_kwargs(),
    )
    assert order[0] == ("configure", "azure")
    assert ("find", None) in order[1:]


def test_an_unsupported_storage_backend_raises(tmp_path, _clean_fsspec_conf):
    """`configure_storage` refuses an unknown backend, and that must surface here rather than
    silently falling back to anonymous local access."""
    import pytest

    with pytest.raises(ValueError, match="not supported"):
        ensure_environment(
            DEFN, registry=str(tmp_path / "registry"), storage="s3",
            _create_environment=lambda name, ctx, **k: "1",
            _build_link=lambda *a, **k: "https://ml.azure.com/x",
            **_kwargs(),
        )


# --- spec 102 D12 / AC8: a run moves a pointer only when asked --------------------------


CHANGED = ImageDefinition(name=DEFN.name, fsd=DEFN.fsd, extras=("azure", "mpc", "cdse"))


def _alias_version(registry_root, alias):
    from rise_eo.image import registry as ireg
    return ireg.resolve(f"fsd-aml-env@{alias}", registry_root).version


def _aliases_file(registry_root):
    return os.path.join(registry_root, "fsd-aml-env", "_aliases.json")


def _ensure(registry_root, create_version, defn=DEFN, **extra):
    return ensure_environment(
        defn, registry=registry_root,
        _create_environment=lambda name, ctx, **k: create_version,
        _build_link=lambda *a, **k: "https://ml.azure.com/x",
        **_kwargs(), **extra,
    )


def test_default_alias_leaves_aliases_untouched(tmp_path):
    registry_root = str(tmp_path / "registry")

    # A first build with no `alias=` writes no `_aliases.json` at all.
    _ensure(registry_root, "1")
    assert not os.path.exists(_aliases_file(registry_root))

    # The maintainer points `current` at v1 on purpose.
    _ensure(registry_root, "1", alias="current")
    assert _alias_version(registry_root, "current") == 1
    with open(_aliases_file(registry_root), "rb") as f:
        before = f.read()

    # Build path: a changed definition publishes v2 and must not repoint `current`.
    built = _ensure(registry_root, "2", defn=CHANGED)
    assert built.registry_version == 2 and built.reused is False
    # Reuse path (live asset), then the force-rebuild path (publish returns the existing v2).
    reused = _ensure(registry_root, "9", defn=CHANGED, _environment_exists=lambda *a, **k: True)
    assert reused.reused is True
    forced = _ensure(
        registry_root, "10", defn=CHANGED, force=True, _environment_exists=lambda *a, **k: True,
    )
    assert forced.registry_version == 2 and forced.reused is False

    with open(_aliases_file(registry_root), "rb") as f:
        assert f.read() == before


def test_explicit_alias_moves_the_pointer_on_a_build(tmp_path):
    registry_root = str(tmp_path / "registry")
    _ensure(registry_root, "1", alias="current")
    _ensure(registry_root, "2", defn=CHANGED, alias="dev-someone")

    assert _alias_version(registry_root, "current") == 1
    assert _alias_version(registry_root, "dev-someone") == 2


def test_explicit_alias_moves_the_pointer_on_a_reuse(tmp_path):
    """The maintainer's promotion: a teammate's build is already registered with a live asset, so
    `ensure_environment(..., alias="current")` takes the reuse path -- which must still honour it."""
    registry_root = str(tmp_path / "registry")
    _ensure(registry_root, "1", alias="current")
    _ensure(registry_root, "2", defn=CHANGED)          # a teammate's build, no alias
    assert _alias_version(registry_root, "current") == 1

    promoted = _ensure(
        registry_root, "9", defn=CHANGED, alias="current",
        _environment_exists=lambda *a, **k: True,
    )
    assert promoted.reused is True and promoted.registry_version == 2
    assert _alias_version(registry_root, "current") == 2
