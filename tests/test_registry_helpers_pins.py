"""Characterisation pins for the registry helpers (refactor slice P13, #145).

Written against the code before the C14/C16 helpers were extracted, and never edited
after: they fix the staged-JSON write, the best-effort discard, the content-digest byte
order and the error messages of `fsd.model.registry` and `fsd.registry._core`.
"""

from __future__ import annotations

import os
import re

import pytest

from fsd.image import digest as image_digest
from fsd.model import registry as model_registry
from fsd.registry import _core as core
from fsd.storage import fs

_STAGE_RE = re.compile(r"\.staging-[0-9a-f]{32}\.json$")

# --- C16: the digest byte order is the contract --------------------------------------

_DIGEST_FILES = [
    ("z/b.txt", b"bee"),
    ("été.txt", "über".encode()),
    ("a.txt", b""),
    ("a/c.bin", b"\x00\x01\x02"),
]
_DIGEST_LITERAL = "sha256:94ef4107685b3f1885d95a5841b279807b039d4bba6b30e8e5eeb37551726faa"


def test_digest_of_unsorted_files_equals_literal():
    assert model_registry._digest_of(_DIGEST_FILES) == _DIGEST_LITERAL
    assert model_registry._digest_of(list(reversed(_DIGEST_FILES))) == _DIGEST_LITERAL


def test_digest_directory_equals_literal(tmp_path):
    for rel, data in _DIGEST_FILES:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    assert image_digest._digest_directory(str(tmp_path)) == _DIGEST_LITERAL


# --- C14: the staged JSON write, for each of the three sites --------------------------


def _model_alias(tmp_path):
    reg = str(tmp_path / "reg")
    d = os.path.join(reg, "m")
    os.makedirs(os.path.join(d, "v1"))

    def call(key, value):
        # `key` is an alias; `value` is ignored, the alias points at v1
        model_registry.set_alias("m", key, 1, reg)

    return d, os.path.join(d, "_aliases.json"), call


def _core_alias(tmp_path):
    reg = str(tmp_path / "reg")
    d = os.path.join(reg, "m")
    os.makedirs(os.path.join(d, "v1"))

    def call(key, value):
        core.set_alias("m", key, 1, reg, {})

    return d, os.path.join(d, "_aliases.json"), call


def _model_deploy(tmp_path):
    reg = str(tmp_path / "reg")
    d = os.path.join(reg, "m", "v1")
    os.makedirs(d)
    state = {}

    def call(key, value):
        state[key] = value
        model_registry.write_deploy_record("m", 1, dict(state), reg)

    return d, os.path.join(d, "_deploy.json"), call


_SITES = {
    "model_set_alias": _model_alias,
    "core_set_alias": _core_alias,
    "model_write_deploy_record": _model_deploy,
}


@pytest.fixture(params=sorted(_SITES))
def site(request, tmp_path):
    return request.param, *_SITES[request.param](tmp_path)


def _listing(directory):
    return sorted(os.listdir(directory))


def test_staged_write_text_is_indented_and_sorted(site):
    _kind, directory, target, call = site
    call("zeta", 1)
    call("alpha", 1)
    with open(target) as f:
        text = f.read()
    assert text == '{\n  "alpha": 1,\n  "zeta": 1\n}'
    assert not any(n.startswith(".staging-") for n in _listing(directory))


def test_staged_write_is_one_write_then_one_rename(site, monkeypatch):
    _kind, directory, target, call = site
    call("first", 1)
    writes, renames = [], []
    real_write, real_rename = fs.write_text, fs.rename

    def spy_write(path, text, **kw):
        writes.append(path)
        return real_write(path, text, **kw)

    def spy_rename(src, dst, **kw):
        renames.append((src, dst))
        return real_rename(src, dst, **kw)

    monkeypatch.setattr(fs, "write_text", spy_write)
    monkeypatch.setattr(fs, "rename", spy_rename)
    call("second", 2)
    assert len(writes) == 1
    assert os.path.dirname(writes[0]) == directory
    assert _STAGE_RE.search(os.path.basename(writes[0]))
    assert renames == [(writes[0], target)]
    assert not os.path.exists(writes[0])
    assert not any(n.startswith(".staging-") for n in _listing(directory))


def test_failed_rename_propagates_same_oserror_and_cleans_stage(site, monkeypatch):
    _kind, directory, target, call = site
    call("first", 1)
    with open(target) as f:
        before = f.read()
    boom = OSError("rename failed")

    def bad_rename(src, dst, **kw):
        raise boom

    monkeypatch.setattr(fs, "rename", bad_rename)
    with pytest.raises(OSError) as ei:
        call("second", 2)
    assert ei.value is boom
    assert not any(n.startswith(".staging-") for n in _listing(directory))
    with open(target) as f:
        assert f.read() == before


def test_keyboard_interrupt_propagates_and_cleans_stage(site, monkeypatch):
    _kind, directory, _target, call = site
    call("first", 1)

    def bad_rename(src, dst, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(fs, "rename", bad_rename)
    with pytest.raises(KeyboardInterrupt):
        call("second", 2)
    assert not any(n.startswith(".staging-") for n in _listing(directory))


def test_failed_rename_and_failed_rm_propagates_the_rename_error(site, monkeypatch):
    _kind, _directory, _target, call = site
    call("first", 1)
    boom = OSError("rename failed")

    def bad_rename(src, dst, **kw):
        raise boom

    def bad_rm(*a, **kw):
        raise RuntimeError("rm failed")

    monkeypatch.setattr(fs, "rename", bad_rename)
    monkeypatch.setattr(fs, "rm", bad_rm)
    with pytest.raises(OSError) as ei:
        call("second", 2)
    assert ei.value is boom


def test_write_deploy_record_non_serialisable_raises_typeerror_and_leaves_no_file(tmp_path):
    reg = str(tmp_path / "reg")
    vdir = os.path.join(reg, "m", "v1")
    os.makedirs(vdir)
    with pytest.raises(TypeError):
        model_registry.write_deploy_record("m", 1, {"bad": {1, 2}}, reg)
    assert os.listdir(vdir) == []


# --- error messages, full str(exc) ----------------------------------------------------


def test_error_messages_model_registry(tmp_path):
    reg = str(tmp_path / "reg")
    with pytest.raises(ValueError) as ei:
        model_registry.set_alias("m", "champion", 9, reg)
    assert str(ei.value) == (
        f"cannot set alias 'champion' -> v9: {os.path.join(reg, 'm', 'v9')!r} does not exist"
    )
    with pytest.raises(ValueError) as ei:
        model_registry.write_deploy_record("m", 9, {}, reg)
    assert str(ei.value) == (
        f"cannot write deploy record: {os.path.join(reg, 'm', 'v9')!r} does not exist"
    )
    with pytest.raises(ValueError) as ei:
        model_registry.set_alias("m", "v3", 1, reg)
    assert str(ei.value) == (
        "alias 'v3' is shadowed by the 'name@v<N>' version-pin shorthand and "
        "could never be resolved -- pick a different alias name."
    )


def test_error_messages_core_registry(tmp_path):
    reg = str(tmp_path / "reg")
    with pytest.raises(ValueError) as ei:
        core.set_alias("m", "champion", 9, reg, {})
    assert str(ei.value) == (
        f"cannot set alias 'champion' -> v9: {os.path.join(reg, 'm', 'v9')!r} does not exist"
    )
    with pytest.raises(ValueError) as ei:
        core.set_alias("m", "v3", 1, reg, {})
    assert str(ei.value) == (
        "alias 'v3' is shadowed by the 'name@v<N>' version-pin shorthand and "
        "could never be resolved -- pick a different alias name."
    )


# --- C14b: the best-effort discard, and the old names ----------------------------------


@pytest.mark.parametrize("discard", [model_registry._discard, core.discard])
def test_discard_missing_path_does_not_raise(discard, tmp_path):
    discard(str(tmp_path / "nope"), {})


@pytest.mark.parametrize("discard", [model_registry._discard, core.discard])
def test_discard_removes_and_swallows_rm_errors(discard, tmp_path, monkeypatch):
    d = tmp_path / "stage"
    d.mkdir()
    (d / "f").write_text("x")
    discard(str(d), {})
    assert not d.exists()

    def bad_rm(*a, **kw):
        raise RuntimeError("rm failed")

    monkeypatch.setattr(fs, "rm", bad_rm)
    discard(str(tmp_path / "anything"), {})


def test_staging_prefix_names_stay_importable():
    assert model_registry._STAGING_PREFIX == ".staging-"
    assert core.STAGING_PREFIX == ".staging-"
