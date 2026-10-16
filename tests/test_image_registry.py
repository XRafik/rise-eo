"""Spec 56 §9 step 3 -- `rise_eo.image.registry`: AC5 (parity with model registry) and AC6
(round-trip), against `tmp_path`."""

from __future__ import annotations

import json
import os

import pytest

from rise_eo.image import digest as digest_mod
from rise_eo.image import registry
from rise_eo.registry import _core as core
from rise_eo.storage import fs

RESOLVED = {
    "base": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04@sha256:" + "a" * 64,
    "rise_eo": "git+https://github.com/nikhilsrajan/fsd@" + "b" * 40,
    "extras": ["azure", "mpc"],
    "extra_pip": [],
    "base_resolved": True,
}


def _digest(resolved=RESOLVED):
    return digest_mod.digest(resolved)


# --- AC6: round-trip -----------------------------------------------------------------


def test_publish_then_resolve_round_trips(tmp_path):
    registry_root = str(tmp_path / "registry")
    d = _digest()

    v = registry.publish(
        "rise-eo-aml-env", registry_root, RESOLVED, d,
        aml={"name": "rise-eo-aml-env", "version": "7", "workspace": "ws"},
    )
    assert v == 1

    got = registry.resolve("rise-eo-aml-env:1", registry_root)
    assert got.definition == RESOLVED
    assert got.digest == d
    assert got.aml == {"name": "rise-eo-aml-env", "version": "7", "workspace": "ws"}


def test_publish_sets_an_alias(tmp_path):
    registry_root = str(tmp_path / "registry")
    v = registry.publish("rise-eo-aml-env", registry_root, RESOLVED, _digest(), alias="current")

    got = registry.resolve("rise-eo-aml-env@current", registry_root)
    assert got.version == v


# --- AC5: registry parity with models --------------------------------------------------


def test_publish_identical_digest_twice_is_idempotent(tmp_path):
    registry_root = str(tmp_path / "registry")
    v1 = registry.publish("rise-eo-aml-env", registry_root, RESOLVED, _digest())
    before = fs.ls(os.path.join(registry_root, "rise-eo-aml-env"))

    v2 = registry.publish("rise-eo-aml-env", registry_root, RESOLVED, _digest())
    after = fs.ls(os.path.join(registry_root, "rise-eo-aml-env"))

    assert v2 == v1
    assert len(after) == len(before)


def test_publish_changed_digest_creates_v2_and_leaves_v1_untouched(tmp_path):
    registry_root = str(tmp_path / "registry")
    v1 = registry.publish("rise-eo-aml-env", registry_root, RESOLVED, _digest())

    other = dict(RESOLVED, extras=["azure"])
    v2 = registry.publish("rise-eo-aml-env", registry_root, other, _digest(other))
    assert v2 == v1 + 1

    v1_path = core.version_path(registry_root, "rise-eo-aml-env", v1)
    with fs.open(os.path.join(v1_path, registry.DEFINITION_FILE), "r") as f:
        assert json.load(f)["definition"] == RESOLVED


def test_complete_json_is_written_last_and_an_incomplete_version_is_invisible(tmp_path, monkeypatch):
    registry_root = str(tmp_path / "registry")
    real_write_bytes = fs.write_bytes
    failed = {"once": False}

    def flaky(path, data, **kw):
        if not failed["once"] and os.path.basename(path) == registry.DEFINITION_FILE:
            failed["once"] = True
            raise OSError("simulated failure")
        return real_write_bytes(path, data, **kw)

    monkeypatch.setattr(core.fs, "write_bytes", flaky)

    with pytest.raises(OSError):
        registry.publish("rise-eo-aml-env", registry_root, RESOLVED, _digest())

    v1 = core.version_path(registry_root, "rise-eo-aml-env", 1)
    assert not fs.exists(os.path.join(v1, registry.DEFINITION_FILE))
    assert not fs.exists(os.path.join(v1, core.COMPLETE_FILE))
    assert core.list_versions(os.path.join(registry_root, "rise-eo-aml-env"), {}) == []

    version = registry.publish("rise-eo-aml-env", registry_root, RESOLVED, _digest())
    assert version == 1
    assert fs.exists(os.path.join(v1, core.COMPLETE_FILE))


def test_aliases_file_is_staged_and_renamed(tmp_path, monkeypatch):
    registry_root = str(tmp_path / "registry")
    registry.publish("rise-eo-aml-env", registry_root, RESOLVED, _digest(), alias="current")

    aliases_path = os.path.join(registry_root, "rise-eo-aml-env", core.ALIASES_FILE)
    written = []
    real_write_bytes = fs.write_bytes

    def spy(path, data, **kw):
        written.append(path)
        return real_write_bytes(path, data, **kw)

    monkeypatch.setattr(core.fs, "write_bytes", spy)
    other = dict(RESOLVED, extras=["azure"])
    registry.publish("rise-eo-aml-env", registry_root, other, _digest(other), alias="current")
    monkeypatch.undo()

    assert aliases_path not in written
    assert any(core.STAGING_PREFIX in p for p in written)


def test_find_by_digest_returns_none_when_absent(tmp_path):
    registry_root = str(tmp_path / "registry")
    assert registry.find_by_digest("rise-eo-aml-env", registry_root, "sha256:deadbeef") is None


def test_status_reports_registered_and_unregistered(tmp_path):
    registry_root = str(tmp_path / "registry")
    d = _digest()

    before = registry.status("rise-eo-aml-env", registry_root, d)
    assert before.state == "unregistered"
    assert before.registered is None

    registry.publish("rise-eo-aml-env", registry_root, RESOLVED, d)
    after = registry.status("rise-eo-aml-env", registry_root, d)
    assert after.state == "registered"
    assert after.registered == 1
