"""`configure_storage` called in one test leaves the next test's fsspec state as it found it."""

from __future__ import annotations

import os

import fsspec
import pytest

from rise_eo.storage import azure as rise_eo_azure

_STATE_BEFORE = {}


def _fsspec_state():
    return os.environ.get("FSSPEC_ABFSS_ANON"), dict(fsspec.config.conf.get("abfss", {}))


def test_a_leaky_caller_configures_azure():
    _STATE_BEFORE["state"] = _fsspec_state()
    rise_eo_azure.configure_storage("azure")
    assert os.environ["FSSPEC_ABFSS_ANON"] == "false"


def test_b_the_leak_does_not_reach_the_next_test():
    if "state" not in _STATE_BEFORE:
        pytest.skip("needs test_a to run first")
    assert _fsspec_state() == _STATE_BEFORE["state"]
