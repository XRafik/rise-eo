"""Fixtures shared by every test module."""

from __future__ import annotations

import os

import fsspec
import pytest


@pytest.fixture(autouse=True)
def clean_fsspec_conf():
    """Snapshot and restore `fsspec.config.conf` and `FSSPEC_ABFSS_ANON` around every test."""
    before = {k: dict(v) for k, v in fsspec.config.conf.items()}
    before_env = os.environ.get("FSSPEC_ABFSS_ANON")
    yield
    fsspec.config.conf.clear()
    fsspec.config.conf.update(before)
    if before_env is None:
        os.environ.pop("FSSPEC_ABFSS_ANON", None)
    else:
        os.environ["FSSPEC_ABFSS_ANON"] = before_env
