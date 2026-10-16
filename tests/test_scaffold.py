"""Scaffold smoke tests: the package imports and exposes its public surface.

These pass on the empty scaffold (no implementation yet) — they only check the
skeleton is wired correctly. Real behavior tests come per spec (see each spec's
Tests section).
"""

import importlib
import importlib.metadata
import os

import pytest


def test_package_imports():
    import fsd

    assert fsd.__version__


def test_version_is_the_installed_distribution_version():
    import fsd

    assert fsd.__version__ == importlib.metadata.version("fsd")


@pytest.mark.skipif(os.environ.get("GITHUB_REF_TYPE") != "tag", reason="runs only when CI builds a tag")
def test_version_matches_the_release_tag():
    import fsd

    assert os.environ["GITHUB_REF_NAME"] == f"v{fsd.__version__}"


def test_submodules_import():
    for name in [
        "fsd.config",
        "fsd.storage.fs",
        "fsd.sources.cdse",
        "fsd.catalog.catalog",
        "fsd.raster.images",
        "fsd.bands.modify",
        "fsd.datacube.builder",
        "fsd.datacube.ops",
        "fsd.datacube.flatten",
        "fsd.workflows.task",
        "fsd.workflows.runners",
        "fsd.workflows.create_datacube",
    ]:
        importlib.import_module(name)


def test_config_defaults():
    from fsd import config

    assert config.REFERENCE_BAND == "B08"
    assert config.SCL_MASK_CLASSES == [0, 1, 3, 7, 8, 9, 10]
