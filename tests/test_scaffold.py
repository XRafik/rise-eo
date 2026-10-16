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
    import rise_eo

    assert rise_eo.__version__


def test_version_is_the_installed_distribution_version():
    import rise_eo

    assert rise_eo.__version__ == importlib.metadata.version("rise-eo")


@pytest.mark.skipif(os.environ.get("GITHUB_REF_TYPE") != "tag", reason="runs only when CI builds a tag")
def test_version_matches_the_release_tag():
    import rise_eo

    assert os.environ["GITHUB_REF_NAME"] == f"v{rise_eo.__version__}"


def test_submodules_import():
    for name in [
        "rise_eo.config",
        "rise_eo.storage.fs",
        "rise_eo.sources.cdse",
        "rise_eo.catalog.catalog",
        "rise_eo.raster.images",
        "rise_eo.bands.modify",
        "rise_eo.datacube.builder",
        "rise_eo.datacube.ops",
        "rise_eo.datacube.flatten",
        "rise_eo.workflows.task",
        "rise_eo.workflows.runners",
        "rise_eo.workflows.create_datacube",
    ]:
        importlib.import_module(name)


def test_config_defaults():
    from rise_eo import config

    assert config.REFERENCE_BAND == "B08"
    assert config.SCL_MASK_CLASSES == [0, 1, 3, 7, 8, 9, 10]
