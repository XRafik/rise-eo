"""Tests for rise_eo.catalog.restamp_cli / inspect_cli (spec 35 §6/§8.11)."""

import uuid

import geopandas as gpd
import pytest
import shapely.geometry as sg

from rise_eo.catalog import declaration as declaration_module
from rise_eo.catalog.declaration import S2_L2A_DECLARATION, CollectionDeclaration
from rise_eo.catalog.inspect_cli import inspect_catalog
from rise_eo.catalog.restamp_cli import restamp_catalog
from rise_eo.storage import fs


def _unstamped_catalog(path):
    gdf = gpd.GeoDataFrame({"id": ["t1"]}, geometry=[sg.Point(0, 0)], crs="EPSG:4326")
    fs.write_parquet(path, gdf)


def test_restamp_stamps_an_unstamped_catalog(tmp_path):
    p = str(tmp_path / "catalog.parquet")
    _unstamped_catalog(p)
    restamp_catalog(p, "sentinel-2-l2a")
    assert declaration_module.from_attrs(fs.read_parquet(p)) == S2_L2A_DECLARATION


def test_restamp_is_idempotent(tmp_path):
    p = str(tmp_path / "catalog.parquet")
    _unstamped_catalog(p)
    restamp_catalog(p, "sentinel-2-l2a")
    restamp_catalog(p, "sentinel-2-l2a")  # same declaration again -- no error
    assert declaration_module.from_attrs(fs.read_parquet(p)) == S2_L2A_DECLARATION


def test_restamp_refuses_differing_stamp_without_force(tmp_path):
    p = str(tmp_path / "catalog.parquet")
    gdf = gpd.GeoDataFrame({"id": ["t1"]}, geometry=[sg.Point(0, 0)], crs="EPSG:4326")
    declaration_module.to_attrs(gdf, CollectionDeclaration(reference_band="B04"))
    fs.write_parquet(p, gdf)

    with pytest.raises(ValueError, match="different stamp"):
        restamp_catalog(p, "sentinel-2-l2a")
    # unchanged.
    assert declaration_module.from_attrs(fs.read_parquet(p)) == CollectionDeclaration(reference_band="B04")


def test_restamp_force_overwrites_differing_stamp(tmp_path):
    p = str(tmp_path / "catalog.parquet")
    gdf = gpd.GeoDataFrame({"id": ["t1"]}, geometry=[sg.Point(0, 0)], crs="EPSG:4326")
    declaration_module.to_attrs(gdf, CollectionDeclaration(reference_band="B04"))
    fs.write_parquet(p, gdf)

    restamp_catalog(p, "sentinel-2-l2a", force=True)
    assert declaration_module.from_attrs(fs.read_parquet(p)) == S2_L2A_DECLARATION


def test_restamp_works_on_a_memory_filesystem_path():
    """The fsspec/non-local proof (spec 35 §8.11) -- works on `abfss://`/`s3://`
    the same way, without a local-path assumption."""
    p = f"memory://{uuid.uuid4()}/catalog.parquet"
    _unstamped_catalog(p)
    restamp_catalog(p, "sentinel-2-l2a")
    assert declaration_module.from_attrs(fs.read_parquet(p)) == S2_L2A_DECLARATION


def test_inspect_catalog_returns_stamped_json(tmp_path):
    p = str(tmp_path / "catalog.parquet")
    _unstamped_catalog(p)
    assert inspect_catalog(p) is None
    restamp_catalog(p, "sentinel-2-l2a")
    assert inspect_catalog(p) == declaration_module.to_json(S2_L2A_DECLARATION)
