"""MPC `max_cloudcover` is "at most" (<=): the STAC query must use `lte`, not `lt`.

Spec 32, Amendment A1. Both MPC search entry points are checked against a fake
`pystac_client.Client`, so nothing touches the network.
"""
from types import SimpleNamespace

import geopandas as gpd
import pystac_client
import pytest
from shapely.geometry import box

from rise_eo.sources import mpc


@pytest.mark.parametrize("name", ["_search_items", "_search_items_unsigned"])
def test_max_cloudcover_is_sent_as_lte(monkeypatch, name):
    recorded = {}

    class _Search:
        def items(self):
            return []

    class _Client:
        def search(self, **kw):
            recorded.update(kw)
            return _Search()

    monkeypatch.setattr(pystac_client.Client, "open", lambda *a, **kw: _Client())
    monkeypatch.setattr(
        mpc, "_import_pc", lambda: SimpleNamespace(sign_inplace=lambda item: item)
    )
    roi = gpd.GeoDataFrame(geometry=[box(15.0, 48.0, 15.1, 48.1)], crs="EPSG:4326")

    getattr(mpc, name)(roi, "2018-06-01", "2018-06-30", max_cloudcover=0.066284)

    assert recorded["query"] == {"eo:cloud_cover": {"lte": 0.066284}}
