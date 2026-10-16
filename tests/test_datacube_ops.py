"""Tests for rise_eo.datacube.ops (spec 04). Synthetic, deterministic."""

import datetime

import numpy as np
import pandas as pd
import pytest

from rise_eo.datacube import ops


def _ts(day):
    return pd.Timestamp(f"2018-01-{day:02d}", tz="UTC")


def test_apply_cloud_mask_scl_masks_only_flagged_pixels():
    # (t=1, H=1, W=2, bands=[B04, SCL]); pixel 0 has SCL=8 (cloud), pixel 1 SCL=4 (ok)
    dc = np.array([[[[100, 8], [200, 4]]]], dtype=np.int32)
    md = {"bands": ["B04", "SCL"]}
    out, _ = ops.apply_cloud_mask_scl(dc, md, mask_classes=[8, 9], mask_value=0)
    assert out[0, 0, 0, 0] == 0     # B04 zeroed where SCL==8
    assert out[0, 0, 1, 0] == 200   # B04 untouched where SCL==4
    assert out[0, 0, 0, 1] == 8 and out[0, 0, 1, 1] == 4  # SCL itself untouched


def test_drop_bands():
    dc = np.arange(2 * 1 * 1 * 3).reshape(2, 1, 1, 3)
    md = {"bands": ["B04", "B08", "SCL"]}
    out, md2 = ops.drop_bands(dc, md, bands_to_drop=["SCL"])
    assert md2["bands"] == ["B04", "B08"]
    assert out.shape == (2, 1, 1, 2)
    assert np.array_equal(out[..., 0], dc[..., 0])  # B04 preserved


def test_median_mosaic_calendar_default_labels_by_window_start():
    # 3 acquisitions: days 1, 6 (window 0: [01-01,01-21)) and 26 (window 1: [01-21,02-10])
    ts = [_ts(1), _ts(6), _ts(26)]
    dc = np.array([10, 0, 30], dtype=np.int32).reshape(3, 1, 1, 1)  # day6 masked (0)
    md = {"bands": ["B04"], "timestamps": ts}
    out, md2 = ops.median_mosaic(
        dc, md, startdate=datetime.datetime(2018, 1, 1),
        enddate=datetime.datetime(2018, 2, 1), mosaic_days=20, mask_value=0,
    )
    assert out.shape == (2, 1, 1, 1)                 # 2 calendar windows
    assert out[0, 0, 0, 0] == 10                     # nanmedian(10, masked) = 10
    assert out[1, 0, 0, 0] == 30
    # calendar default (spec 15): labels are window-start boundaries, not acquisitions
    assert md2["timestamps"] == [_ts(1), _ts(21)]
    assert md2["mosaic_index_intervals"] == [(0, 1), (2, 2)]
    assert md2["previous_timestamps"] == ts


def test_median_mosaic_acquisition_legacy_labels_by_first_acquisition():
    ts = [_ts(1), _ts(6), _ts(26)]
    dc = np.array([10, 0, 30], dtype=np.int32).reshape(3, 1, 1, 1)
    md = {"bands": ["B04"], "timestamps": ts}
    out, md2 = ops.median_mosaic(
        dc, md, startdate=datetime.datetime(2018, 1, 1),
        enddate=datetime.datetime(2018, 2, 1), mosaic_days=20, mask_value=0,
        mosaic_scheme="acquisition",
    )
    assert out.shape == (2, 1, 1, 1)
    assert out[0, 0, 0, 0] == 10 and out[1, 0, 0, 0] == 30
    assert md2["timestamps"] == [ts[0], ts[2]]       # first ts of each bucket (legacy)
    assert md2["mosaic_index_intervals"] == [(0, 1), (2, 2)]


def test_median_mosaic_calendar_cross_shape_identical_timestamps():
    """Two cubes with *different* acquisition dates but the same window inputs get an
    identical timestamps axis — the property flatten (spec 05) relies on."""
    kw = dict(startdate=datetime.datetime(2018, 1, 1),
              enddate=datetime.datetime(2018, 2, 10), mosaic_days=20)
    dc = np.ones((2, 1, 1, 1), dtype=np.int32)
    # cube A samples days 2 & 25; cube B days 8 & 30 — different orbits/tiles
    _, mdA = ops.median_mosaic(dc, {"bands": ["B04"], "timestamps": [_ts(2), _ts(25)]}, **kw)
    _, mdB = ops.median_mosaic(dc, {"bands": ["B04"], "timestamps": [_ts(8), _ts(30)]}, **kw)
    assert mdA["timestamps"] == mdB["timestamps"] == [_ts(1), _ts(21)]


def test_median_mosaic_calendar_emits_empty_window_as_nodata():
    """A calendar window with no acquisitions is still emitted (all-nodata), so the
    timestamps axis length is fixed by the calendar, not the data."""
    ts = [_ts(2), pd.Timestamp("2018-02-15", tz="UTC")]   # nothing in [01-21,02-10)
    dc = np.array([10, 40], dtype=np.int32).reshape(2, 1, 1, 1)
    md = {"bands": ["B04"], "timestamps": ts}
    out, md2 = ops.median_mosaic(
        dc, md, startdate=datetime.datetime(2018, 1, 1),
        enddate=datetime.datetime(2018, 3, 1), mosaic_days=20, mask_value=0,
    )
    # windows: [01-01,01-21) has day2; [01-21,02-10) empty; [02-10,03-02] has 02-15
    assert out.shape == (3, 1, 1, 1)
    assert [int(o[0, 0, 0]) for o in out] == [10, 0, 40]   # middle window = nodata
    assert md2["timestamps"] == [_ts(1), _ts(21), pd.Timestamp("2018-02-10", tz="UTC")]
    assert md2["mosaic_index_intervals"] == [(0, 0), None, (1, 1)]


def test_median_mosaic_calendar_boundary_goes_to_later_window():
    # a timestamp exactly on a window boundary (day 21) lands in the *later* window
    ts = [_ts(5), _ts(21)]
    dc = np.array([10, 20], dtype=np.int32).reshape(2, 1, 1, 1)
    md = {"bands": ["B04"], "timestamps": ts}
    _, md2 = ops.median_mosaic(
        dc, md, startdate=datetime.datetime(2018, 1, 1),
        enddate=datetime.datetime(2018, 2, 10), mosaic_days=20,
    )
    assert md2["mosaic_index_intervals"] == [(0, 0), (1, 1)]  # day21 -> window 1, not 0


def test_median_mosaic_noop_when_days_lt_1():
    dc = np.ones((2, 1, 1, 1), dtype=np.int32)
    md = {"bands": ["B04"], "timestamps": [_ts(1), _ts(2)]}
    out, md2 = ops.median_mosaic(dc, md, startdate=datetime.datetime(2018, 1, 1),
                                 enddate=datetime.datetime(2018, 2, 1), mosaic_days=0)
    assert out is dc and md2 is md


def test_mosaic_ranges_validation():
    ts = [_ts(5), _ts(10)]
    with pytest.raises(ValueError, match="startdate must be"):
        ops._get_mosaic_ts_index_ranges(ts, datetime.datetime(2018, 1, 8),
                                        datetime.datetime(2018, 2, 1), 20)
    with pytest.raises(ValueError, match="not sorted"):
        ops._get_mosaic_ts_index_ranges([_ts(10), _ts(5)],
                                        datetime.datetime(2018, 1, 1),
                                        datetime.datetime(2018, 2, 1), 20)
    # calendar path shares the same guards
    with pytest.raises(ValueError, match="startdate must be"):
        ops._calendar_windows([_ts(5)], datetime.datetime(2018, 1, 8),
                              datetime.datetime(2018, 2, 1), 20)


def test_median_mosaic_rejects_unknown_scheme():
    md = {"bands": ["B04"], "timestamps": [_ts(1), _ts(2)]}
    with pytest.raises(ValueError, match="mosaic_scheme must be one of"):
        ops.median_mosaic(np.ones((2, 1, 1, 1), dtype=np.int32), md,
                          startdate=datetime.datetime(2018, 1, 1),
                          enddate=datetime.datetime(2018, 2, 1), mosaic_scheme="nope")


def test_run_ops_threads_sequence():
    dc = np.array([[[[100, 8]]]], dtype=np.int32)  # (1,1,1,[B04,SCL]) SCL=8
    md = {"bands": ["B04", "SCL"]}
    out, md2 = ops.run_ops(dc, md, [
        (ops.apply_cloud_mask_scl, {"mask_classes": [8]}),
        (ops.drop_bands, {"bands_to_drop": ["SCL"]}),
    ])
    assert md2["bands"] == ["B04"]
    assert out.shape == (1, 1, 1, 1) and out[0, 0, 0, 0] == 0  # masked then SCL dropped


def test_area_median():
    # (t=1, H=2, W=2, b=1) values 10,20,30,0(masked) -> nanmedian(10,20,30)=20
    dc = np.array([10, 20, 30, 0], dtype=np.int32).reshape(1, 2, 2, 1)
    out, md = ops.area_median(dc, {"bands": ["B04"]}, mask_value=0)
    assert out.shape == (1, 1, 1, 1) and out[0, 0, 0, 0] == 20
    assert md["previous_height_width"] == (2, 2)
