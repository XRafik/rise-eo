"""Tests for the pure helpers in `demos/e2e_austria_aml.py` (spec 40 deliverable 3).

The demo run itself is not unit-testable by design (spec 40 §6) -- it needs a cluster.
Its *helpers* are, and three of them are the ones a real run bets 80 GB on:

- `_list_run_ids`/`_new_dispatch_timings`: how a step learns which run(s) its `rise-eo.*`
  call dispatched under, since no verb returns a run id (ADR 0021). Exercised here
  against a real fsspec backend (`memory://`), because the bug it has to survive --
  `fs.glob` returning the filesystem's own scheme-less path form -- only shows up
  against a backend, never against hand-written strings.
- `_asset_key`: the D14 archive-trust comparison key. The catalog's `files` column
  holds bare basenames that repeat on every row, so a key that isn't granule-scoped
  makes the check silently vacuous.
- `_expected_offset`: the independent re-derivation D14 needs to catch a catalog that
  disagrees with its own granules.
"""

from __future__ import annotations

import importlib
import json
import os
import sys

import pytest

from rise_eo.storage import fs

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEMOS_DIR = os.path.join(os.path.dirname(_HERE), "demos")
sys.path.insert(0, _DEMOS_DIR)
demo = importlib.import_module("e2e_austria_aml")


# --- dispatch-run discovery (D2/D9, ADR 0021) --------------------------------------

def _write_timing(root: str, run_id: str, payload: dict | None = None) -> None:
    with fs.open(f"{root}/runs/{run_id}/_timing.json", "w") as f:
        json.dump(payload or {"run_id": run_id, "jobs": {}, "wall": {}}, f)


def test_list_run_ids_extracts_ids_from_a_real_backends_glob_form():
    """`fs.glob` returns the backend's OWN path form -- `memory://` strips the scheme
    exactly as adlfs does. The helper must recover the run id from the tail, never by
    string-comparing a glob hit to a url the caller built."""
    root = "memory://disc_basic"
    _write_timing(root, "20260728T120000Z")
    _write_timing(root, "20260728T130000Z")
    # a run that has not written telemetry yet must not be reported
    with fs.open(f"{root}/runs/20260728T140000Z/_status/0.json", "w") as f:
        json.dump({"unit": 0}, f)

    assert demo._list_run_ids(root) == {"20260728T120000Z", "20260728T130000Z"}


def test_list_run_ids_is_empty_when_nothing_has_dispatched():
    assert demo._list_run_ids("memory://disc_empty") == set()


def test_new_dispatch_timings_returns_only_runs_created_after_the_snapshot():
    """The before/after diff each dispatching step wraps its `rise-eo.*` call in."""
    root = "memory://disc_diff"
    _write_timing(root, "run-old")
    before = demo._list_run_ids(root)

    _write_timing(root, "run-new", {"run_id": "run-new", "jobs": {"0": {}}, "wall": {}})
    new = demo._new_dispatch_timings(root, before)

    assert [t["run_id"] for t in new] == ["run-new"]


def test_new_dispatch_timings_picks_up_both_runs_of_one_training_data_call():
    """D1: `create_training_data(runner="aml")` dispatches TWO runs inside ONE call
    (the cube-build fan-out and the flatten reduce) -- the exact case ADR 0021 cites
    for why telemetry is a file rather than a return value. Both must come back, in a
    stable order."""
    root = "memory://disc_two"
    before = demo._list_run_ids(root)
    _write_timing(root, "20260728T100000Z-build")
    _write_timing(root, "20260728T100500Z-flatten")

    new = demo._new_dispatch_timings(root, before)

    assert [t["run_id"] for t in new] == ["20260728T100000Z-build", "20260728T100500Z-flatten"]


def test_new_dispatch_timings_orders_by_execution_not_by_run_id():
    """Run ids look like sortable timestamps and are not. `create_training_data` mints its
    own right after preflight and gives it to the flatten REDUCE, while the build FAN-OUT
    mints one later when it dispatches -- so the flatten's id sorts first despite running
    second. Real values from 20260729T132222Z: flatten id 132929Z executed 13:32:58, build
    id 133005Z executed 13:30:07. Sorted by id, `3_training_data[0]` was the reduce."""
    root = "memory://disc_order"
    before = demo._list_run_ids(root)
    _write_timing(root, "20260729T132929Z",      # flatten: earlier id, LATER execution
                  {"run_id": "flatten", "jobs": {"0": {}},
                   "wall": {"t_start": "2026-07-29T13:32:58+00:00"}})
    _write_timing(root, "20260729T133005Z",      # build: later id, EARLIER execution
                  {"run_id": "build", "jobs": {str(k): {} for k in range(32)},
                   "wall": {"t_start": "2026-07-29T13:30:07+00:00"}})

    new = demo._new_dispatch_timings(root, before)

    assert [t["run_id"] for t in new] == ["build", "flatten"]
    assert len(new[0]["jobs"]) == 32 and len(new[1]["jobs"]) == 1


def test_new_dispatch_timings_tolerates_a_missing_t_start():
    root = "memory://disc_no_tstart"
    before = demo._list_run_ids(root)
    _write_timing(root, "r1", {"run_id": "r1", "jobs": {}, "wall": {}})
    assert [t["run_id"] for t in demo._new_dispatch_timings(root, before)] == ["r1"]


def test_new_dispatch_timings_is_empty_when_a_step_dispatched_nothing():
    """A fully-resumed step (every output already on blob) dispatches no job, so it
    embeds no telemetry -- and must not raise."""
    root = "memory://disc_none"
    _write_timing(root, "run-old")
    assert demo._new_dispatch_timings(root, demo._list_run_ids(root)) == []


# --- D14 archive-trust keys ---------------------------------------------------------

def test_asset_key_is_granule_scoped_not_a_bare_basename():
    """The catalog's `files` column is bare basenames, identical on every row
    (`B04.tif,B08.tif,…`). Keying on the basename alone would collapse 207 granules
    into 5 names and make D14's object check unable to detect a missing granule."""
    a = demo._asset_key("fs/prefix/imagery/S2/2018/09/28/S2B_MSIL2A_a_N0500_T33UWQ_x/B04.tif")
    b = demo._asset_key("fs/prefix/imagery/S2/2018/09/28/S2B_MSIL2A_b_N0500_T33UVP_y/B04.tif")
    assert a != b
    assert a == "S2B_MSIL2A_a_N0500_T33UWQ_x/B04.tif"


def test_asset_key_matches_across_scheme_and_scheme_less_forms():
    """A url the caller built (`abfss://…`) and the same object as `fs.glob` returned
    it (no scheme) must produce the same key -- that is the whole point."""
    built = "abfss://fs@acct.dfs.core.windows.net/prefix/imagery/x/GRANULE_N0500_id/SCL.tif"
    globbed = "fs/prefix/imagery/x/GRANULE_N0500_id/SCL.tif"
    assert demo._asset_key(built) == demo._asset_key(globbed) == "GRANULE_N0500_id/SCL.tif"


@pytest.mark.parametrize("granule_id,expected", [
    ("S2B_MSIL2A_20180928T100019_N0500_R122_T33UWQ_20230710T001349", -1000),  # >= 04.00
    ("S2A_MSIL2A_20180101T100019_N0400_R122_T33UWQ_20220101T000000", -1000),  # exactly 04.00
    ("S2A_MSIL2A_20180101T100019_N0212_R122_T33UWQ_20180101T000000", 0),      # < 04.00
    ("something-without-a-baseline-token", None),
])
def test_expected_offset_derives_the_esa_offset_from_the_baseline_in_the_id(granule_id, expected):
    """ESA: offset = -1000 for processing baseline >= 04.00, else 0. Derived from the
    id independently of `sources/cdse`, which is what makes the assertion meaningful --
    checking a value against the function that wrote it would catch nothing."""
    assert demo._expected_offset(granule_id) == expected


# --- D14 archive-trust assertions, end to end ---------------------------------------

def _write_tif(fp, *, scale, offset, nodata):
    """A 2x2 uint16 GeoTIFF carrying the spec-34 §1a radiometry tags. Real, because D14
    now reads the COG's own header rather than trusting the catalog about itself."""
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        fp, "w", driver="GTiff", height=2, width=2, count=1, dtype="uint16",
        crs="EPSG:32633", transform=from_origin(500000, 5300000, 10, 10), nodata=nodata,
    ) as dst:
        dst.write(np.ones((2, 2), dtype="uint16") * 1234, 1)
        dst.scales = (scale,)
        dst.offsets = (offset,)


def _build_archive(tmp_path, *, granules=3, sidecar=True, offset=-1000, nodata=None,
                   drop_asset=None, extra_object=False, mpc_style_ids=False,
                   cog_offset=None, cog_scale=None):
    """A miniature imagery archive on disk + its catalog, laid out exactly as a download
    leaves it (spec 59 D2): `<imagery>/<collection>/YYYY/MM/DD/<granule_id>/{B04.tif,SCL.tif,
    MTD_TL.xml}`, catalog at `<imagery>/<collection>/catalog.parquet`, with `files`
    holding bare basenames (identical on every row) -- the property that makes a
    basename-keyed check vacuous.

    `mpc_style_ids` drops the `_N####_` baseline field, as real MPC item ids do.
    `cog_offset`/`cog_scale` override what the COG header declares, so the catalog and
    the granule can be made to disagree.
    """
    import shapely.geometry as sg

    from rise_eo import config
    from rise_eo.catalog.catalog import TileCatalog
    from rise_eo.catalog.declaration import CollectionDeclaration

    imagery = tmp_path / "imagery"
    eff_nodata = config.NODATA if nodata is None else nodata
    rows = []
    for i in range(granules):
        gid = (f"S2B_MSIL2A_2018092{i}T100019_R122_T33UWQ_20230710T00134{i}"
               if mpc_style_ids else
               f"S2B_MSIL2A_2018092{i}T100019_N0500_R122_T33UWQ_20230710T00134{i}")
        folder = imagery / config.SATELLITE_S2L2A / "2018" / "09" / f"2{i}" / gid
        folder.mkdir(parents=True)
        names = ["B04.tif", "SCL.tif"] + (["MTD_TL.xml"] if sidecar else [])
        for name in names:
            if drop_asset is not None and (i, name) == drop_asset:
                continue  # declared in the catalog, absent on "blob"
            if name.endswith(".tif"):
                reflectance = name.startswith("B")
                _write_tif(
                    folder / name,
                    scale=(cog_scale if cog_scale is not None else config.S2_REFLECTANCE_SCALE)
                    if reflectance else 1.0,
                    offset=(cog_offset if cog_offset is not None
                            else offset * config.S2_REFLECTANCE_SCALE) if reflectance else 0.0,
                    nodata=eff_nodata,
                )
            else:
                (folder / name).write_bytes(b"x" * 128)
        rows.append(dict(
            id=gid, collection=config.SATELLITE_S2L2A,
            timestamp=f"2018-09-2{i}T10:00:19Z", s3url=f"s3://eodata/{gid}.SAFE",
            local_folderpath=str(folder), files=",".join(names), cloud_cover=1.0,
            offset=offset, nodata=eff_nodata,
            geometry=sg.box(i, i, i + 1, i + 1),
        ))
    if extra_object:
        stray = imagery / config.SATELLITE_S2L2A / "2018" / "09" / "99" / "leftover"
        stray.mkdir(parents=True)
        _write_tif(stray / "B04.tif", scale=config.S2_REFLECTANCE_SCALE,
                   offset=offset * config.S2_REFLECTANCE_SCALE, nodata=eff_nodata)

    catalog_fp = str(imagery / config.SATELLITE_S2L2A / "catalog.parquet")
    TileCatalog(catalog_fp, declaration=CollectionDeclaration(reference_band="B08")).append(rows)
    return catalog_fp, str(imagery)


def test_archive_trust_passes_on_a_well_formed_archive_including_the_xml_sidecar(tmp_path):
    """The regression that matters most: CDSE declares `MTD_TL.xml` alongside the
    bands, so a check that lists only `*.tif` reports the sidecar as missing and aborts
    the demo run -- AFTER the whole ~80 GB download has been paid for."""
    catalog_fp, imagery = _build_archive(tmp_path, sidecar=True)
    trust = demo._assert_archive_trustworthy(catalog_fp, imagery)
    assert trust["n_catalog_rows"] == 3
    assert trust["n_declared_assets"] == 9        # 3 granules x (B04, SCL, MTD_TL)
    assert trust["n_undeclared_objects"] == 0


def test_archive_trust_detects_a_single_missing_asset(tmp_path):
    """The check has to be granule-scoped to see this at all: every granule declares
    the same 3 basenames, so a basename-keyed comparison still finds `B04.tif` present
    (from the other two granules) and reports a complete archive."""
    catalog_fp, imagery = _build_archive(tmp_path, drop_asset=(1, "B04.tif"))
    with pytest.raises(demo.PreflightFailure, match="not found on blob"):
        demo._assert_archive_trustworthy(catalog_fp, imagery)


def test_archive_trust_surfaces_undeclared_objects_without_failing(tmp_path):
    """A catalog may legitimately be a strict subset of what is on blob (a partial
    re-run's leftovers), so this is counted and reported, not fatal."""
    catalog_fp, imagery = _build_archive(tmp_path, extra_object=True)
    assert demo._assert_archive_trustworthy(catalog_fp, imagery)["n_undeclared_objects"] == 1


def test_archive_trust_ignores_another_collection_in_the_shared_archive(tmp_path):
    """Spec 59 D12: the archive is shared, so S1 granules sit beside S2 under one root.
    They are another catalog's business -- an unscoped `**` glob would count them as
    this catalog's undeclared objects."""
    catalog_fp, imagery = _build_archive(tmp_path)
    s1 = tmp_path / "imagery" / "sentinel-1-rtc" / "2018" / "06" / "26" / "S1B_IW_GRDH_rtc"
    s1.mkdir(parents=True)
    _write_tif(s1 / "vv.tif", scale=1.0, offset=0.0, nodata=0)
    assert demo._assert_archive_trustworthy(catalog_fp, imagery)["n_undeclared_objects"] == 0


def test_archive_trust_rejects_an_offset_its_own_baseline_contradicts(tmp_path):
    """The invisible failure D14 exists for: `N0500` implies -1000, so a row declaring
    0 means every reflectance built from this archive is ~1000 DN high and the pipeline
    still goes green."""
    catalog_fp, imagery = _build_archive(tmp_path, offset=0)
    with pytest.raises(demo.PreflightFailure, match="un-harmonized radiometry"):
        demo._assert_archive_trustworthy(catalog_fp, imagery)


def test_archive_trust_rejects_a_declared_nodata_the_builder_would_not_honor(tmp_path):
    catalog_fp, imagery = _build_archive(tmp_path, nodata=65535)  # valid uint16, wrong value
    with pytest.raises(demo.PreflightFailure, match="config.NODATA"):
        demo._assert_archive_trustworthy(catalog_fp, imagery)


def test_archive_trust_rejects_a_cog_whose_tags_contradict_the_catalog(tmp_path):
    """The `c2bf1f1` black-tile class: the GDAL tag is in REFLECTANCE units to match
    `scale=1/10000`, the catalog column is in DN. Stamping the DN offset (-1000) beside
    a 1/10000 scale makes an `unscale=true` viewer compute `DN/10000 - 1000` -- pure
    black. The old "GDAL tag agrees with STAC" test passed straight through it, because
    both carried the same wrong value."""
    catalog_fp, imagery = _build_archive(tmp_path, offset=-1000, cog_offset=-1000.0)
    with pytest.raises(demo.PreflightFailure, match="stamps OFFSET"):
        demo._assert_archive_trustworthy(catalog_fp, imagery)


def test_archive_trust_still_checks_offset_on_mpc_ids_that_carry_no_baseline(tmp_path):
    """Real MPC item ids drop the `_N####_` field, so `_expected_offset` returns None
    for every row and the baseline cross-check contributes nothing. The COG-tag
    comparison is what has to catch a wrong offset there -- and the result says plainly
    that zero rows were baseline-checked rather than implying the offset was verified
    twice."""
    ok_fp, ok_imagery = _build_archive(tmp_path / "ok", mpc_style_ids=True)
    trust = demo._assert_archive_trustworthy(ok_fp, ok_imagery)
    assert trust["n_offset_baseline_crosschecked"] == 0
    assert trust["n_sampled"] == 3

    bad_fp, bad_imagery = _build_archive(
        tmp_path / "bad", mpc_style_ids=True, offset=-1000, cog_offset=0.0)
    with pytest.raises(demo.PreflightFailure, match="stamps OFFSET"):
        demo._assert_archive_trustworthy(bad_fp, bad_imagery)


def test_archive_trust_rejects_an_offset_outside_the_two_esa_values(tmp_path):
    catalog_fp, imagery = _build_archive(tmp_path, mpc_style_ids=True, offset=-500)
    with pytest.raises(demo.PreflightFailure, match="ESA defines"):
        demo._assert_archive_trustworthy(catalog_fp, imagery)


def test_archive_trust_requires_a_stamped_source_declaration(tmp_path):
    from rise_eo.catalog.catalog import TileCatalog

    catalog_fp, imagery = _build_archive(tmp_path)
    rows = TileCatalog(catalog_fp).read()
    unstamped = str(tmp_path / "imagery" / "unstamped.parquet")
    TileCatalog(unstamped).append(rows.to_dict("records"))
    with pytest.raises(demo.PreflightFailure, match="no stamped CollectionDeclaration"):
        demo._assert_archive_trustworthy(unstamped, imagery)


def test_archive_trust_rejects_a_zero_byte_asset(tmp_path):
    catalog_fp, imagery = _build_archive(tmp_path)
    rows = demo.TileCatalog(catalog_fp).read()
    zeroed = os.path.join(str(rows.iloc[0]["local_folderpath"]), "B04.tif")
    with open(zeroed, "wb"):
        pass
    with pytest.raises(demo.PreflightFailure, match="zero-byte"):
        demo._assert_archive_trustworthy(catalog_fp, imagery)


# --- D2/D11: a stale AML image silently voids the headline metric -------------------

def _dispatch(with_stamps: bool, n=3):
    """One `_timing.json`-shaped block. Without the stamps is what an Environment built
    before spec 40 actually produced on 2026-07-29: `work_seconds` present (it predates
    this spec), all four in-job stamps null."""
    jobs = {}
    for k in range(n):
        job = {"submitted_at": "2026-07-29T10:55:14+00:00",
               "returned_at": "2026-07-29T11:00:20+00:00",
               "work_seconds": 28.5, "job_admission_seconds": None,
               "import_seconds": None, "dispatch_overhead_seconds": 276.9,
               "process_start_at": None, "work_start_at": None,
               "work_end_at": None, "ended_at": None}
        if with_stamps:
            job.update(process_start_at="2026-07-29T10:59:00+00:00",
                       work_start_at="2026-07-29T10:59:10+00:00",
                       work_end_at="2026-07-29T10:59:38+00:00",
                       ended_at="2026-07-29T10:59:39+00:00",
                       job_admission_seconds=225.2)
        jobs[str(k)] = job
    return {"run_id": "r", "jobs": jobs, "wall": {}}


def test_dispatch_telemetry_gate_fails_on_an_image_that_predates_the_stamps():
    """The 2026-07-29 run: green end to end, 97 jobs, four dispatches, and
    `job_admission_seconds: null` on every one -- because the four stamps are written by
    the `rise-eo` inside the AML image, not by the driver's checkout. Nothing failed, which
    is precisely the problem."""
    with pytest.raises(demo.PreflightFailure, match="in-job stamps"):
        demo._assert_dispatch_telemetry_complete([_dispatch(with_stamps=False)],
                                                 step="2_download")


def test_dispatch_telemetry_gate_names_the_fix_not_just_the_symptom():
    with pytest.raises(demo.PreflightFailure) as exc:
        demo._assert_dispatch_telemetry_complete([_dispatch(with_stamps=False)],
                                                 step="2_download")
    msg = str(exc.value)
    assert "rebuild" in msg.lower() and "AZ_ENV_VERSION" in msg
    assert "--run-id" in msg          # and that resuming is cheap


def test_dispatch_telemetry_gate_passes_on_a_current_image():
    demo._assert_dispatch_telemetry_complete([_dispatch(with_stamps=True)], step="2_download")


def test_dispatch_telemetry_gate_accepts_a_partially_stamped_dispatch():
    """One job crashing before it could write `_status` (D3/D15) is a different failure --
    it must not read as a stale image."""
    mixed = _dispatch(with_stamps=True)
    mixed["jobs"]["0"].update(process_start_at=None, job_admission_seconds=None)
    demo._assert_dispatch_telemetry_complete([mixed], step="2_download")


def test_dispatch_telemetry_gate_is_a_noop_when_nothing_was_dispatched():
    """A fully-resumed step dispatches no job at all -- no telemetry is not stale telemetry."""
    demo._assert_dispatch_telemetry_complete([], step="2_download")
    demo._assert_dispatch_telemetry_complete([{"run_id": "r", "jobs": {}, "wall": {}}],
                                             step="2_download")


# --- D4: driver dependencies are a preflight failure, not a step-4 surprise ---------

def test_missing_driver_deps_is_empty_when_everything_is_installed():
    assert demo._missing_driver_deps() == []


def test_missing_driver_deps_names_the_module_the_extra_and_one_install_line(monkeypatch):
    """The 2026-07-29 failure: `ModuleNotFoundError: joblib` at `4_train_bundle`, three
    steps and one ~80 GB download past the point a one-line pip install would have fixed
    it. D4 says preflight is total and fails in seconds -- so it must name the module,
    what needs it, and a single command that fixes every miss at once."""
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda m, *a, **k: None if m in ("joblib", "sklearn", "s2") else real(m))

    lines = demo._missing_driver_deps()

    assert any("joblib" in ln and "model-example" in ln for ln in lines)
    assert any("s2" in ln and "grid" in ln for ln in lines)
    install = lines[-1]
    # ONE line that fixes all three, with each missing extra folded in exactly once.
    assert install.count("pip install") == 1
    assert "grid" in install and "model-example" in install


def test_every_driver_dep_names_a_real_extra_in_pyproject():
    """The table is documentation people act on (E2E_AUSTRIA_AML.md §8.1 mirrors it), so
    an extra that does not exist would send the next operator to a failing command."""
    import re

    pyproject = os.path.join(os.path.dirname(_HERE), "pyproject.toml")
    with open(pyproject) as f:
        text = f.read()
    declared = set(re.findall(r"^([a-z0-9-]+) = \[", text, flags=re.MULTILINE))

    for _module, extra, _why in demo._DRIVER_DEPS:
        assert extra in declared, f"{extra!r} is not an extra in pyproject.toml"


def test_preflight_deps_cover_what_the_steps_actually_import():
    """Guards the drift this check exists to prevent: a step gains an import, nobody adds
    it here, and preflight goes green right up to the crash. Scans the demo script's own
    step functions for third-party imports and requires each to be declared."""
    import ast

    src = os.path.join(os.path.dirname(_HERE), "demos", "e2e_austria_aml.py")
    with open(src) as f:
        tree = ast.parse(f.read())

    declared = {m.split(".")[0] for m, _, _ in demo._DRIVER_DEPS}
    stdlib_or_local = {
        "__future__", "os", "sys", "json", "time", "re", "signal", "datetime", "contextlib",
        "argparse", "importlib", "ast", "statistics", "rise_eo", "adapters",
        "numpy", "pandas", "geopandas", "shapely", "rasterio",  # rise-eo CORE deps
    }

    missed = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            names = [node.module] if node.module and node.level == 0 else []
        for name in names:
            top = name.split(".")[0]
            if top and top not in declared and top not in stdlib_or_local:
                missed.add(top)

    assert not missed, (
        f"third-party import(s) {sorted(missed)} in demos/e2e_austria_aml.py are not in "
        "_DRIVER_DEPS -- preflight would not catch them and the run would die mid-step"
    )


# --- fs.modified (the clock-skew probe's only route to a server-side stamp) ---------

def test_fs_modified_returns_the_backends_own_stamp():
    """`ls`/`glob` return bare strings (`ls` passes `detail=False`), so `fs.modified`
    is the only way spec 40 D11 can compare the driver's clock to storage's."""
    url = "memory://skew/probe.txt"
    fs.write_text(url, "skew-probe")
    assert demo.pd.Timestamp(fs.modified(url)) is not None


def test_fs_modified_returns_none_rather_than_raising_without_an_mtime(tmp_path, monkeypatch):
    """A backend with no mtime must read as UNMEASURED, not as a confident zero skew --
    a silently-zero skew would make every job_admission figure look better-bounded
    than it is."""
    class _NoMtime:
        def modified(self, path):
            raise NotImplementedError

    monkeypatch.setattr(fs, "_fs_and_path", lambda url, opts=None: (_NoMtime(), "p"))
    assert fs.modified("memory://whatever") is None
