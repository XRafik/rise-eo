"""Spec 31 (P1 Azure compute seam) — synthetic/local only, no credentials, no network.

Covers: `to_vsi` translation, the `storage=` config seam (env + `fsspec.config.conf`),
the config-seam facts pinned against the installed adlfs/fsspec (`[azure]` extra —
skips if not installed), `storage_token` (mocked credential), `rio_open` routing
(mocked `rasterio.open`), a `memory://` scheme-routing round-trip, and the
`os.path.join` URL-safety claim §2/§6 relies on.
"""

from __future__ import annotations

import os
import sys
import types

import fsspec
import pytest

from rise_eo.storage import azure as rise_eo_azure
from rise_eo.storage.fs import to_vsi

pytest_adlfs = pytest.importorskip("adlfs", reason="needs the [azure] extra (adlfs)")


# --- to_vsi --------------------------------------------------------------------


def test_to_vsi_abfss_translates():
    assert (
        to_vsi("abfss://fs@acct.dfs.core.windows.net/a/b.tif")
        == "/vsiadls/fs/a/b.tif"
    )


def test_to_vsi_local_passthrough():
    assert to_vsi("/local/a/b.tif") == "/local/a/b.tif"
    assert to_vsi("file:///local/a/b.tif") == "file:///local/a/b.tif"


def test_to_vsi_az_alias():
    assert to_vsi("az://fs/a/b.tif") == "/vsiadls/fs/a/b.tif"


def test_to_vsi_account_extractable():
    assert rise_eo_azure.account_from_url("abfss://fs@acct.dfs.core.windows.net/a/b.tif") == "acct"
    assert rise_eo_azure.account_from_url("/local/a/b.tif") is None
    assert rise_eo_azure.account_from_url("az://fs/a/b.tif") is None


def test_to_vsi_malformed_abfss_raises():
    with pytest.raises(ValueError):
        to_vsi("abfss://not-a-real-host/a/b.tif")


def test_to_vsi_malformed_az_raises():
    with pytest.raises(ValueError):
        to_vsi("az://missing-path")


# --- storage= config seam -------------------------------------------------------


@pytest.fixture()
def _clean_fsspec_conf(monkeypatch):
    """`fsspec.config.conf` is process-global — snapshot + restore around each test."""
    before = {k: dict(v) for k, v in fsspec.config.conf.items()}
    before_env = os.environ.get("FSSPEC_ABFSS_ANON")
    yield
    fsspec.config.conf.clear()
    fsspec.config.conf.update(before)
    if before_env is None:
        os.environ.pop("FSSPEC_ABFSS_ANON", None)
    else:
        os.environ["FSSPEC_ABFSS_ANON"] = before_env


def test_configure_storage_azure_string_sets_env_and_conf(_clean_fsspec_conf):
    rise_eo_azure.configure_storage("azure")
    assert os.environ["FSSPEC_ABFSS_ANON"] == "false"
    assert fsspec.config.conf["abfss"]["anon"] is False


def test_configure_storage_azure_dict_form(_clean_fsspec_conf):
    rise_eo_azure.configure_storage({"backend": "azure"})
    assert os.environ["FSSPEC_ABFSS_ANON"] == "false"
    assert fsspec.config.conf["abfss"]["anon"] is False


def test_configure_storage_none_is_noop(_clean_fsspec_conf):
    os.environ.pop("FSSPEC_ABFSS_ANON", None)
    rise_eo_azure.configure_storage(None)
    assert "FSSPEC_ABFSS_ANON" not in os.environ
    assert "abfss" not in fsspec.config.conf


def test_configure_storage_bad_backend_raises(_clean_fsspec_conf):
    with pytest.raises(ValueError):
        rise_eo_azure.configure_storage("s3")
    with pytest.raises(ValueError):
        rise_eo_azure.configure_storage({"backend": "gcs"})


def test_api_storage_local_or_none_leaves_env_unset(tmp_path, monkeypatch):
    """`storage="local"`/`None` on a verb must not touch FSSPEC_ABFSS_*."""
    monkeypatch.delenv("FSSPEC_ABFSS_ANON", raising=False)
    from rise_eo import api

    assert api._check_local_seams("local", None) == []
    assert api._check_local_seams("local", "local") == []
    assert "FSSPEC_ABFSS_ANON" not in os.environ


def test_configure_storage_local_string_is_noop(_clean_fsspec_conf):
    os.environ.pop("FSSPEC_ABFSS_ANON", None)
    rise_eo_azure.configure_storage("local")
    assert "FSSPEC_ABFSS_ANON" not in os.environ
    assert "abfss" not in fsspec.config.conf


def test_api_check_local_seams_accepts_azure_storage():
    from rise_eo import api

    assert api._check_local_seams("local", "azure") == []
    assert api._check_local_seams("local", {"backend": "azure"}) == []


def test_api_check_local_seams_rejects_bad_backend():
    from rise_eo import api

    errs = api._check_local_seams("local", "s3")
    assert errs and "s3" in errs[0]


def test_api_check_local_seams_runner_still_rejected():
    from rise_eo import api

    errs = api._check_local_seams("batch", None)
    assert errs and "batch" in errs[0]


# --- spec 52 AC6/AC7: deploy/run_inference/verify_adapter/verify_image each authenticate ----


def test_deploy_calls_configure_storage_before_its_first_storage_access(_clean_fsspec_conf):
    """AC6. `deploy` no longer refuses `storage='azure'` (D4 removes the old
    `storage_allowed=False` gate); it must set up authenticated adlfs before it ever touches
    `bundle_path` or `registry`. A bogus `name` (caught by `check_name`, after the
    `configure_storage` call this spec adds but before any storage read) is the cheapest way
    to reach a guaranteed raise without needing a real bundle."""
    os.environ.pop("FSSPEC_ABFSS_ANON", None)
    from rise_eo import api

    with pytest.raises(api.PreflightError):
        api.deploy(
            "/nonexistent/bundle", name="bad/name", registry="/nonexistent/registry",
            environment="rise-eo-infer-sklearn:1", storage="azure",
        )
    assert os.environ["FSSPEC_ABFSS_ANON"] == "false"


def test_run_inference_calls_configure_storage_before_its_first_storage_access(
    _clean_fsspec_conf,
):
    """AC7, in a process where no other verb has run first (the fixture resets the env).

    Driven through **ROI mode on the AML runner**, the one `run_inference` shape where
    `storage="azure"` is actually allowed (`storage_allowed=(roi_mode and runner == "aml")`).
    Opus review, 2026-08-24: the first version of this test used the pre-built-cubes path,
    where the seam check *rejects* `storage="azure"` outright -- so it asserted that a call
    being refused for using blob storage had nonetheless flipped this process to
    authenticated adlfs, which is the global side effect D4 exists to stop, not the
    behavior D4 asks for. Storage is now configured only on a call the seam accepts."""
    os.environ.pop("FSSPEC_ABFSS_ANON", None)
    from rise_eo import api

    with pytest.raises(Exception):  # noqa: B017 - any failure proves nothing else ran first
        api.run_inference(
            "/nonexistent/model.bundle", roi="/nonexistent/roi.geojson",
            output_folderpath="/nonexistent/out", storage="azure", runner="aml",
        )
    assert os.environ["FSSPEC_ABFSS_ANON"] == "false"


def test_run_inference_rejects_a_bad_backend_as_preflight_not_a_bare_value_error(
    _clean_fsspec_conf,
):
    """Opus review, 2026-08-24. `_configure_storage` **raises** `ValueError` on an
    unsupported backend, so calling it ahead of `_raise_preflight` replaced this verb's
    collected `PreflightError` with a bare `ValueError` and threw away every other
    preflight error alongside it. The seam check now raises on its own first, matching
    `deploy`. Also pins the no-side-effect half: a refused call must not leave this
    process switched to authenticated adlfs."""
    os.environ.pop("FSSPEC_ABFSS_ANON", None)
    from rise_eo import api

    with pytest.raises(api.PreflightError, match="s3"):
        api.run_inference(
            "/nonexistent/model.bundle", roi="/nonexistent/roi.geojson",
            output_folderpath=None, storage="s3", runner="aml",
        )
    assert "FSSPEC_ABFSS_ANON" not in os.environ


def test_run_inference_refusing_azure_here_does_not_configure_storage(_clean_fsspec_conf):
    """The pre-built-cubes path refuses `storage="azure"` (inference-on-blob is gated to
    ROI+AML). Refusing it must also leave the process's fsspec state alone -- configuring
    adlfs as a side effect of a rejected call is exactly the accident D4 removes."""
    os.environ.pop("FSSPEC_ABFSS_ANON", None)
    from rise_eo import api

    with pytest.raises(api.PreflightError, match="not supported here yet"):
        api.run_inference(
            "/nonexistent/model.bundle", inference_datacubes="/nonexistent/cubes",
            output_folderpath="/nonexistent/out", storage="azure",
        )
    assert "FSSPEC_ABFSS_ANON" not in os.environ


def test_verify_adapter_calls_configure_storage_before_its_first_storage_access(
    _clean_fsspec_conf,
):
    """AC7, in a process where no other verb has run first (the fixture resets the env).
    `runner="aml"` because that is the shape where `verify_adapter` allows non-local
    storage at all (`storage_allowed=(runner == "aml")`) -- see the `run_inference`
    counterpart above for why a seam-rejected call must no longer configure storage."""
    os.environ.pop("FSSPEC_ABFSS_ANON", None)
    from rise_eo import api

    with pytest.raises(Exception):  # noqa: B017 - any failure proves nothing else ran first
        api.verify_adapter(
            "/nonexistent/model.bundle", roi="/nonexistent/roi.geojson",
            catalog_filepath="/nonexistent/catalog.parquet",
            startdate=None, enddate=None, mosaic_days=None, bands=None,
            export_folderpath="/nonexistent/export", storage="azure", runner="aml",
        )
    assert os.environ["FSSPEC_ABFSS_ANON"] == "false"


def test_verify_image_calls_configure_storage_before_its_first_storage_access(
    _clean_fsspec_conf,
):
    """AC7, in a process where no other verb has run first (the fixture resets the env).
    `bundle_path` does not exist, so `verify_image` catches the read failure internally
    (spec 24: always a shaped result, never a bare traceback) and returns `pass: False`
    rather than raising -- either way, `configure_storage` must already have run."""
    os.environ.pop("FSSPEC_ABFSS_ANON", None)
    from rise_eo.model.verify_image import verify_image

    result = verify_image(
        "/nonexistent/bundle", environment="rise-eo-infer-sklearn:1", runner="aml",
        runner_kwargs={"cluster": "c", "root": "r", "identity_client_id": "i"},
        storage="azure",
    )
    assert result["pass"] is False
    assert os.environ["FSSPEC_ABFSS_ANON"] == "false"


def test_api_check_local_seams_storage_allowed_false_rejects_azure():
    """`run_inference`'s pre-built-cubes / local-ROI paths pass storage_allowed=False --
    inference-on-blob is spec 38 P4 scope, gated separately from `deploy`'s registry-on-blob
    gate, which spec 52 D4 removed."""
    from rise_eo import api

    errs = api._check_local_seams("local", "azure", storage_allowed=False)
    assert errs and "not supported here yet" in errs[0]


# --- config-seam facts (pins the library behavior spec 31 §1 relies on) --------


def test_azureblobfilesystem_protocol_covers_all_three_schemes():
    from adlfs import AzureBlobFileSystem

    assert AzureBlobFileSystem.protocol == ("abfs", "az", "abfss")


def test_apply_config_one_key_covers_the_class():
    from adlfs import AzureBlobFileSystem
    from fsspec.config import apply_config

    out = apply_config(AzureBlobFileSystem, {}, {"abfss": {"anon": False}})
    assert out == {"anon": False}


def test_get_kwargs_from_urls_extracts_account():
    from adlfs import AzureBlobFileSystem

    kw = AzureBlobFileSystem._get_kwargs_from_urls(
        "abfss://data@acct.dfs.core.windows.net/p/x.tif"
    )
    assert kw == {"account_name": "acct"}


# --- storage_token ---------------------------------------------------------------


def test_storage_token_reuses_one_credential_instance(monkeypatch):
    constructed = []

    class _FakeCredential:
        def __init__(self):
            constructed.append(self)

        def get_token(self, scope):
            assert scope == "https://storage.azure.com/.default"
            return types.SimpleNamespace(token="fake-token-123")

    fake_identity_mod = types.ModuleType("azure.identity")
    fake_identity_mod.DefaultAzureCredential = _FakeCredential
    fake_azure_pkg = types.ModuleType("azure")
    monkeypatch.setitem(sys.modules, "azure", fake_azure_pkg)
    monkeypatch.setitem(sys.modules, "azure.identity", fake_identity_mod)
    monkeypatch.setattr(rise_eo_azure, "_credential", None)

    t1 = rise_eo_azure.storage_token()
    t2 = rise_eo_azure.storage_token()

    assert t1 == t2 == "fake-token-123"
    assert len(constructed) == 1  # one credential instance, reused


# --- rio_open routing --------------------------------------------------------------


def test_rio_open_local_path_is_plain_passthrough(monkeypatch):
    from rise_eo import raster

    calls = []

    def fake_open(path, mode="r", **kw):
        calls.append((path, mode, kw))
        return "SENTINEL_DATASET"

    monkeypatch.setattr(raster.rasterio, "open", fake_open)
    result = raster.rio_open("/local/a.tif")

    assert result == "SENTINEL_DATASET"
    assert calls == [("/local/a.tif", "r", {})]


def test_rio_open_per_handle_env_breaks_when_many_are_held_open(monkeypatch):
    """WHY `rio_env` exists (run-book 38 Phase 4, 2026-07-28). `rio_open` enters a
    `rasterio.Env` per handle. rasterio's env stack is LIFO and thread-local: the FIRST
    `Env.__enter__` records no parent, and its `__exit__` clears the stack. So holding N
    handles and closing them in creation order tears the root down first and the next close
    raises `EnvError: No GDAL environment exists`. Merging 300 blob COGs hit exactly this.
    This test pins the trap so nobody 'simplifies' `rio_env` back into a loop of `rio_open`.
    """
    import rasterio.errors

    from rise_eo import raster

    stack = []                       # stand-in for rasterio's thread-local env stack

    class _StackEnv:
        def __init__(self, **kw):
            self._root = False

        def __enter__(self):
            self._root = not stack
            stack.append(self)
            return self

        def __exit__(self, *a):
            if not stack:
                raise rasterio.errors.EnvError("No GDAL environment exists")
            stack.pop()
            if self._root:
                stack.clear()
            return False

    class _FakeDataset:
        def close(self):
            pass

    monkeypatch.setattr(raster.rasterio, "open", lambda *a, **k: _FakeDataset())
    monkeypatch.setattr(raster.rasterio, "Env", _StackEnv)
    monkeypatch.setattr(raster, "storage_token", lambda: "tok")

    urls = [f"abfss://data@acct.dfs.core.windows.net/p/{i}.tif" for i in range(3)]
    handles = [raster.rio_open(u) for u in urls]
    handles[0].close()                                  # root env goes first...
    with pytest.raises(rasterio.errors.EnvError, match="No GDAL environment exists"):
        handles[1].close()                              # ...and the next close blows up

    # `rio_env` is the fix: ONE env for all of them, so N datasets cost one enter/exit.
    stack.clear()
    with raster.rio_env(urls):
        assert len(stack) == 1
    assert stack == []


def test_rio_env_is_a_null_context_for_local_paths():
    from rise_eo import raster

    with raster.rio_env(["/local/a.tif", "/local/b.tif"]):
        pass            # no token fetch, no Env -- the zero-behaviour-change hinge


def test_rio_env_refuses_datasets_on_two_storage_accounts(monkeypatch):
    from rise_eo import raster

    monkeypatch.setattr(raster, "storage_token", lambda: "tok")
    with pytest.raises(ValueError, match="multiple storage accounts"):
        raster.rio_env([
            "abfss://data@acctA.dfs.core.windows.net/p/x.tif",
            "abfss://data@acctB.dfs.core.windows.net/p/y.tif",
        ])


def test_rio_open_remote_path_translates_and_uses_env(monkeypatch):
    from rise_eo import raster

    calls = []
    env_calls = []

    class _FakeDataset:
        def close(self):
            pass

    def fake_open(path, mode="r", **kw):
        calls.append((path, mode, kw))
        return _FakeDataset()

    class _FakeEnv:
        def __init__(self, **kw):
            env_calls.append(kw)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(raster.rasterio, "open", fake_open)
    monkeypatch.setattr(raster.rasterio, "Env", _FakeEnv)
    monkeypatch.setattr(raster, "storage_token", lambda: "tok-abc")

    src = raster.rio_open("abfss://data@acct.dfs.core.windows.net/p/x.tif")

    assert calls == [("/vsiadls/data/p/x.tif", "r", {})]
    assert env_calls == [
        {
            "AZURE_STORAGE_ACCESS_TOKEN": "tok-abc",
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.jp2",
            "AZURE_STORAGE_ACCOUNT": "acct",
        }
    ]
    src.close()


def test_rio_open_worker_thread_gets_its_own_env_not_the_drivers(monkeypatch):
    """AC4 (spec 57 D3) -- the regression guard for the #61 trap (spec 57 §8, §4.1): a worker
    thread entering an env sees ITS OWN token, and the driver thread's env is not relied on.
    rasterio's env stack is thread-local (`rasterio/env.py`, `local = ThreadEnv()`), so an Env
    entered on the driver thread does not exist in a worker thread. Exercised against REAL
    `rasterio.Env`/`hasenv`/`getenv` (not mocked) -- that thread-locality is rasterio's own
    behaviour, not rise-eo's, so mocking it would prove nothing."""
    import threading

    import rasterio.env

    from rise_eo import raster

    monkeypatch.setattr(raster, "storage_token", lambda: "worker-token")

    seen = {}
    driver_env = rasterio.Env(AZURE_STORAGE_ACCESS_TOKEN="driver-token")
    driver_env.__enter__()
    try:
        seen["driver_hasenv"] = rasterio.env.hasenv()

        def _worker():
            # what D3's wrong fix (one rio_env around the threaded loop) would rely on --
            # and what #61's "thread-local" fact actually means: this is False.
            seen["worker_hasenv_before_own_env"] = rasterio.env.hasenv()
            with rasterio.Env(AZURE_STORAGE_ACCESS_TOKEN=raster.storage_token()):
                seen["worker_token"] = rasterio.env.getenv()["AZURE_STORAGE_ACCESS_TOKEN"]

        t = threading.Thread(target=_worker)
        t.start()
        t.join()
    finally:
        driver_env.__exit__(None, None, None)

    assert seen["driver_hasenv"] is True
    assert seen["worker_hasenv_before_own_env"] is False
    assert seen["worker_token"] == "worker-token"  # the worker entered its OWN credentialed env


def test_rio_open_and_rio_env_disable_sidecar_probing_on_remote_opens(monkeypatch):
    """D5 (spec 57): every remote VSI open sets `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` +
    an extension whitelist, so GDAL stops listing the containing directory for sidecars on each
    open (gdal.org config docs, fetched 2026-08-27). Both `rio_open` (one dataset) and `rio_env`
    (N datasets) build from the same `_REMOTE_OPEN_CONFIG` so they cannot drift apart.

    `CPL_VSIL_CURL_ALLOWED_EXTENSIONS` is a WHITELIST -- a remote file whose extension is missing
    from it reads as non-existent -- so it must list EVERY extension the cube builder can hand to
    `rio_open` (review, 2026-08-27; `.tif` alone would have made a remote `.jp2` band file, i.e.
    any `cog=False` download, unopenable)."""
    from rise_eo import raster
    from rise_eo.datacube.builder import _RASTER_EXTS

    env_calls = []

    class _FakeEnv:
        def __init__(self, **kw):
            env_calls.append(kw)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _FakeDataset:
        def close(self):
            pass

    monkeypatch.setattr(raster.rasterio, "open", lambda *a, **k: _FakeDataset())
    monkeypatch.setattr(raster.rasterio, "Env", _FakeEnv)
    monkeypatch.setattr(raster, "storage_token", lambda: "tok")

    raster.rio_open("abfss://data@acct.dfs.core.windows.net/p/x.tif").close()
    with raster.rio_env(["abfss://data@acct.dfs.core.windows.net/p/y.tif"]):
        pass

    assert len(env_calls) == 2
    for kw in env_calls:
        assert kw["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
        allowed = kw["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"].split(",")
        assert set(_RASTER_EXTS).issubset(allowed), (
            f"the whitelist {allowed} drops {sorted(set(_RASTER_EXTS) - set(allowed))}, which "
            "makes those remote band files read as non-existent"
        )


def test_rio_open_write_mode_on_remote_raises():
    from rise_eo import raster

    with pytest.raises(ValueError):
        raster.rio_open("abfss://data@acct.dfs.core.windows.net/p/x.tif", mode="w")


# --- memory:// round-trip (proves scheme-routing without Azure) ------------------


def test_memory_scheme_roundtrip_parquet_and_npy():
    import geopandas as gpd
    import numpy as np
    import shapely.geometry as sg

    from rise_eo.storage import fs

    gdf = gpd.GeoDataFrame({"id": ["a"]}, geometry=[sg.Point(0, 0)], crs="EPSG:4326")
    fs.write_parquet("memory://cat/catalog.parquet", gdf)
    back = fs.read_parquet("memory://cat/catalog.parquet")
    assert list(back["id"]) == ["a"]

    arr = np.arange(6).reshape(2, 3)
    fs.save_npy("memory://cat/arr.npy", arr)
    out = fs.load_npy("memory://cat/arr.npy")
    assert (out == arr).all()


# --- os.path.join URL-safety (§2/§6, pinned) -------------------------------------


def test_os_path_join_is_url_safe_on_abfss():
    joined = os.path.join("abfss://fs@acct.dfs.core.windows.net/a", "b.tif")
    assert joined == "abfss://fs@acct.dfs.core.windows.net/a/b.tif"


# --- #145 P10: both backend parses reject the same shapes with the same text --------

@pytest.mark.parametrize(
    "storage, shown",
    [("s3", "'s3'"), ({"backend": "gcs"}, "'gcs'"), ({}, "None"), (123, "None")],
)
def test_unsupported_backend_text_in_both_places(_clean_fsspec_conf, storage, shown):
    from rise_eo import api

    line = f"storage backend {shown} not supported (only 'azure' in P1)."
    assert api._check_local_seams("local", storage) == [line]
    with pytest.raises(ValueError) as ei:
        rise_eo_azure.configure_storage(storage)
    assert str(ei.value) == line
