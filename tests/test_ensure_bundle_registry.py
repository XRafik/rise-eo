"""Spec 51 §9 step 1 — `api._ensure_bundle` registry resolution (D4, AC6).

`_ensure_bundle` already turns "whatever the caller passed" into a bundle path for
`run_inference`/`verify_adapter`/the `cores>1` fan-out; this covers the one new branch it grows:
resolving `"name:N"` / `"name@alias"` against `registry=`.

Bundles are built with `bundle.save(joblib.Parallel, {}, ..., code=False)`, the same fixture
`tests/test_registry.py` uses — an installed class needs no adapter module written to disk.
"""

from __future__ import annotations

import joblib
import pytest

from rise_eo import api
from rise_eo.model import bundle, registry


def _make_bundle(tmp_path, subdir="src_bundle") -> str:
    return bundle.save(joblib.Parallel, {}, str(tmp_path / subdir), code=False, verbose=False)


# --- registry= given: "name:N" and "name@alias" both resolve ---------------------------


def test_ensure_bundle_resolves_version_pin_when_registry_given(tmp_path):
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    registry.publish(src, "crop-rf", registry_root)

    resolved_path = api._ensure_bundle(
        "crop-rf:1", str(tmp_path / "out"), why="test", registry=registry_root,
    )

    assert resolved_path == registry.version_path(registry_root, "crop-rf", 1)


def test_ensure_bundle_resolves_alias_when_registry_given(tmp_path):
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    registry.publish(src, "crop-rf", registry_root, alias="champion")

    resolved_path = api._ensure_bundle(
        "crop-rf@champion", str(tmp_path / "out"), why="test", registry=registry_root,
    )

    assert resolved_path == registry.version_path(registry_root, "crop-rf", 1)


# --- registry= absent: a ref-shaped "@" string is a PreflightError, never a silent path ---


def test_ensure_bundle_refuses_alias_ref_without_registry_naming_the_missing_argument():
    with pytest.raises(api.PreflightError, match="registry="):
        api._ensure_bundle("crop-rf@champion", "/tmp/out", why="test")


def test_ensure_bundle_refuses_alias_ref_without_registry_even_when_live_adapter_absent():
    # a live adapter passed alongside registry= (nonsensical combination) is refused too,
    # naming the same confusion rather than silently ignoring registry=.
    with pytest.raises(api.PreflightError, match="registry="):
        api._ensure_bundle(joblib.Parallel, "/tmp/out", why="test", registry="/tmp/reg")


# --- registry= absent: everything else is unchanged (D4's existing rows) ---------------


def test_ensure_bundle_treats_version_pin_string_as_a_literal_path_without_registry():
    # "crop-rf:1" has no "@", so it is passed through unchanged -- ":" collides with URL
    # schemes and Windows drive letters, so it is never sniffed (spec 51 step-1 handoff).
    assert api._ensure_bundle("crop-rf:1", "/tmp/out", why="test") == "crop-rf:1"


def test_ensure_bundle_passes_through_a_plain_bundle_path_without_registry(tmp_path):
    src = _make_bundle(tmp_path)
    assert api._ensure_bundle(src, str(tmp_path / "out"), why="test") == src


def test_ensure_bundle_passes_through_an_abfss_url_containing_at_sign_without_registry():
    # abfss URLs legitimately embed "@" (<fs>@<account>.dfs.core.windows.net/<path>,
    # storage/azure.py's _ABFSS_RE) -- must not be mistaken for a "name@alias" ref.
    url = "abfss://models@myaccount.dfs.core.windows.net/crop-rf/v1"
    assert api._ensure_bundle(url, "/tmp/out", why="test") == url


@pytest.mark.parametrize("path", [
    "/data/bundles/rf@2026-08/bundle",   # local path whose directory name contains "@"
    "s3://bucket@weird/crop-rf",         # any URL: the "//" alone settles it
    "relative/rf@v1",                    # relative path, "@" in the last component
])
def test_ensure_bundle_passes_through_a_path_containing_at_sign_without_registry(path):
    # only a SEPARATOR-FREE "name@alias" is ref-shaped; anything with a path component is a
    # path, so AC6's error can never refuse a legitimate bundle location.
    assert api._ensure_bundle(path, "/tmp/out", why="test") == path


# --- registry= given but the ref is bad: a PreflightError naming the verb, not a raw ValueError ---


def test_ensure_bundle_leaves_an_already_resolved_path_alone_when_registry_is_given(tmp_path):
    # `_resolve_model_ref` runs at BOTH `_model_spec`'s call sites and `_ensure_bundle`, so the
    # second call sees a path, not a ref -- it must pass through, not fail to "resolve".
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    registry.publish(src, "crop-rf", registry_root)
    resolved = registry.version_path(registry_root, "crop-rf", 1)

    assert api._ensure_bundle(
        resolved, str(tmp_path / "out"), why="test", registry=registry_root,
    ) == resolved


def test_run_inference_resolves_a_ref_before_reading_the_model_spec(tmp_path):
    # AC1: `run_inference(model=ref)` accepts a ref unchanged. `_model_spec` reads `bundle.json`
    # off `model` BEFORE any dispatch, so an unresolved ref died there with a FileNotFoundError
    # naming `<ref>/bundle.json`. Reaching the *datacube* preflight is the proof it resolved.
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    registry.publish(src, "crop-rf", registry_root, alias="champion")
    cubes = tmp_path / "cubes"
    cubes.mkdir()

    with pytest.raises(api.PreflightError, match="no inference datacubes"):
        api.run_inference(
            model="crop-rf@champion",
            registry=registry_root,
            inference_datacubes=str(cubes),
            output_folderpath=str(tmp_path / "out"),
        )


def test_verify_adapter_resolves_a_ref_before_reading_the_model_spec(tmp_path):
    # same for the other `_model_spec` call site: the error must be about the CALL (a missing
    # export_folderpath), never about `crop-rf@champion/bundle.json`.
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    registry.publish(src, "crop-rf", registry_root, alias="champion")

    with pytest.raises(api.PreflightError, match="export_folderpath"):
        api.verify_adapter(
            model="crop-rf@champion",
            registry=registry_root,
            roi=str(tmp_path / "roi.geojson"),
            catalog_filepath=str(tmp_path / "catalog.parquet"),
            startdate="2018-04-01", enddate="2018-09-01",
            mosaic_days=20, bands=["B04", "B08", "SCL"],
            export_folderpath="",
        )


def test_ensure_bundle_wraps_an_unresolvable_ref_in_a_preflight_error(tmp_path):
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    registry.publish(src, "crop-rf", registry_root)

    with pytest.raises(api.PreflightError, match="cannot resolve"):
        api._ensure_bundle(
            "crop-rf@nosuchalias", str(tmp_path / "out"), why="test", registry=registry_root,
        )


# --- spec 51 step 3: the `[model]` print line (D7's print half, AC10's print half) -----


def test_run_inference_prints_the_model_line_exactly_once_with_environment(
    tmp_path, capsys, monkeypatch,
):
    """The double-print trap: `run_inference` hits `_resolve_model_ref` at BOTH its own call
    site and `_ensure_bundle`'s (via the `cores>1` fan-out). Only the first sees a ref; the
    second sees an already-resolved version path, which is not ref-shaped, so it returns early
    and the line appears once.

    This test **asserts that the second call site actually ran** (`seen`), because the obvious
    way to write it -- an empty `inference_datacubes` folder -- dies at `_raise_preflight`
    *before* `_ensure_bundle` is ever reached, and would pass with only one call site executed.
    So: a real datacube, `cores=2` to force the runner path, and a stubbed runner/finalizer so
    no snakemake or STAC work happens.
    """
    import types

    import numpy as np

    from rise_eo.storage import fs
    from rise_eo.workflows import runners as _runners

    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    v1 = registry.publish(src, "crop-rf", registry_root, alias="champion")
    registry.write_deploy_record(
        "crop-rf", v1,
        {"name": "crop-rf", "version": v1, "environment": "rise-eo-infer-sklearn:6"},
        registry_root,
    )

    cell = tmp_path / "cubes" / "cell_a"
    cell.mkdir(parents=True)
    fs.save_npy(str(cell / "datacube.npy"), np.zeros((1, 2, 2, 2), dtype="float32"))
    fs.save_npy(str(cell / "metadata.pickle.npy"),
                {"timestamps": [0, 1], "bands": ["B04", "B08"]}, allow_pickle=True)

    seen = []
    real_ensure_bundle = api._ensure_bundle

    def spy_ensure_bundle(model, *a, **kw):
        seen.append(model)
        return real_ensure_bundle(model, *a, **kw)

    monkeypatch.setattr(api, "_ensure_bundle", spy_ensure_bundle)
    monkeypatch.setattr(_runners, "run_local_infer_only",
                        lambda *a, **kw: types.SimpleNamespace(returncode=0))
    monkeypatch.setattr(api, "_finalize_outputs", lambda *a, **kw: [])

    api.run_inference(
        model="crop-rf@champion",
        registry=registry_root,
        inference_datacubes=str(tmp_path / "cubes"),
        output_folderpath=str(tmp_path / "out"),
        cores=2,
    )

    # the second call site genuinely ran, and it received an already-resolved path
    assert seen == [registry.version_path(registry_root, "crop-rf", 1)]

    out = capsys.readouterr().out
    line = "[model] crop-rf@champion -> v1 (verified against rise-eo-infer-sklearn:6)"
    assert out.count(line) == 1
    assert out.count("[model]") == 1


def test_ensure_bundle_prints_the_shorter_line_when_deploy_json_is_absent(tmp_path, capsys):
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    registry.publish(src, "crop-rf", registry_root, alias="champion")

    api._ensure_bundle(
        "crop-rf@champion", str(tmp_path / "out"), why="test", registry=registry_root,
    )

    out = capsys.readouterr().out
    assert "[model] crop-rf@champion -> v1\n" in out
    assert "verified against" not in out


def test_ensure_bundle_prints_the_shorter_line_when_deploy_json_has_no_environment(
    tmp_path, capsys,
):
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    v1 = registry.publish(src, "crop-rf", registry_root, alias="champion")
    registry.write_deploy_record(
        "crop-rf", v1, {"name": "crop-rf", "version": v1}, registry_root,
    )

    api._ensure_bundle(
        "crop-rf@champion", str(tmp_path / "out"), why="test", registry=registry_root,
    )

    out = capsys.readouterr().out
    assert "[model] crop-rf@champion -> v1\n" in out
    assert "verified against" not in out


def test_ensure_bundle_prints_the_shorter_line_when_deploy_json_is_malformed(tmp_path, capsys):
    from rise_eo.storage import fs

    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    v1 = registry.publish(src, "crop-rf", registry_root, alias="champion")
    deploy_path = f"{registry.version_path(registry_root, 'crop-rf', v1)}/_deploy.json"
    fs.write_text(deploy_path, "{not valid json")

    resolved = api._ensure_bundle(
        "crop-rf@champion", str(tmp_path / "out"), why="test", registry=registry_root,
    )

    assert resolved == registry.version_path(registry_root, "crop-rf", 1)
    out = capsys.readouterr().out
    assert "[model] crop-rf@champion -> v1\n" in out
    assert "verified against" not in out


def test_ensure_bundle_survives_a_byte_corrupt_deploy_record(tmp_path, capsys):
    # "must never raise" has to cover a record that is not valid UTF-8, not just one that is
    # not valid JSON: a truncated or byte-corrupted `_deploy.json` raises `UnicodeDecodeError`
    # (a `ValueError`, NOT a `json.JSONDecodeError`) out of `json.load`. Catching only
    # `JSONDecodeError`/`OSError` let it escape and kill a run whose model had already resolved.
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    v1 = registry.publish(src, "crop-rf", registry_root, alias="champion")
    deploy_path = f"{registry.version_path(registry_root, 'crop-rf', v1)}/_deploy.json"
    with open(deploy_path, "wb") as f:
        f.write(b'{"environment": "img\xff\xfe:6"}')

    resolved = api._ensure_bundle(
        "crop-rf@champion", str(tmp_path / "out"), why="test", registry=registry_root,
    )

    assert resolved == registry.version_path(registry_root, "crop-rf", 1)
    out = capsys.readouterr().out
    assert "[model] crop-rf@champion -> v1\n" in out
    assert "verified against" not in out


def test_read_deploy_record_returns_none_for_every_unreadable_shape(tmp_path):
    # the contract in one place, at the function itself: absent, empty, non-JSON, non-UTF-8,
    # and a valid-JSON-but-not-a-dict payload all degrade to None, none of them raise.
    vdir = tmp_path / "v1"
    vdir.mkdir()
    deploy_path = vdir / "_deploy.json"

    assert registry.read_deploy_record(str(vdir)) is None       # absent
    deploy_path.write_text("")
    assert registry.read_deploy_record(str(vdir)) is None       # empty
    deploy_path.write_text("{not valid json")
    assert registry.read_deploy_record(str(vdir)) is None       # non-JSON
    deploy_path.write_bytes(b"\xff\xfe\x00binary")
    assert registry.read_deploy_record(str(vdir)) is None       # non-UTF-8
    deploy_path.write_text('["a", "list"]')
    assert registry.read_deploy_record(str(vdir)) is None       # JSON, but not an object
    deploy_path.write_text('{"environment": "img:6"}')
    assert registry.read_deploy_record(str(vdir)) == {"environment": "img:6"}


def test_ensure_bundle_prints_nothing_for_a_plain_bundle_path(tmp_path, capsys):
    src = _make_bundle(tmp_path)
    api._ensure_bundle(src, str(tmp_path / "out"), why="test")
    out = capsys.readouterr().out
    assert "[model]" not in out


def test_verify_adapter_gets_the_model_line_too(tmp_path, capsys):
    # D7's print rule is not verb-specific (spec 47 D5): `verify_adapter` resolves refs too,
    # and it hits the same resolve-succeeded branch.
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    registry.publish(src, "crop-rf", registry_root, alias="champion")

    with pytest.raises(api.PreflightError, match="export_folderpath"):
        api.verify_adapter(
            model="crop-rf@champion",
            registry=registry_root,
            roi=str(tmp_path / "roi.geojson"),
            catalog_filepath=str(tmp_path / "catalog.parquet"),
            startdate="2018-04-01", enddate="2018-09-01",
            mosaic_days=20, bands=["B04", "B08", "SCL"],
            export_folderpath="",
        )

    out = capsys.readouterr().out
    assert "[model] crop-rf@champion -> v1\n" in out
