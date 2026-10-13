"""Spec 51 §9 step 0 — `fsd.model.registry`: layout, `publish`, `resolve`, `migrate`.

One test group per acceptance criterion in `specs/51-deploy-model-registry.md` §4 that step 0
owns (AC1-5, 11, 13; see `handoff-spec51-step0-registry.md`). Step 0 touches no verb, so these
tests call `registry.publish`/`resolve`/`migrate` directly rather than through `fsd.deploy`
(steps 1-3, not implemented yet).

Bundles are built with `bundle.save(joblib.Parallel, {}, ..., code=False)` — an installed class
needs no adapter module written to disk, so fixtures stay a couple of lines each. `joblib` is
already a dependency (used the same way in `tests/test_bundle_transparency.py`).
"""

from __future__ import annotations

import json
import os

import joblib
import pytest

from fsd.model import bundle, registry
from fsd.storage import fs


def _make_bundle(tmp_path, subdir="src_bundle", requirements=None) -> str:
    return bundle.save(
        joblib.Parallel, {}, str(tmp_path / subdir), code=False,
        requirements=requirements, verbose=False,
    )


def _rewrite_bundle_field(bundle_path: str, **fields) -> None:
    """Mutate bundle.json in place so two fixture bundles differ in content."""
    manifest_path = os.path.join(bundle_path, bundle.BUNDLE_MANIFEST)
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest.update(fields)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)


# --- AC1: publish + resolve round-trip ------------------------------------------------


def test_publish_creates_v1_and_resolve_finds_it(tmp_path):
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")

    version = registry.publish(src, "crop-rf", registry_root)

    assert version == 1
    v1 = registry.version_path(registry_root, "crop-rf", 1)
    assert fs.exists(os.path.join(v1, bundle.BUNDLE_MANIFEST))

    resolved = registry.resolve("crop-rf:1", registry_root)
    assert resolved == registry.Resolved("crop-rf", 1, v1)


# --- AC2: identical content twice is idempotent -----------------------------------------


def test_publish_identical_content_twice_is_idempotent(tmp_path):
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")

    v1 = registry.publish(src, "crop-rf", registry_root)
    before = fs.ls(os.path.join(registry_root, "crop-rf"))

    v2 = registry.publish(src, "crop-rf", registry_root)
    after = fs.ls(os.path.join(registry_root, "crop-rf"))

    assert v2 == v1
    assert len(after) == len(before)


# --- AC3: changed content creates v2; v1 untouched --------------------------------------


def test_publish_changed_content_creates_v2_and_leaves_v1_untouched(tmp_path):
    registry_root = str(tmp_path / "registry")
    src1 = _make_bundle(tmp_path, "src1")
    v1 = registry.publish(src1, "crop-rf", registry_root)
    assert v1 == 1

    src2 = _make_bundle(tmp_path, "src2")
    _rewrite_bundle_field(src2, feature={"kind": "callable", "steps": ["different"]})
    v2 = registry.publish(src2, "crop-rf", registry_root)
    assert v2 == 2

    v1_path = registry.version_path(registry_root, "crop-rf", 1)
    with fs.open(os.path.join(v1_path, bundle.BUNDLE_MANIFEST), "r") as f:
        v1_manifest = json.load(f)
    with open(os.path.join(src1, bundle.BUNDLE_MANIFEST)) as f:
        src1_manifest = json.load(f)
    assert v1_manifest == src1_manifest


# --- step 2 follow-through: publish's idempotency check reads a stored digest first ----


def test_publish_idempotency_uses_stored_deploy_digest_without_recomputing_content(
    tmp_path, monkeypatch,
):
    """Once a version carries `_deploy.json` (written by `deploy`, spec 51 D7), re-publishing
    identical content must not re-digest that version's bytes -- it reads the small stored
    digest instead (`registry.publish`'s own docstring, "known follow-through" in the step-2
    handoff)."""
    registry_root = str(tmp_path / "registry")
    src = _make_bundle(tmp_path)
    v1 = registry.publish(src, "crop-rf", registry_root)
    v1_path = registry.version_path(registry_root, "crop-rf", v1)
    digest = registry.content_digest(v1_path)
    registry.write_deploy_record(
        "crop-rf", v1, {"name": "crop-rf", "version": v1, "digest": digest}, registry_root,
    )

    def _forbidden(*a, **kw):
        raise AssertionError("content_digest recomputed despite a stored _deploy.json digest")

    monkeypatch.setattr(registry, "content_digest", _forbidden)

    again = registry.publish(src, "crop-rf", registry_root)

    assert again == v1


# --- step 2 follow-through: migrate carries a version's _deploy.json across ------------


def test_migrate_carries_the_deploy_record_across(tmp_path):
    src_root = str(tmp_path / "src_registry")
    dst_root = str(tmp_path / "dst_registry")
    src = _make_bundle(tmp_path)
    v1 = registry.publish(src, "crop-rf", src_root)
    record = {"name": "crop-rf", "version": v1, "digest": "sha256:deadbeef"}
    registry.write_deploy_record("crop-rf", v1, record, src_root)

    registry.migrate(src_root, dst_root)

    dst_version = registry.resolve("crop-rf:1", dst_root).path
    with open(os.path.join(dst_version, registry.DEPLOY_FILE)) as f:
        assert json.load(f) == record


# --- spec 52 AC2/AC3: a write interrupted before the marker lands is invisible + reusable


def test_publish_leaves_an_unmarked_invisible_version_if_interrupted_mid_write(
    tmp_path, monkeypatch,
):
    """A write that fails before `_complete.json` lands leaves a version `_list_versions`
    does not show, and the next `publish` reuses that number (D1, D2, AC2, AC3)."""
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"weights")
    src = bundle.save(
        joblib.Parallel, {"model": str(artifact)}, str(tmp_path / "src_bundle_2files"),
        code=False, verbose=False,
    )
    registry_root = str(tmp_path / "registry")

    real_write_bytes = fs.write_bytes
    failed = {"once": False}

    def flaky_write_bytes(path, data, **kw):
        if not failed["once"]:
            failed["once"] = True
            raise OSError("simulated failure on the first write")
        return real_write_bytes(path, data, **kw)

    monkeypatch.setattr(registry.fs, "write_bytes", flaky_write_bytes)

    with pytest.raises(OSError):
        registry.publish(src, "crop-rf", registry_root)

    v1 = registry.version_path(registry_root, "crop-rf", 1)
    assert not fs.exists(os.path.join(v1, registry._COMPLETE_FILE))
    assert registry._list_versions(os.path.join(registry_root, "crop-rf"), {}) == []

    # the next publish reuses v1 rather than skipping to v2
    version = registry.publish(src, "crop-rf", registry_root)
    assert version == 1
    assert fs.exists(os.path.join(v1, registry._COMPLETE_FILE))


def test_publish_interrupted_just_before_the_marker_leaves_an_invisible_version(
    tmp_path, monkeypatch,
):
    """Every content file landed, `_complete.json` did not: still invisible, and the next
    `publish` takes the same number (spec 52 X1.1, X1.3)."""
    src = _make_bundle(tmp_path)
    registry_root = str(tmp_path / "registry")
    name_root = os.path.join(registry_root, "crop-rf")

    real_write_text = fs.write_text
    failed = {"once": False}

    def flaky_write_text(path, text, **kw):
        if not failed["once"] and os.path.basename(path) == registry._COMPLETE_FILE:
            failed["once"] = True
            raise OSError("simulated failure writing the marker")
        return real_write_text(path, text, **kw)

    monkeypatch.setattr(registry.fs, "write_text", flaky_write_text)

    with pytest.raises(OSError):
        registry.publish(src, "crop-rf", registry_root)

    v1 = registry.version_path(registry_root, "crop-rf", 1)
    assert fs.exists(os.path.join(v1, bundle.BUNDLE_MANIFEST))
    assert not fs.exists(os.path.join(v1, registry._COMPLETE_FILE))
    assert registry._list_versions(name_root, {}) == []

    assert registry.publish(src, "crop-rf", registry_root) == 1
    assert fs.exists(os.path.join(v1, registry._COMPLETE_FILE))


# --- AC5: alias set on publish; re-deploy with same alias just repoints ----------------


def test_alias_write_and_repoint_touches_no_version(tmp_path):
    registry_root = str(tmp_path / "registry")
    src1 = _make_bundle(tmp_path, "src1")
    v1 = registry.publish(src1, "crop-rf", registry_root, alias="champion")

    aliases_path = os.path.join(registry_root, "crop-rf", registry.ALIASES_FILE)
    with fs.open(aliases_path, "r") as f:
        assert json.load(f) == {"champion": 1}

    src2 = _make_bundle(tmp_path, "src2")
    _rewrite_bundle_field(src2, feature={"kind": "callable", "steps": ["v2"]})
    v2 = registry.publish(src2, "crop-rf", registry_root, alias="champion")
    assert v2 == v1 + 1

    with fs.open(aliases_path, "r") as f:
        assert json.load(f) == {"champion": 2}

    # both versions are exactly what was published, untouched by the alias write
    for v, src in ((v1, src1), (v2, src2)):
        vpath = registry.version_path(registry_root, "crop-rf", v)
        with fs.open(os.path.join(vpath, bundle.BUNDLE_MANIFEST), "r") as f:
            got = json.load(f)
        with open(os.path.join(src, bundle.BUNDLE_MANIFEST)) as f:
            want = json.load(f)
        assert got == want

    resolved = registry.resolve("crop-rf@champion", registry_root)
    assert resolved.version == 2


def test_set_alias_rejects_a_vN_shaped_name(tmp_path):
    registry_root = str(tmp_path / "registry")
    src = _make_bundle(tmp_path)
    registry.publish(src, "crop-rf", registry_root)

    with pytest.raises(ValueError, match="shadowed"):
        registry.set_alias("crop-rf", "v7", 1, registry_root)


def test_bare_name_ref_is_rejected_naming_both_forms(tmp_path):
    with pytest.raises(ValueError, match="name:version.*name@alias|name@alias.*name:version"):
        registry.resolve("crop-rf", str(tmp_path / "registry"))


# --- AC11: alias resolution costs exactly one read, never lists ------------------------


def test_resolve_alias_costs_exactly_one_read_and_never_lists(tmp_path, monkeypatch):
    registry_root = str(tmp_path / "registry")
    src = _make_bundle(tmp_path)
    registry.publish(src, "crop-rf", registry_root, alias="champion")

    counts = {"open": 0, "ls": 0}
    real_open, real_ls = fs.open, fs.ls

    def counting_open(*a, **kw):
        counts["open"] += 1
        return real_open(*a, **kw)

    def counting_ls(*a, **kw):
        counts["ls"] += 1
        return real_ls(*a, **kw)

    monkeypatch.setattr(registry.fs, "open", counting_open)
    monkeypatch.setattr(registry.fs, "ls", counting_ls)

    resolved = registry.resolve("crop-rf@champion", registry_root)

    assert resolved.version == 1
    assert counts["open"] == 1
    assert counts["ls"] == 0


def test_resolve_version_pin_costs_no_reads_and_never_lists(tmp_path, monkeypatch):
    registry_root = str(tmp_path / "registry")
    src = _make_bundle(tmp_path)
    registry.publish(src, "crop-rf", registry_root)

    counts = {"open": 0, "ls": 0}
    real_open, real_ls = fs.open, fs.ls
    monkeypatch.setattr(registry.fs, "open", lambda *a, **kw: (counts.__setitem__("open", counts["open"] + 1), real_open(*a, **kw))[1])
    monkeypatch.setattr(registry.fs, "ls", lambda *a, **kw: (counts.__setitem__("ls", counts["ls"] + 1), real_ls(*a, **kw))[1])

    for ref in ("crop-rf:1", "crop-rf@v1"):
        resolved = registry.resolve(ref, registry_root)
        assert resolved.version == 1

    assert counts["open"] == 0
    assert counts["ls"] == 0


# --- AC13: relocatable (no self-reference) + migrate ------------------------------------


def test_no_written_file_contains_the_registry_root(tmp_path):
    registry_root = str(tmp_path / "registry")
    src1 = _make_bundle(tmp_path, "src1")
    src2 = _make_bundle(tmp_path, "src2")
    _rewrite_bundle_field(src2, feature={"kind": "callable", "steps": ["v2"]})
    registry.publish(src1, "crop-rf", registry_root, alias="champion")
    registry.publish(src2, "crop-rf", registry_root, alias="champion")

    root_str = str(tmp_path)
    for path, _size in fs.find_sizes(registry_root).items():
        if not (path.endswith(".json") or os.path.basename(path).startswith("_")):
            continue
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", errors="replace")
        assert root_str not in text
        assert "://" not in text


def test_migrate_copies_registry_and_refs_resolve_identically(tmp_path):
    src_root = str(tmp_path / "src_registry")
    dst_root = str(tmp_path / "dst_registry")

    b1 = _make_bundle(tmp_path, "b1")
    b2 = _make_bundle(tmp_path, "b2")
    _rewrite_bundle_field(b2, feature={"kind": "callable", "steps": ["v2"]})
    registry.publish(b1, "crop-rf", src_root, alias="champion")
    registry.publish(b2, "crop-rf", src_root, alias="latest")

    registry.migrate(src_root, dst_root)

    for ref in ("crop-rf:1", "crop-rf:2", "crop-rf@champion", "crop-rf@latest"):
        want = registry.resolve(ref, src_root)
        got = registry.resolve(ref, dst_root)
        assert got.version == want.version

        want_digest = registry.content_digest(want.path)
        got_digest = registry.content_digest(got.path)
        assert got_digest == want_digest


def test_migrate_marks_every_copied_version_complete(tmp_path):
    """`migrate` already does D1's write and re-digest steps, so each version it copies
    gets `_complete.json`: without it `_list_versions` would not show the copy."""
    src_root = str(tmp_path / "src_registry")
    dst_root = str(tmp_path / "dst_registry")
    b1 = _make_bundle(tmp_path, "b1")
    registry.publish(b1, "crop-rf", src_root)

    registry.migrate(src_root, dst_root)

    dst_v1 = registry.version_path(dst_root, "crop-rf", 1)
    assert fs.exists(os.path.join(dst_v1, registry._COMPLETE_FILE))
    with fs.open(os.path.join(dst_v1, registry._COMPLETE_FILE), "r") as f:
        marker = json.load(f)
    assert marker["digest"] == registry.content_digest(dst_v1)


def test_migrate_refuses_a_corrupted_copy(tmp_path, monkeypatch):
    src_root = str(tmp_path / "src_registry")
    dst_root = str(tmp_path / "dst_registry")
    src = _make_bundle(tmp_path)
    registry.publish(src, "crop-rf", src_root)

    real_write_bytes = fs.write_bytes

    def corrupting_write_bytes(path, data, **kw):
        if path.endswith(bundle.BUNDLE_MANIFEST) and "dst_registry" in path:
            manifest = json.loads(data.decode("utf-8"))
            manifest["corrupted"] = True
            data = json.dumps(manifest).encode("utf-8")
        return real_write_bytes(path, data, **kw)

    monkeypatch.setattr(registry.fs, "write_bytes", corrupting_write_bytes)

    with pytest.raises(ValueError, match="digest mismatch"):
        registry.migrate(src_root, dst_root)


# --- publish races: a loser must never be handed the winner's version ------------------


def test_publish_a_genuine_collision_retries_instead_of_returning_the_winners_version(
    tmp_path,
):
    """D3: `_write_new_version`'s target-exists check is not a lock, but once a target IS
    complete -- whether from a real race or (as here, for determinism) a competitor that
    simply finished first -- `publish` must not report it as ours. `v1` holds their bundle,
    and a caller that then resolved `crop-rf:1` would run a model it never published (D2,
    and §5's "fail rather than corrupt")."""
    registry_root = str(tmp_path / "registry")
    theirs = _make_bundle(tmp_path, "theirs")
    mine = _make_bundle(tmp_path, "mine")
    _rewrite_bundle_field(mine, feature={"kind": "callable", "steps": ["mine"]})

    registry.publish(theirs, "crop-rf", registry_root)  # claims v1, complete
    version = registry.publish(mine, "crop-rf", registry_root)

    assert version == 2
    mine_path = registry.version_path(registry_root, "crop-rf", version)
    assert registry.content_digest(mine_path) == registry.content_digest(mine)

    v1 = registry.version_path(registry_root, "crop-rf", 1)
    assert registry.content_digest(v1) == registry.content_digest(theirs)


def test_publish_a_competitors_identical_content_is_still_idempotent(tmp_path):
    """Same collision, but the competitor published the same bytes -- that is D2's
    idempotency arriving by another route, so their version is the right answer to
    return, and nothing new is written."""
    registry_root = str(tmp_path / "registry")
    src = _make_bundle(tmp_path, "src")

    registry.publish(src, "crop-rf", registry_root)  # claims v1, complete
    version = registry.publish(src, "crop-rf", registry_root)

    assert version == 1
    assert fs.ls(os.path.join(registry_root, "crop-rf")) == [
        registry.version_path(registry_root, "crop-rf", 1)
    ]


# --- alias writes are never observed half-written --------------------------------------


def test_set_alias_publishes_by_rename_so_a_reader_never_sees_a_partial_file(
    tmp_path, monkeypatch,
):
    """`resolve` reads `_aliases.json` and nothing else (D9), so promoting against a live
    fan-out must leave no window where that read sees a half-written file: the new content
    is staged and renamed onto the name, never written over it in place."""
    registry_root = str(tmp_path / "registry")
    src = _make_bundle(tmp_path)
    registry.publish(src, "crop-rf", registry_root)
    registry.set_alias("crop-rf", "champion", 1, registry_root)

    aliases_path = os.path.join(registry_root, "crop-rf", registry.ALIASES_FILE)
    written = []
    real_write_bytes = fs.write_bytes

    def spy_write_bytes(path, data, **kw):
        written.append(path)
        return real_write_bytes(path, data, **kw)

    monkeypatch.setattr(registry.fs, "write_bytes", spy_write_bytes)
    registry.set_alias("crop-rf", "current", 1, registry_root)
    monkeypatch.undo()

    assert aliases_path not in written
    assert any(registry._STAGING_PREFIX in p for p in written)
    with fs.open(aliases_path, "r") as f:
        assert json.load(f) == {"champion": 1, "current": 1}
    assert not any(
        registry._STAGING_PREFIX in entry
        for entry in fs.ls(os.path.join(registry_root, "crop-rf"))
    )


# --- AC12: a URL registry behaves like a local one --------------------------------------


def test_publish_resolve_round_trip_against_a_url_registry(tmp_path):
    """AC12: behaviour is identical for a local registry path and a URL registry. Unblocked
    by spec 52 D1 (#88): `_write_new_version` no longer renames a directory onto `v<N>`, so
    there is nothing left that hangs against a non-local backend. See also
    `test_deploy.py::test_deploy_set_alias_resolve_run_inference_against_a_url_registry`
    (spec 52 AC8) for the full `deploy` -> `set_alias` -> `resolve` -> `run_inference` chain,
    not just `publish`/`resolve`."""
    src = _make_bundle(tmp_path)
    registry_root = "memory://ac12-registry"
    v = registry.publish(src, "crop-rf", registry_root, alias="champion")
    assert registry.resolve("crop-rf@champion", registry_root).version == v
    assert registry.resolve("crop-rf:1", registry_root).path == registry.version_path(
        registry_root, "crop-rf", 1,
    )


# --- AC1: publish against a non-local backend returns in seconds, not forever ----------


def test_publish_against_a_url_registry_returns_within_a_timeout(tmp_path):
    """AC1's literal wording: a regression to #88 (the infinite retry loop against a
    non-local backend) must fail this test, not hang the suite. `registry.publish` runs on
    a background thread so the test can assert on wall-clock time rather than trust that the
    call returns at all -- `thread.is_alive()` after the join is the failure signal for a
    hang; a raised exception inside the thread is re-raised here so a real bug still reports
    normally instead of just timing out."""
    import threading

    src = _make_bundle(tmp_path)
    registry_root = "memory://ac1-timeout-registry"
    result: dict = {}

    def _publish():
        try:
            result["version"] = registry.publish(src, "crop-rf", registry_root)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread below
            result["error"] = exc

    thread = threading.Thread(target=_publish, daemon=True)
    thread.start()
    thread.join(timeout=10)

    assert not thread.is_alive(), (
        "registry.publish did not return within 10s against memory:// -- this is #88's "
        "infinite retry loop, back."
    )
    if "error" in result:
        raise result["error"]
    assert result["version"] == 1


# --- spec 52 review (Opus, 2026-08-24): exit paths of `_write_new_version` ---------------
#
# The tests below pin two of its exit paths, the idempotent-collision `return` and the
# landed-digest guard, and the marker-only completeness rule. The two "race" tests written
# during implementation look like they cover the collision path but cannot reach it -- their
# competitor publishes *before* `_list_versions` runs, so allocation starts past the
# competitor's version and the loop body never sees an existing target. Reaching it needs the
# listing to be stale, which is what the race actually is and what `monkeypatch` supplies.


def test_write_new_version_returns_a_complete_collision_holding_identical_content(
    tmp_path, monkeypatch,
):
    """D2/D3, path (a): we allocated `v1` off a listing taken before a competitor finished
    `v1` with the *same* bytes. Their version is the right answer -- return it, write
    nothing. Unreachable through `publish` (its own idempotency scan returns first), so the
    stale listing is simulated directly."""
    registry_root = str(tmp_path / "registry")
    name_root = os.path.join(registry_root, "crop-rf")
    src = _make_bundle(tmp_path, "src")
    registry.publish(src, "crop-rf", registry_root)  # competitor completes v1

    files = registry._read_bundle_content(src, None)
    digest = registry._digest_of(files)
    monkeypatch.setattr(registry, "_list_versions", lambda *a, **k: [])  # our stale listing

    assert registry._write_new_version(name_root, files, digest, {}) == 1
    assert fs.ls(os.path.join(registry_root, "crop-rf")) == [
        os.path.join(registry_root, "crop-rf", "v1"),
    ]


def test_write_new_version_skips_a_complete_collision_holding_different_content(
    tmp_path, monkeypatch,
):
    """D2/D3, path (b): same stale listing, but the competitor's `v1` holds *their* bundle.
    Returning 1 would hand the caller a version they never published, so the loop must
    advance to `v2` -- and must leave `v1`'s bytes untouched on the way past."""
    registry_root = str(tmp_path / "registry")
    name_root = os.path.join(registry_root, "crop-rf")
    theirs = _make_bundle(tmp_path, "theirs")
    mine = _make_bundle(tmp_path, "mine")
    _rewrite_bundle_field(mine, feature={"kind": "callable", "steps": ["mine"]})
    registry.publish(theirs, "crop-rf", registry_root)  # competitor completes v1

    files = registry._read_bundle_content(mine, None)
    digest = registry._digest_of(files)
    monkeypatch.setattr(registry, "_list_versions", lambda *a, **k: [])

    assert registry._write_new_version(name_root, files, digest, {}) == 2
    v1 = registry.version_path(registry_root, "crop-rf", 1)
    v2 = registry.version_path(registry_root, "crop-rf", 2)
    assert registry.content_digest(v1) == registry.content_digest(theirs)
    assert registry.content_digest(v2) == digest


def test_publish_moves_on_when_the_landed_bytes_are_not_the_callers(tmp_path, monkeypatch):
    """D1 step 2 / spec 52 §5: two publishers interleaved their files at the same `v<N>`, so
    the re-digest of what landed does not match what we were asked to publish. Claiming that
    directory would serve a model nobody published, so the loop leaves it unmarked and moves
    to `v2`. `v1` has no `_complete.json`, so it is invisible to `_list_versions`."""
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"weights")
    src = bundle.save(
        joblib.Parallel, {"model": str(artifact)}, str(tmp_path / "src_bundle"),
        code=False, verbose=False,
    )
    registry_root = str(tmp_path / "registry")

    real_write_bytes = fs.write_bytes
    tampered = {"once": False}

    def tampering_write_bytes(path, data, **kw):
        # a competitor's bytes land at the first non-manifest file
        if not tampered["once"] and os.path.basename(path) != bundle.BUNDLE_MANIFEST:
            tampered["once"] = True
            return real_write_bytes(path, b"someone else's weights", **kw)
        return real_write_bytes(path, data, **kw)

    monkeypatch.setattr(registry.fs, "write_bytes", tampering_write_bytes)
    version = registry.publish(src, "crop-rf", registry_root)

    assert version == 2
    v1 = registry.version_path(registry_root, "crop-rf", 1)
    v2 = registry.version_path(registry_root, "crop-rf", 2)
    assert registry.content_digest(v2) == registry.content_digest(src)
    assert not fs.exists(os.path.join(v1, registry._COMPLETE_FILE))
    assert registry._list_versions(os.path.join(registry_root, "crop-rf"), {}) == [2]


def test_a_version_with_no_marker_is_invisible_and_is_replaced_by_the_next_publish(tmp_path):
    """Spec 52 X1.1, X1.3: `bundle.json` alone does not make a version. An unmarked `v1` is
    not listed, still resolves by pin (D2), and the next `publish` of different content
    clears it and takes number 1."""
    registry_root = str(tmp_path / "registry")
    name_root = os.path.join(registry_root, "crop-rf")
    old = _make_bundle(tmp_path, "old")
    registry.publish(old, "crop-rf", registry_root)

    v1 = registry.version_path(registry_root, "crop-rf", 1)
    fs.rm(os.path.join(v1, registry._COMPLETE_FILE))

    assert registry._list_versions(name_root, {}) == []
    assert registry.resolve("crop-rf:1", registry_root).path == v1

    nextone = _make_bundle(tmp_path, "nextone")
    _rewrite_bundle_field(nextone, feature={"kind": "callable", "steps": ["next"]})
    assert registry.publish(nextone, "crop-rf", registry_root) == 1
    assert registry.content_digest(v1) == registry.content_digest(nextone)
    assert fs.exists(os.path.join(v1, registry._COMPLETE_FILE))


def test_publish_into_an_incomplete_version_clears_the_previous_attempts_leftovers(tmp_path):
    """Opus review, 2026-08-24. AC2 has the next `publish` reuse an interrupted version *in
    place*, and the re-digest cannot police what it inherits: `content_digest` covers only
    manifest-declared files, so an artifact or a `code/*.py` the new bundle does not declare
    would survive into the version and then be marked complete. `bundle.load` puts a
    version's `code/` on `sys.path`, so a stale module there is importable by whichever
    adapter lands next -- a quiet, wrong-model failure the old stage-then-rename could not
    produce, because every attempt got a fresh staging prefix."""
    registry_root = str(tmp_path / "registry")
    name_root = os.path.join(registry_root, "crop-rf")
    v1 = os.path.join(name_root, "v1")
    # what an interrupted publish leaves: files, but no manifest and no marker
    fs.write_bytes(os.path.join(v1, "stale.bin"), b"a previous attempt's weights")
    fs.write_text(os.path.join(v1, "code", "stale_adapter.py"), "raise RuntimeError('stale')")
    assert registry._list_versions(name_root, {}) == []

    src = _make_bundle(tmp_path, "fresh")
    assert registry.publish(src, "crop-rf", registry_root) == 1

    assert not fs.exists(os.path.join(v1, "stale.bin"))
    assert not fs.exists(os.path.join(v1, "code", "stale_adapter.py"))
    assert registry.content_digest(v1) == registry.content_digest(src)
    assert fs.exists(os.path.join(v1, registry._COMPLETE_FILE))
