"""Guards for the notebooks this repo tracks: the demo notebooks, and every run-book.

`notebooks/build_images.ipynb` is the how-to for building the two AML node images, and
is deliberately public (`.gitignore` un-ignores it explicitly). Every other notebook stays
ignored precisely because notebooks leak: a saved output carries whatever the cell printed,
and cloud tooling prints subscription ids, tenant ids, workspace URLs and home directories
without being asked.

So the exception needs teeth. These tests are the reason the file can be tracked at all:

  * no saved outputs and no execution counts -- an executed notebook must be cleared before
    it is committed (Kernel > Restart & Clear All Outputs);
  * no identifiers in the source -- the private values come from
    `~/.config/rise-eo/config.toml` (spec 54) at run time, never from the file itself.

**Blob paths are the one exception (user, 2026-09-29).** A storage account, container or
`abfss://` path may be written into a notebook directly: it grants nothing without a
credential, and routing it through an environment variable was judged clunky. Everything
else above stays forbidden -- GUIDs (subscription / tenant / client ids), email addresses,
home directories, resource-group / workspace / cluster names.

**Run-books are notebooks too (spec 24 A1).** Every `runbooks/*.ipynb` is tracked (only
`notebooks/*.ipynb` is gitignored), so each one gets the same two guards by glob -- a new
run-book is covered the moment it exists. Two more rules are theirs alone (A1.5): the notebook
names its kernel, and it checks which `rise_eo` it imported.

Synthetic and offline: this reads the checked-in JSON, it never runs a cell.
"""

# Spec 102 A2.6: CI skips the rest of the suite on a docs-only PR but always runs this file. So a test
# that reads a Markdown file belongs in tests/test_docs.py or tests/test_notebooks.py, nowhere else.

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = REPO_ROOT / "notebooks"
RUNBOOKS = REPO_ROOT / "runbooks"

# Every notebook `.gitignore` explicitly un-ignores. Add a name here in the same commit
# that un-ignores it, or it goes public unguarded.
TRACKED_NOTEBOOKS = ["build_images.ipynb", "e2e_austria_aml.ipynb"]

# Every run-book notebook, found by glob rather than listed (spec 24 A1.9).
RUNBOOK_NOTEBOOKS = sorted(p.name for p in RUNBOOKS.glob("*.ipynb"))

# (folder, name) for every notebook the two leak guards cover.
GUARDED = ([("notebooks", n) for n in TRACKED_NOTEBOOKS]
           + [("runbooks", n) for n in RUNBOOK_NOTEBOOKS])


def _path(name, folder="notebooks") -> Path:
    return REPO_ROOT / folder / name


def _cells(name, folder="notebooks"):
    return json.loads(_path(name, folder).read_text())["cells"]


def _source(name, folder="notebooks") -> str:
    return "\n".join("".join(c["source"]) for c in _cells(name, folder))


def _whole_file(name, folder="notebooks") -> str:
    """Source AND outputs, as raw JSON.

    The identifier scan deliberately covers both. Outputs are already banned outright by
    `test_no_saved_outputs`, but that makes the two tests overlap rather than depend on
    each other: if the outputs rule is ever relaxed, an identifier still cannot ride in.
    """
    return _path(name, folder).read_text()


@pytest.mark.parametrize("name", TRACKED_NOTEBOOKS)
def test_the_tracked_notebook_exists(name):
    """If this fails the file was renamed or re-ignored — fix the guard with it, don't
    delete it, or the exception in `.gitignore` silently stops being enforced."""
    assert (NOTEBOOKS / name).exists(), f"{NOTEBOOKS / name} is missing"


@pytest.mark.parametrize("folder, name", GUARDED)
def test_no_saved_outputs(folder, name):
    """An executed notebook must be cleared before commit.

    This is the leak that matters most: the cell that prints a Studio URL embeds the
    workspace's full ARM id — subscription included — in its output, and nothing about
    the source would tell you.
    """
    dirty = [
        i for i, c in enumerate(_cells(name, folder))
        if c.get("cell_type") == "code"
        and (c.get("outputs") or c.get("execution_count") is not None)
    ]
    assert not dirty, (
        f"{folder}/{name}: cells {dirty} carry saved outputs or execution counts. Run "
        "Kernel > Restart & Clear All Outputs, then re-commit."
    )


# Each pattern is a thing cloud tooling prints that must never be committed. Named
# rather than lumped together so a failure says which class of identifier leaked.
_FORBIDDEN = {
    "an Azure GUID (subscription / tenant / client id)":
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    # The lookbehind skips a `container@account` that follows `abfss://` -- a blob path, which
    # is allowed (see the module docstring), not an address.
    "an email address": r"(?<![\w.+/-])[\w.+-]+@[\w-]+\.[\w.]{2,}\b",
    "a local home directory": r"/(?:Users|home)/[a-zA-Z0-9._-]+",
    # No storage-account pattern, on purpose: blob paths may be committed (user, 2026-09-29).
    "a concrete resource group or workspace name": r"\brg-[a-z0-9-]+|\bmlw-[a-z0-9-]+",
    "a concrete compute cluster name": r"\bcluster-[a-z0-9-]+",
}


@pytest.mark.parametrize("folder, name", GUARDED)
@pytest.mark.parametrize("what, pattern", sorted(_FORBIDDEN.items()))
def test_file_carries_no_identifiers(what, pattern, folder, name):
    """The notebook's private values come from `~/.config/rise-eo/config.toml` at run time.

    Anything matching here has been baked into a public file — in a cell, or in an output.
    """
    hits = sorted(set(re.findall(pattern, _whole_file(name, folder))))
    assert not hits, (
        f"{folder}/{name} hardcodes {what}: {hits}. "
        "Read it through rise_eo.config.load() instead."
    )


# --- Spec 102 D13: the same guard over the runnable and living files outside the notebooks ------
#
# Point-in-time records (specs, ADRs, run-books, `docs/*archive*`, `docs/findings/`) truthfully
# say where a run happened and stay as written. Everything below is either run by a contributor
# or read as current, so it must hold no home path, GUID or concrete resource-group/workspace name.
# The email and cluster-name patterns are left out here: this scope is mostly prose, and they flag
# `fs@account.dfs...` (a URL shape) and the word "cluster-loadable".
_SCOPE_GLOBS = (
    "benchmarks/**/*", "demos/**/*", "scripts/**/*", "src/**/*", "docs/howto/**/*",
    "docs/reference/**/*",
)
_SCOPE_FILES = (
    "README.md", "CONTRIBUTING.md", "AGENTS.md", "ARCHITECTURE.md", "CONTEXT.md",
    "LIMITATIONS.md", "ROADMAP.md", "docs/tutorial.md", "docs/adding-a-source.md",
    "docs/history.md",
)
_SCOPE_SUFFIXES = {".py", ".md", ".json", ".sh", ".toml", ".txt", ".yml", ".yaml", ".ipynb"}
_SCOPE_PATTERNS = {
    what: _FORBIDDEN[what]
    for what in (
        "a local home directory",
        "an Azure GUID (subscription / tenant / client id)",
        "a concrete resource group or workspace name",
    )
}
# A literal first path segment after a blob host is a personal prefix (D13): committed blob paths
# keep the account and container but use a placeholder such as `<your-user>` for the prefix. Names
# nobody, so it needs no owner-specific value. Scope files only: the notebook guard above allows
# blob paths wholesale (user, 2026-09-29) and `test_the_demo_notebook_blob_prefix...` pins that cell.
_SCOPE_PATTERNS["a personal blob prefix (literal path segment after the blob host)"] = (
    r"\.(?:dfs|blob)\.core\.windows\.net/[A-Za-z0-9_-]+/"
)


def _scope_files(root: Path) -> list[Path]:
    files = {
        p for g in _SCOPE_GLOBS for p in root.glob(g)
        if p.is_file() and p.suffix in _SCOPE_SUFFIXES
    }
    files |= {root / f for f in _SCOPE_FILES if (root / f).is_file()}
    return sorted(files)


def _scope_offenders(root: Path) -> list[str]:
    out = []
    for path in _scope_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for what, pattern in _SCOPE_PATTERNS.items():
            hits = sorted(set(re.findall(pattern, text)))
            if hits:
                out.append(f"{path.relative_to(root)}: {what}: {hits}")
    return out


def test_runnable_and_living_files_carry_no_identifiers():
    offenders = _scope_offenders(REPO_ROOT)
    assert not offenders, (
        "These files hardcode an identifier. Use a path relative to the repo "
        "(`Path(__file__).resolve().parents[N]`) or a placeholder such as `<your-user>`:\n"
        + "\n".join(offenders)
    )


def test_the_scope_guard_catches_a_planted_home_path(tmp_path):
    """AC9: a planted absolute home path in `benchmarks/` fails the guard; an elided one is fine."""
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "run.py").write_text('ROOT = "/Users/someone/work"\n')
    (tmp_path / "benchmarks" / "ok.md").write_text("see `/Users/…/project` for the layout\n")
    found = _scope_offenders(tmp_path)
    assert len(found) == 1 and found[0].startswith("benchmarks/run.py")


def test_the_scope_guard_catches_a_planted_blob_prefix(tmp_path):
    (tmp_path / "demos").mkdir()
    (tmp_path / "demos" / "bad.md").write_text('AZ_ROOT="abfss://data@acct.dfs.core.windows.net/someone/run"\n')
    (tmp_path / "demos" / "ok.md").write_text('AZ_ROOT="abfss://data@acct.dfs.core.windows.net/<your-user>/run"\n')
    found = _scope_offenders(tmp_path)
    assert len(found) == 1 and found[0].startswith("demos/bad.md")


def test_the_demo_notebook_blob_prefix_is_a_placeholder_that_fails_if_forgotten():
    """D13 (placeholder-only rule): the guard cannot name the owner's prefix without committing it,
    so the runnable Settings cell holds `<your-user>` and asserts it was replaced."""
    src = "".join("".join(c["source"]) for c in _cells("e2e_austria_aml.ipynb") if c["cell_type"] == "code")
    assert re.search(r'AZ_ROOT = "abfss://[^"]*/<your-user>/', src)
    assert 'assert "<" not in AZ_ROOT' in src


def test_blob_paths_are_allowed_but_identifiers_are_not():
    """Blob paths may be committed (user, 2026-09-29) -- and `abfss://container@account...`
    is shaped like an email, which is how the email pattern used to flag every one of them.
    Pin both halves so the allowance cannot quietly widen into letting a real email through."""
    blob = '"AZ_ROOT = \\"abfss://data@myaccount.dfs.core.windows.net/me/demo-runs\\""'
    blob_https = "https://myaccount.blob.core.windows.net/data/me/x.tif"
    for text in (blob, blob_https):
        for what, pattern in _FORBIDDEN.items():
            assert not re.findall(pattern, text), f"{what} flagged a blob path: {text}"

    assert re.findall(_FORBIDDEN["an email address"], "contact: someone@example.org")
    assert re.findall(_FORBIDDEN["an email address"], '"author": "a.b+c@uni.edu"')


@pytest.mark.parametrize("name", TRACKED_NOTEBOOKS)
def test_private_values_still_come_from_rise_eo_config(name):
    """The positive half of the rule above.

    Without this, deleting the `rise_eo.config.load()` call and replacing it with literals
    would still pass every pattern check right up until someone filled in a real name.
    Spec 54 D6 moved the config half of `notebooks/_config.py` into `rise_eo.config`; this
    guard's purpose is unchanged, only its target call.
    """
    src = _source(name)
    assert re.search(r"\brise_eo\.config\.load\(", src), f"{name} no longer calls rise_eo.config.load()"


# --- run-book notebooks only (spec 24 A1.5) --------------------------------------------


def test_the_runbook_template_is_guarded():
    """The glob above must find something, or a typo in RUNBOOKS guards nothing, silently."""
    assert "TEMPLATE.ipynb" in RUNBOOK_NOTEBOOKS, f"no TEMPLATE.ipynb under {RUNBOOKS}"


@pytest.mark.parametrize("name", RUNBOOK_NOTEBOOKS)
def test_runbook_names_its_kernel(name):
    """The kernel is the run-book's job, not the reader's guess: the metadata pre-selects the
    main checkout's `.venv`, and a cell asserts it at run time (`sys.prefix`)."""
    nb = json.loads(_path(name, "runbooks").read_text())
    assert nb["metadata"].get("kernelspec", {}).get("display_name") == ".venv", (
        f"runbooks/{name}: metadata.kernelspec.display_name must be '.venv'")
    assert "sys.prefix" in _source(name, "runbooks"), (
        f"runbooks/{name} never checks which Python it runs on (sys.prefix)")


@pytest.mark.parametrize("name", RUNBOOK_NOTEBOOKS)
def test_runbook_checks_which_rise_eo_it_imported(name):
    """`rise_eo` is installed editable from the MAIN checkout, so a worktree's run-book that just
    `import rise_eo`s runs main's code. The run-book must put its own `src/` first and check."""
    src = _source(name, "runbooks")
    # Run-book records written before the rename (spec 102 A11) still say `fsd.__file__`.
    assert "sys.path.insert" in src and ("rise_eo.__file__" in src or "fsd.__file__" in src), (
        f"runbooks/{name} must put its own src/ on sys.path and assert rise_eo.__file__")
