"""Doc-corpus checks (spec 41 D6). Synthetic, no network.

Implemented here:
  * assertion 4 (P1) - every point-in-time doc parses as a valid D4 status
    header, and every `superseded_by` names a file that exists.
  * assertion 1 (P4) - every `rise_eo.config` key is documented in
    `docs/reference/environment.md`. Re-scoped 2026-08-26 (spec 54 D6): the left-hand side
    of the parity contract moved from `env.example.sh` (retired) to `rise_eo.config.KEYS`, the
    schema `fsd init` now writes.

Assertions 2 and 3 (link resolution, README verb existence) belong to P5.
"""

# Spec 102 A2.6: CI skips the rest of the suite on a docs-only PR but always runs this file. So a test
# that reads a Markdown file belongs in tests/test_docs.py or tests/test_notebooks.py, nowhere else.

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_VALID_STATUSES = {"current", "historical"}
_SUPERSEDED_RE = re.compile(r"^superseded-by-(.+)$")

# Not point-in-time documents (or not governed by D4): indexes, the template
# skeleton (which carries the header pattern as a placeholder, not a real
# status), and non-run-book support files.
_EXCLUDE_BASENAMES = {"README.md", "TEMPLATE.md"}

# Every directory spec 41 D3 classifies as point-in-time. `specs/` and `runbooks/`
# are D6 assertion 4's literal wording; the other three were stamped in P1/P3 and
# are covered here too, so their headers cannot rot untested.
_D4_DIRS = ("specs", "runbooks", "demos", "benchmarks", "docs/findings")


def _d4_targets() -> list[Path]:
    paths = []
    for d in _D4_DIRS:
        for p in sorted((REPO_ROOT / d).glob("*.md")):
            if p.name in _EXCLUDE_BASENAMES:
                continue
            paths.append(p)
    return paths


def _parse_header(path: Path) -> dict:
    text = path.read_text()
    assert text.startswith("---\n"), f"{path}: missing D4 header (must start with '---')"
    end = text.find("\n---\n", 4)
    assert end != -1, f"{path}: D4 header not terminated with a second '---' line"
    block = text[4:end]
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if not line.strip():
            continue
        assert ":" in line, f"{path}: malformed header line {line!r}"
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


@pytest.mark.parametrize("path", _d4_targets(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_d4_header_parses(path: Path):
    fields = _parse_header(path)

    assert "status" in fields, f"{path}: header missing 'status'"
    assert "summary" in fields, f"{path}: header missing 'summary'"
    assert fields["summary"], f"{path}: 'summary' is empty"

    status = fields["status"]
    m = _SUPERSEDED_RE.match(status)
    if m:
        assert "superseded_by" in fields, (
            f"{path}: status is {status!r} but header has no 'superseded_by'"
        )
        assert fields["superseded_by"] == m.group(1), (
            f"{path}: status says superseded-by-{m.group(1)} but "
            f"superseded_by: {fields['superseded_by']!r} disagrees"
        )
    else:
        assert status in _VALID_STATUSES, (
            f"{path}: status {status!r} is not one of "
            f"{_VALID_STATUSES} or 'superseded-by-NN'"
        )
        assert "superseded_by" not in fields, (
            f"{path}: 'superseded_by' set but status is {status!r}, not superseded-by-NN"
        )


def _superseded_targets() -> list[Path]:
    """Only the docs whose header names a `superseded_by`. Filtered at collection, not skipped
    per doc, so CI's `-rs` list stays short enough to read (spec 102 AC1). A malformed header
    is `test_d4_header_parses`'s job, so this filter is a lenient text match."""
    return [
        p for p in _d4_targets()
        if re.search(r"^superseded_by:", p.read_text().split("\n---\n", 1)[0], re.MULTILINE)
    ]


@pytest.mark.parametrize(
    "path", _superseded_targets(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_d4_superseded_by_target_exists(path: Path):
    target = _parse_header(path)["superseded_by"]

    for base_dir in (REPO_ROOT / "specs", REPO_ROOT / "runbooks"):
        matches = list(base_dir.glob(f"{target}-*.md")) + list(base_dir.glob(f"{target}.md"))
        if matches:
            return
    pytest.fail(f"{path}: superseded_by {target!r} names no file in specs/ or runbooks/")


# --------------------------------------------------------------------------
# Assertion 1 (spec 41 D6/D7, re-scoped spec 54 D6): `rise_eo.config` key parity.
#
# The drift this kills is measured: ~50 variables accreted across the run-books
# with no canonical list, including four spellings of one idea (AZ_ARCHIVE /
# _ROOT / _PATH / _CATALOG). Every documentation defect that cost a real run was
# of this class. The parity contract survives spec 54; only its left-hand side
# changed, from a shell template to `rise_eo.config`'s schema.
# --------------------------------------------------------------------------

_ENV_REFERENCE = REPO_ROOT / "docs" / "reference" / "environment.md"

def test_az_vars_are_documented():
    """Every `rise_eo.config` key's `AZ_*` env name appears in the environment reference table."""
    import rise_eo.config as config

    reference = _ENV_REFERENCE.read_text()
    missing = sorted(v for v in config._KEY_TO_ENV.values() if v not in reference)
    assert not missing, f"undocumented in docs/reference/environment.md: {missing}"


# --------------------------------------------------------------------------
# Assertions 2 and 3 (spec 41 D6, P5): links resolve, and the README's verbs
# are real. Assertion 3 alone would have caught the README that called
# `run_inference` a stub for weeks after it shipped and ran on a cluster.
# --------------------------------------------------------------------------

_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
# Root documents + the maintained docs/ tree. Point-in-time corpora are excluded
# deliberately: they are never edited after the fact (D3), so a link that rots
# there is a fact about history, not a defect to fix.
_LINKED_DOCS = (
    "README.md", "ARCHITECTURE.md", "CONTEXT.md", "ROADMAP.md", "CONTRIBUTING.md", "AGENTS.md",
    "LIMITATIONS.md", "runbooks/README.md",
)


def _link_targets(text: str):
    for raw in _MD_LINK_RE.findall(text):
        target = raw.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        yield target


def _docs_with_links() -> list[Path]:
    paths = [REPO_ROOT / n for n in _LINKED_DOCS]
    # ADRs and findings are point-in-time (ADR 0022): their links may go dead when a file they
    # cite is deleted (spec 102 D8). Their index pages are maintained, so those stay checked.
    paths += sorted(
        p for p in (REPO_ROOT / "docs").rglob("*.md")
        if p.parent.name not in ("adr", "findings") or p.name == "README.md"
    )
    return paths


@pytest.mark.parametrize("path", _docs_with_links(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_relative_links_resolve(path: Path):
    broken = [t for t in _link_targets(path.read_text()) if not (path.parent / t).exists()]
    assert not broken, f"{path}: link target(s) do not exist: {broken}"


def test_readme_verbs_exist():
    """Every `fsd.<verb>(` the README calls is really in `rise_eo.__all__`."""
    import rise_eo

    called = set(re.findall(r"\brise_eo\.([a-z_][a-z0-9_]*)\s*\(", (REPO_ROOT / "README.md").read_text()))
    assert called, "README quickstart calls no fsd verbs — did the example disappear?"
    missing = sorted(v for v in called if v not in rise_eo.__all__)
    assert not missing, f"README calls fsd.<verb> that is not in rise_eo.__all__: {missing}"


def _readme_fsd_calls():
    """Every `fsd.<verb>(...)` call in the README's python blocks, as (verb, npos, kwnames).

    Calls that splat (`*args`/`**kwargs`) are skipped — arity is unknowable statically.
    """
    text = (REPO_ROOT / "README.md").read_text()
    for block in re.findall(r"```python\n(.*?)```", text, re.S):
        for node in ast.walk(ast.parse(block)):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                    and fn.value.id == "rise_eo"):
                continue
            if any(isinstance(a, ast.Starred) for a in node.args) or \
                    any(k.arg is None for k in node.keywords):
                continue
            yield fn.attr, len(node.args), [k.arg for k in node.keywords]


def test_readme_calls_bind_to_real_signatures():
    """The README's example calls must actually bind — assertion 3 with teeth.

    Verb *existence* (above) does not prove the call is callable: the P5 review found
    `rise_eo.download(...)` missing its required `max_tiles` and `rise_eo.run_inference(...)`
    passing a `model_bundle=` keyword that does not exist. Both raise TypeError on the
    first line a newcomer copies. This binds each call's real arity and keyword names
    against the live signature, without executing anything.
    """
    import rise_eo

    calls = list(_readme_fsd_calls())
    assert calls, "README's python blocks contain no fsd.<verb>(...) calls to check."
    sentinel = object()
    failures = []
    for verb, npos, kwnames in calls:
        fn = getattr(rise_eo, verb, None)
        if fn is None:
            failures.append(f"fsd.{verb} does not exist")
            continue
        try:
            inspect.signature(fn).bind(*[sentinel] * npos, **dict.fromkeys(kwnames, sentinel))
        except TypeError as exc:
            failures.append(f"fsd.{verb}(...): {exc}")
    assert not failures, "README example call(s) would raise TypeError:\n  " + "\n  ".join(failures)


# --- run-book python snippets reference real fsd attributes --------------------
#
# Run-book snippets are copy-pasted verbatim onto a remote VM, where a typo costs a
# round-trip through VPN + `az login` + a clone. Three have bitten already:
# `git check-ignore -v` inverting its own PASS verdict; `from rise_eo import storage as fs`
# (rise_eo.storage is a PACKAGE -- the functions live in rise_eo.storage.fs), which raised
# AttributeError at run-book 43 Step 1f; and `fs.put(<dir>, ..., recursive=True)`, where
# put/get are file-only and take no `recursive`.
#
# This checks the class the first two belong to: every attribute a snippet reads off a
# module it imported from `fsd` must actually exist. It imports the module (cheap, no
# side effects) but executes no snippet.

_SNIPPET_DIRS = ("runbooks", "docs")
# The selector MUST match every form `_fsd_attr_uses` parses. Selecting only on
# ```python fences made this test vacuous on its first write: run-book 43 has none
# -- all three of its snippets are `python -c "..."` inside ```bash blocks, i.e.
# exactly the file the test was added for.
# ⚠️ The interpreter is NOT always spelled `python`: run-book 58-p2 invokes a specific
# venv through a shell variable (`"$PY" -c "`) because its code lives in a worktree, not
# the main checkout. Keying on the literal `python` silently dropped that whole run-book
# from this test -- the same vacuous-selector trap the comment above records, one spelling
# later. Match ANY `<word> -c "` opener; a non-Python one (`bash -c "`) simply fails
# `ast.parse` below and is skipped, which is already the contract for prose-y snippets.
_SNIPPET_RE = re.compile(r'```(?:python|py)\n|\S+ -c "\n', re.S)


def _docs_with_python_snippets() -> list[Path]:
    paths = []
    for d in _SNIPPET_DIRS:
        for path in sorted((REPO_ROOT / d).rglob("*.md")):
            if _SNIPPET_RE.search(path.read_text()):
                paths.append(path)
    return paths


def _fsd_attr_uses(text: str):
    """`(module, attr)` for every `alias.attr` where `alias` came from an
    `import fsd...`/`from rise_eo... import ...` in the SAME snippet.

    Both `python -c "..."` bodies inside bash blocks and plain ```python blocks are
    covered: the former are extracted first, so a snippet's imports and its uses are
    always parsed together.
    """
    blocks = list(re.findall(r"```(?:python|py)\n(.*?)```", text, re.S))
    blocks += re.findall(r'\S+ -c "\n(.*?)"\n', text, re.S)
    for block in blocks:
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue  # a prose-y or templated snippet; not this test's business
        alias_to_module: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("rise_eo"):
                for a in node.names:
                    alias_to_module[a.asname or a.name] = f"{node.module}.{a.name}"
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith("rise_eo"):
                        alias_to_module[a.asname or a.name] = a.name
        if not alias_to_module:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                module = alias_to_module.get(node.value.id)
                if module:
                    yield module, node.attr


@pytest.mark.parametrize(
    "path", _docs_with_python_snippets(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_doc_snippets_use_real_fsd_attributes(path: Path):
    import importlib

    failures = []
    for module_path, attr in _fsd_attr_uses(path.read_text()):
        obj = None
        for candidate in (module_path, module_path.rsplit(".", 1)[0]):
            try:
                obj = importlib.import_module(candidate)
                break
            except ImportError:
                continue
        if obj is None:
            continue  # an optional-extra module this env lacks; not a doc defect
        if candidate != module_path:  # imported the parent, so resolve the leaf
            leaf = module_path.rsplit(".", 1)[1]
            if not hasattr(obj, leaf):
                failures.append(f"{module_path} does not exist")
                continue
            obj = getattr(obj, leaf)
        if not hasattr(obj, attr):
            # A SUBMODULE is only an attribute of its package once something has imported it,
            # so `import rise_eo` alone leaves `rise_eo.aml` unset and this check would depend on
            # whichever test ran first (Opus review, 2026-08-27 -- `rise_eo.aml` in
            # docs/howto/build-the-images.md was passing only because tests/test_aml_*.py
            # happened to import it earlier in the session). Ask the import system directly.
            try:
                importlib.import_module(f"{module_path}.{attr}")
                continue
            except ImportError:
                pass
            failures.append(
                f"{module_path}.{attr} does not exist "
                f"(is {module_path} a package whose functions live one level deeper?)"
            )
    assert not failures, f"{path.name} snippet references a missing fsd attribute:\n  " + \
        "\n  ".join(sorted(set(failures)))


def test_snippet_selector_is_not_tied_to_the_literal_word_python():
    """Regression: the selector keyed on `python -c "`, so run-book 58-p2 -- which runs a
    specific venv through a shell variable, `"$PY" -c "`, because P2's code lives in a
    worktree rather than the main checkout -- silently fell out of
    `_docs_with_python_snippets()` entirely. The whole run-book stopped being checked,
    with no failure to notice: the parametrized case simply vanished.

    This pins the extractor on both spellings, so the next interpreter spelling
    (`"$PY"`, `$PYTHON`, an absolute path) cannot quietly un-cover a run-book again."""
    text = (
        '```bash\n'
        '"$PY" -c "\n'
        'from rise_eo import api\n'
        'api.download(roi=None)\n'
        '"\n'
        '```\n'
        '```bash\n'
        '.venv/bin/python -c "\n'
        'from rise_eo.catalog import catalog\n'
        'catalog.TileCatalog(\'x\')\n'
        '"\n'
        '```\n'
    )
    assert _SNIPPET_RE.search(text), "a `\"$PY\" -c` snippet must make the file selectable"
    uses = set(_fsd_attr_uses(text))
    assert ("rise_eo.api", "download") in uses, uses
    assert ("rise_eo.catalog.catalog", "TileCatalog") in uses, uses


# --- run-book `<py> -c "..."` snippets must survive the shell wrapper ----------------

_DASH_C_SNIPPET_RE = re.compile(r'\S+ -c "\n(.*?)\n"\n', re.S)


def _dash_c_snippets_with_unescaped_quotes(text: str) -> list[str]:
    """Lines inside a `<interpreter> -c "..."` body carrying an UNESCAPED `"`.

    The body is wrapped in shell double quotes, so a bare `"` ends the string early and
    the rest of the snippet is reinterpreted as shell -- a copy-paste that fails in a way
    that looks nothing like the Python it came from. `\\"` is correct and common (three
    existing run-books rely on it for f-strings); only a bare one is the bug.
    """
    offenders = []
    for body in _DASH_C_SNIPPET_RE.findall(text):
        for line in body.splitlines():
            if re.search(r'(?<!\\)"', line):
                offenders.append(line.strip())
    return offenders


@pytest.mark.parametrize(
    "path", _docs_with_python_snippets(), ids=lambda p: str(p.relative_to(REPO_ROOT))
)
def test_dash_c_snippets_have_no_unescaped_double_quotes(path: Path):
    """Caught for real while writing run-book 58-p2: a comment reading
    `# ... with "'str' object has no attribute 'tzinfo'".` inside a `"$PY" -c "` body
    would have terminated the shell string mid-snippet. `ast.parse` is happy with it --
    the Python is valid -- so only this check sees it."""
    offenders = _dash_c_snippets_with_unescaped_quotes(path.read_text())
    assert not offenders, (
        f"{path.name}: unescaped \" inside a `-c \"...\"` body would end the shell "
        "string early; escape it as \\\" :\n  " + "\n  ".join(offenders)
    )


# --------------------------------------------------------------------------
# Spec 102 D5/D6: numbering guards. Two new files with distinct slugs never conflict
# in git, so a duplicate spec/ADR number would merge silently. These fail CI instead.
# Run-books are out of scope: their prefix is the spec number, so duplicates are by design.
# --------------------------------------------------------------------------

_SPEC_NUM_RE = re.compile(r"^(\d+[a-z]?)-")
_ADR_NUM_RE = re.compile(r"^(\d+)-")
# Already on `main` when the guard landed; allow-listed explicitly, and only this pair.
_KNOWN_DUPLICATE_SPEC_NUMBERS = {"18"}


def _number_key(raw: str) -> str:
    """`05` and `5` (or `0033` and `33`) are one number; `25b` stays distinct from `25`."""
    return raw.lstrip("0") or "0"


def _duplicate_numbers(paths: list[Path], number_re: re.Pattern, allowed: set[str] = frozenset()):
    """`{number: [files]}` for every number shared by more than one of `paths`."""
    by_number: dict[str, list[Path]] = {}
    for p in paths:
        m = number_re.match(p.name)
        if m and _number_key(m.group(1)) not in allowed:
            by_number.setdefault(_number_key(m.group(1)), []).append(p)
    return {n: ps for n, ps in by_number.items() if len(ps) > 1}


def _duplicate_message(dups: dict[str, list[Path]]) -> str:
    return "duplicate number (the PR that merges second renumbers its file):\n  " + "\n  ".join(
        f"{n}: " + ", ".join(p.name for p in ps) for n, ps in sorted(dups.items())
    )


def test_no_duplicate_spec_numbers():
    paths = sorted((REPO_ROOT / "specs").glob("*.md"))
    dups = _duplicate_numbers(paths, _SPEC_NUM_RE, _KNOWN_DUPLICATE_SPEC_NUMBERS)
    assert not dups, _duplicate_message(dups)


def test_no_duplicate_adr_numbers():
    paths = sorted((REPO_ROOT / "docs" / "adr").glob("*.md"))
    dups = _duplicate_numbers(paths, _ADR_NUM_RE)
    assert not dups, _duplicate_message(dups)


def test_duplicate_check_names_both_files():
    """AC2's red case: a second `59-*.md` is caught and both files are named."""
    dups = _duplicate_numbers(
        [Path("specs/59-a.md"), Path("specs/59-b.md"), Path("specs/60-c.md")], _SPEC_NUM_RE
    )
    assert list(dups) == ["59"]
    msg = _duplicate_message(dups)
    assert "59-a.md" in msg and "59-b.md" in msg


def test_duplicate_check_ignores_zero_padding():
    specs = [Path("specs/05-a.md"), Path("specs/5-b.md"), Path("specs/25-c.md"), Path("specs/25b-d.md")]
    assert list(_duplicate_numbers(specs, _SPEC_NUM_RE)) == ["5"]
    adrs = [Path("docs/adr/0033-a.md"), Path("docs/adr/33-b.md")]
    assert list(_duplicate_numbers(adrs, _ADR_NUM_RE)) == ["33"]


def test_known_duplicate_allowlist_covers_exactly_the_18_pair():
    paths = sorted((REPO_ROOT / "specs").glob("18-*.md"))
    assert [p.name for p in paths] == ["18-model-adapter.md", "18-model-bundle-explainer.md"]
    assert _KNOWN_DUPLICATE_SPEC_NUMBERS == {"18"}


def _issue_header_problem(number: int, fields: dict) -> str | None:
    """Specs numbered >= 102 take their number from their tracking issue (D5)."""
    if number < 102:
        return None
    if "issue" not in fields:
        return f"spec {number} has no `issue:` header (required for specs >= 102)"
    if fields["issue"].strip("\"'") != f"#{number}":
        return f"spec {number}: `issue: {fields['issue']}` must be \"#{number}\" (number = issue)"
    return None


@pytest.mark.parametrize(
    "path",
    [p for p in _d4_targets() if p.parent.name == "specs" and _SPEC_NUM_RE.match(p.name)],
    ids=lambda p: p.name,
)
def test_new_specs_carry_their_tracking_issue(path: Path):
    number = int(re.match(r"\d+", path.name).group())
    problem = _issue_header_problem(number, _parse_header(path))
    assert problem is None, f"{path}: {problem}"


def test_issue_header_check_red_cases():
    assert _issue_header_problem(59, {}) is None  # old specs are exempt
    assert _issue_header_problem(102, {"issue": '"#102"'}) is None
    assert "no `issue:` header" in _issue_header_problem(103, {})
    assert "must be" in _issue_header_problem(104, {"issue": '"#103"'})


# Spec 102 D7 / AC12: the index tables are frozen. The spec snapshot lists every spec up to 59 and
# nothing after; newer specs are listed by their tracking issues. The ADR index has no table at all.
_FROZEN_LAST_SPEC = 59


def _spec_files() -> list[Path]:
    return [p for p in (REPO_ROOT / "specs").glob("*.md") if p.name not in _EXCLUDE_BASENAMES]


def _snapshot_problems(listed: set[str], spec_names: list[str]) -> list[str]:
    problems = []
    for name in spec_names:
        m = re.match(r"\d+", name)
        frozen = m is None or int(m.group()) <= _FROZEN_LAST_SPEC
        if frozen and name not in listed:
            problems.append(f"{name} is missing from the frozen snapshot")
        if not frozen and name in listed:
            problems.append(f"{name} is listed, but the snapshot is frozen at spec {_FROZEN_LAST_SPEC}")
    return problems


def test_spec_index_is_a_frozen_snapshot_up_to_59():
    text = (REPO_ROOT / "specs" / "README.md").read_text(encoding="utf-8")
    assert "frozen" in text.lower()
    listed = set(_MD_LINK_RE.findall(text))
    problems = _snapshot_problems(listed, [p.name for p in _spec_files()])
    assert not problems, "\n".join(problems)


def test_snapshot_check_red_cases():
    assert _snapshot_problems({"59-x.md"}, ["59-x.md", "102-y.md"]) == []
    assert "missing" in _snapshot_problems(set(), ["48-x.md"])[0]
    assert "frozen at spec 59" in _snapshot_problems({"102-y.md"}, ["102-y.md"])[0]
    assert "missing" in _snapshot_problems(set(), ["research-notes.md"])[0]


def test_adr_index_has_no_table():
    text = (REPO_ROOT / "docs" / "adr" / "README.md").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| [0")]
    assert not rows, f"docs/adr/README.md is frozen (ADR 0033); the folder listing is the index: {rows[:2]}"


@pytest.mark.parametrize("notice", ["notebooks/shapefiles/NOTICE", "tests/data/tutorial/NOTICE"])
def test_eurocrops_notices_carry_licence_and_citation(notice):
    """Spec 102 D14/AC10: data derived from EuroCrops is CC BY 4.0 and must be credited."""
    text = (REPO_ROOT / notice).read_text(encoding="utf-8")
    assert "CC BY 4.0" in text
    assert "doi:10.5281/zenodo.7851838" in text
    assert "NOT been reconciled" not in text


# Spec 102 D8 / AC11: the docs deleted in P3c stay deleted, and no living doc sends a reader to
# one. A mention is allowed on a line that also names the archive tag. `docs/history.md` is
# exempt: it is the narrative of how those files came and went.
_DELETED_DOCS = (
    "PROGRESS.md", "RECIPES.md", "CHANGES.md", "DROPPED.md", "BUGS.md", "TODO.md",
    "docs/progress-archive.md",
)
_ARCHIVE_TAG = "docs-archive-2026"
_RUNBOOK_PATH_RE = re.compile(r"runbooks/(?:scripts/)?[\w.-]+\.(?:md|py|ipynb)")


def _deleted_doc_problems(name: str, text: str, exists) -> list[str]:
    problems = []
    for n, line in enumerate(text.splitlines(), 1):
        if _ARCHIVE_TAG in line:
            continue
        for doc in _DELETED_DOCS:
            if re.search(rf"(?<![\w/-]){re.escape(Path(doc).name)}", line):
                problems.append(f"{name}:{n} names {doc}")
        for path in _RUNBOOK_PATH_RE.findall(line):
            if not exists(path):
                problems.append(f"{name}:{n} names {path}, which does not exist")
    return problems


def test_deleted_docs_stay_deleted():
    present = [d for d in _DELETED_DOCS if (REPO_ROOT / d).exists()]
    assert not present, f"deleted in spec 102 P3c (D8); readable at tag {_ARCHIVE_TAG}: {present}"


def test_living_docs_do_not_point_at_deleted_docs():
    living = [p for p in _docs_with_links() if p.name != "history.md"]
    # Runnable demos print these pointers at users, and their READMEs are maintained. The demo
    # write-ups (`demos/*.md` other than READMEs) are point-in-time and exempt.
    demos = REPO_ROOT / "demos"
    living += sorted(
        p for p in demos.rglob("*")
        if p.is_file() and (p.suffix in (".py", ".yml", ".yaml") or p.name == "README.md")
    )
    problems = []
    for p in living:
        problems += _deleted_doc_problems(
            str(p.relative_to(REPO_ROOT)), p.read_text(encoding="utf-8"),
            lambda rel: (REPO_ROOT / rel).exists(),
        )
    assert not problems, "\n".join(problems)


def test_deleted_doc_check_red_cases():
    never = lambda rel: False  # noqa: E731
    assert _deleted_doc_problems("x", "see PROGRESS.md", never)
    assert _deleted_doc_problems("x", "see `runbooks/36-aml-runner.md`", never)
    assert not _deleted_doc_problems("x", "PROGRESS.md is at tag docs-archive-2026", never)
    assert not _deleted_doc_problems("x", "`runbooks/58-x.md`", lambda rel: True)
    assert not _deleted_doc_problems("x", "the old TODO file", never)


# --- Setup installs what CI installs (spec 102 P4) ----------------------------------------
# The Setup blocks must name the same extras as CI's install step, so "tests green before you
# start" holds on a fresh clone.
_EXTRAS_RE = re.compile(r'pip install -e "\.\[([^\]]+)\]"')


def _install_extras(text: str) -> list[set[str]]:
    return [set(m.split(",")) for m in _EXTRAS_RE.findall(text)]


def test_setup_installs_the_extras_ci_installs():
    ci = _install_extras((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    assert len(ci) == 1, f"expected one `pip install -e \".[...]\"` in ci.yml, found {len(ci)}"
    for name in ("CONTRIBUTING.md", "AGENTS.md"):
        found = _install_extras((REPO_ROOT / name).read_text(encoding="utf-8"))
        assert found == ci, f"{name} Setup installs {found}, CI installs {ci[0]}"


def test_install_extras_red_cases():
    assert _install_extras('pip install -e ".[dev,local]"') == [{"dev", "local"}]
    assert _install_extras("pip install rise-eo") == []
