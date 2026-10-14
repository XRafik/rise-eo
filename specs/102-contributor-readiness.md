---
status: current
summary: Make fsd contributable by teammates (by hand or through their own agents) and maintainable by a future human maintainer without an agent. Every change reaches a protected `main` through a pull request that passes four gates (CI green, linked issue or short spec, non-author review, real-run evidence where data/cloud/pixels are touched). Add CI with a weekly scheduled run and golden-file tests for on-disk formats. Number specs by their tracking issue, keep ADRs sequential, and have CI fail on duplicate numbers. Freeze the indexes, keep a small set of living docs, and delete and disperse the rest behind an archive tag. Add an in-repo `AGENTS.md` that includes a prior-art rule. Make shared registry aliases move only from `main`. Rename the project and transfer it to the `nasaharvest` org last.
issue: "#102"
---

# Spec 102 — contributor readiness

**Status:** **SIGNED OFF (user, 2026-10-05)**, including §9's two questions. Decisions D1–D18 were each
confirmed by the user in the grilling session of 2026-10-03 → 2026-10-05; this document writes them down
and adds acceptance criteria. **Amended before sign-off:** D6 switched from `<spec>-<n>` ADR ids to sequential
numbers (Q1), and D5 made the `issue:` header required (Q2).
**Tracking issue:** [#102](https://github.com/nikhilsrajan/fsd/issues/102). This is the first spec
numbered by its own tracking issue (D5).
**Related:** [spec 24](24-working-contract.md) (the working contract this partly supersedes),
[spec 41](41-docs-refactor.md) (point-in-time vs living docs; ADRs
[0022](../docs/adr/0022-documents-are-point-in-time-or-continuously-true.md) and
[0023](../docs/adr/0023-three-value-status-header-process-state-in-the-index.md)), ADR
[0025](../docs/adr/0025-one-fact-one-home.md) (one home per fact).

---

## 1. Problem

fsd was built by one person working almost entirely through an agent, and its working contract
lives outside the repo, in a workspace `CLAUDE.md` and about 40 memory notes. Measured on `main` @ `bc01d19`
(2026-10-03):

- **Nothing checks a contribution.** There is no `.github/`, no CI, no `CONTRIBUTING.md` and no
  `AGENTS.md`. A contributor's agent never sees the rules, because they live outside the repo.
- **Parallel branches collide.** Specs and ADRs are numbered sequentially, and `18-` is already used twice.
  Hand-edited index tables have gone stale (`specs/README.md` stops at spec 47, even though ADR 0023 calls it
  "regenerated"). Several files are appended to by every change. Commits touching each one since
  2026-06: `PROGRESS.md` 209, `CHANGES.md` 76, `TODO.md` 56, `RECIPES.md` 43.
- **Only an agent can sustain the documentation load.** A typical change writes or updates a spec
  (with per-source credit), ADRs, `CHANGES.md`, `PROGRESS.md` (plus its archive), `RECIPES.md`, index
  rows and a run-book notebook. A developer without an agent faces a high barrier to entry, and so does
  the future maintainer: **ownership will transfer** (user, 2026-10-05).
- **Shared cloud state is last-writer-wins by default.** `fsd.aml.ensure_environment(alias="current")`
  (`src/fsd/aml/__init__.py:79`) means any AML run, including a test of a branch, silently repoints the shared
  `current` image.
- **On-disk format versions can collide silently.** Two branches that both change `BUNDLE_VERSION = 2` to
  `= 3` make the same edit, so git merges them without a conflict. `BUNDLE_VERSION` has no history
  comment at all.
- **Tracked files contain the owner's own paths.** Eight `benchmarks/*.py` scripts hard-code an absolute home
  directory, and `notebooks/e2e_austria_aml.ipynb`, `demos/E2E_AUSTRIA_AML.md` and
  `runbooks/59-p2-window-a.ipynb` hard-code a personal blob root.
- **Data licensing is unresolved on paper.** `notebooks/shapefiles/NOTICE` says the EuroCrops licence is
  "NOT reconciled". `tests/data/tutorial/fields.geojson` is derived from the same source, but its
  `NOTICE` credits only Copernicus.
- **Many working habits are practised but not written down.** A sweep of 479 commits found 15 habits
  missing from the working contract (the user confirmed all 15 as real), and 10 places where the contract and
  practice disagree. Examples: review fixes in a separate session, the real-run gate, grill-then-ADR.

## 2. Design target

**D0 — Design for the future maintainer, not for the user + Claude.** The *required* process is what one
human maintainer can sustain **without an agent**. Contributors write **code + tests + a PR template**.
For a contract change, they also write a few paragraphs in the issue, and the maintainer writes a **short**
spec. Everything agent-specific is optional *method*, documented in `AGENTS.md`. That covers handoffs, the
Opus/Sonnet split, notebook run-books, the full spec outline with per-source credit, and grilling.
(User, 2026-10-05; this supersedes the earlier "you + Claude as maintainer" framing.)

**Audience (D1):** NASA Harvest teammates first (about 1–5, working by hand or through their own agents,
e.g. Claude Code, Copilot or Cursor). Outsiders are not blocked (the repo is public and MIT), but there is no
onboarding programme for them: no code of conduct, CLA or triage rota.

## 3. Decisions

### D2 — The contract: four gates every PR passes

| # | Gate | Checked by |
|---|---|---|
| 1 | **CI green**: ruff, the fast pytest suite, guard tests, `docs_kwarg_sweep.py` | GitHub Actions |
| 2 | **Linked issue.** A change to a convention, an on-disk format or the public API also needs a **signed-off short spec** (D5). Bug fixes, refactors, docs, and a new collection that follows `docs/adding-a-source.md` need only the issue | Reviewer |
| 3 | **Reviewed by someone other than the author.** For the user + Claude, that means a different session from the one that wrote the change. Every finding is either **fixed in the PR or filed as an issue** | Reviewer |
| 4 | **Real-run evidence** when the change touches real data, the cloud, or pixels: the run's output or a screenshot (e.g. QGIS) pasted into the PR. A contributor without the archive or Azure may ask the reviewer to run it | Reviewer |

Nobody can check *how* the work was produced, so the method is not enforced.

### D3 — Every change reaches `main` through a pull request

- `main` is **protected**: no direct pushes, and the CI check is required before merging. Merges use a
  **merge commit** (this keeps today's `--no-ff` branch boundary).
- **No "required approvals" count.** "Pull request authors cannot approve their own pull requests"
  (GitHub docs), so with that setting a solo maintainer could not merge their own PRs without an admin
  bypass. Gate 3's review is a **PR comment**, and the **maintainer merges**.
- This applies to the user + Claude too. Pushing a feature branch and opening a draft PR becomes routine;
  merging into `main` stays the maintainer's call.
- Side effect: `origin/main` is always the real `main`. That ends the stale-worktree problem
  (`EnterWorktree` branches from `origin/main`) and the "N commits unpushed" state.

### D4 — CI, and how dependency drift is caught

- **One job:** `ubuntu-latest`, Python **3.11** (`requires-python >=3.11`). It installs
  `.[dev,local,s3,notebooks,grid,azure,aml,mpc,titiler,serving]`, i.e. **every extra**, so the
  ~105 skips that come from missing extras disappear. Steps: `ruff check src/ tests/` → `pytest -q` (the
  `network` marker stays off) → `scripts/docs_kwarg_sweep.py`.
- **Triggers:** every PR, and every push to `main`.
- **No version matrix and no split of the slow suite** for now (the ~170 s tutorial test fits inside a
  ~6 min budget). Revisit either if CI time hurts.
- **No lockfile.** CI installs fresh, the same way `pip install` does for a user.
- **A weekly scheduled run** on `main` (GitHub-hosted) turns an upstream break into an **auto-opened
  issue** before a contributor's PR hits it. The issue, rather than an email, is the signal because
  GitHub notifies only "the user who last modified the cron syntax", and that person changes when
  ownership transfers.
- When something breaks, add a version range to `pyproject.toml` with a comment saying why. That is the
  existing rule (`planetary-computer>=1,<2`).
- Caveat: "In a public repository, scheduled workflows are automatically disabled when no repository
  activity has occurred in 60 days". The weekly run is an early warning while the project is active,
  not a guarantee.

### D5 — Specs: numbered by their own tracking issue; a short template

- **Each new spec opens its own tracking issue when drafting starts, and the spec's number is that issue's
  number.** New issues are #102 and up, so they never overlap specs `00`–`59`, which keep their numbers.
  **Don't reuse the number of an old problem issue** (#1–#59 already collide with spec numbers): the old
  issue links to the new spec issue and closes when the spec's work merges.
- Implementation status lives in the tracking issue: open means not done, closed means done (see D7).
- **Short template** (required; about 30 minutes of writing for a human):

  ```markdown
  ---
  status: current
  summary: <one paragraph>          # tests/test_docs.py requires status + summary
  issue: "#NNN"                     # required for specs >= 102 (P1 adds the check)
  ---
  # Spec NNN — <title>
  ## Problem        — what is wrong or missing, measured if possible
  ## Decision       — D1, D2…; each says what changes
  ## Prior art      — the standard practice each decision follows, or "searched X, none fit" (D9)
  ## How to verify  — acceptance checks: tests, plus any real-run evidence (gate 4)
  ## Out of scope
  <!-- optional for big specs: Phases, Risks, Alternatives, Sources with per-source credit -->
  ```
- After sign-off, change a spec by **amending** it (A*n*, each with its own sign-off), not by rewriting it.
- Run-book prefixes stay equal to the spec number. Duplicate prefixes are by design: three run-books for
  spec 31 is not a collision.

### D6 — ADRs: sequential numbers, guarded by a duplicate-number check

- **New ADRs continue the sequential `NNNN-` numbering** (the next one is `0033`), as Nygard's original
  ADR post prescribes: "numbered sequentially and monotonically. Numbers will not be reused."
- **A CI check fails on a duplicate ADR or spec number.** Two new files with distinct slugs never
  conflict in git, so without the check a duplicate number would merge silently. When two PRs pick the
  same number, the PR that merges second renumbers its file. With 1–5 contributors and about 8 ADRs a
  month, that is rare.
- The format is unchanged (Context / Decision / Options rejected / Consequences). An ADR lands **in the same
  PR as its spec**, is never edited afterwards, and is superseded by a new ADR.
- **Considered and rejected (Q1, user, 2026-10-05):** `<spec>-<n>` ids such as `0102-1`. They could never
  collide, but no pre-2022 precedent was found, so the D9 prior-art rule prefers the standard. Renaming
  `0001`–`0032` was also rejected: it would touch 238 citations in 59 files, and the 32 commit messages
  that cite ADRs can never be rewritten.

### D7 — Index tables are frozen

- `specs/README.md` gets one last update to spec 59, marked "frozen snapshot; for newer specs see their
  tracking issue". After that, nobody adds rows.
- `docs/adr/README.md` keeps its introduction and points readers at the directory listing. ADR slugs are
  decision sentences, so the listing reads as an index.
- Each file's `status:` header stays (that half of ADR 0023 still holds). **ADR
  [0033](../docs/adr/0033-index-tables-freeze-status-lives-in-the-tracking-issue.md), which lands with this spec,
  supersedes ADR 0023's "regenerated index" half.**

### D8 — Docs: a small living set; everything else is deleted and dispersed

| Bucket | Docs |
|---|---|
| **Living** (the maintainer keeps them true, changed in the same PR as the code) | `README.md`, **new** `CONTRIBUTING.md`, `ARCHITECTURE.md`, `CONTEXT.md` (gains the *MGRS tile vs grid cell* terminology rule from the workspace `CLAUDE.md`), `LIMITATIONS.md`, `ROADMAP.md`, `docs/history.md` (kept, rarely touched), `docs/tutorial.md`, `docs/adding-a-source.md`, `docs/howto/`, `docs/reference/` (gains `AZURE_INFRA.md` and a new `test-data.md`), `docs/findings/` (gains `RSLEARN_COMPARISON.md`) |
| **Optional agent method** | **new** `AGENTS.md` + a one-line `CLAUDE.md` (D9); `runbooks/TEMPLATE.ipynb` and future notebook run-books |
| **Deleted and dispersed** (in P3, behind the tag `docs-archive-2026`) | see the triage below |

**Triage of the deleted files:**

| File | Its job continues in |
|---|---|
| `PROGRESS.md` | Per-work state → **the draft PR's description**. Order of work ("THE ORDER") → **GitHub milestones + a pinned "Order of work" issue**. Session-start checks → `AGENTS.md` (read the open PRs and the pinned issue; the tests must be green; no pinned counts, because CI replaces them). Standing caveats → issues, or `LIMITATIONS.md` when users need to know |
| `RECIPES.md` | User tasks → `docs/howto/`. Scripts → `scripts/`, each with a docstring saying what it does and how to run it. Maintainer tasks → `CONTRIBUTING.md` "Maintainer tasks" or `AGENTS.md`. One-offs → nowhere (the PR description records them) |
| `CHANGES.md`, `DROPPED.md` | GitHub release notes from PR titles + labels (D15). DROPPED's "the legacy repo could, fsd can't" lines → `LIMITATIONS.md` (the *why* is already in ADRs). Nobody uses `fetch_satdata`, so there is **no migration guide** (user, 2026-10-05) |
| `BUGS.md`, `TODO.md` | GitHub issues; file any open bug not yet filed. `CONTRIBUTING.md` says: "`TODO #NN` = issue #NN" |
| `docs/progress-archive.md` | The tag only |
| `runbooks/*.md` that are finished records, with their driver scripts | The tag only. Their outcomes are already in specs and `history.md` |
| run-books still needed to **re-run** something (e.g. `58-redownload-austria-mpc`, needed by #101) | a `docs/howto/` page, or a notebook in `runbooks/` |
| reusable checks (`runbooks/scripts/docs_kwarg_sweep.py`, and the `85_*` checks if still useful) | `scripts/` |

- **Links:** `tests/test_docs.py` already checks links only in living docs (README, ARCHITECTURE,
  CONTEXT, ROADMAP). Point-in-time specs and ADRs are exempt by design, so their links to deleted files go dead,
  which is allowed. Its `_LINKED_DOCS` drops `PROGRESS.md`. `CONTRIBUTING.md` says: "Docs removed in
  spec 102 are readable at tag `docs-archive-2026`."

### D9 — Agent instructions: three layers, plus rules for all agents

| File | Visibility | Contains |
|---|---|---|
| `fsd/AGENTS.md` | public, read by every agent tool | Setup and test commands; code conventions (storage seam, `(data, profile)`, 5-D band contract, nodata = 0, the terminology rule); the four gates; the rules below; the optional agent method (the spec flow, a model-split *suggestion*, handoffs, worktree → PR) |
| `fsd/CLAUDE.md` | public | One line: `@AGENTS.md` |
| workspace `CLAUDE.md` | private (the user's) | Only what is about the user's machine and preferences: the read-only legacy repos, `AZURE_INFRA_PRIVATE.md`, personal preferences |

`fsd/CLAUDE.md` is required, not cosmetic. By default Claude Code reads `AGENTS.md` only when no
`CLAUDE.md` exists "in your working directory or any directory above it". The user's workspace
`CLAUDE.md` sits above `fsd/`, so without the import their sessions would skip `AGENTS.md`. An import
"works without adding … a setting" on all versions (Claude Code memory docs).

**Rules for all agents, in `AGENTS.md`:**
1. **Hand risky runs to a human.** Agents may run fast local checks (pytest, ruff, reading files).
   Anything that touches the network, the cloud or credentials, or runs longer than a few minutes, is
   handed to the human, who pastes back the result. The stricter "never background a script, never poll
   logs" rule and the notebook run-book format stay in the user's private layer.
2. **Prior-art check** (user, 2026-10-05). Before proposing a mechanism (a design, process or test pattern),
   say whether it is homemade. If it is, look for the established practice documented **before
   2022-11-30** and prefer it. Cite it so a reviewer can verify it (a link, plus what it contributed), or
   say what was searched and that nothing fit. **Never invent a precedent to satisfy this rule.** The
   short spec template has a `Prior art` heading, and the review checklist asks about it.
3. **Privacy:** describe an identifier, never spell it out (GUIDs, resource names, personal paths).

### D10 — Knowledge outside the repo moves in

Project lessons that live only in the user's `~/.claude` memory move into the repo. Personal notes stay private.

| Kind | Example memory notes | Destination |
|---|---|---|
| Review lessons | real-run-beats-review, verify-the-primitive-a-spec-cites, test-the-serialization-boundary, fsd-addressing-granularity, docs-call-sites-rot-silently | `CONTRIBUTING.md` **review checklist**: one line each, ≤ 10 lines, each pointing to its incident |
| Design rationale | fsd-datetime-tz, fsd-metadata-pickle-npy, fsd-geospatial-nitpicks | a "why" line in `ARCHITECTURE.md`, or an ADR if it is a real decision |
| Test-data facts | fsd-austria-archive, dont-trust-a-geometry-filename | `docs/reference/test-data.md` |
| Unfiled bugs | dropping-a-label-polygon-is-a-noop | a GitHub issue |
| Collection switching | switching-a-collection-is-not-a-rename | `docs/adding-a-source.md` |

### D11 — On-disk format versions: golden files

`FSD_DECLARATION_VERSION` (the collection declaration, stamped into every tile catalog's Parquet footer
under `fsd:declaration`) and `BUNDLE_VERSION` (`bundle.json` in every model bundle) each get **golden
files**: `tests/data/formats/declaration.v<N>.json` and `bundle.v<N>.json`.

- **Test A:** write a fixed example with today's code and compare it to the golden file for the current
  version. If the format changed without a bump, the test fails and says "bump the version and add
  `…v<N+1>.json`". If the version was bumped but its file doesn't exist yet, the test fails too.
- **Test B:** every version in `SUPPORTED_BUNDLE_VERSIONS` (and the declaration's equivalent) still
  loads from its golden file. That gives a backward-compatibility check; today only one test fakes a v1 bundle
  (`tests/test_bundle_code.py:344`).
- **Two branches both bumping to v3** each add `…v3.json` with different contents, which git reports as a real
  conflict.
- A homemade "the history comment must match the constant" test was considered and **rejected**: it
  misses the likeliest bug (the format changed and nobody bumped the version).
- The `pyproject.toml` version: PRs never touch it, and the maintainer bumps it only when cutting a release (D15).

### D12 — Shared cloud state

1. **Azure is not required to contribute.** For changes that touch the cloud, the maintainer runs gate 4.
   (CI on fork PRs gets no secrets: "With the exception of `GITHUB_TOKEN`, secrets are not passed to the
   runner when a workflow is triggered from a forked repository.")
2. Teammates with `rise` access work in a **personal namespace**: blob root `<user>/…`, and aliases
   `dev-<user>`.
3. **Shared aliases (`current`, `champion`, `demo-*`) move only from `main`, and only the maintainer moves them.**
4. **Code change:** `fsd.aml.ensure_environment`'s default changes from `alias="current"` to
   **`alias=None`**, so a run moves a pointer only when explicitly asked. This is a behavior change, labelled `breaking` in the release notes.

### D13 — The owner's own paths

| Kind | Rule |
|---|---|
| **Runnable** files (`benchmarks/`, `demos/`, `notebooks/`) | Paths relative to the repo root (`Path(__file__).parents[1] / …`). Blob roots come from a Settings cell set to the placeholder `<your-user>`, plus `assert "<" not in AZ_ROOT` so a forgotten placeholder fails immediately |
| **Point-in-time records** (old run-books, archives) | Left as they are: they truthfully record the run |
| **Guard** | The identifier guard in `tests/test_notebooks.py` extends to `benchmarks/`, `demos/`, `src/` and the living docs. It fails on an absolute home path or a personal blob prefix |
| `pyproject` `authors` | Kept; a `maintainers` field is added at handover |

### D14 — Licensing

- **Code is MIT. Data derived from EuroCrops is CC BY 4.0.** Each data folder's `NOTICE` says so and gives
  the suggested citation: Schneider, M., Chan, A., & Körner, M. (2023). *EuroCrops: A Pan-European crop
  dataset.* Zenodo. doi:10.5281/zenodo.7851838.
- `notebooks/shapefiles/NOTICE` drops "NOT reconciled". `tests/data/tutorial/NOTICE` adds the EuroCrops
  credit next to the Copernicus line.
- **No CLA, and no DCO sign-off.** GitHub's terms already apply "inbound=outbound": "Whenever you add
  Content to a repository containing notice of a license, you license that Content under the same terms"
  (GitHub ToS §D.6). `CONTRIBUTING.md` says so in one line. A DCO can be added later if the org requires it.
- The copyright line is kept. At handover it becomes "Copyright (c) 2026 Nikhil Sasi Rajan and fsd contributors".

### D15 — PR template, labels, releases

- `.github/pull_request_template.md` has: `Closes #NN` · **What changed** (the title becomes the release-note line) ·
  **How verified** (tests; real-run output or a screenshot when data/cloud/pixels are touched) · ☐ docs updated if
  user-facing behaviour changed.
- **Labels:** `breaking`, `feature`, `fix`, `docs`, `internal`. **The maintainer applies them at merge.**
- `.github/release.yml` groups merged PRs by label, and GitHub generates the release notes.
- **Releases:** the maintainer tags `v0.y.z` when there is something worth shipping. `breaking` bumps `y`;
  everything else bumps `z` ("Major version zero (0.y.z) is for initial development. Anything MAY change at
  any time", SemVer §4). The first release is **`v0.1.0`, after this spec lands**.

### D16 — Where each confirmed habit ends up

| Habit (from the 479-commit sweep) | Becomes |
|---|---|
| 1 separate-session review; findings fixed or filed · 2 real-run gate · 5 issue-only small work | **Contract**: gates 3, 4, 2 |
| 6 ADRs immutable, same PR as the spec · 7 spec outline + amendments | **Contract**: the D5/D6 templates, "amend, don't rewrite" |
| 8 point-in-time vs living docs, `status:` headers | **Contract**: kept, enforced by `test_docs.py` |
| 14 no back-compat shims for the archive layout; old artifacts raise with the fix named | **Contract**: a code convention in `AGENTS.md` / `CONTRIBUTING.md` (format versions keep their supported lists, D11) |
| 15 commit-subject style | **Contract, PR titles only**: the PR title is the release-note line |
| 3 check a spec against fsd's own code · 13 turn a lesson into an automated check | **Review checklist** (D10) |
| 4 grill before a spec · 9 worktree per change · 12 manual identifier sweep | **Optional agent method** (`AGENTS.md`); CI's guard (D13) is the contract part of 12 |
| 10 pinned expected counts · 11 THE ORDER | **Replaced**: CI green; milestones + a pinned issue |
| "(user, date)" attribution tags | Optional |

### D17 — The fresh-clone dry run

- **Who:** (b) **a fresh agent session that sees only the cloned repo**, with none of the user's memory and no
  workspace `CLAUDE.md`. (a) A teammate without an agent is **deferred**: none is available now (user,
  2026-10-05). It runs when someone joins.
- **The task:** a real `good first issue`, taken to a PR.
- **PASS:** green CI and all four gates met, with **zero questions to the maintainer that the docs should have
  answered**. Each question asked becomes a doc fix, and that step is re-tried.
- **When:** once in P4, before this spec closes, and again at each maintainer handover (D18).

### D18 — Handover, rename, transfer

`CONTRIBUTING.md` gains a "Maintainer handover" checklist:
1. **Transfer the repo** to [`nasaharvest`](https://github.com/nasaharvest) in **P5, after P4** (user,
   2026-10-05). "All links to the previous repository location are automatically redirected", and issues,
   PRs, the wiki, stars and watchers move with the repo. The transfer needs "permission to create repositories
   in the receiving organization", and the redirect is lost if a new repo is ever created at the old name.
2. Branch protection and CI move with the repo. The weekly failure goes to an auto-opened issue (D4).
3. The successor gets their own `rise` access through the platform admin. Shared aliases and blob roots
   are documented in `docs/reference/AZURE_INFRA.md`.
4. Private values (resource group, workspace names) are handed over **privately** and never committed.
5. Run the D17 dry run with the successor as the reader.

**Rename.** "fsd" (fetch-satdata) undersells a project that runs from imagery through datacubes, training data and a
model contract to inference at scale and served maps. **The new name is chosen before P5** (user: "decide later").
Shortlist, with PyPI checked on 2026-10-05: `sheaf` (install as `sheaf-eo`, because PyPI's `sheaf` is an
empty 0.0.0 placeholder; a sheaf is a harvest bundle, and in mathematics a structure that glues per-cell data
into one whole), `geosheaf`, `harvestac`, `cubecast`. Before P5, check GitHub and the remote-sensing
literature for clashes. The rename is one mechanical PR (`src/fsd` → `src/<name>`, the CLI, the docs), with
no back-compat shim (habit 14; SemVer §4).

## 4. Phases

| Phase | What | Who |
|---|---|---|
| **P0 — switch-over** | ~~Push the 8 local commits on `main`~~ · ~~Open tracking issue~~ (both done 2026-10-05: #102) | user |
| **Spec PR** | This spec + ADR 0033 (D7) merge as the first PR | Claude drafts; user signs off and merges |
| **P1 — scaffolding** | `.github/workflows/ci.yml` (+ the weekly run with its auto-issue) · `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md` (gates, review checklist, maintainer tasks, handover, "TODO #NN = issue #NN") · PR template, `release.yml`, labels · spec/ADR templates · the duplicate-number check · the `issue:` header check for specs ≥ 102 · `docs_kwarg_sweep.py` → `scripts/`. **Then the user enables branch protection** with the CI check required | implementation session; Opus review; user merges |
| **P2 — code guards** | Golden files (D11) · `alias=None` (D12) · path fixes + the identifier guard (D13) · NOTICE fixes (D14) | same |
| **P3 — doc migration** | Tag `docs-archive-2026` · delete and disperse (D8) · knowledge moves in (D10) · freeze the indexes (D7) · `CONTEXT.md` terminology · **the user applies the workspace `CLAUDE.md`/memory rewrite (§8)** | same |
| **P4 — dry run** | D17 (b), then doc fixes; this spec's issue closes; release `v0.1.0` | fresh agent session; user watches |
| **P5 — rename + transfer** | Choose the name → rename PR → transfer to `nasaharvest` (D18) | user, plus an implementation session |

CI must exist before branch protection can require it. The guards land before the docs move, so the
migration PRs get checked by them.

## 5. How to verify (acceptance criteria)

| AC | Phase | Check |
|---|---|---|
| AC1 | P1 | A PR from a branch triggers `ci.yml`; with every extra installed, the pytest skips not caused by `network` are ≤ 5 (each one named in the PR) |
| AC2 | P1 | Adding a second `specs/59-*.md` or a second `docs/adr/0033-*.md` on a branch makes CI fail with a message naming both files; a spec numbered ≥ 102 with no `issue:` header fails `test_docs.py` |
| AC3 | P1 | `ci.yml` has a weekly `schedule:` trigger whose failure step opens an issue (checked by reading the workflow; the first real fire is noted in the issue) |
| AC4 | P1 | `AGENTS.md` exists, `CLAUDE.md` is exactly `@AGENTS.md`, and a fresh Claude Code session in `fsd/` with no workspace `CLAUDE.md` above it shows that `AGENTS.md` was loaded |
| AC5 | P1 | `CONTRIBUTING.md` fits on one screen for the contributor part (≤ ~80 lines before the "Maintainer" sections) and states the four gates, inbound=outbound, and "TODO #NN = issue #NN" |
| AC6 | P1 | After the user enables protection: a direct push to `main` is rejected, and a PR with red CI can't be merged |
| AC7 | P2 | Changing one field written into `bundle.json` without bumping makes Test A fail with the "bump + add v3" message; bumping without adding the golden file fails too; v1 and v2 golden files load |
| AC8 | P2 | `ensure_environment(...)` without `alias=` leaves `_aliases.json` untouched (unit test with the existing fakes) |
| AC9 | P2 | The identifier guard fails on a planted absolute home path in `benchmarks/` and passes on the cleaned tree |
| AC10 | P2 | Both `NOTICE` files name CC BY 4.0 and the EuroCrops citation |
| AC11 | P3 | Tag `docs-archive-2026` exists and points at the last commit before deletion; every file listed in D8 as deleted is gone on `main`, and the living docs link to none of them (`test_docs.py` green) |
| AC12 | P3 | `specs/README.md` lists 00–59 and says "frozen"; `docs/adr/README.md` points at the listing; ADR 0033 (landed with this spec) is the only record superseding 0023's index half, and 0023 itself is unedited |
| AC13 | P3 | Every memory note D10 names has its destination line or issue, and the issue for the label-polygon bug exists |
| AC14 | P4 | The D17 (b) dry run: the PR reaches green CI; the questions asked and the doc fixes made are listed in #102 before it closes |
| AC15 | P4 | Release `v0.1.0` exists with auto-generated notes grouped by label |

## 6. Out of scope

- The teammate dry run (D17 a) until someone is available. Choosing the new name (deferred to before P5).
- A code of conduct, a CLA, a security policy, issue triage rotas (D1).
- A Python version matrix, a lockfile, Dependabot (D4).
- Renaming ADRs `0001`–`0032` (D6). Regenerating any index (D7).
- Spec 59 open item **B** (`data/imagery` vs `tests/outputs/imagery` in the docs) and #101: next in the queue,
  and the first work done under the new process.

## 7. Prior art

Each decision, labelled **standard** or **homemade**:

| Decision | Label | Precedent |
|---|---|---|
| D3 PRs + protected `main`, no self-approval | standard | GitHub branch protection; "Pull request authors cannot approve their own pull requests" |
| D4 weekly run that opens an issue | standard | xarray's `upstream-dev-ci.yaml`: a scheduled run against upstream versions that opens an issue via `scientific-python/issue-from-pytest-log-action` |
| D5 spec number = tracking issue | standard | Kubernetes KEPs: "KEPs are now prefixed with their associated tracking issue number" |
| D5 short template | standard | A trimmed PEP 1 shape (Motivation / Specification / Rationale / Rejected Ideas) |
| D6 sequential ADRs + duplicate check | standard | Nygard (2011): "numbered sequentially and monotonically. Numbers will not be reused." The CI check enforces "not reused" across branches. The `<spec>-<n>` alternative was homemade (no pre-2022 precedent found) and was dropped at sign-off (Q1) |
| D7 status in the tracking issue | standard | KEP: the tracking issue is "where the current state of the KEP is being updated" |
| D7 no index, the directory is the index | standard-ish | Nygard keeps ADRs as numbered files in one directory and describes no index |
| D8 PR titles → release notes | standard | GitHub automatically generated release notes + `.github/release.yml` |
| D8 one archive tag for deleted docs | homemade (low risk) | Git tags are the standard way to name a point in history; using one as a doc archive is our own use |
| D9 `AGENTS.md` | standard | agents.md: "a dedicated, predictable place to provide the context and instructions to help AI coding agents" (format post-2022; there is no pre-2022 equivalent for agent instructions, the nearest being `CONTRIBUTING.md`) |
| D11 golden files | standard | Golden-file tests (Go `testdata/*.golden` with an `-update` flag; Hashimoto, "Advanced Testing with Go", GopherCon 2017) |
| D12 promote shared pointers from `main` only | standard | Kubernetes: "avoid using the `:latest` tag … harder to track which version … more difficult to roll back"; pin a version or digest, and move mutable pointers deliberately |
| D14 mixed code/data licences per folder | standard | REUSE (FSFE): per-file `.license` sidecars or `REUSE.toml`; GitHub ToS §D.6 inbound=outbound |
| D15 0.y.z releases | standard | SemVer §4 |
| D17 dry run | standard | The Joel Test #12, "hallway usability testing" (2000) |
| D18 transfer | standard | GitHub repository transfer (redirects; issues and PRs move) |

**Alternatives rejected, with their precedents:** Rust RFC rename-to-PR-number (`0000-` placeholder renamed
once the PR opens): uniqueness as good as D5, but a rename step on every spec. towncrier news fragments
(`<issue>.<type>` files, which avoid changelog conflicts): GitHub release notes need no file at all. PEP-style
editor-assigned numbers: they need an editor.

## 8. Contract changes outside the repo (the user applies these)

| Where | Change |
|---|---|
| workspace `CLAUDE.md` "Git & branches → land finished worktrees" | → "open a PR; the maintainer merges; prune the worktree after the merge". Pushing a feature branch + a draft PR becomes routine |
| workspace `CLAUDE.md` "Commit only when asked / push only when asked" | Commits on a feature branch + push + draft PR are routine; merging into `main` stays the user's |
| workspace `CLAUDE.md` top line "Read `fsd/PROGRESS.md` first" + the handoff protocol's "flush to PROGRESS.md" | → "read `gh pr list` + the pinned Order-of-work issue"; handoff state goes in the draft PR description |
| workspace `CLAUDE.md` "Keep the living docs current: DROPPED/CHANGES/RECIPES" | → the D8 living set; no RECIPES rule |
| workspace `CLAUDE.md` web-search rule | The standing permission also covers prior-art searches (D9 rule 2) |
| workspace `CLAUDE.md` stale lines | "specs 00..17", "Azure Batch runner" (ADR 0005 says AML), the test-archive numbers, `.[dev]` → `.[dev,local]`, the test ROIs, `tests/manual/` → `runbooks/`, the obsolete notebook-exclusion rule, the commit trailer naming a fixed model version |
| workspace `CLAUDE.md` content that moves into the repo | Terminology → `CONTEXT.md`; code conventions → `AGENTS.md`; "Claude never runs pipelines" → `AGENTS.md` rule 1 (relaxed), with the strict parts kept private |
| memory | Rewrite `worktree-merge-prune-practice` (PR flow) and `fsd-status` (no PROGRESS hook); retire the notes D10 moves into the repo, leaving a pointer |

## 9. Questions at sign-off — ALL RESOLVED (user, 2026-10-05)

- **Q1 — D6 was homemade.** → **Switch to sequential numbers plus the duplicate-number CI check.** D6 was rewritten
  to match.
- **Q2 — the `issue:` header key.** → **Yes, it is required for specs ≥ 102.** P1 adds the check to
  `test_docs.py` (AC2).

## 10. Sources (per-source credit)

- **GitHub Docs, "Events that trigger workflows" (`schedule`)**: scheduled runs use "the latest commit on the
  default branch"; notifications go to "the user who last modified the cron syntax" (why D4 alerts through an
  issue); disabled after 60 days without activity in public repos (D4's caveat).
  <https://docs.github.com/en/actions/writing-workflows/choosing-when-your-workflow-runs/events-that-trigger-workflows>
- **GitHub Docs, "Approving a pull request with required reviews"**: "Pull request authors cannot approve their
  own pull requests" (why D3 sets no required-approval count).
  <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/reviewing-changes-in-pull-requests/approving-a-pull-request-with-required-reviews>
- **GitHub Docs, "Automatically generated release notes"**: `.github/release.yml` with `changelog.categories[*].labels`
  and `exclude` (D8, D15). <https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes>
- **GitHub Docs, "Using secrets in GitHub Actions"**: fork PRs get no secrets except `GITHUB_TOKEN` (D12.1).
  <https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions>
- **GitHub Docs, "Transferring a repository"**: redirects; issues, PRs, wiki, stars and watchers move; requires
  repo-create permission in the org; the redirect is lost if the old name is reused (D18).
  <https://docs.github.com/en/repositories/creating-and-managing-repositories/transferring-a-repository>
- **GitHub Terms of Service §D.6**: inbound=outbound (D14, no CLA/DCO).
  <https://docs.github.com/en/site-policy/github-terms/github-terms-of-service>
- **Kubernetes Enhancements, `keps/README.md`**: KEP number = tracking issue, and the issue is where the current
  state lives (D5, D7). <https://github.com/kubernetes/enhancements/blob/master/keps/README.md>
- **Rust RFCs README**: the `0000-` placeholder renamed to the PR number (a rejected alternative to D5).
  <https://github.com/rust-lang/rfcs/blob/master/README.md>
- **PEP 1**: the PEP section shape the short template trims (D5); editor-assigned numbers (rejected).
  <https://peps.python.org/pep-0001/>
- **Michael Nygard, "Documenting Architecture Decisions" (2011-11-15)**: sequential, monotonic, never-reused
  numbers; superseded records are kept and marked (D6's numbering after Q1; immutability in D6; ADR 0033 supersedes 0023 in part).
  <https://www.cognitect.com/blog/2011/11/15/documenting-architecture-decisions>
- **xarray `upstream-dev-ci.yaml`**: a scheduled upstream-dependency run that opens an issue on failure (D4).
  <https://github.com/pydata/xarray/blob/main/.github/workflows/upstream-dev-ci.yaml>
- **towncrier tutorial**: per-change fragment files named by issue + type avoid changelog conflicts (a rejected
  alternative to D8's release notes). <https://towncrier.readthedocs.io/en/stable/tutorial.html>
- **agents.md**: AGENTS.md at the repo root; the nearest file wins; read by Codex, Jules, Copilot, Cursor and
  others; suggested contents (D9). <https://agents.md/>
- **Claude Code docs, "How Claude remembers your project"**: AGENTS.md is read only when no CLAUDE.md
  exists in or above the working directory; a CLAUDE.md that imports `@AGENTS.md` includes it (why D9's
  `fsd/CLAUDE.md` is required). <https://code.claude.com/docs/en/memory>
- **Golden files: Mitchell Hashimoto, "Advanced Testing with Go" (GopherCon 2017)**, as summarised in
  [Go Time #83](https://changelog.com/gotime/83) and
  [Eli Bendersky, "File-driven testing in Go"](https://eli.thegreenplace.net/2022/file-driven-testing-in-go/):
  expected output stored in a file, compared on every run, regenerated deliberately (D11).
- **Kubernetes docs, "Images"**: avoid `:latest` in production; pin a version or digest (D12.3).
  <https://kubernetes.io/docs/concepts/containers/images/>
- **REUSE FAQ (FSFE)**: `.license` sidecars and `REUSE.toml` for files that can't carry headers (D14).
  <https://reuse.software/faq/>
- **SemVer 2.0.0 §4**: 0.y.z means anything may change (D15, D18's no-shim rename). <https://semver.org/>
- **Joel Spolsky, "The Joel Test" (2000-08-09), #12 hallway usability testing**: the shape of D17.
  <https://www.joelonsoftware.com/2000/08/09/the-joel-test-12-steps-to-better-code/>
- **EuroCrops, Zenodo record 7851838**: CC BY 4.0 + the suggested citation (D14).
  <https://zenodo.org/records/7851838>; **EuroCrops paper (arXiv 2302.10202)**: the Austrian AMA source data is
  CC BY 4.0. <https://arxiv.org/pdf/2302.10202>
- **In-repo evidence**: the 479-commit habit sweep (2026-10-03, Explore agent) and the counts in §1, all
  re-measurable with `git log --since=2026-06-01 --oneline -- <file> | wc -l`.

## Amendment A1 — gate 3 by a reviewer subagent (2026-10-05)

**Status:** requested by the user 2026-10-05 ("set up option 1": a Sonnet implementation session gets an
Opus review without the user passing messages between sessions). Signed off when the user merges the PR
that adds it.

**Problem.** D2 gate 3 says that for the user + Claude, the reviewer is "a different session from the one
that wrote the change". With the model split (implement on a cheaper model, review on a stronger one), that
means the user opens a second session, points it at the PR, and carries the result back. For every PR,
the user is the message bus between two agents.

**Decision.**
- **A1.1** A **reviewer subagent** that the implementing session spawns also satisfies gate 3, if it meets
  all of these: its definition is the committed `.claude/agents/pr-reviewer.md`; it runs on a stronger model
  (`model: opus`, `effort: high`); it starts with a fresh context and gets only the PR number; it has no edit
  tools (`disallowedTools: Edit, Write, NotebookEdit, Agent`); and it posts its **own** PR comment, which is
  the review of record. D2's text is not rewritten (D5: amend, don't rewrite). Read its gate 3 as "a
  different session, or the `pr-reviewer` subagent (A1)".
- **A1.2** The flow lives in `AGENTS.md` ("Review without a relay"): open the PR, spawn `pr-reviewer`, fix or
  file each finding, push, spawn it again to check the fixes. **At most two rounds.** If a **fix in PR**
  finding is still open after the second round, the PR goes to the maintainer, so two agents cannot
  loop forever.
- **A1.3** A separate session started by the user still counts as before. The subagent is an option, not a
  requirement: D2's "the method is not enforced" stands. It is Claude Code-specific, so it lives only in
  `AGENTS.md`'s optional method, labelled as such. `CONTRIBUTING.md` (the human-facing gates) does not
  mention it, and a contributor working by hand or with another agent tool is unaffected.

**Prior art (D9).** This is homemade wiring around an established idea. **IEEE Std 1012** (Verification
and Validation; the 2012 and 2016 editions both predate 2022-11-30) defines *independent* V&V by three kinds
of independence: technical, managerial and financial. Measured against it:
- **Technical: partial.** The reviewer reads the diff itself, with fresh context and a different model. But
  it is the same model family as the author, so errors can be correlated.
- **Managerial: partial.** The author does not choose what gets checked: the review brief is fixed in the
  committed agent file, and the reviewer gets only a PR number. The author does start the review, though.
- **Financial: not applicable.**

So the subagent is a *peer* review in the D2 sense, not IV&V. That is all D2 ever asked for.
Gate 4 (a real run) and the maintainer's merge stay human, and they cover what correlated model errors
miss (CONTRIBUTING checklist line 1: "Real run, not just green tests"). What was searched: only the
definition of independent review (IEEE 1012, above). I did not search for, and do not claim, a pre-2022 rule
that an automated reviewer may stand in for the second person. That is why A1.3 keeps the human-session
path.

**How to verify.** The next PR whose implementing session uses the flow has a `pr-reviewer` comment
headed "Gate-3 review (pr-reviewer subagent, fresh context)". The maintainer sees findings fixed or filed
without having passed any message between sessions. `git check-ignore .claude/agents/pr-reviewer.md`
prints nothing (the file is tracked) and `.claude/worktrees/` is still ignored.

**Out of scope.** A CI-run reviewer (`claude-code-action` on PR open) and a hook that launches a headless
review. Both were weighed on 2026-10-05: the first needs an API key in repo secrets and per-run billing,
and the second backgrounds a process, which `AGENTS.md` rule 1 forbids.

**Outside the repo (the user applies this, as in §8).** Workspace `CLAUDE.md`, "four gates" bullet: "reviewed
by a non-author session" → "reviewed by a non-author session or the `pr-reviewer` subagent (spec 102 A1)".
"Model split & effort": "switch back to Opus for review" → "the Sonnet session spawns `pr-reviewer` (Opus)
for review".

**Sources (per-source credit).**
- **Claude Code docs, "Subagents"** (<https://code.claude.com/docs/en/sub-agents>): the frontmatter fields
  A1.1 uses (`model` accepts the `opus` alias; `effort` overrides the session level; `disallowedTools`
  removes tools; omitting `Agent` stops a subagent from spawning more). Also: project subagents in
  `.claude/agents/` are meant to be checked into version control, which is why `.gitignore` now tracks that
  directory.
- **NASA, "IV&V Overview"** (<https://www.nasa.gov/ivv-overview/>), restating IEEE Std 1012
  (<https://standards.ieee.org/ieee/1012/4021/>, paywalled): "IEEE defines independence in IV&V as three
  parameters: technical independence, managerial independence, and financial independence", plus each
  one's meaning (managerial: the IV&V effort sits in an organization separate from the implementers). The
  partial/partial/n-a mapping onto a subagent reviewer is this amendment's own judgement, not the source's.

## Amendment A2 — docs-only PRs run only the docs guards (2026-10-06)

**Status:** DRAFT, awaiting sign-off. Design in [#110](https://github.com/nikhilsrajan/fsd/issues/110);
ordered before P3 by the maintainer on 2026-10-06 (#102). Signed off when the user merges the PR that adds it.

**Problem.** Every PR runs the full suite (~5 min), including PRs that change only Markdown (e.g. #108).
"No `.py` changed" does not mean "nothing to check": `tests/test_docs.py` checks links, spec `issue:`
headers, ADR numbers and README `fsd.*(` calls; `tests/test_notebooks.py` holds the identifier guard over
the living docs; `scripts/docs_kwarg_sweep.py` checks doc calls against live signatures. A docs-only PR needs
those, not the other ~1000 tests. The obvious fix, `paths-ignore: ['**.md']` on the workflow, breaks merging:
a workflow skipped by a path filter leaves its required check **Pending**, so every docs PR would be blocked.

**Decision.** This changes what D2 gate 1 / D4 run for one class of PR; D4's text is not rewritten (D5).
- **A2.1** `ci.yml` keeps triggering on every PR. A first step in the `test` job lists the PR's changed files
  and sets `code=false` only if **every** one is docs: a `*.md` file anywhere, or a file named `LICENSE` or
  `NOTICE`. Anything else (`.github/`, `pyproject.toml`, `scripts/`, `runbooks/*.ipynb`, a new file type) sets
  `code=true`. It is an allowlist of what is safe to skip, so an unknown file runs everything.
- **A2.2** The required check keeps its name, `test`, and always runs install, ruff,
  `pytest tests/test_docs.py tests/test_notebooks.py` and `docs_kwarg_sweep.py`. The rest of the suite is a
  step with `if: … code == 'true'`. The job always runs, so the check never sits at Pending.
- **A2.3** `push` to `main`, the weekly `schedule` and `workflow_dispatch` always set `code=true`, as does a
  failure of the listing step itself. A misclassified PR is caught at merge or within a week.
- **A2.4** Listing method (CPython's, below): `git fetch origin "$GITHUB_BASE_REF" --depth=1`, then two-dot
  `git diff --name-only --no-renames "origin/$GITHUB_BASE_REF.."` against the merge commit GitHub checks out.
  Not three-dot: with a depth-1 fetch it fails with "no merge base". `--no-renames` because a rename lists only
  its new name, so `git mv src/x.py docs/x.md` would look docs-only.
- **A2.5** The classification is a short script (`scripts/ci_changed_paths.py`: file names on stdin, prints
  `true`/`false`) so it gets a table test, not an untested regex in YAML.
- **A2.6** The rule that keeps A2 safe: **a test that reads a Markdown file lives in `tests/test_docs.py` or
  `tests/test_notebooks.py`.** A comment in `ci.yml` and in each of those two files says so. (Checked
  2026-10-06: no other test reads a committed `.md`; `test_build_fixture.py` reads only a README it generates.)

**Prior art (D9).** Not homemade. **CPython, bpo-40548 (May 2020):** "Always run GitHub action, even on doc
PRs" (`4e363761fc`, GH-19981), then "skip jobs on doc only PRs" (`75d7257b20`, GH-19983). Its `build.yml`
(at `v3.10.0`) carries the reason ("`paths-ignore` is not used to skip documentation-only PRs, because it
prevents to mark a job as mandatory"), a `check_source` job that diffs and greps for any non-docs path, and the
two-dot vs three-dot lesson A2.4 adopts. The third-party alternative, `dorny/paths-filter` (v1.0.0, 2020-05-21),
also predates 2022-11-30; it is not used, to avoid one more pinned action for a ten-line check.

**How to verify.**
- `pytest tests/test_ci_changed_paths.py`: `docs/x.md`, `README.md`, `tests/manual/x.md`, `LICENSE` → `false`;
  `src/fsd/api.py`, `.github/workflows/ci.yml`, `pyproject.toml`, `runbooks/x.ipynb`, a mixed list, an empty
  list → `true`.
- The A2 PR itself edits `ci.yml`, so it runs the full suite (visible in its CI log).
- Real-run evidence (gate 4 for CI): the first docs-only PR after merge shows the full-suite step **skipped**,
  `test` green, the PR mergeable, and a run time under ~2 min. Its link goes in #110 before that issue closes.

**Out of scope.** Making the full suite faster (#109). Splitting into several required checks. Skipping ruff
or the install on docs PRs (the sweep imports fsd).

**Sources (per-source credit).**
- **GitHub docs, "Troubleshooting required status checks"**
  (<https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/troubleshooting-required-status-checks>):
  a workflow skipped by path filtering leaves its checks "in a 'Pending' state and block merging", while a job
  skipped by a conditional "reports 'Success'". Why A2.1 keeps the trigger and A2.2 skips a step instead.
- **CPython `build.yml` at `v3.10.0`** (<https://github.com/python/cpython/blob/v3.10.0/.github/workflows/build.yml>)
  and commits `4e363761fc` / `75d7257b20` (bpo-40548): the whole pattern (diff inside the job, a docs-only
  allowlist, non-PR events always run) and the two-dot diff under a shallow fetch.
- **`dorny/paths-filter` releases** (<https://github.com/dorny/paths-filter/releases/tag/v1.0.0>): the date that
  shows the action route also predates the cutoff (weighed, not used).

## Amendment A3 — a lighter reviewer for small PRs, and a "needs diagnosis" label (2026-10-06)

**Status:** DRAFT, awaiting sign-off. Decided by the maintainer on 2026-10-06 (#102, row 1 of the P3 plan).
Signed off when the user merges the PR that adds it. Builds on A1.

**Problem.** A1 runs every review at `effort: high`, including a 30-line bug fix, so small PRs pay
large-PR review cost. Separately, a reviewer has two labels, **fix in PR** and **file as issue**. A finding
whose cause the reviewer could not pin down then reaches the implementing session (usually the cheaper
model) as an instruction to fix, and that session guesses.

**Decision.**
- **A3.1** A second agent file, `.claude/agents/pr-reviewer-small.md`: `model: opus`, `effort: medium`, the
  same `disallowedTools` as `pr-reviewer`. Two files because `effort` is set only in frontmatter; the Agent
  tool can override `model` per spawn, not `effort`. Its body runs the three checks below first, then says
  "read `.claude/agents/pr-reviewer.md` and follow its body from `## Inputs` on, with this header", so the
  review brief has one home (ADR 0025) and the two files cannot drift.
- **A3.2** A PR is **small** when all three hold:
  (a) **no contract change**: the linked issue is not a spec's tracking issue, and the PR adds or amends no
  spec or ADR;
  (b) **gate 4 does not apply**: no real data, cloud or pixels;
  (c) **≤ 400 changed lines** (additions + deletions) in total, and **≤ 200 under `src/`**:
  `gh pr view <N> --json files --jq '[.files[] | .additions + .deletions] | add'`, and the same with
  `select(.path | startswith("src/"))` before the sum. (`gh pr diff` has no `--stat` flag.)
  The implementing session checks these to choose the agent. `pr-reviewer-small` re-checks them before
  reviewing; if one fails it posts one comment headed `## Gate-3 not started: not small`, names the failed
  check, says "respawn `pr-reviewer`", and stops. That header does not start with `## Gate-3 review`, so it
  does not count as a round.
- **A3.3** Round 2 uses the same agent as round 1. Its re-check covers only the fix diff, so a PR whose fixes
  push it past the thresholds is still re-checked at the round-1 effort.
- **A3.4** A third finding label, **needs diagnosis**, in `pr-reviewer.md` (and so in both agents): the
  reviewer saw a symptom (a failing test, behaviour that contradicts the spec) but could not pin the cause.
  The implementing session does **not** attempt it; it stops and says so in the PR. The maintainer debugs and
  fixes it in a stronger-model session (for the user: Opus, `/effort high`), and the reviewer's round 2 checks
  that fix. The two-round cap (A1.2) is unchanged and counts the same rounds.
- **A3.5** Where it lives: `AGENTS.md` "How we work with agents", bullets "Choosing the reviewer" (which agent
  to spawn, the three checks) and "Acting on review findings" (the three labels). These bullets replace the
  section A1.2 called "Review without a relay" (see A4.7). `CONTRIBUTING.md` is unchanged: per A1.3 the
  human-facing gates do not name the subagents.

**Prior art (D9).** The thresholds are anchored on published review data; the rule that maps PR size to
reviewer effort is **homemade**. SmartBear's Cisco case study found defect-finding best when one review covers
200–400 lines of code, falling off above that. That data is about human reviewers; A3 assumes, without
evidence, that the same size band is a fair cut for an agent's effort level. Searched: SmartBear's study and
its best-practices summary. I found no pre-2022 source on sizing an *automated* reviewer's effort, and claim none.

**How to verify.**
- The next PR that passes A3.2 has a comment headed `## Gate-3 review (pr-reviewer-small subagent, fresh
  context)`. The round lookup in `pr-reviewer.md` (`startswith("## Gate-3 review (pr-reviewer")`) matches both
  agents' headers, so round 2 finds round 1 whichever agent wrote it.
- Spawning `pr-reviewer-small` on a PR over 400 changed lines yields only the `## Gate-3 not started: not small`
  comment.
- `git check-ignore .claude/agents/pr-reviewer-small.md` prints nothing (the file is tracked).

**Out of scope.** A Sonnet reviewer (A1.1 requires a stronger model than the implementer). Choosing the agent
by a hook or CI. Changing the two-round cap.

**Outside the repo (the user applies this, as in §8).** The workspace `CLAUDE.md` "Models and effort" bullet
says "Review follows `fsd/AGENTS.md` and the agent files in `fsd/.claude/agents/`" instead of restating the
flow, so A3 needs no edit there.

**Sources (per-source credit).**
- **SmartBear, "Best practices for peer code review"**
  (<https://smartbear.com/learn/code-review/best-practices-for-peer-code-review/>) and the **Cisco case study**
  (<https://static1.smartbear.co/support/media/resources/cc/book/code-review-cisco-case-study.pdf>): review
  200–400 LOC at a time, since defect-finding drops beyond that. The source of A3.2(c)'s 400; the 200 for `src/`
  is the band's lower edge, this amendment's own choice.
- **Claude Code docs, "Subagents"** (<https://code.claude.com/docs/en/sub-agents>, already cited in A1): `effort`
  is a frontmatter field; the Agent tool's per-spawn override covers `model` only. Why A3.1 needs two files.

## Amendment A4 — the planning session spawns a Sonnet implementer (2026-10-06)

**Status:** DRAFT, awaiting sign-off. Requested by the user 2026-10-06 ("fold A4 into #115"). Signed off when
the user merges the PR that adds it. Builds on A1 and A3.

**Problem.** The model split (Opus plans, Sonnet implements) currently costs a session boundary per PR: the
planning session writes a handoff, the user runs `/handoff`, starts a fresh session, sets model and effort,
points it at the PR, and later relays "needs diagnosis" findings back to Opus. A1 removed the user as the
message bus for review; implementation still has it. The rule that blocks the obvious fix, "Do not spawn
subagents just to write code" (`AGENTS.md` before this PR), assumed a subagent costs more than a session. Both start cold,
and a subagent's tool output stays out of the planner's context, which is the constraint that matters.

**Decision.** Optional method, like A1 (A1.3 applies: `CONTRIBUTING.md` is unchanged and nobody is required
to use it).
- **A4.1** A committed agent file `.claude/agents/implementer.md`: `model: sonnet`, `effort: medium`,
  `disallowedTools: Agent`. After the spec (or, for non-spec work, the issue) is signed off, the planning
  session may spawn it with the spec, the branch and the PR number instead of handing off to a new session.
  `AGENTS.md` drops "Do not spawn subagents just to write code"; its "Writing the code (Claude Code)" bullet
  says the `implementer` "is the only subagent that writes code".
- **A4.2** **Flat, not nested.** The planning session (the *orchestrator*) spawns the implementer, waits for it
  to return, then spawns `pr-reviewer` or `pr-reviewer-small` (A3.2) itself. The implementer has no `Agent`
  tool, so it cannot spawn a reviewer or anything else. The reviewer is still a fresh context on a stronger
  model than the code's author (A1.1).
- **A4.3** **Findings.** *Fix in PR*: the orchestrator continues the **same** implementer (`SendMessage`, which
  keeps its history) with the finding. If it cannot be resumed (for example, `/model` started a new
  orchestrator session, so its transcript is gone), the orchestrator spawns a fresh implementer with the
  branch, the PR number and the command that fetches the latest review (user, 2026-10-06, after the #115
  trial hit this). *File as issue*: the implementer files it. *Needs diagnosis*: the
  orchestrator diagnoses, writes the cause into the PR, and sends the fix to the implementer. Under A4 this
  replaces A3.4's "the maintainer switches to Opus". The orchestrator never writes the fix itself (that would be
  Opus paying for boilerplate). The two-round cap (A1.2) is unchanged.
- **A4.4** **Hard stops.** One orchestration run = one PR. It ends when gates 1–3 hold, or at A1.2's cap, and
  returns to the user. The orchestrator never merges and never starts the next PR (the user's merge is the
  checkpoint between phases). Gate 4 runs stay with the user (`AGENTS.md` rule 1).
- **A4.5** **The implementer's brief.** It works on the given branch in the orchestrator's worktree (the
  orchestrator does not edit while it runs); follows `AGENTS.md`; keeps the PR description current (done /
  next); commits with a `Co-Authored-By` line naming its own model; returns a summary of a few lines. Subagents
  cannot ask the user questions, so an ambiguity in the spec comes back to the orchestrator, which asks the user.
- **A4.6** The handoff protocol (spec 24 D6) still applies when the orchestrator's own context gets heavy.
- **A4.7** **Deviation in the trial (user, 2026-10-06).** At the user's request the orchestrator also rewrote
  `AGENTS.md` and the agent files in plain language, with no spec or amendment references (commit `6122b65`).
  That rewrite carries the reviewer choice and the three labels (A3.5), the implementer flow (A4.1), the A2
  gate-1 wording and the A2.6 test rule, and the `needs diagnosis` label in `pr-reviewer.md` (A3.4). It
  replaces the `AGENTS.md` section A1.2 names, "Review without a relay", with the "Choosing the reviewer" and
  "Acting on review findings" bullets; A1's rules are unchanged. The orchestrator also writes spec text
  answering review findings, since the spec is the planning session's work, not the implementer's.

**Prior art (D9).** **Homemade.** Searched: Baker's Chief Programmer Team (IBM Systems Journal 11(1), 1972,
doi:10.1147/sj.111.0056) as the nearest human model of one lead plus implementers. It is a team structure built
around one lead programmer, not a rule for handing implementation to a cheaper worker under a fixed brief, so it
is not cited as precedent. I found no pre-2022 source for orchestrating model-based coding agents, and claim none.

**How to verify.**
- #115 is the trial: after sign-off, the A2 + A3 + A4 implementation is written by the `implementer` agent. Its
  commits carry a `Co-Authored-By` line naming a Sonnet model, the orchestrator's commits touch only the spec,
  `.claude/agents/implementer.md`, and the files A4.7 lists, and the PR description records how many rounds it
  took.
- `git check-ignore .claude/agents/implementer.md` prints nothing (the file is tracked).
- `implementer.md`'s frontmatter has `disallowedTools` including `Agent`.

**Out of scope.** Nested agents (implementer → reviewer). Auto-merge. Chaining PRs or phases in one run. A
CI-run or hook-run implementer (A1's reasons apply).

**Outside the repo (the user applies this, as in §8).** Applied 2026-10-06. Workspace `CLAUDE.md`, "Models
and effort": "Normally that is the `implementer` agent, which the Opus session spawns after sign-off".
"Handoffs": "Going from plan to code needs no handoff when the `implementer` agent writes the code".

**Sources (per-source credit).**
- **Claude Code docs, "Subagents"** (<https://code.claude.com/docs/en/sub-agents>): `model` accepts the `sonnet`
  alias and `effort` overrides the session level (A4.1); "by default, a subagent can spawn subagents of its own,
  up to three layers below the main conversation", which is why A4.2 removes `Agent` rather than relying on a
  default; `SendMessage` resumes a subagent and "resumed subagents retain their full conversation history" (A4.3);
  `AskUserQuestion` is removed from every subagent (A4.5); "background subagents surface every permission prompt
  in your main session", so the user still answers permission prompts while the implementer runs.
- **F. T. Baker, "Chief programmer team management of production programming"**, IBM Systems Journal 11(1):56–73
  (1972), doi:10.1147/sj.111.0056: searched as a candidate precedent and not used (see Prior art).

## Amendment A5 — the first release is `v0.2.0` (2026-10-06)

**Status:** DRAFT, awaiting sign-off. Decided by the maintainer on 2026-10-06 (P4 planning, #102). Signed off
when the user merges the PR that adds it.

**Problem.** AC15 (§5), the P4 row of §4 and D15 say the first release is `v0.1.0`. A `v0.1.0` tag already exists (commit
`8d1b875`, 2026-09-04) with no GitHub release on it, so that name is taken by a commit that predates this spec.

**Decision.** AC15, the P4 row and D15's last sentence ("The first release is `v0.1.0`, after this spec lands")
read **`v0.2.0`**; their text is not rewritten (D5). The number follows the bump rule earlier in D15:
`breaking` bumps `y`, and PR #107 (labelled `breaking`, merged 2026-10-05) is the first such PR
after `v0.1.0`. The `v0.1.0` tag stays as it is, with no release added to it.

**Prior art (D9).** None needed: the number comes from D15, whose source is SemVer 2.0.0 §4 (cited in §10).

**How to verify.** After P4's doc fixes merge: `gh release view v0.2.0` shows auto-generated notes grouped by
label, and `git tag --contains 8d1b875` lists `v0.2.0`.

**Out of scope.** Releasing `v0.1.0` retroactively. Which PRs go into `v0.2.0` (everything on `main` at tag
time).

## Amendment A6 — what "the cloud" means for gate 4 (2026-10-07)

**Status:** DRAFT, awaiting sign-off. From the P4 dry run (D17 b, attempt 1, #102). Signed off when the user
merges the PR that adds it.

**Problem.** Gate 4 (D2) applies "when the change touches real data, the cloud, or pixels". In the dry run, the agent
fixed #122 (labelled `cloud`) in `fsd.workflows.shard`, `infer_shard` and `runners`, the code an Azure ML job runs,
and called gate 4 not applicable because its tests use fakes. That also made the PR eligible for
`pr-reviewer-small` (A3, condition b). The maintainer then ran the same code with real Snakemake to get the evidence.
"The cloud" needs a definition that does not depend on how the tests are written.

**Decision.** D2 gate 4's text is not rewritten (D5). Read "the cloud" as:
- **A6.1** Code that dispatches to, runs as, or reports from a cloud job: the AML runner (`runner="aml"`), the
  in-job entry points (`fsd.workflows.shard`, `fsd.workflows.infer_shard`), the local runner they call
  (`fsd.workflows.runners`), and the node image definitions. This holds even when unit tests cover the code
  with fakes. Code that a job merely imports (datacube building, sources) falls under "real data" or "pixels"
  as before.
- **A6.2** Evidence: a run of the changed code with the real tools and no fakes (for example a failing shard run
  through real Snakemake on a laptop) counts when the change does not depend on Azure itself. A change to
  dispatch, credentials, images or Azure storage needs a run on Azure. Without access, the contributor says so
  and the reviewer runs it (unchanged).
- **A6.3** The living docs say this where gate 4 is stated: `CONTRIBUTING.md` (the full rule), `AGENTS.md` and
  `pr-reviewer-small.md` (one line each, pointing to `CONTRIBUTING.md`).

**Prior art (D9).** Not a new mechanism: it defines a term in an existing gate. The evidence rule in A6.2 is what
the maintainer accepted on PR #132 (2026-10-07).

**How to verify.** `CONTRIBUTING.md`, `AGENTS.md` and `.claude/agents/pr-reviewer-small.md` each name the in-job
entry points under gate 4. The D17 re-try is not needed for this amendment: it was a `"rule"` observation, not a gap.

**Out of scope.** Automating the gate-4 decision (for example by path in CI). Changing which reviewer A3 picks,
beyond what this definition implies.

## Amendment A7 — five working rules for the implementer and the reviewer (2026-10-08)

**Status:** DRAFT, awaiting sign-off. Requested by the user 2026-10-08, after evaluating the ponytail prompt
pack (<https://github.com/DietrichGebert/ponytail>) for the implementer and `AGENTS.md`. The user chose all five
rules, with rule 5 in a stand-alone wording that does not mention code size. Signed off when the user merges the
PR that adds it. Builds on A1, A3 and A4.

**Problem.** The agent files say what to build and how to check it, but leave four gaps, and `AGENTS.md` has one
more:
- the implementer is told to "match the surrounding code", but not to find everything a change reaches (callers,
  tests, fixtures, docs) or to reuse what fsd already has before writing new code;
- nothing tells the implementer to add a test for new logic. It runs the existing tests;
- its return lists what it ran, not what it did not check, so the orchestrator cannot tell a check that passed from
  one that never ran;
- `pr-reviewer` reads "the functions the diff calls or changes", not the code that calls them, so a changed
  signature or behaviour can break an untouched file without anyone looking at it;
- "Write the smallest code that does the job" (`AGENTS.md`, code style) has no counterweight. An agent could read it
  as permission to drop input checks or error handling. fsd's worst failure is quiet: a swallowed download error
  leaves a datacube with gaps, and nodata is 0, so the gaps can pass for real values.

This is a quality change, not a token-cost change: each rule adds a line or two to a base context of ~30k tokens.

**Decision.** Optional method, like A1 and A4 (A1.3 applies), except A7.5, which is a code convention.
- **A7.1** `implementer.md`, a new Work step 1: "Before you edit, list every place the change must reach: its
  callers (grep them), tests, fixtures, docs and exports. Reuse before you write: an fsd helper first, then the
  standard library, then an installed dependency; write new code only when none fits. Never add a dependency to save
  a few lines."
- **A7.2** `implementer.md`, Work step 2 adds: "New logic with a branch, a loop or a parser gets a test that fails
  without it."
- **A7.3** `implementer.md`, Return adds "what you did not check" to the list.
- **A7.4** `pr-reviewer.md`, "What to check": "When the diff changes a function's signature, return value or
  behaviour, grep its callers and read the ones it could break: a change can break a file it does not touch." The
  Tests bullet starts: "Does risky new logic (a branch, a loop, a parser, a data write) have a test?"
  `pr-reviewer-small` follows `pr-reviewer.md` for the review, so it needs no edit.
- **A7.5** `AGENTS.md`, code conventions, a new bullet: "**Keep the safety checks.** Never drop checks on user input
  (ROIs, dates, config, paths), or an error whose removal would let data be skipped or lost silently." The
  reviewers already check every `AGENTS.md` rule ("Standards"), so they need no extra line. (Wording from review
  round 1, chosen by the user over the first draft, which could be read as if the error itself loses the data.)
- **A7.6** **Not adopted from ponytail:** installing it as a plugin (per-user, so contributors would not get it, and
  `/ponytail-review` would break "one reviewer per PR"); "the shortest working diff wins"; `ponytail:` comments for
  known limits (deferred work goes in an issue, and comments describe the code as it is now); its four-part finding
  format (the reviewer's labels and 300-word fold already do this job).

**Prior art (D9).** Each rule except A7.3, and A7.1's order of reuse, restates a practice documented before
2022-11-30 (sources below):
- A7.1's "reuse an fsd helper before you write" is DRY (*The Pragmatic Programmer*). The rest of A7.1's order
  (standard library, then an installed dependency) and "never add a dependency to save a few lines" are
  homemade, from ponytail;
- A7.2 and the Tests half of A7.4 are Google's review guide, "tests should be added in the same CL as the production
  code";
- the callers half of A7.4 is the same guide's "Context" section: look beyond the lines the review tool shows;
- A7.5 is "fail fast" (Shore, 2004) and PEP 20's "Errors should never pass silently".

A7.3 is **homemade**. Searched: PR-description and code-review practice for a "what was not tested" section. I found
only recent blog posts and issue threads, no pre-2022 practice, and claim none. Ponytail (2026) is the source of
the wording ideas, not prior art. Its own benchmark is author-run, in a setup unlike ours (Opus at default effort,
Bash off, `CLAUDE.md` off, scope chosen by the agent), and its results go both ways, so we cite no numbers and
expect no particular gain.

**How to verify.**
- `grep -n "list every place the change must reach" .claude/agents/implementer.md`,
  `grep -n "what you did not check" .claude/agents/implementer.md`,
  `grep -n "grep its callers" .claude/agents/pr-reviewer.md` and `grep -n "Keep the safety checks" AGENTS.md` each
  print one line.
- `tests/test_docs.py` and `tests/test_notebooks.py` stay green.
- The next PR written by the `implementer` shows A7.3 in its return ("did not check"), and its reviewer comment
  shows a callers check whenever a signature or behaviour changed. This is an observation, not a gate: one run cannot
  measure a change this small.

**Out of scope.** Installing ponytail or any other always-on prompt pack. A Haiku web-reading subagent (still open
from the 2026-10-08 handoff). Adding these rules to `CONTRIBUTING.md`'s review checklist for human reviewers. A
before/after benchmark.

**Sources (per-source credit).**
- **A. Hunt, D. Thomas, *The Pragmatic Programmer*, 20th anniversary ed. (2019), Tip 15, p. 31**, "DRY—Don't
  Repeat Yourself: Every piece of knowledge must have a single, unambiguous, authoritative representation within a
  system" (checked at <https://pragprog.com/tips/>): A7.1's first step, reuse an fsd helper before you write new
  code.
- **Google Engineering Practices, "What to look for in a code review"**
  (<https://google.github.io/eng-practices/review/reviewer/looking-for.html>, published 2019-09; date from the
  `google/eng-practices` repo history). The "Tests" section ("tests should be added in the same CL as the production
  code"; "Will the tests actually fail when the code is broken?") supports A7.2 and A7.4's Tests line. The
  "Context" section ("Sometimes you have to look at the whole file to be sure that the change actually makes
  sense") supports A7.4's callers check.
- **J. Shore, "Fail Fast", *IEEE Software*, 2004, p. 21** (<https://martinfowler.com/ieeeSoftware/failFast.pdf>):
  "when a problem occurs, it fails immediately and visibly". Its example is a missing config value that returns a
  default instead of raising, the case A7.5 names.
- **PEP 20, "The Zen of Python"** (<https://peps.python.org/pep-0020/>, created 2004-08-19): "Errors should never pass
  silently. Unless explicitly silenced." A7.5's error half, in the language fsd is written in.
- **ponytail, by GitHub user DietrichGebert** (<https://github.com/DietrichGebert/ponytail>, MIT): the "before
  you write" scope list, the reuse ladder, the test-for-logic rule, the "what you skipped or did not check"
  ending, and the review's "a change can break code it does not touch" (A7.1–A7.4, reworded in our own words).
  Its benchmark is `benchmarks/results/2026-10-07-agentic.md` (not used as evidence; see Prior art).

## Amendment A8 — the author marks a PR ready when gates 1–3 hold (2026-10-08)

**Status:** DRAFT, awaiting sign-off. Decided by the user on 2026-10-08 (item 3 of the P4 handoff on #102). Signed
off when the user merges the PR that adds it.

**Problem.** `AGENTS.md` says "open a draft PR" but not who marks it ready, or when. Both D17 dry runs ended with a
finished PR (CI green, review approved) still a draft, and the A7 PR (#139) did the same. A draft cannot be merged,
so the maintainer cannot tell a finished PR from one still in progress without reading each description.

**Decision.**
- **A8.1** The **author** marks the PR ready (`gh pr ready <N>`) when gates 1–3 hold: CI green, an issue linked,
  the review posted and every finding fixed or filed. In an agent run (A4), the author is the planning session,
  and the run ends after it marks the PR ready (A4.4 otherwise unchanged). The implementer and the reviewers do
  not change a PR's state; `implementer.md` and `pr-reviewer.md` say so.
- **A8.2** A gate that waits on the maintainer does not hold a PR back from ready. If the review (gate 3) or the real
  run (gate 4) waits on the maintainer, the author marks the PR ready once the other gates hold and the PR
  description says what waits. Draft means "still being worked on"; ready means "waiting on the maintainer" (to
  review, to run, or to merge). (From review round 1: without the gate-3 case, a contributor whose reviewer is the
  maintainer had no step that asked for the review.)
- **A8.3** The living docs say this where the PR flow is stated: `AGENTS.md` ("How a change reaches `main`", and
  the "Writing the code" bullet's end of a run) and `CONTRIBUTING.md` (the four-gates section). The `AGENTS.md`
  handoff bullet says "the PR description", not "the draft PR description".

**Prior art (D9).** Not homemade: the two states and who changes them are GitHub's (sources below). One deliberate
difference: GitHub says to mark a PR ready "when you're ready to get feedback", while A8 marks it ready after the
review when a reviewer agent is available, because fsd's gate 3 can be met inside the draft. When the review needs
the maintainer, A8.2 falls back to GitHub's meaning.

**How to verify.** `grep -n "gh pr ready" AGENTS.md` prints one line; `CONTRIBUTING.md`'s four-gates section says
to mark the PR ready when gates 1–3 hold. The next agent-run PR ends ready, not as a draft. That is an
observation, not a gate.

**Out of scope.** A `CODEOWNERS` file: fsd has none, so marking a PR ready requests no review automatically.
Marking ready automatically from CI. Who merges (unchanged: the maintainer).

**Outside the repo (the user applies this, as in §8).** Workspace `CLAUDE.md`, "Git": "Commit, push and open a
draft PR without asking" gains "and mark it ready when gates 1–3 hold".

**Sources (per-source credit).**
- **GitHub Docs, "Changing the stage of a pull request"**
  (<https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/changing-the-stage-of-a-pull-request>):
  "When you're ready to get feedback on your pull request, you can mark your draft pull request as ready for
  review", and "No one can merge the pull request until you mark the pull request as ready for review again" (the
  author marks it ready, A8.1; a draft blocks the merge, the Problem). Marking ready "will request reviews from any
  code owners" (why a `CODEOWNERS` file is named in Out of scope).
- **GitHub Blog, "Introducing draft pull requests"** (2019-02-14,
  <https://github.blog/news-insights/product-news/introducing-draft-pull-requests/>): "With draft pull requests,
  you can clearly tag when you're coding a work in progress" (draft = still being worked on, A8.2).

## Amendment A9 — gate 4 skips a proven comments-only diff; local-only paths may skip `fsd.storage` (2026-10-09)

**Status:** DRAFT, awaiting sign-off. Decided by the user on 2026-10-09: questions Q2 and Q4 of the refactor audit
(#145). Signed off when the user merges the PR that adds it. Builds on A6.

**Problem.**
- **Gate 4 and comments.** Since A6.1, gate 4 covers code that dispatches to, runs as or reports from a cloud job
  "even when unit tests cover the code with fakes". Read literally, that includes an edit to a comment or a
  docstring in such a file. The audit found comments there that are false: `runners.py` says it is "the only place
  in `fsd/` that imports `azure-ai-ml`", but `model/verify_image.py` imports it too. It also found about 110 lines
  of oversized docstrings in `runners.py` and `task.py`. Each fix changes no instruction Python runs, yet the rule
  asks for a real run. So false comments in the files that most need true ones stay false the longest.
- **Local-only paths.** `AGENTS.md` and ADR 0003 say all file I/O goes through `fsd.storage`, with one exception:
  raster pixel reads. The audit found about 15 sites that use the standard library on paths that can only ever be
  local:
  - the user config file `~/.config/fsd`;
  - adapter source trees read to package a bundle;
  - docker build contexts;
  - node-local scratch directories, the "scratch → `transfer` → `rename`" pattern that ADR 0003's own
    Consequences describe;
  - GDAL's temporary writes.

  The rule calls them violations. Routing them through `fs` would mean adding helpers (`fs` has no `mkdtemp` or
  `shutil` equivalent) and would change nothing. The sites the rule exists for, a path that may be a URL, are a
  different set. For example, `download_cli` writes `_result.json` under `--dst`, which may be `abfss://…`. Lumping
  the two together makes the real ones hard to see.

**Decision.**
- **A9.1** Gate 4 does not apply to a diff whose executable code is unchanged. Run the four checks on a clean
  checkout of the PR head, with `<base>` = `git merge-base origin/main HEAD` (the script reads the working tree).
  All four must hold, and the PR description pastes the commands and their output as the evidence:
  - `scripts/comment_astcheck.py <base>` exits 0 and prints no `NEW` line. It compares the docstring-stripped ASTs
    of every `.py` file under `src/fsd`.
  - `git diff --diff-filter=ADR --name-only <base> -- src/` prints nothing, so no file was added, deleted or
    renamed. The script reports a new file without failing, and never sees a deleted one.
  - `git diff --name-only <base> -- src/ ':(exclude)*.py'` prints nothing, so no Snakefile or other non-`.py` file
    under `src/` changed. The script reads only `.py` files, and such a file still follows gate 4 as before.
  - `grep -rn "__doc__" src/fsd` prints nothing (true on 2026-10-09). The script ignores docstrings, so the
    exemption holds only while no code reads one at runtime.

  (From review round 1: the third and fourth checks were in the prose, not in what a contributor pastes.)
- **A9.2** The `fsd.storage` rule's exception widens to: raster pixel **reads** through rasterio/GDAL, **and paths
  that are local by construction**. Those are:
  - the temporary and scratch directories the process creates itself, which covers GDAL's temporary writes;
  - the user config file;
  - a source tree that is read in order to package it, for example `bundle.save(code=[…])` and
    `ImageDefinition(build_context=…)`. It is local by construction even when a caller supplies it, because its
    consumer (the bundle writer, `docker build`) works only on local disk.

  Any other path that comes from a caller, a CLI flag or config may be a URL, so it always goes through
  `fsd.storage`. A GDAL **write** to a caller's path stays forbidden: it goes scratch → `transfer` (TODO #39,
  `raster/cog.py`). (From review round 1: the first draft said "pixel I/O", which would have allowed it.)
- **A9.3** ADR 0034 records A9.2 and supersedes ADR 0003 in part: its "one documented exception" clause. ADR 0003's
  Status line says so, which `docs/adr/README.md` asks for. Nothing else in ADR 0003 changes.
- **A9.4** The living docs say this where the rules are stated:
  - `AGENTS.md`: the gate 4 line and the "All file I/O goes through `fsd.storage`" bullet;
  - `CONTRIBUTING.md`: gate 4;
  - `docs/reference/code-comments.md`: "Reviewing a trim", which already requires the script on every comment pass.

  The `fsd/storage/fs.py` module docstring, which says "no other module in fsd opens files directly", is fixed in
  the audit's comment slice P2 (#145), not here, so that this PR stays docs-only.

**Prior art (D9).**
- **A9.1 restates an established practice.** The Black formatter proves a reformatted file safe the same way:
  "When run with `--safe` (the default), *Black* checks that the code before and after is semantically
  equivalent. This check is done by comparing the AST of the source with the AST of the target." `--safe` predates
  2022: Black 19.10b0's changelog says "`--safe` now works with Python 2 code". Black treats docstrings as a known
  exception to strict AST equality. `comment_astcheck.py` strips docstrings for the same reason, and A9.1's
  `__doc__` condition is what makes that safe. fsd already treats non-code diffs differently in CI (A2).
- **A9.2 is partly homemade.** fsspec, the library under `fsd.storage`, keeps its own process-local temporaries in
  the standard library's temp space, not in a remote-capable filesystem: `MMapCache` says "If None, a temporary
  file is created using tempfile.TemporaryFile()", and the caching filesystems take `cache_storage="TMP"`, which
  "is a temporary directory, and will be cleaned up by the OS when this process ends". The boundary wording,
  "local by construction" against "comes from a caller and may be a URL", is homemade. Searched: storage
  abstraction rules with an exception for temporary/scratch files, and fsspec temp-file practice. Nothing stated
  that boundary as a rule.

**How to verify.**
- `grep -n "comment_astcheck" AGENTS.md CONTRIBUTING.md` prints the new gate 4 lines.
- `grep -n "local by construction" AGENTS.md docs/adr/0034-*.md` prints the widened exception.
- `grep -n "0034" docs/adr/0003-*.md` prints ADR 0003's updated Status line.
- The audit's first comment slice that touches a gate-4 file cites A9.1 and pastes the script's output instead of
  a run. That is an observation, not a gate.

**Out of scope.**
- An AST check for Snakefiles or notebooks.
- Moving the sites that may be remote onto `fsd.storage` (audit findings A12 and E19, code slices of #145).
- Fixing `fs.py`'s docstring (slice P2).
- Any change to which code counts as "the cloud" (A6.1 is unchanged).

**Outside the repo.** None.

**Sources (per-source credit).**
- **Black, "The Black code style"** (<https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html>):
  the `--safe` AST-equivalence quotation, and docstrings as the documented exception (A9.1 prior art; why the
  `__doc__` condition exists).
- **Black changelog** (<https://black.readthedocs.io/en/stable/change_log.html>), 19.10b0: "`--safe` now works with
  Python 2 code" (the practice predates 2022-11-30).
- **fsspec API reference** (<https://filesystem-spec.readthedocs.io/en/latest/api.html>): `MMapCache`'s
  `tempfile.TemporaryFile()` default, and `cache_storage="TMP"` for `CachingFileSystem`,
  `WholeFileCacheFileSystem` and `SimpleCacheFileSystem` (A9.2's partial prior art).
- **fsd ADR 0003** (`docs/adr/0003-all-file-io-through-storage-seam.md`): the rule A9.2 narrows, and its
  Consequences paragraph, which already relies on node-local scratch.

## Amendment A10 — gate 3 counts a review only from an account with write access; code owners after the transfer (2026-10-13)

**Status:** DRAFT, awaiting sign-off. Decided by the user on 2026-10-13 ("1+2", in a review of the #145 working loop).
Signed off when the user merges the PR that adds it. Line numbers at `main` `6dfae80`. Builds on A1 and A8.

**Problem.**
- **The author can meet gate 3.** D2's gate 3 asks for a review by "someone other than the author", and A1 lets a
  reviewer subagent count. The subagent posts under the account of whoever spawned it. So a contributor who spawns
  `pr-reviewer` on their own PR gets a gate-3 comment posted by the PR's author, and nothing tells it apart from a
  comment they typed by hand. While the maintainer is the only author this does no harm; it does with the first
  contributor (D1: the repo is public).
- **Anyone can post a review.** Any GitHub account can comment on a public repo. `AGENTS.md:140` reads the review of
  record as the latest comment that starts with `## Gate-3 review`, and `pr-reviewer.md:26` finds its previous round
  the same way. Neither checks who posted the comment, so a stranger's comment with that header would be read as the
  review, or as round 1.
- **D3's "no required approvals" fits one maintainer only.** With two or more maintainers, GitHub's own approval check
  can carry gate 3, and a comment is the weaker of the two.

**Decision.**
- **A10.1** A gate-3 review counts only when the account that posted it has **write access** to the repo:
  `gh api 'repos/{owner}/{repo}/collaborators/<login>/permission' --jq .permission` prints `admin` or `write`. GitHub
  maps the maintain role to `write`. A reviewer subagent (A1) counts when the account that spawned it has write access.
  Any other review, including a contributor's own reviewer agent on their own PR, is a pre-review: welcome, but not
  gate 3. A contributor without write access marks the PR ready when the other gates hold and says that gate 3 waits
  on the maintainer (A8.2).
- **A10.2** To read the review of record, list the accounts that posted the PR's `## Gate-3 review` comments, keep
  those with write access (A10.1), and read the latest comment by one of them. The `pr-reviewer` round lookup keeps
  only comments posted by the account it runs as (`gh api user --jq .login`): its own previous round was posted by
  that account, so it needs no permission call.
- **A10.3** **Code owners after the transfer.** Once the repo is in the `nasaharvest` org (D18) **and** two or more
  maintainers are active:
  - add `.github/CODEOWNERS` with one line, `* @<org>/<maintainers-team>`. It names a team, not people, so a handover
    changes the team and not the file;
  - in branch protection, require 1 approval and turn on "Require review from Code Owners".

  Gate 3 is then that approval, and a reviewer agent's comment (A1) becomes a pre-review. Until both conditions hold,
  D3's "no required approvals" stands.
- **A10.4** The living docs say this where gate 3 is stated:
  - `CONTRIBUTING.md`: gate 3 (line 25); "Maintainer tasks", the branch-protection bullet (line 76), says when
    required approvals turn on; "Maintainer handover" gains the A10.3 step after step 2 (line 98).
  - `AGENTS.md`: gate 3 (line 43), the reviewer bullet's "counts for gate 3" (lines 126–127) and the command
    that reads a review (lines 139–140).
  - `.claude/agents/pr-reviewer.md`: the round lookup (line 26) and "the review of record" (lines 97–98).
    `pr-reviewer-small.md` reuses the lookup (line 14), so it does not change.
  - Not in this PR: `runbooks/102-p4-dry-run.ipynb` counts a `## Gate-3 review` comment from any account; issue #179
    tracks it (from review round 1).

**Prior art (D9).** Not homemade.
- A10.1 applies GitHub's rule for required reviews to fsd's comment-based gate. GitHub counts "approving reviews from
  people with write permissions in the repository or from a designated code owner". D3 turned required approvals off
  only because a solo maintainer cannot approve their own PRs; A10.1 keeps the comment and adopts GitHub's test of
  whose review counts. The Apache Software Foundation draws the same line for code changes: "PMC members have formally
  binding votes, but in general communities encourage all their members to vote, even if their votes are only
  advisory." That page is undated, and no archived copy could be fetched, so its age is not shown here.
- A10.3 is GitHub's code-owners feature, announced 2017-07-06: "With protected branches enabled, a code owner for each
  owned file has to leave a review before anyone can merge a pull request to that branch."

**How to verify.**
- `grep -n "write access" CONTRIBUTING.md AGENTS.md` prints the gate-3 lines; `grep -n "CODEOWNERS" CONTRIBUTING.md`
  prints the handover step and the branch-protection bullet.
- `grep -n "gh api user" .claude/agents/pr-reviewer.md` prints the round lookup.
- On PR #177, the A10.2 steps find one account, whose permission is `admin`, and read its round-2 review. For an
  account that is not a collaborator (`octocat`), the permission call prints `read` (checked 2026-10-13), so that
  account's comments would be dropped.
- A10.3 is checked after the transfer, not now: with the settings on, a PR without a code owner's approval cannot be
  merged.

**Out of scope.**
- A reviewer that runs in CI and posts as a bot, which would prove a review ran (weighed in A1's Out of scope).
- Proving that a comment by an account with write access came from a real `pr-reviewer` run: A10 trusts those
  accounts, as GitHub does.
- Adding `CODEOWNERS` or changing branch protection now. Who merges (unchanged).
- A docs test that pins the commands in `AGENTS.md`.

**Outside the repo.** None.

**Sources (per-source credit).**
- **GitHub Docs, "About protected branches"**
  (<https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches>):
  "You can require approving reviews from people with write permissions in the repository or from a designated code
  owner" (A10.1's rule); "If you do, any pull request that affects code with a code owner must be approved by that
  code owner before the pull request can be merged into the protected branch" (A10.3).
- **GitHub Docs, "About code owners"**
  (<https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners>):
  the file goes in `.github/`, the root or `docs/`; "the CODEOWNERS file must be on the base branch of the pull
  request"; "When the code owner is a team, that team must be visible and it must have write permissions"; "the last
  matching pattern takes the most precedence" (A10.3).
- **GitHub REST docs, "Get repository permissions for a user"**
  (<https://docs.github.com/en/rest/collaborators/collaborators#get-repository-permissions-for-a-user>): `permission`
  is one of admin, write, read or none, and "the maintain role is mapped to write and the triage role is mapped to
  read" (A10.1's `admin` or `write`).
- **GitHub Blog, "Introducing code owners"** (2017-07-06,
  <https://github.blog/news-insights/product-news/introducing-code-owners/>): the quotation in Prior art (A10.3 predates
  2022-11-30).
- **Apache Software Foundation, "Voting"** (<https://www.apache.org/foundation/voting.html>, undated): binding votes
  against advisory ones (A10.1's prior art).

## Amendment A11 — the new name is `rise-eo`, under a component convention for RISE (2026-10-14)

**Status:** DRAFT, awaiting sign-off. Decided by the user on 2026-10-14, in an interview that followed the rename
handoff. Signed off when the user merges the PR that adds it. Line numbers at `main` `cdcba01`. Amends D18
("Rename") and the P5 row of §4.

**Problem.**
- **D18 left the name open.** Its shortlist (`sheaf`, `geosheaf`, `harvestac`, `cubecast`) was picked as a
  stand-alone brand.
- **fsd is now part of RISE.** [`nasaharvest/rise`](https://github.com/nasaharvest/rise) (Rapid Intelligence System
  for the Environment) installs fsd through its `satellite` extra (`fsd @ git+…@v0.1.0`), next to RDM (reference
  data management), which lives inside the RISE repo. The new name has to work as a RISE component, and RISE has no
  rule yet for how a component is named or where its code lives. The maintainer also maintains RISE, so this
  amendment sets that rule.
- **The obvious names clash.** PyPI's `rise` (the Jupyter slideshow extension) ships a top-level `rise` package, and
  RISE's own `pyproject.toml` uses the same distribution name. PyPI's `rdm` (an unrelated tool) ships a top-level
  `rdm` package, the same import name as RISE's RDM.
- **"No shim" had no list.** D18 says the rename has no back-compat shim but does not say what carries the name
  outside `import fsd`: on-disk keys, saved bundles, the config folder, environment variables, STAC ids and AML
  names all do.

**Decision.**
- **A11.1 RISE's structure.**
  - The umbrella repo `nasaharvest/rise` holds no library code except the `rise` command (A11.4). It holds one extra
    per component, each pinned to an exact tag, the notebooks that use more than one component, and the docs on how
    the components fit together.
  - Each component is its own repo, `nasaharvest/rise-<role>`, with its own CI, releases, issues and license. Each
    picks its own visibility: `rise-eo` stays public (MIT); `rise-rdm` matches RISE's.
  - Components never import each other. They meet through files with a documented schema; for example, RDM's
    standardized field file is the label file `create_training_data` reads.
  - A part becomes a new component only when its maintainer, release cadence or dependencies differ from every
    existing one. Otherwise it is a module inside an existing component.
  - The umbrella's `requires-python` is the highest minimum of its components. Components declare lower bounds only
    on the dependencies they share.
  - A deployed service (for example `ukrop`, the database RDM's output feeds) is not a component; RISE links to it
    and never installs it.
- **A11.2 Component names.** Repo `nasaharvest/rise-<role>`, distribution `rise-<role>`, import `rise_<role>`,
  command `rise-<role>`, config folder `rise-<role>`, environment variables `RISE_<ROLE>_*`. Not a shared
  `rise.<role>` namespace: PyPI's `rise` ships a regular `rise` package, and one regular package in an environment
  breaks a namespace. Not bare role names: `rdm` already clashes.
- **A11.3 fsd becomes `rise-eo`** (EO: Earth observation). The name covers the whole path, from imagery through
  datacubes, training data and the model contract to inference and served maps; "fetch" was its first step only.
  The D18 shortlist is dropped: brand names read badly next to RISE's role names.
- **A11.4 The `rise` command.** The umbrella owns it; no component installs a script called `rise`.
  - `rise init` runs `rise-<role> init` for each installed component that has one. Today that is only `rise-eo`, so
    it does what `fsd init` does now.
  - `rise <role> …` passes through to `rise-<role> …`.
  - `rise-eo` ships the command `rise-eo` (`rise-eo init`, `rise-eo config`; today `fsd = "fsd.cli:main"`), so it
    also works without the umbrella.
  - The umbrella's distribution moves off the name `rise`, because pip treats two distributions with one name as one
    project, so installing either replaces the other. The new name is chosen in the RISE PR (A11.6, step 5).
- **A11.5 What the rename changes, with no shim.**

  | What carries the name | After the rename |
  |---|---|
  | package, distribution, command | `src/rise_eo`, `rise-eo`, `rise-eo` |
  | Python names that carry the name, e.g. the constant `FSD_DECLARATION_VERSION` (`declaration.py:52`) | renamed with the code (`RISE_EO_DECLARATION_VERSION`). The on-disk key it writes stays (row below) |
  | environment variables `FSD_CONFIG_DIR`, `FSD_WRITE_READ_LOG`, `FSD_WRITE_TIMINGS`, and `FSD_OUTPUTS_DIR` (used only in `demos/mini_mpc/` and `docs/howto/serve-xyz.md`) | `RISE_EO_CONFIG_DIR`, `RISE_EO_WRITE_READ_LOG`, `RISE_EO_WRITE_TIMINGS`, `RISE_EO_OUTPUTS_DIR` |
  | config folder `~/.config/fsd` (`%APPDATA%\fsd`, `$XDG_CONFIG_HOME/fsd`; `config.py:194-215`) | `rise-eo` in each place. The old folder is not read. When config is missing and the old file exists, the error prints the command that moves it |
  | on-disk keys `fsd_bundle_version` (`bundle.py:454`), `fsd_declaration_version` (`declaration.py:278`), the catalog footer and datacube key `fsd:declaration` (`declaration.py:53`), the deploy record's `fsd_version` (`api.py:2655`) | **unchanged.** Renaming a key is a format change (D11) with nothing in it for a user; the formats and their golden files stay as they are |
  | the image definition: the field `ImageDefinition(fsd=…)` (`definition.py:55`, documented in `docs/howto/build-the-images.md`) and its resolved `"fsd"` key, which goes into every image digest and into `image.json` in the image registry (`digest.py:184`; read at `aml/__init__.py:57`, `verify_image.py:185`) | field and key become `rise_eo`. Every digest changes, which the image rebuild below needs anyway. An image registered before the rename has no `rise_eo` key, so `verify_image` against it fails; it could not run the renamed code anyway |
  | wheel globs `fsd-*.whl` (`definition.py:85-146`, `verify_image.py:91-93`, `.gitignore:97`) | `rise_eo-*.whl`, not `rise-eo-*.whl`: a wheel's file name replaces each run of `-_.` in the distribution name with `_` |
  | saved bundles whose embedded adapter imports `fsd` (`bundle.py:6-18`) | `load` fails and says the bundle was saved with fsd ≤ v0.2, and to change `fsd` to `rise_eo` in the named file under `code/` and save again. The known bundle (DemoRF, in the registry) is saved again once; adapter files such as RISE's `notebooks/demo_model/my_adapter.py` change their import |
  | the "is this fsd itself" checks in bundling (`bundle.py:147`, `bundle.py:368`) | test for `rise_eo` |
  | STAC defaults: `catalog_id="fsd"` (`stac.py:618`), `fsd-inference` (`stac.py:407`, `api.py:1569`, `api.py:1614`) | `rise-eo`, `rise-eo-inference`. Outputs already written keep their ids |
  | AML job and experiment names `fsd-…` (`runners.py:729-1262`), temp-file prefixes | `rise-eo-…` |
  | AML environment names in the docs (`fsd-aml-env`, `fsd-infer-sklearn`) | `rise-eo-aml-env`, `rise-eo-infer-sklearn`. The images are rebuilt anyway: node commands become `python -m rise_eo.workflows.…`. Old environments stay registered |
  | point-in-time docs (specs, ADRs, findings, run-book records; ADR 0022) | unchanged; they keep "fsd" as written. Living docs, notebooks, examples and code change |

- **A11.6 Order** (replaces the P5 row of §4):
  1. #145's X4, X5 and C1 merge, and no other PR is open.
  2. The rename PR (A11.7).
  3. Release `v0.3.0`: the rename plus #145's X1–X5 and C1. Before tagging, the maintainer labels each of those PRs
     (D15); the merged #174, #177 and #180 have no label yet.
  4. Transfer the repo to `nasaharvest` (D18's handover checklist), then rename it to `rise-eo`. GitHub redirects old
     links after both, so RISE's current pin keeps installing.
  5. A RISE PR: pin `rise-eo@v0.3.0`, raise `requires-python` from `>=3.10` to `>=3.11` (A11.1; `rise-eo` needs
     3.11), rename the umbrella distribution, add the `rise` command, and write A11.1–A11.2 into RISE's docs.

  Moving RDM to `nasaharvest/rise-rdm` is RISE work and does not wait on any of these steps.
- **A11.7 The rename PR, for the session that does it.**
  - One mechanical PR: move `src/fsd` to `src/rise_eo`; `pyproject.toml` (name, script, package data); every import
    and call in `src/`, `tests/`, `scripts/` (including `docs_kwarg_sweep.py`'s import), `examples/`, `demos/`,
    notebooks, Snakefiles and living docs; `test_docs.py`'s `startswith("fsd")`; then the rows of A11.5.
  - Size: 301 tracked files mention `fsd` as a word (`git grep -lw fsd`; 134 of them `.md`). That count includes
    the point-in-time docs, which stay, and two binary fixtures under `tests/data/tutorial/`: `catalog.parquet` holds
    the unchanged key `fsd:declaration`, and `SCL.tif` matches only by chance in its compressed bytes. Neither changes. Easy to miss: `.github/workflows/ci.yml`, `.gitignore`, both `NOTICE` files and the
    agent files under `.claude/agents/`.
  - Reviewer: `pr-reviewer` (well over 200 lines under `src/`).
  - Gate 4 applies: rebuild the images with the renamed package, then one real AML run (download, training data,
    inference) and one local run, handed over as a run-book.

**Prior art (D9).** Not homemade; each piece copies Jupyter, which split into separate repos under one name long
before 2022-11-30.
- A11.1's umbrella is Jupyter's metapackage (`jupyter` 1.0.0 on PyPI, 2015-08-12): "an empty metapackage for user
  convenience, only expressing dependencies on multiple Jupyter packages".
- A11.2's prefix is Jupyter's: `jupyter-server` (on PyPI since 2018-07-28) imports as `jupyter_server`, from its own
  repo.
- A11.4's command is Jupyter's: since 2015, `jupyter_core`'s `jupyter` command "searches PATH for `jupyter-name`"
  and runs it, so `jupyter lab` runs `jupyter-lab`.
- Rejected: a namespace package, the Open Data Cube style (`odc-geo` imports as `odc.geo`, first on PyPI
  2022-01-25). It works when the project owns the top-level name; RISE does not (PyPA's rule, in Sources).
- A11.1's rule on when to split. eo-learn shipped as subpackages that one umbrella pinned to exact versions (1.0,
  2022), then merged back into one package in 1.5.0 (2023-09-06), a release that "focuses on making `eo-learn` much
  simpler to install". This is evidence of what a split by code layer costs, not a precedent: it postdates
  2022-11-30.

**Clash check for `rise-eo` (2026-10-14).**

| Where | Searched | Found |
|---|---|---|
| PyPI | JSON API for `rise-eo`, `rise-rdm`, `rise-satellite`, `rise-reference`, `rise-core`, `rise-imagery`, `rise-sat`, `rise-labels`, `rdm` | all free except `rdm` (an unrelated tool, 0.11.1) |
| GitHub | `gh search repos "<name> in:name"` for `rise-eo`, `rise_eo`, `rise-rdm`, `rise_rdm`, `rise-reference` | nothing in geospatial or remote sensing |
| Web and literature | `"RISE-EO" remote sensing OR "earth observation"`; `"rise_eo" OR "rise-eo" python package satellite` | no tool, dataset or paper named RISE-EO |

The umbrella's own name has namesakes in Earth observation: "RISE (Remote Imaging Support for Emergencies)", a
rapid-mapping tool from the Luxembourg Red Cross, WASDI and LIST, and the RISESAT microsatellite. They are noted
here, not acted on: RISE's name is not this spec's to choose, and `rise-eo` is distinct from both.

**How to verify.**
- This PR: the docs tests pass, and `grep -n "rise-eo" specs/102-contributor-readiness.md` prints A11.
- The rename PR:
  - `rise-eo --help` and `python -c "import rise_eo"` work in a fresh install;
  - `grep -rnw fsd src tests` prints only the A11.5 rows marked unchanged and the bundle error message, and
    `git grep -n FSD_` prints nothing outside point-in-time docs;
  - the golden tests pass with no new golden file;
  - loading a bundle saved by `v0.2.0` with an adapter that imports `fsd` fails with the A11.5 message;
  - the gate-4 runs pass.
- The RISE PR: `rise init` runs `rise-eo init`.

**Out of scope.** The rename itself (A11.7 plans it). The RISE PR and RDM's move to `rise-rdm` (RISE work). The
umbrella's new distribution name. Renaming RISE. Publishing to PyPI: free names keep it possible, and PyPI refuses a
package that depends on a git URL, so the umbrella can be published only after its components are.

**Outside the repo.** In `nasaharvest/rise`:
- A11.6 step 5.
- RDM's move:
  1. Extract `rdm/` with its history into `nasaharvest/rise-rdm`.
  2. Rename `src/rdm` to `src/rise_rdm`, add a `pyproject.toml` (dependencies from the umbrella's `rdm` extra) and a
     license, and tag `v0.1.0`.
  3. Point the umbrella's `rdm` extra at it.
  4. Replace the notebooks' `sys.path` lines with the installed package.

In the workspace: `CLAUDE.md` and memory say `rise-eo` once the rename merges.

**Sources (per-source credit).**
- **`nasaharvest/rise`** (read with `gh` on 2026-10-14): `README.md` and `docs/packaging.md` (FSD installed from its own
  repo through the `satellite` extra; RDM bundled under `rdm/src/rdm`), `pyproject.toml` (`name = "rise"`, the pin
  `fsd[…] @ git+…@v0.1.0`), `rdm/README.md` (RDM standardizes field data for `ukrop`). The Problem's account of RISE.
- **PyPI, `rise`** (<https://pypi.org/project/rise/>, 5.7.1) and its source
  [`damianavila/RISE`](https://github.com/damianavila/RISE) (`classic/rise/__init__.py`; `classic/setup.py` declares no
  scripts): the slideshow owns the `rise` distribution and a top-level `rise` package but no `rise` command. Supports
  A11.2's rejection of a namespace, A11.4's rename of the umbrella's distribution, and A11.4's free `rise` command.
- **PyPI, `rdm`** (<https://pypi.org/project/rdm/>, 0.11.1, "Regulatory Documentation Manager") and its source
  [`innolitics/rdm`](https://github.com/innolitics/rdm) (a top-level `rdm/` package): A11.2's "bare names clash".
- **PyPA, "Packaging namespace packages"**
  (<https://packaging.python.org/en/latest/guides/packaging-namespace-packages/>): "It is extremely important that every
  distribution that uses the namespace package omits the `__init__.py`". Supports A11.2's rejection.
- **PyPI, `jupyter`** (<https://pypi.org/project/jupyter/>, 1.0.0 uploaded 2015-08-12): "an empty metapackage for user
  convenience, only expressing dependencies on multiple Jupyter packages". A11.1's umbrella.
- **PyPI, `jupyter-server`** (<https://pypi.org/project/jupyter-server/>, first upload 2018-07-28) and
  [`jupyter-server/jupyter_server`](https://github.com/jupyter-server/jupyter_server): distribution `jupyter-server`,
  import `jupyter_server`, its own repo. A11.2's prefix.
- **`jupyter/jupyter_core`, `jupyter_core/command.py`**
  (<https://github.com/jupyter/jupyter_core/blob/main/jupyter_core/command.py>; subcommand commits from 2015-05-26):
  the `jupyter` command "searches PATH for `jupyter-name`". A11.4's dispatch.
- **`opendatacube/odc-geo`** (<https://github.com/opendatacube/odc-geo>, `pyproject.toml`; PyPI first upload
  2022-01-25): distribution `odc-geo`, import `odc.geo`. The namespace style A11.2 rejects.
- **eo-learn `CHANGELOG.md`** (<https://github.com/sentinel-hub/eo-learn/blob/master/CHANGELOG.md>): 1.0.x, "`eo-learn`
  abstract package from now on requires fixed versions of `eo-learn` subpackages"; 1.5.0 (2023-09-06), "`eo-learn` is
  now distributed as a single package", in a release that "focuses on making `eo-learn` much simpler to install".
  A11.1's rule on when to split.
- **PyPI's upload checks, `warehouse/forklift/metadata.py`**
  (<https://github.com/pypi/warehouse/blob/main/warehouse/forklift/metadata.py>): uploads are rejected with "Can't
  have direct dependency". Out of scope's note on PyPI.
- **PyPA, "Binary distribution format"**
  (<https://packaging.python.org/en/latest/specifications/binary-distribution-format/>): in a wheel's file name, "any
  run of `-_.` characters (HYPHEN-MINUS, LOW LINE and FULL STOP) should be replaced with `_`". A11.5's wheel-glob row.
- **EURISY, "RISE – Rapid Mapping to Transform Humanitarian Response"** (<https://www.eurisy.eu/?p=12508>): RISE is
  "Remote Imaging Support for Emergencies", built by the Luxembourg Red Cross, WASDI and LIST. The namesakes note.
- **Clash-check searches** (2026-10-14): the PyPI JSON API and `gh search repos`, as listed in the table, and the two
  web searches in the table. The first of those also returned a JpGU 2019 abstract on the RISESAT microsatellite
  (<https://confit.atlas.jp/guide/event/jpgu2019/subject/MSD42-P01/detail>). The namesakes note.
