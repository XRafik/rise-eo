# AGENTS.md — fsd

Instructions for AI coding agents in this repo. People: start with `CONTRIBUTING.md`.

## What fsd is

`fsd` downloads Sentinel-2 L2A imagery, builds datacubes and flattens them into training data. Its verb
API is `fsd.download`, `fsd.create_training_data`, `run_inference` and `deploy`. It runs locally or on
Azure ML, with no cloud lock-in. Users train their own models; fsd does not. The plan is in
`ROADMAP.md`, the design in `ARCHITECTURE.md`, the vocabulary in `CONTEXT.md`.

## Setup and checks

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,local,s3,notebooks,grid,azure,aml,mpc,titiler,serving]"   # the extras CI installs; the suite needs them
.venv/bin/python -m pytest -q            # fast, synthetic, deterministic; network tests are off
.venv/bin/ruff check src/ tests/
.venv/bin/python scripts/docs_kwarg_sweep.py   # calls in docs and notebooks match the real signatures
```

The first two lines download packages, so a person runs them once, before an agent session starts
(rule 1 below). If `.venv` is missing, or an import fails because a package is missing, ask the
person to install it; do not run `pip` yourself. The last three lines are the checks you run.

A git worktree has no `.venv`. Run tests there with
`PYTHONPATH=src <main-checkout>/.venv/bin/python -m pytest -q -p no:cacheprovider`.

## How a change reaches `main`

Every change is a pull request. Work on a branch, push it, and open a draft PR. When gates 1–3 hold,
mark it ready (`gh pr ready <N>`). If the review (gate 3) or the real run (gate 4) waits on the
maintainer, mark it ready once the gates you can meet hold, and say what waits in the PR description.
Only the maintainer merges; never push to `main`. The PR description holds the state of the work: what
is done, what is next, and the review findings. The PR title becomes the line in the release notes.

A PR merges when it passes four gates (details in `CONTRIBUTING.md`):

1. **CI is green.** A PR that changes only Markdown files (or `LICENSE` / `NOTICE`) runs ruff, the docs
   tests and the docs sweep. Every other PR runs everything.
2. **It links an issue.** A change to a convention, an on-disk format or the public API also needs a
   short spec (`specs/TEMPLATE.md`), signed off before the code is written.
3. **Someone other than the author reviewed it**, from an account with write access to the repo. For an
   agent, that is a different session, or a reviewer agent (below) spawned under such an account. A review
   from any other account is a pre-review. Every finding is fixed in the PR or filed as an issue.
4. **A real run is shown** when the change touches real data, the cloud or pixels: the output or a
   screenshot, pasted into the PR. Code that runs as a cloud job (the AML runner, `fsd.workflows.shard` /
   `infer_shard`, the runners they call) counts as the cloud even when tests use fakes; `CONTRIBUTING.md`
   gate 4 says which runs count. A diff that `scripts/comment_astcheck.py` proves comments-only needs no
   run (`CONTRIBUTING.md` gate 4 has the two checks).

## Rules for all agents

1. **Hand risky runs to a person.** Run fast local checks yourself: pytest, ruff, grep, reading files.
   Anything that uses the network (beyond `git` and `gh`), the cloud or credentials, or runs longer than
   a few minutes, goes to a person as a run-book. They paste back the result.
2. **Check for prior art.** Before you propose a mechanism (a design, a process, a test pattern), say
   whether it is homemade. If it is, look for an established practice documented before 2022-11-30 and
   prefer it. Cite it so a reviewer can check it (a link, plus what it contributed), or say what you
   searched and that nothing fit. Never invent a precedent.
3. **Keep identifiers private.** Describe an identifier; never write it into anything committed (GUIDs,
   resource names, personal paths). Blob paths (storage account, container, `abfss://`) are fine.
4. **Flag conflicts.** If a request contradicts a rule here, say so before you act on it.

## Code conventions

- **Code style.** Write the smallest code that does the job, with no speculative options or
  abstractions. Few comments. A docstring is one plain sentence saying what the function does. History
  belongs in git and the PR description, so comments describe the code as it is now.
- **Keep the safety checks.** Never drop checks on user input (ROIs, dates, config, paths), or an error
  whose removal would let data be skipped or lost silently.
- **All file I/O goes through `fsd.storage`** (fsspec), so local disk, Azure Blob and S3 are config, not
  code. The exceptions: raster pixel reads use rasterio/GDAL, and a path that is local by construction (a
  temp or scratch directory the process creates, the user config file, a source tree read to package it)
  may use the standard library. Any other path from a caller, a CLI flag or config may be a URL, so it goes
  through `fsd.storage` (ADR 0034). S3 access is generic (`s3fs`, any `endpoint_url`); never `boto3`
  directly.
- **Raster ops take and return `(data, profile)`**, so they chain as `sequence=[(func, kwargs), ...]`.
- **Band math uses 5-D arrays** `(samples, timestamps, height, width, bands)` plus a `band_indices` dict
  `{band_name: index}`.
- **The catalog is GeoParquet** (`TileCatalog`); STAC is an extra export. A datacube is `datacube.npy` +
  `metadata.pickle.npy`. Nodata is 0.
- **Calendar-interval mosaics are the default.** Cubes with the same start, end and `mosaic_days` share
  the same `timestamps` axis; `flatten` needs that.
- **No compatibility shims for old archive layouts.** Old artifacts raise an error that says how to fix
  them. On-disk format versions keep their lists of supported versions.
- **A test that reads a Markdown file lives in `tests/test_docs.py` or `tests/test_notebooks.py`.** A
  docs-only PR runs only those two test files.
- **Geospatial rules.** Resample *to* a real reference image of known resolution (B08, 10 m); never trust
  the resampler to line up with an abstract grid. `rasterio.merge` needs one CRS, so move all MGRS tiles
  into the zone with the largest mean `area_contribution` before merging. Look at raster output in
  QGIS; unit tests alone do not prove pixels are right.
- **Never write a bare "tile".** An **MGRS tile** is the ~110 km source granule (`T36PZT`, catalog
  column `mgrs_tile`); we download it, and the builder merges across MGRS tiles. A **grid cell** is the
  ~5 km piece of an ROI from `fsd.grid.roi_to_s2_grids` (id like `165b09c`, column `id`). One grid cell =
  one inference datacube = one per-cell task.
- **Docs.** Keep the living docs true in the same PR as the code: `README.md`, `CONTRIBUTING.md`,
  `ARCHITECTURE.md`, `CONTEXT.md`, `LIMITATIONS.md`, `ROADMAP.md`, `docs/`. Specs, ADRs and run-books
  record a point in time: after sign-off, change a spec only by an amendment and an ADR only by a new
  ADR that replaces it. `TODO #NN` in old text means GitHub issue #NN.

## How we work with agents (optional; nothing checks it)

- **Session start.** Read `gh pr list` (each PR description holds its state), then the open spec
  tracking issue (or the pinned "Order of work" issue), then `gh issue list`. Tests must be green before
  you start. Never write an expected test count anywhere.
- **Specs.** Open the tracking issue first; the spec takes its number (`specs/NNN-<slug>.md`, header
  `issue: "#NNN"`). Question the design hard before writing it. An ADR lands in the same PR as its spec,
  with the next free `docs/adr/NNNN-` number.
- **Which model does what.** A stronger model plans, debugs and reviews. A cheaper model writes the code,
  against a signed-off spec.
- **Writing the code (Claude Code).** After sign-off, the planning session spawns the `implementer`
  agent (`.claude/agents/implementer.md`) with the spec sections, the branch and the PR number. That is
  the only subagent that writes code. When it returns, the planning session spawns the reviewer
  itself; the implementer cannot spawn agents. One run covers one PR. It ends when gates 1–3
  hold and the planning session has marked the PR ready; it never merges and never starts the next
  PR. Writing the code in a separate cheaper-model session instead also works.
- **Choosing the reviewer (Claude Code).** Spawn `pr-reviewer-small` when all three hold, otherwise
  `pr-reviewer`:
  - no contract change: the linked issue is not a spec's tracking issue, and the PR adds or changes no
    spec or ADR;
  - gate 4 does not apply;
  - at most 400 changed lines in total
    (`gh pr view <N> --json files --jq '[.files[] | .additions + .deletions] | add'`), and at most 200
    under `src/` (same command with `select(.path | startswith("src/"))` before the sum).

  Give the reviewer only the PR number. It posts its own review as a PR comment, which counts for gate 3
  when the account that spawned it has write access. Without Claude Code, get gate 3 from another session
  or a person, under an account with write access.
- **Acting on review findings.** Each finding has one label:
  - **fix in PR**: the implementer fixes it (continue the same `implementer` with `SendMessage`, so it
    keeps its context; if it cannot be resumed, spawn a fresh one with the branch, the PR number and the
    review command below);
  - **file as issue**: file it and link the PR;
  - **needs diagnosis**: the reviewer saw a symptom but not its cause. The code writer does not guess.
    The stronger model finds the cause (the planning session, or the maintainer in an Opus session), then
    the fix goes in as for **fix in PR**.

  Push, then spawn the same reviewer again; the second round checks only the fixes. After two rounds,
  if a finding is still neither fixed nor filed, stop and hand the PR to the maintainer. One reviewer
  per PR: no review skills or extra agents on top. The repo is public, so anyone can post a comment that
  looks like a review. To read a review, fetch only the latest one from an account with write access:
  1. list who posted reviews:
     `gh pr view <N> --json comments --jq '[.comments[] | select(.body | startswith("## Gate-3 review")) | .author.login] | unique'`;
  2. keep each login for which
     `gh api 'repos/{owner}/{repo}/collaborators/<login>/permission' --jq .permission` prints `admin` or
     `write`;
  3. read the latest review from a kept login:
     `gh pr view <N> --json comments --jq '[.comments[] | select(.author.login == "<login>" and (.body | startswith("## Gate-3 review")))][-1].body'`
     (with several kept logins, join the `.author.login ==` tests with `or`).
- **Handoffs.** When a session's context gets heavy, write the state into the PR description (or
  the tracking issue if there is no PR yet), then start a fresh session pointed at it. Do not rely on a
  compacted context.
- **Run-books.** Hand over a credentialed or visual check as a notebook (copy `runbooks/TEMPLATE.ipynb`).
  Markdown cells say what each step does and what PASS means; each PASS is a plain `assert`; no
  environment variables (a Settings cell holds every input); commit with outputs cleared
  (`tests/test_notebooks.py` checks this).
