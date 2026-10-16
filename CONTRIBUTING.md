# Contributing to rise-eo

Code is MIT; by opening a PR you license your contribution under the repo's terms (GitHub ToS §D.6,
inbound = outbound). No CLA, no sign-off line. Agents: also read `AGENTS.md`.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,local,s3,notebooks,grid,azure,aml,mpc,titiler,serving]"
.venv/bin/python -m pytest -q && .venv/bin/ruff check src/ tests/
```

## Every change is a pull request that passes four gates

`main` is protected: no direct pushes. Work on a branch and open a PR from the template, as a draft
while you are still working. When gates 1–3 hold, mark it ready ("Ready for review" on GitHub): that tells
the maintainer it is waiting to be merged. If the review or the real run waits on the maintainer, mark it
ready once the other gates hold, and say what waits in the PR description.

1. **CI green**: ruff, the fast pytest suite, the guard tests, `scripts/docs_kwarg_sweep.py`. A PR that
   changes only Markdown files (or `LICENSE` / `NOTICE`) runs ruff, the docs guards and the sweep, not the rest.
2. **Linked issue.** Changing a convention, an on-disk format or the public API also needs a signed-off
   short spec (`specs/TEMPLATE.md`; its number is its tracking issue's number). Bug fixes, refactors, docs and
   a new collection that follows `docs/adding-a-source.md` need only the issue.
3. **Reviewed by someone other than the author, from an account with write access to the repo**, as a PR
   comment. A review by anyone else, including your own reviewer agent when you have no write access, is a
   welcome pre-review: mark the PR ready and say that gate 3 waits on the maintainer. Every finding is fixed
   in the PR or filed as an issue.
4. **Real-run evidence** when you touch real data, the cloud or pixels: paste the output or a screenshot
   (QGIS). "The cloud" includes code that dispatches, runs as or reports from a cloud job: the AML runner,
   `rise_eo.workflows.shard` / `infer_shard`, the `rise_eo.workflows.runners` they call, image definitions. That holds
   even when tests use fakes. A run with the real tools and no fakes (e.g. real Snakemake on a laptop) counts,
   unless the change depends on Azure itself. No archive or Azure access? Say so; the reviewer runs it.
   A comments-only diff needs no run when all four hold on a clean checkout of the PR head, with
   `<base>` = `git merge-base origin/main HEAD`; paste the four outputs instead (spec 102 A9.1):
   `scripts/comment_astcheck.py <base>` exits 0 with no `NEW` line;
   `git diff --diff-filter=ADR --name-only <base> -- src/` prints nothing;
   `git diff --name-only <base> -- src/ ':(exclude)*.py'` prints nothing;
   `grep -rn "__doc__" src/rise_eo` prints nothing.

The PR title becomes the release-note line. The maintainer labels and merges (merge commit).

## Conventions

- Specs and ADRs: copy `specs/TEMPLATE.md` / `docs/adr/TEMPLATE.md`. ADR numbers are sequential; if two
  PRs pick the same number, the one that merges second renumbers (CI fails on duplicates).
- Keep the living docs true in the same PR (`README.md`, `ARCHITECTURE.md`, `CONTEXT.md`,
  `LIMITATIONS.md`, `ROADMAP.md`, `docs/`). Specs, ADRs and run-books are point-in-time.
- `TODO #NN` in old text = GitHub issue #NN. File deferred work as an issue.
- Docs removed in spec 102 (the progress log and its archive, the changelog, recipes, the dropped,
  bug and TODO lists, finished run-books) are readable at tag `docs-archive-2026`.
- Call a MGRS tile (the ~110 km source granule) and a grid cell (the ~5 km ROI subdivision) by those
  names; never a bare "tile". See `CONTEXT.md`.

## Review checklist

Each line points at the incident that taught it.

- [ ] **Real run, not just green tests.** Synthetic fixtures encode today's assumptions (spec 50: a real run
  found stale `input.csv` rows adopted and a ~3600-call serial blob sweep after two review rounds).
- [ ] **Verify the primitive a spec cites.** A docstring or our own issue is not evidence (#74: "no `.part`
  here" missed that `fs.transfer` was already atomic).
- [ ] **Test the serialization boundary.** Shard CSVs and `input.csv` retype values (`"05.00"` becomes `5.0`);
  test the round trip, not the in-memory rows.
- [ ] **A new verb kwarg is forwarded on every runner.** A default hides a dropped kwarg (`19b5ad8`: the AML
  download dropped `collection=`, so an S1 request discovered S2 granules); add a forwarding test per hop.
- [ ] **Address per unit path.** Never hash a set; watch control files written once per run (spec 58 D13).
- [ ] **Doc call sites rot silently.** pytest never runs a notebook cell; `scripts/docs_kwarg_sweep.py` does
  (four dead `scl_mask_classes=` call sites after spec 58 P1).
- [ ] **Prior art.** If the mechanism is homemade, does the PR say what standard practice it replaced?
- [ ] **Spec vs. code.** Was the spec checked against rise-eo's own code, not only against its sources?
- [ ] **A lesson becomes a check.** If review caught a class of bug, did the PR add a test or guard for the class?

## Maintainer tasks

- **Merge** with a merge commit once the four gates hold; apply one label (`breaking`, `feature`, `fix`,
  `docs`, `internal`). After merging, fast-forward local `main` and delete the branch and worktree.
- **Branch protection** (Settings → Branches → `main`): require a pull request, require the `test` status
  check, **no required approvals** (authors cannot approve their own PRs), merge commits only, no direct pushes.
  Once the repo is in the `nasaharvest` org **and** two or more maintainers are active, add `.github/CODEOWNERS`
  with one line naming the maintainers team (`* @<org>/<maintainers-team>`), require 1 approval and turn on
  "Require review from Code Owners". Gate 3 is then that approval (spec 102 A10.3).
- **Release.** Tag `v0.y.z` when there is something worth shipping; `breaking` bumps `y`, everything else bumps `z`.
  Only the maintainer edits the `pyproject.toml` version, the one place it is written (`rise_eo.__version__` reads it
  from the installed package). First merge a PR that sets the new version. Then, on an up-to-date `main`, create
  the release with the tag read from `pyproject.toml`. The check stops a missed bump: `gh release create` would
  otherwise publish a release on an old tag that has none (`v0.1.0` is one).

  ```bash
  v="v$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
  if git ls-remote --exit-code --tags origin "refs/tags/$v" >/dev/null; then echo "$v exists: bump pyproject.toml first"
  else gh release create "$v" --target main --generate-notes; fi
  ```

  `.github/release.yml` groups the notes by label. CI runs on the new tag and fails if tag and version differ.
- **Weekly CI run** (Mondays, on `main`): a failure opens an issue. Fix it by pinning a range in `pyproject.toml`
  with a comment saying why. GitHub disables scheduled runs after 60 days without repository activity.
- **Shared cloud aliases** (`current`, `champion`, `demo-*`) move only from `main`, and only the maintainer
  moves them. Teammates with Azure access use a personal namespace (`dev-<user>`).
- **Sweep for private identifiers** before pushing prose about a real run (it has leaked four
  times). The CI guard catches the usual shapes; this catches your own concrete values. Keep them
  one per line in a file outside the repo, then scan only tracked files:
  `while read -r v; do git ls-files -z | xargs -0 grep -lF "$v"; done < <your-values-file>`.
  Replace a hit with a placeholder (`st<proj>`) and describe the value, never spell it, even when
  writing up the leak. A pushed leak stays in history unless it is rewritten.
- **Order of work:** one pinned issue ([#125](https://github.com/nikhilsrajan/fsd/issues/125)) holds
  the order; each piece of work keeps its own state in its PR description or issue.

## Maintainer handover

1. Transfer the repo to the `nasaharvest` org (needs repo-create permission there; old links redirect, and the
   redirect is lost if a repo is ever created at the old name).
2. Branch protection and CI move with the repo; check the weekly failure still opens an issue.
3. With two or more maintainers, add `.github/CODEOWNERS` and turn on the code-owner approval ("Maintainer
   tasks", branch protection).
4. The successor gets their own `rise` access through the platform admin; shared aliases and blob roots are
   in `docs/reference/AZURE_INFRA.md`.
5. Hand private values (resource group, workspace names) over privately; never commit them.
6. Dry run: a fresh agent session or the successor takes a `good first issue` to a green PR using only these docs.
   Every question it has to ask becomes a doc fix.
