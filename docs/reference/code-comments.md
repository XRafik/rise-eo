# Code comments — what earns a line in `src/`

rise-eo is spec-first, and that produced a habit: every decision got narrated where it landed. The
result is 28% prose in `src/rise_eo` (0.49 prose lines per code line) and 986 backward references
across 422 functions — about 2.3 `spec NN` / `Dn` tags each. Most of it reads as a changelog.

This page is the rule for what stays. It applies to `src/`; tests and run-books have their own
looser norms.

## The one distinction

> **Cut the changelog. Keep the hazard.**

A comment that says *what changed, when, and in which spec* is a worse version of `git log`,
the PR that made the change, and `specs/` — all of which are still there, and are the right place for it.

A comment that says *what breaks if you undo this* cannot be recovered from any of them, because
by the time someone undoes it they have already stopped looking.

Both of these were live in `create_datacube.py` on 2026-08-21. The first is noise; the second
caught a real bug during review the same day:

```python
# ❌ changelog — cut
# #65 FIXED (spec 47): this leg is no longer silent. Before this, a long run
# looked hung; measured 96s on the Austria e2e, 2026-08-21.

# ✅ hazard — keep
# `export_folderpath` is derived from the id, so two shapes sharing an id are two
# work-units writing the SAME folder, concurrently. On blob that collides on the
# block-blob commit; locally it silently overwrites. Refuse rather than race.
```

## Docstrings

Say what the function does and what its contract is. Lead with a single sentence that would let a
caller use it without reading the body.

- **One sentence is a complete docstring** when the name plus signature already carry the rest.
  `exists(path) -> bool` needs nothing.
- Document arguments only where the type does not: units, accepted values, what `None` means, who
  owns the file being written.
- Document what it **raises** when a caller must catch it.
- Do not restate the body. If the docstring is a prose transcript of the next ten lines, delete it.
- Do not open with the history of the signature.

**Length guide:** 1 line for a wrapper, up to ~8 for something with a real contract. Past ~15,
either the function does too much or the explanation belongs in `specs/`.

## Inline comments

Default to none. A comment earns its place when a reader who understands Python would still get it
wrong. In practice that is:

1. **An invariant** — "row order must follow the shapefile, so results are placed by index."
2. **A failure mode** — "a half-written cube must not count as present."
3. **A deliberate non-obvious choice** — "threads, not processes: this is latency-bound blob I/O
   and the GIL is released for the duration of each call."
4. **A trap in a dependency** — "`os.path.abspath` is only safe for local paths; on a URL it
   corrupts the scheme."
5. **A prohibition with a reason** — "never hand this path to `gpd.read_file`: GDAL has no
   `abfss://` driver and reports it as file-not-found."

Everything else goes. In particular, delete: section-divider banners that only name the next
function, comments that translate the line below into English, and any sentence whose subject is
the project rather than the code (*"this spec adds…"*, *"we used to…"*, *"as of the 2026-08-20
run…"*).

## References to specs, decisions and issues

Keep them — they are how "why is this here?" gets answered — but **at most one per function**, and
not scattered inline.

- **Module docstring:** one `Spec: specs/NN-name.md` line. This is the anchor for the whole file.
- **Function docstring:** at most one trailing reference, and only when the function encodes a
  decision a reader would otherwise second-guess. Put it on its own last line: `Spec: 50 D6.`
- **Inline:** only alongside a hazard comment, and only the bare id — `(#84)`, not
  `(spec 50 D9, filed as #84, blocked on the array layer)`.
- **Never** date-stamp a comment. `git blame` has the date and is never wrong.

Issue numbers stay first-class: `CLAUDE.md` records that ~448 `TODO #NN` references point into
GitHub Issues, and `TODO #47 == issue #47`. Trimming must not break that mapping — drop the prose
around a reference, not the reference itself.

## When the story genuinely matters

Sometimes the history *is* the point: an approach was tried, it failed, and it will look tempting
again. That is real knowledge and it must not be lost — but it does not belong inline.

Put it in the living doc that already exists for it, and leave a pointer of at most two lines:

| the story | where it goes |
|---|---|
| a capability that was removed | a `LIMITATIONS.md` row (the *why* lives in its ADR or spec) |
| behaviour kept but changed | the PR title (it becomes the release-note line) |
| an approach evaluated and rejected | the spec's *Alternatives considered*, or `ARCHITECTURE.md` §4 if it was built and reverted |
| a command or script worth reusing | a `docs/howto/` page, or `scripts/` with a docstring |
| deferred work | a GitHub issue |

```python
# ❌ 15 lines in fs.py explaining a retry helper that was removed and why it never worked

# ✅
# No write retry here: see ARCHITECTURE.md §4, "the InvalidBlockList retry". Retrying a
# deterministic id collision turns a legible failure into an error storm.
```

## Reviewing a trim

A trim is correct when:

- every remaining comment answers "what breaks if I change this", not "what happened here";
- no reference was deleted that an issue or spec still depends on;
- the file's behaviour is untouched — a comment pass must never change code, and `pytest -q` plus
  `ruff check` are run to prove it;
- the diff is comments-only, so it can be reviewed by reading the removals alone.

Three tools (each docstring says how to run it):

- `scripts/comment_refcount.py src/rise_eo` — prose lines and backward references per file, densest first;
- `scripts/comment_parens.py [--apply] src/rise_eo` — removes reference-only parentheticals
  (`(spec 50)`, `(D4)`) from comments and docstrings; never one that holds a `#`;
- `scripts/comment_astcheck.py <git-ref> src/rise_eo` — proves the diff is comments-only by comparing
  docstring-stripped ASTs. **Run it on every comment pass**: in #85 it caught three "comment" edits
  that were inside a `raise ValueError(...)` and an argparse `help=`. Its output, with the three
  other checks in `CONTRIBUTING.md` gate 4, stands in for gate 4's real run on a cloud-job file.

Target after a pass: **~0.30 prose lines per code line**, down from 0.49. That is a guide, not a
quota — `storage/fs.py` is a thin seam over fsspec whose whole value is knowing which backend
traps it hides, and it will stay above the line on purpose.
