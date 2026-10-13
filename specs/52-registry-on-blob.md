---
status: current
summary: A registry version is published in place with a completion marker written last (D1), no directory rename -- fixing #88's infinite retry loop against a non-local backend. `_list_versions` becomes marker-aware with a legacy carve-out for pre-spec content (D2/D5; the carve-out is dropped by the 2026-10-13 amendment). `deploy`, `run_inference`, `verify_adapter` and `verify_image` gain `configure_storage` calls so a blob registry actually authenticates (D4, #86). Amends spec 51 D2; completes spec 51 AC12.
---

# Spec 52 — the registry on blob: `fsd.deploy` publishes to the cloud

**Status:** **SIGNED OFF (user, 2026-08-24)** — all four §7 questions resolved at their proposed
defaults. Ready to implement. · **Opened:** 2026-08-24 · **Amends:** spec 51 D2
**Closes:** [#88](https://github.com/nikhilsrajan/fsd/issues/88) — verified on real Azure
2026-08-25, closed. **[#86](https://github.com/nikhilsrajan/fsd/issues/86) was claimed and is NOT
closed** — unreachable on the path the run-book exercises; see §10.5 and
[#90](https://github.com/nikhilsrajan/fsd/issues/90). · **Completes:** spec 51 AC12

> **Rewritten 2026-08-24 after the user asked whether this was being over-engineered.** It was.
> The first draft reproduced Azure's native atomic directory rename in order to preserve a
> guarantee — race arbitration between two simultaneous publishers — that the user had already
> said does not occur in this team. Preserving it required a conditional-header protocol,
> continuation-token handling, file-vs-directory dispatch and a new Azure dependency, and the
> complexity grew each time either of us looked at it. This version drops the guarantee, and the
> complexity with it. §6 keeps the rejected design intact, so reinstating it later is a decision
> rather than a rediscovery.

---

## 1. The problem, in three sentences

A registry is a folder of models. `fsd.deploy` puts a bundle in it under a name and a version, so
a run can say `crop-rf@champion` instead of a path. **That folder cannot live on blob today**, so
it lives on the user's laptop and the registry is private to one machine.

Two things stop it, and only one of them is interesting.

**#88 — `registry.publish` never returns against a non-local backend.** Publishing writes the
version's files into `.staging-<uuid>/` and then **renames that directory** onto `v<N>`. Renaming
a directory is instant, so no reader can ever see a half-written model. Blob storage has no
directories to rename; fsspec substitutes copy-then-delete, the delete fails, and
`_write_new_version` reads every error from the rename as *"I lost a race, try `v<N+1>`"* — true
for `os.rename`, false here. The failure is deterministic, so the loop increments forever. No
timeout, no error, no output.

**#86 — `deploy` never authenticates.** `storage.azure.configure_storage` is the single function
that switches adlfs from anonymous to credentialed, and only `download`, `create_training_data`
and `flatten_training_data` call it. `deploy`, `run_inference`, `verify_adapter` and
`verify_image` do not. A blob registry would be read and written anonymously. Today's notebook
works only because an earlier verb in the same kernel set a *process-global* flag as a side
effect; a fresh kernel resuming straight to inference is already unauthenticated.

### How small this actually is

The registry calls `fs.rename` three times. **Only one of them is broken:**

| call site | renames | works on blob today? |
|---|---|---|
| `set_alias` (`registry.py:199`) | `.staging-<uuid>.json` → `_aliases.json` | **yes** — verified |
| `write_deploy_record` (`registry.py:305`) | `.staging-<uuid>.json` → `_deploy.json` | **yes** — verified |
| `_write_new_version` (`registry.py:394`) | `.staging-<uuid>/` → `v<N>/` | **no** — this is #88 |

A *file* rename is copy-then-delete of one object, which fsspec does correctly, including
overwriting an existing target (verified on `memory://`, 2026-08-24). `builder.py:365` renames a
temp sidecar file the same way and is likewise fine. **`registry.py:394` is the only directory
rename in the codebase**, and D1 removes it. `storage.fs.rename` is therefore **not changed by
this spec at all.**

---

## 2. Scope

**In:** `_write_new_version` publishes without a directory rename; `_list_versions` learns to
ignore an unfinished version; the retry loop stops misreading a hard failure as a race;
`configure_storage` is called by the four verbs that skip it; `deploy` accepts `storage="azure"`.

**Out:** `storage.fs.rename` (unchanged — §1); a lock service (spec 51 §5 stands); `set_alias` and
`write_deploy_record` (already work); migrating existing registries (D5); `run_inference` fetching
models from the registry instead of staging per run (§7 Q2).

---

## 3. Decisions

### D1 — a version is published in place, and a marker written last says it is complete

Replace stage-then-rename with three steps, in this order:

1. write the version's files directly under `v<N>/`;
2. **re-digest what landed** and confirm it equals the caller's content digest;
3. write `v<N>/_complete.json` — carrying that digest — **last**.

A single-object write is atomic on every backend fsd targets ("partial updates are not supported
with `Put Blob`"), so step 3 is the all-or-nothing moment the directory rename used to provide. No
staging prefix, no rename, no backend-specific code, no new dependency.

**Step 2 is not bookkeeping, it is the guarantee.** Publishing already re-digests the landed bytes
today (`registry.py:405`); that check is what makes publication *provably* the caller's content,
and it is strictly stronger than the rename ever was. Putting the marker after it means
`_complete.json` asserts **"these bytes were verified"**, not merely "the writer stopped". An
interrupted upload, a crashed process, or two writers interleaving files all leave a version with
no marker — which D2 makes invisible.

### D2 — only `_list_versions` reads the marker, so resolution stays free

`_list_versions` counts a `v<N>` directory only when `v<N>/_complete.json` exists. Nothing else
changes.

This deliberately does **not** touch `resolve`. Spec 51 D9 requires resolution to be one small
read and never a listing — a version pin (`crop-rf:3`) reads nothing at all today, and a marker
check would put a round-trip on the hot path where N nodes resolve a model. It is also
unnecessary: `_list_versions` is called only by `publish` and `migrate`, so an unfinished version
can never be handed out as a *new* version, and a pin naming an unfinished one fails loudly at
bundle load.

### D3 — a publish loop that cannot hang

`_write_new_version` currently treats every `OSError` as a lost race and retries forever. After D1
there is no rename left to fail, but the loop remains for genuine version collisions, so:

- retry **only** when the target already exists and does not hold the caller's digest;
- re-raise anything else, wrapped with the registry root, instead of retrying it;
- bound the loop at **`_MAX_PUBLISH_ATTEMPTS = 16`** and raise on exhaustion.

Worth doing even if D1 were rejected: an unbounded loop over a deterministic failure is the defect
that gave this spec its reason to exist.

### D4 — `configure_storage` is called by every verb that touches the seam

`deploy`, `run_inference`, `verify_adapter` and `verify_image` each call
`configure_storage(storage)` in preflight, where the other three verbs already do — after argument
validation, before the first storage access.

`configure_storage` sets **process-global** fsspec state, so this is four one-line calls, not a
`storage_options` refactor; the registry functions' existing `storage_options=` parameters are
left exactly as they are. This replaces today's accident, where authentication works only because
some *earlier* verb in the same process happened to set the flag.

**`deploy` stops refusing non-local storage:** `_check_local_seams(..., storage_allowed=False)`
was correct while a blob registry could not work, and is what this spec removes.

### D5 — existing registries keep working, and nothing is migrated

> **Superseded by the 2026-10-13 amendment (end of this spec):** only `_complete.json` makes a version.

A version published before this spec has no `_complete.json`, so D2 would make it invisible. That
is unacceptable for zero benefit, so `_list_versions` treats a directory holding `bundle.json` but
no marker as **complete (legacy)**. The only registry that exists is `notebooks/demo_registry`,
days old and local; it keeps resolving, and it can be moved to blob as a plain file copy (spec 51
D11 guarantees every ref still resolves; `registry.migrate` re-digests to prove the copy landed).

The legacy rule is a small permanent cost — an unfinished *legacy* version cannot be told from a
finished one — and it is bounded: everything published after this spec carries a marker.

> **Amendment (raised in implementation, 2026-08-24 — resolved by Opus review, pending the
> user's confirmation).** D5 as originally written conflicted with AC2: `_list_versions`'
> legacy rule (`bundle.json` present, no marker ⇒ complete) would have counted a freshly
> interrupted *post-spec-52* publish too, since an interruption anywhere after `bundle.json`
> landed left exactly that on-disk shape. Resolved as follows.
>
> **The residual window, stated plainly.** `bundle.json` with no marker means one of two
> things: content published before this spec (guaranteed complete — the old stage-then-rename
> either landed a whole version or nothing), or a post-spec publish interrupted between its
> manifest write and its marker write. Nothing on disk separates them. D1 orders the writes so
> the second case is as narrow as this design can make it (**Opus review, 2026-08-24: narrow, but
> not the "one object write" originally claimed — `content_digest(target)` re-reads the whole
> bundle inside that window. See §10.4; the conclusion is unaffected**), and where the two still
> collide **the legacy
> reading wins**: reading legacy content as incomplete would hide a real published version and
> let the next `publish` allocate over it, losing a model, whereas reading an interrupted
> version as complete strands a folder and burns a version number. Strand the folder.
>
> **`migrate` is not legacy.** `_migrate_version` already performs D1's steps 1 and 2, so it
> writes `_complete.json` last as well. Migrated content is first-class marked, and the legacy
> rule's only remaining job is genuinely pre-spec-52 registries.

---

## 4. Acceptance criteria

1. `registry.publish` against `memory://` returns a version — **and does so in seconds**, asserted
   with a timeout so a regression to #88's hang fails rather than stalls the suite.
2. Interrupting a publish **before its `bundle.json` lands** leaves a version that
   `_list_versions` does not count and that the next `publish` reuses in place.
   `_write_new_version` writes `bundle.json` last of the content files precisely so this
   covers every interruption during the write — all the bytes and all the elapsed time are
   in the artifacts. An interruption in the remaining window (after `bundle.json`, before
   `_complete.json`) leaves a stranded unmarked version indistinguishable from legacy
   content; D5 accepts that deliberately, and §5 already accepts such strays consuming
   version numbers.
3. `_complete.json` is written only after the landed content re-digests to the caller's digest; a
   version whose bytes do not match never gets a marker.
4. `_write_new_version` retries only a genuine version collision, re-raises everything else, and
   is bounded at 16 attempts.
5. A version published before this spec (no marker, has `bundle.json`) still resolves — D5. **Superseded (2026-10-13 amendment); AC2's exception, AC5a's "legacy" premise too.**
5a. `migrate` writes `_complete.json` for every version it copies (the amendment to D5): a
   migrated registry's versions list identically even with the legacy rule hypothetically
   removed, because they are marked, not merely legacy-complete.
6. `deploy(..., storage="azure")` is accepted, and `deploy` calls `configure_storage` before its
   first storage access.
7. `run_inference`, `verify_adapter` and `verify_image` each call `configure_storage`, asserted in
   a process where no other verb has run.
8. **Spec 51 AC12 holds for real:** `deploy` → `set_alias` → `resolve` → `run_inference` behave
   identically against a local registry path and a `memory://` one — the test spec 51 shipped
   skipped, now unskipped and passing.
9. `resolve` for a version pin still performs no listing and no marker read (spec 51 D9).
10. `pytest -q` and `ruff check src/ tests/ demos/ examples/` clean; no network in unit tests.

---

## 5. Risks

> **Superseded by the 2026-10-13 amendment (end of this spec):** only `_complete.json` makes a version.

**Two simultaneous publishers can corrupt a version, and nothing prevents it.** Both write into
`v1/`, their files interleave, and neither set of bytes is intact. D1 step 2 means neither writer
*marks* it, so neither claims it and both move on — the damage is a stranded folder rather than a
bad model being *returned by `publish`*. **Corrected by Opus review, 2026-08-24: that folder is
not invisible.** Both writers wrote a `bundle.json`, and D5's legacy rule reads
`bundle.json`-without-a-marker as complete, so `_list_versions` counts it: it burns a version
number and stays reachable by an explicit pin or a hand-set alias. Nothing *hands it out* — no
caller resolves "the latest version" through `_list_versions` (D2) — so the risk is unchanged in
kind, but the original wording overstated the containment and is fixed here rather than left to
mislead a later reader. This is the guarantee the rename used to
provide and that this spec gives up, on the user's statement (2026-08-24) that concurrent
publication of the same model does not happen in this team. **If that changes, §6 is the way
back.**

**The `abfss://` path is still not proven by this suite.** `memory://` exercises the same code,
and after D1 there is no backend-specific branch left to diverge — which is most of the argument
for this design. But "no branch" is not "tested". The real proof is a run-book: publish two
versions to a blob registry, repoint an alias, run inference off the ref, and confirm a re-publish
of identical content is a no-op. **Green tests are not done here** — see MEMORY
`real-run-beats-review`.

**Stranded unmarked versions are never cleaned up.** `fs.rm(recursive=True)` is unreliable on
`abfss://` (#50), so a blob registry will slowly collect unmarked `v<N>` folders from interrupted
publishes. They are invisible and harmless, but they consume version numbers and storage. Out of
scope; noted so it is not refiled later as a bug.

---

## 6. Alternatives considered

**Azure's native atomic directory rename** — the first draft of this spec, and the design the user
initially chose before asking whether it was over-engineered. ADLS Gen2 with a hierarchical
namespace really does rename a directory atomically (`Path::Create` with `x-ms-rename-source`);
adlfs simply never calls it, but `DataLakeDirectoryClient.rename_directory` does. It would
preserve **both** of the rename's jobs, race arbitration included, and change no on-disk layout.

Rejected as disproportionate. It requires: a new declared dependency; a permanent Azure-specific
branch inside the storage seam; `etag="*", match_condition=MatchConditions.IfMissing` on every
call, because **`Path::Create` overwrites its destination by default** and an unconditional rename
would let a losing racer silently destroy the winner's published version; handling
`x-ms-continuation` for directories too large to rename in one invocation; and file-vs-directory
dispatch inside `fs.rename`. All of that buys a guarantee against a scenario the user says does
not occur — and it would still leave S3 and every other backend broken, whereas D1 fixes all of
them at once.

**A lock service.** Spec 51 §5 rejected it; nothing here reopens that.

**Leaving it local and documenting the limitation.** The standing decision for about an hour on
2026-08-24, reversed by the user the same day. The reasoning still holds for anyone who does not
need a *shared* registry: the model reaches the cluster either way, because `verify_image` and
`run_inference` stage it per run.

---

## 7. Questions at sign-off — ALL RESOLVED (user, 2026-08-24)

Every question was signed off **at its proposed default**. Recorded individually so a later reader
sees a decision, not an unread list.

1. **[RESOLVED — default stands] `_MAX_PUBLISH_ATTEMPTS = 16`.** It bounds only genuine version
   collisions. Folded into **D3**.
2. **[RESOLVED — stays out of scope] `run_inference` keeps staging the model per run.** A blob
   registry makes fetching-from-the-registry possible — `_stage_bundle` would become a
   blob-to-blob copy that could be skipped — but it changes how nodes get models and needs its own
   decision. **Do not implement it as part of this spec**, and do not treat the per-run `[stage]`
   line as a defect while doing so.
3. **[RESOLVED — default stands] `deploy` keeps an explicit `storage=` kwarg**, matching every
   other verb, rather than inferring the backend from the registry URL's scheme. Inference would
   be the silent-fallback pattern spec 51 D4 refused elsewhere. Folded into **D4**.
4. **[RESOLVED — not now] `_complete.json`'s digest does NOT replace the `_deploy.json` digest read
   in `publish`'s idempotency scan.** It would be a real saving (the scan could skip re-digesting
   every version) but it is an optimisation, and this spec exists because the previous draft grew
   too large. The marker carries the digest anyway, so taking this later costs nothing.

---

## 9. Implementation note

Per CLAUDE.md's model split, implementation is a **Sonnet session at `/effort medium`** against
this signed-off spec, handed back to Opus `/effort high` for review before merging. Phased so each
step is independently revertible:

0. **The registry core (D1, D2, D3, D5)** — `_write_new_version` writes in place, re-digests, then
   writes `_complete.json`; `_list_versions` becomes marker-aware with the legacy rule; the retry
   loop is bounded and stops misreading hard failures as races. Pure `fsd.model.registry`, no verb
   touched, fully unit-testable on `memory://`. This is AC1–AC5 and AC9, i.e. most of the spec.
1. **Verb wiring (D4)** — four `configure_storage` calls, and `deploy` stops refusing
   `storage="azure"`. AC6, AC7. Small and mechanical; do it only once step 0 is green.
2. **End-to-end on a URL registry** — unskip spec 51's AC12 test and drive
   `deploy` → `set_alias` → `resolve` → `run_inference` against `memory://`. AC8.

**Then the part tests cannot do (§5):** a run-book the *user* executes against the real Azure
account — publish two versions to an `abfss://` registry, repoint an alias, run inference off the
ref, confirm a re-publish of identical content is a no-op. Write it to `fsd/runbooks/` using
`TEMPLATE.md`; Claude never runs it. **This spec is not done on green tests.**

---

## 10. Opus review of the implementation (2026-08-24)

Reviewed at `/effort high` against the shipped branch, per CLAUDE.md's model split. Four findings,
all fixed in-branch. **The D5/AC2 amendment itself was re-derived independently and stands** — the
one correction to it is §10.4, which does not change its conclusion.

1. **`run_inference` and `verify_adapter` turned a preflight error into a bare `ValueError`.**
   D4's `configure_storage` call was placed before each verb's `_raise_preflight`, which is
   genuinely *necessary* (`_resolve_model_ref` and `fs.read_geo` touch storage first, as the
   handoff argued) — but `configure_storage` **raises** on an unsupported backend, so
   `storage="s3"` escaped as a `ValueError` instead of the verb's `PreflightError`, discarding
   every other accumulated preflight error with it. Confirmed by calling both verbs. And the
   *accepted* case had the side effect the handoff had reasoned was absent:
   `run_inference(storage="azure")` on the pre-built-cubes path is refused by the seam gate, yet
   still flipped the whole process to authenticated adlfs on its way to being refused — the exact
   accident D4 exists to remove. **Fixed:** the seam check raises on its own, before
   `configure_storage`, in both verbs — the shape `deploy` already had. `verify_adapter`'s date
   errors are no longer batched with seam errors; a seam misconfiguration is fatal by itself.
2. **Publishing into an incomplete version inherited the previous attempt's leftovers.** AC2 has
   the next `publish` reuse an interrupted `v<N>` **in place**, and D1 step 2 cannot police what
   it inherits: `content_digest` covers only *manifest-declared* files, so an artifact or a
   `code/*.py` the new bundle does not declare survived into the version and was then marked
   complete. `bundle.load` puts a version's `code/` on `sys.path`, so a stale module left there is
   importable by whichever adapter lands next — a quiet wrong-model failure the old
   stage-then-rename could not produce, because every attempt got a fresh staging prefix.
   **Fixed:** `_write_new_version` `_discard`s an existing incomplete target before writing.
   Best-effort like every other `_discard`, so #50's unreliable recursive rm leaves a blob
   registry no worse off than before the call existed.
3. **Four branches were unpinned — found by mutation testing, not by reading.** Deleting the
   idempotent-collision `return`, disabling the landed-digest guard, and replacing D5's legacy
   `bundle.json` check with `return False` each left the whole suite **green**. The two "race"
   tests rewritten during implementation look like they cover the collision path but cannot reach
   it: the competitor publishes *before* `_list_versions` runs, so allocation starts past their
   version and the loop body never sees an existing target. So the narrowing flagged at handoff
   did lose something the ACs require — AC4's "retries only a genuine version collision", and
   **AC5, which had no test at all**. **Fixed:** five tests added, each verified to kill its
   mutation; the collision pair reaches the branch by simulating the stale listing that *is* the
   race. D3's re-raise was already covered and stays so.
4. **The residual window is not "one object write wide".** Between `bundle.json` landing and
   `_complete.json` landing sits `content_digest(target)` — a full re-read and re-digest of the
   whole bundle, which on blob is many round trips, not one write. The amendment's wording is
   corrected in D5. **The conclusion is unaffected, and on inspection is stronger than it was
   argued:** everything in that window is *post*-write, so a version stranded there holds complete
   content, and reading it as legacy-complete is right — it is unverified, not partial.

5. **#86 is not closed by this spec — corrected 2026-08-25 after the real run.** D4's
   `configure_storage` call is unreachable on `run_inference`'s pre-built-cubes path, because
   `storage="azure"` is refused there by the seam gate
   (**[#90](https://github.com/nikhilsrajan/fsd/issues/90)**). Step 4 is the run-book's only step
   that goes through a verb at all, so the run-book as written cannot exercise D4 anywhere. Two
   further facts from that run: adlfs's `anon` default is `None` (its anonymous branch is a
   fallback reached only when credential discovery fails), and steps 1-3 read and wrote a real
   storage account with **no** `configure_storage` call — so #86's stated failure mode may not
   occur under a developer `az login` at all. The same run also found that a blob-resolved ref
   cannot be loaded on the local run path
   (**[#89](https://github.com/nikhilsrajan/fsd/issues/89)**). Both are addressed in
   `specs/53-blob-registry-on-the-local-run-path.md`. **#88 is genuinely fixed and closed** —
   publish returned v1 in 32.9 s against `abfss://`, where it previously never returned.

## 8. Best-practice alignment / sources

Cross-validation run at draft (2026-08-24) under `CLAUDE.md`'s standing permission for spec
searches. It did real work twice: it **overturned** the original assumption that no atomic
directory move exists on object storage (one does, on ADLS Gen2 — which is why §6 is a real
alternative rather than a straw man), and it then **overturned that alternative's own first
draft** by establishing that the rename overwrites its destination unless made conditional.

### External

- **[Put Blob (REST API)](https://learn.microsoft.com/en-us/rest/api/storageservices/put-blob)**:
  supplies **D1's foundation** — "partial updates are not supported with Put Blob", so a
  single-object write lands whole or not at all. That is what makes `_complete.json` a sound
  completion marker rather than a hopeful one.
- **[Manage concurrency in Blob Storage](https://learn.microsoft.com/en-us/azure/storage/blobs/concurrency-manage)**:
  supplies **§5's honesty** — the default is last-writer-wins, and snapshot isolation means a
  reader sees a consistent version of a blob rather than a partial one. It also names leases and
  optimistic concurrency as the real answers if arbitration is ever needed, which is what §5's "if
  that changes" would reach for.
- **[Azure Data Lake Storage hierarchical namespace](https://learn.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-namespace)**:
  supplies **§6's central claim** — an HNS account "enables atomic directory manipulation" by
  "updating a single entry (the parent directory)". This is what makes the rejected alternative
  genuinely viable, and why `fs.rename`'s docstring was half right rather than simply wrong.
- **[Path - Create (ADLS Gen2 REST API)](https://learn.microsoft.com/en-us/rest/api/storageservices/datalakestoragegen2/path/create)**:
  supplies **§6's cost** — rename is `Path::Create` with `x-ms-rename-source`, but *"by default,
  the destination is overwritten […] To fail if the destination already exists, use a conditional
  request with `If-None-Match: "*"`"*, and a directory rename may return `x-ms-continuation`
  rather than completing in one call. Both are why §6 was rejected as disproportionate.

### Internal — primary evidence, read or executed rather than cited

- **`adlfs 2026.8.0`, inspected directly**: `AzureBlobFileSystem` defines neither `_rename` nor
  `_mv`; `.mv` resolves to `fsspec.spec.AbstractFileSystem.mv`, whose body is `copy()` then
  `rm(path1, recursive=recursive)`. This is the whole of #88, and it disproves `fs.rename`'s claim
  that the move is "one metadata operation" on Azure. Read from the installed package, per MEMORY
  `verify-the-primitive-a-spec-cites`.
- **File-rename probe on `memory://`** (2026-08-24): a single-file `fs.rename` succeeds on a
  non-local backend, including onto an existing target. This is what confines #88 to
  `registry.py:394` and keeps `set_alias`, `write_deploy_record` and `builder.py:365` out of scope
  — the single most scope-reducing fact in this spec.
- **Marker-publish prototype on `memory://`** (2026-08-24): publishing with no rename works, and a
  version left without its marker is correctly invisible to a marker-aware listing. D1 is that
  prototype.
- **`fsd/model/registry.py:405`**: publication already re-digests the landed content on the
  success path. D1 step 2 promotes that existing check into the publish protocol rather than
  adding one.
- **`fsd/storage/azure.py::configure_storage`**: sets process-global fsspec state, which is what
  lets D4 be four one-line calls.

---

## Amendment (2026-10-13) — drop the legacy rule; only `_complete.json` makes a version (#145 X1)

**Status:** SIGNED OFF (user, 2026-10-13), before any code in the X1 PR (#174) was written. Decided by the
user on 2026-10-09 (question Q3a of the refactor audit, #145 Decisions comment) and 2026-10-13 (the three
X1 questions). Supersedes D5, its 2026-08-24 amendment,
AC2's second half, AC5 and AC5a, and the "not invisible" correction in §5's first risk.

**Problem.**
- D5 kept every pre-spec-52 version visible: a `v<N>` with `bundle.json` and no `_complete.json` counts as
  complete (`_is_version_complete`, `model/registry.py:309-324`). It was meant for one registry,
  `notebooks/demo_registry`, "days old and local" at the time.
- It has a cost on every new publish. An interrupted publish that got as far as `bundle.json` looks
  finished, so it burns a version number and stays reachable by a pin (D5 amendment; §5). To shrink that
  window, `_write_new_version` writes `bundle.json` last, out of the caller's order (`registry.py:393`),
  and a hazard comment guards the order.
- The image registry (`registry/_core.py:51-55`, spec 56) never had the rule. So the two registries
  disagree about what a version is, which also blocks merging them (#145 C1).
- `AGENTS.md` says "no compatibility shims"; the audit lists this as one (#145, rubric row 12).

**Decision.**
- **X1.1** A version is complete only when `v<N>/_complete.json` exists. `_is_version_complete` drops
  its `bundle.json` fallback and matches `registry/_core.is_version_complete`.
- **X1.2** `_write_new_version` writes the files in the caller's order. With X1.1, an interruption at any
  point before `_complete.json` leaves an unmarked, invisible version, whatever order the files landed
  in, so the manifest-last sort has no job.
- **X1.3** What an unmarked `v<N>` now means, in every case: it is invisible to `_list_versions`, so
  `publish`'s idempotency scan and `migrate` skip it. When `publish` next allocates that number, it
  clears the folder and rewrites it (unchanged code: `_discard` of an incomplete target). `resolve`
  is unchanged (D2): a pin or an alias that names an unmarked version still reads it.
- **X1.4** Old content is not migrated or stamped. The user confirmed on 2026-10-13 that no registry in
  use holds an unmarked version. The one known case is `notebooks/demo_registry/crop-rf/v1` (local,
  gitignored, 2026-08-24, alias `demo-eurocrops-at`). Under X1.3, a later `publish` of different content
  to `crop-rf` in that registry would replace v1, and the alias would then name the new model. The user
  chose on 2026-10-13 to delete that demo registry.
- **X1.5** Comments and tests that describe the old rule are rewritten or removed in the same PR:
  - `_is_version_complete`'s docstring;
  - `_write_new_version`'s docstring: the "not invisible" sentence, the ⚠️ ordering hazard and the
    residual-window paragraph;
  - the "legacy rule" comment in `_migrate_version`;
  - `_core.py:52`'s "unlike `fsd.model.registry`";
  - in `tests/test_registry.py`: the manifest-last test, the interleaving test's `[1, 2]` assertion and
    the legacy test.

**Prior art.**
- Apache Oozie, Coordinator spec §5.1 (synchronous datasets): with no `done-flag`, "the coordinator will
  wait for the presence of a `_SUCCESS` file in the directory"; a named flag means the dataset "will be
  considered ready (done) when the file exists". Directory presence counts only when the flag is
  explicitly empty. X1.1 is the default case: the marker is the only signal.
- Hadoop, S3A committers page: "After a Job is committed, all its work must be visible. A file named
  `_SUCCESS` may be written to the output directory" (`mapreduce.fileoutputcommitter.marksuccessfuljobs`,
  default true). That is the writer side of the same convention, matching D1 step 3.
- In fsd: the image registry (spec 56) has used marker-only completeness from its first version.

**How to verify.**
- A test: a `v<N>` with `bundle.json` and no `_complete.json` is not listed. The next `publish` of
  different content returns that same N, and the folder then holds the new content, marked.
- A test: an interruption after `bundle.json` lands but before `_complete.json` leaves a version that
  `_list_versions` does not count and that the next `publish` reuses in place. This is AC2 without its
  exception.
- The interleaving test asserts `_list_versions == [2]`: the unmarked `v1` is now invisible.
- `grep -n "legacy" src/fsd/model/registry.py src/fsd/registry/_core.py` prints nothing.
- `grep -n "BUNDLE_MANIFEST" src/fsd/model/registry.py` no longer shows the sort key.
- No gate 4: the changed code has no backend branch, and no AML node calls `fsd.model.registry`.
  `run_inference` resolves the ref on the driver (`api.py:1289`), and the node loads the bundle by path
  (`workflows/infer_task.py`). The suite runs the changed code on local and `memory://` registries.

**Risks.**
- Two concurrent publishers of one name: B can now clear A's version while A is re-digesting it, where
  before B would have counted it as complete and moved on. A then sees a digest mismatch and moves to
  `v<N+1>`. No wrong model is marked. §5 already rests on "concurrent publication of the same model does
  not happen in this team", and §6 is still the way back.

**Out of scope.**
- Merging `registry/_core.py` into `model/registry.py` (#145 C1). That follows X1.
- Cleaning up stranded unmarked folders (§5's third risk; #50).
