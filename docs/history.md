---
status: current
summary: How fsd got the shape it has — eight eras, each named by the question it was answering, each recording the forks taken, the forks dropped, and the measurement that decided them. Append-only (spec 43 D3); ends at the consumer-repo run of 2026-09-02.
---

# A history of fsd

**What this is.** The *why*. Every other document here answers "what was decided" (`../specs/`,
`adr/`), "what is true now" ([`../ARCHITECTURE.md`](../ARCHITECTURE.md)), or "what changed"
([`CHANGES.md` at tag `docs-archive-2026`](https://github.com/nikhilsrajan/fsd/blob/docs-archive-2026/CHANGES.md)). This one answers **what happened** — and in particular the
things the other registers structurally cannot hold: the roads not taken. An
[ADR](adr/0025-one-fact-one-home.md) records a decision *made*, in active voice. The fork examined
for a week and declined has no row anywhere else.

**How to read it.** Eight eras, each named by the question the project was actually facing.
Nothing here is the only home for a fact — every claim links to the register that owns it. If you
want to *run* something, this is the wrong document: start at
[`../README.md`](../README.md), then [`tutorial.md`](tutorial.md).

**Where it stops.** At the consumer-repo run of 2026-09-02. This file is **append-only** (spec 43
D3): a closed era is never rewritten, only appended to.

---

## Before the first era: what fsd was for

fsd is a clean-room rewrite that folded three older repositories — `fetch_satdata`, `rsutils`,
`cdseutils` — into one installable package, so that a researcher could go from *an area on a map*
to *a trained model's predictions as georeferenced imagery* without owning a cluster.

The framing that outlasted everything else, and which decided a fork 25 days later: **fsd is an
alternative to Google Earth Engine, for running simpler models at scale.** Not a modelling
library. fsd deliberately never trains a model; training stays on the user's side, behind the
[ModelAdapter contract](../specs/18-model-adapter.md). What fsd owns is everything either side of
that: getting pixels, shaping them into a cube, and running someone else's model over a region
without them thinking about nodes.

Two commitments were made at the start and never broken, and both are stated as *absences* — the
form [ARCHITECTURE.md](../ARCHITECTURE.md)'s invariants section takes throughout. **No module
opens a remote path outside `fsd.storage`**, so that "where the bytes live" is configuration
rather than code. And **the unit of work never knows what dispatched it**, so that the thing that
runs on a laptop and the thing that runs on 32 cluster nodes are the same code. Nearly every good
outcome below is downstream of those two absences; several of the bad ones came from a place where
they had quietly not been honoured yet.

One theme recurs so often it is worth naming before you meet it eight times: **existence is not
identity.** A file that is *there* is not evidence that it is the *right* file, produced by the
request you think produced it. That single confusion is behind more defects in this history than
any other cause.

---

## Era 1 — What does fsd actually have to reproduce?
**2026-07-06 → 07-11**

The rewrite's first job was archaeology: work out what the three legacy repos *did*, and port the
behaviour rather than the code. By 2026-07-06 the local pipeline existed end to end — the
[API façade](../specs/16-packaging-and-api.md), a [STAC export view](../specs/17-stac-catalog.md)
that turned 579 catalogued tiles into Items in 0.06 s, the
[ModelAdapter contract](../specs/18-model-adapter.md), and an
[ROI→grid-cell demo](../specs/19-e2e-demo.md).

**The fork that defined the era was taken on purpose and then bit immediately.** Porting
*behaviour* faithfully means faithfully porting the bugs. [Spec 20](../specs/20-datacube-tile-merge-bug.md)
is the exhibit: `_stack_datacube` kept only **one** MGRS tile per `(timestamp, band)`, because it
stored them in a dict. Any shape straddling an MGRS-tile boundary silently lost every other
same-acquisition tile's coverage. The worst grid cell in the demo held **0.6 % valid pixels
despite roughly 80 % raw coverage**. This was a legacy bug carried across intact, invisible until
inference grid cells became the first shapes big enough to straddle a tile boundary. Fixed by
merging all same-`(timestamp, band)` images onto the reference grid: **0.6 % → 82.8 %** on that
cell, and the demo's merged map went **90 % → 96 % valid with zero dead grid cells, from nine.**

It is the first instance of the era-spanning shape: a dict keyed on `(timestamp, band)` treats the
*existence* of one tile as though it were the whole answer.

**The fork dropped: `multiprocessing.Pool`.** [Spec 22](../specs/22-unify-inference-runner.md)
retired the last parallel fan-out that was not on the runner seam. It bought nothing locally — the
point was that scaling out later should be a `runner=` swap and nothing else. That bet is
collected in era 4.

**The measurement that decided the era's biggest rewrite: 8.8 MB/s probe, ~0.2 files/s
aggregate.** Downloads were converting JP2→COG inline on the transfer threads, and GDAL holds the
GIL, so conversion starved the transfers.
[Spec 25](../specs/25-download-convert-redesign.md) split the worker into a thread stage and a
*process* stage chained by callbacks, bounded by a semaphore. Reviewing that implementation found
a defect nobody had flagged: an exception inside a completion callback leaked its permit, and
`download()` then **hung forever** — the silent-hang failure mode that the next spec's whole
premise was meant to exclude. It was fixed *first*, as
[spec 25b](../specs/25b-pipeline-exception-safety.md), before anything ran over a network.

**And a fork about how the project itself would work.** The trigger was a process failure, not a
code failure: a long download got launched that the user could not see progress on or stop, and
polling its logs burned context for nothing. [Spec 24](../specs/24-working-contract.md) made the
working process a designed artifact — Claude never runs long, networked or side-effecting scripts;
those become **run-books** the user runs, pasting back a machine-readable `_result.json` that gets
diffed against criteria written *before* the run. Opus specs and reviews, Sonnet implements
against a signed-off spec. Every era after this one is shaped by it, which is why it is here and
not in a footnote.

---

## Era 2 — What does fsd owe a viewer?
**2026-07-13 → 07-15**

The pipeline met real data at full size. The Austria end-to-end ran for real: **207 granules /
44.61 GB, 300 grid cells, 900 training fields, a 6830×6868 merged map at 99.2 % valid, in ~100
minutes**, download and inference taking 45 % and 44 % of the clock.

Running it at full size surfaced three defects that the same code had passed every test with. The
sharpest: `cog_outputs_to_items` derived each STAC Item's id from the COG filename stem, which was
the constant `"output"` — so a 300-cell run produced a collection with **300 identical links and
one item file on disk**. Existence is not identity, again, and this time in the metadata.

**The fork dropped, and it is the era's whole point: fsd does not build a dashboard.**
The work started as a local titiler + Leaflet viewer to verify the inference STAC.
[Spec 27](../specs/27-titiler-leaflet-stac-verify.md) was written and signed off pending — then a
design discussion reframed it and **it was superseded before a line was implemented.** The
argument: NASA Harvest's STACNotator already *is* the viewer, and it consumes exactly two standard
APIs (a STAC API and titiler-pgstac's XYZ endpoints). So fsd's job is not to serve tiles; it is to
emit artifacts standard enough that a **stock** pgSTAC + titiler-pgstac deployment serves them.
fsd becomes, in the phrase used at the time, *"just another MPC"* — and owns only the COG + catalog
+ render-config contract, with the serving stack a deployment decision rather than fsd code.

That contract was then validated in two tiers rather than asserted:
[Tier 1](../specs/29-tier1-prestyled-xyz-validation.md), a pre-styled XYZ URL pasted into
STACNotator's bring-your-own-layer; and [Tier 2](../specs/30-tier2-mini-mpc-validation.md), a
local "mini-MPC" — real pgSTAC, real stac-fastapi, real titiler-pgstac in Docker. Tier 2 **ran
green**: a tile fetched as `200 image/png`, and QGIS rendered the 300-cell crop map in class
colours over the true slanted cell footprints, through the full register→searchId→XYZ path.

**A measurement that changed a plan, and a fork deferred to be measured rather than argued.**
The download was **link-bound: 26 MB/s probe against 17 MB/s aggregate, with four transfer streams
slower than one.** That is a property of a laptop uplink which inverts on a datacenter NIC, so
retuning was explicitly deferred to the cloud rather than tuned against the wrong hardware.

**The era ends by turning the question on the project itself.** On 2026-07-15 the user stopped
feature work and asked, plainly: *am I accreting endless TODOs, or is there a critical path?* The
diagnostic's verdict was not "scope creep" — the register was well-triaged and nearly every item
real. The pattern was subtler and worse: **the project kept finishing locally-completable work
*around* its critical path, because that path was blocked and the blocker had never been named.**
The blocker turned out to be entirely activation energy — unfamiliar Azure. Naming it dissolved
it: an access probe ran green the same day, a 370-byte write-read round-trip proving the personal
identity already had the storage role everyone had been quietly worried about.

---

## Era 3 — Where does radiometric truth get established?
**2026-07-16 → 07-18**

**The fork taken: Microsoft Planetary Computer as a second source.** MPC serves Sentinel-2 L2A as
already-COG on Azure, which made the entire JP2→COG conversion problem — the thing era 1 spent
three specs solving — simply *evaporate* for that path.

It also brought a correctness debt to a head. Sentinel-2's processing baseline 04.00 introduced a
`BOA_ADD_OFFSET` of −1000, and reprocessing stamps new baselines onto *old* acquisition dates — so
the offset must be keyed on **baseline, not date**.
[Spec 32](../specs/32-mpc-source-baseline-harmonization.md) derived it from
`s2:processing_baseline` and stored it as a catalog column.

**Then the reversal, and it is the most instructive moment in this history.** The MPC pivot had
*deleted* spec 31's §5, the `stage → convert → put` ingest step, on the reasoning that MPC needs no
conversion. The user's counter-argument, verified against the code rather than accepted on
assertion: MPC did not *remove* normalization, it **moved** it — from format (JP2→COG) to
radiometry (the baseline offset) — and the radiometry had been put in the datacube builder. In the
project's own words: **§5's shape was right and deleting it was the error.**

The evidence was structural. `build_datacube`'s chain contained `_apply_boa_offsets`,
`apply_cloud_mask_scl`, `drop_bands(["SCL"])` and a hardcoded `REFERENCE_BAND="B08"`. It was **an
S2 L2A builder wearing a generic name**, so every future source would have to either cosplay as
Sentinel-2 or force a rewrite. The project had already logged the consequence without seeing the
pattern — an open item noting that CHIRPS and ERA5 have no SCL band, filed as a one-off. The fix
was to generalise §5 rather than delete it: `stage → normalize → put`, per source, which became
[spec 34](../specs/34-ingest-normalization-contract.md).

**The measurement that made the argument undeniable was found in fsd's own test data.** Every
granule in the Austria archive is baseline `N0500` — offset −1000 — but the CDSE ingest path
hardcoded `boa_add_offset = 0`, and the catalog predated the column entirely. **Every datacube
ever built from that archive is ~1000 DN too high, including the 300-cell crop map from era 2.**
Not a seam problem, and harmless to the infrastructure tests it still serves; but it is the exact
failure the pivot describes — the downloader did not normalize, the wrongness got baked into an
artifact, and *the catalog asserts it needs no fix*. The archive was never re-ingested. It is
still fine for infrastructure work and still wrong for science, and the workspace's own
`CLAUDE.md` says so at the top.

**The storage seam shipped anyway, and that was deliberate.** A well-argued "let us redesign
ingest first" has exactly the shape of the avoidance pattern the era-2 diagnostic had just
diagnosed. The guard applied was explicit: **the seam still ships.** It did — 20 granules / 40
files / 2.27 GB uploaded to blob at ~13.4 MB/s over VPN, every catalog row carrying an `abfss://`
path, and GDAL reading fsd's own uploaded COG back through `/vsiadls/`. The riskiest claims in
[spec 31](../specs/31-p1-azure-storage-seam.md) were proven on real data **before any seam code
was written for them**.

Two smaller things from this era outlived it. A standing practice: **every spec that leans on
external facts must be cross-validated against primary sources before sign-off, crediting what
each source specifically contributed** — the section you can find at the end of any spec from here
on. And a piece of housekeeping with teeth: the 159 GiB benchmark archive had been deleted for
disk pressure while the docs still described it as the test set, so a session planned work against
data that was not there.

---

## Era 4 — Does any of this survive leaving one machine?
**2026-07-20 → 07-29**

The seam bet from era 1 came due. [Spec 36](../specs/36-scale-runner.md) put a scale runner behind
the same CLI unit-of-work the local runner already drove; specs
[37](../specs/37-download-on-aml.md), [38](../specs/38-inference-on-aml.md) and
[39](../specs/39-training-data-on-aml.md) took download, inference and training-data assembly onto
Azure Machine Learning in turn.

**The fork taken, quietly, against the earlier plan: AML rather than Azure Batch.** The roadmap
had named Batch as the scale target from the beginning; the infrastructure that actually existed
had a far larger AML fleet, and AML is what shipped. The roadmap kept saying "Batch" for another
ten days, until an era-5 review caught it as a stale second home for a fact.

**The era's climax, on 2026-07-29:** `demos/e2e_austria_aml.py` ran unattended on the cluster.
**1127.7 s — 18.8 minutes — 8/8 steps, 97 jobs, 213 MPC granules, 300 grid cells → 300 output COGs
plus STAC and a merged map.** The milestone was not the script. Two roadmap phases had said
"pending cluster validation" for weeks, and this run *was* that validation: a laptop triggering a
cloud download → build → flatten with the arrays coming home, and an ROI's inference fanned out
over the cluster, both in one command.

**What it measured is more interesting than that it passed.** Job admission was **36 % of the
entire run — 403 s of 1128 s, nearly all of it a single cold start.** The download step spent
**286 s on admission against 84 s of execution**: 71 % of that step was the cluster scaling from
zero to 32 nodes. Node utilisation ran 5 % (download, cold), 42 % (build), 37 % (inference). A
known sharding imbalance reproduced exactly — 32 shards over 4 bands, so one band per shard, a
**16.4× spread between the fastest and slowest**. And clock skew on the VM was **−0.88 s ± 0.03**
against roughly 8 s on the laptop.

That last set of numbers settled a priority argument by arithmetic: admission at 403 s dwarfs the
~30 s the sharding fix would recover, so **the honest headline lever was cluster warm-up policy,
not sharding.** Both were recorded in [`findings/`](findings/) rather than argued about.

**The defects this era found are the reason [`findings/`](findings/) exists.** Six of them, and
the archive's own summary of their provenance is the point: *"found by running it — none by
review — all cost or nearly cost a run."* A `gpd.read_file` on an `abfss://` URL reported "No such
file or directory" for a file that plainly existed, because GDAL has no `abfss://` driver. A
**stale AML image silently voided an entire 25-minute run** — the telemetry stamps are written by
the `fsd` *inside the image*, so the run produced correct science and a void measurement. Dispatch
telemetry was ordered by run id rather than execution order, which had been silently mislabelling
every figure the plotter drew.

This is the era that earned the project's most-cited working rule, and the honest version of it is
narrower than the slogan: review and execution find **different classes** of defect. Review caught
the download hang in era 1 and five more in era 7. But nothing in a test suite catches an image
that is stale, a driver that lies about ordering, or a URL scheme GDAL has never heard of.

---

## Era 5 — Can anyone but the author read this?
**2026-07-30 → 07-31**

With the pipeline proven, the corpus became the problem. Measured on 2026-07-29: **201 markdown
files, 284,441 words**, of which `PROGRESS.md` alone was 37.7k.

**The fork dropped, and it is the one this document exists to honour.** The original request was
for a narrative plus *"≤~5 docs following the C4 model"*. C4 is a model for diagramming a system
you deploy; fsd is an installable library plus a pipeline that runs in three modes, and the mapping
was awkward. [Spec 41 D0](../specs/41-docs-refactor.md) **demoted C4 from file-count driver to the
section outline of a single file** — [`ARCHITECTURE.md`](../ARCHITECTURE.md) — and dropped the file
count, the Component level as a separate document, and Level 4 entirely. The trap it avoided is
worth keeping: C4's "container" means *a separately runnable thing*, and c4model.com's own page
opens with **"Not Docker!"** — so fsd's containers are the driver, the AML node, blob storage and
the catalog, and emphatically *not* its Docker images.

**The organising idea that replaced it** came from Diátaxis, and one decision followed from it that
governs the whole corpus: **every document is either point-in-time or continuously-true**
([ADR 0022](adr/0022-documents-are-point-in-time-or-continuously-true.md)). Point-in-time documents
— specs, run-books, findings, this file's own era sections — are *never substantially edited after
the fact*, a rule adopted wholesale from PEP 1. They are superseded, not rewritten.

That single rule then decided things that look unrelated. When ~450 references to numbered items
had to survive a migration to GitHub Issues, the choice was **to force the issue numbers to align
rather than rewrite the references** ([ADR 0024](adr/0024-todo-migrates-to-issues-with-forced-number-alignment.md)) —
because rewriting them would mean editing 30 point-in-time documents. `PROGRESS.md` was split
**3,691 → 93 lines**, with 61 entries moved verbatim into
[`progress-archive.md`](https://github.com/nikhilsrajan/fsd/blob/docs-archive-2026/docs/progress-archive.md) — moved, because deferring the history document made
that log the primary source for a spec not yet written. This one.

**The measurement that justified the whole era arrived as a defect in its own output.** The
rewritten `README.md` had been calling `run_inference` a **stub** long after it had shipped *and*
run on a cluster. Worse, the review of the fixed README found that **two of the three calls in its
60-second example raised `TypeError` before doing any work** — the first code any newcomer copies.
And the test that was supposed to prevent exactly this had passed, because it checked that each
verb existed in `fsd.__all__`: **existence, not callability.** The fix was a test that
`inspect.signature().bind()`s every README call against the live function, executing nothing.

The same theme, for the fifth time in five eras. A name in a namespace is not a working call, the
way a file on disk is not the right file.

---

## Era 6 — Should fsd exist at all?
**Open 2026-07-06, decided 2026-07-31**

The longest-open fork in the project, and the only one that questioned the premise. AllenAI's
**rslearn** is a broad, mature, Apache-2.0 superset of much of fsd's pipeline — 30+ data sources, a
window-based data model, compositors including SCL-aware ones, and a full PyTorch Lightning
training stack with foundation models. The question could not be dodged: *are we reinventing this?*

It was held open for 25 days on purpose — parked at the era-2 diagnostic as orthogonal to the
critical path, with one binding condition attached: **do not build more data sources until it is
called.** Then it was answered the way the project answers things, by measuring on a branch with an
isolated environment rather than arguing.

**What the probes measured**, all on a real VM against real blob storage:

- rslearn **reads and writes ADLS Gen2 under managed identity unmodified** — one `pip install
  adlfs`, no patching. The strongest point *for* borrowing, and it was found first.
- Blob read throughput, warm and like-for-like: **fsd's `/vsiadls/` path ~107 MB/s against
  rslearn's ~22 MB/s** — which vindicated the era-3 seam design rather than fsd's own preferences.
- fsd's calendar-interval contract did not survive the translation: **9 timesteps where fsd returns
  10, and 7 where fsd returns 9** once a period came up empty. rslearn's period yields one
  first-coverage scene; fsd takes a per-pixel median. Different definitions, not a tuning knob.
- Install weight: **5.3 GB, 2.9 GB cold download, 88.5 s — and the stock install does not import**,
  on an undeclared dependency.

**The decision, 2026-07-31: no rslearn for download. No full switch, no hybrid, no pilot.** The
reasoning that made it durable was not any single number, it was the category: *fsd is an
alternative to Google Earth Engine for running simpler models at scale; rslearn is a foundation-
model library that happens to ship an acquisition layer.* Two tools, two jobs — merging the
acquisition layers is a category error. `spike/rslearn` does not merge to `main`.

**What was kept from the fork, rather than thrown away.** rslearn's real value — its model library
and heavy foundation models — became a *named successor project*: run rslearn on Azure the way fsd
now can. The transferable asset is fsd's scale-out know-how (the runner seam, the storage seam, the
environment and identity work), **not** fsd's pipeline code; and the spike had already established
that such a project would not start with a storage problem, it would start at the runner, which
rslearn lacks entirely. Two byproducts were adopted outright: rslearn's lazy per-read signing
answered an open fsd design question, and its `class_path` + `init_args` pattern became the model
for a `Source` abstraction fsd still does not have.

**Two things about *how* this was decided are worth more than the verdict.** The report carried a
**standing bias warning on its front page** — the same analysis that had read the source
unfavourably was writing the supposedly neutral verdict — and guarded against it by requiring the
section on rslearn's genuine advantages to be written first and to be at least as quantified as the
section against. And the final planned probe, a pixel-equivalence diff, was **deliberately never
run**: an earlier probe had already shown such a diff would measure the quality of fsd's own
adapter shim rather than anything about rslearn. Knowing which measurement not to take is a
finding.

---

## Era 7 — Does the driver tell the truth about what ran?
**2026-08-19 → 08-21**

With the cluster path working, the era's question was whether it was *honest*. Four specs, and one
shape running through all of them, stated in the archive as: **a driver-side fact the code already
has in memory gets dropped on the floor instead of acted on or reported.**

- Bundles began [carrying their adapter's source](../specs/44-bundle-carried-adapter-code.md), so
  what ran on a node could be read rather than inferred.
- [Bundle transparency and image verification](../specs/45-bundle-transparency-and-image-verification.md)
  made `bundle.save` report what it had actually embedded, and refuse bundles it could not load.
- [Run addressability](../specs/46-run-addressability-and-grid-dedup.md) gave a run a name derived
  from what was requested, and dropped grid cells another cell already covered — measured on the
  real ROIs as 9→1 and 300→299.
- [Driver-side honesty](../specs/47-driver-side-honesty.md) took the four worst cases directly.

**The measurements are all measurements of silence and of lies.** `runners.py` contained exactly
one `print()` in 1,169 lines, and four driver-side legs were **completely silent for 627 s, ~1000 s,
and 30 minutes 10 seconds respectively** — indistinguishable, from the outside, from a hang. A
download that had nothing to do still paid a **full cold-start fan-out, measured at 5m31s**,
because the driver discovered the whole asset list and dispatched all of it without diffing against
what the catalog already held.

And the era's best defect was found in its own fix. The new merge progress bar **measured the wrong
thing**: `rasterio.open` reads a COG's *header*, while the pixels are read later inside the merge
itself. On the 300-COG case it cited — roughly 1000 s over the WAN — the bar reached 100 % at the
end of the header scan and the expensive phase then ran in silence. The review's verdict is the
line worth keeping: **worse than no bar, because it asserts completion.**

**Existence versus identity, again, and this time it silently produced wrong output.** A second
`run_inference` into the same output folder with a **different** ROI resumed the cached work list
by existence alone, and re-inferred the *first* ROI's cells. The fix made the output folder the
declared identity of a run, and made a mismatch raise rather than proceed.

**The fork dropped here was a stronger check that could not yet be built honestly.** The obvious
hardening is to verify a downloaded file by comparing its *size* rather than merely its presence.
It was explicitly declined in favour of existence-only, for the good reason: downloads are not yet
written atomically, so a **truncated** file is indistinguishable from a complete one at that layer,
and size-comparing would have looked like rigour while resting on the same unsound foundation. It
was filed instead, with atomic writes named as the prerequisite that must land first — as was an
opt-in existence pass that had been signed off and was then deliberately left unimplemented,
because no acceptance criterion covered it and pretending otherwise would have made the spec's
record of itself untrue.

---

## Era 8 — Is it pleasant to use from outside its own checkout?
**2026-08-21 → 09-02**

Everything so far had been driven from fsd's own checkout, by its author. The era's question came
from noticing that this proves nothing about the thing fsd claims to be — an installable package.

**The fork dropped first, and it retired a gate that had already been signed off.** The plan had
included a cold-start documentation test: follow the tutorial literally, stop at the first
instruction that fails. The user killed it with an argument that generalises: *"the tutorials I was
pointed to were something you pointed me to do. That felt artificial and did not help find the real
pain points."* **A doc-following exercise measures whether the docs are followable, not whether the
package is usable.** What replaced it was harder and slower: drive the real notebook as a user
would, then stand up a **separate repository** that installs fsd as a pinned dependency and holds
the demo notebook — because pip-install friendliness can only be measured from outside.

The work that followed was all shaped by that vantage point.
[`fsd init`](../specs/54-user-level-config.md) gave the package its first console script and a
user-level config file. [Spec 55](../specs/55-root-leaves-the-config.md) then amended it in *both*
directions: the storage root **left** the config, because it is a per-run destination rather than a
durable setting, while the two registry locations **entered** it as optional keys. That
durable-versus-per-run distinction became the test for anything proposed for the config file since.
The AML image recipe moved out of a notebook into a
[declarative, hash-keyed definition](../specs/56-image-definitions-and-registry.md) that builds only
if absent.

**Two defects from this era are worth reading twice, because both are the project's oldest theme in
new clothes.** Spec 56's review found an **infinite image-rebuild loop**: publishing is idempotent
by digest, so rebuilding an unchanged definition allocated no new version and dropped the one just
built, leaving the registry permanently naming a deleted asset. And before that, a cube that merely
*existed* was skipped as "already landed" — and then stamped with *this* request's identity over
the previous request's pixels. The review's note is the cleanest statement of the theme in the
whole corpus: **existence standing in for identity is exactly what the identity stamp exists to
prevent.**

**The era, and this history, close on a measurement.** The consumer notebook ran end to end from
the separate repository on 2026-09-02, with fsd installed as a dependency rather than checked out —
which by itself answered the era's question. The post-run window that
[spec 57](../specs/57-collect-and-stac-round-trips.md) had targeted went **777 s → 36 s (21.6×)**
against a prediction of under 100 s: `[collect]` **616 s → 26 s**, `[stac]` **161 s → 10 s**.

The reason that number is trustworthy is the part worth carrying forward. A faster run on a
different day invites the obvious objection — *maybe the network was simply better.* The run
answered it with a **control variable already in the data**: the `[merge]` leg, which the
optimisation deliberately does not touch, improved 2.4× over the same link. Crediting *all* of that
to a better network still leaves ~9.9× for collect and ~6.7× for stac as the code's own. The
control was not designed in advance; it was recognised in the results because one leg had been
deliberately left alone.

---

## Where it stood on 2026-09-02

fsd runs its full pipeline — download → datacube → training data → inference → COGs + STAC — on a
laptop and on an Azure ML cluster, from an installed package in a repository that is not its own,
with storage and runner as configuration rather than code. That was the goal stated on day one, and
it is met.

What is *not* met, stated plainly because a history that only records wins is a brochure:

- **The pipeline is Sentinel-2 L2A only.** Era 3 diagnosed the generic-builder problem precisely
  and shipped the ingest contract that makes a second source possible; the `Source` abstraction it
  implies still does not exist, and the code still dispatches on a hardcoded pair of source names.
- **The radiometry debt is fixed in code and still live in the test archive.** Cubes built from the
  Austria imagery remain ~1000 DN high. Fine for infrastructure, not for science.
- **The cluster's dominant cost is warm-up, not work.** Job admission was 36 % of the demo run and
  nothing has yet been done about cluster warm-up policy, which the measurements name as the real
  lever.
- **No tag has been cut.** The version has been deliberately withheld until the asset layout stops
  moving, so that a pin means something.
- **rslearn-on-Azure** was named as a successor project and has not been started.

The registers all remain: [`../specs/`](../specs/README.md) for what was designed,
[`adr/`](adr/README.md) for what was decided, [`findings/`](findings/README.md) for what was
measured, [`progress-archive.md`](https://github.com/nikhilsrajan/fsd/blob/docs-archive-2026/docs/progress-archive.md) for the day-by-day. This file is the thread
through them, and it ends here until the next era does.
