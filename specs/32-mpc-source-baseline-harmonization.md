---
status: current
summary: MPC as a second source, plus processing-baseline radiometry harmonization.
---

# Spec 32 — MPC source (Sentinel-2 L2A) + processing-baseline harmonization

> **Amendment A1 (2026-10-14, SIGNED OFF by the user 2026-10-14):** `max_cloudcover` means "at most" (≤) on MPC too:
> the STAC query uses `lte`, not `lt` (#145 X5; end of this spec).

> **Status: ✅ DONE — CODE REVIEWED + MERGED + RUNBOOK v2 FULLY VALIDATED ON REAL MPC DATA
> (2026-07-16).** All three steps PASS; correctness debt **#10 is fixed for MPC and proven end to
> end**. Verified independently from the artifacts, not from the `pass` flag (v1 taught us that flag
> could lie):
> - **The cutover boundary was hit exactly** — the real items came back with baselines **`03.00`
>   (pre) → offset `0`** and **`04.00` (post) → offset `−1000`**. `04.00` *is* the first offset
>   baseline, so this exercises `_baseline_tuple(...) >= (4, 0)` **precisely on the boundary**: had
>   the code written `>` instead of `>=`, it would have returned `0` and been silently wrong. Real
>   data landed on the one value that distinguishes them.
> - **Step 3 A/B vs an unharmonized control** (`_result_step3.json`, cube `(2, 550, 606, 1)`):
>   `pre_identical_to_control = true` (offset 0 → bit-identical), and the post-baseline slice equals
>   the control **exactly −1000** across all **202 831** non-clipping pixels (`np.array_equal`, not a
>   tolerance). **Zero** pixels fell in `(0, 1000]`, so nothing clipped and the mean delta is exactly
>   **1000.0**.
> - **The science:** the pre-vs-post gap was **2187.1 DN** unharmonized → **1187.1 DN** harmonized.
>   The fix removed exactly the 1000 DN baseline artifact; the 1187 remainder is real scene change
>   (two January Austria dates). That is precisely the silent error #10 described — a median mosaic
>   over a window spanning both dates would have blended 400 with 2587 where the comparable values
>   are 400 and 1587.
> - **Both spec-flagged open items confirmed live:** `s2:processing_baseline` + `s2:mgrs_tile` exist
>   as assumed, and `storage.transfer` streamed signed MPC HTTPS cleanly (**no `aiohttp` fallback
>   needed** → the fsspec `http` transport open item resolves in our favour).
>
> **Follow-ons logged (none blocking):** **TODO #34** (MPC serves duplicate reprocessed
> acquisitions; fsd doesn't dedup), **TODO #35** (`build_datacube` requires SCL even when masking
> isn't wanted — the v1 step-3 crash), **TODO #36** (CDSE-vs-MPC speed — parked; the local reading is
> confounded by VPN + full-tile copy for a 0.18 % ROI), plus **pinning `planetary-computer`**
> (resolved version now observable from the runbook's install).
>
> **Code review: PASS (Opus@high, 2026-07-16), no code changes required.** Merged to `main`
> (fast-forward, `1cf1568`+`0da4d15`).
> Re-verified independently at review: `pytest -q` **234 passed / 3 skipped**, `ruff check
> src/ tests/` clean. **The #10 guard test is not vacuous** — mutation-tested: removing the
> `_apply_boa_offsets` call makes `test_build_datacube_harmonizes_boa_offset_before_median_mosaic`
> fail; and applying the offset *after* the median would yield `clip(median(200,1200)−1000)=0`,
> not the asserted 200. Ordering (D1/D2), baseline-not-date keying (D3), band exemption,
> catalog back-compat, and `api.download` positional back-compat all confirmed against the spec.
> **Three minor non-blocking notes** (deliberately not fixed — logged, not silently redesigned):
> (1) `sources/mpc._mgrs_tile_from_item` is **dead code** — defined + unit-tested but never called
> (the catalog has no `mgrs_tile` column; `builder._mgrs_tile` derives it from the product id
> instead); harmless, kept for the Phase-2 caller the spec's §1 implies. (2) The reused
> `cdse._finalize_catalog_gdf` raises a **CDSE-worded** message ("CDSE returned non-unique tile
> ids") reachable from the MPC path — cosmetic only (message not asserted anywhere; MPC item ids
> are unique). (3) **`planetary-computer` is unpinned** in the `[mpc]` extra, whereas this spec's
> "Open items" asked to *pin the version* (cf. the `rio-tiler>=6,<8` discipline) — best resolved
> with real data: the runbook's step-1 `pip install -e ".[dev,mpc]"` reports the resolved version;
> pin it then. Implemented to the letter (no redesign); see `CHANGES.md` for the exact landing.
> Originally: **SIGNED OFF (2026-07-16).** Opus@high (interview → grill → cross-validate → spec). A new **data source**:
> Microsoft Planetary Computer (MPC) Sentinel-2 L2A, whose assets
> are **already COGs on Azure** — so this source has **no `jp2→COG` conversion** (unlike CDSE, spec
> 14/25), which is the point. It also fixes **correctness debt #10** (the S2 processing-baseline
> radiometric offset that silently mixes incomparable reflectances across years) at the right layer.
>
> **Conscious deferral at sign-off (2026-07-16):** the source stays **download-based as written**
> (Phase 1 downloads tiles; catalog stores stable local/owned paths). The **stream-in-place
> (`/vsicurl`) vs download/copy-to-`rise`** comparison — including whether a small windowed stream
> read beats a full-tile download for fsd's read pattern — is **deferred to Phase 2**, to be
> measured *after* at-scale cloud datacube creation is set up, not argued in the abstract now. The
> tradeoff (fsd reads only a ~5 km window from a ~110 km tile; full-tile download only amortizes
> under high per-tile cell reuse; on Azure the real axis is MPC-blob-stream vs `rise`-blob-stream,
> gated by rate-limits/ownership/cost not raw read speed) is captured for that later comparison.
>
> **This is Phase 1** of the plan agreed 2026-07-16 (local, hotspot-friendly): build the MPC source
> + nail baseline handling on a **single MGRS tile, single band, two dates straddling 2022-01-25**.
> **Phase 2** (Azure, at scale) is spec 31 (storage seam), retargeted — its old §5 (CDSE
> stage-local-convert-put) is **deleted**, since MPC removes the conversion problem entirely.
>
> **Consciously accepted governance flag:** adding a source is the "build more data sources
> (#11/#21)" work the 2026-07-15 diagnostic parked pending the **rslearn Plan B/C** call (rslearn
> may already ingest MPC → some risk of rework). The user accepted this eyes-open: the source is
> small (reuses the existing `pystac-client` discovery) and it is the fastest unblock for testing
> datacube creation on real Azure-native COGs. [[fsd-diagnostic-triage]]
>
> **Facts confirmed against MPC (web, 2026-07-16 — cited in §7):** MPC serves **raw, unharmonized
> DN**; the per-band offset is **not** in the STAC (`raster:bands` absent on band assets) → it must
> be derived from the item property **`s2:processing_baseline`**; assets are COG
> (`profile=cloud-optimized`); reflectance nodata = **0**.
>
> **Interview decisions locked (2026-07-16), all as recommended:**
> - **D1 — harmonize at BUILD, not download.** Download stays a **pure COG byte-copy** (no
>   re-encode); harmonization is applied **per source image** in the builder, **before** the median
>   mosaic (a calendar window can straddle the baseline change).
> - **D2 — harmonization math = align-to-pre-baseline, uint16.** For baseline ≥ 04.00 reflectance
>   bands: `harmonized = clip(DN − 1000, 0, 65535)`; else unchanged. Keeps fsd's **uint16 +
>   nodata=0** datacube contract. Nodata-safe (0→0). SCL / non-reflectance bands exempt.
> - **D3 — offset source = `s2:processing_baseline`** (per item), **captured into the catalog** as
>   an additive `boa_add_offset` column. Keyed on **baseline, not acquisition date** (MPC
>   reprocessing can stamp a ≥04.00 baseline on a pre-2022 date — the offset then still applies).
> - **D4 — signing = the official `planetary-computer` package** behind a new optional **`[mpc]`**
>   extra; **anonymous** by default (optional subscription key via env for higher rate limits). No
>   `CdseCredentials` for this source.
>
> **Spec-first (spec 24):** this session writes the spec only. A **Sonnet@medium** session
> implements against it. The credentialed/networked demo is a **runbook** the *user* runs
> (hotspot-friendly: one tile, one band), pasting back `_result.json`.

## Motivation

fsd's Azure/Batch endgame wants to read Sentinel-2 that already lives **on Azure as COGs**. MPC is
exactly that: a global S2 L2A archive (2016→present, Sen2Cor, COG) served from Azure West Europe —
the same region as `rise` storage. Sourcing from MPC:

1. **Deletes the `jp2→COG` conversion** (the heaviest, most fragile part of the CDSE path, spec
   25) — MPC assets are already COG, so a download is a pure byte-copy and a build can even stream
   them in place later (Phase 2).
2. **Forces us to handle the processing-baseline offset correctly** (#10). On 2022-01-25 ESA
   introduced baseline 04.00, which adds a `BOA_ADD_OFFSET` (−1000 for L2A reflectance) so
   `reflectance = (DN + offset) / 10000`. MPC does **not** harmonize this and does **not** expose it
   per-band in STAC. A datacube that median-mosaics pre- and post-2022 DN together is **silently
   wrong**. This spec fixes it band-aware, at the layer where it belongs.

Two integration facts from the codebase shape the design:

- **Band-flattening lives in `builder.flatten_catalog`** (tile-row → one row per band file), and
  **per-source-image loading is in the builder's `_add_image_metadata` / `images.load_images`**,
  which runs **before** the `ops.median_mosaic` op-chain. Harmonization must be applied there,
  per image — a datacube-level op is too late (the median would already have mixed baselines; a
  calendar window straddling Jan 2022 is the concrete failure).
- **The catalog schema is a fixed `COLUMNS` list** (`catalog/catalog.py`); the offset is an
  **additive** column (default 0 → CDSE rows and old catalogs are unaffected).

## Scope

**In (Phase 1, local, hotspot-friendly):**
- `sources/mpc.py` — a *general* MPC S2 L2A source (ROI→MGRS tiles, all bands), reusing the STAC
  discovery pattern; **only the test is narrow** (single tile / single band).
- The additive `boa_add_offset` catalog column + its derivation from `s2:processing_baseline`.
- The build-time, band-aware harmonization op (D2), applied per source image before the mosaic.
- `[mpc]` extra (`planetary-computer`); thin `api.download(source="mpc")` wiring.
- pytest (synthetic) + a single-tile/single-band runbook (real MPC, user-run).

**Out (named, deferred):**
- **Azure at scale** (download-to-`rise` / stream-in-place, output artifacts on blob) — **Phase 2 /
  spec 31**. Including the Phase-2 fork: *copy MPC COGs into `rise`* vs *stream MPC in place via
  `/vsicurl` and write only outputs to `rise`* (decided at the Phase-1→2 boundary).
- **Signed-URL expiry / re-sign** for long builds — a non-issue at Phase-1 scale (seconds); a
  Phase-2 concern (TODO).
- **Retrofitting CDSE with the same offset fix** — the column is additive and CDSE-ready, but
  wiring CDSE's baseline capture is a follow-on (TODO). CDSE rows get `boa_add_offset = 0` for now.
- **rslearn Plan B/C** — orthogonal, still parked.

## Design

### 1. `sources/mpc.py` — discovery + download (mirrors CDSE, minus conversion)

- **Discovery**: `pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")`
  with the `planetary_computer` **sign modifier** so returned assets carry SAS-signed HTTPS hrefs;
  search collection `sentinel-2-l2a`, `intersects=ROI`, `datetime=[start,end]`,
  `query={"eo:cloud_cover": {"lt": max_cloudcover}}` (**superseded by Amendment A1**: `lte`).
  Anonymous; optional
  `PC_SDK_SUBSCRIPTION_KEY` env raises rate limits.
- **Asset→band**: MPC keys bands directly (`"B04"`, `"SCL"`, …) — simpler than CDSE's `Bxx_YYm`.
  Map requested `bands` → asset keys; each asset href is a signed COG URL.
- **Download = pure copy**: `storage.transfer(signed_href, dst)` — src is HTTPS (fsspec `http`),
  dst is local (Phase 1). **No `to_cog`, no re-encode** (already COG). The existing atomic
  `.part`+rename in `transfer` gives safe resume (`exists && size>0` skip), same as CDSE.
- **Catalog rows** (mirror `cdse._items_to_gdf` / `_append_downloaded`): `id`, `satellite`
  (`sentinel-2-l2a`), `timestamp` (`item.datetime`, tz-aware UTC), `s3url` (the MPC item href or
  ""), `local_folderpath`, `files`, `cloud_cover` (`eo:cloud_cover`), **`boa_add_offset`**
  (§2), `geometry`. The MGRS tile id comes from `item.properties["s2:mgrs_tile"]` (or the item id).
- **Scope note**: Phase 1 may use a straightforward sequential/threaded download (single tile/band
  is trivial); reusing CDSE's full `download_resume` circuit-breaker orchestration is a Phase-2
  nicety, not required here. `api.download` gains `source: str = "cdse"` and dispatches to
  `sources.mpc.download` when `"mpc"`; `creds` is not required for `source="mpc"` (relax the
  preflight for that branch).

### 2. `boa_add_offset` — derive from `s2:processing_baseline`, store in the catalog

- Add **`boa_add_offset`** (int) to `catalog/catalog.COLUMNS` (before `geometry`, which stays last).
  Backward-compatible: `TileCatalog.read` fills a **missing column with 0** (old catalogs / CDSE).
- Derivation, **per item, keyed on baseline** (not date):
  ```python
  # baseline like "04.00", "05.09", "02.14"
  offset = -1000 if _baseline_tuple(item.properties["s2:processing_baseline"]) >= (4, 0) else 0
  ```
  Stored as the **reflectance-band** offset for the whole tile-row. (The per-band exemption is
  applied at flatten, §3 — SCL/masks are not reflectance.) If `s2:processing_baseline` is missing
  on an item, **raise** (deterministic — no silent 0; this is the correctness-critical field).

### 3. Band-aware application: flatten derives the per-band offset; the builder applies it

- **`builder.flatten_catalog`** gains a per-band **`boa_add_offset`** output column:
  `offset if _is_reflectance(band) else 0`, where `_is_reflectance` = band matches `B01…B12`/`B8A`
  (i.e. `^B\d` or `B8A`); `SCL`, `AOT`, `WVP`, `visual`, … → 0. So each band-flattened row carries
  the exact offset to apply to *that* image.
- **New raster op `raster/ops`-style helper `apply_boa_offset(data, profile, *, offset)`** on the
  locked `(data, profile) -> (data, profile)` convention:
  ```python
  if offset == 0:
      return data, profile
  # upcast to signed to avoid uint16 underflow, align to pre-baseline, clamp, restore dtype
  out = np.clip(data.astype(np.int32) + offset, 0, 65535).astype(data.dtype)
  return out, profile
  ```
  nodata (0) with `offset=-1000` → `clip(-1000,0,·)=0` → stays nodata (order-independent, D2).
- **Applied per source image in the builder's load stage** (in/next to `_add_image_metadata`,
  right after `images.load_images` returns `data_profile_list`, **before** any mosaic op): for each
  non-dropped image `i`, `data_profile_list[i] = apply_boa_offset(*data_profile_list[i],
  offset=catalog_gdf.iloc[i]["boa_add_offset"])`. This guarantees each image is on the pre-baseline
  scale **before** `median_mosaic` collapses a (possibly baseline-straddling) calendar window.
- **Do not** put harmonization in the `ops.run_ops` sequence — that runs on the assembled 5-D cube,
  which is too late (offset is per-source-image, and the median has already mixed baselines).

### 4. Where nothing changes

The catalog query/filter, `area_contribution` multi-tile merge (spec 20), reference-band
resampling (B08), SCL masking, calendar mosaic (spec 15), flatten, and the 5-D band-math contract
are **untouched** — MPC items are the same MGRS granules on the same grid, so the whole
download→datacube→flatten core consumes MPC catalog rows unchanged once §1–§3 land.

## Tests

**pytest (synthetic, no network, deterministic):**
- **baseline→offset** — `_baseline_tuple` parses `"04.00"/"05.09"/"02.14"`; `>= (4,0)` → −1000,
  else 0; a **reprocessed pre-2022 date with baseline 05.xx still yields −1000** (the date-vs-
  baseline trap); missing `s2:processing_baseline` **raises**.
- **flatten band-exemption** — a tile-row with `boa_add_offset=-1000` explodes so `B04` rows carry
  −1000 and `SCL` rows carry 0.
- **`apply_boa_offset` op** — `offset=0` passthrough; `offset=-1000` on a hand-built uint16 array:
  DN 1500→500, DN 500→0 (clamped), DN 0 (nodata)→0; dtype preserved; no uint16 underflow.
- **build-time integration** — a synthetic two-timestamp cube (one "old" image offset 0, one "new"
  image offset −1000, same fabricated tile/band) run through the builder load+mosaic: assert the
  new image is shifted **before** the median, so the mosaic of a baseline-straddling window is
  computed on aligned reflectances (guards the exact #10 failure).
- **catalog back-compat** — reading a parquet without `boa_add_offset` fills 0; round-trip with the
  column preserves it.
- **regression** — full existing suite stays green (CDSE rows default to 0; op is a no-op at 0).

**Runbook (real MPC, user-run) — `runbooks/32-mpc-baseline.md`:**

> **⚠️ CORRECTED 2026-07-16 (runbook v2) — this paragraph as originally signed off was defective,
> and the defect was mine (Opus), not the implementer's.** It survived sign-off, cross-validation,
> implementation, *and* the Opus review. The first real run exposed four errors:
> 1. **"band B04 only" + "build a 2-timestamp datacube" are mutually impossible.** `build_datacube`
>    hardcodes `apply_cloud_mask_scl` → `drop_bands(["SCL"])`, so **SCL is structurally required**;
>    `bands=['B04']` always raises `ValueError: SCL band not present in datacube`. → v2 downloads
>    and builds **B04 + SCL**, which also makes the band exemption **live** (SCL must come back `0`
>    while B04 comes back `−1000`) rather than "moot". The underlying builder limitation is real and
>    deferred to **TODO #35** (needs its own spec — it changes a core contract, and TODO #11's
>    non-optical sources have no SCL at all).
> 2. **"tiny COGs" / "hotspot-sized" was false.** MPC assets are **full-tile (~110 km) COGs** —
>    a single B04 measured **96–272 MB**. v1 also downloaded *the whole date range between* the two
>    chosen items → **9 items / 1.7 GB**, not 2. → v2 uses two **tight ±1 h windows**.
> 3. **The 2-timestamp claim was arithmetically wrong:** `mosaic_days=120` over a 120-day window
>    gives **T=1**. → v2 computes `mosaic_days` from the two acquisitions so each gets its own
>    calendar window.
> 4. **The PASS criteria could not fail:** step 2's `pass` only checked `failed_count == 0` (never
>    the offsets it claimed to test), and step 3's was a "plausible range" judgement. → v2 asserts
>    the offsets explicitly, and replaces the vague check with an **A/B against an unharmonized
>    control** (same cube built twice, offsets forced to 0 in the control): the post-baseline slice
>    must equal the control **exactly −1000** on non-clipping pixels, the pre-baseline slice must be
>    **bit-identical**. A real measurement, not a plausibility call.
>
> **Lesson:** the spec's Tests section was never traced against `build_datacube`'s actual op chain
> or against a real asset size. Cross-validating external facts (§7) did not catch an
> *internal* inconsistency with our own code.

- One MGRS tile, **bands B04 + SCL**, two acquisitions **straddling 2022-01-25** (one baseline
  <04.00, one ≥04.00). Steps: install `.[mpc]`; `download(source="mpc", …)` those two items
  (~320 MB — full-tile COGs); assert the **catalog** records `boa_add_offset` 0 and −1000
  respectively and that flatten exempts SCL (0) while B04 carries the offset; build a 2-timestamp
  datacube **plus an unharmonized control**; assert the harmonized post-baseline slice is the
  control shifted by exactly −1000 DN. `_result.json` per spec 24 with a self-contained `expected`
  block. No conversion, no scale.

## Deliverables (for the Sonnet@medium implement session)

- `sources/mpc.py` (new) — discovery (signed), asset→band, pure-copy download, catalog rows.
- `catalog/catalog.py` — add `boa_add_offset` to `COLUMNS`; read fills missing with 0.
- `datacube/builder.py` — `flatten_catalog` adds per-band `boa_add_offset`; apply `apply_boa_offset`
  per source image in the load stage (before mosaic).
- `datacube/ops.py` (or `raster/…`) — `apply_boa_offset` + `_is_reflectance` helper.
- `api.py` — `download(source="cdse"|"mpc")` dispatch; relax `creds`/preflight for `mpc`.
- `pyproject.toml` — new `[mpc]` extra (`planetary-computer`; `pystac-client` already core).
- `runbooks/32-mpc-baseline.md` (placeholders where needed; hotspot-sized).
- Tests per the Tests section.
- Living docs: `CHANGES.md` (new source; catalog gains `boa_add_offset`; #10 fixed for MPC),
  `DROPPED.md`/none, `TODO.md` (CDSE offset retrofit; Phase-2 stream-vs-copy fork; signed-URL
  re-sign; full `download_resume` orchestration for MPC), `RECIPES.md` (MPC download recipe),
  `specs/31` banner (§5 deleted; retargeted to Phase 2), `specs/10` pointer (MPC = another
  first-class source), `PROGRESS.md`, memory ([[fsd-pipeline-contract]], [[fsd-status]]).

## Best-practice alignment / sources (cross-validated 2026-07-16)

Per-source credit — **exactly what each reliable source contributed to a decision above**:

- **MPC STAC collection JSON**
  (`planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a`) — *verified directly*:
  the `B04` (and other band) assets carry `eo:bands`/`gsd`/`type` but **no `raster:bands`** array,
  and `type` is `image/tiff; …profile=cloud-optimized`. **Contributed:** the facts that MPC exposes
  **no per-band offset/scale/nodata in STAC** (→ must derive from `s2:processing_baseline`, §D3/§2)
  and that **assets are COG** (→ pure-copy download, no re-encode, §D1/§1).
- **MPC issue #134 "Sentinel-2 BOA_ADD_OFFSET harmonisation"**
  (`github.com/microsoft/PlanetaryComputer/issues/134`) — **contributed:** MPC **does not
  harmonize** the offset (open, unresolved; GEE/Sentinel-Hub do) and the `BOA_ADD_OFFSET` "lives in
  the product-metadata XML, not easy to read across many images" (→ we own harmonization at build,
  D1; and derive from the cheap `s2:processing_baseline` property rather than parsing MTD XML, §2).
- **MPC dataset page** (`planetarycomputer.microsoft.com/dataset/sentinel-2-l2a`) — **contributed:**
  the collection description (2016→present, Sen2Cor L2A → COG, Azure-hosted) grounding the
  "already-COG on Azure" premise (Motivation, §1).
- **ClearSKY — "Sentinel-2 Scaling & Harmonization"**
  (`clearsky.vision/knowledge/sentinel2-scaling-harmonization`) — **contributed:** the exact scaling
  relation `ρ = (DN + ADD_OFFSET) / QUANTIFICATION_VALUE`, that the per-band add-offset exists only
  since PB 04.00, and that post-baseline DN encodes values down to ~−1000 vs near-zero before (→ the
  harmonization math and the nodata-collision reasoning, §D2/§3).
- **ESA STEP forum — "Changes in band data after 25 Jan 2022 / baseline 04.00 harmonizeValues"**
  (`forum.step.esa.int/t/…/36270`) — **contributed:** the baseline-04.00 cutover date (2022-01-25)
  and that ESA's own `harmonizeValues` aligns new→old by subtracting 1000 (→ D2's "align-to-pre-
  baseline, clamp ≥0" is the ecosystem-standard choice, not an fsd invention).

## Open items to confirm at sign-off / flag for the implementer

- **Exact STAC keys on a live MPC item** — verify `s2:processing_baseline` and `s2:mgrs_tile`
  property names against a real item (the runbook's first download naturally surfaces them).
- ~~**`planetary-computer` signing API** — `pc.sign_inplace` as a `Client(..., modifier=…)` vs signing
  items post-search; pick one and pin the version (mirror the rio-tiler/pin discipline).~~
  ✅ **RESOLVED (2026-07-16).** `pc.sign_inplace` as the `Client(..., modifier=…)` modifier
  (`sources/mpc.py::_search_items`), **validated against real MPC traffic** by runbook 32 v2. Version
  **pinned `planetary-computer>=1,<2`** — the runbook's install resolved **1.0.0**, so the bound is
  set from a verified fact rather than a guess (that was why the pin was deliberately left open at
  code review). `<2` guards the next major, where `sign_inplace` would be allowed to change; mirrors
  `rio-tiler>=6,<8`.
- **`http` transport for `transfer`** — confirm `storage.transfer(signed_https, local)` streams via
  the fsspec `http` backend cleanly (large-file streaming, no full in-memory buffer); if not, a
  `/vsicurl`-based or `requests`-stream `get` is the fallback.
- **Non-reflectance band set** — `_is_reflectance` must exempt exactly SCL/AOT/WVP/visual/preview;
  confirm no B-prefixed non-reflectance asset exists in the requested band list.

## Amendment A1 (2026-10-14) — cloud cover "at most" on MPC too (#145 X5)

**Status:** SIGNED OFF (user, 2026-10-14), before any code in the X5 PR (#184) was written. Decided by the user on 2026-10-09 (question Q6 of the refactor audit, #145
Decisions comment: "'At most' (≤) on both sources. MPC's `lt` → `lte`.") and 2026-10-14 (the three X5
questions: home here, one test, a small gate-4 run-book). Supersedes the `lt` in "Discovery" (§1).
Line numbers below are at `main` cdfb764, before the X5 code.

**Problem.**
- MPC sends `{"eo:cloud_cover": {"lt": max_cloudcover}}` to the STAC API in both searches:
  `_search_items` (`sources/mpc.py:126`, local `download` and `query_catalog`) and
  `_search_items_unsigned` (`:151`, the AML driver's discovery). The server drops a granule whose
  cloud cover equals the threshold.
- Both sources then filter locally with `<=` (`cdse._finalize_catalog_gdf`, `sources/cdse.py:284`;
  MPC imports it at `mpc.py:47`). CDSE sends no cloud query to its server, so on CDSE a granule at the
  threshold is kept.
- So the same `max_cloudcover` keeps a granule on CDSE and drops it on MPC, and `max_cloudcover=0`
  returns no granule at all on MPC, not the cloud-free ones.

**Decision.**
- **X5.1** Both MPC searches send `{"eo:cloud_cover": {"lte": max_cloudcover}}`. `max_cloudcover`
  means "cloud cover at most this" on every source. Nothing else changes: CDSE and the local `<=`
  filter already mean that.

**Prior art.**
- STAC API Query extension (https://github.com/stac-api-extensions/query, checked 2026-10-14; maturity
  "Stable"): the comparison operators are "eq, neq, lt, lte, gt, gte, startsWith, endsWith, contains,
  in". This is where `lte` is defined.
- MPC's conformance list (https://planetarycomputer.microsoft.com/api/stac/v1/conformance, checked
  2026-10-14) includes `https://api.stacspec.org/v1.0.0/item-search#query`: MPC implements that
  extension, `lte` included. Gate 4 checks it against a live search.
- In-repo: the local filter already uses `<=` (`cdse.py:284`).

**How to verify.**
- **T1** (new file) parametrised over `_search_items` and `_search_items_unsigned`, with
  `pystac_client.Client.open` replaced by a fake that records the `query` passed to `search` (and
  `mpc._import_pc` stubbed for the signed one): called with `max_cloudcover=0.066284`, each records
  `{"eo:cloud_cover": {"lte": 0.066284}}`. Catches: either site left at `lt` (no existing test reads
  the query: every MPC test replaces the whole search function).
- **Gate 4** (the change decides which real granules come back): `runbooks/145-x5-mpc-cloudcover.ipynb`,
  live MPC searches with no download, on the tutorial fixture's ROI (`s2grid=4772924`) and window
  (2018-04-01 to 2018-10-01). An unfiltered search first reads the lowest live cloud cover `c` in
  that window and its item (the fixture's catalog records 0.066284 for
  `S2B_MSIL2A_20180928T100019_R122_T33UWP`). PASS:
  (a) `fsd.sources.mpc.query_catalog(..., max_cloudcover=c)` returns that granule;
  (b) at two thresholds `t` (`c`, and the median cover of an unfiltered raw search), a raw
  `pystac_client` search with `{"lte": t}` returns **every** item the unfiltered search lists with
  cover `<= t` (none dropped), and a raw search with `{"lt": c}` returns none at `c`.
  (a) alone would also pass if MPC ignored the filter, since fsd filters again locally with `<=`.
  (b) checks the property fsd relies on: MPC's `lte` never drops an item at or below the threshold.
  It does not require MPC to drop items above it: the first gate-4 run (2026-10-14) showed MPC's
  `lte` at `c` = 0.066284 returning 5 items, up to 0.176506, while `lt` returned none. The Query
  extension lists the operator but does not define it. fsd's local `<=` filter
  (`_finalize_catalog_gdf`) removes the extra items, so the result fsd returns is exactly "at
  most". (Rewritten twice in gate-3 review and gate 4.) All searches run on the driver, so a local
  run counts; no AML run and no image build.

**Risks.**
- A run whose `max_cloudcover` equals a granule's cloud cover gets one more granule on MPC than
  before. Values are floats with six decimals, so this matters mostly for whole numbers such as 0.

**Out of scope.**
- Merging the two MPC search functions, which build the same query.
- A run-book step for `discover_shard_rows` (the AML discovery path): T1 pins its query, and gate 4's
  (a) shows MPC honours `lte`.
