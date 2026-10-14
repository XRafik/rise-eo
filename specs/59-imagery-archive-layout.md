---
status: current
summary: The imagery archive is flat per run, keyed by each provider's own item id, and silently double-counts one acquisition fetched from two sources. Lay it out as `{root}/{collection}/YYYY/MM/DD/{granule}/`, name each granule by its canonical (provider-independent, processing-inclusive) name so the same granule from two sources collides and two processings never do, select a processing per acquisition with one `processing=` grammar on both `download` (default "latest") and the build verbs (default: duplicates raise), record provenance, and publish a band file only after it is stamped. Closes #74; prepares #101.
---

# Spec 59 — imagery archive layout

**Status:** **SIGNED OFF (user, 2026-09-30)**, including §8's derived point. Decisions D1–D12 were each confirmed by the user in the
grilling session of 2026-09-29; this document writes them down and adds acceptance criteria.
**Amended 2026-09-30 (before sign-off):** D7 rewritten — `download` takes `processing=`; D10 rewritten —
the byte copy was already atomic, the defect is stamping after publishing (§8).
**Amendment A1 (2026-10-03, SIGNED OFF by the user 2026-10-03):** D5 adds one public helper,
`fsd.archive_catalog_filepath(dst_folderpath, collection)`, so no caller rebuilds the archive's
inner layout by hand (found in the D12 review: five hand-built `{root}/{collection}/catalog.parquet`).
**Amendment A2 (2026-10-13, SIGNED OFF by the user 2026-10-13):** a `setup`-written `input.csv` with no `processing` column is
refused, extending D8's "no shim" (#145 X2; end of this spec).
**Amendment A3 (2026-10-14, DRAFT):** a catalog row or file with a missing or empty `offset` or `nodata`
is refused instead of read as 0; an empty `scale` is filled from the declaration (#145 X4, #173; end of
this spec).
· **Opened:** 2026-09-29
**Closes:** [#74](https://github.com/nikhilsrajan/fsd/issues/74) (a download stamps its file in place after publishing it — retitled 2026-09-30).
**Prepares:** [#101](https://github.com/nikhilsrajan/fsd/issues/101) (guard inference against processing
versions the model was not trained on) — this spec records the provenance #101 compares.
**Related:** [spec 02](02-catalog.md) (the catalog), [spec 33 research](research-s2-reprocessing-dedup.md)
(the S2 reprocessing dedup this generalizes), [spec 34](34-ingest-normalization-contract.md) (per-row
`offset`), [spec 58](58-collection-agnostic-verbs.md) (D4 cube digest, D9 `properties_filter`, D12
schema policy, D13/D14 variants), ADRs [0006](../docs/adr/0006-catalog-is-file-based-geoparquet.md),
[0011](../docs/adr/0011-ingest-stores-raw-dn-declares-radiometry.md),
[0030](../docs/adr/0030-source-and-collection-are-orthogonal-axes.md),
[0031](../docs/adr/0031-collection-strings-resolve-on-the-driver.md), and the new
[ADR 0032](../docs/adr/0032-the-archive-is-lossless-duplicates-raise.md).
**Origin:** the user, 2026-09-29, after the first green S1 AML run: `imagery/` "looked crowded", and
the same acquisition from MPC and CDSE should not be two unrelated things — *"the raw data matters,
not the source"*. The grilling session then reversed the first sketch (collide on acquisition, latest
processing replaces) because a silent replace is exactly the class of bug fsd has already paid for
(the Austria archive's ~1000 DN radiometry debt).

---

## 1. The problem

Four facts, all verified in code or against live data on 2026-09-29.

1. **Flat and provider-shaped.** MPC writes `{root}/{item.id}/{band}.tif`
   (`sources/mpc.py` `_select_item_files`). CDSE mirrors EODATA,
   `{root}/Sentinel-2/MSI/L2A/YYYY/MM/DD/{SAFE}/{band}.tif` (`cdse._download_folderpath`). The Austria
   archive is 184 granule folders in one directory, and the two sources lay out the same collection
   differently.
2. **The same granule has two names.** MPC's item id drops ESA's baseline field
   (`S2B_MSIL2A_20180918T100019_R122_T33UWP_20201009T023142`); CDSE's id is the full ESA product name
   (`…_N0500_R122_T33UWP_2023…`). The catalog key is `id` alone (`TileCatalog.append`,
   `drop_duplicates(subset="id")`), so one acquisition fetched from both sources is **two rows**, and the
   median mosaic counts it twice. Nothing warns.
3. **Same acquisition ≠ same bytes.** MPC's 2018 S2 L2A is **not** ESA's product: its
   `s2:granule_id` reads `S2B_OPER_MSI_L2A_TL_ESRI_20201009T023150_…_N02.12` — Sen2Cor run by Esri in
   2020 at baseline 02.12 (offset 0). CDSE serves ESA's Collection-1 reprocessing at baseline 05.00
   (offset −1000). A layout that collided them on acquisition would union band files from two
   processings under one row's single `offset` (`TileCatalog.append` unions `files` per id) — silently
   wrong radiometry.
4. **One catalog file is one collection** (`TileCatalog.append` raises "declaration conflict"), so
   `sentinel-1-rtc` and `sentinel-2-l2a` cannot share `imagery/catalog.parquet` today — loud, but it
   forces a per-collection root on the caller. HLS (spec 58 P3) hits the same wall.

Two more, found while designing:

5. **CDSE discovery never dedupes.** MPC collapses same-acquisition items to the newest
   `s2:generation_time` (spec 33, `_dedupe_reprocessed_items`); CDSE only asserts id uniqueness
   (`cdse._finalize_catalog_gdf`), although CDSE documents near-duplicate products by design (§9).
6. **The idempotency skip can trust an unstamped file** (#74, re-scoped 2026-09-30). The byte copy is
   already atomic — `fs.transfer` streams into `<dst>.part` and renames on success (`storage/fs.py`,
   since `19ba57f`, 2026-07-02); #74's original "truncated file under the final name" premise missed
   that. What is **not** atomic is the step after it: `stamp_or_reencode` edits the file **in place,
   under its final name** (`"r+"`), in both `mpc._transfer_and_stamp_one` and `cdse._convert_one`. A
   kill or an exception in that window leaves a non-empty, final-named file without its scale/offset/
   nodata tags (or with a half-edited header), and the `size > 0` skip accepts it forever. CDSE is the
   sharper case: a stamp exception returns `"ConvertError"` but leaves the unstamped `.tif` on disk.
   Under this spec's collision rule such a file would also count as done for *every other source* that
   later tops up the same granule — the bug gets wider, so this spec fixes it.

## 2. Scope

**In:** the on-disk layout; granule naming; the acquisition key; catalog schema; the build-time
duplicate check; the `processing=` selector on `download` and the build verbs (incl. CDSE, which never deduplicated); provenance in cube
metadata and the training-data stamp; stamp-then-publish downloads (#74); the verbs' path semantics; notebooks,
docs and the re-download run-book.

**Out:** the train-vs-inference processing guard (**#101**, its own spec — this spec only records what
it needs); two runs downloading into one archive concurrently (§6); a migration tool for existing
archives (standing policy, spec 58 D12 — re-download instead); SNAP-compatible `.SAFE` layout (#6,
unaffected).

## 3. Decisions

### D1 — The archive is lossless; duplicates raise, never replace

Nothing is ever overwritten or hidden by a newer processing. Two processings of one acquisition
**coexist on disk**; the ambiguity surfaces as an error at the one place it matters — a build that
would mosaic both. Full rationale: **ADR 0032**.

Cross-source comparison is the caller's choice of root: to compare MPC against CDSE for the same
acquisitions, download each into a **different `dst_folderpath`**. Inside one archive, fsd does not
keep sources apart — it keeps *processings* apart (D3).

### D2 — Layout: `{root}/{collection}/YYYY/MM/DD/{granule}/`

```
{root}/
  sentinel-2-l2a/
    catalog.parquet
    2018/09/18/
      S2B_MSIL2A_20180918T100019_N0212_R122_T33UWP_20201009T023142/
        B04.tif  B08.tif  SCL.tif
  sentinel-1-rtc/
    catalog.parquet
    2018/06/26/
      S1B_IW_GRDH_1SDV_20180626T165009_20180626T165034_011547_015393_rtc/
        vh.tif  vv.tif
```

- `{collection}` is the **STAC collection id the bytes were fetched as** — never a build variant name
  (spec 58 D13/D14). A variant cannot be downloaded (the name goes to the provider's STAC search
  verbatim), so this is already true; the spec makes it a rule.
- `YYYY/MM/DD` is the **acquisition date (UTC)** from the collection's granule-name parser (D4), falling
  back to the UTC date of the catalog `timestamp` for a collection without one. It must come from the
  name, not `timestamp`: HLS's STAC `datetime` differs from the sensing time in its own id (§9), and the
  date level must be identical for one granule whatever source fetched it.
- **One `catalog.parquet` per collection directory.** Spec 58's "one catalog = one collection = one
  declaration" rule is unchanged; it now matches the directory structure instead of fighting it.
- CDSE's EODATA mirror (`Sentinel-2/MSI/L2A/…`) is retired; both sources write the same layout.

### D3 — A granule folder is named by its canonical granule name

The **canonical granule name** keeps every processing field and drops every provider-specific one, so
it is **equal across sources exactly when the bytes are the same processing**:

| collection | canonical granule name | from MPC | from CDSE |
|---|---|---|---|
| `sentinel-2-l2a` | ESA product name without `.SAFE` | `s2:product_uri` minus `.SAFE` | item `id` |
| `sentinel-1-rtc` | MPC item id | item `id` | not served (spec 58 D15) |
| `hls2-s30`, `hls2-l30` | item id, incl. `.v2.0` | item `id` | not served |
| any other | item id | item `id` | item `id` |

- Same granule from MPC, then CDSE → same folder → the transfer is skipped, a new band is added
  (`files` unioned, same processing, same `offset` — correct).
- Different baseline → different folder → both kept; D6 raises at build.
- The catalog **`id` becomes the canonical granule name**, so the catalog key collides exactly when the
  folder does.

The parser lives in the collection's module (`fsd/collections/s2_l2a.py` etc.) as a plain function,
**not** on `CollectionDeclaration` — a declaration may not hold callables (ADR 0031). It is called only
where items are discovered: on the driver for MPC local and AML, and inside the single CDSE job on
the node, where it is ordinary installed fsd code (not a registry lookup, so ADR 0031's concern does
not arise). A user-registered collection without a parser gets the "any other" row.

### D4 — The acquisition key detects duplicates; it never names a folder

The **acquisition key** is the canonical granule name minus its processing fields — the identity of
the physical observation:

| collection | acquisition key | dropped |
|---|---|---|
| `sentinel-2-l2a` | `S2B_MSIL2A_20180918T100019_R122_T33UWP` | `_N0212`, trailing discriminator |
| `sentinel-1-rtc` | `S1B_IW_GRDH_1SDV_20180626T165009_20180626T165034_011547_015393` | `_rtc` |
| `hls2-*` | `HLS.S30.T33UWP.2021123T100031` | `.v2.0` |
| any other | the granule name itself (no cross-processing detection) | nothing |

The split mirrors Landsat's product id, which separates acquisition fields (sensor, path/row,
acquisition date) from processing fields (processing date, collection, tier) (§9). It is **stored** as
a catalog column (D8), so the build-time check never re-derives it on a node.

⚠️ **Known false positive, carried over from spec 33:** a datatake split across two datastrips yields
two S2 products with the same acquisition key that may cover different parts of the tile (§9, CDSE
forum). D6 and D7 treat them as duplicates, as spec 33 already does for MPC.

⚠️ **Known false negative:** S1 NRT and reprocessed slices of one datatake can be cut at different
start/stop times, giving different keys for overlapping data. Not detected; stated, not solved.

### D5 — `dst_folderpath` is the archive root; `download` returns the collection's catalog

- `download(dst_folderpath=root, collection=c)` writes under `{root}/{c}/` and returns
  `{root}/{c}/catalog.parquet` (as today, a catalog path comes back).
- `create_training_data`, `run_inference`, `verify_adapter` keep **`catalog_filepath=`** unchanged.
- `create_training_data(download=True)` derives the root as the catalog's **grandparent**.
- Preflight raises when a `catalog_filepath`'s parent directory name is not `collection=`, naming both
  — the directory is now part of the contract, so a mismatched pair is caught before any work.
- **A1 (signed off 2026-10-03).** The archive root is the only path a caller writes; everything under
  it is fsd's layout. A caller that needs the catalog path *before* downloading (CDSE's
  `download_resume` takes it as input; a notebook's Settings cell passes it to
  `create_training_data`) gets it from **`fsd.archive_catalog_filepath(dst_folderpath, collection)`**
  → `{dst_folderpath}/{collection}/catalog.parquet`, exported in `fsd.__all__`.
  - **One source of truth:** `download` computes its return value by calling this helper (it wraps
    the private `_granules.collection_root`), so the returned path and the helper's path cannot
    diverge.
  - **Pure path function:** no I/O, no storage configuration, no collection validation — a wrong
    `collection` already fails D5's directory preflight at the verb that uses the path.
  - **Name:** not `catalog_filepath`, because six verbs in `fsd.api` take a `catalog_filepath=`
    parameter that would shadow it inside their bodies.
  - **Callers move to it:** `demos/e2e_austria.py`, `demos/e2e_austria_aml.py` (or `download`'s
    return value, where the call comes first), `notebooks/e2e_austria_aml.ipynb`. Tests that *assert*
    the layout keep the literal path — a contract check spells out the expected answer.

### D6 — A build over >1 processing of one acquisition raises; `processing=` selects per acquisition

After `properties_filter` (spec 58 D9) and the window filter, the surviving rows are grouped by
`acquisition_key`. A group with more than one row is a **duplicate acquisition**.

New verb parameter on every verb that builds (`create_training_data`, `run_inference`,
`verify_adapter`), alongside `properties_filter`:

`processing: str | None = None`

| value | per acquisition group |
|---|---|
| `None` (build default) | raise if any group has >1 row |
| `"latest"` | keep the latest row (ordering below) |
| a PEP 440 specifier, e.g. `">=04.00"`, `"==05.00"`, `">=05.00,<06"` | keep the rows whose `processing_version` satisfies it, then as `"latest"`; a group left empty is **dropped and reported** |

**One grammar, one meaning, two decision points.** The same `processing=` values mean the same thing
on `download` (D7), where they choose what to *fetch*, and on the build verbs, where they choose what to
*use*. Only the defaults differ, deliberately: a download must pick something from what a provider
offers, so it defaults to `"latest"`; a build over what is already on disk must not pick silently, so it
defaults to `None`.

**"Latest" ordering.** Within a group, rows are ordered by `processing_version` (PEP 440), then by
`processing_datetime`; a null sorts below any value. `"latest"` keeps the maximum. **It raises only
when it must choose and cannot** — two or more rows that tie on both keys (e.g. two S1 RTC granules of
one acquisition, which carry neither), naming them. A group of one never needs ordering, so `"latest"`
is harmless on a collection with no version.

- **Per acquisition, not per window.** A plain row filter (`properties_filter={"processing_version":
  "05.00"}`) would also drop dates where only 02.12 exists; that is why selection is a separate knob
  (the user's case, 2026-09-29).
- **Mixing processings across dates is sound** at the offset level: the builder applies each row's
  `offset` before the median (`datacube/builder.py` `_apply_offsets`). Processor-level differences
  (Esri Sen2Cor 2020 vs ESA Collection-1) remain a science caveat, as they already are for any cube
  spanning the 2022 baseline cutover — that is #101's territory.
- **Parsing uses `packaging.specifiers.SpecifierSet`**, already a core dependency. PEP 440 normalizes
  S2's zero-padded baselines: `SpecifierSet("<=5.11").contains("05.11")` is `True`, and
  `SpecifierSet(">=04.00").contains("02.12")` is `False` (checked 2026-09-29).
- **A version specifier against a collection that publishes no version (S1 RTC) raises at preflight**,
  naming the collection — the same "don't silently no-op" rule as spec 58 D6's `max_cloudcover` gate.
  `"latest"` and `None` do not raise there; they behave as the ordering rule above says.
- **The error is the discovery mechanism** (as spec 58 D9): it lists each duplicate group with `source`,
  `processing_version`, `processing_datetime`, and prints the arguments that would resolve it.
- **Where it runs:** the same choke points as D9 — `workflows.create_datacube.setup` (so coverage
  accounting sees the rows the build will use) and `build_datacube` (so no entry point routes around
  it). Idempotent.
- **Cube identity:** `processing` joins `params_key` exactly as `properties_filter` did — appended only
  when not `None`, so **no existing cube path moves** for a build that does not use it.
- `properties_filter` gains the ability to match the new first-class columns `source` and
  `processing_version` (today it matches the `properties` JSON only, `catalog.filter_by_properties`).
  `source` is a set (D8); a filter value matches a row whose set contains it. These two names are
  **reserved**: they resolve to the column, never to a same-named key inside `properties`, and the
  unknown-key raise (D9.1) treats them as always carried.

### D7 — `download` selects a processing per acquisition, defaulting to `"latest"`, and reports what it leaves ambiguous

**Which processing does `download` fetch?** Whatever the chosen `source` offers, narrowed by a new
`processing=` parameter on `download` — D6's grammar, applied per acquisition to **one provider's
search results**, after `properties_filter` and **before** `max_tiles` (so the cap counts what will
actually transfer, as spec 58 D9 already requires). Added 2026-09-30 after sign-off review: the user
asked which processing `download` gets, and the draft's answer ("the provider's newest, no choice")
meant a version could only be chosen by switching source.

`processing: str = "latest"`

| value | what one download fetches, per acquisition |
|---|---|
| `"latest"` (default) | the provider's latest (D6 ordering) — spec 33's MPC rule, generalized from `(item.datetime, s2:mgrs_tile)` to the acquisition key and **extended to CDSE**, which never deduplicated (CDSE's own guidance is "use the most recent", §9) |
| a specifier, e.g. `"==05.00"` (match training data), `">=05.00"` (refuse old baselines) | the latest processing that satisfies it; an acquisition with none at this source is **skipped and reported**, e.g. CDSE has deleted pre-Collection-1 baselines, so `"==02.12"` from CDSE mostly finds nothing |

- **Every choice is printed** — each skipped granule and each acquisition dropped by a specifier — so
  nothing a download decides is silent.
- `None` is not accepted on `download`: "fetch every processing the provider still serves" is not a
  use case anyone has asked for, and on MPC — which keeps superseded items live (spec 33 research, Q1)
  — it would fetch processings the provider itself treats as obsolete. Passing it raises, naming
  `"latest"`. *(Derived while writing D7; accepted at sign-off, 2026-09-30.)*
- **The ambiguity is reported when it is created.** After the catalog append, `download` prints how
  many of the acquisitions it touched now hold >1 processing in the archive, with the versions and
  sources, and the build argument that will be needed:

  ```
  [download] 184 acquisitions; 184 matched processing=">=05.00" at cdse
  [download] archive now holds >1 processing for 184 of them (mpc 02.12, cdse 05.00);
             builds over them need processing="latest" or a specifier
  ```

- **Forwarded at every hop, per runner** — `api.download` → `sources.mpc.download` /
  `discover_shard_rows` (driver-side, AML) / `sources.cdse.download` (local, and inside the single CDSE
  AML job, where it rides the command line as a plain string). Spec 58 P2's bug 1 was exactly a kwarg
  that reached one runner branch and not the other; AC 15 pins all four.
- `create_training_data(download=True)` forwards its own `processing` to the download leg when it is
  `"latest"` or a specifier; when it is `None`, the download leg uses its default `"latest"` — so a
  training run fetches one processing per acquisition and the build then sees no ambiguity it did
  not already have.
- `processing` on `download` does **not** enter any cube path; only the build's value does (D6).

Anything that accumulates **across** downloads (MPC reprocesses after your first download; you add CDSE
later) still lands in its own folder, is reported by the line above, and raises at build under D6's
default.

### D8 — Catalog schema

| column | change | value |
|---|---|---|
| `id` | **meaning changes** | canonical granule name (D3) |
| `acquisition_key` | **new** | D4 |
| `processing_version` | **new** | normalized string: S2 `"05.00"` (MPC `s2:processing_baseline`, CDSE `processing:version`), HLS `"2.0"` (from the id), S1 RTC null |
| `processing_datetime` | **new** | UTC; S2 MPC `s2:generation_time`, CDSE `processing:datetime`; null when the provider exposes none |
| `source` | **new** | comma-joined sorted set of sources that contributed files (`"mpc"`, `"cdse,mpc"`), unioned on append like `files` |
| `local_folderpath` | value changes | now under D2's layout |

Every other column is unchanged. The STAC export (`catalog/stac.py`) writes `processing_version` /
`processing_datetime` as the STAC processing extension's `processing:version` /
`processing:datetime` (§9).

**No shim, no migration tool** (spec 58 D12's standing policy): a pre-59 catalog is not read. It raises
naming the missing columns and "re-download". The re-download is D12's cost.
> **Extended by Amendment A2 (2026-10-13, end of this spec):** a `setup`-written `input.csv` with no
> `processing` column is refused too.
> **Extended by Amendment A3 (2026-10-14, end of this spec):** so is a catalog row or file whose
> `offset` or `nodata` is missing or empty; an empty `scale` is filled from the declaration.

### D9 — Provenance is recorded in every cube and training-data stamp

Each cube's `metadata.pickle.npy` and each `_flatten_stamp.json` gain:

```json
"provenance": {"sentinel-2-l2a": {"ids": ["S2B_MSIL2A_…", "…"],
                                  "processing_version": ["02.12", "05.00"],
                                  "source": ["cdse", "mpc"]}}
```

`ids` only in the cube (one cube = a bounded set of granules); the stamp carries the version and
source sets only (a training set spans thousands of granules). **Nothing reads provenance in this
spec** — it is recorded so #101 can compare training against inference, and so a stale cube (D11) is
diagnosable. It is metadata, not identity: it does not enter `params_key`.

### D10 — Only a fully stamped file is ever published under its final name (closes #74)

**Rule: stamp, then publish.** The rename (or put) that creates `dst` is the **last** step of an
ingest, never followed by an in-place edit. Rewritten 2026-09-30 (the user asked what `.part` + uuid
changed): the draft proposed `<dst>.part-<uuid4>` for the byte copy, which is already atomic (§1
fact 6); the real defect is the order of rename and stamp.

```
stage = f"{dst}.stage"
fs.transfer(src, stage)       # unchanged, already atomic: stage.part -> rename -> stage
stamp_or_reencode(stage)      # the in-place edit now happens on the staged file
os.replace(stage, dst)        # the ONLY step that creates dst
```

- **MPC, local destination:** as above. A kill anywhere before the last line leaves no `dst`, so the
  next run re-fetches it; a leftover `.stage` is simply overwritten by the next attempt.
- **MPC, remote destination:** already this shape (local scratch → stamp → `fs.put`); unchanged.
  Whether that put is atomic per backend (adlfs block-list commit) is **not verified**, and this spec
  does not claim it; the run-book (§4) checks it by killing a transfer mid-put.
- **CDSE:** `to_cog(jp2, stage)` → stamp `stage` → `os.replace(stage, dst)`. A stamp exception now
  removes `stage` (in the existing `finally`) and leaves no `.tif` at all, instead of an unstamped one.
- **Name:** `<dst>.stage`, not `<dst>.part` — `fs.transfer` already uses `.part` for its own sidecar,
  and reusing the suffix would give `B04.tif.part.part` mid-copy. Fixed, not per-attempt: a per-attempt
  `uuid4` suffix only helps two writers of one destination **at the same moment**, which cannot happen
  within one run (each asset is in exactly one shard) and is out of scope across runs (D12, §6).
- Implementation check (AC 12): GDAL identifies GeoTIFF by content, so `stamp_or_reencode` on a
  `.stage` name must still take the in-place path; the re-encode fallback must write a COG regardless
  of the missing `.tif` extension. Test both rather than assume.

### D11 — The build-skip stays a presence test

Spec 49's skip does not re-check which granules a cube was built from. A cube built with
`processing="latest"` before a newer processing arrived keeps its path and is skipped on re-run —
the same behaviour as any catalog growth today, and deliberately so: the cube is addressed per unit,
never by hashing a set of granules (memory rule, spec 58 D13's near-miss). D9's provenance makes the
staleness detectable; the escape is a new `run_folderpath` or deleting the cube. Stated in §6.

### D12 — One shared archive in notebooks and docs; the re-download is owned here

- Notebooks, how-tos and the tutorial point `dst_folderpath` at **one shared archive**:
  `{AZ_ROOT}/imagery` on blob, `tests/outputs/imagery` locally. Runs keep their own
  `{AZ_ROOT}/<demo>/runs/<id>` for cubes, training data and outputs. A repeat run downloads nothing
  (#64's shortfall diff returns zero).
- The AML shard-catalog merge (`_merge_shard_catalogs`) stays single-writer **per run**; two runs
  downloading into one archive at once is out of scope (§6).
- **Re-download, no migration** (standing policy): the Austria S2 archive (184 granules, 67.2 GB,
  `B04,B08,SCL`, MPC) and the S1 imagery under blob run folder `demo-20260928T142634Z` are re-fetched
  into the new layout by the run-book. Every existing cube path, `_flatten_stamp.json` and known-empty
  manifest under those roots is regenerated with them — not because `params_key` moves (it does not,
  D6), but because the catalogs they were built from are replaced.

## 4. Phases

**P1 — code, network-free.** D2–D11. Ends green on pytest with synthetic items: MPC and CDSE fixtures
for the *same* ESA product resolve to the same folder and `id`; two baselines resolve to two folders
and one acquisition key.

**P2 — run-book, then notebooks.** Re-download Window A (`s2grid=4772924`, S2 + S1) into a fresh shared
archive; add one CDSE granule for an acquisition MPC already holds, to exercise D6 on real data; kill
one transfer mid-flight locally and mid-put on blob (D10). Then D12's notebook/doc repointing and the
full Austria re-download.

## 5. Acceptance criteria

**P1**

1. `pytest -q` and `ruff check src tests demos examples` clean.
2. An MPC item and a CDSE item for the **same** ESA product yield the same canonical granule name,
   the same `{collection}/YYYY/MM/DD/` folder, and one catalog row whose `source` is `"cdse,mpc"`.
3. Two S2 items of one acquisition at baselines 02.12 and 05.00 yield two folders, two rows, **one**
   `acquisition_key`.
4. An HLS fixture whose STAC `datetime` falls on a different UTC date from its id's sensing time is
   filed under the **id's** date.
5. A build over AC3's catalog with `processing=None` raises, listing the group with source, version
   and processing datetime and the resolving arguments.
6. On a catalog where acquisition A has {02.12, 05.00} and acquisition B only {02.12}:
   `processing="latest"` keeps A@05.00 and B@02.12; `processing=">=04.00"` keeps A@05.00, drops B,
   and reports the drop; `processing="==05.00"` likewise.
7. Any version specifier against `sentinel-1-rtc` raises at preflight naming the collection, on
   `download` and on a build. `processing="latest"` downloads and builds S1 as before; two S1 rows of
   one acquisition (no version, no datetime) make `"latest"` raise naming both (D6 ordering).
8. An S2 build with `processing=None` resolves to **the same cube path it did before this spec**;
   two builds differing only in `processing` resolve to different paths.
9. `download` with the default `"latest"`: CDSE given two items of one acquisition keeps the later one
   and prints the skipped one (D7); MPC selects identically to before on spec 33's fixtures.
10. `properties_filter={"source": "mpc"}` matches a row whose `source` is `"cdse,mpc"`;
    `{"processing_version": "05.00"}` matches the first-class column.
11. `create_training_data(catalog_filepath=".../sentinel-1-rtc/catalog.parquet",
    collection="sentinel-2-l2a")` raises at preflight naming both (D5).
12. A stamp that raises (MPC and CDSE) or a kill injected between transfer and stamp leaves **no file
    under the final name**; a re-run transfers it again rather than skipping (D10). A file published
    through the `.stage` path carries the declared scale/offset/nodata tags, via both the in-place
    stamp and the forced re-encode fallback.
13. A pre-59 catalog raises on read, naming the missing columns and "re-download" (D8).
14. A built cube's metadata and `_flatten_stamp.json` carry D9's `provenance` block.
15. `download(processing=...)` reaches the selection on **all four paths** — MPC local, MPC AML
    (`discover_shard_rows`, driver-side), CDSE local, CDSE AML (the node's command line) — pinned by
    forwarding tests in the shape of spec 58 P2's bug-1 tests; `create_training_data(download=True)`
    forwards a specifier, and forwards `"latest"` when its own value is `None`.
16. `download(processing=">=05.00")` on a fixture offering {02.12, 05.00} for acquisition A and only
    {02.12} for B fetches A@05.00, skips B and reports it; `max_tiles` counts after the selection;
    `download(processing=None)` raises naming `"latest"`.
17. A download that leaves an acquisition with >1 processing in the archive prints D7's report line
    naming the versions and sources; one that leaves none prints nothing extra.

**P2**

18. Window A run-book green: S2 and S1 downloaded into one shared root land in two collection
    directories with two catalogs; a second identical download reports zero missing assets.
19. The added CDSE granule for an acquisition MPC already holds (different baseline) makes the next
    build raise, and that download printed D7's report line; `processing="latest"` then builds, and
    the cube's provenance names both sources' versions only where they were used.
20. Blob mid-put kill: the run-book records whether a partial blob is visible under the final name.
    Either outcome is a result; if it is visible, D10's remote leg becomes a filed follow-up, not a
    silent pass.
21. Visual QGIS check of one S2 and one S1 cube from the new layout (per `CLAUDE.md`).

**A1 (signed off 2026-10-03)**

22. The offline download test (`tests/test_spec59_p1.py:624`, D5) also asserts the returned path
    `== fsd.archive_catalog_filepath(root, S2)`; a pure-function test pins the helper for a local
    root and an `abfss://` root.
23. No Python under `demos/` and no code cell in a tracked notebook builds `…catalog.parquet` from a
    collection by hand (a pytest grep). Prose in `docs/` that *explains* the layout is exempt.

**P2 Window A result (2026-09-30, `runbooks/59-p2-window-a.ipynb`): AC 18–21 green.** AC 20: a
blob upload killed 1.5 s in left **no blob** under the final name (observed once, at one delay; the
mechanism was not examined), so D10's remote leg needs no follow-up. AC 19's D7 line was not captured on the real
run (never asserted; AC 17's tests cover it).

## 6. Risks

- **The re-download is the long pole**, again (67.2 GB S2 + S1 scenes at ~3.7 GB each). P1 is
  network-free so the code lands first.
- **Stale cubes (D11).** A skipped cube can reflect an older processing than `"latest"` now resolves
  to. Detectable via provenance; not prevented.
- **Concurrent writers.** Two runs downloading into one shared archive can race on the collection's
  `catalog.parquet` (last writer wins). The shared archive (D12) makes this *possible* where per-run
  roots made it impossible. Acceptable while there is one user; revisit with contributor readiness.
- **Datastrip-split false positives (D4)** — genuinely different coverage treated as duplicates,
  inherited from spec 33.
- **`s2:product_uri` absent on an MPC item.** The S2 parser then cannot recover the baseline field; it
  **raises** naming the item rather than falling back to the MPC id (which would silently break D3's
  cross-source equality).
- **`source` as a set** loses which band came from which source. Deliberate: the same canonical name
  means the same processing, so the bytes are interchangeable; per-file provenance would be noise.

## 7. Alternatives considered

- **Collide on the acquisition key, latest processing replaces** (the first sketch). Rejected by the
  user: a silent replace can change radiometry under an existing catalog. Earth Search's Landsat ids
  work this way and its README concedes some historical items still point at superseded processings
  (§9). ADR 0032.
- **ODC-style archive** — keep the old processing but hide it from queries. Lossless, but the hiding is
  itself a silent choice; D6's raise asks instead.
- **Provider priority at discovery** (EODAG) — picks a *provider*, not a processing; "the raw data
  matters, not the source".
- **Keep provider item ids as folder names** — the same ESA product from two sources gets two folders
  (MPC's id has no baseline field) and double storage; "collide if same processing" then needs a
  separate check.
- **Tile-first hierarchy** (`{collection}/{tile}/YYYY/…`, Earth Search's S2 COG layout) — S1 has no
  tile, so every collection would need its own partition key.
- **`acquisition_key` derived on read** — one fact, one home, but the parser would have to run on
  nodes and a user-registered collection has none.
- **A row-filter specifier in `properties_filter`** (`{"processing_version": ">=04.00"}`) plus an
  implicit pick — makes one knob do two jobs, and only for one key.
- **`archive=` + `collection=` replacing `catalog_filepath=` on every verb** — cleaner long-term, but
  rewrites four signatures, the notebooks and the tutorial again straight after spec 58.

## 8. Questions at sign-off — ALL RESOLVED (user, 2026-09-30)

D1–D12 were resolved in the 2026-09-29 grilling. **D7 was amended 2026-09-30** at the user's question
("which processing does `download` fetch?"): `download` gained `processing=` (default `"latest"`) and a
post-download ambiguity report, and D6's `"latest"` was redefined so it raises only when it must choose
and cannot. **D10 was rewritten 2026-09-30** at the user's question ("wasn't `.part` already used?"):
it was — `fs.transfer` has been atomic since 2026-07-02, and #74's premise (copied into the first D10)
missed that; the real defect is the in-place stamp after the rename. The user chose stamp-then-publish
with a fixed `.stage` name and no uuid. One point was derived while writing rather than asked:

1. `download(processing=None)` raises instead of fetching every processing a provider serves (D7).
   **Accepted with the sign-off, 2026-09-30.**

## 9. Best-practice alignment / sources

Per-source credit — what each source contributed:

- **USGS, *What is the naming convention for Landsat Collections Level-1 scenes?***
  (usgs.gov/faqs) — the product id `LXSS_LLLL_PPPRRR_YYYYMMDD_yyyymmdd_CC_TX` separates **acquisition
  date** from **processing date, collection number and tier**. The model for D4's split between the
  acquisition key and the processing fields.
- **STAC processing extension** (github.com/stac-extensions/processing) — defines
  `processing:version` ("the version of the primary processing software or processing chain that
  produced the data") and `processing:datetime` ("processing date and time … in UTC"). The names and
  meanings of D8's two new columns and their STAC export.
- **STAC best practices** (radiantearth/stac-spec `best-practices.md`) — item ids "STRONGLY
  RECOMMENDED" unique per collection and usable as file names (D3: the canonical granule name is both
  the folder and the id, scoped by the collection directory); items in **subdirectories** when they
  have sidecar files, and "limit the number of Items in a Catalog or Collection, grouping /
  partitioning as relevant" — D2's date levels, i.e. the crowding fix.
- **stactools-packages/sentinel2 issue #130** (via spec 33's research) — the STAC-ecosystem view that
  an id should identify a location and time, with reprocessed versions expressed separately. D4 keeps
  that identity as `acquisition_key` but, per D1, does **not** make it the folder name.
- **CDSE STAC, live item** (stac.dataspace.copernicus.eu/v1, `sentinel-2-l2a`, fetched 2026-09-29) —
  the id is the ESA product name without `.SAFE`; `datetime` equals the id's sensing time;
  `processing:version` (`"05.13"`) and `processing:datetime` equal the baseline and the discriminator;
  `platform` is lowercase (`sentinel-2c`) and the tile is `grid:code` = `MGRS-…`. Basis for D3's CDSE
  column and for parsing names rather than normalizing properties.
- **MPC `sentinel-2-l2a` collection + the Austria catalog** (planetarycomputer.microsoft.com, fetched
  2026-09-29) — providers list **Esri as processor**, "processed to L2A … using Sen2Cor"; real 2018
  items carry `s2:granule_id` `…_ESRI_20201009…_N02.12` and `s2:product_uri` with the `N0212` field. §1
  fact 3 and D3's MPC column.
- **MPC `sentinel-1-rtc` collection** (fetched 2026-09-29) — Catalyst is the processor; **no
  `processing:*` or version field** is published; `s1:product_timeliness` includes `Reprocessing`. Why
  S1's `processing_version` is null and D6 refuses `"latest"` for it.
- **MPC `hls2-s30`, live item** (fetched 2026-09-29) — id `HLS.S30.T59VNH.2026270T234621.v2.0` carries
  the version; its `datetime` (23:49:35) differs from the id's sensing time (23:46:21). D2's rule that
  the date level comes from the name, and D4's HLS row.
- **CDSE forum, "Sentinel-2 L2A duplicate products (and border artefact)"** (via spec 33's research) —
  near-duplicate products from datastrip splits are produced by design; guidance "use the most
  recent". D7's extension of the dedup to CDSE, and D4's false-positive warning.
- **Element 84 Earth Search README** (github.com/Element84/earth-search) — Landsat ids omit the
  processing date so a USGS reprocess **replaces** the item; the README notes some historical items
  still point at superseded processings (dead links). §7's evidence against replace-on-acquisition.
- **Open Data Cube CLI docs** (opendatacube.readthedocs.io) — `dataset archive` removes datasets from
  active use, `restore` reverses it, `purge` deletes only archived datasets, and `find-duplicates`
  searches for duplicates within a product. The "archive, don't overwrite" alternative in §7, and
  precedent for duplicate detection as a first-class operation.
- **EODAG configuration docs** (eodag.readthedocs.io) — a provider is chosen by **priority** at search
  time (`set_preferred_provider`), with fallback. The provider-priority alternative in §7.
- **PEP 440 / `packaging`** — version specifier syntax and normalization (`05.11` == `5.11`). D6's
  specifier grammar; verified locally with `packaging` 25.0.
- **rslearn tile store** (read-only reference, `rslearn/`) — `tiles/{layer}/{item_name}/{bandset}/`,
  per layer and source item name, no cross-source dedup. Confirms no borrowed layout solves D1.

## 10. Implementation note — build order

1. Collection parsers (`canonical_name`, `acquisition_key`, `acquisition_date`) + their tests.
2. Schema (D8) + `TileCatalog` read raise + `source` union.
3. Source modules: D2/D3 paths, D10 atomic writes.
3a. D6 ordering + specifier selection as one shared function; D7 `download(processing=)` on all four paths + the ambiguity report.
4. D6 check + `processing=` + `properties_filter` on columns + `params_key`.
5. D5 path semantics + preflight.
6. D9 provenance.
7. P2 run-book, then notebooks/docs (D12).

## Amendment A2 (2026-10-13) — refuse a `setup`-written `input.csv` with no `processing` column (#145 X2)

**Status:** SIGNED OFF (user, 2026-10-13), before any code in the X2 PR (#177) was written. Decided by the user on 2026-10-09 (question Q3b of the
refactor audit, #145 Decisions comment) and 2026-10-13 (the three X2 questions: `verify_adapter` stays
out, fail loudly, record it in this spec). Extends D8's "No shim" paragraph from the catalog to
`input.csv`. Corrects Q3b's "old run folders rebuild once", which holds for neither site Q3b named.
Line numbers below are at `main` b839628, before the X2 code.

**Problem.**
- `setup` writes a `processing` column into every `input.csv` row (`_append_input_rows`,
  `workflows/create_datacube.py:213`). A file written before this spec has none.
  `_flatten_identity` covers that case with a shim: `params.setdefault("processing", [""])`
  (`api.py:904-906`). D8 says "no shim" for a pre-59 catalog; `input.csv` got one anyway.
- Dropping the shim alone makes such a run folder re-flatten on **every** `create_training_data` call,
  not once:
  1. `_row_matches_window` skips a column the row lacks (`create_datacube.py:609-611`), so the old rows
     count as present and `setup` never runs.
  2. `build_shortfall_only` rewrites `input.csv` from those rows, still without the column (`:844-846`).
  3. The request's identity always holds `processing` (`api.py:968`). The identity read from the file
     would not, so the stamp never matches.
- `flatten_training_data` also accepts a hand-built `input_csv` with only `id`, `datacube_filepath` and
  an optional label (e.g. `tests/test_training_data_aml.py:465`). It never came from `setup`, so a
  missing `processing` column there does not mean "written before spec 59".

**Decision.**
- **X2.1** `flatten_training_data` raises `PreflightError` when its `input_csv` has at least one of the
  run-parameter columns `setup` writes (`bands`, `mosaic_days`, `startdate`, `enddate`, `collection`,
  `mosaic_scheme`, `properties_filter`) and no `processing` column. The check runs right after the file
  is read, before the stamp check and before any reduce, so nothing is written. The message names the
  file, says `setup` wrote it before spec 59, and gives the two ways out:
  - delete the run folder and run again;
  - or call `create_training_data` with `overwrite=True`. That rebuilds every cube and deletes and
    rewrites `input.csv` from the catalog (`create_datacube.py:985-987`), so it needs a spec-59
    catalog (D8).

  `create_training_data` reaches the check through `flatten_training_data`, which it always calls.
- **X2.2** `_flatten_identity` drops the `setdefault`. `params` then holds exactly the run-parameter
  columns the file has. This is already the rule for the other seven columns. A hand-built file with
  none of them gets no `processing` key. One-time cost: a stamp written from a hand-built file before
  this amendment holds `processing: [""]`, so the next call re-flattens once and rewrites the stamp.
  Later calls match.
- **X2.3** Unchanged: `verify_adapter` adds `processing` to its identity only when it is not `None`
  (`api.py:2284-2286`). That is D6's cube-identity rule ("appended only when not `None`"), not a shim.
  Dropping it would make every existing `processing=None` export folder raise `PreflightError`
  (`api.py:2295`). Q3b named it by mistake.
- **X2.4** Unchanged: `create_training_data`'s top-level short-circuit never reads `input.csv`
  (spec 50 D3). A run folder whose stamp already matches the request is still handed back as current,
  and X2.1 does not fire for it. That includes a pre-59 run folder flattened after spec 59 merged:
  the shim wrote `processing: [""]` into its stamp, and with `processing=None` its cube paths are
  unchanged (D6), so a `processing=None` request matches. A stamp written before spec 59 has no
  `processing` key, never matches, and so reaches X2.1. Making the short-circuit read `input.csv`
  would undo spec 50 D3. (Corrected in review, 2026-10-13: the first text said every older stamp lacks
  the key. The decision is unchanged.)
- **X2.5** In the same PR: delete the shim's comment (`api.py:904-905`); add one sentence on the
  refusal to `flatten_training_data`'s docstring. No existing test pins the shim.
- **Data.** The workspace holds four pre-59 `input.csv` files, all spec-58 scratch runs under
  `tests/outputs/p58_p2/` and `tests/outputs/p58_redownload/` (gitignored). `flatten_training_data`
  on any of them now raises. The two flatten stamps there (`p58_p2/training_s1`, `training_s2`) have
  no `processing` key, so `create_training_data` reaches the refusal too (X2.4).

**Prior art.** In-repo only; the decision relies on no outside fact.
- D8 here: "No shim, no migration tool (spec 58 D12's standing policy): a pre-59 catalog is not read. It
  raises naming the missing columns and 're-download'." X2.1 applies the same rule to `input.csv`.
- #145 Q5 (X4): a hand-built catalog with no `offset`/`nodata` fails loudly rather than defaulting.
- Spec 52's 2026-10-13 amendment (#145 X1): a pre-spec rule is dropped and old content is not migrated.

**How to verify.**
- A test: an `input.csv` with `setup`'s run-parameter columns and no `processing` column makes
  `flatten_training_data` raise `PreflightError`. The message names the file and spec 59, and no array
  or stamp is written. It fails on the old code, which flattens.
- A test: `create_training_data` on a run folder whose cubes all exist and whose `input.csv` lacks the
  column raises the same error. It fails on the old code.
- A test: a hand-built `input.csv` (`id`, `datacube_filepath` only) still flattens; its stamp's `params`
  has no `processing` key; a second call skips. The key assertion fails on the old code.
- `grep -n 'setdefault("processing"' src/fsd/api.py` prints nothing.
- No gate 4: the check and `_flatten_identity` run on the driver only (one caller, `api.py:1116`,
  before any dispatch).

**Risks.**
- A hand-built `input.csv` that copies some of `setup`'s columns but not `processing` now raises. Fix:
  add an empty `processing` column.

**Out of scope.**
- `_row_matches_window` (`create_datacube.py:609-611`) and `_dedupe_on_unit_identity` (`:458`) skip
  any column a row lacks. That tolerance is wider than `processing` (it also covers pre-58 files with
  no `properties_filter`). Recorded on #145 as a finding.
- `create_datacube` on a pre-59 run folder: it never flattens, so X2.1 does not fire there.
- `verify_adapter` (X2.3).

## Amendment A3 (2026-10-14) — a missing or empty `offset` / `nodata` fails loudly (#145 X4, #173)

**Status:** DRAFT. Decided by the user on 2026-10-09 (question Q5 of the refactor audit, #145 Decisions
comment: "Fail loudly") and 2026-10-14 (the four X4 questions: one rule at `append` plus a check in
`read()` and strict readers; `offset`/`nodata` raise, `scale` fills from the declaration; no repair;
home here). Extends D8's "No shim" paragraph from the catalog's columns to its `offset`/`nodata`
values. Line numbers below are at `main` cdcba01, before the X4 code.

**Problem.**
- `TileCatalog.append` fills `offset: 0`, `scale: 1.0`, `nodata: 0` when a column is missing from
  **every** new row (`_APPEND_DEFAULTS`, `catalog/catalog.py:80-88`, applied at `:252-254`). An S2 row
  at baseline ≥ 04.00 needs `offset` −1000 (`sources/_s2_radiometry.py` `offset_for_item`, which raises
  rather than guess). A hand-built row without it gets 0 silently, and the cube's reflectance is off by
  1000 DN.
- A column that **some** rows leave empty (`None`, or a key one row lacks) is not filled: pandas makes it
  float64 with NaN (#173). NaN gets past the readers' `or 0` / `is None` guards (NaN is truthy and not
  `None`):
  - the builder (`datacube/builder.py:236-238` in `flatten_catalog`, then `:392` and `:667`) calls
    `int(NaN)`, which raises `ValueError` on the AML node, far from the append that caused it;
  - the STAC export (`catalog/stac.py:239-243` in `_granule_item`) writes NaN into `raster:bands`. That
    serialises as a bare `NaN` token, which JSON does not allow (RFC 8259 §6).
- `read()` checks the spec-59 columns only (`catalog.py:311`). Its docstring says a hand-built file
  without `offset`/`nodata` passes and the builder then reads 0 (`:306-308`).
- Measured on 2026-10-14: every catalog file under `tests/` (the tutorial fixture and five under
  `tests/outputs/p59_p2/`) stores `offset`/`nodata` as int64 and `scale` as float64, with no NaN. Both
  sources always write all three columns (`sources/mpc.py:180-182`, `sources/cdse.py:259-261`). So only
  hand-built rows and files are affected: removing the two defaults fails 13 existing tests in 5 files,
  all with hand-built rows.

**Decision.**
- **X4.1** `append` raises `ValueError` before it reads or writes any file when a new row has no
  `offset` or no `nodata`: the column is missing from every row, or the row's cell is empty (`None`,
  NaN, or the key is absent while other rows have it). The message names the column, the ids of the
  rows that lack it (at most the first five), and says to pass the value explicitly: for S2,
  `offset_for_item`'s value; `0` for a collection with no offset. `offset` and `nodata` leave
  `_APPEND_DEFAULTS`.
- **X4.2** `append` fills `scale` from the declaration it stamps (`effective.scale`, where `effective`
  is the `declaration=` argument, else the constructor's, else the existing file's stamp), when the
  column is missing from every row or a row's cell is empty. With no declaration at all it fills `1.0`,
  `CollectionDeclaration`'s default and today's value. This replaces today's `scale: 1.0` default,
  which is wrong for S2 (1/10000). Sources write `declaration.scale` already, so their rows do not
  change. `scale` is filled, not refused, because the declaration states it exactly; `nodata` is not
  filled, because the sources write `config.NODATA` rather than the declaration's `nodata` (#181).
- **X4.3** `read()` raises `ValueError` when the file lacks an `offset`, `scale` or `nodata` column, or
  any of them holds NaN. The check runs after the spec-59 column check, in the same style: the message
  names the file and the column and says "re-download" (or re-append the rows). No repair: D8's "No
  shim" applies. `TileCatalog.filter` and `to_stac` read through `read()`, so they get the check.
- **X4.4** The two readers stop defaulting: `flatten_catalog` reads `row["offset"]` and
  `row["nodata"]`; `_granule_item` reads `row["offset"]`, `row["scale"]` and `row["nodata"]`, each
  verbatim. A hand-built frame that lacks one raises `KeyError` naming it. One visible effect: a
  hand-built `scale` of 0 now reaches `raster:bands` as 0 instead of being replaced by the
  declaration's scale.
- **X4.5** In the same PR, make these docstrings and comments true: the `_APPEND_DEFAULTS` comment
  (`catalog.py:69-79`), `read()` (`:301-309`), `flatten_catalog` (`builder.py:213-229`: "missing
  `offset`/`nodata` columns default every row to 0"), and `CollectionDeclaration`'s `nodata`
  (`catalog/declaration.py:119-120`): after X4 the fallback serves only a hand-built band-flattened
  frame passed straight to `build_datacube`.

**Prior art.** In-repo, plus one outside fact.
- D8 here ("No shim, no migration tool"); Amendment A2 applied the same rule to `input.csv` (#145 X2).
- `offset_for_item` (`sources/_s2_radiometry.py`) already refuses to guess an offset for S2; X4.1 moves
  the same refusal to the point where any row is written.
- #145 X3 (spec 58/59 shard rows, 2026-10-14): `download_shard` reads row keys strictly instead of
  defaulting; X4.4 does the same in the readers.
- RFC 8259 §6 (https://www.rfc-editor.org/rfc/rfc8259.html, checked 2026-10-14): "Numeric values that
  cannot be represented in the grammar below (such as Infinity and NaN) are not permitted." This is why
  a NaN in `raster:bands` makes the item invalid JSON (Problem). The STAC raster extension's rules for a
  NaN `nodata` are not needed: after X4, NaN cannot reach the export.

**How to verify.** Tests go in a new file. Each names the mistake it catches that no existing test
does.
- **T1** `append`, parametrised over `offset` and `nodata` × {the column missing from every row; `None`
  in one of two rows}, to a catalog path that does not exist yet: raises `ValueError` whose message
  names the column and the id of the row that lacks it; the file still does not exist. Catches: the
  silent column default (old code writes 0) and the per-row NaN (old code writes a float column with
  NaN). `test_append_rows_without_offset_or_nodata_default_to_zero` pins the opposite and is deleted.
- **T2** `append` with `declaration=S2_L2A_DECLARATION`, parametrised over {the `scale` column missing
  from every row; `None` in one of two rows, the other row giving 1/10000}: every stored `scale` equals
  1/10000. Catches: filling `1.0` (old code, column missing) or leaving NaN (old code, empty cell).
- **T3** `read()` on a file written with `fs.write_parquet` from a frame that otherwise passes `read()`,
  parametrised over {no `offset` column; NaN `offset`; NaN `scale`; NaN `nodata`}: raises `ValueError`
  whose message names the file path and the column. Catches: a check that skips one of the three
  columns, or that tests presence but not NaN. The second half of
  `test_read_does_not_backfill_a_legacy_catalog_missing_offset_nodata` pins the opposite (a file with
  none of the three passes) and changes to expect this error.
- **T4** `flatten_catalog` on a hand-built frame without `offset`, and `tile_catalog_to_items` on one
  without `nodata`: each raises `KeyError`. Catches: a reader that keeps a default.
  `test_flatten_catalog_offset_and_nodata_default_to_zero_when_column_missing` pins the opposite and is
  deleted.
- `grep -n -E 'get\("(offset|nodata|scale)"' src/fsd/datacube/builder.py src/fsd/catalog/stac.py` prints
  nothing.
- Existing tests whose hand-built rows or frames lack the columns gain `offset=0, nodata=0` (the 13 the
  measurement found, plus the `flatten_catalog` frames in `test_datacube_builder.py`). In
  `tests/test_stac_split_pins.py` (#170), the case with no `offset`/`scale`/`nodata` columns changes to
  expect `KeyError`, and the case with `scale` 0 changes to expect 0 in `raster:bands` (X4.4). No other
  expected value changes.
- **Gate 4** (the builder change runs on AML nodes): `runbooks/145-x4-cube-equality.ipynb`, a copy of
  `145-p12-cube-equality.ipynb` with `BASE = "cdcba01"`, builds the cube from the tracked tutorial
  fixture with `BASE` and with this branch. PASS: the cubes, metadata and sidecars are equal as in P12.
  A local run counts: real catalogs always carry the columns, and nothing here depends on Azure.

**Risks.**
- A hand-built row or file without `offset`/`nodata` (a notebook, an fsd-agnostic top-up) now raises.
  Fix: add the values. This is the point of Q5.

**Out of scope.**
- `build_datacube`'s own fallbacks on a band-flattened frame (`builder.py:391-394` → the declaration's
  `nodata`; `_apply_offsets` `:658-667`, a missing `offset` column is a no-op). After X4,
  `flatten_catalog` always fills both columns, so only a hand-built band-flattened frame reaches them.
- #181: the sources write `config.NODATA` (0) as every row's `nodata`, including S1 RTC, whose
  declaration and rasters say −32768.
- Casting `offset`/`nodata` to int64 on append. The column dtype follows the input: once empty cells
  are refused, integer input stays int64 (#173 asked for this; it now holds without a cast).
- `properties` keeps its `"{}"` default.
