# Manual test guides

Step-by-step runbooks you read and execute by hand to verify a module against its
spec. One file per module; they grow over time — ask to **augment** a file when you
want more checks.

| Guide | Module | Spec |
|-------|--------|------|
| `storage.md` | `rise_eo.storage.fs` | `specs/10-storage-and-scale.md` |
| `realdata.md` | `rise_eo.raster.images` + `rise_eo.bands.modify` (real tile, QGIS) | `specs/07-raster.md`, `06-bands.md`, `09-notebooks.md` |
| `datacube.md` | `rise_eo.datacube.builder` (real **multi-CRS** tiles, QGIS) | `specs/03-datacube.md`, `04-datacube-ops.md` |
| `debug_s3paths.md` | _superseded_ — BUG-001 S3 listing debug (the STAC pivot removed S3 listing) | `BUGS.md` at tag `docs-archive-2026` |

These complement (don't replace) automated `pytest` in `tests/`. Use a manual guide
for credentialed / remote / exploratory checks; deterministic local checks are also
good candidates for automation when you want them hands-free.
