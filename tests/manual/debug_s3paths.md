# Debug runbook — fetching a tile's S3 paths (legacy boto3 vs rise-eo s3fs)

> **SUPERSEDED (2026-07-01).** The premise below (an s3fs-vs-boto3 *signing*
> difference on recursive listing) was **disproven**: the user's multi-run debug
> showed the **legacy boto3** path *also* fails intermittently with the same errors,
> and sometimes fully succeeds — i.e. the cause is **CDSE server-side** (node
> credential-replication inconsistency), not rise-eo. See
> `debug-attempts/s3_paths_fetch/cdse_s3_intermittent_auth_report.md` and BUG-001 in
> `BUGS.md` at tag `docs-archive-2026`. Kept for history; steps 1–4 are still a fine way to *observe* the
> intermittency, but the fix is client resilience, not a listing-method change.

Goal (original): isolate **BUG-001** (`BUGS.md` at tag `docs-archive-2026`). Shallow listing of a `.SAFE`
works in rise-eo, but the **recursive** listing my file-selection uses
(`fs.glob(".../**/*.jp2")`) fails with `SignatureDoesNotMatch`, while the legacy
**boto3** `objects.filter(Prefix=...)` lists recursively fine. This runbook builds
the legacy known-good reference, then diffs the rise-eo path against it so we can see
exactly what to change.

Run top-to-bottom in one Python session (dev env active). Tick a box per step.
Never print secret values.

---

## 0. Setup

```python
from rise_eo.sources.cdse import CdseCredentials
from rise_eo.storage import fs
import os

ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))  # workspace root
creds = CdseCredentials.from_json(f"{ROOT}/secrets/cdse_credentials.json")
opts = creds.s3_storage_options()

BUCKET = "eodata"
SAFE_PREFIX = ("Sentinel-2/MSI/L2A_N0500/2018/01/30/"
               "S2A_MSIL2A_20180130T080151_N0500_R035_T36PZT_20230915T000622.SAFE")
SAFE_URL = f"s3://{BUCKET}/{SAFE_PREFIX}"
ENDPOINT = opts["client_kwargs"]["endpoint_url"]
print("endpoint:", ENDPOINT, "| creds:", creds)  # creds repr is masked
```

- [ ] Prints the endpoint and a **masked** creds repr (`sh_client_id=set, ...`).

---

## 1. Sanity — shallow listing works (creds OK)

```python
print(fs.ls(SAFE_URL, **opts))
```

- [ ] Lists ~9 top-level entries (`DATASTRIP`, `GRANULE`, `HTML`, ...). If this
      fails with `InvalidAccessKeyId`, stop — the S3 keys are the problem, not this
      bug (see BUG-001 problem #1).

---

## 2. Legacy known-good — boto3 recursive `filter(Prefix=...)`

This mirrors `cdseutils.sentinel2.get_s3paths_single_url` (the path that reliably
lists a whole `.SAFE`). If this works, recursive listing IS possible with these
keys/endpoint — proving the rise-eo failure is an s3fs-signing difference.

```python
import boto3

s3 = boto3.resource(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=creds.s3_access_key,
    aws_secret_access_key=creds.s3_secret_key,
    region_name="default",
)
keys = [obj.key for obj in s3.Bucket(BUCKET).objects.filter(Prefix=SAFE_PREFIX)]
jp2s = [k for k in keys if k.endswith(".jp2") and "IMG_DATA" in k]
print("total objects:", len(keys), "| IMG_DATA jp2s:", len(jp2s))
print("sample jp2:", jp2s[0].split("/")[-1] if jp2s else None)
```

- [ ] Prints a non-trivial object count and IMG_DATA `.jp2` files (e.g.
      `T36PZT_..._B02_10m.jp2`). **This is the reference behavior rise-eo must match.**

---

## 3. rise-eo current — recursive s3fs glob (the failing call)

```python
try:
    print(fs.glob(f"{SAFE_URL}/**/*.jp2", **opts))
except Exception as e:
    print("FAIL:", type(e).__name__, "-", str(e)[:100])
```

- [ ] Fails with `SignatureDoesNotMatch` (the bug). If it *succeeds*, note the
      `s3fs` / `aiobotocore` versions — the bug may already be fixed by an upgrade.

---

## 4. Isolate — which s3fs listing modes fail?

```python
import s3fs
s3fs_fs = s3fs.S3FileSystem(key=creds.s3_access_key, secret=creds.s3_secret_key,
                            client_kwargs={"endpoint_url": ENDPOINT})

def probe(label, fn):
    try:
        out = fn(); print(f"[{label}] OK ({len(out)})")
    except Exception as e:
        print(f"[{label}] {type(e).__name__}: {str(e)[:70]}")

probe("shallow ls (delimiter=/)", lambda: s3fs_fs.ls(f"{BUCKET}/{SAFE_PREFIX}"))
probe("ls GRANULE",               lambda: s3fs_fs.ls(f"{BUCKET}/{SAFE_PREFIX}/GRANULE"))
probe("find (recursive)",         lambda: s3fs_fs.find(f"{BUCKET}/{SAFE_PREFIX}"))
probe("glob **/*.jp2",            lambda: s3fs_fs.glob(f"{BUCKET}/{SAFE_PREFIX}/**/*.jp2"))
```

- [ ] Record which succeed. Expectation: shallow `ls` OK; recursive `find`/`glob`
      fail. This pins the bug to **recursive (`delimiter=""`) requests** in s3fs.

---

## 5. Candidate fixes — try until rise-eo matches the legacy reference

Try each; note which makes step 3 (or an equivalent recursive listing) succeed.

**(a) Signature / region tweaks on s3fs**
```python
s3fs_fs = s3fs.S3FileSystem(
    key=creds.s3_access_key, secret=creds.s3_secret_key,
    client_kwargs={"endpoint_url": ENDPOINT, "region_name": "default"},
    config_kwargs={"signature_version": "s3v4"},
)
probe("v4+region find", lambda: s3fs_fs.find(f"{BUCKET}/{SAFE_PREFIX}"))
```

**(b) Shallow-`ls` recursive walk (avoids delimiter="" entirely)**
```python
def walk_jp2(base):
    out = []
    for p in fs.ls(base, **opts):
        if p.rstrip("/").endswith(SAFE_PREFIX.rstrip("/")):
            continue
        leaf = p.split("/")[-1]
        if leaf.endswith(".jp2"):
            out.append(p)
        elif "." not in leaf:            # a "folder"
            out += walk_jp2(f"s3://{p}" if not p.startswith("s3://") else p)
    return out
print("walk found jp2s:", len(walk_jp2(SAFE_URL)))
```

**(c) boto3-backed listing inside the CDSE source** — keep `storage.transfer` for the
byte copy, but list via boto3 `filter(Prefix=...)` (step 2). Only if s3fs recursive
listing can't be made to sign correctly.

- [ ] Record which candidate works. That decides how `_select_tile_files` is
      reworked (update BUG-001 + spec 01, then re-run `tests/test_cdse.py`).

---

## What to report back
- Step 2 (legacy boto3) result — does recursive listing work? (expected: yes)
- Step 4 — exact set of s3fs modes that fail.
- Step 5 — which candidate fix makes rise-eo match the legacy reference.
- `s3fs` / `aiobotocore` / `botocore` versions (`pip show s3fs aiobotocore`).
