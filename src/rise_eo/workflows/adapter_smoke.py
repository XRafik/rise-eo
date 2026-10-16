"""Node-side adapter-import smoke.

The ONE check the driver cannot do alone: whether the adapter (module:attr) actually
imports and its artifact actually loads **inside the real inference Environment** -- the
driver's own venv is not guaranteed to mirror the node's image (ADR 0002). Runs as a
single one-node job, once, before the N-node fan-out (`run_aml_inference`); a missing
`sklearn` or an un-importable adapter fails here in ~40-380s (one node's cold start)
instead of on every fan-out node.

No pipeline logic: fetch the staged bundle to scratch (reuses
`infer_shard.fetch_bundle_to_scratch`), check the bundle's declared requirements against this
Environment, `bundle.load` it (resolving the import, usually from the bundle's own `code/`,
and reading the artifact), and confirm `predict` is callable. Writes a `_status/*.json`.

Run as: python -m rise_eo.workflows.adapter_smoke <bundle_url> --status-url <url>
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile

from rise_eo.model import bundle as _bundle
from rise_eo.storage import fs
from rise_eo.workflows.infer_shard import fetch_bundle_to_scratch


def run_smoke(bundle_url: str, status_url: str) -> dict:
    scratch_dir = tempfile.mkdtemp(prefix="fsd-smoke-bundle-")
    error = None
    try:
        local_bundle = fetch_bundle_to_scratch(bundle_url, scratch_dir)
        # Check DECLARED dependencies before importing, so a missing `sklearn` is
        # reported as a named dependency rather than as an ImportError traceback from deep
        # inside the adapter. fsd never installs them -- they belong in the Environment image
        # Dependency installation stays front-loaded to image build time.
        manifest = _bundle.read_spec(local_bundle)
        missing = _bundle.check_requirements(manifest.get("requirements"))
        if missing:
            raise RuntimeError(
                "the inference Environment does not satisfy the bundle's declared requirements: "
                + "; ".join(missing)
                + ". Install them in the Environment image and rebuild it."
            )
        adapter = _bundle.load(local_bundle)
        if not callable(getattr(adapter, "predict", None)):
            raise TypeError(f"{type(adapter).__name__}.predict is not callable")
    except Exception as exc:  # noqa: BLE001 - always report, never crash the job silently
        error = str(exc)
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    status = {"status": "ok" if error is None else "failed", "error": error}
    with fs.open(status_url, "w") as f:
        json.dump(status, f, indent=2)
    return status


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m rise_eo.workflows.adapter_smoke",
        description="One-node adapter-import smoke, run once before the AML inference "
                     "fan-out (spec 38 D11).",
    )
    p.add_argument("bundle_url", help="the staged bundle folder (any rise_eo.storage URL)")
    p.add_argument("--status-url", required=True)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    status = run_smoke(args.bundle_url, args.status_url)
    if status["status"] != "ok":
        raise SystemExit(f"adapter-import smoke failed: {status}")


if __name__ == "__main__":
    main()
