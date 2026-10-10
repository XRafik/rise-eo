"""Infer-only unit-of-work: infer one or more PRE-BUILT datacubes -> COG(s).

Spec: specs/22-unify-inference-runner.md

No build step (unlike `infer_task`). Replaces `engine.run_local`'s retired `mp.Pool`: the
`cores>1` pre-built-cubes inference fans out through this task via the Snakemake infer-only
runner. **No `mp.Pool` here** — parallelism across tasks is Snakemake's; a task loops
**sequentially** over its row group (`--rows i:j`), loading the bundle **once** (the
`cubes_per_task` amortiser). Existing outputs are skipped unless `--overwrite` (idempotency).

Run as:  python -m fsd.workflows.infer_only_task --input-csv <csv> --rows i:j --bundle <path>
             [--predict-batch-size N] [--no-skip-nan] [--overwrite]

The CSV has columns `datacube_filepath`, `output_filepath`.
"""

from __future__ import annotations

import argparse

import pandas as pd

from fsd.model import bundle as _bundle
from fsd.model import engine
from fsd.storage import fs


def run_infer_only(
    input_csv: str,
    rows: tuple[int, int],
    bundle_path: str,
    *,
    predict_batch_size: int | None = None,
    skip_nan: bool = True,
    overwrite: bool = False,
) -> list[str]:
    """Infer rows `[lo, hi)` of `input_csv` to their `output_filepath`s. One bundle load, then a
    plain sequential loop (no pool). Skips a row whose output already exists unless `overwrite`."""
    with fs.open(input_csv, "r") as f:
        df = pd.read_csv(f)
    lo, hi = rows
    adapter = _bundle.load(bundle_path)                 # ONE load for the whole group
    written = []
    for _, r in df.iloc[lo:hi].iterrows():
        dc_fp, out_fp = str(r["datacube_filepath"]), str(r["output_filepath"])
        if not overwrite and fs.exists(out_fp):
            continue
        engine.infer_datacube_to_cog(
            adapter, dc_fp, out_fp,
            predict_batch_size=predict_batch_size, skip_nan=skip_nan,
        )
        written.append(out_fp)
    return written


def _parse_rows(s: str) -> tuple[int, int]:
    lo, hi = s.split(":")
    return int(lo), int(hi)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m fsd.workflows.infer_only_task",
        description="Infer one or more pre-built datacubes -> COG(s) (one infer-only work-unit).",
    )
    p.add_argument("--input-csv", required=True, help="CSV with datacube_filepath, output_filepath")
    p.add_argument("--rows", required=True, help="row slice 'i:j' of the CSV to process")
    p.add_argument("--bundle", required=True, help="model bundle path")
    p.add_argument("--predict-batch-size", type=int, default=None)
    p.add_argument("--no-skip-nan", action="store_true",
                   help="predict on every pixel (default skips all-NaN pixels)")
    p.add_argument("--overwrite", action="store_true", help="re-infer even if the output exists")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    run_infer_only(
        args.input_csv, _parse_rows(args.rows), args.bundle,
        predict_batch_size=args.predict_batch_size, skip_nan=not args.no_skip_nan,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
