"""Print a catalog Parquet file's stamped `CollectionDeclaration`, footer-only --
no row group is read (`python -m rise_eo.catalog.inspect_cli`).

Recovers the sidecar-JSON option's one real advantage (human legibility)
without its separation risk: the
declaration always lives in the catalog file itself, this just prints it.

Run as:  python -m rise_eo.catalog.inspect_cli <catalog.parquet>
"""

from __future__ import annotations

import argparse
import json

from rise_eo.catalog import declaration as declaration_module
from rise_eo.storage import fs


def inspect_catalog(path: str) -> dict | None:
    """The raw stamped declaration JSON at `path`, or `None` if unstamped."""
    return fs.peek_parquet_attrs(path).get(declaration_module.ATTRS_KEY)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m rise_eo.catalog.inspect_cli",
        description="Print a catalog's stamped declaration (footer-only read, spec 35 §6).",
    )
    p.add_argument("catalog", help="catalog.parquet path (any rise_eo.storage URL)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    raw = inspect_catalog(args.catalog)
    if raw is None:
        print(f"{args.catalog}: no fsd:declaration stamp.")
        return
    print(json.dumps(raw, indent=2))


if __name__ == "__main__":
    main()
