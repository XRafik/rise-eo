"""Table test for `scripts/ci_changed_paths.py` (spec 102 A2.5). Synthetic, no network."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "ci_changed_paths.py"
_spec = importlib.util.spec_from_file_location("ci_changed_paths", SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


@pytest.mark.parametrize("paths, full", [
    (["docs/x.md"], False),
    (["README.md"], False),
    (["tests/manual/x.md"], False),
    (["LICENSE"], False),
    (["notebooks/shapefiles/NOTICE"], False),
    (["README.md", "docs/a/b.md", "LICENSE"], False),
    (["src/rise_eo/api.py"], True),
    ([".github/workflows/ci.yml"], True),
    (["pyproject.toml"], True),
    (["runbooks/x.ipynb"], True),
    (["README.md", "src/rise_eo/api.py"], True),
    (["x.md.py"], True),
    ([], True),
    ([""], True),
])
def test_needs_full_suite(paths, full):
    assert mod.needs_full_suite(paths) is full


def test_cli_reads_stdin():
    def run(text):
        return subprocess.run([sys.executable, str(SCRIPT)], input=text, capture_output=True,
                              text=True, check=True).stdout.strip()
    assert run("README.md\nLICENSE\n") == "false"
    assert run("README.md\nsrc/rise_eo/api.py\n") == "true"
    assert run("") == "true"
