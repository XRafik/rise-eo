"""Check every kwarg passed to a rise-eo verb in the notebooks, docs and demos against the live signature.

Catches call sites left behind when a verb parameter is removed or renamed -- the failure
pytest cannot see, because prose and notebook cells are never executed. Found four dead
`scl_mask_classes=` call sites after spec 58 P1 removed it (3 notebook cells + docs/tutorial.md).

Run from the fsd repo root: .venv/bin/python scripts/docs_kwarg_sweep.py
Exits 1 if any call site is stale.
"""
from __future__ import annotations

import ast
import inspect
import json
import pathlib
import sys

sys.path.insert(0, "src")

import rise_eo  # noqa: E402
from rise_eo.workflows import create_datacube  # noqa: E402

TARGETS: dict[str, inspect.Signature] = {}
for _name in ("download", "create_training_data", "run_inference", "verify_adapter",
              "build_datacube", "compute_n_timestamps", "deploy"):
    _f = getattr(rise_eo, _name, None)
    if _f is not None:
        TARGETS[_name] = inspect.signature(_f)
for _name in ("run_create_datacube", "setup", "params_key", "build_shortfall_only"):
    _f = getattr(create_datacube, _name, None)
    if _f is not None:
        TARGETS[_name] = inspect.signature(_f)

bad = 0


def check(source: str, origin: str) -> None:
    """Report every kwarg in `source` that the called fsd verb does not accept."""
    global bad
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return  # an illustrative fragment, not runnable code
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        sig = TARGETS.get(name)
        if sig is None:
            continue
        params = sig.parameters
        if any(p.kind == p.VAR_KEYWORD for p in params.values()):
            continue
        for kw in node.keywords:
            if kw.arg is not None and kw.arg not in params:
                print(f"STALE  {origin}: {name}(... {kw.arg}=...) is not in the signature")
                bad += 1


def md_python_blocks(path: pathlib.Path):
    """Yield (code, first_line) for each python fenced block, pairing fences by state."""
    inside, info, buf, start = False, "", [], 0
    for n, line in enumerate(path.read_text().split("\n"), 1):
        if line.startswith("```"):
            if not inside:
                inside, info, buf, start = True, line[3:].strip().lower(), [], n
            else:
                if info in ("python", "py", ""):
                    yield "\n".join(buf), start
                inside = False
        elif inside:
            buf.append(line)


for nb_path in pathlib.Path("notebooks").glob("*.ipynb"):
    nb = json.loads(nb_path.read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            check("".join(cell["source"]), f"{nb_path} cell {i}")

md_files = [*pathlib.Path("docs").rglob("*.md"), *pathlib.Path("demos").glob("*.md"),
            pathlib.Path("README.md")]
for md in md_files:
    for code, line in md_python_blocks(md):
        check(code, f"{md} ~line {line}")

for py in [*pathlib.Path("demos").glob("*.py"), *pathlib.Path("examples").rglob("*.py")]:
    check(py.read_text(), str(py))

print(f"sweep done: {bad} stale call site(s)")
sys.exit(1 if bad else 0)
