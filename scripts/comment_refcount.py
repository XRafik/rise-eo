"""Count prose lines and backward references per file under src/rise_eo.

Usage: .venv/bin/python scripts/comment_refcount.py src/rise_eo   (or any subpackage)
Prints refs / prose lines / code lines / prose-per-code per file, densest first, then a TOTAL.
Needs Python >= 3.8 (`ast.Constant`); an older interpreter silently reports zero docstrings.
The convention it audits: docs/reference/code-comments.md.
"""
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

REF = re.compile(r"spec\s+\d+|\bAC\d+\b|\bD\d+\b|#\d+|20\d\d-\d\d-\d\d", re.I)


def stats(p: Path):
    src = p.read_text()
    lines = src.splitlines()
    total = len(lines)
    tree = ast.parse(src)
    doc_lines = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = n.body
            if (
                b
                and isinstance(b[0], ast.Expr)
                and isinstance(b[0].value, ast.Constant)
                and isinstance(b[0].value.value, str)
            ):
                doc_lines.update(range(b[0].lineno, b[0].end_lineno + 1))
    comment_lines = set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            comment_lines.add(tok.start[0])
    prose = doc_lines | comment_lines
    blank = {i + 1 for i, line in enumerate(lines) if not line.strip()}
    code = total - len(prose) - len(blank - prose)
    refs = sum(len(REF.findall(lines[i - 1])) for i in sorted(prose))
    return total, code, len(prose), refs


root = Path(sys.argv[1] if len(sys.argv) > 1 else "src/rise_eo")
rows = []
for p in sorted(root.rglob("*.py")):
    try:
        rows.append((p, *stats(p)))
    except Exception as e:
        print(f"ERR {p}: {e}", file=sys.stderr)
rows.sort(key=lambda r: -r[4])
T = C = P = R = 0
for p, t, c, pr, r in rows:
    T, C, P, R = T + t, C + c, P + pr, R + r
    if r or pr:
        print(f"{r:5d} refs {pr:5d} prose {c:5d} code {pr / max(c, 1):5.2f}  {p}")
print(f"\nTOTAL {T} lines / {C} code / {P} prose ({P / max(C, 1):.2f} per code line) / {R} refs")
