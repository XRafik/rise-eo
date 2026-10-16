"""Prove a diff is comments-only: compare docstring-stripped ASTs vs a git ref.

Usage: .venv/bin/python scripts/comment_astcheck.py <git-ref> [root]
Exits 1 and prints every file whose executable AST changed.
"""
import ast
import subprocess
import sys
from pathlib import Path


def strip(tree):
    for n in ast.walk(tree):
        body = getattr(n, "body", None)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                n.body = body[1:] or [ast.Pass()]
    return tree


def sig(src):
    return ast.dump(strip(ast.parse(src)), include_attributes=False)


ref = sys.argv[1]
root = Path(sys.argv[2] if len(sys.argv) > 2 else "src/rise_eo")
bad = []
checked = 0
for p in sorted(root.rglob("*.py")):
    rel = p.as_posix()
    old = subprocess.run(
        ["git", "show", f"{ref}:{rel}"], capture_output=True, text=True
    )
    if old.returncode != 0:
        print(f"NEW  {rel} (not in {ref})")
        continue
    checked += 1
    try:
        if sig(old.stdout) != sig(p.read_text()):
            bad.append(rel)
    except SyntaxError as e:
        bad.append(f"{rel}: {e}")

if bad:
    print("AST CHANGED:")
    for b in bad:
        print("  " + b)
    sys.exit(1)
print(f"AST identical across {checked} files vs {ref} — comments-only.")
