"""The self-describing model **bundle** — see also `specs/18-model-bundle-explainer.md`.

Spec: specs/18-model-adapter.md

A bundle is a folder carrying the things fsd needs to run a model anywhere: the **code** (an
adapter class, referenced by a `module:attribute` import string, and usually *carried inside
the bundle*), the **artifact(s)** (weights, referenced by paths relative to the bundle), and
the **spec** (required bands, T, output dtype/nodata/names) — mirrored as plain text so fsd
can validate a run *without* importing the code or loading the model (model-free preflight).

    bundle/
      bundle.json     # manifest (below)
      rf.joblib       # artifact(s)
      code/           # the adapter's source, layout preserved
        my_adapter.py

`save` derives the `module:attr` string from the adapter object automatically, and
finds and embeds the adapter's source; `load` puts `code/` on `sys.path`, resolves the ref back to
a class, instantiates it, injects absolute artifact paths, and calls `.load()`.

**Code moves into the bundle; dependencies stay in the image.** The inference image differs
only by *dependency family* (sklearn vs torch), never by model, so an adapter never needs its
own Docker image. Dependencies are **declared** (`requirements`) and checked by the one-node
smoke job — fsd never installs anything at run time.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import site
import sys
import sysconfig

from fsd.storage import fs

__all__ = [
    "adapter_code_files",
    "adapter_ref",
    "check_requirements",
    "classify_adapter_source",
    "load",
    "manifest_code_files",
    "read_spec",
    "resolve_ref",
    "save",
]

BUNDLE_MANIFEST = "bundle.json"
BUNDLE_VERSION = 2
SUPPORTED_BUNDLE_VERSIONS = (1, 2)

#: Where embedded adapter source lives inside the bundle.
CODE_DIR = "code"

#: Guardrails on auto-detection. A package root is walked automatically, which is convenient
#: right up until it sweeps a `data/` folder into every run's upload -- so refuse loudly and early.
MAX_CODE_FILES = 64
MAX_CODE_BYTES = 5 * 1024 * 1024

_CODE_EXCLUDE_DIRS = frozenset({
    "__pycache__", ".git", ".venv", "venv", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache",
})
_CODE_EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".pyd", ".so")

_SPEC_FIELDS = (
    "required_bands", "n_timestamps", "output_dtype", "output_nodata", "output_band_names",
)


def resolve_ref(ref: str):
    """`'crop_mapper.adapters:CropRF'` -> the `CropRF` class object (not an instance).

    An import path in `module:attribute` form (the setuptools entry-point / gunicorn
    convention). `module` must be importable (i.e. on `sys.path`), which is usually satisfied
    by the bundle's own `code/` directory rather than by a pip install.
    """
    module_path, sep, attr = ref.partition(":")
    if not sep or not attr:
        raise ValueError(f"adapter ref must be 'module:attribute', got {ref!r}")
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def adapter_ref(adapter) -> str:
    """Derive the `module:attribute` string for an adapter instance/class."""
    cls = adapter if isinstance(adapter, type) else type(adapter)
    return f"{cls.__module__}:{cls.__qualname__}"


# --- where does the adapter's code come from? --------------------------------


def _installed_roots() -> tuple[str, ...]:
    """Directories that mean "this module shipped with the environment, not with the user"."""
    roots = []
    for getter in (site.getsitepackages, lambda: [site.getusersitepackages()]):
        # `site` is not guaranteed in every embedding (e.g. python -S); a missing root just
        # means one fewer place we recognise as "installed", which is safe.
        with contextlib.suppress(Exception):
            roots += list(getter() or [])
    paths = sysconfig.get_paths()
    roots += [paths[k] for k in ("purelib", "platlib", "stdlib", "platstdlib") if k in paths]
    return tuple(os.path.realpath(r) for r in roots if r)


def _is_installed(filepath: str) -> bool:
    real = os.path.realpath(filepath)
    return any(real.startswith(root + os.sep) for root in _installed_roots())


def classify_adapter_source(adapter) -> tuple[str, str | None]:
    """`("bundled", <abs source file>)`, `("installed", None)`, or `("unresolvable", <why>)`.

    Three origins, three behaviors:

    * **local source** -- a real file that is not under a site-packages/stdlib root. **Embed
      it**; this is the case code-in-the-bundle exists for.
    * **installed package** -- the adapter really is a pip dependency of the image, which is
      still valid. **Skip**; the manifest records no `code` block.
    * **unresolvable** -- `__main__` (a script or, in practice, a notebook cell, where
      `getsourcefile` yields a `/tmp/ipykernel_*/1234.py` that does not exist), a C extension, or
      anything else with no readable source. **Refuse** at `save` -- otherwise the run fails much
      later with a `ModuleNotFoundError` on a cluster node, after a cold start.

    Note an intentional edge: an **editable install** (`pip install -e .`) resolves to its source
    tree, not site-packages, so it classifies as local and gets embedded. That is the safe answer
    -- an editable install is by definition not what shipped in the image -- but it is surprising,
    which is why `MAX_CODE_*` exists and `code=False` is the escape hatch.
    """
    cls = adapter if isinstance(adapter, type) else type(adapter)
    module_name = cls.__module__

    if module_name in ("__main__", "__mp_main__", None):
        return "unresolvable", (
            f"{cls.__qualname__} is defined in {module_name!r} (a script or notebook cell), so it "
            "has no importable module name. Move the class into a .py file next to your notebook "
            "and import it (e.g. `sys.path.insert(0, './my_model'); from my_adapter import "
            f"{cls.__qualname__}`), then bundle it."
        )
    if module_name.split(".")[0] == "fsd":
        # fsd itself is always in the image; never drag src/fsd/ into a bundle.
        return "installed", None

    try:
        src = inspect.getsourcefile(cls)
    except (TypeError, OSError) as exc:
        return "unresolvable", f"cannot locate source for {module_name}:{cls.__qualname__} ({exc})"
    if not src:
        return "unresolvable", (
            f"{module_name}:{cls.__qualname__} has no Python source file (a C extension?)"
        )
    src = os.path.abspath(src)
    if not os.path.exists(src):
        return "unresolvable", (
            f"{module_name}:{cls.__qualname__} reports source at {src!r}, which does not exist. "
            "Classes defined in a notebook cell do this. Move the class into a .py file and "
            "import it, then bundle it."
        )
    if _is_installed(src):
        return "installed", None
    return "bundled", src


def _module_root(module_name: str, source_file: str) -> str:
    """The directory that must be on `sys.path` for `module_name` to import from `source_file`.

    Walk up one level per dot, plus one for the file itself: `my_adapter` -> the containing
    directory; `my_pkg.adapters` -> the parent of `my_pkg/`.
    """
    root = source_file
    for _ in range(module_name.count(".") + 1):
        root = os.path.dirname(root)
    return root


def _walk_code_files(root: str, start: str) -> list[str]:
    """Every embeddable file under `start`, as paths relative to `root` (layout preserved)."""
    rels: list[str] = []
    if os.path.isfile(start):
        return [os.path.relpath(start, root)]
    for dirpath, dirnames, filenames in os.walk(start):
        dirnames[:] = sorted(
            d for d in dirnames if d not in _CODE_EXCLUDE_DIRS and not d.startswith(".")
        )
        for name in sorted(filenames):
            if name.startswith(".") or name.endswith(_CODE_EXCLUDE_SUFFIXES):
                continue
            rels.append(os.path.relpath(os.path.join(dirpath, name), root))
    return rels


def adapter_code_files(adapter) -> tuple[str, list[str]] | None:
    """Auto-detection: `(root_dir, [paths relative to root_dir])`, or `None` if not embeddable.

    For a plain module the set is one `.py`; for a package it is the **whole package tree with its
    layout preserved** -- which is the bug fsd fixes relative to MLflow's `code_paths`, whose
    flattening forces users to rewrite their imports.

    Raises `ValueError` for the `unresolvable` origin and for a set over `MAX_CODE_FILES` /
    `MAX_CODE_BYTES`.
    """
    origin, detail = classify_adapter_source(adapter)
    if origin == "installed":
        return None
    if origin == "unresolvable":
        raise ValueError(f"cannot bundle the adapter's source: {detail}")

    cls = adapter if isinstance(adapter, type) else type(adapter)
    src = detail
    root = _module_root(cls.__module__, src)
    top = cls.__module__.split(".")[0]
    pkg_dir = os.path.join(root, top)
    start = pkg_dir if os.path.isdir(pkg_dir) else src

    rels = _walk_code_files(root, start)
    total = sum(os.path.getsize(os.path.join(root, r)) for r in rels)
    if len(rels) > MAX_CODE_FILES or total > MAX_CODE_BYTES:
        raise ValueError(
            f"the adapter's code at {start!r} is {len(rels)} files / {total / 1e6:.1f} MB, over the "
            f"limit of {MAX_CODE_FILES} files / {MAX_CODE_BYTES / 1e6:.0f} MB. Pass the files "
            "explicitly, e.g. save(..., code=['my_pkg/adapters.py', 'my_pkg/__init__.py']), or "
            "save(..., code=False) to keep the adapter as a pip dependency of the image."
        )
    return root, rels


# --- catch a bundle that saves fine but dies on a cold-start node ------------------


def _check_adapter_at_top(module_name: str, rels: list[str]) -> None:
    """Before anything is copied, the adapter's OWN module must sit at the top level of the
    resolved `code/` tree -- `my_adapter` -> `my_adapter.py`, `my_pkg.adapters` ->
    `my_pkg/adapters.py`.

    If it does not, the manifest's unqualified `module:attr` ref can never import once
    `code/` is on `sys.path`: `resolve_ref` looks up exactly `module_name`, nothing deeper
    (#72).

    A pure path computation over `rels` -- no imports, no network, no node -- so it
    catches `code=[...]` files pulled from two different trees (the common root then
    sits ABOVE the adapter's own directory) before a single byte is copied.
    """
    expected = module_name.replace(".", os.sep) + ".py"
    if expected in rels:
        return

    basename = os.path.basename(expected)
    matches = [r for r in rels if r == expected or r.endswith(os.sep + basename)]
    if not matches:
        raise ValueError(
            f"the adapter's module {module_name!r} is not among the files this bundle "
            f"would embed ({sorted(rels)}); code/ must contain {expected!r}."
        )
    adapter_rel = matches[0]
    adapter_dir = os.path.dirname(adapter_rel)
    prefix = adapter_dir + os.sep if adapter_dir else ""
    culprits = [r for r in rels if r != adapter_rel and not r.startswith(prefix)]
    culprit = culprits[0] if culprits else adapter_rel
    raise ValueError(
        f"{basename} would not be importable at the top of code/ -- the resolved "
        f"import root puts it at {adapter_rel!r} instead of {expected!r}, because "
        f"{culprit!r} sits outside its directory and pulled the common root up. "
        "Fix: keep every embedded file under one directory (pass a single containing "
        "folder to code=[...], or omit code= and let auto-detection embed just the "
        "adapter's own module)."
    )


def _disp(path: str) -> str:
    """Cosmetic only -- format an absolute path relative to cwd for an error message,
    matching the `./demo_model/my_adapter.py` style callers write in `code=[...]`."""
    rel = os.path.relpath(path)
    return rel if rel.startswith("..") else os.path.join(".", rel)


def _is_installed_distribution(name: str) -> bool:
    """Is top-level import name `name` satisfied by something OTHER than a file this
    bundle would embed -- stdlib, a C extension/namespace package, or a pip-installed
    distribution? Uses `importlib.util.find_spec` to LOCATE the module (finders don't
    execute module code) -- an embedded file is never imported here, but classifying a
    dependency name this way is the same mechanism `classify_adapter_source` already uses
    for the adapter itself, via `_is_installed`."""
    if name in sys.stdlib_module_names:
        return True
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, ModuleNotFoundError):
        return False
    if spec is None:
        return False
    if spec.origin is None:  # builtin or namespace package
        return True
    return _is_installed(spec.origin)


def _top_level_imports(tree: ast.AST) -> list[str]:
    """The first dotted name of every absolute import in `tree`, in `ast.walk` order, duplicates kept."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # relative import -- resolved within the package, not a "sibling"
            if node.module:
                names.append(node.module.split(".")[0])
    return names


def _sibling_rel(root: str, name: str) -> str | None:
    """`name`'s file under `root` relative to it (`<name>.py` before `<name>/__init__.py`), or None."""
    if os.path.isfile(os.path.join(root, name + ".py")):
        return name + ".py"
    if os.path.isfile(os.path.join(root, name, "__init__.py")):
        return os.path.join(name, "__init__.py")
    return None  # doesn't resolve under root -- an external dependency, not a sibling


def _check_no_unembedded_siblings(root: str, rels: list[str]) -> None:
    """Refuse a bundle whose embedded `.py` files import a SIBLING module that lives under
    the same resolved root but was not embedded (#71).

    The node raises `ModuleNotFoundError` after a cold start, and the only reason `save`
    succeeded is that the driver still has the sibling on its own `sys.path`/cwd.

    Parses each embedded file with `ast` (never imports it -- the module may need
    dependencies the driver lacks). For every `import X` / `from X import ...`
    anywhere in the file (by its first dotted name, so `import a.b` is `a`): skip
    stdlib/an installed distribution/`fsd` itself (a dependency, declared
    via `requirements=`, never embedded); if `X` resolves to a file under `root` that
    is not in `rels`, refuse, naming the file and the one-line fix. Scanning is
    transitive over embedded files (a sibling that imports a sibling), bounded by the
    same `rels` this bundle would already embed (MAX_CODE_FILES/MAX_CODE_BYTES were
    already enforced upstream in `adapter_code_files`/`_resolve_explicit_code`).

    Known limits, all shared with MLflow's `infer_code_paths`: a module imported dynamically
    (`importlib.import_module(name)`) is invisible to an ast scan, and a non-Python data file
    opened by relative path is not detected.
    """
    processed: set[str] = set()
    checked_names: set[str] = set()
    queue = [r for r in rels if r.endswith(".py")]

    while queue:
        rel = queue.pop()
        if rel in processed:
            continue
        processed.add(rel)
        path = os.path.join(root, rel)
        with open(path, "rb") as f:
            source = f.read()
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue  # not this function's job to validate the adapter's syntax

        for name in _top_level_imports(tree):
            if name in checked_names:
                continue
            checked_names.add(name)
            if name == "fsd" or _is_installed_distribution(name):
                continue

            target_rel = _sibling_rel(root, name)
            if target_rel is None:
                continue

            if target_rel not in rels:
                fix_files = {os.path.join(root, r) for r in rels}
                fix_files.add(os.path.join(root, target_rel))
                fix = ", ".join(repr(_disp(p)) for p in sorted(fix_files))
                raise ValueError(
                    f"{rel} imports {name!r}, which resolves to {_disp(os.path.join(root, target_rel))!r} "
                    "but is not embedded. The node would raise ModuleNotFoundError after a cold "
                    f"start.\nFix: code=[{fix}]"
                )
            if target_rel not in processed:
                queue.append(target_rel)


def _resolve_explicit_code(code) -> tuple[str, list[str]]:
    """`code=[...]` -- take the user at their word, but keep layout relative to a common root."""
    paths = [os.path.abspath(p) for p in code]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"code files do not exist: {missing}")
    root = os.path.commonpath([os.path.dirname(p) if os.path.isfile(p) else p for p in paths])
    rels: list[str] = []
    for p in paths:
        rels += _walk_code_files(root, p)
    return root, rels


# --- declared dependencies (never installed) ---------------------------------


def check_requirements(requirements) -> list[str]:
    """Which of `requirements` are unsatisfied *here*? Human-readable lines; `[]` means ok.

    Declared, never installed: dependency installation stays front-loaded to image build time.
    This is the check the one-node smoke job runs, so a missing `sklearn` fails once, before
    the fan-out, naming the dependency instead of raising a traceback on every node.

    PEP 508 is parsed by `packaging`, not by hand: it gets extras, environment markers and version
    specifiers right in ways a hand-rolled parser does not.
    """
    from packaging.requirements import Requirement

    problems: list[str] = []
    for spec in requirements or []:
        try:
            req = Requirement(spec)
        except Exception as exc:  # noqa: BLE001 - a bad declaration is itself the problem
            problems.append(f"{spec!r} is not a valid requirement ({exc})")
            continue
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            installed = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{spec}: not installed")
            continue
        if req.specifier and not req.specifier.contains(installed, prereleases=True):
            problems.append(f"{spec}: installed {req.name}=={installed}")
    return problems


# --- manifest ----------------------------------------------------------------


def _feature_descriptor(adapter) -> dict:
    """Human-readable provenance for the transform.

    The *executable* version usually ships in the bundle too, under `code/`, so this is a
    summary rather than the only record."""
    seq = getattr(adapter, "feature_sequence", None)
    if seq is None:
        return {"kind": "callable", "steps": ["<adapter.features>"]}
    steps = [getattr(fn, "__name__", repr(fn)) for fn, _ in seq]
    return {"kind": "sequence", "steps": steps}


def _manifest_from_adapter(
    adapter, artifacts_rel: dict[str, str], code_rel: list[str] | None, requirements,
) -> dict:
    manifest = {
        "fsd_bundle_version": BUNDLE_VERSION,
        "adapter": adapter_ref(adapter),
        "artifacts": artifacts_rel,
        "code_origin": "bundled" if code_rel else "installed",
        "feature": _feature_descriptor(adapter),
    }
    if code_rel:
        manifest["code"] = {"root": CODE_DIR, "files": sorted(code_rel)}
    if requirements:
        manifest["requirements"] = list(requirements)
    for field in _SPEC_FIELDS:
        val = getattr(adapter, field, None)
        manifest[field] = list(val) if isinstance(val, (list, tuple)) else val
    return manifest


def manifest_code_files(manifest: dict) -> list[str]:
    """Bundle-relative paths of the embedded code, `[]` if none.

    The single place that knows the `code` block's shape, so the two manifest-driven transports
    (`runners._stage_bundle`, `infer_shard.fetch_bundle_to_scratch`) stay dumb -- they enumerate
    files, they never list directories -- a property the remote transports depend on.
    """
    code = manifest.get("code") or {}
    root = code.get("root", CODE_DIR)
    return [f"{root}/{rel}" for rel in code.get("files", [])]


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "kB", "MB"):
        if size < 1024 or unit == "MB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"  # pragma: no cover - a code bundle over ~1 GB is already refused


def _print_save_report(root: str | None, code_rel: list[str] | None, manifest: dict) -> None:
    """Say what `save` embedded, before it returns (#70). Print, not `logging` --
    the audience is a notebook cell, matching the repo's rule that a consequential
    operation shows what it did (memory `long-process-progress`)."""
    req = manifest.get("requirements")
    req_str = f" | requirements: {', '.join(req)}" if req else ""
    if not code_rel:
        print(f"[bundle] no code embedded (adapter is a pip dependency of the image) "
              f"| adapter {manifest['adapter']}{req_str}", flush=True)
        return

    # The resolved import ROOT is what §7 Q2 calls out as the thing a caller cannot
    # infer -- including one who passed code=[...] themselves, since the root is
    # COMPUTED from their paths, not chosen by them (this is exactly the #72 surprise).
    print(f"[bundle] code root {_disp(root)} -> {CODE_DIR}/", flush=True)
    total = 0
    for rel in sorted(code_rel):
        size = os.path.getsize(os.path.join(root, rel))
        total += size
        print(f"[bundle]   {rel} ({_human_size(size)})", flush=True)
    plural = "" if len(code_rel) == 1 else "s"
    print(f"[bundle] {len(code_rel)} file{plural}, {_human_size(total)} "
          f"| adapter {manifest['adapter']}{req_str}", flush=True)


def read_spec(bundle_path: str) -> dict:
    """Read just `bundle.json` — the spec, with NO import/model-load (model-free preflight).

    Deliberately import-free: the driver validates bands/T against a manifest and never puts a
    bundle's code on its own `sys.path`. Only a process that is about to *run* the model imports it.
    """
    with fs.open(os.path.join(str(bundle_path), BUNDLE_MANIFEST), "r") as f:
        return json.load(f)


def save(
    adapter,
    artifacts: dict[str, str],
    dst: str,
    *,
    overwrite: bool = True,
    code=None,
    requirements=None,
    verbose: bool = True,
) -> str:
    """Write a bundle at `dst`: copy each artifact in, embed the adapter's source, and dump the
    manifest read off `adapter`. Returns the bundle folder path. Storage-seam aware (blob later).

    `artifacts` maps a name -> a local source filepath, e.g. `{"model": "rf.joblib"}`.

    `code` controls source embedding:

    * `None` (default) -- **auto-detect** from the adapter class. A local module or package is
      embedded under `code/` with its layout preserved; a pip-installed adapter is left alone; an
      adapter defined in `__main__`/a notebook cell **raises** with the fix in the message.
    * a list of paths -- embed exactly these (files or directories).
    * `False` -- never embed; the adapter is a pip dependency of the inference image.

    `requirements` is an optional list of PEP 508 strings recorded for the smoke job to check.
    fsd never installs them.

    Before anything is copied, two checks turn a bundle that would save fine and
    die on a cluster node into a `save`-time `ValueError` naming the fix: the adapter's own module
    must sit at the TOP of the resolved `code/` tree (#72), and every sibling import an embedded
    file makes must itself be embedded (#71) -- a dependency (declared via `requirements=`) is
    left alone.

    `verbose=True` (default) prints what got embedded -- the resolved import root, the file
    list with sizes, the adapter ref and the declared requirements -- before returning.
    `verbose=False` silences it; the return value is unaffected either way.
    """
    dst = str(dst)
    fs.makedirs(dst)

    artifacts_rel: dict[str, str] = {}
    for name, src in artifacts.items():
        rel = os.path.basename(src)
        dst_path = os.path.join(dst, rel)
        if fs.exists(dst_path) and not overwrite:
            raise FileExistsError(dst_path)
        with open(src, "rb") as fsrc, fs.open(dst_path, "wb") as fdst:
            fdst.write(fsrc.read())
        artifacts_rel[name] = rel

    detected = None
    if code is False:
        detected = None
    elif code is None:
        detected = adapter_code_files(adapter)
    else:
        detected = _resolve_explicit_code(code)

    code_rel: list[str] | None = None
    root: str | None = None
    if detected is not None:
        root, rels = detected
        cls = adapter if isinstance(adapter, type) else type(adapter)
        _check_adapter_at_top(cls.__module__, rels)
        _check_no_unembedded_siblings(root, rels)

        for rel in rels:
            dst_path = os.path.join(dst, CODE_DIR, rel)
            fs.makedirs(os.path.dirname(dst_path))
            with open(os.path.join(root, rel), "rb") as fsrc, fs.open(dst_path, "wb") as fdst:
                fdst.write(fsrc.read())
        code_rel = rels

    manifest = _manifest_from_adapter(adapter, artifacts_rel, code_rel, requirements)
    with fs.open(os.path.join(dst, BUNDLE_MANIFEST), "w") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        _print_save_report(root, code_rel, manifest)
    return dst


# --- putting the bundle's code on sys.path -----------------------------------


def _module_source_bytes(module) -> bytes | None:
    path = getattr(module, "__file__", None)
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return f.read()


def _guard_module_collision(module_name: str, code_root: str) -> None:
    """A1: refuse to load a bundle whose adapter module name is already taken by *different* code.

    The failure this prevents is a silent wrong answer, not a crash: `sys.modules` caches by name,
    so a second bundle whose module has the same name would quietly get the **first** bundle's
    class and predict with the wrong model (reported against MLflow as mlflow#12377, which cannot
    detect it -- fsd can, because fsd knows the exact file it means to provide).

    Amendment A1: compare **content**, not path. Save-then-load in one process is a normal flow
    (`api._ensure_bundle` does it), and there the imported module and the bundle's copy always sit
    at different paths while being the same code. Byte-identical => not a collision.
    """
    parts = module_name.split(".")
    # Check every ANCESTOR package too, not just the leaf. An already-imported parent package
    # keeps its own `__path__`, so `import mypkg.adapters` would resolve through the stale
    # `mypkg` and never look at the bundle's copy at all -- the same silent-wrong-model failure,
    # one level up and even harder to see.
    for depth in range(1, len(parts) + 1):
        name = ".".join(parts[:depth])
        existing = sys.modules.get(name)
        if existing is None:
            continue

        stem = os.path.join(code_root, *parts[:depth])
        for cand in (stem + ".py", os.path.join(stem, "__init__.py")):
            if os.path.exists(cand):
                with open(cand, "rb") as f:
                    incoming = f.read()
                break
        else:
            continue  # the bundle does not provide this name; nothing to collide with.

        current = _module_source_bytes(existing)
        if current is None:
            raise ValueError(
                f"module {name!r} is already imported but its source cannot be read (no readable "
                f"__file__), so this bundle's {cand!r} cannot be verified as the same code. Load "
                "the bundle in a fresh process."
            )
        if current != incoming:
            raise ValueError(
                f"module {name!r} is already imported from {existing.__file__!r} with DIFFERENT "
                f"source than this bundle's {cand!r}. Python caches modules by name, so loading "
                "this bundle here would silently use the already-imported code and predict with "
                "the wrong model. Load each bundle in its own process."
            )


def _activate_bundle_code(bundle_path: str, manifest: dict) -> None:
    """Prepend `<bundle>/code` to `sys.path` so the bundle's adapter wins.

    Bundle-first is deliberate: `sys.path` is searched in order and the first match wins, so an
    embedded adapter shadows a same-named module in the image. The bundle is the authority on which
    model this is. **This mutates the interpreter's `sys.path`** -- a documented side effect of
    `load`, and the reason `read_spec` exists for anything that must stay import-free.
    """
    if not manifest.get("code"):
        return
    code_root = os.path.join(bundle_path, manifest["code"].get("root", CODE_DIR))
    module_name = manifest["adapter"].partition(":")[0]
    _guard_module_collision(module_name, code_root)
    if code_root not in sys.path:          # idempotent: repeated loads add no duplicates
        sys.path.insert(0, code_root)


def load(bundle_path: str, *, validate: bool = True):
    """Turn a bundle folder back into a ready-to-use adapter.

    Puts the bundle's `code/` on `sys.path` (**mutates `sys.path`**), resolves the
    adapter `module:attr` -> class, instantiates it, injects **absolute** artifact paths onto
    `adapter.artifacts`, (optionally) checks the class's declared spec matches the manifest, and
    calls `adapter.load()`.

    `bundle_path` must be a **local** directory (the node fetches it to scratch first).
    """
    bundle_path = str(bundle_path)
    manifest = read_spec(bundle_path)

    version = manifest.get("fsd_bundle_version")
    if version not in SUPPORTED_BUNDLE_VERSIONS:
        raise ValueError(
            f"unsupported fsd_bundle_version={version!r} (this fsd reads "
            f"{list(SUPPORTED_BUNDLE_VERSIONS)}). Re-save the bundle with this version of fsd."
        )

    # A version-1 bundle has no `code` block, which is indistinguishable from a version-2
    # installed-package bundle: both mean "resolve the ref from the environment" (today's
    # behavior). Nothing that already works stops working.
    _activate_bundle_code(bundle_path, manifest)

    cls = resolve_ref(manifest["adapter"])
    adapter = cls()
    adapter.artifacts = {
        name: os.path.join(bundle_path, rel) for name, rel in manifest["artifacts"].items()
    }

    if validate:
        bundled = bool(manifest.get("code"))
        for field in _SPEC_FIELDS:
            declared = getattr(adapter, field, None)
            declared = list(declared) if isinstance(declared, (list, tuple)) else declared
            # Skip fields the class leaves UNSET (it defers them to the trained model / bundle):
            # None, an empty list, or n_timestamps==0 (the base default). This lets ONE adapter
            # class back models trained on different T without hardcoding it — the bundle is
            # authoritative. Fields the class *does* pin are still drift-checked.
            if declared is None or declared == [] or (field == "n_timestamps" and declared == 0):
                continue
            if manifest.get(field) is not None and declared != manifest[field]:
                # The same check means different things per origin. For an installed adapter it
                # catches genuine code/bundle version skew -- the image's pip version against
                # this bundle. For a bundled adapter both sides came out of one `save()` call,
                # so a disagreement can only mean the bundle was edited.
                why = (
                    "the adapter source INSIDE this bundle — the bundle has been edited"
                    if bundled else
                    f"{manifest['adapter']}.{field}={declared!r} (code/bundle drift)"
                )
                raise ValueError(f"bundle.json {field}={manifest[field]!r} disagrees with {why}.")

    adapter.load()
    return adapter
