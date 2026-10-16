"""`ImageDefinition` -- an AML node image, declared as data.

Spec: specs/56-image-definitions-and-registry.md

A frozen dataclass, no methods that touch the network. It renders a Dockerfile
(`render_dockerfile`) and a build context (`write_context`); it does not build one --
that is `rise_eo.aml.ensure_environment`'s job.

**`local` is REQUIRED on a node image (#80).** `snakemake` moved to the `[local]` extra, and
the in-job entrypoints (`workflows/shard.py`, `workflows/infer_shard.py`) call straight back
into `runners.run_local` -- a node runs the same Snakemake orchestration a laptop does.
An image built without it fails ~30 min into a dispatch, not at build time. So the node
extras are `[local,azure,mpc]`.

**Why `[azure,mpc]` and not `[aml]` or `[grid]`:** `azure` brings `adlfs` +
`azure-identity` + `azure-keyvault-secrets` -- blob I/O
through the storage seam, managed-identity auth, Key Vault creds on the node. `mpc` brings
`planetary-computer` -- it signs asset hrefs on the node, right before transfer. NOT
`aml`: `azure-ai-ml` is the driver-side dispatch SDK: the node never submits jobs. NOT
`grid`: `s2`/`s2cell` split the ROI into grid cells on the driver, before any job exists --
a node only ever reads the cells it is handed.

**The escape hatch** (#79: "caller may pass their own build context"): `ImageDefinition(
build_context="./images/mine")` takes a directory the caller owns; `rise_eo.image.digest`
digests its contents unchanged, and `render_dockerfile`/`write_context` are not used --
the builder is pointed at `build_context` directly.
"""

from __future__ import annotations

import dataclasses
import glob
import os
import subprocess
import sys

DEFAULT_BASE = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest"

__all__ = ["ImageDefinition"]


@dataclasses.dataclass(frozen=True)
class ImageDefinition:
    """`name` never affects the digest -- it is where the image is registered, not
    what it contains. `rise_eo` is a pip-installable reference: `"git+https://...@<ref>"`, a
    PyPI spec, or `"path:/local/checkout"` for a rise-eo developer's own tree (the one case
    `resolve()` builds a wheel for). `extras`/`extra_pip` are rise-eo's own extras
    (`azure`, `mpc`, ...) and any additional plain pip packages (an inference image's
    `scikit-learn`, `joblib`). `build_context`, when set, is the escape hatch above and
    makes every other field except it purely informational.
    """

    name: str
    base: str = DEFAULT_BASE
    rise_eo: str | None = None
    extras: tuple[str, ...] = ()
    extra_pip: tuple[str, ...] = ()
    build_context: str | None = None

    def derive(self, **overrides) -> "ImageDefinition":
        """A new definition sharing every field not overridden -- `BASE.derive(name=...,
        extra_pip=...)` is how an inference image is built from a general-purpose one
        without repeating `base`/`rise_eo`/`extras`."""
        return dataclasses.replace(self, **overrides)

    def render_dockerfile(self) -> str:
        """The two Dockerfiles in `notebooks/images/` collapse to this: a `FROM` line and
        one `RUN pip install` line. Raises if `build_context` is set -- that mode's
        Dockerfile is the caller's, not rise-eo's to render."""
        if self.build_context is not None:
            raise ValueError(
                "render_dockerfile() does not apply when build_context= is set -- the "
                "caller's directory already holds the Dockerfile rise-eo will build."
            )
        if not self.rise_eo:
            raise ValueError(
                "ImageDefinition.rise_eo must be set to render a Dockerfile (or pass "
                "build_context= for a caller-owned context)."
            )
        extras_suffix = f"[{','.join(self.extras)}]" if self.extras else ""
        extra_pip_suffix = f" {' '.join(self.extra_pip)}" if self.extra_pip else ""
        lines = [f"FROM {self.base}", ""]
        if self.rise_eo.startswith("path:"):
            lines += [
                "COPY rise_eo-*.whl /tmp/",
                "",
                f'RUN python -m pip install --no-cache-dir "$(ls /tmp/rise_eo-*.whl){extras_suffix}"'
                f"{extra_pip_suffix} \\\n && python -m pip cache purge || true",
            ]
        else:
            spec = f"rise-eo{extras_suffix} @ {self.rise_eo}"
            lines += [
                f'RUN python -m pip install --no-cache-dir "{spec}"{extra_pip_suffix} \\\n'
                " && python -m pip cache purge || true",
            ]
        return "\n".join(lines) + "\n"

    def write_context(self, dir: str) -> str:
        """Materialize a build context at the local directory `dir`: the rendered
        Dockerfile, plus a built wheel for a `path:` `rise_eo` reference -- the one slow path;
        everything else installs straight from the pip-installable `rise_eo` reference inside the
        image, with no local build. Returns `dir`. Not used when
        `build_context` is set -- point the builder at that directory directly."""
        if self.build_context is not None:
            raise ValueError(
                "write_context() does not apply when build_context= is set -- the "
                "caller's directory IS the build context."
            )
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, "Dockerfile"), "w") as f:
            f.write(self.render_dockerfile())
        if self.rise_eo and self.rise_eo.startswith("path:"):
            # A wheel already here is the one `digest.resolve(..., wheel_dir=dir)` just built
            # and hashed; rebuilding would put a DIFFERENT file in the image than the one the
            # registry records.
            if not glob.glob(os.path.join(dir, "rise_eo-*.whl")):
                _build_wheel(self.rise_eo[len("path:"):], dir)
        return dir


def _build_wheel(src_dir: str, dest_dir: str) -> str:
    """`pip wheel {src_dir} --no-deps -w {dest_dir}` -- the same call
    `build_images.ipynb` made, now library code. `--no-deps`: the rise-eo wheel alone; the
    rendered Dockerfile's `pip install` resolves its dependencies inside the image.

    `--no-build-isolation` first, then a retry without it. With it, the build backend has to
    already be importable here -- true in a rise-eo dev venv on 3.11, and the only way
    `tests/test_image_digest.py` builds a wheel without reaching the network. Without it, pip
    downloads `setuptools` into an isolated env, which is what a 3.12+ venv (no bundled
    setuptools) needs. Trying the offline form first keeps the network-free path fast and
    still works where it cannot apply."""
    def _run(extra: list[str]):
        return subprocess.run(
            [sys.executable, "-m", "pip", "wheel", src_dir, "--no-deps", *extra,
             "-w", dest_dir],
            capture_output=True, text=True,
        )

    result = _run(["--no-build-isolation"])
    if result.returncode != 0:
        result = _run([])
    if result.returncode != 0:
        raise RuntimeError(f"pip wheel {src_dir!r} failed: {result.stderr[-2000:]}")
    wheels = sorted(glob.glob(os.path.join(dest_dir, "rise_eo-*.whl")))
    if not wheels:
        raise RuntimeError(f"pip wheel {src_dir!r} produced no rise_eo-*.whl in {dest_dir!r}")
    return wheels[-1]
