"""`rise_eo.aml.environment` -- the AML side of an image.

Spec: specs/56-image-definitions-and-registry.md

**The only module under `rise_eo.aml` that touches Azure** -- `az ml environment ...` and
`az ml workspace show`, via `subprocess`. Every function takes its Azure coordinates
explicitly (`resource_group`, `workspace`); rise-eo hard-codes nothing (§7 Q2). Tests stub these
three functions rather than calling `az` -- `ensure_environment` (`rise_eo/aml/__init__.py`)
takes them as injectable keyword arguments for exactly that reason.
"""

from __future__ import annotations

import os
import subprocess

__all__ = ["build_link", "create_environment", "environment_exists", "latest_registered"]

_ENV_YML_TEMPLATE = (
    "$schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json\n"
    "name: {name}\n"
    "build:\n"
    "  path: .\n"
    "  dockerfile_path: Dockerfile\n"
)


def latest_registered(name: str, *, resource_group: str, workspace: str) -> str | None:
    """The highest version AML currently holds for `name`, or `None` if it has never been
    registered.

    Proves the AML **asset** exists -- never that its image finished building."""
    out = subprocess.run(
        ["az", "ml", "environment", "list", "-n", name, "-g", resource_group,
         "-w", workspace, "--query", "[].version", "-o", "tsv"],
        capture_output=True, text=True,
    )
    versions = [v for v in out.stdout.split() if v.isdigit()]
    return max(versions, key=int) if versions else None


def create_environment(
    name: str, context_dir: str, *, resource_group: str, workspace: str,
) -> str:
    """`az ml environment create` from the Dockerfile at `context_dir`, returning the
    version AML assigned. Writes `environment.yml` there if `write_context`/the caller
    didn't already leave one (the `build_context=` escape hatch may).

    Guarded on purpose: `az` can print something other than a version without failing
    (a broken `az` once silently produced `built rise-eo-aml-env:No module named 'rpds.rpds'`),
    so a non-numeric version raises, loudly.
    """
    env_yml = os.path.join(context_dir, "environment.yml")
    if not os.path.exists(env_yml):
        with open(env_yml, "w") as f:
            f.write(_ENV_YML_TEMPLATE.format(name=name))

    out = subprocess.run(
        ["az", "ml", "environment", "create", "-f", env_yml,
         "-g", resource_group, "-w", workspace, "--query", "version", "-o", "tsv"],
        capture_output=True, text=True, cwd=context_dir,
    )
    version = out.stdout.strip()
    if not version.isdigit():
        raise RuntimeError(
            f"{name}: az returned {version!r} (stderr: {out.stderr.strip()[:400]}) -- not a "
            "version number. A broken `az` (or a half-deleted `ml` extension) can produce "
            "exactly this; see notebooks/build_images.ipynb's Troubleshooting."
        )
    return version


def environment_exists(
    name: str, version: str, *, resource_group: str, workspace: str,
) -> bool:
    """Does AML still hold `name:version`? A registry entry whose asset was
    deleted is stale -- `ensure_environment` must not hand back a version that will fail
    at job submission."""
    out = subprocess.run(
        ["az", "ml", "environment", "show", "-n", name, "-v", str(version),
         "-g", resource_group, "-w", workspace, "--query", "name", "-o", "tsv"],
        capture_output=True, text=True,
    )
    return bool(out.stdout.strip())


def build_link(name: str, version: str, *, resource_group: str, workspace: str) -> str:
    """Studio URL for one environment version -- the only way to see build status: an AML
    v2 image build is an ACR task run, not an AML job, so nothing in `az ml job list` shows
    it -- polling for one prints `0/0` forever.
    Returns a URL string -- a library function must not import IPython; the caller (a
    notebook) does `display(Markdown(...))` with it.

    Studio's `wsid` query parameter IS the workspace's ARM resource id -- asked of `az`
    rather than string-built, so a typo in `resource_group`/`workspace` surfaces here
    rather than as a silent 404 on the Studio page.
    """
    ws = subprocess.run(
        ["az", "ml", "workspace", "show", "-n", workspace, "-g", resource_group,
         "--query", "id", "-o", "tsv"], capture_output=True, text=True,
    )
    wsid = ws.stdout.strip()
    if not wsid.startswith("/subscriptions/"):
        raise RuntimeError(
            f"could not resolve the workspace id: {wsid!r} / {ws.stderr.strip()[:300]}"
        )
    tenant = subprocess.run(
        ["az", "account", "show", "--query", "tenantId", "-o", "tsv"],
        capture_output=True, text=True,
    ).stdout.strip()
    tid = f"&tid={tenant}" if tenant else ""
    return f"https://ml.azure.com/environments/{name}/version/{version}?wsid={wsid}{tid}"
