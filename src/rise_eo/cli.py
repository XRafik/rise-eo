"""rise-eo's console script.

Spec: specs/54-user-level-config.md

`rise-eo init` writes `~/.config/rise-eo/config.toml` (or wherever `rise_eo.config.config_dir()`
resolves to); `rise-eo config` shows the resolved values and where each came from. Both are
operator-facing tools that ship with the library -- `src/rise_eo/` itself never calls this
module or reads the file it writes.
"""

from __future__ import annotations

import argparse
import os
import sys

from rise_eo import config as rise_eo_config


def _cmd_init(args: argparse.Namespace) -> int:
    if args.blank:
        existing = rise_eo_config._read_file_values()
        if any(existing.values()) and not args.force:
            print(
                f"rise-eo init --blank: {rise_eo_config.config_path()} already holds values. "
                "--blank would erase nothing but writes an empty template over your edits; "
                "pass --force if that is what you want.",
                file=sys.stderr,
            )
            return 2
        # write_config drops empty values, so a blank file cannot go through it (
        # keys present and empty, so the file PARSES and `load()` names the gaps instead of
        # `tomllib` reporting a syntax error).
        path = rise_eo_config.write_blank_config()
        print(f"Wrote {path}")
        return 0

    if args.from_env_file:
        env_values = rise_eo_config.parse_env_file(args.from_env_file)
        updates = {
            rise_eo_config.ENV_TO_KEY[env_name]: value
            for env_name, value in env_values.items()
            if env_name in rise_eo_config.ENV_TO_KEY
        }
    elif args.set:
        updates = {}
        for item in args.set:
            key, sep, value = item.partition("=")
            if not sep:
                print(f"rise-eo init --set: {item!r} is not key=value", file=sys.stderr)
                return 2
            if key not in rise_eo_config.KEYS:
                print(
                    f"rise-eo init --set: unknown key {key!r} "
                    f"(expected one of {', '.join(rise_eo_config.KEYS)})",
                    file=sys.stderr,
                )
                return 2
            updates[key] = value
    else:
        if not sys.stdin.isatty():
            # Prompting is the default form, so a piped shell or a CI step lands here and used
            # to die on a bare EOFError out of `input()`. Name the three non-interactive forms.
            print(
                "rise-eo init: no terminal to prompt on. Use one of:\n"
                "  rise-eo init --blank                  write an empty config.toml to fill in\n"
                "  rise-eo init --set key=value …        set named keys\n"
                "  rise-eo init --from-env-file PATH     import an env.local.sh-shaped file",
                file=sys.stderr,
            )
            return 2
        existing = rise_eo_config._read_file_values()
        updates = {}
        for key in rise_eo_config.KEYS:
            default = existing.get(key, "")
            optional = " (optional)" if key in rise_eo_config.OPTIONAL_KEYS else ""
            prompt = f"{key}{optional} [{default}]: " if default else f"{key}{optional}: "
            entered = input(prompt).strip()
            updates[key] = entered or default

    path = rise_eo_config.write_config(updates)
    print(f"Wrote {path}")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    file_values = rise_eo_config._read_file_values()
    print(f"config file: {rise_eo_config.config_path()}")
    for key in rise_eo_config.KEYS:
        env_value = os.environ.get(rise_eo_config._KEY_TO_ENV[key])
        file_value = file_values.get(key)
        if env_value:
            source, value = "env", env_value
        elif file_value:
            source, value = "file", file_value
        else:
            source, value = None, None
        if source:
            print(f"  {key} = {value!r}  ({source})")
        elif key in rise_eo_config.OPTIONAL_KEYS:
            print(f"  {key} unset (optional)")
        else:
            print(f"  {key} unset")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rise-eo")
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init", help="write the user-level config file")
    group = init_parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from-env-file", metavar="PATH",
        help="parse an env.local.sh-shaped file and write the AZ_* values it yields",
    )
    group.add_argument(
        "--set", metavar="KEY=VALUE", action="append",
        help="set one config key (repeatable); other keys are left as they are",
    )
    group.add_argument(
        "--blank", action="store_true",
        help="write config.toml with every key present and empty; prompt for nothing",
    )
    init_parser.add_argument(
        "--force", action="store_true",
        help="with --blank: overwrite a config.toml that already holds values",
    )
    init_parser.set_defaults(func=_cmd_init)

    config_parser = sub.add_parser(
        "config", help="print the resolved config and where each value came from"
    )
    config_parser.set_defaults(func=_cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
