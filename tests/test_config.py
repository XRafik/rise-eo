"""Tests for `rise_eo.config`'s user-level config loader (spec 54, §4 criteria 3-8).

Every test that touches disk goes through `isolated_config_dir`, which points
`RISE_EO_CONFIG_DIR` at `tmp_path` -- spec 37 §7's rule, generalised: no test may reach the
developer's real `~/.config/rise-eo`.
"""

from __future__ import annotations

import os

import pytest

import rise_eo
import rise_eo.config as config


@pytest.fixture
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RISE_EO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    for env_name in config._KEY_TO_ENV.values():
        monkeypatch.delenv(env_name, raising=False)
    return tmp_path


def _write_required(config_mod, **overrides):
    """The five REQUIRED keys, filled in, so a test can isolate the one thing it is about."""
    values = {k: overrides.get(k, k[:2]) for k in config_mod.REQUIRED_KEYS}
    return config_mod.write_config(values)


# --- AC 3: rise_eo.config resolves off a bare `import rise_eo` ------------------------------


def test_config_reachable_off_bare_import():
    assert rise_eo.config.load is config.load


# --- AC 4: precedence -----------------------------------------------------------------


def test_precedence_file_only(isolated_config_dir):
    _write_required(config, workspace="file-ws")
    cfg = config.load()
    assert cfg.workspace == "file-ws"


def test_precedence_env_overrides_file(isolated_config_dir, monkeypatch):
    _write_required(config, workspace="file-ws")
    monkeypatch.setenv("AZ_ML_WORKSPACE", "env-ws")
    cfg = config.load()
    assert cfg.workspace == "env-ws"


def test_precedence_kwarg_overrides_env(isolated_config_dir, monkeypatch):
    _write_required(config, workspace="file-ws")
    monkeypatch.setenv("AZ_ML_WORKSPACE", "env-ws")
    cfg = config.load(workspace="kwarg-ws")
    assert cfg.workspace == "kwarg-ws"


def test_precedence_empty_string_falls_through(isolated_config_dir, monkeypatch):
    _write_required(config, workspace="file-ws")
    monkeypatch.setenv("AZ_ML_WORKSPACE", "")
    cfg = config.load(workspace="")
    assert cfg.workspace == "file-ws"


# --- AC 5: location resolution, all five D1 branches -----------------------------------


def test_config_dir_prefers_rise_eo_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RISE_EO_CONFIG_DIR", str(tmp_path / "rise-eo-dir"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config.config_dir() == tmp_path / "rise-eo-dir"


def test_config_dir_falls_back_to_xdg(tmp_path, monkeypatch):
    monkeypatch.delenv("RISE_EO_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert config.config_dir() == tmp_path / "xdg" / "rise-eo"


def test_config_dir_falls_back_to_posix_default(tmp_path, monkeypatch):
    monkeypatch.delenv("RISE_EO_CONFIG_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(os, "name", "posix")
    monkeypatch.setattr(config.Path, "home", classmethod(lambda cls: tmp_path))
    assert config.config_dir() == tmp_path / ".config" / "rise-eo"


# D1's fifth branch -- `%APPDATA%\\rise-eo` when `os.name == "nt"` -- has no test on purpose:
# `pathlib.Path()` consults `os.name` at construction, so forcing it to "nt" on POSIX makes
# every `Path(...)` raise `NotImplementedError: cannot instantiate 'WindowsPath'`, including
# pytest's own. Testing it would mean a Path shim in `config_dir()` that exists only for the
# test. Verified by reading, not by assertion.


def test_config_dir_relative_rise_eo_config_dir_is_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("RISE_EO_CONFIG_DIR", "relative/path")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(config.Path, "home", classmethod(lambda cls: tmp_path))
    assert config.config_dir() == tmp_path / ".config" / "rise-eo"


def test_config_dir_relative_xdg_config_home_is_ignored(tmp_path, monkeypatch):
    monkeypatch.delenv("RISE_EO_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/xdg")
    monkeypatch.setattr(config.Path, "home", classmethod(lambda cls: tmp_path))
    assert config.config_dir() == tmp_path / ".config" / "rise-eo"


# --- AC 6: TOML round-trips -------------------------------------------------------------


def test_toml_round_trips_adversarial_values(isolated_config_dir):
    import tomllib

    values = {
        "subscription_id": 'has "quotes"',
        "resource_group": "has\\backslash",
        "workspace": "has # a hash",
        "cluster": "has\na newline",
        "uami_client_id": "has non-ascii café",
        "model_registry": "",
        "image_registry": "abfss://c@a.dfs.core.windows.net/r",
    }
    # U+007F is a control character TOML forbids raw in a basic string, and it sits ABOVE the
    # printable range -- an `ord(ch) < 0x20` guard misses it and writes a file tomllib rejects.
    values["workspace"] = "has \x7f a delete"
    text = config._emit_toml(values)
    parsed = tomllib.loads(text)["azure"]
    assert parsed == {k: values[k] for k in config.KEYS}


# --- AC 7: MissingConfig names every missing key and mentions `rise-eo init` ---------------


def test_missing_config_lists_every_missing_key(isolated_config_dir):
    config.write_config({"subscription_id": "only-this-one"})
    with pytest.raises(config.MissingConfig) as exc:
        config.load()
    msg = str(exc.value)
    for key in ("resource_group", "workspace", "cluster", "uami_client_id"):
        assert key in msg
    assert "subscription_id" not in exc.value.missing
    assert "rise-eo init" in msg


def test_missing_config_is_a_key_error(isolated_config_dir):
    with pytest.raises(KeyError):
        config.load()


# --- AC 7b: neither load() nor init writes to os.environ --------------------------------


def test_load_does_not_mutate_environ_reading_file(isolated_config_dir):
    _write_required(config)
    before = dict(os.environ)
    config.load()
    assert dict(os.environ) == before


def test_load_does_not_mutate_environ_reading_env(isolated_config_dir, monkeypatch):
    for key, env_name in config._KEY_TO_ENV.items():
        monkeypatch.setenv(env_name, f"value-{key}")
    before = dict(os.environ)
    config.load()
    assert dict(os.environ) == before


def test_write_config_does_not_mutate_environ(isolated_config_dir):
    before = dict(os.environ)
    config.write_config({"workspace": "w"})
    assert dict(os.environ) == before


# --- AC 8: no test reaches the real ~/.config/rise-eo ---------------------------------------


def test_isolated_config_dir_is_not_the_real_one(isolated_config_dir):
    assert config.config_dir() != config.Path.home() / ".config" / "rise-eo"


# --- parse_env_file (moved verbatim from notebooks/_config.py) --------------------------


def test_parse_env_file_reads_quoted_and_bare_forms(tmp_path):
    f = tmp_path / "env.local.sh"
    f.write_text(
        "# a comment line\n"
        "export AZ_RG='my-rg'                 # resource group\n"
        'export AZ_ML_WORKSPACE="my-ws"       # workspace\n'
        "export AZ_CLUSTER=bare-value         # unquoted\n"
        "not an export line\n"
    )
    assert config.parse_env_file(f) == {
        "AZ_RG": "my-rg", "AZ_ML_WORKSPACE": "my-ws", "AZ_CLUSTER": "bare-value",
    }


def test_parse_env_file_skips_empty_and_derived_values(tmp_path):
    f = tmp_path / "env.local.sh"
    f.write_text(
        "export AZ_RG=''\n"
        'export AZ_DERIVED="${AZ_RG}/x"\n'
        'export AZ_FROM_CMD="$(az account show --query id -o tsv)"\n'
        "export AZ_REAL='kept'\n"
    )
    assert config.parse_env_file(f) == {"AZ_REAL": "kept"}


# --- write_config merge behaviour --------------------------------------------------------


def test_write_config_merges_over_existing_values(isolated_config_dir):
    config.write_config({"workspace": "first", "cluster": "c1"})
    config.write_config({"cluster": "c2"})
    values = config._read_file_values()
    assert values["workspace"] == "first"
    assert values["cluster"] == "c2"


def test_key_to_env_map_is_a_bijection():
    assert len(config._KEY_TO_ENV) == len(config.ENV_TO_KEY) == len(config.KEYS)
    for key, env_name in config._KEY_TO_ENV.items():
        assert config.ENV_TO_KEY[env_name] == key


# --- spec 55: `root` is gone, the registries are optional -------------------------------


def test_root_is_not_a_config_key():
    """Spec 55 D1. `root` names a per-run destination, so the caller passes it."""
    assert "root" not in config.KEYS
    assert "AZ_ROOT" not in config._KEY_TO_ENV.values()


def test_load_rejects_root_kwarg(isolated_config_dir):
    """AC 4 -- passing `root=` is exactly the mistake a spec-54-era caller makes, so it must
    fail loudly rather than being silently accepted and dropped."""
    _write_required(config)
    with pytest.raises(TypeError, match="root"):
        config.load(root="abfss://nope")


def test_required_and_optional_split(isolated_config_dir):
    """AC 1 + AC 2: the five raise when unset, the two do not."""
    assert config.KEYS == config.REQUIRED_KEYS + config.OPTIONAL_KEYS
    _write_required(config)
    cfg = config.load()
    assert cfg.model_registry is None and cfg.image_registry is None
    assert cfg.subscription_id


def test_missing_config_names_only_required_keys(isolated_config_dir):
    with pytest.raises(config.MissingConfig) as exc:
        config.load()
    assert set(exc.value.missing) == set(config.REQUIRED_KEYS)
    for optional in config.OPTIONAL_KEYS:
        assert optional not in exc.value.missing


def test_optional_key_precedence(isolated_config_dir, monkeypatch):
    """AC 5: kwarg > env > file for an optional key too, with empty falling through."""
    _write_required(config)
    config.write_config({"image_registry": "abfss://file-reg"})
    assert config.load().image_registry == "abfss://file-reg"
    monkeypatch.setenv("AZ_IMAGE_REGISTRY", "abfss://env-reg")
    assert config.load().image_registry == "abfss://env-reg"
    assert config.load(image_registry="abfss://kwarg-reg").image_registry == "abfss://kwarg-reg"
    monkeypatch.setenv("AZ_IMAGE_REGISTRY", "")
    assert config.load(image_registry="").image_registry == "abfss://file-reg"


def test_write_blank_config_writes_every_key_empty(isolated_config_dir):
    """AC 6 half: a blank file must PARSE, so `load()` names the gaps rather than tomllib
    reporting a syntax error."""
    path = config.write_blank_config()
    assert path.exists()
    assert config._read_file_values() == {k: "" for k in config.KEYS}
    with pytest.raises(config.MissingConfig):
        config.load()


# --- spec 102 A11.5: the config folder is `rise-eo`; the old `fsd` folder is not read --------


def _only_old_config(tmp_path, monkeypatch):
    monkeypatch.delenv("RISE_EO_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    for env_name in config._KEY_TO_ENV.values():
        monkeypatch.delenv(env_name, raising=False)
    old = tmp_path / "xdg" / "fsd"
    old.mkdir(parents=True)
    (old / "config.toml").write_text('[azure]\nsubscription_id = "from-the-old-folder"\n')
    return old / "config.toml"


def test_old_fsd_config_is_not_read_and_the_error_prints_the_move_command(tmp_path, monkeypatch):
    old = _only_old_config(tmp_path, monkeypatch)
    with pytest.raises(config.MissingConfig) as exc:
        config.load()
    msg = str(exc.value)
    assert "subscription_id" in exc.value.missing        # the old value was not picked up
    new = tmp_path / "xdg" / "rise-eo" / "config.toml"
    assert f"mv '{old}' '{new}'" in msg


def test_no_move_hint_when_there_is_no_old_config(tmp_path, monkeypatch):
    old = _only_old_config(tmp_path, monkeypatch)
    old.unlink()
    with pytest.raises(config.MissingConfig) as exc:
        config.load()
    assert "mv " not in str(exc.value)
