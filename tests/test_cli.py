"""Tests for `rise_eo.cli` (spec 54 D5). `main()` is called with an argv list, never shelled
out to -- the `rise-eo` console script itself is exercised manually (AC 1/2), not here.
"""

from __future__ import annotations

import os

import pytest

import rise_eo.config as config
from rise_eo.cli import main


@pytest.fixture
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RISE_EO_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    for env_name in config._KEY_TO_ENV.values():
        monkeypatch.delenv(env_name, raising=False)
    return tmp_path


def test_init_set_writes_named_keys_only(isolated_config_dir, capsys):
    rc = main(["init", "--set", "workspace=w1", "--set", "cluster=c1"])
    assert rc == 0
    values = config._read_file_values()
    assert values["workspace"] == "w1"
    assert values["cluster"] == "c1"
    assert values["subscription_id"] == ""
    out = capsys.readouterr().out
    assert str(config.config_path()) in out


def test_init_set_rejects_unknown_key(isolated_config_dir, capsys):
    rc = main(["init", "--set", "not_a_key=x"])
    assert rc != 0
    assert not config.config_path().exists()


def test_init_set_rejects_malformed_pair(isolated_config_dir):
    rc = main(["init", "--set", "no-equals-sign"])
    assert rc != 0


def test_init_from_env_file_maps_az_names_to_keys(isolated_config_dir, tmp_path):
    f = tmp_path / "env.local.sh"
    f.write_text(
        "export AZ_ML_WORKSPACE='ws-from-file'\n"
        "export AZ_CLUSTER='c-from-file'\n"
        "export AZ_ROOT='abfss://ignored'\n"
    )
    rc = main(["init", "--from-env-file", str(f)])
    assert rc == 0
    values = config._read_file_values()
    assert values["workspace"] == "ws-from-file"
    assert values["cluster"] == "c-from-file"
    # AZ_ROOT is not a config key since spec 55 D1; it is parsed and dropped, silently
    # (the "say so" decision was dropped at sign-off -- no shipped users to warn).
    assert "root" not in values


def test_init_merges_over_existing_values(isolated_config_dir):
    main(["init", "--set", "workspace=first", "--set", "cluster=c1"])
    main(["init", "--set", "cluster=c2"])
    values = config._read_file_values()
    assert values["workspace"] == "first"
    assert values["cluster"] == "c2"


def test_config_reports_provenance(isolated_config_dir, monkeypatch, capsys):
    main(["init", "--set", "workspace=ws-from-file"])
    monkeypatch.setenv("AZ_CLUSTER", "c-from-env")
    rc = main(["config"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "workspace = 'ws-from-file'  (file)" in out
    assert "cluster = 'c-from-env'  (env)" in out
    assert "subscription_id unset" in out
    assert "image_registry unset (optional)" in out      # AC 8


def test_init_from_env_file_does_not_mutate_environ(isolated_config_dir, tmp_path):
    """AC 7b's third case: the write path leaves the caller's environment alone.

    `--from-env-file` is the one `init` form that reads `AZ_*`-shaped names off disk, so it is
    the one most likely to grow a "helpful" `os.environ` export later (spec 54 D4).
    """
    f = tmp_path / "env.local.sh"
    f.write_text("export AZ_ML_WORKSPACE='ws'\nexport AZ_CLUSTER='c-from-file'\n")
    before = dict(os.environ)
    assert main(["init", "--from-env-file", str(f)]) == 0
    assert dict(os.environ) == before


def test_init_interactive_prompts_and_keeps_existing_on_empty_input(
    isolated_config_dir, monkeypatch, capsys
):
    """D5's primary form: prompt per key, existing value shown as the default and kept on Enter."""
    main(["init", "--set", "workspace=ws-existing", "--set", "cluster=c-existing"])
    capsys.readouterr()
    prompts, answers = [], {"subscription_id": "s-typed", "image_registry": "abfss://typed"}

    def fake_input(prompt):
        prompts.append(prompt)
        key = prompt.split(" ")[0].rstrip(":")
        return answers.get(key, "")          # Enter on everything else

    monkeypatch.setattr("builtins.input", fake_input)
    # A user at a terminal is being simulated, so the terminal has to be simulated too --
    # pytest's stdin is not a tty, which spec 55 D4's guard correctly refuses to prompt on.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    assert main(["init"]) == 0
    values = config._read_file_values()
    assert values["subscription_id"] == "s-typed"        # typed in
    assert values["image_registry"] == "abfss://typed"   # an optional key prompts too
    assert values["workspace"] == "ws-existing"          # kept on empty input
    assert values["cluster"] == "c-existing"
    assert [p for p in prompts if p.startswith("workspace")] == ["workspace [ws-existing]: "]
    assert [p for p in prompts if p.startswith("image_registry")] == ["image_registry (optional): "]
    assert len(prompts) == len(config.KEYS)


def test_init_never_prints_a_pre_existing_value(isolated_config_dir, capsys):
    """`rise-eo init` prints only the path written, never a value -- D5's rule against echoing
    a value it did not just receive from the user."""
    main(["init", "--set", "image_registry=abfss://secret-looking-value"])
    capsys.readouterr()
    main(["init", "--set", "cluster=c2"])
    out = capsys.readouterr().out
    assert "abfss://secret-looking-value" not in out


# --- spec 55 D4: --blank, and a non-tty that explains itself ----------------------------


def test_init_blank_writes_an_empty_template(isolated_config_dir, capsys):
    """AC 6. Prompts for nothing, and the file it leaves behind must parse."""
    rc = main(["init", "--blank"])
    assert rc == 0
    assert config._read_file_values() == {k: "" for k in config.KEYS}
    assert str(config.config_path()) in capsys.readouterr().out


def test_init_blank_refuses_to_clobber_a_filled_file(isolated_config_dir, capsys):
    main(["init", "--set", "workspace=do-not-lose-me"])
    capsys.readouterr()
    assert main(["init", "--blank"]) != 0
    assert config._read_file_values()["workspace"] == "do-not-lose-me"
    assert main(["init", "--blank", "--force"]) == 0
    assert config._read_file_values()["workspace"] == ""


def test_init_blank_is_exclusive_with_the_other_forms(isolated_config_dir):
    with pytest.raises(SystemExit):
        main(["init", "--blank", "--set", "cluster=c"])


def test_init_without_a_tty_names_the_non_interactive_forms(
    isolated_config_dir, monkeypatch, capsys
):
    """AC 7 -- this used to raise EOFError out of `input()`."""
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    rc = main(["init"])
    assert rc != 0
    err = capsys.readouterr().err
    for form in ("--blank", "--set", "--from-env-file"):
        assert form in err
    assert not config.config_path().exists()
