"""Spec 47 D4/D5 — `rise_eo.progress.ticker`, the shared throttled progress helper."""

import re

from rise_eo import progress


def test_ticker_prints_expected_shape(capsys):
    """Pinned as a regex, not as `endswith("s")`: when `elapsed` rounds to 0.0 the rate is 0
    and the eta segment is legitimately `eta ?` rather than `eta <n>s` (the zero-rate branch
    `test_ticker_zero_rate_prints_unknown_eta` covers deliberately). Asserting a trailing
    "s" made this test depend on the wall clock, and it failed under load."""
    tick = progress.ticker(300, "setup", unit="shapes")
    tick(34, force=True)
    out = capsys.readouterr().out.strip()
    assert re.fullmatch(
        r"\[setup\] 34/300 shapes \(11%\) \| [\d.]+ shapes/s \| elapsed \d+s \| eta (\?|\d+s)",
        out,
    ), out


def test_ticker_throttles_unless_forced(capsys):
    tick = progress.ticker(10, "x", unit="items", throttle_s=60.0)
    tick(1, force=True)
    tick(2)                        # inside the throttle window -> suppressed
    out = capsys.readouterr().out
    assert out.count("[x]") == 1


def test_ticker_force_always_prints(capsys):
    tick = progress.ticker(10, "x", unit="items", throttle_s=60.0)
    tick(1, force=True)
    tick(2, force=True)
    out = capsys.readouterr().out
    assert out.count("[x]") == 2


def test_ticker_show_rate_false_drops_rate_segment(capsys):
    tick = progress.ticker(10, "aml", unit="jobs terminal", show_rate=False)
    tick(3, force=True)
    out = capsys.readouterr().out
    assert "/s" not in out
    assert "elapsed" in out and "eta" in out


def test_ticker_show_eta_false_drops_eta_segment(capsys):
    tick = progress.ticker(1, "aml", unit="jobs terminal", show_rate=False, show_eta=False)
    tick(0, force=True)
    out = capsys.readouterr().out
    assert "eta" not in out
    assert "elapsed" in out


def test_ticker_suffix_appended_as_trailing_segment(capsys):
    tick = progress.ticker(10, "aml", unit="jobs terminal", show_rate=False)
    tick(3, force=True, suffix="7 running")
    out = capsys.readouterr().out.strip()
    assert out.endswith("| 7 running")


def test_ticker_zero_rate_prints_unknown_eta(capsys):
    tick = progress.ticker(10, "x", unit="items")
    tick(0, force=True)
    out = capsys.readouterr().out
    assert "eta ?" in out
