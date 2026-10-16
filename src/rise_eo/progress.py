"""`rise_eo.progress` — one shared, throttled progress ticker.

Spec: specs/47-driver-side-honesty.md

Promoted from `workflows.create_datacube.setup`'s `_tick` closure, which #65 named as the
bar to match: `[setup] 34/300 shapes (11%) | 7.9 shapes/s | elapsed 4s | eta 33s`. Every
driver-side loop that can run for minutes on latency-bound I/O (blob uploads, an AML poll
loop, a per-cell merge) should print through this one implementation rather than growing a
second copy of the same throttle + rate + ETA math.
"""

from __future__ import annotations

import time
from collections.abc import Callable

__all__ = ["ticker"]


def ticker(
    total: int,
    label: str,
    *,
    unit: str = "items",
    show_rate: bool = True,
    show_eta: bool = True,
    throttle_s: float = 2.0,
) -> Callable[..., None]:
    """Returns `tick(done, force=False, suffix="")`, throttled to one line per
    `throttle_s` seconds unless `force=True` -- call with `force=True` for the first and
    last tick of a run so both endpoints always print regardless of throttling.

    Line shape: `[label] done/total unit (pct%) | rate unit/s | elapsed Es | eta Xs[ |
    suffix]`. `show_rate=False` drops the rate segment (e.g. the AML poll leg: a
    fan-out's per-second job-completion rate is not a meaningful number). `show_eta=False`
    drops the eta segment (a single-job wait has no rate to derive an ETA from --
    print the elapsed and omit the ETA rather than inventing one). `suffix`, if given, is
    appended as a trailing ` | suffix` segment on that call only.
    """
    t0 = time.time()
    last_print = 0.0

    def tick(done: int, force: bool = False, suffix: str = "") -> None:
        nonlocal last_print
        now = time.time()
        if not force and now - last_print < throttle_s:
            return
        last_print = now
        elapsed = now - t0
        rate = done / elapsed if elapsed > 0 and done else 0.0
        pct = 100 * done / total if total else 100.0
        parts = [f"[{label}] {done}/{total} {unit} ({pct:.0f}%)"]
        if show_rate:
            parts.append(f"{rate:.1f} {unit}/s")
        parts.append(f"elapsed {elapsed:.0f}s")
        if show_eta:
            eta = f"{(total - done) / rate:.0f}s" if rate else "?"
            parts.append(f"eta {eta}")
        if suffix:
            parts.append(suffix)
        print(" | ".join(parts), flush=True)

    return tick
