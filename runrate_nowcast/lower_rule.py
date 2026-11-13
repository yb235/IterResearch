"""How a "surpassed / topped / over $X" claim (kind = lower) enters the state-space filter (sub-task N).

Pre-registered in DATA_PLAN.md (section 子任务 N, commit abce18a) before any full rerun:

  baseline  one-sided: true value > X (the original rule; config.LOWER_UPPER_RATIO = None)
  iv115     PRIMARY. X <= true value <= 1.15 X, a two-sided truncated-normal update
  iv110     the same with upper edge 1.10 X
  iv125     the same with upper edge 1.25 X
  point     the claim is treated as a point claim at X (the row's tier noise)

The rule applies to every kind == "lower" row of metric run_rate or revenue, all segments, both
companies, except forecast claims (is_forecast = 1, which config.USE_FORECASTS keeps out of the
evidence anyway). It is applied inside model.Model when the claims are compiled into filter
observations, so every consumer of the filter gets it (backtest, calibrate's refits, the crossing
profile, bayes, panel_model / ml_fixed labels) and data.select, the evidence rows and the target
selection are unchanged. The upper edge is derived from the claim itself: it adds no row and no
information from after the vintage.

The setting is process-wide (config.LOWER_UPPER_RATIO). Worker processes started by fork inherit it.
In-process caches keyed on the evidence rows (panel_model._NOISE_CACHE, ml_fixed._RHO_CACHE) do not
see the rule, so run one variant per process or call ml_fixed.clear_caches() after switching.
"""
from __future__ import annotations

import contextlib

from . import config

VARIANTS = {"baseline": None, "point": "point", "iv110": 1.10, "iv115": 1.15, "iv125": 1.25}
PRIMARY = "iv115"
ORDER = ("baseline", "point", "iv110", "iv115", "iv125")


def setting(variant: str):
    if variant not in VARIANTS:
        raise ValueError(f"unknown lower-claim variant {variant!r}; one of {sorted(VARIANTS)}")
    return VARIANTS[variant]


def apply(variant: str) -> None:
    """Set the rule for the rest of this process."""
    config.LOWER_UPPER_RATIO = setting(variant)


@contextlib.contextmanager
def lower_rule(variant: str):
    """Use a variant inside the block, then restore the previous setting."""
    old = config.LOWER_UPPER_RATIO
    config.LOWER_UPPER_RATIO = setting(variant)
    try:
        yield
    finally:
        config.LOWER_UPPER_RATIO = old


def describe(variant: str) -> str:
    v = setting(variant)
    if v is None:
        return "lower claims one-sided (true value > X)"
    if v == "point":
        return "lower claims as points at X"
    return f"lower claims two-sided, X <= true value <= {v:.2f} X"
