"""Audit of hand-set choices: which ones actually move the conclusions?

Each entry below is a number, rule or specification chosen by hand rather than estimated
from data. The audit sets each one to plausible alternatives (usually half and double,
or the competing rule) with everything else fixed, then records how four headline
outputs move:
    oct31   official-basis run-rate on 2026-10-31 (median, $B)
    q4      Q4 2026 revenue (mean, $B)
    mae     backtest log error over the pre-publication targets
    cov80   backtest 80%-interval coverage
A choice is "material" if any alternative moves oct31 or q4 by more than half the
baseline 80% half-width, or moves backtest MAE by more than 0.01. Material choices must
be estimated from data or carried as stated uncertainty. The rest can stay as they are.

python -m runrate_nowcast audit            # ~8 minutes, writes reports/audit_<asof>.md
"""
from __future__ import annotations

import contextlib
import copy
import math
import time

import pandas as pd

from . import backtest, calibrate, config, data, report
from . import model as model_mod

ASOF = "2026-11-10"
CROSSING = ["rr_2026_03", "rr_2026_04", "rr_2026_05", "cc_2025_11"]          # "crossed $X" claims
BORDERLINE_OFFICIAL = ["rr_2025_10", "rr_2025_12", "rr_2026_03"]           # official only via a press report


@contextlib.contextmanager
def patched(obj, attr, value):
    old = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield
    finally:
        setattr(obj, attr, old)


@contextlib.contextmanager
def patched_item(d, key, value):
    old = d[key]
    d[key] = value
    try:
        yield
    finally:
        d[key] = old


TARGETS = None    # fixed backtest target ids, set from the baseline run


def evaluate(df, params, asof=ASOF, with_backtest=True) -> dict:
    _, res, _ = report.run_nowcast(df, params, asof)
    o = res.runrate("2026-10-31", "t28x13")
    q4 = res.period_revenue("2026-10-01..2026-12-31")
    out = {"oct31": o["median"], "oct31_q10": o["q10"], "oct31_q90": o["q90"],
           "q4": q4["mean"], "q4_q10": q4["q10"], "q4_q90": q4["q90"]}
    if with_backtest:
        s = backtest.summarize(backtest.run_backtest(df, params, target_ids=TARGETS))
        out.update({"mae": s["mae_log_model"], "cov80": s["coverage80"]})
    return out


def _p(params, **kw):
    p = copy.deepcopy(params)
    for k, v in kw.items():
        p[k] = v
    return p


def _tier(params, tier, sd):
    p = copy.deepcopy(params)
    p.setdefault("tiers", {}).setdefault(tier, {})["log_sd"] = sd
    return p


def _segments(params, field, mult):
    p = copy.deepcopy(params)
    for s, v in config.SEGMENTS.items():
        p.setdefault("segments", {}).setdefault(s, {})[field] = v[field] * mult
    return p


def _indicators(params, **kw):
    p = copy.deepcopy(params)
    for name, spec in p["indicators"].items():
        if spec.get("role") == "model":
            spec.update(kw)
    return p


def _df_edit(df, fn):
    d = df.copy()
    fn(d)
    return d


def _shift_dates(d, ids, days):
    m = d.obs_id.isin(ids)
    d.loc[m, "ref_date"] = d.loc[m, "ref_date"] + pd.Timedelta(days=days)


def _ind_lag(df, days):
    d = df.copy()
    m = d.metric == "indicator"
    d.loc[m, "ref_date"] = d.loc[m, "ref_date"] - pd.Timedelta(days=days)
    return d


def checks():
    """(name, category, current setting, [(label, runner)]) where runner(df, params) -> metrics."""
    E = evaluate
    L = config.DEFINITION_LAG_DAYS
    return [
        # ---- source noise
        ("official tier noise", "source noise", "0.04 (data-checked, n=6)",
         [("0.02", lambda df, p: E(df, _tier(p, "official", 0.02))), ("0.08", lambda df, p: E(df, _tier(p, "official", 0.08)))]),
        ("leak_tier1 tier noise", "source noise", "0.08 (hand-set)",
         [("0.04", lambda df, p: E(df, _tier(p, "leak_tier1", 0.04))), ("0.16", lambda df, p: E(df, _tier(p, "leak_tier1", 0.16)))]),
        ("leak_single tier noise", "source noise", "0.15 (hand-set)",
         [("0.08", lambda df, p: E(df, _tier(p, "leak_single", 0.08))), ("0.30", lambda df, p: E(df, _tier(p, "leak_single", 0.30)))]),
        ("third_party tier noise", "source noise", "0.15 (hand-set)",
         [("0.08", lambda df, p: E(df, _tier(p, "third_party", 0.08))), ("0.30", lambda df, p: E(df, _tier(p, "third_party", 0.30)))]),
        ("GAAP noise", "source noise", "0.003 (rounding)",
         [("0.001", lambda df, p: E(df, _tier(p, "gaap", 0.001))), ("0.01", lambda df, p: E(df, _tier(p, "gaap", 0.01)))]),
        # ---- claim encoding
        ("t28x13 definition lag", "claim encoding", "14 days (from the S-1 definition, via secondary source)",
         [("7", lambda df, p: _with_lag(df, p, "t28x13", 7)), ("21", lambda df, p: _with_lag(df, p, "t28x13", 21))]),
        ("'unknown' definition lag", "claim encoding", "7 days (hand-set)",
         [("0", lambda df, p: _with_lag(df, p, "unknown", 0)), ("14", lambda df, p: _with_lag(df, p, "unknown", 14))]),
        ("date uncertainty (date_sd_days)", "claim encoding", "window width / sqrt(12) from source wording",
         [("x0.5", lambda df, p: E(_df_edit(df, lambda d: d.__setitem__("date_sd_days", d.date_sd_days * 0.5)), p)),
          ("x2", lambda df, p: E(_df_edit(df, lambda d: d.__setitem__("date_sd_days", d.date_sd_days * 2)), p))]),
        ("crossing-date placement", "claim encoding", "window midpoint from source wording",
         [("-7d", lambda df, p: E(_df_edit(df, lambda d: _shift_dates(d, CROSSING, -7)), p)),
          ("+7d", lambda df, p: E(_df_edit(df, lambda d: _shift_dates(d, CROSSING, 7)), p))]),
        ("'over $X' as lower bound", "claim encoding", "rule: truncated observation",
         [("as point", lambda df, p: E(_df_edit(df, lambda d: d.__setitem__("kind", d.kind.replace({"lower": "point"}))), p))]),
        ("borderline 'official' tiering", "claim encoding", "3 claims official via press report",
         [("demote", lambda df, p: E(_df_edit(df, lambda d: d.__setitem__(
             "tier", d.tier.where(~d.obs_id.isin(BORDERLINE_OFFICIAL), "leak_tier1"))), p))]),
        # ---- total-layer structure / priors
        ("long-run growth g_bar (damping target)", "structure/prior", "0.03/month (hand-set)",
         [("0.0", lambda df, p: E(df, _p(p, growth_longrun=0.0))), ("0.06", lambda df, p: E(df, _p(p, growth_longrun=0.06)))]),
        ("sigma_l (not identified)", "structure/prior", "MAP = prior",
         [("x0.5", lambda df, p: E(df, _p(p, sigma_l=p["sigma_l"] * 0.5))), ("x2", lambda df, p: E(df, _p(p, sigma_l=p["sigma_l"] * 2)))]),
        ("sigma_b (not identified)", "structure/prior", "MAP = prior",
         [("x0.25", lambda df, p: E(df, _p(p, sigma_b=p["sigma_b"] * 0.25))), ("x3", lambda df, p: E(df, _p(p, sigma_b=p["sigma_b"] * 3)))]),
        ("prior on starting level/growth/gap", "structure/prior", "l: 0±0.5, g: 0.15±0.2, b: 0±0.25",
         [("tight x0.5", lambda df, p: E(df, _p(p, prior_l_sd=0.25, prior_g_sd=0.1, prior_b_sd=0.125))),
          ("wide x2", lambda df, p: E(df, _p(p, prior_l_sd=1.0, prior_g_sd=0.4, prior_b_sd=0.5)))]),
        # ---- segment layer

        ("sum-to-one constraint noise", "segment", "0.005 weekly (hand-set)",
         [("0.001", lambda df, p: _with_attr(df, p, config, "CONSTRAINT_SD", 0.001)),
          ("0.02", lambda df, p: _with_attr(df, p, config, "CONSTRAINT_SD", 0.02))]),
        ("segment share drift", "segment", "0.005 / 0.02 per sqrt(day) (hand-set)",
         [("x0.5", lambda df, p: E(df, _segments(p, "sigma", 0.5))), ("x2", lambda df, p: E(df, _segments(p, "sigma", 2)))]),
        ("segment share priors (sd)", "segment", "0.3-1.0 (hand-set)",
         [("x0.5", lambda df, p: E(df, _segments(p, "prior_sd", 0.5))), ("x2", lambda df, p: E(df, _segments(p, "prior_sd", 2)))]),
        # ---- indicator layer
        ("indicator params (beta, drift, noise)", "indicator", "MAP fit on full sample",
         [("round: beta .85 drift .01", lambda df, p: E(df, _indicators(p, beta=0.85, drift=0.01, obs_sd=0.04))),
          ("round: beta 1 drift .03", lambda df, p: E(df, _indicators(p, beta=1.0, drift=0.03, obs_sd=0.04)))]),
        ("indicator lead/lag", "indicator", "0 days (assumed contemporaneous)",
         [("leads 7d", lambda df, p: E(_ind_lag(df, -7), _shift_breaks(p, -7))),
          ("lags 7d", lambda df, p: E(_ind_lag(df, 7), _shift_breaks(p, 7)))]),
        ("break variance", "indicator", "0.5 (hand-set)",
         [("0.2", lambda df, p: _with_attr(df, p, model_mod, "BREAK_SD", 0.2)),
          ("1.0", lambda df, p: _with_attr(df, p, model_mod, "BREAK_SD", 1.0))]),
        ("Ramp drift (2 points, not identified)", "indicator", "0.005 = prior",
         [("0.02", lambda df, p: E(df, _one_indicator(p, "ramp_token_spend_anthropic", drift=0.02)))]),
        ("Ramp included", "indicator", "role=model with 2 snapshots",
         [("drop Ramp", lambda df, p: E(df[df.indicator != "ramp_token_spend_anthropic"], p))]),
        ("PyPI break handling", "indicator", "break at 2026-08-25",
         [("no break", lambda df, p: E(df, _one_indicator(p, "pypi_anthropic", breaks=[])))]),
    ]


def _shift_breaks(params, lag):
    """Break dates live in the indicator's calendar and must move with a lag shift."""
    p = copy.deepcopy(params)
    for spec in p["indicators"].values():
        spec["breaks"] = [f"{pd.Timestamp(d) - pd.Timedelta(days=lag):%Y-%m-%d}" for d in spec.get("breaks", [])]
    return p


def _with_lag(df, p, definition, days):
    with patched_item(config.DEFINITION_LAG_DAYS, definition, float(days)):
        return evaluate(df, p)


def _with_attr(df, p, obj, attr, value):
    with patched(obj, attr, value):
        return evaluate(df, p)


def _one_indicator(params, name, **kw):
    p = copy.deepcopy(params)
    p["indicators"][name].update(kw)
    return p


def run_audit(df=None, params=None, verbose=True) -> tuple[dict, pd.DataFrame]:
    df = data.load_all() if df is None else df
    params = calibrate.load_params() if params is None else params
    global TARGETS
    TARGETS = backtest.run_backtest(df, params).target.tolist()
    base = evaluate(df, params)
    if verbose:
        print("baseline", {k: round(v, 3) for k, v in base.items()})
    rows = []
    for name, cat, current, variants in checks():
        for label, runner in variants:
            t0 = time.time()
            r = runner(df, params)
            rows.append({"choice": name, "category": cat, "current": current, "alternative": label,
                         "d_oct31": r["oct31"] - base["oct31"], "d_q4": r["q4"] - base["q4"],
                         "d_mae": r["mae"] - base["mae"], "d_cov80": r["cov80"] - base["cov80"],
                         "oct31": r["oct31"], "q4": r["q4"], "mae": r["mae"], "cov80": r["cov80"]})
            if verbose:
                print(f"  {name:42s} {label:26s} oct31 {r['oct31']:6.1f} ({r['oct31'] - base['oct31']:+5.1f})  "
                      f"q4 {r['q4']:5.1f} ({r['q4'] - base['q4']:+5.1f})  mae {r['mae']:.3f}  cov80 {r['cov80']:.2f}  "
                      f"[{time.time() - t0:.0f}s]")
    tab = pd.DataFrame(rows)
    half_oct = (base["oct31_q90"] - base["oct31_q10"]) / 2
    half_q4 = (base["q4_q90"] - base["q4_q10"]) / 2
    g = tab.groupby("choice", sort=False)
    tab["material"] = tab.choice.map(
        (g.d_oct31.apply(lambda s: s.abs().max()) > 0.5 * half_oct)
        | (g.d_q4.apply(lambda s: s.abs().max()) > 0.5 * half_q4)
        | (g.d_mae.apply(lambda s: s.abs().max()) > 0.01))
    base["half_oct"], base["half_q4"] = half_oct, half_q4
    return base, tab


def markdown(base, tab, asof=ASOF) -> str:
    out = [f"# Audit of hand-set choices (as of {asof})\n",
           f"Baseline: Oct-31 official run-rate {base['oct31']:.1f} [{base['oct31_q10']:.1f}, {base['oct31_q90']:.1f}], "
           f"Q4 revenue {base['q4']:.1f} [{base['q4_q10']:.1f}, {base['q4_q90']:.1f}], backtest MAE {base['mae']:.3f}, "
           f"80% coverage {base['cov80']:.0%}.\n",
           f"**Material** = an alternative moves Oct-31 by > {0.5 * base['half_oct']:.1f}, Q4 by > {0.5 * base['half_q4']:.1f} "
           "(half the 80% half-width), or backtest MAE by > 0.01.\n",
           "| Choice | Category | Current | Alternative | ΔOct-31 | ΔQ4 | ΔMAE | Δcov80 | Material |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        out.append(f"| {r.choice} | {r.category} | {r.current} | {r.alternative} | {r.d_oct31:+.1f} | {r.d_q4:+.1f} | "
                   f"{r.d_mae:+.3f} | {r.d_cov80:+.0%} | {'**yes**' if r.material else ''} |")
    return "\n".join(out) + "\n"
