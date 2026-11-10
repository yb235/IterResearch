"""Nowcast report, alerts and claim plausibility."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import backtest, config, data
from .model import Model, Z80, observation_predictive, period_key


def default_queries(asof) -> list[tuple]:
    """Months from 4 months before `asof` to year end, plus the current quarter and the year."""
    asof = pd.Timestamp(asof)
    first = (asof - pd.DateOffset(months=4)).to_period("M")
    last = pd.Timestamp(year=asof.year, month=12, day=31).to_period("M")
    qs = [(p.start_time.normalize(), p.end_time.normalize()) for p in pd.period_range(first, last, freq="M")]
    q = asof.to_period("Q")
    qs.append((q.start_time.normalize(), q.end_time.normalize()))
    y = asof.to_period("Y")
    qs.append((y.start_time.normalize(), y.end_time.normalize()))
    return qs


def run_nowcast(df: pd.DataFrame, params: dict | None, asof) -> tuple[Model, object, pd.DataFrame]:
    asof = pd.Timestamp(asof)
    sel = data.select(df, vintage=asof)
    model = Model(params)
    queries = default_queries(asof)
    end = max(asof, max(e for _, e in queries))
    res = model.run(sel, end=end, queries=queries)
    return model, res, sel


def claim_probability(df, params, row, asof) -> float | None:
    """Probability of a bound claim under a model that never saw that claim's origin."""
    sel = data.select(df, vintage=asof)
    sel = sel[sel.origin_id != row.origin_id]
    model = Model(params)
    res = model.run(sel, end=max(pd.Timestamp(asof), pd.Timestamp(row.t_eff)))
    mu, sd = observation_predictive(res, row, model.p["tiers"])
    y = math.log(row.value) if row.metric == "run_rate" else row.value
    if row.kind == "lower":
        return float(norm.sf((y - mu) / sd))
    if row.kind == "upper":
        return float(norm.cdf((y - mu) / sd))
    return None


def alerts(df, params, since, asof, z_thresh=config.ALERT_Z) -> pd.DataFrame:
    """Claims published in [since, asof], scored against the forecast made the day before."""
    sel = data.select(df, vintage=asof)
    new = sel[sel.published >= pd.Timestamp(since)]
    rows = []
    for r in new.itertuples():
        out = backtest.prepublication_forecast(df, r, params)
        if out is None:
            continue
        mu, sd, n_ev, _ = out
        y = math.log(r.value) if r.metric == "run_rate" else float(r.value)
        z = (y - mu) / sd
        if r.kind == "point":
            flag = abs(z) > z_thresh
            p = float(2 * norm.sf(abs(z)))
        elif r.kind == "lower":
            p = float(norm.sf(z))             # prob. the floor is true
            flag = p < norm.sf(z_thresh)
        else:
            p = float(norm.cdf(z))
            flag = p < norm.sf(z_thresh)
        fmt = (lambda v: math.exp(v)) if r.metric == "run_rate" else (lambda v: v)
        rows.append({"obs_id": r.obs_id, "published": r.published.date(), "tier": r.tier,
                     "kind": r.kind, "claim": r.value, "expected": fmt(mu),
                     "q10": fmt(mu - Z80 * sd), "q90": fmt(mu + Z80 * sd), "z": z, "p": p,
                     "ALERT": bool(flag)})
    return pd.DataFrame(rows)


def fit_table(res, sel, tiers) -> pd.DataFrame:
    rows = []
    for r in sel.itertuples():
        try:
            mu, sd = observation_predictive(res, r, tiers)
        except (IndexError, KeyError):
            continue
        if r.metric == "run_rate":
            fitted, y = math.exp(mu), math.log(r.value)
        else:
            fitted, y = mu, r.value
        rows.append({"obs_id": r.obs_id, "tier": r.tier, "kind": r.kind, "claim": r.value,
                     "fitted": fitted, "z_insample": (y - mu) / sd})
    return pd.DataFrame(rows)


def _fmt_rr(d):
    return f"{d['median']:.1f} [{d['q10']:.1f}, {d['q90']:.1f}]"


def markdown(df, params, asof) -> str:
    asof = pd.Timestamp(asof)
    model, res, sel = run_nowcast(df, params, asof)
    tiers = model.p["tiers"]
    out = []
    out.append(f"# Anthropic run-rate nowcast as of {asof:%Y-%m-%d}\n")
    out.append("Units: $B. Intervals are 80% (q10 to q90). The **GAAP basis** is the instantaneous annualized "
               "revenue that matches the S-1 quarterly revenue. The **official basis** applies Anthropic's run-rate "
               "definition (trailing 28 days ×13, a 14-day lag) plus the estimated gap b, so it is the number "
               "the company would announce on that date.\n")
    p = model.p
    out.append(f"Parameters: sigma_l={p['sigma_l']:.4f}, sigma_g={p['sigma_g']:.4f}, sigma_b={p['sigma_b']:.4f}, "
               f"growth reversion kappa={p.get('growth_reversion', 0):.3f}/month toward {p.get('growth_longrun', 0):.2f}; "
               f"{len(sel)} observations used (after dedup); smoother converged in {res.iterations} passes.\n")

    out.append("## Run-rate\n")
    out.append("| Date | GAAP basis (instantaneous) | Official basis (t28×13 + b) | Growth (log/month) |")
    out.append("|---|---|---|---|")
    last_q = (asof.to_period("Q") - 1).end_time.normalize()
    dates = sorted({last_q - pd.DateOffset(months=3) + pd.offsets.MonthEnd(0), last_q,
                    (asof - pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0), asof,
                    asof.to_period("Q").end_time.normalize()})
    for d in dates:
        g, gs = res.growth(d)
        tag = " (forecast)" if d > asof else ""
        out.append(f"| {d:%Y-%m-%d}{tag} | {_fmt_rr(res.runrate(d))} | {_fmt_rr(res.runrate(d, 't28x13'))} | "
                   f"{g:+.3f} ± {gs:.3f} |")
    b, bs = res.gap(asof)
    out.append(f"\nGap b between official run-rate and GAAP at {asof:%Y-%m-%d}: {b:+.3f} ± {bs:.3f} "
               f"(official ≈ GAAP basis × {math.exp(b):.3f})\n")

    out.append("## Revenue by period\n")
    out.append("| Period | Mean | 80% interval | Status |")
    out.append("|---|---|---|---|")
    for pr in res.periods:
        r = res.period_revenue(pr.key)
        obs = sel[(sel.metric == "revenue") & (sel.period_start == pr.start) & (sel.period_end == pr.end)
                  & (sel.tier == "gaap")]
        status = "GAAP disclosed" if len(obs) else ("estimate" if pr.end <= asof else "forecast")
        out.append(f"| {pr.key} | {r['mean']:.2f} | [{r['q10']:.2f}, {r['q90']:.2f}] | {status} |")

    out.append("\n## Bound claims: how likely is each one true? (evaluated by a model that never saw it)\n")
    out.append("| Claim | Source tier | Probability true |")
    out.append("|---|---|---|")
    for r in sel[sel.kind != "point"].itertuples():
        pr_ = claim_probability(df, params, r, asof)
        word = ">" if r.kind == "lower" else "<"
        out.append(f"| {r.obs_id}: {r.metric} {word} {r.value:g} @ {r.ref_date:%Y-%m-%d} | {r.tier} | {pr_:.0%} |")

    out.append("\n## In-sample fit by observation (smoothed; |z|>2 means this claim conflicts with everything else)\n")
    ft = fit_table(res, sel, tiers)
    out.append("| obs_id | tier | kind | claim | fitted | z |")
    out.append("|---|---|---|---|---|---|")
    for r in ft.itertuples():
        out.append(f"| {r.obs_id} | {r.tier} | {r.kind} | {r.claim:g} | {r.fitted:.2f} | {r.z_insample:+.2f} |")

    removed = data.dedup_report(df)
    if len(removed):
        out.append("\n## Rows not used\n")
        out.append("| obs_id | reason |")
        out.append("|---|---|")
        for r in removed.itertuples():
            out.append(f"| {r.obs_id} | {r.reason} |")
    return "\n".join(out) + "\n"
