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
    res = model.run(sel, end=end, queries=queries, segments=list(config.SEGMENTS))
    res.indicator_specs = model.p["indicators"]
    return model, res, sel


def claim_probability(df, params, row, asof) -> float | None:
    """Probability of a bound claim under a model that never saw that claim's origin."""
    sel = data.select(df, vintage=asof)
    sel = sel[sel.origin_id != row.origin_id]
    model = Model(params)
    seg = getattr(row, "segment", "total")
    res = model.run(sel, end=max(pd.Timestamp(asof), pd.Timestamp(row.t_eff)),
                    segments=[seg] if seg != "total" else None)
    mu, sd = observation_predictive(res, row, model)
    y = row.value if row.metric == "revenue" else math.log(row.value)
    if row.kind == "lower":
        return float(norm.sf((y - mu) / sd))
    if row.kind == "upper":
        return float(norm.cdf((y - mu) / sd))
    return None


def alerts(df, params, since, asof, z_thresh=config.ALERT_Z) -> pd.DataFrame:
    """Claims published in [since, asof], scored against the forecast made the day before."""
    sel = data.select(df, vintage=asof)
    new = sel[sel.published >= pd.Timestamp(since)]
    ind = new[new.metric == "indicator"]
    latest = ind.sort_values("ref_date").groupby("indicator").tail(1)
    new = pd.concat([new[new.metric != "indicator"], latest])
    rows = []
    for r in new.itertuples():
        out = backtest.prepublication_forecast(df, r, params)
        if out is None:
            continue
        mu, sd, n_ev, _ = out
        y = float(r.value) if r.metric == "revenue" else math.log(r.value)
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
        fmt = (lambda v: v) if r.metric == "revenue" else (lambda v: math.exp(v))
        rows.append({"obs_id": r.obs_id, "published": r.published.date(), "tier": r.tier,
                     "kind": r.kind, "claim": r.value, "expected": fmt(mu),
                     "q10": fmt(mu - Z80 * sd), "q90": fmt(mu + Z80 * sd), "z": z, "p": p,
                     "ALERT": bool(flag)})
    return pd.DataFrame(rows)


def fit_table(res, sel, model) -> pd.DataFrame:
    rows = []
    for r in sel[sel.metric != "indicator"].itertuples():
        try:
            mu, sd = observation_predictive(res, r, model)
        except (IndexError, KeyError, ValueError):
            continue
        if r.metric == "revenue":
            fitted, y = mu, r.value
        else:
            fitted, y = math.exp(mu), math.log(r.value)
        z = (y - mu) / sd
        conflict = abs(z) > 2 if r.kind == "point" else (z > 2 if r.kind == "lower" else z < -2)
        rows.append({"obs_id": r.obs_id, "tier": r.tier, "kind": r.kind, "claim": r.value,
                     "segment": getattr(r, "segment", "total"), "fitted": fitted, "z_insample": z,
                     "conflict": conflict})
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
               f"{len(sel)} observations used after dedup ({int((sel.metric == 'indicator').sum())} indicator-weeks); "
               f"smoother converged in {res.iterations} passes.\n")

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

    out.append("## Segments (official basis)\n")
    out.append("Only consumption vs subscription is audited (S-1, H1 2026). Cloud channel and Claude Code are overlays "
               "estimated from disclosures and estimates; intervals are wide where data are thin.\n")
    out.append(f"| Segment | Share {last_q:%Y-%m-%d} | Share {asof:%Y-%m-%d} | Run-rate {last_q:%Y-%m-%d} | Run-rate {asof:%Y-%m-%d} |")
    out.append("|---|---|---|---|---|")
    for sname in res.layout.segments:
        sh0, sh1 = res.share(last_q, sname), res.share(asof, sname)
        r0, r1 = res.runrate(last_q, "t28x13", segment=sname), res.runrate(asof, "t28x13", segment=sname)
        out.append(f"| {config.SEGMENTS[sname]['label']} | {sh0['median']:.1%} [{sh0['q10']:.0%}, {sh0['q90']:.0%}] | "
                   f"{sh1['median']:.1%} [{sh1['q10']:.0%}, {sh1['q90']:.0%}] | {_fmt_rr(r0)} | {_fmt_rr(r1)} |")

    if res.layout.indicators:
        out.append("\n## Leading indicators\n")
        out.append("Each indicator: log(I) = a + beta·log(segment revenue) + noise, with a drifting intercept a. "
                   "\"Read as revenue\" = 4-week change in the indicator / beta. \"Model revenue\" = the filter's "
                   "4-week change in the indicator's segment revenue. The gap is what the filter attributed to drift.\n")
        out.append("| Indicator | Segment | beta | Latest week | 4-week change | Read as revenue | Model revenue (same weeks) |")
        out.append("|---|---|---|---|---|---|---|")
        for name in res.layout.indicators:
            spec = model.p["indicators"][name]
            rows_i = sel[(sel.metric == "indicator") & (sel.indicator == name)].sort_values("ref_date")
            last = rows_i.iloc[-1]
            prev = rows_i[rows_i.ref_date <= last.ref_date - pd.Timedelta(days=27)]
            if len(prev):
                prev = prev.iloc[-1]
                dlog = math.log(last.value / prev.value)
                segn = spec.get("segment", "total")
                l1 = res.log_runrate(last.ref_date, segment=segn)[0]
                l0 = res.log_runrate(prev.ref_date, segment=segn)[0]
                out.append(f"| {name} | {segn} | {spec['beta']:.2f} | {last.ref_date:%Y-%m-%d} | {dlog:+.1%} | "
                           f"{dlog / spec['beta']:+.1%} | {l1 - l0:+.1%} |")

    fc = df[(df.get("is_forecast", 0) == 1) & (df.include == 1) & (df.published <= asof)]
    if len(fc):
        out.append("\n## Forecast claims vs the model (not used in the filter; scored for track record)\n")
        out.append("| Claim | Made | Target date | Horizon | Claim value | Model (official basis) | Claim / model |")
        out.append("|---|---|---|---|---|---|---|")
        for r in fc.sort_values("published").itertuples():
            try:
                mo = res.runrate(r.ref_date, "t28x13")
            except IndexError:
                continue
            word = ">" if r.kind == "lower" else ("<" if r.kind == "upper" else "")
            h = (r.ref_date - r.published).days / config.DAYS_PER_MONTH
            out.append(f"| {r.obs_id} | {r.published:%Y-%m-%d} | {r.ref_date:%Y-%m-%d} | {h:.1f} mo | {word}{r.value:g} | "
                       f"{_fmt_rr(mo)} | {r.value / mo['median']:.2f} |")

    out.append("\n## Revenue by period\n")
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

    out.append("\n## In-sample fit by claim (smoothed)\n")
    out.append("z = (claim − fitted)/sd. A point claim conflicts with everything else when |z| > 2. A lower "
               "bound conflicts only when z > 2 (the fit sits below the floor); an upper bound only when z < −2.\n")
    ft = fit_table(res, sel, model)
    out.append("| obs_id | segment | tier | kind | claim | fitted | z | conflict |")
    out.append("|---|---|---|---|---|---|---|---|")
    for r in ft.itertuples():
        out.append(f"| {r.obs_id} | {r.segment} | {r.tier} | {r.kind} | {r.claim:g} | {r.fitted:.2f} | "
                   f"{r.z_insample:+.2f} | {'**yes**' if r.conflict else ''} |")

    removed = data.dedup_report(df)
    if len(removed):
        out.append("\n## Rows not used\n")
        out.append("| obs_id | reason |")
        out.append("|---|---|")
        for r in removed.itertuples():
            out.append(f"| {r.obs_id} | {r.reason} |")
    return "\n".join(out) + "\n"
