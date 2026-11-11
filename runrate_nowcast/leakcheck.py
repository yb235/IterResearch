"""Leak audit: could any forecast see information it would not have had at its vintage?

1. Hard checks, every target and vintage rule: no evidence published after the vintage, no
   filing rows (S-1 and amendments are targets only), not the target's own origin. The same
   guard (backtest.check_no_leak) also runs inside every backtest forecast.
2. Indirect filing leakage: non-filing rows published after the first filing number became
   public, from sources that may have derived them from the filing (third-party, secondary).
   Optionally (sensitivity=True), how much each affected forecast moves without them.
3. Breaks used before they were knowable: for each catalog break, the first day the break
   rule (robust.detect_breaks) flags it using only data up to that day. A target whose
   vintage falls between the break and that day used it with hindsight.
4. Parameters: which fitted values each pipeline uses and whether they come from data public
   at the vintage (static list, kept next to the code it describes).
5. Alternative-data vintages: indicator values recorded with a publication date on or after
   the observation, and snapshots fetched after the fact.
6. Choices made after seeing results: cannot be tested on the same targets; listed.
"""
from __future__ import annotations

import pandas as pd

from . import altdata, backtest, calibrate, config, data, robust

RISKY_TIERS = ("third_party", "secondary")


def _targets(df: pd.DataFrame) -> pd.DataFrame:
    full = data.select(df, filings=True)
    return full[full.tier.isin(backtest.TARGET_TIERS) & (full.kind == "point")
                & full.metric.isin(["run_rate", "revenue"]) & (full.is_forecast == 0) & (full.segment == "total")]


def hard_checks(df: pd.DataFrame, rules=("publication", "period_end")) -> pd.DataFrame:
    rows = []
    for rule in rules:
        for tg in _targets(df).itertuples():
            v = backtest.target_vintage(tg, rule)
            ev = data.select(df, vintage=v)
            ev = ev[ev.origin_id != tg.origin_id]
            status = "ok"
            try:
                backtest.check_no_leak(ev, v, tg)
            except AssertionError as e:
                status = f"LEAK: {e}"
            ind = ev[ev.metric == "indicator"]
            rows.append({"rule": rule, "target": tg.obs_id, "vintage": v.date(), "n_claims": int((ev.metric != "indicator").sum()),
                         "n_indicator_weeks": len(ind), "latest_published": ev.published.max().date() if len(ev) else None,
                         "filing_rows": int(ev.get("filing", pd.Series(0, index=ev.index)).sum()),
                         "indicator_ref_after_published": int((ind.ref_date > ind.published).sum()), "status": status})
    return pd.DataFrame(rows)


def first_filing_public(df: pd.DataFrame) -> pd.Timestamp:
    return pd.Timestamp(df[df.filing == 1].published.min())


def post_filing_rows(df: pd.DataFrame) -> pd.DataFrame:
    t0 = first_filing_public(df)
    d = df[(df.metric != "indicator") & (df.filing != 1) & (df.include == 1) & (df.published >= t0)]
    d = d.assign(risk=[("excluded (filing=2)" if f == 2 else "may derive from filing" if t in RISKY_TIERS else "independent source")
                       for f, t in zip(d.filing, d.tier)])
    return d[["obs_id", "origin_id", "tier", "value", "kind", "published", "risk", "notes"]].sort_values("published")


def post_filing_sensitivity(df: pd.DataFrame, params: dict, rule: str = "publication") -> pd.DataFrame:
    """Re-forecast targets whose vintage is after the first filing without the risky rows."""
    t0 = first_filing_public(df)
    pf = post_filing_rows(df)
    risky = set(pf[pf.risk != "independent source"].obs_id)
    out = []
    base = df[df.metric != "indicator"]
    variants = {"claims_only": base[(base.segment == "total") & (base.metric != "share")], "with_indicators": df}
    for tg in _targets(df).itertuples():
        v = backtest.target_vintage(tg, rule)
        if v < t0:
            continue
        for name, d in variants.items():
            r = {}
            with_all = d.assign(filing=d.filing.where(~d.obs_id.isin(risky), 0))   # put the risky rows back in
            for label, dd in (("all", with_all), ("without_risky", d[~d.obs_id.isin(risky)])):
                f = backtest.prepublication_forecast(dd, tg, params, refit=True, crossing="profile", vintage=v)
                r[label] = None if f is None else f[0]
            if r["all"] is None or r["without_risky"] is None:
                continue
            import math
            conv = (lambda m: math.exp(m)) if tg.metric == "run_rate" else (lambda m: m)
            out.append({"target": tg.obs_id, "vintage": v.date(), "variant": name, "actual": tg.value,
                        "forecast_all": conv(r["all"]), "forecast_without_risky": conv(r["without_risky"])})
    o = pd.DataFrame(out)
    if len(o):
        o["shift_pct"] = 100 * (o.forecast_all / o.forecast_without_risky - 1)
    return o


def break_detectability(max_days: int = 90) -> pd.DataFrame:
    raw = altdata.load_raw()
    cat = altdata.catalog()
    rows = []
    for spec in cat.itertuples():
        for b in [x for x in str(getattr(spec, "breaks", "") or "").split(";") if x and x != "nan"]:
            b = pd.Timestamp(b)
            first = None
            for k in range(0, max_days, 2):
                T = b + pd.Timedelta(days=k)
                found = robust.detect_breaks(raw[raw.date <= T])
                if len(found) and ((pd.to_datetime(found.date) - b).abs() <= pd.Timedelta(days=7)).any():
                    first = T
                    break
            rows.append({"indicator": spec.indicator, "break_date": b.date(), "first_detectable": None if first is None else first.date(),
                         "days_after": None if first is None else (first - b).days})
    return pd.DataFrame(rows).drop_duplicates(subset=["break_date", "first_detectable"])


def break_exposure(df: pd.DataFrame, breaks: pd.DataFrame, rules=("publication", "period_end")) -> pd.DataFrame:
    rows = []
    for rule in rules:
        for tg in _targets(df).itertuples():
            v = backtest.target_vintage(tg, rule)
            for b in breaks.itertuples():
                if b.first_detectable is None:
                    continue
                if pd.Timestamp(b.break_date) <= v < pd.Timestamp(b.first_detectable):
                    rows.append({"rule": rule, "target": tg.obs_id, "vintage": v.date(), "break_date": b.break_date,
                                 "first_detectable": b.first_detectable})
    return pd.DataFrame(rows)


def alt_data_vintages() -> pd.DataFrame:
    raw = altdata.load_raw()
    raw = raw.assign(published=pd.to_datetime(raw.published), date=pd.to_datetime(raw.date))
    g = raw.groupby("indicator")
    return pd.DataFrame({"first_date": g.date.min().dt.date, "last_date": g.date.max().dt.date,
                         "published_min": g.published.min().dt.date, "published_max": g.published.max().dt.date,
                         "rows": g.size(), "published_before_date": g.apply(lambda x: int((x.published < x.date).sum()))})


PARAMETERS = [
    ("State-space backtest, refit=True", "process noise (sigma_l, sigma_g, sigma_b, kappa)", "refit on data public at the vintage", "ok"),
    ("State-space backtest, refit=True", "indicator beta / drift / obs_sd", "reset to catalog priors, refit at the vintage", "ok"),
    ("State-space backtest, refit=True", "crossing position rho", "profiled at the vintage with unfitted parameters (before this audit: with full-sample fits)", "fixed"),
    ("State-space backtest, refit=False", "all of params.json", "full-sample fits", "LEAKS: not used for forecast results"),
    ("Bayesian backtest", "about 20 parameters", "MAP + Laplace at the vintage from priors; warm starts now ordered by vintage", "ok"),
    ("ML backtest", "smoother labels", "process noise refit at the vintage (before this audit: full-sample params.json)", "fixed"),
    ("all", "source noise by tier, growth_longrun, definition lag, catalog priors, holiday rule", "set by hand with the whole history in view", "hindsight, see audit.py sensitivity"),
    ("Price-adjusted indicators", "model mix weights", "uninformed 50/50 (ok); measured GitHub mix up to the date (ok); Nov 2026 mix", "nov2026_mix is HINDSIGHT"),
    ("Price-adjusted indicators", "gamma (price elasticity)", "grid; real-time likelihood weights in pricing.gamma_weights", "picking gamma = 1 after seeing errors is HINDSIGHT"),
]

HINDSIGHT = [
    "Which indicators have role=model (npm, PyPI, Ramp) was decided after the layer backtest on the same targets.",
    "The crossing-position model was added after seeing errors on crossing claims; rho itself is estimated per vintage.",
    "Catalog priors (beta 0.75, drift, obs_sd), tier noise and the holiday window were chosen with the full history visible.",
    "The run-rate definition (t28x13, 14-day lag) became public around May 2026; earlier vintages use it with hindsight (the Bayesian model mixes 7 and 14 days).",
    "Price adjustment with gamma = 1 has the lowest error after the fact, but at every vintage the likelihood then available puts about 70-90% of the weight on gamma = 0. Reporting gamma = 1 as the model would be a hindsight choice.",
    "GitHub commit counts were all measured on 2026-11-11. Counts for earlier weeks include commits that became public later (repositories opened later, late pushes), so the backfilled series is not what a forecaster would have seen at the time.",
    "Each of these makes the backtest look better than a true real-time forecast would have. Only targets that arrive after a forecast is registered (forecasts/registry.jsonl) are free of it.",
]


def run(df: pd.DataFrame | None = None, params: dict | None = None, sensitivity: bool = False) -> str:
    df = data.load_all() if df is None else df
    params = calibrate.load_params() if params is None else params
    hc = hard_checks(df)
    pf = post_filing_rows(df)
    bd = break_detectability()
    be = break_exposure(df, bd)
    av = alt_data_vintages()
    n_leak = int((hc.status != "ok").sum())
    md = ["# Leak audit", "",
          f"Filings are inputs: {config.USE_FILINGS}. First filing number public: {first_filing_public(df).date()}.", "",
          f"## 1. Hard checks ({len(hc)} target-vintage pairs): {'no leaks' if n_leak == 0 else f'{n_leak} LEAKS'}", "",
          hc.to_markdown(index=False), "",
          "## 2. Rows published after the first filing number", "",
          pf.to_markdown(index=False), ""]
    if sensitivity:
        ps = post_filing_sensitivity(df, params)
        md += ["Forecasts with and without the rows that may derive from the filing:", "",
               ps.round(2).to_markdown(index=False) if len(ps) else "(no affected targets)", ""]
    md += ["## 3. Breaks: when they became detectable", "", bd.to_markdown(index=False), "",
           "Targets whose vintage falls between a break and its first detectable day:", "",
           be.to_markdown(index=False) if len(be) else "none", "",
           "## 4. Parameters", "",
           pd.DataFrame(PARAMETERS, columns=["pipeline", "parameters", "estimated from", "status"]).to_markdown(index=False), "",
           "## 5. Alternative-data vintages", "",
           "Values are recorded with the date they were fetched or published. A snapshot fetched today cannot enter a backtest before today; daily download counts are not revised.", "",
           av.to_markdown(), "",
           "## 6. Choices made after seeing results (not testable on the same targets)", ""]
    md += [f"- {h}" for h in HINDSIGHT]
    return "\n".join(md) + "\n"
