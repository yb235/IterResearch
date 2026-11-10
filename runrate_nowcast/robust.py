"""Specification uncertainty in the indicator layer: estimate what can be estimated, average the rest.

The audit (audit.py) found that the nowcast is sensitive to three indicator choices the
backtest cannot tell apart: the elasticity beta, the intercept drift, and the
lead/lag between an indicator and revenue. A single point choice of these understates
uncertainty. Here they are integrated over a grid instead:

    weight(spec) ∝ prior(spec) * p(all observations | spec)

p(...) is the online filter's prediction-error likelihood, so specs are compared on how
well they predicted each observation before seeing it. Priors:
    beta ~ N(1, 0.25)   log(drift multiplier) ~ N(0, 1)   lag ~ uniform on the grid
The nowcast is then the weighted mixture of the per-spec nowcasts.

Two non-indicator choices the audit flagged are integrated the same way, with flat priors
over their grids: the noise of third-party estimates (likelihood is flat), and the lag
of Anthropic's run-rate definition (14 days by definition; 7 fits slightly better).

Breaks are detected by a rule, not by eye: a counting change is a day when the
Anthropic series and the same source's competitor series (openai) both step by more
than 25% between the 7-day means before and after.
"""
from __future__ import annotations

import contextlib
import copy
import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import altdata, config, data, report
from .model import Model

SDK = ("npm_anthropic_sdk", "pypi_anthropic")
LAGS = (-14, -7, 0, 7, 14)              # + = indicator reflects revenue `lag` days earlier
BETAS = (0.85, 1.0, 1.15)
DRIFT_MULTS = (0.5, 0.7, 1.0, 1.4)
THIRD_PARTY_SD = (0.08, 0.15, 0.30)     # not identified by the data (flat likelihood)
DEF_LAGS = (7, 14)                      # t28x13 lag: 14 from the definition, data mildly prefer 7-10


def detect_breaks(raw: pd.DataFrame | None = None, pairs=(("pypi_anthropic", "pypi_openai"),
                                                       ("npm_anthropic_sdk", "npm_openai")),
                  threshold: float = 0.25, window: int = 7, persist: int = 28) -> pd.DataFrame:
    """Days where an Anthropic series and its same-source competitor step together, and stay stepped.

    Step: 7-day mean after vs before moves by more than `threshold` in both series, in the
    same direction. Persistence: the 28-day means (skipping the week around the day) also
    differ by more than threshold/2 in both. Holiday dips that recover fail this test.
    """
    raw = altdata.load_raw() if raw is None else raw
    out = []
    for a, b in pairs:
        s, lp = {}, {}
        for name in (a, b):
            x = raw[raw.indicator == name].set_index("date").value.sort_index()
            x = x[x > 0]
            # Calendar rule: the year-end holiday (Dec 20 - Jan 6) depresses every series and
            # then recovers. Drop it so its rebound is not read as a step.
            md = x.index.month * 100 + x.index.day
            x = x[~((md >= 1220) | (md <= 106))].asfreq("D")
            before = x.rolling(window, min_periods=window - 2).mean()
            after = x[::-1].rolling(window, min_periods=window - 2).mean()[::-1]
            s[name] = np.log(after / before.shift(1))
            long_before = x.rolling(persist, min_periods=persist // 2).mean().shift(window + 1)
            long_after = x[::-1].rolling(persist, min_periods=persist // 2).mean()[::-1].shift(-window)
            lp[name] = np.log(long_after / long_before)
        both = pd.concat(s, axis=1).dropna()
        longs = pd.concat(lp, axis=1).reindex(both.index)
        step = (both[a].abs() > math.log(1 + threshold)) & (both[b].abs() > math.log(1 + threshold)) \
            & (np.sign(both[a]) == np.sign(both[b]))
        lasting = (longs[a] * np.sign(both[a]) > math.log(1 + threshold / 2)) \
            & (longs[b] * np.sign(both[b]) > math.log(1 + threshold / 2))
        hit = both[step & lasting]
        # keep the strongest day in each cluster
        last = None
        for d, r in hit.iterrows():
            if last is not None and (d - last).days <= window:
                continue
            seg = hit[(hit.index >= d) & (hit.index <= d + pd.Timedelta(days=window))]
            best = seg[a].abs().idxmax()
            out.append({"source_pair": f"{a} / {b}", "date": best, "step_anthropic": float(np.expm1(seg.loc[best, a])),
                        "step_competitor": float(np.expm1(seg.loc[best, b]))})
            last = d
    return pd.DataFrame(out)


def _shift_indicators(df, lag_days):
    d = df.copy()
    m = d.metric == "indicator"
    d.loc[m, "ref_date"] = d.loc[m, "ref_date"] - pd.Timedelta(days=lag_days)
    return d


def _spec_params(params, beta, drift_mult, lag=0):
    p = copy.deepcopy(params)
    for name in SDK:
        if name in p["indicators"]:
            p["indicators"][name]["beta"] = beta
            p["indicators"][name]["drift"] = params["indicators"][name]["drift"] * drift_mult
    # Breaks are dates in the indicator's own calendar, so they move with the lag.
    for spec in p["indicators"].values():
        spec["breaks"] = [f"{pd.Timestamp(d) - pd.Timedelta(days=lag):%Y-%m-%d}" for d in spec.get("breaks", [])]
    return p


@contextlib.contextmanager
def definition_lag(days):
    old = config.DEFINITION_LAG_DAYS["t28x13"]
    config.DEFINITION_LAG_DAYS["t28x13"] = float(days)
    try:
        yield
    finally:
        config.DEFINITION_LAG_DAYS["t28x13"] = old


def _full_params(params, beta, dm, lag, tp_sd):
    p = _spec_params(params, beta, dm, lag)
    p.setdefault("tiers", {}).setdefault("third_party", {})["log_sd"] = tp_sd
    return p


def spec_grid(df, params, asof, lags=LAGS, betas=BETAS, drift_mults=DRIFT_MULTS,
              tp_sds=THIRD_PARTY_SD, def_lags=DEF_LAGS, verbose=False) -> pd.DataFrame:
    rows = []
    for dlag in def_lags:
        with definition_lag(dlag):
            for lag in lags:
                sel = data.select(_shift_indicators(df, lag), vintage=asof)
                for beta, dm, tp in itertools.product(betas, drift_mults, tp_sds):
                    ll = Model(_full_params(params, beta, dm, lag, tp)).run(sel, smooth=False).loglik
                    lp = norm.logpdf(beta, 1.0, 0.25) + norm.logpdf(math.log(dm), 0.0, 1.0)
                    rows.append({"def_lag": dlag, "lag": lag, "beta": beta, "drift_mult": dm, "tp_sd": tp,
                                 "loglik": ll, "logprior": lp})
        if verbose:
            print(f"  definition lag {dlag}: best loglik {max(r['loglik'] for r in rows if r['def_lag'] == dlag):.2f}")
    g = pd.DataFrame(rows)
    lw = g.loglik + g.logprior
    g["weight"] = np.exp(lw - lw.max())
    g["weight"] /= g.weight.sum()
    return g.sort_values("weight", ascending=False).reset_index(drop=True)


def averaged_nowcast(df, params, asof, grid: pd.DataFrame, mass: float = 0.99, n_draws: int = 20000,
                     seed: int = 0, dates=("2026-10-31", "2026-11-10"), period="2026-10-01..2026-12-31"):
    """Mixture nowcast over the specs that carry `mass` of the posterior weight."""
    keep = grid[grid.weight.cumsum().shift(fill_value=0) < mass].copy()
    keep["weight"] /= keep.weight.sum()
    comps = []
    for r in keep.itertuples():
        p = _full_params(params, r.beta, r.drift_mult, r.lag, r.tp_sd)
        with definition_lag(r.def_lag):
            _, res, _ = report.run_nowcast(_shift_indicators(df, r.lag), p, asof)
            c = {"def_lag": r.def_lag, "lag": r.lag, "beta": r.beta, "drift_mult": r.drift_mult,
                 "tp_sd": r.tp_sd, "weight": r.weight}
            for d in dates:
                c[f"rr_{d}"] = res.log_runrate(d, "t28x13")
            q = res.period_revenue(period)
            c["rev"] = (q["mean"], q["sd"])
        comps.append(c)
    rng = np.random.default_rng(seed)
    w = np.array([c["weight"] for c in comps])
    idx = rng.choice(len(comps), size=n_draws, p=w / w.sum())
    summary = {}
    for d in dates:
        mu = np.array([comps[i][f"rr_{d}"][0] for i in idx])
        sd = np.array([comps[i][f"rr_{d}"][1] for i in idx])
        x = np.exp(rng.normal(mu, sd))
        summary[f"runrate_{d}"] = {"median": float(np.median(x)), "q10": float(np.quantile(x, 0.1)),
                                   "q90": float(np.quantile(x, 0.9))}
    m = np.array([comps[i]["rev"][0] for i in idx])
    s = np.array([comps[i]["rev"][1] for i in idx])
    s2 = np.log1p((s / m) ** 2)
    x = np.exp(rng.normal(np.log(m) - s2 / 2, np.sqrt(s2)))
    summary[f"revenue_{period}"] = {"mean": float(x.mean()), "q10": float(np.quantile(x, 0.1)),
                                    "q90": float(np.quantile(x, 0.9))}
    return pd.DataFrame(comps), summary


def lag_posterior(grid: pd.DataFrame, col: str = "lag") -> pd.Series:
    return grid.groupby(col).weight.sum().sort_index()
