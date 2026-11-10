"""Estimate the model's noise parameters from the data instead of picking them by hand.

fit_process_noise: maximum a posteriori estimate of (sigma_l, sigma_g) from the online
    filter's prediction-error likelihood, with weak log-normal priors around the
    defaults (there are only ~20 observations).
calibrate_sources: compare each tier's past claims with the audited path implied by
    GAAP revenue alone, and estimate each tier's bias and noise sd. Small tiers are
    shrunk toward the defaults.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm

from . import config, data
from .model import Model


def load_params(path: str = config.PARAMS_PATH, with_indicators: bool = True) -> dict:
    """params.json, plus indicator specs from the catalog (fitted overrides win)."""
    p = {}
    if os.path.exists(path):
        with open(path) as f:
            p = json.load(f)
    if with_indicators and os.path.exists(config.CATALOG_PATH):
        from . import altdata
        p["indicators"] = altdata.indicator_params(overrides=p.get("indicators"))
    return p


def save_params(params: dict, path: str = config.PARAMS_PATH) -> None:
    keep = {k: v for k, v in params.items() if k != "indicators"}
    fitted = {k: {kk: vv for kk, vv in v.items() if kk in ("beta", "drift", "obs_sd")}
              for k, v in params.get("indicators", {}).items() if v.get("fitted")}
    if fitted:
        keep["indicators"] = fitted
    with open(path, "w") as f:
        json.dump(keep, f, indent=2, sort_keys=True)
        f.write("\n")


# --------------------------------------------------------------------------- process noise
def fit_process_noise(df: pd.DataFrame, base: dict | None = None, prior_sd_log: float = 1.0):
    """MAP estimate of sigma_l, sigma_g, sigma_b and the growth reversion rate kappa.

    The objective is the online filter's prediction-error likelihood plus weak
    log-normal priors. kappa's prior is centred on 0.1/month (half-life ~7 months)
    with a log sd of 1.5, which is wide enough that the data decide.
    """
    base = dict(base or {})
    sel = data.select(df)
    d0 = config.DEFAULT_PARAMS
    names = ["sigma_l", "sigma_g", "sigma_b", "growth_reversion"]
    centers = np.log([d0["sigma_l"], d0["sigma_g"], 0.002, 0.1])
    prior_sds = np.array([prior_sd_log, prior_sd_log, 1.5, 1.5])

    def negpost(theta):
        p = dict(base, **{k: float(v) for k, v in zip(names, np.exp(theta))})
        ll = Model(p).run(sel, smooth=False).loglik
        lp = float(np.sum(norm.logpdf(theta, centers, prior_sds)))
        return -(ll + lp)

    x0 = np.array([math.log(base.get(k) or math.exp(c)) for k, c in zip(names, centers)])
    opt = minimize(negpost, x0, method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400})
    h = 1e-2
    sds = []
    for i in range(len(names)):
        e = np.zeros(len(names)); e[i] = h
        d2 = (negpost(opt.x + e) - 2 * opt.fun + negpost(opt.x - e)) / h ** 2
        sds.append(1 / math.sqrt(d2) if d2 > 0 else float("nan"))
    out = {k: float(v) for k, v in zip(names, np.exp(opt.x))}
    out.update({"log_sd": dict(zip(names, sds)), "neg_log_post": float(opt.fun),
                "converged": bool(opt.success)})
    return out


def fit_indicators(df: pd.DataFrame, params: dict, names=None, verbose=True) -> dict:
    """MAP estimate of each model indicator's elasticity beta and intercept drift.

    The objective is the same online-filter likelihood over all observations (claims and
    indicators), with priors from the catalog: beta ~ N(beta_prior, beta_sd) and
    log(drift) ~ N(log(drift_prior), 1). Indicators are fit one at a time, the others
    held at their current values.
    """
    sel = data.select(df)
    params = json.loads(json.dumps(params))
    specs = params["indicators"]
    names = names or [k for k, v in specs.items() if v.get("role") == "model"
                      and k in set(sel.indicator)]
    out = {}
    for name in names:
        prior = dict(specs[name])

        def negpost(theta, name=name, prior=prior):
            beta, logd, logs = theta
            specs[name] = dict(prior, beta=float(beta), drift=float(math.exp(logd)), obs_sd=float(math.exp(logs)))
            ll = Model(params).run(sel, smooth=False).loglik
            lp = norm.logpdf(beta, prior["beta"], prior["beta_sd"]) + \
                norm.logpdf(logd, math.log(prior["drift"]), 1.0) + \
                norm.logpdf(logs, math.log(prior["obs_sd"]), 0.7)
            return -(ll + lp)

        x0 = [prior["beta"], math.log(prior["drift"]), math.log(prior["obs_sd"])]
        opt = minimize(negpost, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 200})
        beta, drift, sd = float(opt.x[0]), float(math.exp(opt.x[1])), float(math.exp(opt.x[2]))
        specs[name] = dict(prior, beta=beta, drift=drift, obs_sd=sd, fitted=True)
        out[name] = {"beta": beta, "drift": drift, "obs_sd": sd, "prior_beta": prior["beta"],
                     "prior_drift": prior["drift"], "prior_obs_sd": prior["obs_sd"], "neg_log_post": float(opt.fun)}
        if verbose:
            print(f"  {name}: beta {prior['beta']:.2f} -> {beta:.3f}, drift {prior['drift']:.4f} -> {drift:.4f}, "
                  f"obs_sd {prior['obs_sd']:.3f} -> {sd:.3f}")
    return out, params


# --------------------------------------------------------------------------- sources
def gaap_anchor(df: pd.DataFrame, params: dict | None = None):
    """Smoothed path using audited revenue only: the reference truth for scoring sources."""
    sel = data.select(df)
    gaap = sel[sel.tier == "gaap"]
    return Model(params).run(gaap)


def source_residuals(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Each non-GAAP claim minus the GAAP-implied value at its effective date."""
    res = gaap_anchor(df, params)
    sel = data.select(df)
    rows = []
    D = config.DAYS_PER_MONTH
    claims = sel[(sel.tier != "gaap") & sel.metric.isin(["run_rate", "revenue"]) & (sel.segment == "total")]
    for r in claims.itertuples():
        if r.metric == "run_rate":
            try:
                m, P = res.state(r.t_eff)
            except IndexError:
                continue
            truth, tsd = float(m[0]), math.sqrt(P[0, 0])
            date_sd = abs(m[1]) / D * r.date_sd_days
            e = math.log(r.value) - truth
            rows.append({"obs_id": r.obs_id, "tier": r.tier, "metric": r.metric, "kind": r.kind,
                         "claim": r.value, "gaap_implied": math.exp(truth), "resid": e,
                         "resid_sd": math.sqrt(tsd ** 2 + date_sd ** 2)})
        else:
            key = f"{r.period_start:%Y-%m-%d}..{r.period_end:%Y-%m-%d}"
            try:
                pr = res.period_revenue(key)
            except KeyError:
                continue
            e = math.log(r.value / pr["mean"])
            rows.append({"obs_id": r.obs_id, "tier": r.tier, "metric": r.metric, "kind": r.kind,
                         "claim": r.value, "gaap_implied": pr["mean"], "resid": e,
                         "resid_sd": pr["sd"] / pr["mean"]})
    return pd.DataFrame(rows)


def _fit_tier(g: pd.DataFrame, sd0: float, bias_prior_sd=0.15, sd_prior_log=0.5):
    def nlp(theta):
        beta, logs = theta
        s = math.exp(logs)
        ll = 0.0
        for r in g.itertuples():
            # resid = log(claim) - truth. Point: resid ~ N(beta, tot^2).
            tot = math.sqrt(s * s + r.resid_sd ** 2)
            if r.kind == "point":
                ll += norm.logpdf(r.resid, beta, tot)
            elif r.kind == "lower":   # reported value >= claim  <=>  resid <= beta + noise
                ll += float(log_ndtr((beta - r.resid) / tot))
            else:                     # reported value <= claim  <=>  resid >= beta + noise
                ll += float(log_ndtr((r.resid - beta) / tot))
        lp = norm.logpdf(beta, 0, bias_prior_sd) + norm.logpdf(logs, math.log(sd0), sd_prior_log)
        return -(ll + lp)

    opt = minimize(nlp, [0.0, math.log(sd0)], method="Nelder-Mead")
    return float(opt.x[0]), float(math.exp(opt.x[1]))


def calibrate_sources(df: pd.DataFrame, params: dict | None = None):
    """Per tier: n, mean residual, fitted bias and sd (shrunk toward the defaults).

    The official tier's bias is the gap between the reported run-rate and GAAP
    revenue. Inside the filter that gap is the state b, so the bias written back
    for other tiers is relative to the official tier.
    """
    resid = source_residuals(df, params)
    rows = []
    for (tier, metric), g in resid.groupby(["tier", "metric"]):
        sd0 = config.TIERS[tier]["log_sd"]
        beta, sd = _fit_tier(g, sd0)
        pts = g[g.kind == "point"]
        rows.append({"tier": tier, "metric": metric, "n": len(g), "n_point": len(pts),
                     "mean_resid": float(pts.resid.mean()) if len(pts) else float("nan"),
                     "rmse": float(np.sqrt((pts.resid ** 2).mean())) if len(pts) else float("nan"),
                     "bias_fit": beta, "sd_fit": sd, "sd_default": sd0})
    return resid, pd.DataFrame(rows)


def tier_overrides(table: pd.DataFrame, min_points: int = 3) -> dict:
    """Turn calibrate_sources output into params['tiers'] overrides (run-rate claims only).

    A tier is overridden only once it has at least `min_points` point claims to
    compare. Below that the fit mostly echoes the prior, so the defaults stay.
    """
    rr = table[(table.metric == "run_rate") & (table.n_point >= min_points)].set_index("tier")
    if "official" not in rr.index:
        return {}
    b_off = rr.loc["official", "bias_fit"]
    out = {}
    for tier, r in rr.iterrows():
        out[tier] = {"log_sd": round(float(r.sd_fit), 4),
                     "bias": 0.0 if tier == "official" else round(float(r.bias_fit - b_off), 4)}
    return out
