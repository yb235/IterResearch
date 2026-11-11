"""Parameter posterior for the state-space model: MAP + Laplace approximation + mixture predictive.

Why: the strict out-of-sample backtest showed intervals ~1.5-1.7x too narrow. One cause
is plugging in point estimates of parameters (noise levels, indicator elasticities,
drifts) as if they were known. Here the parameters get a posterior and predictions
average over it:

    log p(theta | y) = log p(y | theta) + log p(theta)
        p(y | theta): online-filter prediction-error likelihood (all observations)
    theta_hat = argmax (L-BFGS-B), H = -Hessian at theta_hat (finite differences)
    theta ~ N(theta_hat, H^-1)   (Laplace approximation)
    predictive = mixture over K draws of the per-draw smoothed predictive

Parameters (transformed to an unbounded scale) and priors:
    sigma_l, sigma_g, sigma_b, kappa       log;  N(log 0.004, 1), N(log 0.010, 1), N(log 0.002, 1.5), N(log 0.1, 1.5)
    factor drift (dev_activity)            log;  N(log 0.01, 1.5)
    per indicator: beta                    N(catalog beta, catalog beta_sd)
                   drift, obs_sd           log;  N(log catalog value, 1) and N(log catalog value, 0.7)
    third-party noise                      log;  N(log 0.15, 0.7)
    v_q, growth-volatility log-scale       N(0, 0.7^2) per calendar quarter (piecewise stochastic volatility)
Indicators with fewer than MIN_IND_OBS weekly observations at the vintage keep their
catalog values (their parameters would only echo the prior).

Discrete structure (indicator lag, run-rate definition lag) is compared with the
Laplace evidence  log p(y) ~ log p(y|th) + log p(th) + d/2 log 2pi - 1/2 log|H|.
"""
from __future__ import annotations

import copy
import math
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from . import altdata, backtest, calibrate, config, data, robust
from .model import Model, observation_predictive

MIN_IND_OBS = 8
VOL_SD = 0.7


# --------------------------------------------------------------------------- parameter vector
def build_spec(sel: pd.DataFrame, params: dict, last_date) -> list[dict]:
    """Entries: name, kind ('log'|'lin'), prior mean/sd on the transformed scale, setter path."""
    cat = altdata.indicator_params()
    spec = [
        {"name": "sigma_l", "kind": "log", "mu": math.log(0.004), "sd": 1.0, "path": ("sigma_l",)},
        {"name": "sigma_g", "kind": "log", "mu": math.log(0.010), "sd": 1.0, "path": ("sigma_g",)},
        {"name": "sigma_b", "kind": "log", "mu": math.log(0.002), "sd": 1.5, "path": ("sigma_b",)},
        {"name": "kappa", "kind": "log", "mu": math.log(0.1), "sd": 1.5, "path": ("growth_reversion",)},
    ]
    ind = sel[sel.metric == "indicator"]
    counts = ind.groupby("indicator").size() if len(ind) else pd.Series(dtype=int)
    facs = set()
    for name, n in counts.items():
        if n < MIN_IND_OBS or name not in cat:
            continue
        c = cat[name]
        spec += [
            {"name": f"{name}.beta", "kind": "lin", "mu": c["beta"], "sd": c["beta_sd"], "path": ("indicators", name, "beta")},
            {"name": f"{name}.drift", "kind": "log", "mu": math.log(c["drift"]), "sd": 1.0, "path": ("indicators", name, "drift")},
            {"name": f"{name}.obs_sd", "kind": "log", "mu": math.log(c["obs_sd"]), "sd": 0.7, "path": ("indicators", name, "obs_sd")},
        ]
        if c.get("factor"):
            facs.add(c["factor"])
    for f in sorted(facs):
        spec.append({"name": f"factor.{f}.drift", "kind": "log", "mu": math.log(0.01), "sd": 1.5, "path": ("factors", f, "drift")})
    if ((sel.tier == "third_party") & (sel.metric == "run_rate")).any():
        spec.append({"name": "third_party.sd", "kind": "log", "mu": math.log(0.15), "sd": 0.7,
                     "path": ("tiers", "third_party", "log_sd")})
    for q in pd.period_range(config.MODEL_START, pd.Timestamp(last_date), freq="Q"):
        spec.append({"name": f"vol.{q}", "kind": "lin", "mu": 0.0, "sd": VOL_SD, "path": ("vol_scales", str(q))})
    return spec


def base_params(params: dict) -> dict:
    """Catalog indicator specs (not full-sample fits) as the starting point, so a vintage never sees the future."""
    p = copy.deepcopy(params)
    p["indicators"] = altdata.indicator_params()
    p.setdefault("factors", {})
    p.setdefault("vol_scales", {})
    p.setdefault("tiers", {})
    return p


def apply(theta, spec, params) -> dict:
    p = copy.deepcopy(params)
    for x, e in zip(theta, spec):
        v = math.exp(x) if e["kind"] == "log" else float(x)
        d = p
        for k in e["path"][:-1]:
            d = d.setdefault(k, {})
        d[e["path"][-1]] = v
    return p


def theta0(spec):
    return np.array([e["mu"] for e in spec], dtype=float)


def log_prior(theta, spec):
    return float(sum(norm.logpdf(x, e["mu"], e["sd"]) for x, e in zip(theta, spec)))


# --------------------------------------------------------------------------- parallel evaluation
_W = {}          # data shared with forked worker processes


def _nlp_worker(theta):
    return _neg_logpost(theta, _W["sel"], _W["params"], _W["spec"])


def _neg_logpost(theta, sel, params, spec):
    try:
        ll = Model(apply(theta, spec, params)).run(sel, smooth=False).loglik
    except (np.linalg.LinAlgError, ValueError, OverflowError, FloatingPointError):
        return 1e10
    if not np.isfinite(ll):
        return 1e10
    return -(ll + log_prior(theta, spec))


class _Evaluator:
    """Evaluates the negative log posterior at many points, in parallel when workers > 1."""
    def __init__(self, sel, params, spec, workers):
        self.workers = workers
        self.pool = None
        if workers > 1:
            import multiprocessing as mp
            _W.update(sel=sel, params=params, spec=spec)
            self.pool = mp.get_context("fork").Pool(workers)
        self.args = (sel, params, spec)

    def many(self, points):
        if self.pool is not None:
            return np.array(self.pool.map(_nlp_worker, list(points), chunksize=1))
        return np.array([_neg_logpost(x, *self.args) for x in points])

    def close(self):
        if self.pool is not None:
            self.pool.close(); self.pool.join(); self.pool = None


# --------------------------------------------------------------------------- posterior
class Posterior:
    def __init__(self, sel, params, spec, workers=4):
        self.sel, self.params, self.spec = sel, params, spec
        self.workers = workers
        self.theta_hat = None
        self.cov = None
        self.logpost_hat = None
        self.log_evidence = None

    def neg_logpost(self, theta):
        try:
            ll = Model(apply(theta, self.spec, self.params)).run(self.sel, smooth=False).loglik
        except (np.linalg.LinAlgError, ValueError, OverflowError, FloatingPointError):
            return 1e10
        if not np.isfinite(ll):
            return 1e10
        return -(ll + log_prior(theta, self.spec))

    def fit(self, maxiter=100, x0=None, eps=1e-3):
        x0 = theta0(self.spec) if x0 is None else np.asarray(x0, float)
        bounds = [(e["mu"] - 4 * e["sd"], e["mu"] + 4 * e["sd"]) for e in self.spec]
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        ev = _Evaluator(self.sel, self.params, self.spec, self.workers)
        d = len(x0)

        def fg(x):     # value and forward-difference gradient in one parallel batch
            pts = [x] + [x + eps * np.eye(d)[i] for i in range(d)]
            f = ev.many(pts)
            return f[0], (f[1:] - f[0]) / eps

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                opt = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
        finally:
            ev.close()
        self.theta_hat, self.logpost_hat = opt.x, -opt.fun
        return self

    def laplace(self, h=0.02):
        d = len(self.theta_hat)
        x = self.theta_hat
        E = np.eye(d) * h
        pts = [x + E[i] for i in range(d)] + [x - E[i] for i in range(d)]
        pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
        pts += [x + E[i] + E[j] for i, j in pairs] + [x - E[i] - E[j] for i, j in pairs]
        ev = _Evaluator(self.sel, self.params, self.spec, self.workers)
        try:
            f = ev.many(pts)
        finally:
            ev.close()
        f0 = -self.logpost_hat
        fp, fm = f[:d], f[d:2 * d]
        fpp, fmm = f[2 * d:2 * d + len(pairs)], f[2 * d + len(pairs):]
        Hm = np.zeros((d, d))
        Hm[np.diag_indices(d)] = (fp - 2 * f0 + fm) / h ** 2
        for k, (i, j) in enumerate(pairs):
            # H_ij from f(x+ei+ej), f(x-ei-ej) and the one-step values
            Hm[i, j] = Hm[j, i] = (fpp[k] + fmm[k] - fp[i] - fm[i] - fp[j] - fm[j] + 2 * f0) / (2 * h ** 2)
        # Keep the curvature at least as tight as the prior, and positive definite.
        prior_prec = np.diag([1 / e["sd"] ** 2 for e in self.spec])
        w, V = np.linalg.eigh(0.5 * (Hm + Hm.T))
        w = np.maximum(w, 1e-3)
        Hpd = V @ np.diag(w) @ V.T
        Hpd = np.where(np.diag(Hpd) > 0, Hpd, Hpd + prior_prec)
        self.cov = np.linalg.inv(Hpd)
        sign, logdet = np.linalg.slogdet(Hpd)
        self.log_evidence = self.logpost_hat + 0.5 * d * math.log(2 * math.pi) - 0.5 * logdet
        return self

    def draws(self, k=24, seed=0):
        rng = np.random.default_rng(seed)
        th = rng.multivariate_normal(self.theta_hat, self.cov, size=k - 1)
        return [self.theta_hat] + list(th)

    def summary(self) -> pd.DataFrame:
        sd = np.sqrt(np.diag(self.cov)) if self.cov is not None else np.full(len(self.spec), np.nan)
        rows = []
        for x, s, e in zip(self.theta_hat, sd, self.spec):
            f = math.exp if e["kind"] == "log" else (lambda v: v)
            rows.append({"param": e["name"], "map": f(x), "q10": f(x - 1.2816 * s), "q90": f(x + 1.2816 * s),
                         "prior_center": f(e["mu"])})
        return pd.DataFrame(rows)


def crossing_for_vintage(df, vintage, exclude_origin=None) -> float:
    """Crossing position estimated from data public at the vintage, with default (unfitted) parameters."""
    return robust.best_crossing_position(df, base_params({}), vintage, exclude_origin=exclude_origin)


def fit_posterior(df, params, vintage, exclude_origin=None, laplace=True, lag=0, def_lag=None, warm=None,
                  crossing=None):
    """Posterior from data public at `vintage` (optionally without one origin).

    crossing: None (keep the table's dates), a number (that crossing position), or
    "profile" (estimated from data public at the vintage).
    """
    if crossing is not None:
        rho = crossing_for_vintage(df, vintage, exclude_origin) if crossing == "profile" else float(crossing)
        df = data.with_crossing_position(df, rho)
    sel = data.select(robust._shift_indicators(df, lag) if lag else df, vintage=vintage)
    if exclude_origin is not None:
        sel = sel[sel.origin_id != exclude_origin]
    p = base_params(params)
    if lag:
        p = _shift_breaks(p, lag)
    spec = build_spec(sel, p, pd.Timestamp(vintage) + pd.offsets.QuarterEnd(1))
    x0 = None
    if warm is not None:      # warm start: previous MAP values for parameters present in both specs
        prev = dict(zip([e["name"] for e in warm.spec], warm.theta_hat))
        x0 = np.array([prev.get(e["name"], e["mu"]) for e in spec])
    post = Posterior(sel, p, spec).fit(x0=x0)
    post.df = df
    if laplace:
        post.laplace()
    return post


def _shift_breaks(params, lag):
    p = copy.deepcopy(params)
    for spec in p["indicators"].values():
        spec["breaks"] = [f"{pd.Timestamp(d) - pd.Timedelta(days=lag):%Y-%m-%d}" for d in spec.get("breaks", [])]
    return p


def mixture_predictive(post: Posterior, row, end, k=24, queries=None, segments=None):
    """Mean and sd of a claim's predictive, mixing over posterior draws (and the MAP-only value)."""
    mus, sds = [], []
    for i, th in enumerate(post.draws(k)):
        p = apply(th, post.spec, post.params)
        m = Model(p)
        try:
            res = m.run(post.sel, end=end, queries=queries, segments=segments)
            mu, sd = observation_predictive(res, row, m)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        if np.isfinite(mu) and np.isfinite(sd):
            mus.append(mu); sds.append(sd)
        if i == 0:
            map_pred = (mu, sd)
    mus, sds = np.array(mus), np.array(sds)
    mix_mu = float(mus.mean())
    mix_sd = float(math.sqrt((sds ** 2).mean() + mus.var()))
    return map_pred, (mix_mu, mix_sd)


# --------------------------------------------------------------------------- strict backtest
def bayes_backtest(df, params, k=24, verbose=True, crossing="profile") -> pd.DataFrame:
    """Strict real-time test: posterior refit at every target's vintage, from catalog priors."""
    full = data.select(df, filings=True)          # filings are targets, never evidence
    targets = full[full.tier.isin(backtest.TARGET_TIERS) & (full.kind == "point") & full.metric.isin(["run_rate", "revenue"])
                   & (full.is_forecast == 0) & (full.segment == "total")]
    rows = []
    prev = None
    targets = targets.sort_values("published")
    for tg in targets.itertuples():
        vintage = tg.published - pd.Timedelta(days=1)
        ev = data.select(df, vintage=vintage)
        ev = ev[ev.origin_id != tg.origin_id]
        if (ev.metric != "indicator").sum() < 3 or pd.Timestamp(tg.t_eff) < pd.Timestamp(config.MODEL_START):
            continue
        post = fit_posterior(df, params, vintage, exclude_origin=tg.origin_id, warm=prev, crossing=crossing)
        prev = post
        # the target itself may be a crossing claim: use its date under the vintage's crossing position
        tg = post.df[post.df.obs_id == tg.obs_id].assign(t_eff=lambda x: x.apply(data.effective_date, axis=1)).iloc[0]
        queries = [(tg.period_start, tg.period_end)] if tg.metric == "revenue" else None
        end = max(pd.Timestamp(tg.t_eff), vintage)
        (m0, s0), (m1, s1) = mixture_predictive(post, tg, end, k=k, queries=queries)
        y = math.log(tg.value) if tg.metric == "run_rate" else float(tg.value)
        for label, mu, sd in (("map_plugin", m0, s0), ("laplace_mixture", m1, s1)):
            err = (y - mu) if tg.metric == "run_rate" else math.log(y / mu)
            z = (y - mu) / sd
            rows.append({"target": tg.obs_id, "metric": tg.metric, "vintage": vintage.date(), "method": label,
                         "n_params": len(post.spec), "actual": tg.value, "forecast_mu": mu, "forecast_sd": sd,
                         "log_err": err, "z": z, "in80": abs(z) < 1.2816, "in50": abs(z) < 0.6745})
        if verbose:
            print(f"  {tg.obs_id}: map z={rows[-2]['z']:+.2f}  laplace z={rows[-1]['z']:+.2f}  ({len(post.spec)} params)", flush=True)
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    g = bt.groupby("method")
    return pd.DataFrame({"n": g.size(), "mae_log": g.log_err.apply(lambda s: s.abs().mean()),
                         "coverage80": g.in80.mean(), "coverage50": g.in50.mean(),
                         "sd_z": g.z.std(ddof=0), "mean_z": g.z.mean()})


# --------------------------------------------------------------------------- current nowcast
def posterior_nowcast(df, params, asof, dates=None, period=None, k=32,
                      lags=(0,), def_lags=(14,), verbose=True):
    """Posterior-averaged official-basis run-rate and quarter revenue, mixing over (lag, def_lag) by evidence."""
    asof = pd.Timestamp(asof)
    q = asof.to_period("Q")
    period = period or (q.start_time.normalize(), q.end_time.normalize())
    month_end = (asof - pd.DateOffset(months=1)) + pd.offsets.MonthEnd(0)
    dates = dates or [month_end, asof]
    combos = []
    for dl in def_lags:
        with robust.definition_lag(dl):
            for lag in lags:
                post = fit_posterior(df, params, asof, lag=lag)
                combos.append((dl, lag, post))
                if verbose:
                    print(f"  def_lag {dl}, indicator lag {lag:+d}: log evidence {post.log_evidence:.2f}", flush=True)
    ev = np.array([c[2].log_evidence for c in combos])
    w = np.exp(ev - ev.max()); w /= w.sum()
    rng = np.random.default_rng(0)
    out_rr = {d: [] for d in dates}
    out_rev = []
    for (dl, lag, post), wi in zip(combos, w):
        n_draws = max(int(round(k * wi)), 1 if wi > 0.02 else 0)
        if not n_draws:
            continue
        with robust.definition_lag(dl):
            for th in post.draws(n_draws, seed=int(1000 * wi)):
                p = apply(th, post.spec, post.params)
                m = Model(p)
                res = m.run(post.sel, end=max(asof, period[1]), queries=[period], segments=list(config.SEGMENTS))
                for d in dates:
                    mu, sd = res.log_runrate(d, "t28x13")
                    out_rr[d].append(rng.normal(mu, sd, 200))
                r = res.period_revenue(f"{pd.Timestamp(period[0]):%Y-%m-%d}..{pd.Timestamp(period[1]):%Y-%m-%d}")
                s2 = math.log1p((r["sd"] / r["mean"]) ** 2)
                out_rev.append(rng.normal(math.log(r["mean"]) - s2 / 2, math.sqrt(s2), 200))
    summ = {}
    for d in dates:
        x = np.exp(np.concatenate(out_rr[d]))
        summ[f"runrate_{pd.Timestamp(d):%Y-%m-%d}"] = {"median": float(np.median(x)), "q10": float(np.quantile(x, .1)),
                                                     "q90": float(np.quantile(x, .9))}
    x = np.exp(np.concatenate(out_rev))
    summ["revenue_quarter"] = {"mean": float(x.mean()), "q10": float(np.quantile(x, .1)), "q90": float(np.quantile(x, .9))}
    weights = pd.DataFrame([{"def_lag": dl, "lag": lag, "log_evidence": post.log_evidence, "weight": wi}
                            for (dl, lag, post), wi in zip(combos, w)])
    best = combos[int(np.argmax(w))][2]
    return summ, weights, best.summary()
