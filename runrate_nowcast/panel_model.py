"""Pooled two-company growth model (Anthropic + OpenAI), trained and tested by vintage.

Step G of DATA_PLAN.md. The specification below was written before any backtest of this
module was run, and is not changed afterwards to improve a score. Bug fixes found after the
first run are reported separately (README / report), with before and after numbers.

SPECIFICATION (fixed 2026-11-12, before the first run)
======================================================

Information set at a vintage v
  * Claims: the company's claims table (Anthropic: data/observations.csv, OpenAI:
    data/openai_claims.csv). Rows with filing != 0 are dropped when the table is loaded,
    so they are never inputs (they are read only by the scoring code, as targets). Only
    total-company run_rate and revenue rows are used (no segment or share rows). At v:
    published <= v, include == 1, is_forecast == 0, never the target's own origin_id; then
    data.select(vintage=v) once (one row per origin). Crossing claims keep data.load's
    dating rule (neutral rho = 0.5; params.json holds no fitted crossing position).
    Asserted in code (`_assert_public`), next to backtest.check_no_leak.
  * Features: only features.asof_features(v, entity). Asserted: the frame ends at the week
    that contains v.

Labels (per company, per vintage)
  * The claims-only state-space Model (model.py) run on the public claims of that company.
    Process noise (sigma_l, sigma_g, sigma_b, growth_reversion) is refit at every vintage
    on those claims only (MAP, same objective and priors as calibrate.fit_process_noise),
    as ml.labels_smoother does. Other parameters: config defaults.
  * Model start: Anthropic config.MODEL_START (2025-01-01); OpenAI 2023-12-01 (its first
    claims). Revenue rows whose period starts before the model start are dropped (OpenAI
    FY2023). Default priors for both.
  * Label y(d) = smoothed l + b at day d (log run-rate on the company's own run-rate
    definition; OpenAI's 4 weeks x 12 net basis and Anthropic's t28x13 gross basis are
    never mixed: each company has its own level and only growth rates are pooled).
    Label sd(d) = sqrt(h' P h), h = e_l + e_b, from the smoothed covariance.
  * Anchors (real data points): point total run-rate claims at their effective date, and
    point total revenue claims covering at most 100 days at the period midpoint (the
    ml.anchors rule, restricted to kind == point). Label weeks are the Thursdays (feature
    week labels) from max(first anchor, model start) to the last anchor t0. Nothing after
    t0 is used as a label. A company needs >= 3 anchors to be forecast, >= 2 to contribute
    training pairs.

Features (rates per week, from features.asof_features(v, company))
  Level series, built per week, forward-filled at most 4 weeks:
    openrouter_log_tokens   log tokens                  (log series -> log difference)
    openrouter_share        share of the three authors  (share -> difference)
    github_repo_share       share                       (share -> difference)
    ramp_adoption           share of businesses         (share -> difference)
    vercel_spend_share      share                       (share -> difference)
    log_price               log price index, Anthropic only (log difference; NaN for OpenAI)
    npm_level, pypi_level   cumulative sum of npm_dlog / pypi_dlog from the first observed
                            week (a missing week adds 0), i.e. a log download index
  Not used: openrouter_dlog_tokens and ramp_adoption_chg (duplicates of the above);
  cloud_bedrock_growth and cloud_foundry_growth (Anthropic-only quarterly growth
  statements, not levels, mostly missing). No level of any series that grows with adoption
  enters a model.

Training pairs (both companies; for the control, the target company only)
  For each label week t and h in {4, 8, 13}, with t >= the first feature week,
  t - h >= the first label week and t + h <= t0 (the company's last anchor):
    y   = (y(t+h) - y(t)) / h                     weekly log growth (the target)
    mom = (y(t) - y(t-h)) / h                     own label growth over the previous h weeks
    x_f = (F_f(t+h) - F_f(t)) / h                 weekly change of each feature (NaN if missing)
    w   = h^2 / (sd(t)^2 + sd(t+h)^2 + 2 * 0.03^2)   inverse variance of y; normalised to mean 1
  One model covers all three horizons (rate form), so any gap can be forecast.

Models (each fit at every vintage on the pairs public at v)
  Linear models share one preprocessing, fit on the training rows only:
    company fixed effect: y, mom and every x_f are demeaned within company (weighted
    means), so the effect is not penalised; a missing x_f is then set to 0 (= the company
    mean, no information); columns are scaled by their weighted RMS.
  (a) trend  ridge on [mom]                       company effect + momentum, no features
  (b) pca    ridge on [mom, PC_1..PC_k]           PCs: weighted PCA of the scaled feature
                                                  changes, k in {1, 2}
  (c) ridge  ridge on [mom, all x_f]
  (d) gbm    HistGradientBoostingRegressor on [is_anthropic, mom, all x_f] (missing values
             handled natively), target y, sample weights w; learning_rate 0.05,
             min_samples_leaf 15
  Ridge: sklearn Ridge(fit_intercept=False) with sample weights, alpha in
  {0.01, 0.1, 1, 10, 100, 1000, 10000}. GBM grid: max_depth {2, 3} x max_iter {50, 200}.
  Hyperparameters are chosen by time-ordered cross-validation inside the training pairs:
  the distinct start weeks are cut into 5 consecutive blocks; for blocks 2..5, validate on
  pairs starting in the block and train on pairs that END before the block starts (purged,
  no overlap); a fold needs >= 20 training and >= 5 validation pairs; loss = weighted mean
  absolute error of y; ties -> the first grid value. With no usable fold: alpha 10, k 1,
  max_depth 2, max_iter 50.
  Variants: "pooled" (both companies' pairs) and "own" (target company only). The own-
  company version of every model is the pre-declared control for pooling.

Forecast for a target, made at vintage v
  t0 = the target company's last anchor public at v. For a date d:
    d <= t0: the smoothed label path itself (same for every model);
    d >  t0: y(t0) + g * rate_hat, g = (d - t0) in weeks, where rate_hat is the model's
             weekly growth given x_f = (F_f(week of d) - F_f(week of t0)) / (weeks between)
             and mom over the last m = clip(round(g), 4, 13) weeks before t0.
  Run-rate target: d = its effective date (ref_date minus the definition lag); forecast =
  exp of the above on the run-rate basis (l + b).
  Revenue target: integrate the daily path over the period: sum of exp(level(s)) / 365,
  with level = smoothed l (GAAP basis) up to t0 and l(t0) + g(s) * rate_hat(s) after.
  A target is skipped (no panel forecast) when the target company has < 3 anchors or
  < 10 training pairs.

Backtest
  Primary set: the Anthropic targets of backtest.run_backtest (GAAP / official point
  totals, not forecasts; filing rows only as targets), t_eff >= MODEL_START, vintage by
  backtest.target_vintage under both rules (publication, period_end).
  Secondary set: OpenAI's included, non-forecast point total run-rate and revenue claims
  (all tiers, one per origin) with t_eff >= MODEL_START, same two rules.
  Error: log(actual / forecast). Also written: the claims-only smoother's own prediction
  from the same label run ("SS claims only (panel labels)") and backtest._naive on the
  same public claims ("naive last claim").
  Pre-declared comparisons: each model pooled vs own company only (primary: (b) pca);
  (a) vs (b)-(d) (do features help); (b) pooled vs A claims only, E OpenRouter tokens x
  price, G GitHub new-repo share and the naive last claim, paired on common targets
  (forecast_eval.paired).
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from . import backtest, config, data
from . import features as F
from .model import Model, L, B, period_key

COMPANIES = ("anthropic", "openai")
CLAIMS_PATHS = {"anthropic": config.DATA_PATH,
                "openai": os.path.join(config.PKG_DIR, "data", "openai_claims.csv")}
MODEL_STARTS = {"anthropic": config.MODEL_START, "openai": "2023-12-01"}

HORIZONS = (4, 8, 13)
LABEL_SD_FLOOR = 0.03
MIN_ANCHORS = 3
MIN_ANCHORS_CONTRIB = 2
MIN_TARGET_PAIRS = 10
FFILL_WEEKS = 4
MAX_REV_ANCHOR_DAYS = 100
WEEK = pd.Timedelta(days=7)

# feature name -> (asof_features column, how the level is built)
FEATURE_SPEC = {
    "openrouter_log_tokens": ("openrouter_log_tokens", "level"),
    "openrouter_share": ("openrouter_share", "level"),
    "github_repo_share": ("github_repo_share", "level"),
    "ramp_adoption": ("ramp_adoption", "level"),
    "vercel_spend_share": ("vercel_spend_share", "level"),
    "log_price": ("log_price", "level"),
    "npm_level": ("npm_dlog", "cumsum"),
    "pypi_level": ("pypi_dlog", "cumsum"),
}
FEATURES = list(FEATURE_SPEC)

MODELS = ("trend", "pca", "ridge", "gbm")
POOLS = ("pooled", "own")
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
PCA_K = (1, 2)
GBM_GRID = tuple({"max_depth": d, "max_iter": n} for d in (2, 3) for n in (50, 200))
GBM_FIXED = {"learning_rate": 0.05, "min_samples_leaf": 15, "random_state": 0}
GRIDS = {
    "trend": tuple({"alpha": a} for a in ALPHAS),
    "pca": tuple({"alpha": a, "k": k} for k in PCA_K for a in ALPHAS),
    "ridge": tuple({"alpha": a} for a in ALPHAS),
    "gbm": GBM_GRID,
}
DEFAULT_HYPER = {"trend": {"alpha": 10.0}, "pca": {"alpha": 10.0, "k": 1},
                 "ridge": {"alpha": 10.0}, "gbm": {"max_depth": 2, "max_iter": 50}}
CV_BLOCKS = 5
CV_MIN_TRAIN = 20
CV_MIN_VALID = 5

NOISE_KEYS = ("sigma_l", "sigma_g", "sigma_b", "growth_reversion")


# --------------------------------------------------------------------------- claims
def load_claims(company: str, path: str | None = None) -> pd.DataFrame:
    """Total-company run-rate and revenue claims of one company, as loaded by data.load.

    Filing rows (filing != 0) are removed here, so nothing downstream can use them.
    """
    df = data.load(path or CLAIMS_PATHS[company])
    df = df[(df.segment == "total") & df.metric.isin(["run_rate", "revenue"])]
    df = df[df.filing.fillna(0).astype(int) == 0].copy()
    df["company"] = company
    return df.reset_index(drop=True)


def load_targets(company: str, path: str | None = None) -> pd.DataFrame:
    """Scoring targets. Anthropic: the backtest.run_backtest selection (filings allowed as
    targets). OpenAI: every included, non-forecast point total claim (all tiers)."""
    df = data.load(path or CLAIMS_PATHS[company])
    full = data.select(df[df.metric != "indicator"], filings=True)
    t = full[(full.kind == "point") & full.metric.isin(["run_rate", "revenue"])
             & (full.is_forecast == 0) & (full.segment == "total")]
    if company == "anthropic":
        t = t[t.tier.isin(backtest.TARGET_TIERS)]
    t = t[pd.to_datetime(t.t_eff) >= pd.Timestamp(config.MODEL_START)]
    return t.assign(company=company).reset_index(drop=True)


def _assert_public(sel: pd.DataFrame, vintage: pd.Timestamp, exclude_origin) -> None:
    """Hard guard: claims public by the vintage, no filing row, never the target's origin."""
    if len(sel) and (pd.to_datetime(sel.published) > vintage).any():
        raise AssertionError(f"claims published after {vintage:%Y-%m-%d}: "
                             f"{sel[sel.published > vintage].obs_id.tolist()}")
    if len(sel) and (sel.filing.fillna(0).astype(int) != 0).any():
        raise AssertionError(f"filing rows in claims: {sel[sel.filing != 0].obs_id.tolist()}")
    if exclude_origin is not None and (sel.origin_id == exclude_origin).any():
        raise AssertionError(f"target origin {exclude_origin} in claims")


def public_claims(claims: pd.DataFrame, vintage, exclude_origin=None,
                  start: str | None = None) -> pd.DataFrame:
    """Claims usable at the vintage (selected once with data.select), see the spec."""
    v = pd.Timestamp(vintage)
    raw = claims[(claims.published <= v) & (claims.include == 1) & (claims.is_forecast == 0)
                 & (claims.filing.fillna(0).astype(int) == 0)]
    if exclude_origin is not None:
        raw = raw[raw.origin_id != exclude_origin]
    sel = data.select(raw, vintage=v, forecasts=False, filings=False)
    if start is not None and len(sel):
        s = pd.Timestamp(start)
        sel = sel[~((sel.metric == "revenue") & (pd.to_datetime(sel.period_start) < s))].reset_index(drop=True)
    _assert_public(sel, v, exclude_origin)
    return sel


def anchors(sel: pd.DataFrame) -> pd.Series:
    """Real data points (log run-rate basis), indexed by date: point run-rates at t_eff,
    point revenue of <= 100 days at the period midpoint (revenue x 365 / days)."""
    pts = []
    rr = sel[(sel.metric == "run_rate") & (sel.kind == "point")]
    pts += [(pd.Timestamp(r.t_eff), math.log(r.value)) for r in rr.itertuples()]
    rev = sel[(sel.metric == "revenue") & (sel.kind == "point")]
    for r in rev.itertuples():
        days = (r.period_end - r.period_start).days + 1
        if days <= MAX_REV_ANCHOR_DAYS:
            pts.append((r.period_start + pd.Timedelta(days=days // 2), math.log(r.value * 365 / days)))
    if not pts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return pd.Series([v for _, v in pts], index=pd.DatetimeIndex([d for d, _ in pts])).groupby(level=0).mean().sort_index()


# --------------------------------------------------------------------------- process noise
_NOISE_CACHE: dict = {}


def _noise_key(company, sel: pd.DataFrame) -> tuple:
    cols = ["obs_id", "value", "published", "t_eff", "kind"]
    return (company,) + tuple(map(tuple, sel[cols].astype(str).to_numpy()))


def fit_noise(sel: pd.DataFrame, start: str) -> dict:
    """MAP process noise from the online filter's likelihood: calibrate.fit_process_noise's
    objective and priors, for already-selected claims and a given model start."""
    d0 = config.DEFAULT_PARAMS
    names = list(NOISE_KEYS)
    centers = np.log([d0["sigma_l"], d0["sigma_g"], 0.002, 0.1])
    prior_sds = np.array([1.0, 1.0, 1.5, 1.5])

    def negpost(theta):
        p = {k: float(v) for k, v in zip(names, np.exp(theta))}
        ll = Model(p, start=start).run(sel, smooth=False).loglik
        return -(ll + float(np.sum(norm.logpdf(theta, centers, prior_sds))))

    opt = minimize(negpost, centers.copy(), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-3, "maxiter": 400})
    return {k: float(v) for k, v in zip(names, np.exp(opt.x))}


def vintage_noise(company: str, sel: pd.DataFrame) -> dict:
    key = _noise_key(company, sel)
    if key not in _NOISE_CACHE:
        _NOISE_CACHE[key] = fit_noise(sel, MODEL_STARTS[company])
    return _NOISE_CACHE[key]


# --------------------------------------------------------------------------- labels
@dataclass
class Labels:
    company: str
    vintage: pd.Timestamp
    sel: pd.DataFrame
    anchors: pd.Series
    y: pd.Series                   # weekly label (log run-rate basis), Thursdays
    sd: pd.Series
    t0: pd.Timestamp | None
    start: pd.Timestamp | None     # first label day
    res: object = None             # model.Result
    model: object = None
    params: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.t0 is not None

    def level(self, d, basis: str = "runrate") -> float:
        m, _ = self.res.state(pd.Timestamp(d))
        return float(m[L] + (m[B] if basis == "runrate" else 0.0))


def _thursdays(first, last) -> pd.DatetimeIndex:
    """Week labels (Thursday of weeks ending Sunday) whose Thursday lies in [first, last]."""
    first, last = pd.Timestamp(first), pd.Timestamp(last)
    lab0 = F._week_label([first])[0]
    if lab0 < first:
        lab0 += WEEK
    if lab0 > last:
        return pd.DatetimeIndex([])
    return pd.date_range(lab0, last, freq="7D")


def company_labels(company: str, claims: pd.DataFrame, vintage, exclude_origin=None,
                   params: dict | None = None, end=None, queries=None) -> Labels:
    """Smoothed weekly labels for one company at a vintage (see the spec).

    params: process noise to use; None refits it on the claims public at the vintage.
    end / queries: extend the run to a target date / add a revenue period to integrate.
    """
    v = pd.Timestamp(vintage)
    start = MODEL_STARTS[company]
    sel = public_claims(claims, v, exclude_origin, start=start)
    anch = anchors(sel)
    anch = anch[anch.index >= pd.Timestamp(start)]
    if len(anch) < MIN_ANCHORS_CONTRIB:
        return Labels(company, v, sel, anch, pd.Series(dtype=float), pd.Series(dtype=float), None, None)
    p = dict(vintage_noise(company, sel) if params is None else params)
    model = Model(p, start=start)
    t0 = anch.index.max()
    last = max([v, t0] + ([pd.Timestamp(end)] if end is not None else []))
    res = model.run(sel, end=last, queries=queries)
    first = max(anch.index.min(), pd.Timestamp(start))
    weeks = _thursdays(first, t0)
    h = np.zeros(res.ms.shape[1])
    h[L] = h[B] = 1.0
    ys, sds = [], []
    for w in weeks:
        m, P = res.state(w)
        ys.append(float(h @ m))
        sds.append(math.sqrt(max(float(h @ P @ h), 0.0)))
    return Labels(company, v, sel, anch, pd.Series(ys, index=weeks), pd.Series(sds, index=weeks),
                  t0, first, res, model, p)


# --------------------------------------------------------------------------- features
def feature_levels(company: str, vintage, sources: dict | None = None) -> pd.DataFrame:
    """Weekly level-form features (FEATURE_SPEC) from features.asof_features(vintage)."""
    v = pd.Timestamp(vintage)
    f = F.asof_features(v, company, sources=sources)
    if len(f) and f.index.max() > F._week_label([v])[0]:
        raise AssertionError("as-of features extend past the week of the vintage")
    out = pd.DataFrame(index=f.index)
    for name, (col, how) in FEATURE_SPEC.items():
        s = f[col].astype(float)
        if how == "cumsum":
            first, last = s.first_valid_index(), s.last_valid_index()
            lev = s.fillna(0.0).cumsum()
            if first is None:
                lev[:] = np.nan
            else:
                lev[(lev.index < first) | (lev.index > last + FFILL_WEEKS * WEEK)] = np.nan
            out[name] = lev
        else:
            out[name] = s.ffill(limit=FFILL_WEEKS)
    return out


# --------------------------------------------------------------------------- training pairs
def make_pairs(lab: Labels, flev: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Training pairs of one company (see the spec): y, mom, weight and feature rates."""
    cols = ["company", "t", "t1", "h", "y", "mom", "w_raw"] + FEATURES
    if not lab.ok or not len(lab.y) or not len(flev):
        return pd.DataFrame(columns=cols)
    y, sd = lab.y, lab.sd
    first_feat = flev.index.min()
    rows = []
    for t in y.index:
        if t < first_feat:
            continue
        for h in horizons:
            t1, tm = t + h * WEEK, t - h * WEEK
            if t1 not in y.index or tm not in y.index or t1 not in flev.index or t not in flev.index:
                continue
            x = (flev.loc[t1, FEATURES].to_numpy(float) - flev.loc[t, FEATURES].to_numpy(float)) / h
            w = h ** 2 / (sd[t] ** 2 + sd[t1] ** 2 + 2 * LABEL_SD_FLOOR ** 2)
            rows.append([lab.company, t, t1, h, (y[t1] - y[t]) / h, (y[t] - y[tm]) / h, w] + list(x))
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- models
def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    m = ~np.isnan(x)
    return float(np.sum(w[m] * x[m]) / np.sum(w[m])) if m.any() and np.sum(w[m]) > 0 else np.nan


class LinearPanel:
    """Company fixed effect (within demeaning) + scaling (+ weighted PCA) + ridge.

    Every statistic (company means, scales, principal components, coefficients) is computed
    in fit() from the training rows only; predict() only applies them.
    """

    def __init__(self, kind: str, alpha: float, k: int = 1):
        self.kind, self.alpha, self.k = kind, float(alpha), int(k)

    def _demean(self, P: pd.DataFrame):
        cols = ["mom"] + FEATURES
        X = P[cols].to_numpy(float).copy()
        for i, c in enumerate(P["company"].to_numpy()):
            mu = self.means_.get(c, self.means_["__all__"])
            X[i] = X[i] - mu[1:]
        X[np.isnan(X)] = 0.0
        return X

    def fit(self, P: pd.DataFrame):
        from sklearn.linear_model import Ridge
        w = P["w"].to_numpy(float)
        cols = ["y", "mom"] + FEATURES
        M = P[cols].to_numpy(float)
        self.means_ = {"__all__": np.array([_wmean(M[:, j], w) for j in range(M.shape[1])])}
        for c, g in P.groupby("company"):
            wc = g["w"].to_numpy(float)
            Mc = g[cols].to_numpy(float)
            mu = np.array([_wmean(Mc[:, j], wc) for j in range(Mc.shape[1])])
            self.means_[c] = np.where(np.isnan(mu), self.means_["__all__"], mu)
        self.means_["__all__"] = np.nan_to_num(self.means_["__all__"])
        for c in list(self.means_):
            self.means_[c] = np.nan_to_num(self.means_[c])
        yd = P["y"].to_numpy(float) - np.array([self.means_.get(c)[0] for c in P["company"]])
        X = self._demean(P)
        sc = np.sqrt(np.sum(w[:, None] * X ** 2, axis=0) / np.sum(w))
        self.scale_ = np.where(sc > 1e-12, sc, 1.0)
        Z = X / self.scale_
        self.components_ = None
        D = self._design(Z, fitting=True, w=w)
        self.ridge_ = Ridge(alpha=self.alpha, fit_intercept=False).fit(D, yd, sample_weight=w)
        self.n_train_ = len(P)
        return self

    def _design(self, Z: np.ndarray, fitting: bool = False, w=None) -> np.ndarray:
        mom, Xf = Z[:, :1], Z[:, 1:]
        if self.kind == "trend":
            return mom
        if self.kind == "ridge":
            return Z
        if self.kind == "pca":
            if fitting:
                ws = np.sqrt(w / np.mean(w))[:, None]
                active = np.any(Xf != 0, axis=0)
                comps = np.zeros((0, Xf.shape[1]))
                if active.any():
                    _, s, vt = np.linalg.svd(ws * Xf, full_matrices=False)
                    keep = min(self.k, int(np.sum(s > 1e-10)))
                    comps = vt[:keep]
                self.components_ = comps
            return np.hstack([mom, Xf @ self.components_.T])
        raise ValueError(self.kind)

    def predict(self, P: pd.DataFrame) -> np.ndarray:
        Z = self._demean(P) / self.scale_
        D = self._design(Z)
        base = np.array([self.means_.get(c, self.means_["__all__"])[0] for c in P["company"]])
        return base + D @ self.ridge_.coef_

    def elasticities(self) -> dict:
        """Coefficients in the original units (weekly growth per unit of weekly feature change);
        for pca, the feature coefficients implied by the retained components."""
        coef = self.ridge_.coef_
        out = {"mom": float(coef[0] / self.scale_[0])}
        if self.kind == "ridge":
            beta = coef[1:]
        elif self.kind == "pca":
            beta = self.components_.T @ coef[1:]
        else:
            return out
        out.update({n: float(b / s) for n, b, s in zip(FEATURES, beta, self.scale_[1:])})
        return out


class GBMPanel:
    def __init__(self, max_depth: int, max_iter: int):
        self.max_depth, self.max_iter = int(max_depth), int(max_iter)

    @staticmethod
    def _X(P: pd.DataFrame) -> np.ndarray:
        return np.column_stack([(P["company"] == "anthropic").to_numpy(float),
                                P[["mom"] + FEATURES].to_numpy(float)])

    def fit(self, P: pd.DataFrame):
        from sklearn.ensemble import HistGradientBoostingRegressor
        X = self._X(P)
        # a column with no value in the training rows cannot split (and sklearn's binning
        # rejects it): keep the columns observed at least once in training
        self.cols_ = np.where(~np.all(np.isnan(X), axis=0))[0]
        self.est_ = HistGradientBoostingRegressor(max_depth=self.max_depth, max_iter=self.max_iter, **GBM_FIXED)
        self.est_.fit(X[:, self.cols_], P["y"].to_numpy(float), sample_weight=P["w"].to_numpy(float))
        self.n_train_ = len(P)
        return self

    def predict(self, P: pd.DataFrame) -> np.ndarray:
        return self.est_.predict(self._X(P)[:, self.cols_])


def build(kind: str, hyper: dict):
    return GBMPanel(**hyper) if kind == "gbm" else LinearPanel(kind, **hyper)


def _normalise(P: pd.DataFrame) -> pd.DataFrame:
    P = P.copy()
    P["w"] = P["w_raw"] / P["w_raw"].mean()
    return P


def cv_folds(P: pd.DataFrame, n_blocks: int = CV_BLOCKS):
    """Time-ordered, purged folds: (train index, validation index) per usable block."""
    starts = np.sort(P["t"].unique())
    if len(starts) < n_blocks:
        return []
    blocks = np.array_split(starts, n_blocks)
    out = []
    for b in blocks[1:]:
        b0 = b[0]
        va = P.index[P["t"].isin(b)]
        tr = P.index[P["t1"] < b0]
        if len(tr) >= CV_MIN_TRAIN and len(va) >= CV_MIN_VALID:
            out.append((tr, va))
    return out


def select_hyper(kind: str, P: pd.DataFrame) -> tuple[dict, int]:
    """Hyperparameters by time-ordered CV on the training pairs; (choice, folds used)."""
    folds = cv_folds(P)
    if not folds:
        return dict(DEFAULT_HYPER[kind]), 0
    best, best_loss = None, np.inf
    for hyp in GRIDS[kind]:
        losses = []
        for tr, va in folds:
            m = build(kind, hyp).fit(_normalise(P.loc[tr]))
            Pv = P.loc[va]
            e = np.abs(Pv["y"].to_numpy(float) - m.predict(Pv))
            wv = Pv["w_raw"].to_numpy(float)
            losses.append(float(np.sum(wv * e) / np.sum(wv)))
        loss = float(np.mean(losses))
        if loss < best_loss - 1e-15:
            best, best_loss = hyp, loss
    return dict(best), len(folds)


def fit_model(kind: str, P: pd.DataFrame):
    """CV-selected hyperparameters, then a fit on all training pairs."""
    P = P.reset_index(drop=True)
    hyp, nf = select_hyper(kind, P)
    m = build(kind, hyp).fit(_normalise(P))
    m.hyper_, m.cv_folds_ = hyp, nf
    return m


# --------------------------------------------------------------------------- forecasting
def _feature_rate(flev: pd.DataFrame, t0: pd.Timestamp, d: pd.Timestamp) -> np.ndarray:
    w0 = F._week_label([t0])[0]
    w1 = min(F._week_label([d])[0], flev.index.max())
    n = (w1 - w0).days / 7
    if n < 1 or w0 not in flev.index or w1 not in flev.index:
        return np.full(len(FEATURES), np.nan)
    return (flev.loc[w1, FEATURES].to_numpy(float) - flev.loc[w0, FEATURES].to_numpy(float)) / n


def _momentum(lab: Labels, t0: pd.Timestamp, g: float) -> float:
    m = int(min(max(round(g), min(HORIZONS)), max(HORIZONS)))
    for mm in range(m, 0, -1):                      # shorten the window if labels start later
        tm = t0 - mm * WEEK
        if tm >= lab.start:
            return (lab.level(t0) - lab.level(tm)) / mm
    return np.nan


def growth_rows(lab: Labels, flev: pd.DataFrame, dates) -> pd.DataFrame:
    """Prediction inputs for dates after t0: one row per date (company, mom, feature rates, g)."""
    rows = []
    for d in dates:
        g = (pd.Timestamp(d) - lab.t0).days / 7
        rows.append([lab.company, g, _momentum(lab, lab.t0, g)] + list(_feature_rate(flev, lab.t0, pd.Timestamp(d))))
    return pd.DataFrame(rows, columns=["company", "g", "mom"] + FEATURES)


def target_dates(target) -> tuple[list, str]:
    if target.metric == "run_rate":
        return [pd.Timestamp(target.t_eff)], "runrate"
    return list(pd.date_range(target.period_start, target.period_end, freq="D")), "gaap"


def path_forecast(lab: Labels, flev: pd.DataFrame, target, predict) -> float:
    """Forecast of a target from the label path up to t0 and predicted growth after it.

    predict: function(DataFrame of growth_rows) -> weekly growth rate per row, or None for
    the label path only (all dates must then be <= t0).
    """
    dates, basis = target_dates(target)
    after = [d for d in dates if d > lab.t0]
    lv = {}
    for d in dates:
        if d <= lab.t0:
            lv[d] = lab.level(d, basis)
    if after:
        G = growth_rows(lab, flev, after)
        rate = predict(G)
        start = lab.level(lab.t0, basis)
        for d, r, g in zip(after, rate, G["g"]):
            lv[d] = start + float(r) * float(g)
    path = np.array([lv[d] for d in dates])
    if np.isnan(path).any():
        return np.nan
    if target.metric == "run_rate":
        return float(math.exp(path[0]))
    return float(np.exp(path).sum() / config.DAYS_PER_YEAR)


def smoother_forecast(lab: Labels, target) -> float:
    """The claims-only smoother's own prediction of the target from the same label run."""
    if target.metric == "run_rate":
        return float(math.exp(lab.level(target.t_eff, "runrate")))
    return float(lab.res.period_revenue(period_key(target.period_start, target.period_end))["mean"])


# --------------------------------------------------------------------------- one target
def forecast_target(target, vintage, claims: dict, sources: dict, models=MODELS, pools=POOLS,
                    params: dict | None = None) -> list[dict]:
    """All panel forecasts (and the two references) of one target at one vintage.

    claims: {company: load_claims(company)}; sources: features.load_sources().
    params: {company: process noise} to skip refitting (tests only).
    """
    v = pd.Timestamp(vintage)
    tc = target.company
    other = [c for c in COMPANIES if c != tc]
    dates, _ = target_dates(target)
    queries = [(target.period_start, target.period_end)] if target.metric == "revenue" else None
    labs = {tc: company_labels(tc, claims[tc], v, exclude_origin=target.origin_id,
                               params=(params or {}).get(tc), end=max(dates), queries=queries)}
    for c in other:
        labs[c] = company_labels(c, claims[c], v, params=(params or {}).get(c))
    # the target's own origin must not be in any claims used
    for c, lab in labs.items():
        backtest.check_no_leak(lab.sel, v, target)
    lab = labs[tc]
    base = {"target": target.obs_id, "company": tc, "metric": target.metric, "vintage": v.date(),
            "actual": float(target.value), "n_anchors": len(lab.anchors),
            "t0": None if lab.t0 is None else lab.t0.date()}
    out = []
    if len(lab.anchors) < MIN_ANCHORS or not lab.ok:
        return out
    gap = max((max(dates) - lab.t0).days / 7, 0.0)
    base["gap_weeks"] = round(gap, 2)

    def row(model, pool, fc, **extra):
        le = math.log(target.value / fc) if fc and fc > 0 and not math.isnan(fc) else np.nan
        return {**base, "model": model, "pool": pool, "forecast": fc, "log_err": le, **extra}

    out.append(row("SS claims only (panel labels)", "own", smoother_forecast(lab, target), n_train_pairs=0))
    nv = backtest._naive(target, lab.sel)
    if nv is not None:
        fc = math.exp(nv) if target.metric == "run_rate" else nv
        out.append(row("naive last claim", "own", fc, n_train_pairs=0))

    flevs = {c: feature_levels(c, v, sources) for c in labs}
    pairs = {c: make_pairs(labs[c], flevs[c]) for c in labs}    # empty for a company with < 2 anchors
    n_own = len(pairs[tc])
    if n_own < MIN_TARGET_PAIRS:
        return out
    for pool in pools:
        P = pd.concat([pairs[c] for c in ([tc] if pool == "own" else [tc] + other) if len(pairs[c])],
                      ignore_index=True)
        # training labels end at each company's last anchor, which is public at v
        assert (P["t1"] <= v).all() and all((P[P.company == c]["t1"] <= labs[c].t0).all() for c in P.company.unique())
        for kind in models:
            m = fit_model(kind, P)
            fc = path_forecast(lab, flevs[tc], target, m.predict)
            out.append(row(f"panel {kind} ({pool})", pool, fc, n_train_pairs=len(P),
                           n_target_pairs=n_own, hyper=str(m.hyper_), cv_folds=m.cv_folds_))
    return out


def fitted_models(target, vintage, claims: dict, sources: dict, kinds=("ridge", "pca"), pool: str = "pooled",
                  params: dict | None = None) -> dict:
    """The models forecast_target fits for one target and pool (for reporting what they learned)."""
    v = pd.Timestamp(vintage)
    tc = target.company
    labs = {c: company_labels(c, claims[c], v, exclude_origin=target.origin_id if c == tc else None,
                              params=(params or {}).get(c)) for c in COMPANIES}
    pairs = {c: make_pairs(labs[c], feature_levels(c, v, sources)) for c in COMPANIES}
    P = pd.concat([pairs[c] for c in ([tc] if pool == "own" else COMPANIES) if len(pairs[c])], ignore_index=True)
    return {k: fit_model(k, P) for k in kinds} | {"pairs": P}


# --------------------------------------------------------------------------- backtest
def target_jobs(rule: str, companies=COMPANIES) -> list[tuple]:
    jobs = []
    for c in companies:
        for t in load_targets(c).itertuples():
            jobs.append((c, t.obs_id, backtest.target_vintage(t, rule)))
    return jobs


def run_job(job, claims=None, sources=None) -> list[dict]:
    company, obs_id, v = job
    warnings.simplefilter("ignore")
    claims = claims or {c: load_claims(c) for c in COMPANIES}
    sources = sources or F.load_sources()
    t = load_targets(company).set_index("obs_id").loc[obs_id]
    tgt = _Row(obs_id=obs_id, **t.to_dict())
    return forecast_target(tgt, v, claims, sources)


class _Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def run_backtest(rule: str, companies=COMPANIES, n_jobs: int = 1, verbose: bool = True) -> pd.DataFrame:
    jobs = target_jobs(rule, companies)
    rows = []
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for job, res in zip(jobs, ex.map(run_job, jobs)):
                rows += res
                if verbose:
                    print(f"  {rule} {job[0]} {job[1]} done ({len(res)} rows)", flush=True)
    else:
        claims = {c: load_claims(c) for c in COMPANIES}
        sources = F.load_sources()
        for job in jobs:
            res = run_job(job, claims, sources)
            rows += res
            if verbose:
                print(f"  {rule} {job[0]} {job[1]} done ({len(res)} rows)", flush=True)
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(0, "rule", rule)
    return out
