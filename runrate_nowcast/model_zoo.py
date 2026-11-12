"""A broad set of ML methods on the pooled two-company panel, under one fixed protocol.

The protocol below was written before any backtest of this module was run and is not changed
afterwards to improve a score. Bug fixes found after the first run are reported separately
(report / README), with before and after numbers.

PROTOCOL (fixed 2026-11-12, before the first run)
=================================================

Data, labels, training pairs, forecasts, targets: exactly panel_model.py (its docstring), reused
through its functions, not restated:
  * labels: pm.company_labels (claims-only state-space path per company, process noise refit on
    the claims public at the vintage, the target's own origin excluded, filing rows never read);
  * training pairs: 4 / 8 / 13-week weekly log growth of the label, `mom` (own label growth over
    the previous h weeks), weekly change of each feature, weight w = inverse label variance
    (normalised to mean 1 within each fit); pairs end at each company's last anchor t0 <= v;
  * forecast: the label path up to t0, then y(t0) + g * rate_hat; revenue targets integrate the
    daily path (pm.path_forecast logic, generalised only to take a feature list);
  * targets and vintages: pm.target_jobs / pm.load_targets with backtest.target_vintage, both rules
    (publication, period_end). Anthropic targets are the test set; OpenAI claims are a second test
    set. A target is skipped when its company has < 3 anchors or < 10 own training pairs.
  * pools: "pooled" (Anthropic + OpenAI pairs) for every target; "own" (target company only) for
    Anthropic targets, as the control for pooling.

Feature sets (`feature_set` argument)
  "g_spec": panel_model.FEATURE_SPEC (the 8 level series of sub-task G), built exactly as
            pm.feature_levels.
  any other name: features.FEATURE_SETS[name] (written by another sub-task). A dict
            {name: (asof column, "level" | "cumsum")} is used as given. A list of asof_features
            columns is mapped by this rule: a column that is in FEATURE_SPEC keeps that spec; a
            column whose name contains "dlog" or ends in "_chg", "_diff", "_growth" or "_rate" is
            a change or rate and enters as its cumulative sum ("cumsum"); every other column is a
            level ("level"). A change column is dropped when its level is in the set (the
            window change of the level already carries it): openrouter_dlog_tokens when
            openrouter_log_tokens is in, and any "<X>_chg" when "<X>" is in.
            asof_features gets feature_set=<name> if its signature accepts it.
            [The general "<X>_chg" rule was added after reading the column list of
            FEATURE_SETS["all"] and before any run with it.]

Model zoo (fixed list; MEMBERS)
  Preprocessing "linear" (as pm.LinearPanel, fit on training rows only): company fixed effect by
  within-company weighted demeaning of y, mom and every feature change; missing -> 0 (the company
  mean); columns scaled by their weighted RMS; the demeaned y is divided by its weighted RMS (so
  grids are scale-free; for ridge this changes nothing). Prediction = company mean + model.
  Preprocessing "raw" (as pm.GBMPanel): [is_anthropic, mom, feature changes], y as is, missing
  values handled natively by the trees; columns with no value in the training rows are dropped.
  Sample weights w are used by every member (exceptions noted).

   member        prep    model                                             grid (CV)
   trend         linear  Ridge on [mom]                                    alpha in pm.ALPHAS
   ridge         linear  Ridge on [mom, x]                                 alpha in pm.ALPHAS
   lasso         linear  Lasso on [mom, x]                                 alpha {.001,.003,.01,.03,.1,.3}
   enet          linear  ElasticNet on [mom, x]                            alpha {.003,.01,.03,.1} x l1_ratio {.2,.5,.8}
   pca           linear  Ridge on [mom, weighted PCs of x] (= pm "pca")    alpha in pm.ALPHAS x k {1,2}
   pls           linear  weighted PLS1 (NIPALS) on [mom, x]                n_components {1,2,3}
   bayes_ridge   linear  BayesianRidge (evidence maximisation)             none
   ard           linear  ARDRegression (no sample_weight in sklearn: rows scaled by sqrt(w),
                         exact weighted least squares without intercept)   none
   kernel_ridge  linear  KernelRidge, RBF kernel on [mom, x]               alpha {.1,1,10} x gamma {.01,.03,.1}
   gp            linear  GaussianProcessRegressor, amplitude x RBF + white noise; the noise
                         variance of pair i is s / w_i (heteroscedastic, from the label
                         variance); amplitude and length scale by marginal likelihood
                         (no restarts)                                     s {.1,.3,1}
   knn           linear  k nearest neighbours (Euclidean on [mom, x]), prediction = w-weighted
                         mean of the neighbours' y                         k {5,10,20,40}
   rf            raw     RandomForestRegressor, 200 trees                  min_samples_leaf {5,20} x max_features {.5,1}
   extra_trees   raw     ExtraTreesRegressor, 200 trees                    same as rf
   hgb           raw     HistGradientBoostingRegressor (= pm "gbm")        pm.GBM_GRID, pm.GBM_FIXED
   lightgbm      raw     LGBMRegressor, lr .05, min_child_samples 15       num_leaves {3,7} x n_estimators {50,200}
   xgboost       raw     XGBRegressor, hist, lr .05, min_child_weight 15   max_depth {2,3} x n_estimators {50,200}
   mlp           linear  MLPRegressor, tanh, adam lr .01, early stopping
                         (20% random validation split inside the training
                         rows, 20 epochs patience)                         hidden {(4,),(8,)} x alpha {.01,1}
   midas         linear  MIDAS-style ridge: weekly distributed lag of feature changes with Almon
                         polynomial weights. For a window (t, t+h] and lag j weeks,
                         m_j = (F(t+h-j) - F(t-j)) / h (m_0 = x, the plain feature change);
                         regressors Z_p = sum_{j=1..13} (j/13)^p m_j for p = 0..degree (missing if
                         any m_j is missing). Ridge on [mom, x, Z]: lag 0 enters freely, lags
                         1..13 through the Almon polynomial. At forecast time the window is
                         (week of t0, last feature week <= d], as for the other members.
                         alpha in pm.ALPHAS x degree {1,2}.
                         [Revised before the first backtest: the first draft put lags 0..13 all
                         under one polynomial, which cannot represent a contemporaneous-only
                         effect, so it did not nest ridge; found by a synthetic unit test.]
  All random seeds are 0; every member runs single-threaded. If lightgbm or xgboost cannot be
  imported, that member is dropped and the report says so.

Hyperparameters
  pm.cv_folds on the training pairs (5 consecutive blocks of start weeks; validate on blocks 2..5,
  train on pairs that END before the block starts; >= 20 training and >= 5 validation pairs).
  Loss = weighted (w_raw) mean absolute error of y, averaged over folds; ties -> the first grid
  value. With no usable fold: the first grid value (pm.DEFAULT_HYPER for trend/ridge/pca/hgb).

Combinations (computed from the training pairs only, never from any test target)
  AUTO (the headline model, declared in advance): at each vintage and pool, the member with the
       lowest inner-CV loss at its selected hyperparameters; ties -> the first in MEMBERS order;
       with no usable fold -> trend. Its forecast is that member's forecast.
  stack: non-negative weights summing to one on the members' out-of-fold predictions (the CV
       predictions at each member's selected hyperparameters, validation blocks 2..5), chosen by
       weighted (w_raw) least squares; rows where any member's OOF prediction is missing are
       dropped; a member with no OOF predictions (every grid value failed) gets weight 0; with
       no usable fold -> equal weights. Forecast rate = weighted sum of the members'
       predicted rates. (The OOF predictions of the selected hyperparameters were also used to
       select them, so they are slightly optimistic; this is accepted, not corrected.)
  Members that fail to fit get no forecast and an infinite CV loss.

Reported (forecast_zoo_<feature_set>_<rule>_2026-11-10.csv and forecast_zoo_2026-11-10.md)
  Main result: AUTO (pooled) on Anthropic targets, both rules. Also: stack (pooled); AUTO and
  stack (own) as the pre-declared controls for pooling; every member's score, the spread of
  member scores, and the best member labelled optimistic (selected after the fact); references
  A claims only, E OpenRouter tokens x price, G GitHub new-repo share (forecast_eval.collect) and
  the naive last claim (forecast_backtest_<rule>), on common targets; H2-2026 errors; paired
  comparisons (forecast_eval.paired) of AUTO and stack against each reference and pooled vs own;
  OpenAI targets (AUTO, stack, members, the claims-only smoother and the naive last claim).
"""
from __future__ import annotations

import inspect
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import backtest
from . import features as F
from . import panel_model as pm

# --------------------------------------------------------------------------- availability
def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:          # noqa: BLE001 - any import failure means the member is dropped
        return False


HAS_LIGHTGBM = _has("lightgbm")
HAS_XGBOOST = _has("xgboost")

ALL_MEMBERS = ("trend", "ridge", "lasso", "enet", "pca", "pls", "bayes_ridge", "ard", "kernel_ridge",
               "gp", "knn", "rf", "extra_trees", "hgb", "lightgbm", "xgboost", "mlp", "midas")
DROPPED = tuple(m for m, ok in (("lightgbm", HAS_LIGHTGBM), ("xgboost", HAS_XGBOOST)) if not ok)
MEMBERS = tuple(m for m in ALL_MEMBERS if m not in DROPPED)
COMBOS = ("AUTO", "stack")
AUTO_FALLBACK = "trend"
RAW_MEMBERS = ("rf", "extra_trees", "hgb", "lightgbm", "xgboost")
SS_NAME = "SS claims only (panel labels)"
NAIVE_NAME = "naive last claim (panel)"

MIDAS_LAGS = 13
MIDAS_DEG = 2
N_TREES = 200

GRIDS = {
    "trend": tuple({"alpha": a} for a in pm.ALPHAS),
    "ridge": tuple({"alpha": a} for a in pm.ALPHAS),
    "lasso": tuple({"alpha": a} for a in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)),
    "enet": tuple({"alpha": a, "l1_ratio": r} for a in (0.003, 0.01, 0.03, 0.1) for r in (0.2, 0.5, 0.8)),
    "pca": tuple({"alpha": a, "k": k} for k in pm.PCA_K for a in pm.ALPHAS),
    "pls": tuple({"n_components": k} for k in (1, 2, 3)),
    "bayes_ridge": ({},),
    "ard": ({},),
    "kernel_ridge": tuple({"alpha": a, "gamma": g} for a in (0.1, 1.0, 10.0) for g in (0.01, 0.03, 0.1)),
    "gp": tuple({"s": s} for s in (0.1, 0.3, 1.0)),
    "knn": tuple({"k": k} for k in (5, 10, 20, 40)),
    "rf": tuple({"min_samples_leaf": m, "max_features": f} for m in (5, 20) for f in (0.5, 1.0)),
    "extra_trees": tuple({"min_samples_leaf": m, "max_features": f} for m in (5, 20) for f in (0.5, 1.0)),
    "hgb": pm.GBM_GRID,
    "lightgbm": tuple({"num_leaves": nl, "n_estimators": n} for nl in (3, 7) for n in (50, 200)),
    "xgboost": tuple({"max_depth": d, "n_estimators": n} for d in (2, 3) for n in (50, 200)),
    "mlp": tuple({"hidden": h, "alpha": a} for h in ((4,), (8,)) for a in (0.01, 1.0)),
    "midas": tuple({"alpha": a, "degree": d} for d in (1, 2) for a in pm.ALPHAS),
}
DEFAULT_HYPER = {m: dict(g[0]) for m, g in GRIDS.items()}
DEFAULT_HYPER.update({k: dict(v) for k, v in pm.DEFAULT_HYPER.items() if k in GRIDS})
DEFAULT_HYPER["hgb"] = dict(pm.DEFAULT_HYPER["gbm"])
SMALL_GRIDS = {m: g[:1] for m, g in GRIDS.items()}       # tests only


# --------------------------------------------------------------------------- feature sets
def _dup_of(col: str):
    """The level column whose window change duplicates this change column (None if none)."""
    if col == "openrouter_dlog_tokens":
        return "openrouter_log_tokens"
    return col[:-len("_chg")] if col.endswith("_chg") else None


def feature_spec(feature_set: str = "g_spec") -> dict:
    """{feature name: (asof_features column, "level" | "cumsum")} for a feature set (see the protocol)."""
    if feature_set == "g_spec":
        return dict(pm.FEATURE_SPEC)
    sets = getattr(F, "FEATURE_SETS", None)
    if not sets or feature_set not in sets:
        raise ValueError(f"feature set {feature_set!r} not defined in features.FEATURE_SETS "
                         f"(have: {sorted(sets) if sets else 'none'})")
    entry = sets[feature_set]
    if isinstance(entry, dict) and all(isinstance(v, (tuple, list)) and len(v) == 2 for v in entry.values()):
        return {k: (v[0], v[1]) for k, v in entry.items()}
    cols = list(entry)
    by_col = {col: (name, how) for name, (col, how) in pm.FEATURE_SPEC.items()}
    spec = {}
    for col in cols:
        if _dup_of(col) in cols:
            continue
        if col in by_col:
            name, how = by_col[col]
        elif "dlog" in col or col.endswith(("_chg", "_diff", "_growth", "_rate")):
            name, how = col + "_cum", "cumsum"
        else:
            name, how = col, "level"
        spec[name] = (col, how)
    return spec


def _asof(vintage, company, sources, feature_set):
    kw = {}
    if feature_set != "g_spec" and "feature_set" in inspect.signature(F.asof_features).parameters:
        kw["feature_set"] = feature_set
    return F.asof_features(vintage, company, sources=sources, **kw)


def feature_levels(company: str, vintage, sources: dict | None = None, feature_set: str = "g_spec") -> pd.DataFrame:
    """Weekly level-form features of a feature set; for "g_spec" identical to pm.feature_levels."""
    v = pd.Timestamp(vintage)
    spec = feature_spec(feature_set)
    f = _asof(v, company, sources, feature_set)
    if len(f) and f.index.max() > F._week_label([v])[0]:
        raise AssertionError("as-of features extend past the week of the vintage")
    out = pd.DataFrame(index=f.index)
    for name, (col, how) in spec.items():
        s = f[col].astype(float) if col in f else pd.Series(np.nan, index=f.index)
        if how == "cumsum":
            first, last = s.first_valid_index(), s.last_valid_index()
            lev = s.fillna(0.0).cumsum()
            if first is None:
                lev[:] = np.nan
            else:
                lev[(lev.index < first) | (lev.index > last + pm.FFILL_WEEKS * pm.WEEK)] = np.nan
            out[name] = lev
        else:
            out[name] = s.ffill(limit=pm.FFILL_WEEKS)
    return out


def midas_cols(feats) -> list[str]:
    return [f"midas:{f}:{p}" for f in feats for p in range(MIDAS_DEG + 1)]


_ALMON = np.array([[(j / MIDAS_LAGS) ** p for j in range(1, MIDAS_LAGS + 1)]
                   for p in range(MIDAS_DEG + 1)])     # (degree + 1) x lags 1..J


def _midas_values(Lv: np.ndarray, i0: int, i1: int, n: float) -> np.ndarray:
    """Almon terms for the window (row i0, row i1] of the level matrix Lv (weeks x features):
    m_j = (Lv[i1 - j] - Lv[i0 - j]) / n, Z_p = sum_{j=1..J} (j / J)^p m_j; NaN if any m_j is missing."""
    J = MIDAS_LAGS
    nf = Lv.shape[1]
    if i0 - J < 0:
        return np.full(nf * (MIDAS_DEG + 1), np.nan)
    js = np.arange(1, J + 1)
    m = (Lv[i1 - js] - Lv[i0 - js]) / n               # J x features
    Z = _ALMON @ m                                     # (deg+1) x features; NaN propagates
    return Z.T.reshape(-1)                             # feature-major: f0p0, f0p1, f0p2, f1p0, ...


# --------------------------------------------------------------------------- training pairs
def make_pairs(lab: pm.Labels, flev: pd.DataFrame, horizons=pm.HORIZONS, midas: bool = True) -> pd.DataFrame:
    """pm.make_pairs for the columns of flev, plus the MIDAS terms of each pair's window."""
    feats = list(flev.columns)
    cols = ["company", "t", "t1", "h", "y", "mom", "w_raw"] + feats + (midas_cols(feats) if midas else [])
    if not lab.ok or not len(lab.y) or not len(flev):
        return pd.DataFrame(columns=cols)
    y, sd = lab.y, lab.sd
    first_feat = flev.index.min()
    pos = {d: i for i, d in enumerate(flev.index)}
    Lv = flev.to_numpy(float)
    rows = []
    for t in y.index:
        if t < first_feat:
            continue
        for h in horizons:
            t1, tm = t + h * pm.WEEK, t - h * pm.WEEK
            if t1 not in y.index or tm not in y.index or t1 not in pos or t not in pos:
                continue
            x = (Lv[pos[t1]] - Lv[pos[t]]) / h
            w = h ** 2 / (sd[t] ** 2 + sd[t1] ** 2 + 2 * pm.LABEL_SD_FLOOR ** 2)
            r = [lab.company, t, t1, h, (y[t1] - y[t]) / h, (y[t] - y[tm]) / h, w] + list(x)
            if midas:
                r += list(_midas_values(Lv, pos[t], pos[t1], h))
            rows.append(r)
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- forecast rows
def growth_rows(lab: pm.Labels, flev: pd.DataFrame, dates, midas: bool = True) -> pd.DataFrame:
    """pm.growth_rows for the columns of flev (+ MIDAS terms of the window (week of t0, week of d])."""
    feats = list(flev.columns)
    pos = {d: i for i, d in enumerate(flev.index)}
    Lv = flev.to_numpy(float)
    w0 = F._week_label([lab.t0])[0]
    rows = []
    for d in dates:
        d = pd.Timestamp(d)
        g = (d - lab.t0).days / 7
        w1 = min(F._week_label([d])[0], flev.index.max())
        n = (w1 - w0).days / 7
        ok = n >= 1 and w0 in pos and w1 in pos
        x = (Lv[pos[w1]] - Lv[pos[w0]]) / n if ok else np.full(len(feats), np.nan)
        r = [lab.company, g, pm._momentum(lab, lab.t0, g)] + list(x)
        if midas:
            r += list(_midas_values(Lv, pos[w0], pos[w1], n) if ok else np.full(len(midas_cols(feats)), np.nan))
        rows.append(r)
    return pd.DataFrame(rows, columns=["company", "g", "mom"] + feats + (midas_cols(feats) if midas else []))


@dataclass
class ForecastInputs:
    target: object
    dates: list
    basis: str
    before: dict          # date -> label level (dates <= t0)
    after: list           # dates > t0
    G: pd.DataFrame       # growth rows for `after`
    start: float          # label level at t0 on the target's basis


def forecast_inputs(lab: pm.Labels, flev: pd.DataFrame, target) -> ForecastInputs:
    dates, basis = pm.target_dates(target)
    after = [d for d in dates if d > lab.t0]
    before = {d: lab.level(d, basis) for d in dates if d <= lab.t0}
    G = growth_rows(lab, flev, after) if after else None
    return ForecastInputs(target, dates, basis, before, after, G, lab.level(lab.t0, basis))


def path_value(fi: ForecastInputs, rate) -> float:
    """pm.path_forecast given the predicted weekly growth rate of each date after t0."""
    lv = dict(fi.before)
    if fi.after:
        for d, r, g in zip(fi.after, rate, fi.G["g"]):
            lv[d] = fi.start + float(r) * float(g)
    path = np.array([lv[d] for d in fi.dates])
    if np.isnan(path).any():
        return np.nan
    if fi.target.metric == "run_rate":
        return float(math.exp(path[0]))
    return float(np.exp(path).sum() / pm.config.DAYS_PER_YEAR)


def path_forecast(lab, flev, target, predict) -> float:
    fi = forecast_inputs(lab, flev, target)
    return path_value(fi, predict(fi.G) if fi.after else [])


# --------------------------------------------------------------------------- members
def _feats_of(P: pd.DataFrame) -> list[str]:
    fixed = {"company", "t", "t1", "h", "y", "mom", "w_raw", "w", "g"}
    return [c for c in P.columns if c not in fixed and not c.startswith("midas:")]


class Prep:
    """pm.LinearPanel's company fixed effect, missing -> 0 and RMS scaling (training rows only),
    for the columns ["mom"] + feats + MIDAS terms; plus y scaled by its weighted RMS."""

    def fit(self, P: pd.DataFrame, xcols: list[str]):
        self.xcols = list(xcols)
        w = P["w"].to_numpy(float)
        cols = ["y"] + self.xcols
        M = P[cols].to_numpy(float)
        allm = np.array([pm._wmean(M[:, j], w) for j in range(M.shape[1])])
        self.means_ = {}
        for c, g in P.groupby("company"):
            wc = g["w"].to_numpy(float)
            Mc = g[cols].to_numpy(float)
            mu = np.array([pm._wmean(Mc[:, j], wc) for j in range(Mc.shape[1])])
            self.means_[c] = np.nan_to_num(np.where(np.isnan(mu), allm, mu))
        self.means_["__all__"] = np.nan_to_num(allm)
        X = self._demean(P)
        sc = np.sqrt(np.sum(w[:, None] * X ** 2, axis=0) / np.sum(w))
        self.scale_ = np.where(sc > 1e-12, sc, 1.0)
        yd = P["y"].to_numpy(float) - self.base(P)
        ys = math.sqrt(float(np.sum(w * yd ** 2) / np.sum(w)))
        self.yscale_ = ys if ys > 1e-12 else 1.0
        return X / self.scale_, yd / self.yscale_, w

    def _demean(self, P):
        X = P[self.xcols].to_numpy(float).copy()
        for i, c in enumerate(P["company"].to_numpy()):
            X[i] -= self.means_.get(c, self.means_["__all__"])[1:]
        X[np.isnan(X)] = 0.0
        return X

    def transform(self, P):
        return self._demean(P) / self.scale_

    def base(self, P):
        return np.array([self.means_.get(c, self.means_["__all__"])[0] for c in P["company"]])


def _wpls_fit(Z, y, w, k):
    """Weighted PLS1 (NIPALS) without centering (the inputs are already centred within company).
    Returns coefficients b with prediction Z @ b."""
    X, r = Z.copy(), y.copy()
    Ws, Ps, qs = [], [], []
    for _ in range(k):
        wa = X.T @ (w * r)
        nrm = np.linalg.norm(wa)
        if nrm < 1e-12:
            break
        wa /= nrm
        t = X @ wa
        tt = float(np.sum(w * t * t))
        if tt < 1e-12:
            break
        p = X.T @ (w * t) / tt
        q = float(np.sum(w * r * t)) / tt
        X = X - np.outer(t, p)
        r = r - q * t
        Ws.append(wa), Ps.append(p), qs.append(q)
    if not Ws:
        return np.zeros(Z.shape[1])
    W, Pm, q = np.array(Ws).T, np.array(Ps).T, np.array(qs)
    return W @ np.linalg.solve(Pm.T @ W, q)


class _WKNN:
    def __init__(self, k):
        self.k = int(k)

    def fit(self, Z, y, w):
        self.Z, self.y, self.w = Z, y, w
        return self

    def predict(self, Zq):
        k = min(self.k, len(self.y))
        d2 = ((Zq[:, None, :] - self.Z[None, :, :]) ** 2).sum(axis=2)
        idx = np.argsort(d2, axis=1, kind="stable")[:, :k]
        ww, yy = self.w[idx], self.y[idx]
        return (ww * yy).sum(axis=1) / ww.sum(axis=1)


class LinearMember:
    """Members on the 'linear' preprocessing."""

    def __init__(self, kind: str, **hyp):
        self.kind, self.hyp = kind, hyp

    def _cols(self, P):
        feats = _feats_of(P)
        if self.kind == "trend":
            return ["mom"]
        if self.kind == "midas":
            deg = int(self.hyp["degree"])
            return ["mom"] + feats + [f"midas:{f}:{p}" for f in feats for p in range(deg + 1)]
        return ["mom"] + feats

    def fit(self, P: pd.DataFrame):
        from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, Lasso, Ridge
        self.prep_ = Prep()
        Z, y, w = self.prep_.fit(P, self._cols(P))
        k, h = self.kind, self.hyp
        self.components_ = None
        if k in ("trend", "ridge", "midas"):
            self.est_ = Ridge(alpha=h["alpha"], fit_intercept=False).fit(Z, y, sample_weight=w)
        elif k == "pca":
            mom, Xf = Z[:, :1], Z[:, 1:]
            ws = np.sqrt(w / np.mean(w))[:, None]
            comps = np.zeros((0, Xf.shape[1]))
            if np.any(Xf != 0):
                _, s, vt = np.linalg.svd(ws * Xf, full_matrices=False)
                comps = vt[:min(int(h["k"]), int(np.sum(s > 1e-10)))]
            self.components_ = comps
            self.est_ = Ridge(alpha=h["alpha"], fit_intercept=False).fit(np.hstack([mom, Xf @ comps.T]), y,
                                                                         sample_weight=w)
        elif k == "lasso":
            self.est_ = Lasso(alpha=h["alpha"], fit_intercept=False, max_iter=20000).fit(Z, y, sample_weight=w)
        elif k == "enet":
            self.est_ = ElasticNet(alpha=h["alpha"], l1_ratio=h["l1_ratio"], fit_intercept=False,
                                   max_iter=20000).fit(Z, y, sample_weight=w)
        elif k == "pls":
            self.coef_ = _wpls_fit(Z, y, w, int(h["n_components"]))
        elif k == "bayes_ridge":
            self.est_ = BayesianRidge(fit_intercept=False).fit(Z, y, sample_weight=w)
        elif k == "ard":
            sw = np.sqrt(w)
            self.est_ = ARDRegression(fit_intercept=False).fit(Z * sw[:, None], y * sw)
        elif k == "kernel_ridge":
            from sklearn.kernel_ridge import KernelRidge
            self.est_ = KernelRidge(alpha=h["alpha"], kernel="rbf", gamma=h["gamma"]).fit(Z, y, sample_weight=w)
        elif k == "gp":
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel
            kern = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(3.0, (0.3, 100.0))
            self.est_ = GaussianProcessRegressor(kernel=kern, alpha=h["s"] / w, normalize_y=False,
                                                 n_restarts_optimizer=0, random_state=0).fit(Z, y)
        elif k == "knn":
            self.est_ = _WKNN(h["k"]).fit(Z, y, w)
        elif k == "mlp":
            from sklearn.neural_network import MLPRegressor
            self.est_ = MLPRegressor(hidden_layer_sizes=tuple(h["hidden"]), alpha=h["alpha"], activation="tanh",
                                     solver="adam", learning_rate_init=0.01, max_iter=2000, early_stopping=True,
                                     validation_fraction=0.2, n_iter_no_change=20, random_state=0
                                     ).fit(Z, y, sample_weight=w)
        else:
            raise ValueError(k)
        return self

    def predict(self, P: pd.DataFrame) -> np.ndarray:
        Z = self.prep_.transform(P)
        if self.kind == "pca":
            Z = np.hstack([Z[:, :1], Z[:, 1:] @ self.components_.T])
        if self.kind == "pls":
            f = Z @ self.coef_
        else:
            f = np.asarray(self.est_.predict(Z), float).reshape(-1)
        return self.prep_.base(P) + self.prep_.yscale_ * f


class RawMember:
    """Tree members on the 'raw' design [is_anthropic, mom, feature changes] (as pm.GBMPanel)."""

    def __init__(self, kind: str, **hyp):
        self.kind, self.hyp = kind, hyp

    @staticmethod
    def _X(P):
        return np.column_stack([(P["company"] == "anthropic").to_numpy(float),
                                P[["mom"] + _feats_of(P)].to_numpy(float)])

    def fit(self, P: pd.DataFrame):
        X = self._X(P)
        self.cols_ = np.where(~np.all(np.isnan(X), axis=0))[0]
        X = X[:, self.cols_]
        y, w = P["y"].to_numpy(float), P["w"].to_numpy(float)
        k, h = self.kind, self.hyp
        if k == "hgb":
            from sklearn.ensemble import HistGradientBoostingRegressor
            est = HistGradientBoostingRegressor(max_depth=h["max_depth"], max_iter=h["max_iter"], **pm.GBM_FIXED)
        elif k in ("rf", "extra_trees"):
            from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
            C = RandomForestRegressor if k == "rf" else ExtraTreesRegressor
            est = C(n_estimators=N_TREES, min_samples_leaf=h["min_samples_leaf"], max_features=h["max_features"],
                    random_state=0, n_jobs=1)
        elif k == "lightgbm":
            import lightgbm
            est = lightgbm.LGBMRegressor(n_estimators=h["n_estimators"], num_leaves=h["num_leaves"],
                                         learning_rate=0.05, min_child_samples=15, subsample=1.0,
                                         colsample_bytree=1.0, reg_lambda=0.0, random_state=0, n_jobs=1,
                                         deterministic=True, force_row_wise=True, verbose=-1)
        elif k == "xgboost":
            import xgboost
            est = xgboost.XGBRegressor(n_estimators=h["n_estimators"], max_depth=h["max_depth"], learning_rate=0.05,
                                       min_child_weight=15, tree_method="hist", random_state=0, n_jobs=1)
        else:
            raise ValueError(k)
        self.est_ = est.fit(X, y, sample_weight=w)
        return self

    def predict(self, P: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.est_.predict(self._X(P)[:, self.cols_]), float).reshape(-1)


def build(member: str, hyper: dict):
    return (RawMember if member in RAW_MEMBERS else LinearMember)(member, **hyper)


# --------------------------------------------------------------------------- fitting with CV
@dataclass
class Fitted:
    member: str
    model: object = None
    hyper: dict = field(default_factory=dict)
    cv_loss: float = math.inf
    oof: np.ndarray | None = None
    n_folds: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.model is not None


def _fold_loss(P, va, pred):
    e = np.abs(P.loc[va, "y"].to_numpy(float) - pred)
    wv = P.loc[va, "w_raw"].to_numpy(float)
    return float(np.sum(wv * e) / np.sum(wv))


def fit_member(member: str, P: pd.DataFrame, folds, grids=None) -> Fitted:
    """CV over the member's grid on the training pairs, then a fit on all of them."""
    grid = (grids or GRIDS)[member]
    best, best_loss, best_oof = None, math.inf, None
    if folds:
        for hyp in grid:
            oof = np.full(len(P), np.nan)
            losses = []
            try:
                for tr, va in folds:
                    m = build(member, hyp).fit(pm._normalise(P.loc[tr]))
                    pv = m.predict(P.loc[va])
                    oof[np.asarray(va)] = pv
                    losses.append(_fold_loss(P, va, pv))
                loss = float(np.mean(losses))
            except Exception:          # noqa: BLE001 - a failing configuration is never selected
                loss = math.nan
            if np.isfinite(loss) and loss < best_loss - 1e-15:
                best, best_loss, best_oof = hyp, loss, oof
    if best is None:
        best, best_loss, best_oof = dict(DEFAULT_HYPER[member] if not grids else grid[0]), math.inf, None
    try:
        model = build(member, best).fit(pm._normalise(P))
    except Exception as e:             # noqa: BLE001
        return Fitted(member, None, dict(best), math.inf, None, len(folds), repr(e))
    return Fitted(member, model, dict(best), best_loss, best_oof, len(folds))


def fit_zoo(P: pd.DataFrame, members=MEMBERS, grids=None) -> dict[str, Fitted]:
    """Every member fit on the training pairs P with its own CV (same folds for all)."""
    P = P.reset_index(drop=True)
    folds = pm.cv_folds(P)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {m: fit_member(m, P, folds, grids) for m in members}


def auto_choice(fits: dict[str, Fitted], members=None) -> str:
    """AUTO: the member with the lowest inner-CV loss (ties: the first in member order); with no
    finite CV loss, the fallback member. Uses nothing but the members' CV losses on training pairs."""
    order = [m for m in (members or list(fits)) if m in fits and fits[m].ok]
    finite = [(fits[m].cv_loss, i, m) for i, m in enumerate(order) if np.isfinite(fits[m].cv_loss)]
    if not finite:
        return AUTO_FALLBACK if AUTO_FALLBACK in order else (order[0] if order else AUTO_FALLBACK)
    return min(finite)[2]


def stack_weights(fits: dict[str, Fitted], P: pd.DataFrame) -> dict[str, float]:
    """Non-negative weights summing to one on the members' OOF predictions (weighted least squares)."""
    from scipy.optimize import minimize
    names = [m for m, f in fits.items() if f.ok]
    if not names:
        return {}
    have = [m for m in names if fits[m].oof is not None]
    if not have:                                   # no usable fold: equal weights
        return {m: 1.0 / len(names) for m in names}
    names = have                                   # members without OOF predictions get weight 0
    M = np.column_stack([fits[m].oof for m in names])
    P = P.reset_index(drop=True)
    rows = np.all(np.isfinite(M), axis=1)
    if rows.sum() < 1:
        return {m: 1.0 / len(names) for m in names}
    M, y, w = M[rows], P["y"].to_numpy(float)[rows], P["w_raw"].to_numpy(float)[rows]
    w = w / w.mean()
    sc = float(np.sqrt(np.mean(y ** 2))) or 1.0
    M, y = M / sc, y / sc

    def obj(a):
        r = y - M @ a
        return float(np.sum(w * r * r)), -2.0 * M.T @ (w * r)

    k = len(names)
    res = minimize(obj, np.full(k, 1.0 / k), jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                   constraints=[{"type": "eq", "fun": lambda a: np.sum(a) - 1.0, "jac": lambda a: np.ones_like(a)}],
                   options={"ftol": 1e-12, "maxiter": 500})
    a = np.clip(res.x, 0.0, None)
    a = a / a.sum() if a.sum() > 0 else np.full(k, 1.0 / k)
    return {m: float(x) for m, x in zip(names, a)}


# --------------------------------------------------------------------------- one target
def forecast_target(target, vintage, claims: dict, sources: dict, feature_set: str = "g_spec",
                    members=MEMBERS, pools=None, params: dict | None = None, grids=None,
                    return_fits: bool = False):
    """All zoo forecasts (members, AUTO, stack, and two references) of one target at one vintage.

    pools: default ("pooled", "own") for Anthropic targets and ("pooled",) otherwise.
    params: {company: process noise} to skip refitting (tests only); grids: override (tests only).
    """
    v = pd.Timestamp(vintage)
    tc = target.company
    if pools is None:
        pools = ("pooled", "own") if tc == "anthropic" else ("pooled",)
    other = [c for c in pm.COMPANIES if c != tc]
    dates, _ = pm.target_dates(target)
    queries = [(target.period_start, target.period_end)] if target.metric == "revenue" else None
    labs = {tc: pm.company_labels(tc, claims[tc], v, exclude_origin=target.origin_id,
                                  params=(params or {}).get(tc), end=max(dates), queries=queries)}
    for c in other:
        labs[c] = pm.company_labels(c, claims[c], v, params=(params or {}).get(c))
    for c, lab_c in labs.items():
        backtest.check_no_leak(lab_c.sel, v, target)
    lab = labs[tc]
    base = {"target": target.obs_id, "company": tc, "metric": target.metric, "vintage": v.date(),
            "actual": float(target.value), "feature_set": feature_set}
    out, fits_out = [], {}
    if len(lab.anchors) < pm.MIN_ANCHORS or not lab.ok:
        return (out, fits_out) if return_fits else out
    gap = max((max(dates) - lab.t0).days / 7, 0.0)
    extra0 = {"n_anchors": len(lab.anchors), "t0": lab.t0.date(), "gap_weeks": round(gap, 2)}

    def row(model, pool, fc, **extra):
        le = math.log(target.value / fc) if fc and fc > 0 and not math.isnan(fc) else np.nan
        return {**base, "forecast": fc, "log_err": le, "model": model, "pool": pool, **extra0, **extra}

    out.append(row(SS_NAME, "own", pm.smoother_forecast(lab, target)))
    nv = backtest._naive(target, lab.sel)
    if nv is not None:
        out.append(row(NAIVE_NAME, "own", math.exp(nv) if target.metric == "run_rate" else nv))

    flevs = {c: feature_levels(c, v, sources, feature_set) for c in labs}
    pairs = {c: make_pairs(labs[c], flevs[c]) for c in labs}
    n_own = len(pairs[tc])
    if n_own < pm.MIN_TARGET_PAIRS:
        return (out, fits_out) if return_fits else out
    fi = forecast_inputs(lab, flevs[tc], target)
    for pool in pools:
        P = pd.concat([pairs[c] for c in ([tc] if pool == "own" else [tc] + other) if len(pairs[c])],
                      ignore_index=True)
        assert (P["t1"] <= v).all() and all((P[P.company == c]["t1"] <= labs[c].t0).all() for c in P.company.unique())
        fits = fit_zoo(P, members, grids)
        fits_out[pool] = fits
        rates = {}
        info = {"n_train_pairs": len(P), "n_target_pairs": n_own}
        for m, f in fits.items():
            if f.ok and fi.after:
                try:
                    rates[m] = np.asarray(f.model.predict(fi.G), float)
                except Exception:      # noqa: BLE001
                    rates[m] = None
            elif f.ok:
                rates[m] = np.array([])
            fc = path_value(fi, rates[m]) if rates.get(m) is not None else np.nan
            out.append(row(m, pool, fc, hyper=str(f.hyper), cv_loss=f.cv_loss, cv_folds=f.n_folds, **info))
        ok = {m: f for m, f in fits.items() if rates.get(m) is not None}
        choice = auto_choice(ok, list(members))
        fc_auto = path_value(fi, rates[choice]) if choice in rates and rates[choice] is not None else np.nan
        out.append(row("AUTO", pool, fc_auto, auto_choice=choice, hyper=str(fits[choice].hyper) if choice in fits else "",
                       cv_loss=fits[choice].cv_loss if choice in fits else np.nan, **info))
        wts = stack_weights(ok, P)
        if wts:
            r = sum(wts[m] * rates[m] for m in wts) if fi.after else np.array([])
            fc_stack = path_value(fi, r)
        else:
            fc_stack = np.nan
        out.append(row("stack", pool, fc_stack,
                       stack_weights=";".join(f"{m}={x:.3f}" for m, x in wts.items() if x >= 0.0005), **info))
    return (out, fits_out) if return_fits else out


# --------------------------------------------------------------------------- backtest
_WORKER: dict = {}


def _init_worker():
    _WORKER["claims"] = {c: pm.load_claims(c) for c in pm.COMPANIES}
    _WORKER["sources"] = F.load_sources()


def run_job(job, feature_set: str = "g_spec", claims=None, sources=None) -> list[dict]:
    company, obs_id, v = job
    warnings.simplefilter("ignore")
    if claims is None:
        if not _WORKER:
            _init_worker()
        claims, sources = _WORKER["claims"], _WORKER["sources"]
    import time
    t_start = time.time()
    t = pm.load_targets(company).set_index("obs_id").loc[obs_id]
    tgt = pm._Row(obs_id=obs_id, **t.to_dict())
    rows = forecast_target(tgt, v, claims, sources, feature_set=feature_set)
    secs = round(time.time() - t_start, 1)
    return [{**r, "job_seconds": secs} for r in rows]


def run_backtest(rule: str, feature_set: str = "g_spec", companies=pm.COMPANIES, n_jobs: int = 1,
                 verbose: bool = True) -> pd.DataFrame:
    jobs = pm.target_jobs(rule, companies)
    jobs = sorted(jobs, key=lambda j: j[0] != "anthropic")           # longest jobs (two pools) first
    rows = []
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker) as ex:
            for job, res in zip(jobs, ex.map(partial(run_job, feature_set=feature_set), jobs)):
                rows += res
                if verbose:
                    print(f"  {feature_set} {rule} {job[0]} {job[1]} done ({len(res)} rows)", flush=True)
    else:
        _init_worker()
        for job in jobs:
            res = run_job(job, feature_set)
            rows += res
            if verbose:
                print(f"  {feature_set} {rule} {job[0]} {job[1]} done ({len(res)} rows)", flush=True)
    out = pd.DataFrame(rows)
    if len(out):
        out.insert(0, "rule", rule)
    return out
