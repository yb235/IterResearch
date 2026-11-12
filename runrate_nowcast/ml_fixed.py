"""Corrected ML pipeline: ridge toward prior elasticities (M1) and its average with the claims-only path (M2).

The specification below was written before any backtest of this module was run and is not changed
afterwards to improve a score. Bug fixes found after the first run are reported separately (report),
with before and after numbers. It answers the five findings of reports/zoo_diagnostics_2026-11-12.md:
(1) scores mixed pre-t0 label paths with real forecasts, (2) the inner CV trained on hindsight labels,
(3) shrinkage toward zero erased the signals, (4) standardization over imputed zeros without clipping,
(5) crossing claims dated at rho 0.5 while A / E / G profile rho at the vintage.

SPECIFICATION (fixed 2026-11-12, before the first run)
======================================================

Reused unchanged from panel_model.py (its docstring): claims tables and their public-at-v selection
(pm.public_claims: published <= v, include, not a forecast, never a filing, never the target's origin),
the claims-only state-space labels (pm.company_labels: process noise refit on the claims public at the
vintage, label y(d) = smoothed l + b on Thursdays from the first anchor to the last anchor t0), anchors
(>= 3 to forecast, >= 2 to contribute pairs), training pairs (4 / 8 / 13-week weekly log growth y, `mom`
= own label growth over the previous h weeks, weekly feature change x_f, inverse-variance weight w,
normalised to mean 1 within each fit), forecasts (label path up to t0, then y(t0) + g * rate_hat; revenue
targets integrate the daily path; mom over the last clip(round(g), 4, 13) weeks before t0; feature rate
over (week of t0, last feature week <= week of d]), targets (pm.load_targets) and vintages
(backtest.target_vintage, rules publication and period_end). A target gets no M1 / M2 forecast when its
company has < 3 anchors or < 10 own training pairs.

1. Crossing dates (finding 5)
   At every vintage w at which labels are built (the forecast vintage v and every nested-CV fold vintage
   u), each company's crossing claims are re-dated at a crossing position rho profiled at w:
     Anthropic: rho = robust.best_crossing_position(claims, backtest.unfitted_params(), w,
                exclude_origin=<target origin if the company is the target's>), exactly the call in
                backtest.prepublication_forecast(crossing="profile"); `claims` is pm.load_claims (the A
                frame of forecast_statespace_ml.py; the filing rows it lacks are dropped by data.select
                there anyway).
     OpenAI:    the same grid (robust.RHOS), online-filter likelihood, flat prior and posterior mean, on
                OpenAI's claims public at w as pm.public_claims selects them, with Model(unfitted params,
                start = OpenAI's model start 2023-12-01). (Its claims start before Anthropic's model start;
                no reference model profiles OpenAI, this is the same rule on that company's own claims.)
   Claims are re-dated with data.with_crossing_position(claims, rho) before pm.company_labels; the
   target's t_eff / ref_date are re-derived at the target company's rho at v (as prepublication_forecast
   re-derives its row). Everything downstream (t0, label path, target dates) uses the re-dated values.
   [Detail added before the first backtest, found by the unit tests: when a company has no claim public
   at w (early nested-CV fold vintages), rho = 0.5; no labels can be built then, so it changes nothing.]

2. Features (finding 4)
   The 8 g_spec level series of pm.FEATURE_SPEC, built as pm.feature_levels (from
   features.asof_features(v, company, feature_set="g_spec"), identical values), plus three revenue-proxy
   transforms of the same as-of levels:
     or_spend          openrouter_log_tokens + log_price (Anthropic). OpenAI has no price series
                       (features.ENTITIES[...]["price"] is False): or_spend = openrouter_log_tokens there
                       (price treated as constant), as in the diagnostics' D1 `or_spend`.
     log_gh_share      log(github_repo_share)
     log_ramp_adoption log(ramp_adoption)
   So 11 features; pair inputs are their weekly changes x_f, plus mom.
   Preprocessing (fit on the training rows of each fit only; "fixed" prep):
     a. Drop: a feature is dropped from the fit if fewer than 26 distinct pair start weeks t among the
        fit's training rows (any company in the pool) have it observed (non-NaN x_f). mom is never dropped.
     b. Clip: for mom and each kept feature, lo / hi = 1st / 99th percentile (numpy linear interpolation,
        unweighted) of its observed training values (all companies in the fit). Training values and
        prediction inputs are clipped to [lo, hi]; NaN stays NaN.
     c. Company effect: weighted (w) mean of y and of every clipped input per company over that company's
        observed rows (fallback for a company with no observed value: the mean over all observed rows);
        y and inputs are demeaned within company.
     d. Standardize: scale_j = sqrt(sum w * xd^2 / sum w) over the observed rows only (all companies);
        scale_j = 1 if below 1e-12. z = xd / scale_j; a missing input is then set to z = 0 (company mean).
   Prediction: clip, demean with the company's training means, divide by the scales, NaN -> 0;
   rate_hat = company mean of y + z . gamma.

3. M1, ridge toward prior elasticities (finding 3)
   gamma = argmin sum_i w_i (yd_i - z_i . gamma)^2 + lambda * sum_j (gamma_j - gamma0_j)^2, solved in
   closed form: gamma = (Z'WZ + lambda I)^-1 (Z'W yd + lambda gamma0), w normalised to mean 1.
   Prior means in original units (weekly log growth per unit of weekly input change): beta0 = 1.0 for
   or_spend, log_gh_share and log_ramp_adoption (the catalog prior, indicator_catalog.csv: beta 1 for
   openrouter_anthropic_tokens, github_repo_share_claude, ramp_adoption_anthropic); 0 for the other 8
   features and for mom. In standardized units gamma0_j = beta0_j * scale_j. One lambda for all
   coefficients. lambda grid LAMBDAS = (1e6, 1e5, 1e4, 1e3, 1e2, 10, 1, 0.1) (ordered strongest first);
   chosen by the nested CV below; ties (within 1e-12) -> the larger lambda; no usable fold -> 100.

4. Nested-vintage CV (finding 2)
   Blocks: the distinct start weeks t of the TARGET company's vintage-v pairs, cut into 5 consecutive
   blocks (np.array_split); validate on blocks 2..5. For a block whose first start week is b0:
     * fold vintage u = b0 - 1 day: for each company in the pool, labels are rebuilt from the claims
       published on or before u (i.e. before the block starts), with rho profiled at u (item 1) and
       process noise refit at u; features are features as of u; pairs as above (pm rules).
     * purge: of those pairs, only the ones with t1 <= b0 - 13 weeks are training rows (>= 13 weeks
       between the end of every training window and the start of the validation block, so no validation
       pair's mom window overlaps a training window).
     * validation rows: the target company's vintage-v pairs (labels at v, features as of v) whose start
       week t is in the block.
     * a fold is usable with >= 20 training rows, >= 10 of them from the target company, and >= 5
       validation rows.
   Loss of a lambda = w_raw-weighted mean absolute error of y on the validation rows, averaged over the
   usable folds; the preprocessing (item 2) is refit on each fold's training rows. The pooled and own
   fits share the same blocks and fold vintages.

5. M2 = equal-weight average, in logs, of M1 and the claims-only state-space path forecast:
   M2 = exp(0.5 log M1 + 0.5 log SS), SS = pm.smoother_forecast of the target company's own label run at v
   (claims-only smoother at the profiled rho, process noise refit at v; written as "SS claims-only path").

6. Pools: "pooled" (Anthropic + OpenAI pairs) is primary; "own" (target company only) is the control,
   for Anthropic targets. OpenAI targets: pooled only.

7. Scoring
   Error = log(actual / forecast). A target is post-t0 when its last date (run-rate t_eff, or the
   revenue period end, at the profiled rho) is after t0 (the target company's last anchor public at v);
   before t0 every ML model returns the label path. Primary: mean |log error| of M1 and M2 (pooled) on the
   post-t0 Anthropic targets, each rule. Also reported: all targets. Comparisons on the targets that M1,
   M2, A claims only, E OpenRouter tokens x price, G GitHub new-repo share (forecast_eval.collect), the
   naive last claim (forecast_backtest_<rule>) and the old zoo AUTO (pooled, g_spec,
   forecast_zoo_g_spec_<rule>) all forecast; paired comparisons (forecast_eval.paired) of M1 / M2 vs E,
   G, A and old AUTO on post-t0 and on all targets; pooled vs own. Second test set: OpenAI claims, M1 / M2
   (pooled) vs SS, the panel's naive last claim and old AUTO (pooled).

8. Ablations (declared in advance; diagnostics only, never candidates, never used to choose anything):
     "M1 prior 0"   beta0 = 0 for every input (ridge toward zero); else as M1, nested CV.
     "M1 inner CV"  lambda by the old inner CV: pm.cv_folds on the pool's vintage-v pairs (hindsight
                    labels, validation rows of all companies, same loss); else as M1.
     "M1 old prep"  pm.LinearPanel's preprocessing: no drop, no clip, missing -> company mean, scale over
                    all training rows including the imputed ones; prior means scaled by those scales;
                    nested CV.
     "M1 rho 0.5"   claims, labels and target date at rho 0.5 (both companies); lambda fixed at M1's
                    nested-CV choice for that pool; else as M1. Also "SS claims-only path rho 0.5".
     "M1 prior 1/3" beta0 = 1/3 on each of the three proxies (the prior then says revenue growth equals
                    their average, not their sum); else as M1, nested CV.
                    [Added before the first backtest, after checking the inputs of one forecast
                    (rr_2026_09_s1a, publication; no error or actual value looked at): with beta0 = 1 on all
                    three proxies the prior adds up their three deviations from the training means
                    (-0.134, -0.043, -0.015 per week there), where the state-space models treat each
                    indicator as a separate measurement of one level. M1 itself is kept as specified.]
   Each M1 / ablation row also records the mean contributions after t0 (weekly growth) of the company
   mean (rate_base), mom and each proxy, from the fitted coefficients.

Outputs: reports/forecast_mlfixed_<rule>_2026-11-10.csv (target, company, metric, vintage, actual,
forecast, log_err, model, pool, post_t0, then diagnostics) and reports/forecast_mlfixed_2026-11-10.md,
written by scripts/forecast_ml_fixed.py (4 worker processes, every fit single-threaded).
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import backtest, data, robust
from . import features as F
from . import model_zoo as mz
from . import panel_model as pm
from .model import Model

COMPANIES = pm.COMPANIES
WEEK = pm.WEEK
PRIOR_ELASTICITY = {"or_spend": 1.0, "log_gh_share": 1.0, "log_ramp_adoption": 1.0}
FEATURES = list(pm.FEATURES) + list(PRIOR_ELASTICITY)
LAMBDAS = (1e6, 1e5, 1e4, 1e3, 1e2, 10.0, 1.0, 0.1)          # strongest first: ties -> the larger lambda
DEFAULT_LAMBDA = 100.0
TIE_TOL = 1e-12
PURGE_WEEKS = 13
MIN_FEATURE_WEEKS = 26
CLIP_PCT = (1.0, 99.0)
CV_BLOCKS = 5
CV_MIN_TRAIN = 20
CV_MIN_TRAIN_TARGET = 10
CV_MIN_VALID = 5

M1, M2 = "M1", "M2"
SS_NAME = "SS claims-only path"
NAIVE_NAME = "naive last claim (panel)"
ABL_PRIOR0, ABL_INNER, ABL_OLDPREP, ABL_RHO05 = "M1 prior 0", "M1 inner CV", "M1 old prep", "M1 rho 0.5"
ABL_PRIOR3 = "M1 prior 1/3"
SS_RHO05 = "SS claims-only path rho 0.5"
ABLATIONS = (ABL_PRIOR0, ABL_INNER, ABL_OLDPREP, ABL_RHO05, ABL_PRIOR3, SS_RHO05)
CONTRIB = ("mom", "or_spend", "log_gh_share", "log_ramp_adoption")
CSV_COLS = ["target", "company", "metric", "vintage", "actual", "forecast", "log_err", "model", "pool", "post_t0",
            "rule", "t0", "gap_weeks", "t_eff", "rho", "lam", "n_folds", "cv_curve", "feats_dropped",
            "beta_mom", "beta_or_spend", "beta_log_gh_share", "beta_log_ramp_adoption", "rate_base", "rate_hat",
            "contrib_mom", "contrib_or_spend", "contrib_log_gh_share", "contrib_log_ramp_adoption",
            "n_train_pairs", "n_target_pairs", "n_anchors", "job_seconds"]


# --------------------------------------------------------------------------- crossing position (spec 1)
_RHO_CACHE: dict = {}


def clear_caches() -> None:
    """Drop the in-process caches (rho profiles and process-noise fits); tests use it."""
    _RHO_CACHE.clear()
    pm._NOISE_CACHE.clear()


def _profile_rho(company: str, claims: pd.DataFrame, w: pd.Timestamp, exclude_origin=None,
                 rhos=robust.RHOS) -> float:
    """robust.crossing_profile's rule (online-filter loglik per rho, flat prior, posterior mean) on the
    company's claims as pm.public_claims selects them, with the company's model start."""
    start = pm.MODEL_STARTS[company]
    ll = []
    for rho in rhos:
        sel = pm.public_claims(data.with_crossing_position(claims, rho), w, exclude_origin, start=start)
        ll.append(Model(backtest.unfitted_params(), start=start).run(sel, smooth=False).loglik)
    ll = np.asarray(ll, float)
    wt = np.exp(ll - ll.max())
    return float(np.dot(np.asarray(rhos, float), wt / wt.sum()))


def crossing_rho(company: str, claims: pd.DataFrame, vintage, exclude_origin=None) -> float:
    """Crossing position profiled on the company's claims public at the vintage (spec item 1)."""
    w = pd.Timestamp(vintage)
    key = (company, exclude_origin, tuple(sorted(claims.loc[claims.published <= w, "obs_id"].astype(str))))
    if key not in _RHO_CACHE:
        sel = data.select(claims, vintage=w)
        if exclude_origin is not None:
            sel = sel[sel.origin_id != exclude_origin]
        if not len(sel):                     # nothing public yet: no labels either (spec item 1)
            _RHO_CACHE[key] = 0.5
        elif company == "anthropic":           # exactly backtest.prepublication_forecast(crossing="profile")
            _RHO_CACHE[key] = float(robust.best_crossing_position(claims, backtest.unfitted_params(), w,
                                                                  exclude_origin=exclude_origin))
        else:
            _RHO_CACHE[key] = _profile_rho(company, claims, w, exclude_origin)
    return _RHO_CACHE[key]


_TABLES: dict = {}


def target_at_rho(target, rho: float):
    """The target with t_eff / ref_date re-derived at crossing position rho (as prepublication_forecast)."""
    c = target.company
    if c not in _TABLES:
        _TABLES[c] = data.load(pm.CLAIMS_PATHS[c])
    d = data.with_crossing_position(_TABLES[c], rho)
    r = d[d.obs_id == target.obs_id].iloc[0]
    return pm._Row(**{**target.__dict__, "t_eff": pd.Timestamp(data.effective_date(r)),
                      "ref_date": pd.Timestamp(r.ref_date)})


def labels_at(company: str, claims: pd.DataFrame, vintage, exclude_origin=None, params=None, end=None,
              queries=None, rho: float | None = None) -> pm.Labels:
    """pm.company_labels on the claims re-dated at rho (profiled at the vintage unless given)."""
    if rho is None:
        rho = crossing_rho(company, claims, vintage, exclude_origin)
    lab = pm.company_labels(company, data.with_crossing_position(claims, rho), vintage, exclude_origin=exclude_origin,
                            params=params, end=end, queries=queries)
    lab.rho = float(rho)
    return lab


# --------------------------------------------------------------------------- features (spec 2)
def feature_levels(company: str, vintage, sources: dict | None = None) -> pd.DataFrame:
    """pm.FEATURE_SPEC levels (as pm.feature_levels) plus the three revenue-proxy transforms."""
    v = pd.Timestamp(vintage)
    f = F.asof_features(v, company, sources=sources, feature_set="g_spec")
    if len(f) and f.index.max() > F._week_label([v])[0]:
        raise AssertionError("as-of features extend past the week of the vintage")
    out = pd.DataFrame(index=f.index)
    for name, (col, how) in pm.FEATURE_SPEC.items():
        s = f[col].astype(float)
        if how == "cumsum":
            first, last = s.first_valid_index(), s.last_valid_index()
            lev = s.fillna(0.0).cumsum()
            if first is None:
                lev[:] = np.nan
            else:
                lev[(lev.index < first) | (lev.index > last + pm.FFILL_WEEKS * WEEK)] = np.nan
            out[name] = lev
        else:
            out[name] = s.ffill(limit=pm.FFILL_WEEKS)
    tok = out["openrouter_log_tokens"]
    out["or_spend"] = tok + out["log_price"] if F.ENTITIES[company]["price"] else tok
    gh, ramp = out["github_repo_share"], out["ramp_adoption"]
    out["log_gh_share"] = np.log(gh.where(gh > 0))
    out["log_ramp_adoption"] = np.log(ramp.where(ramp > 0))
    return out[FEATURES]


def make_pairs(lab: pm.Labels, flev: pd.DataFrame) -> pd.DataFrame:
    """pm's training pairs (y, mom, w_raw, weekly feature changes) for the FEATURES columns."""
    return mz.make_pairs(lab, flev, midas=False)


# --------------------------------------------------------------------------- preprocessing (spec 2)
class Prep:
    """kind "fixed": drop short features, clip to the training 1st-99th percentile, company means and
    scales on observed rows only, missing -> 0 after standardizing. kind "old": pm.LinearPanel's
    preprocessing (no drop, no clip, missing -> company mean, scale over all rows)."""

    def __init__(self, kind: str = "fixed"):
        if kind not in ("fixed", "old"):
            raise ValueError(kind)
        self.kind = kind

    def fit(self, P: pd.DataFrame):
        w = P["w"].to_numpy(float)
        feats = [f for f in FEATURES if f in P.columns]
        if self.kind == "fixed":
            feats = [f for f in feats if P.loc[P[f].notna(), "t"].nunique() >= MIN_FEATURE_WEEKS]
        self.feats = feats
        self.dropped = [f for f in FEATURES if f in P.columns and f not in feats]
        self.cols = ["mom"] + feats
        X = P[self.cols].to_numpy(float)
        if self.kind == "fixed":
            self.lo = np.array([np.nanpercentile(X[:, j], CLIP_PCT[0]) for j in range(X.shape[1])])
            self.hi = np.array([np.nanpercentile(X[:, j], CLIP_PCT[1]) for j in range(X.shape[1])])
            X = np.clip(X, self.lo, self.hi)
        M = np.column_stack([P["y"].to_numpy(float), X])
        allm = np.array([pm._wmean(M[:, j], w) for j in range(M.shape[1])])
        self.means_ = {}
        comp = P["company"].to_numpy()
        for c in pd.unique(comp):
            m = comp == c
            mu = np.array([pm._wmean(M[m, j], w[m]) for j in range(M.shape[1])])
            self.means_[c] = np.nan_to_num(np.where(np.isnan(mu), allm, mu))
        self.means_["__all__"] = np.nan_to_num(allm)
        Xd = X - self._mu(comp)
        obs = ~np.isnan(Xd)
        if self.kind == "fixed":
            num = np.sum(np.where(obs, w[:, None] * Xd ** 2, 0.0), axis=0)
            den = np.sum(np.where(obs, w[:, None], 0.0), axis=0)
            sc = np.sqrt(np.divide(num, den, out=np.zeros_like(num), where=den > 0))
        else:
            X0 = np.where(obs, Xd, 0.0)
            sc = np.sqrt(np.sum(w[:, None] * X0 ** 2, axis=0) / np.sum(w))
        self.scale_ = np.where(sc > 1e-12, sc, 1.0)
        return self

    def _mu(self, comp) -> np.ndarray:
        return np.array([self.means_.get(c, self.means_["__all__"])[1:] for c in comp]).reshape(len(comp), -1)

    def clip(self, X: np.ndarray) -> np.ndarray:
        return np.clip(X, self.lo, self.hi) if self.kind == "fixed" else X

    def Z(self, P: pd.DataFrame) -> np.ndarray:
        X = self.clip(P[self.cols].to_numpy(float))
        Z = (X - self._mu(P["company"].to_numpy())) / self.scale_
        Z[np.isnan(Z)] = 0.0
        return Z

    def base(self, P: pd.DataFrame) -> np.ndarray:
        return np.array([self.means_.get(c, self.means_["__all__"])[0] for c in P["company"]])

    def prior_gamma(self, prior: float = 1.0) -> np.ndarray:
        """Prior means in standardized units: prior * beta0 * scale (mom and non-proxy features: 0);
        prior 1 = M1, 0 = ridge toward zero, 1/3 = the "prior 1/3" ablation."""
        beta0 = np.array([0.0] + [PRIOR_ELASTICITY.get(f, 0.0) * float(prior) for f in self.feats])
        return beta0 * self.scale_


def _normal_eq(Z, yd, w):
    return Z.T @ (w[:, None] * Z), Z.T @ (w * yd)


def solve_prior_ridge(A0, b0, lam: float, g0) -> np.ndarray:
    """gamma = (Z'WZ + lam I)^-1 (Z'W yd + lam gamma0)."""
    return np.linalg.solve(A0 + lam * np.eye(len(b0)), b0 + lam * np.asarray(g0, float))


class PriorRidge:
    """M1: ridge toward prior elasticities with a company effect (spec item 3)."""

    def __init__(self, lam: float, prior: float = 1.0, prep: str = "fixed"):
        self.lam, self.prior, self.prep = float(lam), float(prior), prep

    def fit(self, P: pd.DataFrame):
        Pn = pm._normalise(P.reset_index(drop=True))
        self.prep_ = Prep(self.prep).fit(Pn)
        Z = self.prep_.Z(Pn)
        yd = Pn["y"].to_numpy(float) - self.prep_.base(Pn)
        A0, b0 = _normal_eq(Z, yd, Pn["w"].to_numpy(float))
        self.gamma0_ = self.prep_.prior_gamma(self.prior)
        self.coef_ = solve_prior_ridge(A0, b0, self.lam, self.gamma0_)
        self.n_train_ = len(P)
        return self

    def predict(self, G: pd.DataFrame) -> np.ndarray:
        return self.prep_.base(G) + self.prep_.Z(G) @ self.coef_

    def elasticities(self) -> dict:
        """Coefficients in original units (weekly log growth per unit of weekly input change)."""
        return {c: float(g / s) for c, g, s in zip(self.prep_.cols, self.coef_, self.prep_.scale_)}

    def contributions(self, G: pd.DataFrame, company: str) -> dict:
        """Mean over the rows of G of the company mean and of each CONTRIB input's term (weekly growth)."""
        Z = self.prep_.Z(G) * self.coef_[None, :]
        out = {"rate_base": float(self.prep_.means_.get(company, self.prep_.means_["__all__"])[0])}
        for k in CONTRIB:
            out[f"contrib_{k}"] = float(Z[:, self.prep_.cols.index(k)].mean()) if k in self.prep_.cols else np.nan
        return out


# --------------------------------------------------------------------------- CV (spec 3, 4)
def cv_curve(folds, prior: float = 1.0, prep: str = "fixed", lambdas=LAMBDAS):
    """Mean over folds of the w_raw-weighted MAE of y on the validation rows, per lambda (None: no fold)."""
    if not folds:
        return None
    L = []
    for Ptr, Pva in folds:
        Pn = pm._normalise(Ptr.reset_index(drop=True))
        pr = Prep(prep).fit(Pn)
        Z = pr.Z(Pn)
        A0, b0 = _normal_eq(Z, Pn["y"].to_numpy(float) - pr.base(Pn), Pn["w"].to_numpy(float))
        g0 = pr.prior_gamma(prior)
        Zv, bv = pr.Z(Pva), pr.base(Pva)
        yv, wv = Pva["y"].to_numpy(float), Pva["w_raw"].to_numpy(float)
        L.append([float(np.sum(wv * np.abs(yv - bv - Zv @ solve_prior_ridge(A0, b0, lam, g0))) / np.sum(wv))
                  for lam in lambdas])
    return np.mean(np.array(L), axis=0)


def choose_lambda(curve, lambdas=LAMBDAS) -> float:
    if curve is None or not np.isfinite(curve).any():
        return DEFAULT_LAMBDA
    m = np.nanmin(curve)
    return float(lambdas[int(np.flatnonzero(curve <= m + TIE_TOL)[0])])


@dataclass
class Fold:
    b0: pd.Timestamp                 # first start week of the validation block
    u: pd.Timestamp                  # fold vintage: b0 - 1 day
    block: np.ndarray                # the block's start weeks
    train: dict                      # company -> purged training pairs from labels / features as of u
    rho: dict


def fold_blocks(P_target: pd.DataFrame):
    """(b0, block start weeks) for blocks 2..5 of the target company's vintage-v pair start weeks."""
    starts = np.sort(P_target["t"].unique())
    if len(starts) < CV_BLOCKS:
        return []
    return [(pd.Timestamp(b[0]), b) for b in np.array_split(starts, CV_BLOCKS)[1:]]


def nested_folds(target, P_target: pd.DataFrame, claims: dict, sources: dict, companies, params=None) -> list[Fold]:
    """Training pairs of each fold rebuilt from what was public before the block starts (spec item 4)."""
    params = params or {}
    out = []
    for b0, block in fold_blocks(P_target):
        u = b0 - pd.Timedelta(days=1)
        train, rhos = {}, {}
        for c in companies:
            ex = target.origin_id if c == target.company else None
            lab = labels_at(c, claims[c], u, exclude_origin=ex, params=params.get(c))
            if len(lab.sel) and (pd.to_datetime(lab.sel.published) > u).any():
                raise AssertionError("fold labels use claims published after the fold vintage")
            pr = make_pairs(lab, feature_levels(c, u, sources))
            if len(pr):
                pr = pr[pr["t1"] <= b0 - PURGE_WEEKS * WEEK]
            train[c], rhos[c] = pr.reset_index(drop=True), lab.rho
        out.append(Fold(b0, u, block, train, rhos))
    return out


def pool_folds(folds: list[Fold], tc: str, P_target: pd.DataFrame, companies) -> list[tuple]:
    """(training rows, validation rows) of the usable folds for a pool."""
    res = []
    for f in folds:
        parts = [f.train[c] for c in companies if len(f.train.get(c, ()))]
        if not parts:
            continue
        Ptr = pd.concat(parts, ignore_index=True)
        Pva = P_target[P_target["t"].isin(f.block)]
        if len(Ptr) >= CV_MIN_TRAIN and int((Ptr["company"] == tc).sum()) >= CV_MIN_TRAIN_TARGET \
                and len(Pva) >= CV_MIN_VALID:
            res.append((Ptr, Pva))
    return res


def inner_folds(P: pd.DataFrame) -> list[tuple]:
    """The old inner CV (ablation): pm.cv_folds on the vintage-v pairs, all companies' validation rows."""
    P = P.reset_index(drop=True)
    return [(P.loc[tr], P.loc[va]) for tr, va in pm.cv_folds(P)]


# --------------------------------------------------------------------------- forecasts
def forecast_inputs(lab: pm.Labels, flev: pd.DataFrame, target) -> mz.ForecastInputs:
    dates, basis = pm.target_dates(target)
    after = [d for d in dates if d > lab.t0]
    before = {d: lab.level(d, basis) for d in dates if d <= lab.t0}
    G = mz.growth_rows(lab, flev, after, midas=False) if after else None
    return mz.ForecastInputs(target, dates, basis, before, after, G, lab.level(lab.t0, basis))


def path_forecast(model, fi: mz.ForecastInputs) -> tuple[float, float]:
    """(forecast, mean predicted weekly growth after t0)."""
    if not fi.after:
        return mz.path_value(fi, []), np.nan
    rate = np.asarray(model.predict(fi.G), float)
    return mz.path_value(fi, rate), float(rate.mean())


def pool_companies(pool: str, tc: str) -> list[str]:
    return [tc] if pool == "own" else [tc] + [c for c in COMPANIES if c != tc]


def _pool_pairs(pairs: dict, cs) -> pd.DataFrame:
    return pd.concat([pairs[c] for c in cs if len(pairs[c])], ignore_index=True)


def _fit_cols(m: PriorRidge, fi: mz.ForecastInputs, company: str) -> dict:
    """Dropped features, elasticities (original units) and contributions after t0 of a fitted model."""
    el = m.elasticities()
    out = {"feats_dropped": ";".join(m.prep_.dropped), **{f"beta_{k}": el.get(k, np.nan) for k in CONTRIB}}
    if fi.after:
        out.update(m.contributions(fi.G, company))
    return out


def _fmt_curve(curve) -> str:
    return "" if curve is None else ";".join(f"{lam:g}={x:.5f}" for lam, x in zip(LAMBDAS, curve))


def forecast_target(target, vintage, claims: dict, sources: dict, params: dict | None = None,
                    ablations: bool = True, pools=None) -> list[dict]:
    """M1, M2 (each pool), the claims-only path, the panel naive and the ablations for one target.

    claims: {company: pm.load_claims(company)}; sources: features.load_sources().
    params: {company: process noise} to skip refitting (tests only).
    """
    v = pd.Timestamp(vintage)
    tc = target.company
    params = params or {}
    pools = pools or (("pooled", "own") if tc == "anthropic" else ("pooled",))
    others = [c for c in COMPANIES if c != tc]
    rho = {tc: crossing_rho(tc, claims[tc], v, target.origin_id)}
    rho.update({c: crossing_rho(c, claims[c], v) for c in others})
    tg = target_at_rho(target, rho[tc])
    dates, _ = pm.target_dates(tg)
    queries = [(tg.period_start, tg.period_end)] if tg.metric == "revenue" else None
    labs = {tc: labels_at(tc, claims[tc], v, target.origin_id, params.get(tc), end=max(dates), queries=queries,
                          rho=rho[tc])}
    labs.update({c: labels_at(c, claims[c], v, None, params.get(c), rho=rho[c]) for c in others})
    for lab_c in labs.values():
        backtest.check_no_leak(lab_c.sel, v, tg)
    lab = labs[tc]
    base = {"target": target.obs_id, "company": tc, "metric": target.metric, "vintage": v.date(),
            "actual": float(target.value)}
    out: list[dict] = []
    if len(lab.anchors) < pm.MIN_ANCHORS or not lab.ok:
        return out
    last = max(dates)
    head = {"post_t0": bool(last > lab.t0), "t0": lab.t0.date(), "gap_weeks": round(max((last - lab.t0).days / 7, 0.0), 2),
            "t_eff": pd.Timestamp(tg.t_eff).date(), "rho": rho[tc], "n_anchors": len(lab.anchors)}

    def row(model, pool, fc, **extra):
        le = math.log(target.value / fc) if fc and fc > 0 and np.isfinite(fc) else np.nan
        return {**base, "forecast": fc, "log_err": le, "model": model, "pool": pool, **head, **extra}

    ss = pm.smoother_forecast(lab, tg)
    out.append(row(SS_NAME, "own", ss))
    nv = backtest._naive(tg, lab.sel)
    if nv is not None:
        out.append(row(NAIVE_NAME, "own", math.exp(nv) if tg.metric == "run_rate" else nv))

    flevs = {c: feature_levels(c, v, sources) for c in labs}
    pairs = {c: make_pairs(labs[c], flevs[c]) for c in labs}
    n_own = len(pairs[tc])
    if n_own < pm.MIN_TARGET_PAIRS:
        return out
    fi = forecast_inputs(lab, flevs[tc], tg)
    need = list(dict.fromkeys(c for p in pools for c in pool_companies(p, tc)))
    folds = nested_folds(target, pairs[tc], claims, sources, need, params)
    lam_m1 = {}
    for pool in pools:
        cs = pool_companies(pool, tc)
        P = _pool_pairs(pairs, cs)
        assert (P["t1"] <= v).all() and all((P[P.company == c]["t1"] <= labs[c].t0).all() for c in P.company.unique())
        info = {"n_train_pairs": len(P), "n_target_pairs": n_own}
        pf = pool_folds(folds, tc, pairs[tc], cs)
        curve = cv_curve(pf, prior=1.0, prep="fixed")
        lam = choose_lambda(curve)
        m1 = PriorRidge(lam).fit(P)
        fc1, rate = path_forecast(m1, fi)
        fit_info = {"lam": lam, "n_folds": len(pf), "cv_curve": _fmt_curve(curve), **_fit_cols(m1, fi, tc),
                    "rate_hat": rate, **info}
        out.append(row(M1, pool, fc1, **fit_info))
        fc2 = math.exp(0.5 * math.log(fc1) + 0.5 * math.log(ss)) if fc1 > 0 and ss > 0 else np.nan
        out.append(row(M2, pool, fc2, **fit_info))
        lam_m1[pool] = lam
        if not ablations:
            continue
        for name, kw, fl in ((ABL_PRIOR0, {"prior": 0.0, "prep": "fixed"}, pf),
                             (ABL_INNER, {"prior": 1.0, "prep": "fixed"}, inner_folds(P)),
                             (ABL_OLDPREP, {"prior": 1.0, "prep": "old"}, pf),
                             (ABL_PRIOR3, {"prior": 1.0 / 3.0, "prep": "fixed"}, pf)):
            c_ = cv_curve(fl, **kw)
            l_ = choose_lambda(c_)
            m = PriorRidge(l_, **kw).fit(P)
            fc, r_ = path_forecast(m, fi)
            out.append(row(name, pool, fc, lam=l_, n_folds=len(fl), cv_curve=_fmt_curve(c_), **_fit_cols(m, fi, tc),
                           rate_hat=r_, **info))
    if ablations:
        out += _rho05_rows(target, v, claims, sources, params, pools, lam_m1, base, head)
    return out


def _rho05_rows(target, v, claims, sources, params, pools, lam_m1, base, head) -> list[dict]:
    """Ablation "M1 rho 0.5" (lambda fixed at M1's choice) and the claims-only path at rho 0.5."""
    tc = target.company
    tg = target_at_rho(target, 0.5)
    dates, _ = pm.target_dates(tg)
    queries = [(tg.period_start, tg.period_end)] if tg.metric == "revenue" else None
    labs = {tc: labels_at(tc, claims[tc], v, target.origin_id, params.get(tc), end=max(dates), queries=queries, rho=0.5)}
    labs.update({c: labels_at(c, claims[c], v, None, params.get(c), rho=0.5) for c in COMPANIES if c != tc})
    lab = labs[tc]
    if len(lab.anchors) < pm.MIN_ANCHORS or not lab.ok:
        return []
    h = {**head, "t0": lab.t0.date(), "t_eff": pd.Timestamp(tg.t_eff).date(), "rho": 0.5,
         "gap_weeks": round(max((max(dates) - lab.t0).days / 7, 0.0), 2)}

    def row(model, pool, fc, **extra):
        le = math.log(target.value / fc) if fc and fc > 0 and np.isfinite(fc) else np.nan
        return {**base, "forecast": fc, "log_err": le, "model": model, "pool": pool, **h, **extra}

    out = [row(SS_RHO05, "own", pm.smoother_forecast(lab, tg))]
    flevs = {c: feature_levels(c, v, sources) for c in labs}
    pairs = {c: make_pairs(labs[c], flevs[c]) for c in labs}
    if len(pairs[tc]) < pm.MIN_TARGET_PAIRS:
        return out
    fi = forecast_inputs(lab, flevs[tc], tg)
    for pool in pools:
        P = _pool_pairs(pairs, pool_companies(pool, tc))
        m = PriorRidge(lam_m1[pool]).fit(P)
        fc, r_ = path_forecast(m, fi)
        out.append(row(ABL_RHO05, pool, fc, lam=lam_m1[pool], **_fit_cols(m, fi, tc), rate_hat=r_,
                       n_train_pairs=len(P), n_target_pairs=len(pairs[tc])))
    return out


# --------------------------------------------------------------------------- backtest
_WORKER: dict = {}


def _init_worker():
    _WORKER["claims"] = {c: pm.load_claims(c) for c in COMPANIES}
    _WORKER["sources"] = F.load_sources()


def run_job(job, ablations: bool = True) -> list[dict]:
    company, obs_id, v = job
    warnings.simplefilter("ignore")
    if not _WORKER:
        _init_worker()
    t_start = time.time()
    t = pm.load_targets(company).set_index("obs_id").loc[obs_id]
    tg = pm._Row(obs_id=obs_id, **t.to_dict())
    rows = forecast_target(tg, v, _WORKER["claims"], _WORKER["sources"], ablations=ablations)
    secs = round(time.time() - t_start, 1)
    return [{**r, "job_seconds": secs} for r in rows]


def run_backtest(rule: str, companies=COMPANIES, n_jobs: int = 1, ablations: bool = True,
                 verbose: bool = True) -> pd.DataFrame:
    jobs = sorted(pm.target_jobs(rule, companies), key=lambda j: j[0] != "anthropic")
    rows = []
    if n_jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_worker) as ex:
            for job, res in zip(jobs, ex.map(partial(run_job, ablations=ablations), jobs)):
                rows += res
                if verbose:
                    print(f"  {rule} {job[0]} {job[1]} done ({len(res)} rows)", flush=True)
    else:
        for job in jobs:
            res = run_job(job, ablations)
            rows += res
            if verbose:
                print(f"  {rule} {job[0]} {job[1]} done ({len(res)} rows)", flush=True)
    out = pd.DataFrame(rows)
    if len(out):
        out["rule"] = rule
    return out.reindex(columns=CSV_COLS)
