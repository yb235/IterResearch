"""ML on interpolated labels, validated only on real anchors.

Idea: the real data points (GAAP quarters, run-rate claims) are too few to train on.
Fill the gaps between them with estimated weekly labels, train a model that maps
leading-indicator moves to revenue moves, then check it on the real points.

Two traps, and how this module avoids them:
  1. Interpolated labels are not new information. Every label between two anchors
     comes from those two anchors. Validation therefore uses only real anchors: the same
     13 pre-publication targets as backtest.py.
  2. Leakage. For a target published on day P, labels are built only from anchors
     public before P (the target's own origin excluded), and features only from
     indicator data public before P.

Labels (between the first and last anchor public at the vintage):
  interp    log-linear interpolation through the anchors. Run-rate claims sit at their
            effective date; a GAAP quarter sits at its midpoint as revenue x 365/days.
  smoother  the claims-only state-space path (handles GAAP quarters as exact integrals)

The model learns growth, not levels. A tree cannot extrapolate a level it has never
seen, and a nowcast is always beyond the last label. Training pairs (t0, t1), 1-13
weeks apart: y = label(t1) - label(t0), X = change in each log indicator over the same
weeks, plus the gap in weeks. Prediction = label at the last anchor + predicted growth
from there to the target date.

Models: trend (extrapolate the last 8 weeks of label growth, no indicators), ridge
(RidgeCV, alpha chosen by time-ordered CV), gbm (HistGradientBoosting, settings chosen
by time-ordered CV).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from . import altdata, backtest, calibrate, config, data, robust
from .model import Model, L, B

FEATURES = ["npm_anthropic_sdk", "npm_openai", "npm_claude_code", "npm_claude_agent_sdk",
            "npm_openai_codex", "pypi_anthropic", "pypi_openai"]
MAX_H = 13


# --------------------------------------------------------------------------- features
def weekly_features(raw: pd.DataFrame, vintage) -> pd.DataFrame:
    """Weekly log indicator levels public on/before `vintage`, holiday-cleaned and break-adjusted."""
    vintage = pd.Timestamp(vintage)
    r = raw[(raw.indicator.isin(FEATURES)) & (raw.value > 0) & (raw.date + pd.Timedelta(days=1) <= vintage)].copy()
    # break adjustment by the same rule as the filter, using only data public at the vintage
    try:
        br = robust.detect_breaks(r)
    except Exception:
        br = pd.DataFrame()
    for b in br.itertuples():
        a, c = b.source_pair.split(" / ")
        for name, step in ((a, b.step_anthropic), (c, b.step_competitor)):
            m = (r.indicator == name) & (r.date < b.date)
            r.loc[m, "value"] = r.loc[m, "value"] * (1 + step)
    md = r.date.dt.month * 100 + r.date.dt.day
    r = r[~((md >= altdata.HOLIDAY_START) | (md <= altdata.HOLIDAY_END))]
    out = {}
    for name, g in r.groupby("indicator"):
        s = g.set_index("date").value.sort_index().resample("W-SUN")
        wk = pd.DataFrame({"v": s.mean(), "n": s.count()})
        wk = wk[(wk.n >= 5) & (wk.index + pd.Timedelta(days=1) <= vintage)]
        out[name] = np.log(wk.v)
    f = pd.DataFrame(out)
    f.index = f.index - pd.Timedelta(days=3)          # week centre (Thursday)
    if {"npm_anthropic_sdk", "npm_openai"} <= set(f):
        f["npm_share"] = f.npm_anthropic_sdk - f.npm_openai
    return f.sort_index()


# --------------------------------------------------------------------------- labels
def anchors(df: pd.DataFrame, vintage, exclude_origin=None) -> pd.Series:
    sel = data.select(df[df.metric != "indicator"], vintage=vintage)
    if exclude_origin is not None:
        sel = sel[sel.origin_id != exclude_origin]
    pts = []
    rr = sel[(sel.metric == "run_rate") & (sel.kind == "point") & (sel.segment == "total") & (sel.is_forecast == 0)]
    pts += [(pd.Timestamp(r.t_eff), math.log(r.value)) for r in rr.itertuples()]
    rev = sel[(sel.metric == "revenue") & (sel.segment == "total")]
    for r in rev.itertuples():
        days = (r.period_end - r.period_start).days + 1
        if days <= 100:
            pts.append((r.period_start + pd.Timedelta(days=days // 2), math.log(r.value * 365 / days)))
    if not pts:
        return pd.Series(dtype=float)
    s = pd.Series([v for _, v in pts], index=[d for d, _ in pts]).groupby(level=0).mean().sort_index()
    return s[s.index >= pd.Timestamp(config.MODEL_START)]


def labels_interp(anch: pd.Series, weeks: pd.DatetimeIndex) -> pd.Series:
    w = weeks[(weeks >= anch.index.min()) & (weeks <= anch.index.max())]
    x = (anch.index - anch.index.min()).days.to_numpy(float)
    xw = (w - anch.index.min()).days.to_numpy(float)
    return pd.Series(np.interp(xw, x, anch.to_numpy()), index=w)


def labels_smoother(df, vintage, exclude_origin, weeks, params, last_anchor) -> pd.Series:
    sel = data.select(df[df.metric != "indicator"], vintage=vintage)
    if exclude_origin is not None:
        sel = sel[sel.origin_id != exclude_origin]
    res = Model(params).run(sel, end=max(pd.Timestamp(vintage), last_anchor))
    w = weeks[(weeks >= res.dates[0]) & (weeks <= last_anchor)]
    return pd.Series([res.state(d)[0][L] + res.state(d)[0][B] for d in w], index=w)


# --------------------------------------------------------------------------- training pairs
def make_pairs(lab: pd.Series, feats: pd.DataFrame, max_h=MAX_H):
    idx = lab.index.intersection(feats.index)
    lab, F = lab[idx], feats.loc[idx]
    X, y, t1 = [], [], []
    vals, fv = lab.to_numpy(), F.to_numpy()
    for i in range(len(idx)):
        for j in range(i + 1, min(i + max_h, len(idx) - 1) + 1):
            X.append(np.concatenate([fv[j] - fv[i], [j - i]]))
            y.append(vals[j] - vals[i])
            t1.append(idx[j])
    cols = list(F.columns) + ["gap_weeks"]
    return pd.DataFrame(X, columns=cols), np.array(y), pd.DatetimeIndex(t1)


class _Cols:
    """Fit on columns with enough data in every CV fold; predict with the same columns."""
    def __init__(self, est, cols):
        self.est, self.cols = est, cols

    def predict(self, X):
        return self.est.predict(X[self.cols])


def _fit(kind, X, y):
    # a column must have data in the first CV training fold, or some folds would see it all-empty
    first_fold = X.iloc[: max(len(X) // 5, 10)]
    cols = [c for c in X.columns if first_fold[c].notna().sum() >= 10]
    return _Cols(_fit_raw(kind, X[cols], y), cols)


def _fit_raw(kind, X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = TimeSeriesSplit(n_splits=4)
    if kind == "ridge":
        m = make_pipeline(SimpleImputer(strategy="constant", fill_value=0.0), StandardScaler(),
                          RidgeCV(alphas=np.logspace(-3, 3, 13), cv=cv))
        return m.fit(X, y)
    if kind == "gbm":
        grid = {"max_depth": [2, 3], "learning_rate": [0.03, 0.1], "max_iter": [100, 300]}
        gs = GridSearchCV(HistGradientBoostingRegressor(min_samples_leaf=20, random_state=0), grid, cv=cv,
                          scoring="neg_mean_absolute_error")
        return gs.fit(X, y).best_estimator_
    raise ValueError(kind)


# --------------------------------------------------------------------------- prediction for one target
def predict_target(df, raw, target, params, label_kind="interp", models=("trend", "ridge", "gbm")):
    vintage = pd.Timestamp(target.published) - pd.Timedelta(days=1)
    anch = anchors(df, vintage, exclude_origin=target.origin_id)
    if len(anch) < 3:
        return None
    feats = weekly_features(raw, vintage)
    weeks = feats.index
    last_anchor = anch.index.max()
    if label_kind == "interp":
        lab = labels_interp(anch, weeks)
    else:
        lab = labels_smoother(df, vintage, target.origin_id, weeks, params, last_anchor)
    lab = lab.dropna()
    if len(lab) < 6:
        return None
    t0 = lab.index.max()
    # weeks the target needs
    if target.metric == "run_rate":
        need = [pd.Timestamp(target.t_eff)]
    else:
        need = list(pd.date_range(target.period_start, target.period_end, freq="D"))
    X, y, _ = make_pairs(lab, feats)
    fitted = {k: _fit(k, X, y) for k in models if k != "trend"} if len(X) >= 30 else {}
    recent = lab[lab.index >= t0 - pd.Timedelta(weeks=8)]
    slope = (recent.iloc[-1] - recent.iloc[0]) / max((recent.index[-1] - recent.index[0]).days / 7, 1)
    preds = {}
    for k in models:
        path = []
        for d in need:
            if d <= t0:                                    # covered by labels: interpolate
                xd = (lab.index - lab.index[0]).days.to_numpy(float)
                path.append(float(np.interp((d - lab.index[0]).days, xd, lab.to_numpy())))
                continue
            w1 = feats.index[feats.index <= d]
            if not len(w1):
                path.append(np.nan)
                continue
            w1 = w1[-1]
            gap = (d - t0).days / 7
            if k == "trend" or k not in fitted:
                path.append(lab[t0] + slope * gap)
            else:
                dx = feats.loc[w1] - feats.loc[t0] if t0 in feats.index else feats.loc[w1] * np.nan
                xrow = pd.DataFrame([np.concatenate([dx.to_numpy(), [gap]])], columns=X.columns)
                path.append(lab[t0] + float(fitted[k].predict(xrow)[0]))
        path = np.array(path)
        if np.isnan(path).any():
            preds[k] = np.nan
        elif target.metric == "run_rate":
            preds[k] = path[0]                              # log run-rate
        else:
            preds[k] = float(np.exp(path).mean() * len(need) / 365)   # $B revenue
    return preds, len(anch), t0


def run(df=None, raw=None, params=None, label_kinds=("interp", "smoother"), verbose=True) -> pd.DataFrame:
    warnings.simplefilter("ignore")
    df = data.load_all() if df is None else df
    raw = altdata.load_raw() if raw is None else raw
    params = calibrate.load_params() if params is None else params
    full = data.select(df)
    targets = full[full.tier.isin(backtest.TARGET_TIERS) & (full.kind == "point") & full.metric.isin(["run_rate", "revenue"])
                   & (full.is_forecast == 0) & (full.segment == "total")]
    rows = []
    for tg in targets.itertuples():
        if pd.Timestamp(tg.t_eff) < pd.Timestamp(config.MODEL_START):
            continue
        for lk in label_kinds:
            out = predict_target(df, raw, tg, params, lk)
            if out is None:
                continue
            preds, n_anch, t0 = out
            y = math.log(tg.value)
            for k, p in preds.items():
                if p is None or (isinstance(p, float) and math.isnan(p)):
                    continue
                err = (y - p) if tg.metric == "run_rate" else math.log(tg.value / p)
                rows.append({"target": tg.obs_id, "metric": tg.metric,
                             "vintage": (tg.published - pd.Timedelta(days=1)).date(),
                             "labels": lk, "model": k, "n_anchors": n_anch, "last_label": t0.date(),
                             "log_err": err})
        if verbose:
            print(f"  {tg.obs_id} done", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- current nowcast
def nowcast(df=None, raw=None, params=None, asof=None, dates=None, label_kind="smoother") -> dict:
    """Ridge nowcast with all data public at `asof`, on the official run-rate basis.

    Labels sit at effective dates, so the official run-rate reported on day d is the label
    at d - 14 days (t28x13). Returns {date: run-rate} plus the learned elasticities.
    Dates past the last indicator week cannot be predicted (no features yet).
    """
    df = data.load_all() if df is None else df
    raw = altdata.load_raw() if raw is None else raw
    params = calibrate.load_params() if params is None else params
    asof = pd.Timestamp(asof or pd.Timestamp.today().normalize())
    anch = anchors(df, asof)
    feats = weekly_features(raw, asof)
    if label_kind == "interp":
        lab = labels_interp(anch, feats.index)
    else:
        lab = labels_smoother(df, asof, None, feats.index, params, anch.index.max())
    lab = lab.dropna()
    t0 = lab.index.max()
    X, y, _ = make_pairs(lab, feats)
    fit = _fit("ridge", X, y)
    est, sc = fit.est.steps[-1][1], fit.est.steps[1][1]
    out = {"label_kind": label_kind, "last_label": t0, "last_label_value": math.exp(lab[t0]),
           "alpha": float(est.alpha_), "elasticities": {c: round(float(v), 3) for c, v in zip(fit.cols, est.coef_ / sc.scale_)},
           "runrate": {}}
    lag = pd.Timedelta(days=round(config.DEFINITION_LAG_DAYS["t28x13"]))
    for d in dates or [asof - pd.offsets.MonthBegin(1) - pd.Timedelta(days=1), asof]:
        eff = pd.Timestamp(d) - lag
        if eff <= t0:
            xd = (lab.index - lab.index[0]).days.to_numpy(float)
            out["runrate"][pd.Timestamp(d)] = math.exp(float(np.interp((eff - lab.index[0]).days, xd, lab.to_numpy())))
            continue
        w1 = feats.index[feats.index <= eff]
        if not len(w1) or t0 not in feats.index:
            continue
        dx = feats.loc[w1[-1]] - feats.loc[t0]
        xrow = pd.DataFrame([np.concatenate([dx.to_numpy(), [(eff - t0).days / 7]])], columns=X.columns)
        out["runrate"][pd.Timestamp(d)] = math.exp(lab[t0] + float(fit.predict(xrow)[0]))
    return out


def markdown(res: pd.DataFrame, ss_strict: pd.DataFrame | None, now: list[dict], asof) -> str:
    out = [f"# ML on interpolated labels (as of {pd.Timestamp(asof):%Y-%m-%d})\n",
           "Labels between real data points are estimated (log-linear interpolation, or the claims-only "
           "state-space path). Models learn indicator growth -> revenue growth. Validation uses only the real "
           "pre-publication targets, with labels and features restricted to data public before each target.\n"]
    piv = res.pivot_table(index="target", columns=["labels", "model"], values="log_err")
    piv.columns = [f"{a}/{b}" for a, b in piv.columns]
    if ss_strict is not None:
        piv = ss_strict.set_index("target")[["SS_claims", "SS_indicators"]].join(piv, how="inner")
    out.append("## Per-target log error\n")
    out.append("| target | " + " | ".join(piv.columns) + " |")
    out.append("|---|" + "---|" * len(piv.columns))
    for t, r in piv.iterrows():
        out.append(f"| {t} | " + " | ".join(f"{v:+.3f}" for v in r) + " |")
    out.append("\n## Mean |log error|\n")
    out.append("| model | all targets | " + " |")
    out.append("|---|---|")
    for c in piv.columns:
        out.append(f"| {c} | {piv[c].abs().mean():.3f} |")
    out.append("\n## Current ML nowcast (official run-rate basis)\n")
    for n in now:
        rr = ", ".join(f"{d:%Y-%m-%d}: {v:.1f}" for d, v in n["runrate"].items())
        out.append(f"- {n['label_kind']} labels: {rr}. Last label {n['last_label']:%Y-%m-%d} = "
                   f"{n['last_label_value']:.1f}; ridge alpha {n['alpha']:.3g}; elasticities {n['elasticities']}")
    return "\n".join(out) + "\n"
