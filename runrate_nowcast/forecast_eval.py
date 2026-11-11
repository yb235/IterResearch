"""Collect the no-filing forecast backtests into one table and summarize them.

Inputs (written by the backtests, reports/):
  forecast_backtest_<rule>_<asof>.csv          state-space, claims only and + indicators
  forecast_bayes_<rule>_<asof>.csv             Bayesian (Laplace mixture)
  forecast_ml_<rule>_<asof>.csv                ML ridge (smoother / interpolated labels)
  forecast_price_<scheme>_g<gamma>_<rule>_<asof>.csv   C: + price-adjusted indicators
  forecast_price_<scheme>_g<gamma>_github_<rule>_<asof>.csv   D: + GitHub commits (scheme none = no price adjustment)
  forecast_openrouter_<variant>_<rule>_<asof>.csv      E: OpenRouter weekly Anthropic tokens
  forecast_all_<variant>_<rule>_<asof>.csv             F: every real-time signal at once
  forecast_mlall_<rule>_<asof>.csv                     ML on every real-time signal
rule: publication (day before release) or period_end (date the number refers to).
Errors are log(actual / forecast): negative = forecast too high.
"""
from __future__ import annotations

import glob
import math
import os
import re

import numpy as np
import pandas as pd

from . import config, data


def _z(row_err, sd):
    return row_err / sd if sd and sd > 0 else np.nan


def collect(asof: str = "2026-11-10", reports_dir: str = config.OUTPUT_DIR) -> pd.DataFrame:
    obs = data.load().set_index("obs_id")
    rows = []
    for rule in ("publication", "period_end"):
        f = os.path.join(reports_dir, f"forecast_backtest_{rule}_{asof}.csv")
        if os.path.exists(f):
            s = pd.read_csv(f)
            for key, name in (("claims_only", "A claims only"), ("with_indicators", "B + indicators")):
                for r in s.dropna(subset=[f"log_err_model_{key}"]).itertuples():
                    rows.append({"rule": rule, "model": name, "target": r.target, "metric": r.metric, "vintage": r.vintage,
                                 "actual": r.actual, "forecast": getattr(r, f"forecast_{key}"),
                                 "log_err": getattr(r, f"log_err_model_{key}"), "z": getattr(r, f"z_{key}")})
        f = os.path.join(reports_dir, f"forecast_bayes_{rule}_{asof}.csv")
        if os.path.exists(f):
            b = pd.read_csv(f)
            for r in b[b.method == "laplace_mixture"].itertuples():
                fc = math.exp(r.forecast_mu) if r.metric == "run_rate" else r.forecast_mu
                rows.append({"rule": rule, "model": "Bayes + indicators", "target": r.target, "metric": r.metric,
                             "vintage": r.vintage, "actual": r.actual, "forecast": fc, "log_err": r.log_err, "z": r.z})
        f = os.path.join(reports_dir, f"forecast_ml_{rule}_{asof}.csv")
        if os.path.exists(f):
            m = pd.read_csv(f)
            for r in m[m.model == "ridge"].itertuples():
                act = float(obs.loc[r.target, "value"])
                rows.append({"rule": rule, "model": f"ML ridge ({r.labels})", "target": r.target, "metric": r.metric,
                             "vintage": r.vintage, "actual": act, "forecast": act / math.exp(r.log_err),
                             "log_err": r.log_err, "z": np.nan})
        for f in sorted(glob.glob(os.path.join(reports_dir, f"forecast_price_*_{rule}_{asof}.csv"))):
            m_ = re.match(r".*forecast_price_(.+?)_g([\d.]+)(_github)?_" + rule + "_", f)
            if not m_:
                continue
            scheme, g, gh = m_.groups()
            if gh:
                label = "D + GitHub commits" + ("" if scheme == "none" else f", price-adjusted ({scheme}, gamma {g})")
            else:
                label = f"C price-adjusted ({scheme}, gamma {g})"
            p = pd.read_csv(f)
            for r in p.itertuples():
                rows.append({"rule": rule, "model": label, "target": r.target,
                             "metric": r.metric, "vintage": r.vintage, "actual": r.actual, "forecast": r.forecast,
                             "log_err": r.log_err_model, "z": r.z})
        for f in sorted(glob.glob(os.path.join(reports_dir, f"forecast_openrouter_*_{rule}_{asof}.csv"))):
            variant = re.match(r".*forecast_openrouter_(.+?)_" + rule + "_", f).group(1)
            label = {"or_only": "E OpenRouter tokens", "or_price": "E OpenRouter tokens x price",
                     "b_plus_or": "E B + OpenRouter tokens", "opencode": "E OpenCode downloads"}.get(variant, f"E {variant}")
            for r in pd.read_csv(f).itertuples():
                rows.append({"rule": rule, "model": label, "target": r.target, "metric": r.metric, "vintage": r.vintage,
                             "actual": r.actual, "forecast": r.forecast, "log_err": r.log_err_model, "z": r.z})
        for f in sorted(glob.glob(os.path.join(reports_dir, f"forecast_all_*_{rule}_{asof}.csv"))):
            variant = re.match(r".*forecast_all_(.+?)_" + rule + "_", f).group(1)
            label = {"all_price": "F all signals x price", "all_raw": "F all signals",
                     "core_price": "F core signals x price (selected by backtest)"}.get(variant, f"F {variant}")
            for r in pd.read_csv(f).itertuples():
                rows.append({"rule": rule, "model": label, "target": r.target, "metric": r.metric, "vintage": r.vintage,
                             "actual": r.actual, "forecast": r.forecast, "log_err": r.log_err_model, "z": r.z})
        f = os.path.join(reports_dir, f"forecast_mlall_{rule}_{asof}.csv")
        if os.path.exists(f):
            m = pd.read_csv(f)
            for r in m[m.model == "ridge"].itertuples():
                act = float(obs.loc[r.target, "value"])
                rows.append({"rule": rule, "model": f"ML all signals ({r.labels})", "target": r.target, "metric": r.metric,
                             "vintage": r.vintage, "actual": act, "forecast": act / math.exp(r.log_err),
                             "log_err": r.log_err, "z": np.nan})
    d = pd.DataFrame(rows)
    # Ensemble, defined before its results were seen: equal-weight mean of log forecasts of the
    # all-signal state-space model (x price) and the two all-signal ML ridge models.
    parts = ["F all signals x price", "ML all signals (interp)", "ML all signals (smoother)"]
    if len(d) and set(parts) <= set(d.model):
        e = d[d.model.isin(parts)]
        for (rule, t), h in e.groupby(["rule", "target"]):
            if h.model.nunique() < len(parts):
                continue
            le = float(h.log_err.mean())            # mean log error = log(actual) - mean log forecast
            act = float(h.actual.iloc[0])
            rows_e = {"rule": rule, "model": "Ensemble F + ML (all signals)", "target": t, "metric": h.metric.iloc[0],
                      "vintage": h.vintage.iloc[0], "actual": act, "forecast": act / math.exp(le), "log_err": le, "z": np.nan}
            d = pd.concat([d, pd.DataFrame([rows_e])], ignore_index=True)
    if len(d):
        d["filing_target"] = d.target.map(lambda t: int(obs.loc[t, "filing"]) == 1)
        d["pct_err"] = 100 * (d.forecast / d.actual - 1)
    return d


def summary(d: pd.DataFrame, common: bool = True) -> pd.DataFrame:
    """Per rule and model. With common=True, only targets every model in that rule forecast."""
    out = []
    for rule, g in d.groupby("rule"):
        if common:
            n_models = g.model.nunique()
            keep = g.groupby("target").model.nunique()
            g = g[g.target.isin(keep[keep == n_models].index)]
        for model, h in g.groupby("model"):
            z = h.z.dropna()
            out.append({"rule": rule, "model": model, "n": len(h),
                        "mae_log": h.log_err.abs().mean(), "bias_log": h.log_err.mean(),
                        "mae_log_filing_targets": h[h.filing_target].log_err.abs().mean(),
                        "mae_log_official_claims": h[~h.filing_target].log_err.abs().mean(),
                        "cov80": (z.abs() < 1.2816).mean() if len(z) else np.nan,
                        "sd_z": z.std(ddof=0) if len(z) else np.nan})
    return pd.DataFrame(out).sort_values(["rule", "mae_log"])


def paired(d: pd.DataFrame, a: str, b: str, rule: str, n_boot: int = 10000, seed: int = 0) -> dict:
    g = d[d.rule == rule]
    ea = g[g.model == a].set_index("target").log_err.abs()
    eb = g[g.model == b].set_index("target").log_err.abs()
    common = ea.index.intersection(eb.index)
    diff = (eb[common] - ea[common]).to_numpy()
    if not len(diff):
        return {}
    boots = np.random.default_rng(seed).choice(diff, size=(n_boot, len(diff))).mean(axis=1)
    lo, hi = np.percentile(boots, [5, 95])
    return {"rule": rule, "comparison": f"{b} vs {a}", "n": len(diff), "mean_diff": diff.mean(), "ci90_lo": lo, "ci90_hi": hi,
            "improved": int((diff < 0).sum()), "worse": int((diff > 0).sum())}


H2_TARGETS = ("rev_2026q3", "rr_2026_09_s1a", "rr_2026_10_s1a")


def markdown(d: pd.DataFrame, asof: str = "2026-11-10") -> str:
    out = [f"# Forecasts without the filings: backtest ({asof})", "",
           "Inputs: company statements, leaks, third-party estimates and alternative data, each only from the day it",
           "was public. Never inputs: the S-1 and its amendments (filing = 1) and third-party numbers published after a",
           "filing that may derive from it (filing = 2). Parameters are refit at every vintage. Errors are",
           "log(actual / forecast); negative means the forecast was too high.", ""]
    for rule in ("publication", "period_end"):
        g = d[d.rule == rule]
        if not len(g):
            continue
        out += [f"## Vintage: {rule}", "", summary(g, common=False).round(3).to_markdown(index=False), "",
                "H2 2026 targets, % error (+ = forecast too high):", "",
                g[g.target.isin(H2_TARGETS)].pivot_table(index="model", columns="target", values="pct_err").round(1).to_markdown(), ""]
        pairs = [paired(d, a, b, rule) for a, b in (("A claims only", "B + indicators"),
                                                    ("B + indicators", "C price-adjusted (github, gamma 1)"),
                                                    ("B + indicators", "D + GitHub commits"),
                                                    ("B + indicators", "Bayes + indicators"),
                                                    ("A claims only", "E OpenRouter tokens"),
                                                    ("A claims only", "E OpenRouter tokens x price"),
                                                    ("B + indicators", "E B + OpenRouter tokens"),
                                                    ("A claims only", "E OpenCode downloads"))]
        pairs = [p for p in pairs if p]
        pairs += [p for p in (paired(d, "E OpenRouter tokens x price", "F all signals x price", rule),
                              paired(d, "ML ridge (interp)", "ML all signals (interp)", rule),
                              paired(d, "A claims only", "Combination: weights from past accuracy (all signals)", rule)) if p]
        if pairs:
            out += ["Paired differences in |log error| (negative = second model better; 90% bootstrap interval):", "",
                    pd.DataFrame(pairs).round(3).to_markdown(index=False), ""]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- forecast combination
COMBINE = ["A claims only", "B + indicators", "C price-adjusted (github, gamma 1)", "D + GitHub commits",
           "E OpenRouter tokens", "E OpenRouter tokens x price", "E OpenCode downloads",
           "ML all signals (interp)", "ML all signals (smoother)"]
PRIOR_MSE = 0.1 ** 2          # shrinks weights toward equal while few past errors are known


def combine_realtime(d: pd.DataFrame, members=COMBINE) -> pd.DataFrame:
    """Two combinations of the single-signal forecasts, both defined before their results were seen.

    equal      mean of log forecasts over the members that forecast the target
    inv_mse    weights proportional to 1 / (PRIOR_MSE + mean squared log error on earlier targets),
               using only targets whose actual value was public by this forecast's vintage
               (Bates-Granger). With no earlier targets the weights are equal.
    """
    pub = data.load().set_index("obs_id").published
    out = []
    for rule, g in d[d.model.isin(members)].groupby("rule"):
        for t, h in g.groupby("target"):
            v = pd.Timestamp(h.vintage.iloc[0])
            act = float(h.actual.iloc[0])
            le = h.set_index("model").log_err
            past = g[g.target.map(pub) <= v]
            mse = past.groupby("model").log_err.apply(lambda s: float((s ** 2).sum())) / past.groupby("model").size()
            npast = past.groupby("model").size()
            known = {m: mse[m] for m in le.index if npast.get(m, 0)}
            fill = float(np.mean(list(known.values()))) if known else 0.0       # no history: the average member
            w = pd.Series({m: 1.0 / (PRIOR_MSE + known.get(m, fill)) for m in le.index})
            for name, ww in (("Combination: equal weights (all signals)", pd.Series(1.0, index=le.index)),
                             ("Combination: weights from past accuracy (all signals)", w)):
                e = float((le * ww).sum() / ww.sum())
                out.append({"rule": rule, "model": name, "target": t, "metric": h.metric.iloc[0], "vintage": h.vintage.iloc[0],
                            "actual": act, "forecast": act / math.exp(e), "log_err": e, "z": np.nan,
                            "filing_target": h.filing_target.iloc[0], "pct_err": 100 * (math.exp(-e) - 1)})
    return pd.DataFrame(out)
