"""Vintage backtest: forecast each official figure using only what was public the day before.

For each target (GAAP revenue or official run-rate):
  vintage = target.published - 1 day
  evidence = rows published on/before the vintage, excluding the target's own origin
  forecast = predictive distribution of the target claim (the smoothed state plus the
             target tier's measurement noise)
It reports the z-score, PIT and whether the 50%/80% intervals cover the actual value,
next to a naive benchmark. A calibrated model's 80% interval should cover about 80%
of targets.

Caveat: process-noise parameters are fit on the full sample, so this is not a strict
real-time test. Pass refit=True to refit them at every vintage (slower).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import calibrate, config, data
from .model import Model, observation_predictive

TARGET_TIERS = ("gaap", "official")


def _naive(row, ev: pd.DataFrame) -> float | None:
    """Benchmark an analyst could compute by hand from the same evidence.

    Run-rate target: the latest point run-rate claim (random walk), in log units.
    Revenue target: join the point run-rate claims log-linearly, hold flat outside
    them, and integrate over the period: sum of daily rate / 365, in $B.
    """
    rr = ev[(ev.metric == "run_rate") & (ev.kind == "point")].sort_values("ref_date")
    if not len(rr):
        return None
    if row.metric == "run_rate":
        prior = rr[rr.ref_date <= row.ref_date]
        return None if not len(prior) else math.log(float(prior.iloc[-1].value))
    x = (rr.ref_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(float)
    y = np.log(rr.value.to_numpy(float))
    days = pd.date_range(row.period_start, row.period_end, freq="D")
    xd = (days - pd.Timestamp("2000-01-01")).days.to_numpy(float)
    return float(np.exp(np.interp(xd, x, y)).sum() / config.DAYS_PER_YEAR)


def prepublication_forecast(df: pd.DataFrame, row, params: dict | None = None, refit: bool = False,
                            min_evidence: int = 3):
    """Forecast one claim using only rows public the day before it, excluding its own origin.

    Returns (mu, sd, n_evidence, evidence_df). mu and sd are in log units for
    run-rate claims and in $B for revenue claims. Returns None if there is too little
    evidence.
    """
    vintage = pd.Timestamp(row.published) - pd.Timedelta(days=1)
    ev = data.select(df, vintage=vintage)
    ev = ev[ev.origin_id != row.origin_id]
    if len(ev) < min_evidence:
        return None
    p = dict(params or {})
    if refit:
        p.update({k: v for k, v in calibrate.fit_process_noise(df[df.published <= vintage], p).items()
                  if k in ("sigma_l", "sigma_g", "sigma_b", "growth_reversion")})
    model = Model(p)
    queries = [(row.period_start, row.period_end)] if row.metric == "revenue" else None
    end = max(pd.Timestamp(row.t_eff), vintage)
    res = model.run(ev, end=end, queries=queries)
    mu, sd = observation_predictive(res, row, model.p["tiers"])
    return mu, sd, len(ev), ev


def run_backtest(df: pd.DataFrame, params: dict | None = None, refit: bool = False,
                 min_evidence: int = 3) -> pd.DataFrame:
    full = data.select(df)
    targets = full[full.tier.isin(TARGET_TIERS) & (full.kind == "point")]
    rows = []
    for tg in targets.itertuples():
        out = prepublication_forecast(df, tg, params, refit, min_evidence)
        if out is None:
            continue
        mu, sd, n_ev, ev = out
        y = math.log(tg.value) if tg.metric == "run_rate" else float(tg.value)
        z = (y - mu) / sd
        nv = _naive(tg, ev)
        if tg.metric == "run_rate":
            err_model, err_naive = y - mu, (None if nv is None else y - nv)
            lo, hi = math.exp(mu - 1.2816 * sd), math.exp(mu + 1.2816 * sd)
            point = math.exp(mu)
            naive_pt = None if nv is None else math.exp(nv)
        else:
            err_model = math.log(y / mu)
            err_naive = None if nv is None else math.log(y / nv)
            lo, hi = mu - 1.2816 * sd, mu + 1.2816 * sd
            point, naive_pt = mu, nv
        rows.append({
            "target": tg.obs_id, "metric": tg.metric,
            "vintage": (tg.published - pd.Timedelta(days=1)).date(),
            "n_evidence": n_ev, "actual": tg.value, "forecast": point, "q10": lo, "q90": hi,
            "z": z, "pit": float(norm.cdf(z)), "in50": abs(z) < 0.6745, "in80": abs(z) < 1.2816,
            "log_err_model": err_model, "naive": naive_pt, "log_err_naive": err_naive,
        })
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> dict:
    if not len(bt):
        return {}
    ok = bt.dropna(subset=["log_err_naive"])
    return {
        "n": int(len(bt)),
        "coverage50": float(bt.in50.mean()),
        "coverage80": float(bt.in80.mean()),
        "mean_z": float(bt.z.mean()),
        "sd_z": float(bt.z.std(ddof=0)),
        "mae_log_model": float(np.mean(np.abs(bt.log_err_model))),
        "mae_log_naive_same_targets": float(np.mean(np.abs(ok.log_err_naive))) if len(ok) else float("nan"),
        "mae_log_model_same_targets": float(np.mean(np.abs(ok.log_err_model))) if len(ok) else float("nan"),
    }
