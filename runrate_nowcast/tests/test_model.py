"""Checks on the filter math (against Monte Carlo) and on recovering a known synthetic path.

Run: python -m pytest runrate_nowcast/tests -q
"""
import math

import numpy as np
import pandas as pd
import pytest

from runrate_nowcast import backtest, config, data
from runrate_nowcast.model import B, G, L, Model, Obs

D, Y = config.DAYS_PER_MONTH, config.DAYS_PER_YEAR


def _obs_df(rows):
    df = pd.DataFrame(rows)
    for c in ("ref_date", "period_start", "period_end", "published"):
        df[c] = pd.to_datetime(df.get(c))
    for c, v in {"definition": "instant", "date_sd_days": 0.0, "include": 1,
                 "source": "", "url": "", "notes": ""}.items():
        df[c] = df[c].fillna(v) if c in df else v
    df["origin_id"] = df["origin_id"].fillna(df.obs_id) if "origin_id" in df else df.obs_id
    return df


def test_predict_moments_match_monte_carlo():
    """With no observations, accumulated revenue and the log level match a simulation."""
    p = {"prior_l": math.log(20), "prior_l_sd": 0.3, "prior_g": 0.2, "prior_g_sd": 0.1,
         "sigma_l": 0.005, "sigma_g": 0.01, "sigma_b": 0.0}
    days = 60
    start = "2026-01-01"
    end = pd.Timestamp(start) + pd.Timedelta(days=days - 1)
    res = Model(p, start=start).run(None, end=end, queries=[(start, end)], smooth=False)
    m, P = res.m_f[-1], res.P_f[-1]

    rng = np.random.default_rng(0)
    n = 200_000
    l = rng.normal(p["prior_l"], p["prior_l_sd"], n)
    g = rng.normal(p["prior_g"], p["prior_g_sd"], n)
    A = np.zeros(n)
    for _ in range(days):
        A += np.exp(l + 0.5 * g / D) / Y
        l = l + g / D + rng.normal(0, p["sigma_l"], n)
        g = g + rng.normal(0, p["sigma_g"], n)
    assert m[L] == pytest.approx(l.mean(), abs=0.005)
    assert math.sqrt(P[L, L]) == pytest.approx(l.std(), rel=0.02)
    assert m[3] == pytest.approx(A.mean(), rel=0.005)
    assert math.sqrt(P[3, 3]) == pytest.approx(A.std(), rel=0.05)


def test_truncated_update_matches_rejection_sampling():
    rng = np.random.default_rng(1)
    m = np.array([3.0, 0.1, 0.05])
    Araw = rng.normal(size=(3, 3))
    P = Araw @ Araw.T * 0.01 + np.diag([0.04, 0.01, 0.01])
    H = np.array([1.0, 0.0, 1.0])
    o = Obs("x", 0, "run_rate", "lower", math.log(25), H, 0.0, 0.05 ** 2, 0.0, "official")
    mm, PP, inn = Model()._update(m.copy(), P.copy(), o, pd.Timestamp("2026-01-01"))

    x = rng.multivariate_normal(m, P, 2_000_000)
    z = x @ H + rng.normal(0, 0.05, len(x))
    keep = x[z > math.log(25)]
    assert inn.p_tail == pytest.approx(len(keep) / len(x), rel=0.02)
    assert mm == pytest.approx(keep.mean(0), abs=3e-3)
    assert PP == pytest.approx(np.cov(keep.T), abs=2e-3)


def _simulate(seed=2):
    """Truth: growth slows from 0.30/month to 0.08/month; run-rate quoted on t28x13 with gap 0.06."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    days = pd.date_range(start, "2026-11-10", freq="D")
    l = np.zeros(len(days))
    l[0] = math.log(1.2)
    gpath = np.interp(np.arange(len(days)), [0, 300, 500, len(days) - 1], [0.18, 0.30, 0.12, 0.08])
    for t in range(1, len(days)):
        l[t] = l[t - 1] + gpath[t - 1] / D + rng.normal(0, 0.003)
    daily = np.exp(l) / Y
    rows = []
    for q in pd.period_range("2025Q1", "2026Q3", freq="Q"):
        s, e = q.start_time.normalize(), q.end_time.normalize()
        rev = daily[(days >= s) & (days <= e)].sum()
        rows.append({"obs_id": f"rev{q}", "metric": "revenue", "kind": "point", "value": rev,
                     "ref_date": e, "period_start": s, "period_end": e, "published": "2026-11-01",
                     "tier": "gaap"})
    b_true = 0.06
    for i, d in enumerate(pd.date_range("2025-02-15", "2026-10-31", freq="45D")):
        t = (d - start).days - 14
        val = math.exp(l[t] + b_true + rng.normal(0, 0.04))
        rows.append({"obs_id": f"rr{i}", "metric": "run_rate", "kind": "point", "value": val,
                     "ref_date": d, "published": d, "tier": "official", "definition": "t28x13"})
    return days, l, b_true, _obs_df(rows)


def test_synthetic_recovery():
    days, l_true, b_true, df = _simulate()
    sel = data.select(df)
    qs = [(m.start_time.normalize(), m.end_time.normalize()) for m in pd.period_range("2026-07", "2026-09", freq="M")]
    res = Model({"sigma_l": 0.004, "sigma_g": 0.016, "sigma_b": 0.0015}).run(sel, end=days[-1], queries=qs)
    # Accumulated revenue reproduces each GAAP quarter.
    for r in sel[sel.metric == "revenue"].itertuples():
        key = f"{r.period_start:%Y-%m-%d}..{r.period_end:%Y-%m-%d}"
        assert res.period_revenue(key)["mean"] == pytest.approx(r.value, rel=0.01)
    # Monthly revenue adds up to the quarter.
    months = sum(res.period_revenue(f"{s:%Y-%m-%d}..{e:%Y-%m-%d}")["mean"] for s, e in qs)
    assert months == pytest.approx(res.period_revenue("2026-07-01..2026-09-30")["mean"], rel=0.005)
    # The truth falls inside the 80% band on most days, and always inside a 99.9% band.
    idx = np.arange(60, len(days) - 30)
    z = (l_true[idx] - res.ms[idx, L]) / np.sqrt(res.Ps[idx, L, L])
    assert 0.6 < np.mean(np.abs(z) < 1.2816) <= 1.0
    assert np.max(np.abs(z)) < 3.3
    # The run-rate-vs-GAAP gap is recovered.
    b, bs = res.gap(days[-1])
    assert abs(b - b_true) < 3 * bs + 0.02


def test_dedup_keeps_most_reliable_and_earliest_date():
    df = _obs_df([
        {"obs_id": "a", "origin_id": "o1", "metric": "run_rate", "kind": "point", "value": 10,
         "ref_date": "2026-01-01", "published": "2026-01-05", "tier": "secondary"},
        {"obs_id": "b", "origin_id": "o1", "metric": "run_rate", "kind": "point", "value": 10,
         "ref_date": "2026-01-01", "published": "2026-01-06", "tier": "leak_tier1"},
        {"obs_id": "c", "origin_id": "o2", "metric": "run_rate", "kind": "point", "value": 11,
         "ref_date": "2026-01-10", "published": "2026-01-10", "tier": "official", "include": 0},
    ])
    sel = data.select(df)
    assert sel.obs_id.tolist() == ["b"]
    assert sel.published.iloc[0] == pd.Timestamp("2026-01-05")


def test_real_data_backtest_runs_and_is_roughly_calibrated():
    df = data.load()
    bt = backtest.run_backtest(df)
    s = backtest.summarize(bt)
    assert s["n"] >= 10
    assert 0.5 <= s["coverage80"] <= 1.0
    assert s["mae_log_model_same_targets"] < s["mae_log_naive_same_targets"]
