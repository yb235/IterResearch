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
    for c, v in {"definition": "instant", "date_sd_days": 0.0, "include": 1, "source": "", "url": "",
                 "notes": "", "segment": "total", "indicator": "", "is_forecast": 0}.items():
        df[c] = df[c].fillna(v) if c in df else v
    df["is_forecast"] = df["is_forecast"].astype(int)
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


# ----------------------------------------------------------------- segment / indicator layers
def _indicator_df(days, l_true, start_obs="2025-01-05", step=None, beta=1.0, drift_sd=0.0, seed=5, name="ind_x"):
    rng = np.random.default_rng(seed)
    rows, a = [], 2.0
    for d in pd.date_range(start_obs, days[-1], freq="7D"):
        t = (d - days[0]).days
        a += rng.normal(0, drift_sd)
        v = math.exp(a + beta * l_true[t] + rng.normal(0, 0.02))
        if step is not None and d >= pd.Timestamp(step):
            v *= 0.65
        rows.append({"obs_id": f"{name}@{d:%Y-%m-%d}", "metric": "indicator", "kind": "point", "value": v,
                     "ref_date": d, "published": d + pd.Timedelta(days=1), "tier": "altdata", "indicator": name})
    return _obs_df(rows)


def _spec(**kw):
    s = {"segment": "total", "beta": 1.0, "obs_sd": 0.02, "drift": 0.002, "prior_sd": 5.0, "breaks": []}
    s.update(kw)
    return s


def test_partition_shares_sum_to_one():
    days, l_true, _, df = _simulate()
    share = _obs_df([{"obs_id": "sub", "metric": "share", "kind": "point", "value": 0.2, "ref_date": "2026-05-01",
                      "published": "2026-10-26", "tier": "gaap", "segment": "subscription"}])
    sel = data.select(pd.concat([df, share], ignore_index=True))
    res = Model({"sigma_l": 0.004, "sigma_g": 0.016}).run(sel, end=days[-1])
    for d in ["2025-06-01", "2026-05-01", "2026-11-01"]:
        tot = res.share(d, "api")["mean"] + res.share(d, "subscription")["mean"]
        assert tot == pytest.approx(1.0, abs=0.03)
    assert res.share("2026-05-01", "subscription")["median"] == pytest.approx(0.2, rel=0.1)


def test_indicator_tracks_growth_between_anchors():
    """Hide Q3 from the model: an indicator proportional to revenue should recover the Q3 path."""
    days, l_true, _, df = _simulate()
    no_q3 = df[df.obs_id != "rev2026Q3"]
    no_q3 = no_q3[~((no_q3.metric == "run_rate") & (pd.to_datetime(no_q3.ref_date) > "2026-06-30"))]
    ind = _indicator_df(days, l_true, drift_sd=0.005)
    p = {"sigma_l": 0.004, "sigma_g": 0.016, "indicators": {"ind_x": _spec(drift=0.003)}}
    t = (pd.Timestamp("2026-09-30") - days[0]).days
    with_ind = Model(p).run(data.select(pd.concat([no_q3, ind], ignore_index=True)), end=days[-1])
    without = Model(p).run(data.select(no_q3), end=days[-1])
    err_with = abs(with_ind.ms[t, L] - l_true[t])
    err_without = abs(without.ms[t, L] - l_true[t])
    assert err_with < err_without
    assert err_with < 0.05


def test_break_absorbs_a_counting_change():
    """A 35% step in the indicator on a known break date must not show up as a revenue drop."""
    days, l_true, _, df = _simulate()
    base = df[pd.to_datetime(df.published) <= "2026-07-01"]           # anchors end mid-2026
    ind = _indicator_df(days, l_true, step="2026-08-25")
    t = (pd.Timestamp("2026-10-15") - days[0]).days
    errs = {}
    for label, breaks in (("with", ["2026-08-25"]), ("without", [])):
        p = {"sigma_l": 0.004, "sigma_g": 0.016, "indicators": {"ind_x": _spec(breaks=breaks)}}
        res = Model(p).run(data.select(pd.concat([base, ind], ignore_index=True)), end=days[-1])
        errs[label] = res.ms[t, L] - l_true[t]
    assert abs(errs["with"]) < 0.06
    assert errs["without"] < -0.2          # read as a large revenue fall


def test_weekly_aggregation_and_publication_dates():
    from runrate_nowcast import altdata
    raw = pd.DataFrame({"date": pd.date_range("2026-01-05", "2026-01-25", freq="D"), "indicator": "x",
                        "value": 100.0, "published": pd.NaT, "source": "", "note": ""})
    cat = pd.DataFrame([{"indicator": "x", "segment": "api", "role": "model", "beta": 1, "beta_sd": 0.2, "obs_sd": 0.05,
                         "drift": 0.003, "prior_sd": 5, "lag_days": 0, "aggregate": "weekly_mean",
                         "publish_delay_days": 1, "frequency": "daily", "access": "", "fetcher": "npm:x",
                         "source_url": "", "description": "", "breaks": ""}])
    obs = altdata.indicator_observations(cat=cat, raw=raw)
    assert len(obs) == 3                                     # three full Mon-Sun weeks
    assert obs.ref_date.iloc[0] == pd.Timestamp("2026-01-08")   # week centre (Thursday)
    assert obs.published.iloc[0] == pd.Timestamp("2026-01-12")  # Sunday + 1 day


def test_forecast_claims_excluded_unless_requested():
    df = _obs_df([
        {"obs_id": "f", "metric": "run_rate", "kind": "point", "value": 100, "ref_date": "2026-12-31",
         "published": "2026-08-01", "tier": "leak_tier1", "is_forecast": 1},
        {"obs_id": "o", "metric": "run_rate", "kind": "point", "value": 50, "ref_date": "2026-06-01",
         "published": "2026-06-02", "tier": "official"},
    ])
    assert data.select(df).obs_id.tolist() == ["o"]
    assert set(data.select(df, forecasts=True).obs_id) == {"o", "f"}


def test_break_rule_matches_catalog_and_ignores_holidays():
    """Catalog breaks must be exactly what the rule detects; a holiday dip is not a break."""
    from runrate_nowcast import altdata, robust
    found = robust.detect_breaks()
    detected = {(r.source_pair.split(" / ")[0], r.date.strftime("%Y-%m-%d")) for r in found.itertuples()}
    cat = altdata.indicator_params()
    listed = {(k, d) for k, v in cat.items() for d in v.get("breaks", []) if not k.endswith("openai")}
    assert listed == detected

    days = pd.date_range("2025-11-01", "2026-03-31", freq="D")
    base = pd.Series(100.0, index=days)
    md = days.month * 100 + days.day
    holiday = base.where(~((md >= 1222) | (md <= 103)), 55.0)           # dips, then recovers
    raw = pd.concat([pd.DataFrame({"date": days, "indicator": n, "value": holiday.values, "published": pd.NaT,
                                   "source": "", "note": ""}) for n in ("pypi_anthropic", "pypi_openai")])
    assert robust.detect_breaks(raw, pairs=(("pypi_anthropic", "pypi_openai"),)).empty
    stepped = base.where(days < "2026-02-10", 60.0)                      # permanent counting change
    raw2 = pd.concat([pd.DataFrame({"date": days, "indicator": n, "value": stepped.values, "published": pd.NaT,
                                    "source": "", "note": ""}) for n in ("pypi_anthropic", "pypi_openai")])
    hit = robust.detect_breaks(raw2, pairs=(("pypi_anthropic", "pypi_openai"),))
    assert len(hit) == 1 and abs((hit.date.iloc[0] - pd.Timestamp("2026-02-10")).days) <= 1


def test_ml_uses_only_data_public_before_the_vintage():
    """Leakage guard: anchors and features must be public on or before the vintage."""
    from runrate_nowcast import altdata, ml
    df, raw = data.load_all(), altdata.load_raw()
    vintage = pd.Timestamp("2026-05-27")
    anch = ml.anchors(df, vintage)
    sel = data.select(df[df.metric != "indicator"], vintage=vintage)
    assert set(anch.index) <= set(pd.to_datetime(sel.t_eff)) | {
        r.period_start + pd.Timedelta(days=((r.period_end - r.period_start).days + 1) // 2)
        for r in sel[sel.metric == "revenue"].itertuples()}
    assert (sel.published <= vintage).all()
    feats = ml.weekly_features(raw, vintage)
    # week centre + 3 days = Sunday week end; data for that week is public the next day
    assert (feats.index + pd.Timedelta(days=4) <= vintage).all()
