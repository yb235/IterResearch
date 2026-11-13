"""Per-model price index (pricing.py) and the gamma-by-evidence price study (price_study.py)."""
import math

import numpy as np
import pandas as pd
import pytest

from runrate_nowcast import backtest, calibrate, data, pricing, price_study


def _q(rows):
    d = pd.DataFrame(rows, columns=["week_start", "query", "total_count"])
    d["week_start"] = pd.to_datetime(d.week_start)
    d["total_count"] = d.total_count.astype(float)
    return d


@pytest.fixture(scope="module")
def prices():
    return pricing.load_prices()


def test_exact_counts_never_subtraction(prices):
    # Week of 2026-10-05: "Sonnet 5" also matches 5.5 (a prefix), so only "Sonnet 5 noreply" is used.
    q = _q([("2026-10-05", "Claude", 1000), ("2026-10-05", "Claude Opus", 700),
            ("2026-10-05", "Claude Opus 5.5", 450), ("2026-10-05", "Claude Opus 5 noreply", 50),
            ("2026-10-05", "Claude Sonnet", 280), ("2026-10-05", "Claude Sonnet 5.5", 200),
            ("2026-10-05", "Claude Sonnet 5", 9999), ("2026-10-05", "Claude Sonnet 5 noreply", 50),
            ("2026-10-05", "Claude Haiku", 20)])
    c = pricing.model_counts(q, prices).iloc[0]
    # two exact counts in a family: their sum is the family total (family counts come from other days)
    assert c["Opus 5.5"] == 450 and c["Opus 5"] == 50
    assert c["Sonnet 5.5"] == 200 and c["Sonnet 5"] == 50  # the prefix count "Sonnet 5" is never used
    assert c["Haiku 5.5"] == 20                            # no exact counts: newest released (10-07) takes the family
    assert c["named"] == pytest.approx(770) and c["coverage"] == pytest.approx(0.77)
    assert bool(c["measured"]) is False                    # < 10k named commits
    # one exact count: family count x within-family split
    q1 = _q([("2026-10-05", "Claude Opus", 700), ("2026-10-05", "Claude Opus 5.5", 450)])
    assert pricing.model_counts(q1, prices).iloc[0]["Opus 5.5"] == 700


def test_family_total_falls_back_to_sum_of_exact_counts(prices):
    q = _q([("2026-11-02", "Claude Opus 5.5", 90), ("2026-11-02", "Claude Opus 5 noreply", 10)])
    c = pricing.model_counts(q, prices).iloc[0]
    assert c["Opus 5.5"] == 90 and c["Opus 5"] == 10 and c["named"] == 100 and np.isnan(c["total"])


def _shares(rows):
    s = pd.DataFrame(rows).set_index("week_start")
    s.index = pd.to_datetime(s.index)
    return s.fillna(0.0)


def test_price_level_uses_only_public_sample_weeks_and_successor_rule(prices):
    sh = _shares([{"week_start": "2026-09-07", "Opus 5": 0.6, "Sonnet 5": 0.4},
                  {"week_start": "2026-10-05", "Opus 5.5": 0.8, "Sonnet 5.5": 0.2}])
    assert math.isnan(pricing.model_price_level("2026-09-13", "list", sh, prices))   # week ends 09-13, public 09-14
    assert pricing.model_price_level("2026-09-14", "list", sh, prices) == pytest.approx(0.6 * 25 + 0.4 * 10)
    # Opus 5.5 ($20) released 09-22: the Opus share moves to it before it is measured
    assert pricing.model_price_level("2026-09-23", "list", sh, prices) == pytest.approx(0.6 * 20 + 0.4 * 10)
    # the 10-05 sample (ends 10-11) is not used before 10-12
    assert pricing.model_price_level("2026-10-11", "list", sh, prices) == pytest.approx(0.6 * 20 + 0.4 * 10)
    assert pricing.model_price_level("2026-10-12", "list", sh, prices) == pytest.approx(0.8 * 20 + 0.2 * 10)
    # a later price list never changes an earlier level
    early = prices[prices.release_date <= "2026-09-30"]
    assert pricing.model_price_level("2026-09-23", "list", sh, early) == \
        pricing.model_price_level("2026-09-23", "list", sh, prices)


def test_index_is_flat_before_first_measured_week(prices):
    sh = _shares([{"week_start": "2025-12-01", "Opus 4.5": 0.7, "Sonnet 4.5": 0.3},
                  {"week_start": "2026-07-13", "Opus 4.8": 0.7, "Sonnet 5": 0.3}])
    idx = pricing.model_price_index(["2025-06-01", "2025-11-25", "2025-12-08", "2026-07-20"], "list", sh, prices)
    assert list(idx.iloc[:3]) == [1.0, 1.0, 1.0]
    assert idx.iloc[3] == pytest.approx((0.7 * 25 + 0.3 * 10) / (0.7 * 25 + 0.3 * 15))


def test_cache_basket_cuts_more_than_list_for_the_5_5_models(prices):
    ratio_list = pricing.model_price("Opus 5.5", "list", prices) / pricing.model_price("Opus 5", "list", prices)
    ratio_cache = pricing.model_price("Opus 5.5", "cache", prices) / pricing.model_price("Opus 5", "cache", prices)
    assert ratio_list == pytest.approx(0.8) and ratio_cache < ratio_list
    # before the 5.5 models cache reads were 1/10 of input everywhere: both baskets give the same ratios
    assert pricing.model_price("Sonnet 5", "cache", prices) / pricing.model_price("Sonnet 4.6", "cache", prices) == \
        pytest.approx(10 / 15)


def test_adjust_usage_indicators_leaves_claims_and_spend_alone():
    df = data.load_all()
    idx = pd.Series(0.5, index=pd.DatetimeIndex(sorted(set(pd.to_datetime(df[df.metric == "indicator"].ref_date)))))
    adj = pricing.adjust_usage_indicators(df, 1.0, index=idx)
    claims = df.metric != "indicator"
    pd.testing.assert_frame_equal(adj[claims], df[claims])
    spend = df.indicator.isin(pricing.SPEND_INDICATORS)
    assert (adj.loc[spend, "value"] == df.loc[spend, "value"]).all()
    usage = (df.metric == "indicator") & ~spend
    assert np.allclose(adj.loc[usage, "value"], 0.5 * df.loc[usage, "value"])
    assert pricing.adjust_usage_indicators(df, 0.0).equals(df)


def test_gamma_weights_and_mixture():
    w = pricing.gamma_weights_from_loglik({0.0: -10.0, 0.5: -10.0, 1.0: -10.0 + math.log(2)})
    assert w.to_dict() == pytest.approx({0.0: 0.25, 0.5: 0.25, 1.0: 0.5})
    base = {"target": ["t1"], "metric": ["run_rate"], "vintage": ["2026-10-01"], "n_evidence": [5],
            "actual": [80.0], "naive": [70.0], "log_err_naive": [math.log(80 / 70)]}
    comps = {0.0: pd.DataFrame(dict(base, mu=[math.log(100)], sd=[0.1], ll_claims=[-50.0], ll_joint=[0.0])),
             1.0: pd.DataFrame(dict(base, mu=[math.log(80)], sd=[0.1], ll_claims=[-10.0], ll_joint=[-100.0]))}
    m = backtest.gamma_mixture(comps).iloc[0]
    assert m.w_gamma_1 > 0.999 and m.forecast == pytest.approx(80, rel=1e-6) and m.in80
    j = backtest.gamma_mixture(comps, likelihood="joint").iloc[0]
    assert j.w_gamma_0 > 0.999 and j.forecast == pytest.approx(100, rel=1e-6) and not j.in80


def test_paired_bootstrap_sign():
    a = pd.Series([0.2, 0.3, 0.1], index=list("xyz"))
    b = pd.Series([0.1, -0.2, 0.05], index=list("xyz"))
    r = backtest.paired_bootstrap(a, b)
    assert r["n"] == 3 and r["mean_diff_abs_err"] < 0 and r["targets_improved"] == 3


def test_github_indicator_rows_public_after_their_week():
    g = price_study.github_raw()
    assert (g.published == g.date + pd.Timedelta(days=1)).all()
    assert (g.date.dt.dayofweek == 6).all()                   # samples are Monday-Sunday weeks
    a = price_study.frame("A")
    assert (a.metric != "indicator").all() and (a.segment == "total").all()


def test_loglik_columns_see_only_data_public_at_the_vintage():
    """ll_claims and the forecast do not move when data published after the vintage change."""
    df = data.load_all()
    p = calibrate.load_params()
    one = ["rr_2026_05"]
    a = backtest.run_backtest(df, p, target_ids=one, loglik=True).iloc[0]
    v = pd.Timestamp(a.vintage)
    late = df.published > v
    df2 = df.copy()
    df2.loc[late, "value"] = df2.loc[late, "value"] * 3
    b = backtest.run_backtest(df2, p, target_ids=one, loglik=True).iloc[0]
    assert a.ll_claims == pytest.approx(b.ll_claims) and a.mu == pytest.approx(b.mu)
    assert not np.isclose(a.ll_claims, a.ll_joint)
