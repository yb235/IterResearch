"""Sub-task N: the rule for "surpassed $X" claims (kind = lower), config.LOWER_UPPER_RATIO / lower_rule.py.

1. Off by default, and then the filter is exactly the committed (pre-N) model.py.
2. The two-sided update is the truncated-normal moment match (scipy, rejection sampling).
3. With a rule on, every non-forecast lower run-rate / revenue claim gets an upper side r X derived
   from the claim itself, in the same compiled observation as its lower side (so the same row, the
   same `published`); no evidence row is added, the targets do not change, check_no_leak holds.
4. A lower claim no longer allows a posterior far above r X.
5. The rule reaches the other consumers of the filter (panel_model / ml_fixed labels, bayes).
"""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import truncnorm

from runrate_nowcast import backtest, bayes, calibrate, config, data, lower_rule
from runrate_nowcast import panel_model as pm
from runrate_nowcast.model import Model, Obs, log_ndtr_diff, lower_claim_rule

REF_COMMIT = "abce18a"          # DATA_PLAN 子任务 N registered; model.py before the rule existed
D0 = pd.Timestamp("2026-01-01")


@pytest.fixture(scope="module")
def ref_model(tmp_path_factory):
    """model.py as committed in REF_COMMIT, importable next to the current package."""
    try:
        code = subprocess.run(["git", "show", f"{REF_COMMIT}:runrate_nowcast/model.py"], cwd=config.PKG_DIR,
                              capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {REF_COMMIT} not available")
    path = tmp_path_factory.mktemp("ref") / "model_ref.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location("runrate_nowcast._model_ref", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                 # dataclasses look the module up while it executes
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def claims():
    df = data.load()
    return df[(df.segment == "total") & (df.metric != "share")]


@pytest.fixture(scope="module")
def params():
    return calibrate.load_params(with_indicators=False)


def _variants():
    return list(lower_rule.ORDER)


# --------------------------------------------------------------------------- 1. default off
def test_default_is_off():
    assert config.LOWER_UPPER_RATIO is None
    assert lower_rule.VARIANTS["baseline"] is None and lower_rule.PRIMARY == "iv115"
    assert lower_rule.VARIANTS == {"baseline": None, "point": "point", "iv110": 1.10, "iv115": 1.15, "iv125": 1.25}


@pytest.mark.parametrize("vintage", ["2025-10-20", "2026-08-20", "2026-11-09"])
def test_default_reproduces_committed_model(ref_model, claims, params, vintage):
    """Rule off: smoothed path, online likelihood and innovations equal the pre-N model exactly."""
    sel = data.select(claims, vintage=vintage)
    assert (sel.kind == "lower").any()
    new = Model(params).run(sel, end="2026-12-31")
    old = ref_model.Model(params).run(sel, end="2026-12-31")
    np.testing.assert_array_equal(new.ms, old.ms)
    np.testing.assert_array_equal(new.Ps, old.Ps)
    assert new.loglik == old.loglik
    assert [(i.obs_id, i.kind, i.p_tail) for i in new.innovations] == [(i.obs_id, i.kind, i.p_tail) for i in old.innovations]
    f_new, f_old = Model(params).run(sel, smooth=False), ref_model.Model(params).run(sel, smooth=False)
    assert f_new.loglik == f_old.loglik


def test_rule_on_then_off_restores(claims, params):
    sel = data.select(claims, vintage="2026-11-09")
    before = Model(params).run(sel, smooth=False).loglik
    with lower_rule.lower_rule("iv115"):
        on = Model(params).run(sel, smooth=False).loglik
    assert config.LOWER_UPPER_RATIO is None
    assert on != before and Model(params).run(sel, smooth=False).loglik == before


def test_bad_setting_rejected(claims, params):
    sel = data.select(claims, vintage="2026-11-09")
    with pytest.raises(ValueError):
        lower_rule.apply("iv200")
    old = config.LOWER_UPPER_RATIO
    try:
        config.LOWER_UPPER_RATIO = 0.9
        with pytest.raises(ValueError):
            Model(params).run(sel, smooth=False)
    finally:
        config.LOWER_UPPER_RATIO = old


# --------------------------------------------------------------------------- 2. the update
@pytest.mark.parametrize("a,b", [(-3, -2.5), (-1, 0.5), (0.2, 0.4), (5, 5.3), (20, 20.3), (-40, -39.6), (3, 30)])
def test_interval_update_matches_truncnorm(a, b):
    o = Obs("x", 0, "run_rate", "lower", a, np.array([1.0, 0.0, 0.0]), 0.0, 0.0, 0.0, "official", y_hi=b)
    m, P, inn = Model()._update(np.zeros(3), np.eye(3), o, D0)
    tm, tv = truncnorm.stats(a, b, moments="mv")
    assert m[0] == pytest.approx(float(tm), rel=1e-7, abs=1e-9)
    assert P[0, 0] == pytest.approx(float(tv), rel=1e-6)
    assert m[1:] == pytest.approx([0.0, 0.0]) and P[1:, 1:] == pytest.approx(np.eye(2))
    assert inn.loglik == pytest.approx(log_ndtr_diff(b, a)) and inn.kind == "lower"


def test_log_ndtr_diff_tails():
    from scipy.stats import norm
    for a, b in [(-1.0, 2.0), (0.1, 0.2), (-0.3, -0.1)]:
        assert log_ndtr_diff(b, a) == pytest.approx(math.log(norm.cdf(b) - norm.cdf(a)), rel=1e-10)
    assert np.isfinite(log_ndtr_diff(40.3, 40.0)) and np.isfinite(log_ndtr_diff(-40.0, -40.3))


def test_interval_update_matches_rejection_sampling():
    rng = np.random.default_rng(3)
    m = np.array([3.4, 0.1, 0.05])
    Araw = rng.normal(size=(3, 3))
    P = Araw @ Araw.T * 0.01 + np.diag([0.04, 0.01, 0.01])
    H = np.array([1.0, 0.0, 1.0])
    lo, hi = math.log(25), math.log(25 * 1.15)
    o = Obs("x", 0, "run_rate", "lower", lo, H, 0.0, 0.05 ** 2, 0.0, "official", y_hi=hi)
    mm, PP, inn = Model()._update(m.copy(), P.copy(), o, D0)
    x = rng.multivariate_normal(m, P, 2_000_000)
    z = x @ H + rng.normal(0, 0.05, len(x))
    keep = x[(z > lo) & (z < hi)]
    assert inn.p_tail == pytest.approx(len(keep) / len(x), rel=0.02)
    assert mm == pytest.approx(keep.mean(0), abs=3e-3)
    assert PP == pytest.approx(np.cov(keep.T), abs=2e-3)


# --------------------------------------------------------------------------- 3. compiled observations
def _compiled(df, vintage, variant, start=config.MODEL_START):
    sel = data.select(df, vintage=vintage, forecasts=True)       # forecasts too: they must stay one-sided
    sel = sel[~((sel.metric == "revenue") & (sel.period_start < pd.Timestamp(start)))]
    model = Model(backtest.unfitted_params(), start=start)
    with lower_rule.lower_rule(variant):
        return sel, {o.obs_id: o for o in model._compile(sel, model.layout(sel, None))}


@pytest.mark.parametrize("company", ["anthropic", "openai"])
@pytest.mark.parametrize("variant", ["iv110", "iv115", "iv125", "point"])
def test_upper_side_on_every_lower_claim(company, variant):
    df = data.load(pm.CLAIMS_PATHS[company])
    df = df[df.metric != "indicator"]
    sel, obs = _compiled(df, "2026-11-13", variant, pm.MODEL_STARTS[company])
    _, base = _compiled(df, "2026-11-13", "baseline", pm.MODEL_STARTS[company])
    r = lower_rule.VARIANTS[variant]
    lowers = sel[(sel.kind == "lower") & sel.metric.isin(["run_rate", "revenue"]) & (sel.is_forecast == 0)]
    lowers = lowers[lowers.obs_id.isin(obs)]
    assert len(lowers) >= 3
    for row in sel.itertuples():
        if row.obs_id not in obs:
            continue
        o, o0 = obs[row.obs_id], base[row.obs_id]
        # same row, same day (the date comes from the row: the upper side shares its `published` and t_eff)
        assert (o.t, o.y, o.base_var, o.date_sd) == (o0.t, o0.y, o0.base_var, o0.date_sd)
        assert o0.y_hi is None and o0.kind == row.kind
        if row.obs_id in set(lowers.obs_id):
            if variant == "point":
                assert o.kind == "point" and o.y_hi is None
            elif row.metric == "run_rate":
                assert o.kind == "lower" and o.y_hi == pytest.approx(o.y + math.log(r)) and o.y == math.log(row.value)
            else:
                assert o.kind == "lower" and o.y_hi == pytest.approx(r * row.value) and o.y == row.value
        else:                                    # points, upper bounds and forecasts are untouched
            assert o.kind == row.kind and o.y_hi is None


def test_forecast_lower_claims_stay_one_sided():
    row = pd.Series({"kind": "lower", "metric": "run_rate", "is_forecast": 1})
    with lower_rule.lower_rule("iv115"):
        assert lower_claim_rule(row, 4.0) == ("lower", None)
        now = pd.Series({"kind": "lower", "metric": "run_rate", "is_forecast": 0})
        assert lower_claim_rule(now, 4.0)[1] == pytest.approx(4.0 + math.log(1.15))
        assert lower_claim_rule(pd.Series({"kind": "lower", "metric": "share", "is_forecast": 0}), 4.0) == ("lower", None)
        assert lower_claim_rule(pd.Series({"kind": "upper", "metric": "run_rate", "is_forecast": 0}), 4.0) == ("upper", None)


@pytest.mark.parametrize("vintage", ["2025-09-10", "2026-08-20", "2026-11-09", "2026-11-13"])
def test_evidence_rows_unchanged(claims, vintage):
    """No row is added or edited: data.select returns the same frame under every variant."""
    base, full = data.select(claims, vintage=vintage), data.select(claims, filings=True)
    for v in _variants():
        with lower_rule.lower_rule(v):
            pd.testing.assert_frame_equal(data.select(claims, vintage=vintage), base)
            pd.testing.assert_frame_equal(data.select(claims, filings=True), full)


@pytest.mark.parametrize("rule", ["publication", "period_end"])
def test_targets_unchanged_and_no_leak(monkeypatch, claims, rule):
    """run_backtest picks the same targets under every variant; every evidence set passes
    check_no_leak (called inside prepublication_forecast); no lower claim is ever a target."""
    seen = {}

    class Rec:
        def __init__(self, p=None, start=config.MODEL_START):
            pass

        def run(self, ev, end=None, queries=None, segments=None):
            seen[cur].append((ev.obs_id.tolist(), ev.published.max()))
            return None

    monkeypatch.setattr(backtest, "Model", Rec)
    monkeypatch.setattr(backtest, "observation_predictive", lambda res, row, model: (1.0, 1.0))
    out = {}
    for v in _variants():
        cur = v
        seen[v] = []
        with lower_rule.lower_rule(v):
            bt = backtest.run_backtest(claims, None, vintage_rule=rule)
        out[v] = bt[["target", "vintage"]]
    for v in _variants():
        pd.testing.assert_frame_equal(out[v], out["baseline"])
        assert seen[v] == seen["baseline"]
    lowers = set(claims[claims.kind == "lower"].obs_id)
    assert len(out["baseline"]) >= 10 and not (set(out["baseline"].target) & lowers)
    assert all(pd.Timestamp(pub) <= pd.Timestamp(vint) for (ids, pub), vint
               in zip(seen["iv115"], out["iv115"].vintage))


# --------------------------------------------------------------------------- 4. the cap
def test_lower_claim_caps_posterior(claims, params):
    """At vintage 2026-11-09 the evidence holds WSJ ">$80B at end of October" (rr_2026_10_wsj). One-sided,
    the claims-only model extrapolates H1 growth far above it; with [X, 1.15X] it stays near the interval."""
    full = data.select(claims, filings=True)
    row = full[full.obs_id == "rr_2026_10_s1a"].iloc[0]
    res = {}
    for v in _variants():
        with lower_rule.lower_rule(v):
            mu, sd, n, ev = backtest.prepublication_forecast(claims, row, params, refit=False)
        assert "rr_2026_10_wsj" in set(ev.obs_id)
        res[v] = math.exp(mu)
    assert res["baseline"] > 1.5 * 80
    assert res["point"] == pytest.approx(80, rel=0.05)
    for v, r in (("iv110", 1.10), ("iv115", 1.15), ("iv125", 1.25)):
        assert 80 * 0.98 < res[v] < r * 80 * 1.02, (v, res[v])
    assert res["point"] < res["iv110"] < res["iv115"] < res["iv125"] < res["baseline"]


def test_smoothed_path_inside_interval(claims, params):
    sel = data.select(claims, vintage="2026-11-09")
    lo = sel[(sel.kind == "lower") & (sel.metric == "run_rate")]
    with lower_rule.lower_rule("iv115"):
        res = Model(params).run(sel, end="2026-12-31")
    for r in lo.itertuples():
        mu, sd = res.log_runrate(r.ref_date, r.definition)
        assert math.log(r.value) - 0.05 < mu < math.log(1.15 * r.value) + 0.05, (r.obs_id, math.exp(mu))


# --------------------------------------------------------------------------- 5. other consumers
def test_rule_reaches_panel_labels():
    """pm.company_labels (panel_model and ml_fixed labels) runs Model on the company's claims:
    OpenAI's lower claims (revenue and run-rate) change its label path under the rule."""
    cl = pm.load_claims("openai")
    assert ((cl.kind == "lower") & (cl.metric == "revenue")).any()
    noise = {"sigma_l": 0.004, "sigma_g": 0.01, "sigma_b": 0.002, "growth_reversion": 0.1}
    base = pm.company_labels("openai", cl, "2026-06-30", params=noise)
    with lower_rule.lower_rule("iv115"):
        on = pm.company_labels("openai", cl, "2026-06-30", params=noise)
    assert base.t0 == on.t0 and list(base.anchors.index) == list(on.anchors.index)   # anchors are point claims
    assert not np.allclose(base.y.to_numpy(), on.y.to_numpy())


def test_rule_reaches_bayes_likelihood(claims):
    sel = data.select(claims, vintage="2026-11-09")
    p = bayes.base_params({})
    spec = bayes.build_spec(sel, p, "2026-12-31")
    th = bayes.theta0(spec)
    base = bayes._neg_logpost(th, sel, p, spec)
    with lower_rule.lower_rule("iv115"):
        on = bayes._neg_logpost(th, sel, p, spec)
    assert np.isfinite(base) and np.isfinite(on) and on != base
