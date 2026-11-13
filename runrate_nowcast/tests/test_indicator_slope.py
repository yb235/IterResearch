"""Sub-task P: indicator slope (P1), fixed drift (P1'), beta prior sd (P1'') — indicator_slope.py.

1. Off by default, and then the filter is bit-identical to model.py as committed in 1ff145b (sub-task P
   registered, before any change), on claims-only and indicator frames, with and without the iv115 rule;
   calibrate.fit_indicators equals the committed calibrate.py.
2. The slope state: layout, transition, priors; with an indicator that grows faster than revenue, P1
   learns the excess growth and the baseline does not.
3. P1' and P1'': the flags reach the filter and the per-vintage refit.
4. Leak hygiene for P1: a forecast does not change when every row published after the vintage is deleted,
   or when the target's value is tripled (refit at the vintage, crossing profiled, both vintage rules,
   both evidence sets).
"""
from __future__ import annotations

import importlib.util
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from runrate_nowcast import backtest, calibrate, config, data, lower_rule
from runrate_nowcast import indicator_slope as isl
from runrate_nowcast.model import Model

REF_COMMIT = "1ff145b"          # DATA_PLAN 子任务 P registered; model.py / calibrate.py before the change


def _load_ref(tmp_path_factory, rel, name):
    try:
        code = subprocess.run(["git", "show", f"{REF_COMMIT}:runrate_nowcast/{rel}"], cwd=config.PKG_DIR,
                              capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {REF_COMMIT} not available")
    path = tmp_path_factory.mktemp("ref") / f"{name}.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location(f"runrate_nowcast.{name}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                 # dataclasses look the module up while it executes
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ref_model(tmp_path_factory):
    return _load_ref(tmp_path_factory, "model.py", "_model_ref_p")


@pytest.fixture(scope="module")
def ref_calibrate(tmp_path_factory):
    # relative imports resolve to the current package: config, data and model (identical by default, test 1)
    return _load_ref(tmp_path_factory, "calibrate.py", "_calibrate_ref_p")


@pytest.fixture
def iv115(monkeypatch):
    monkeypatch.setattr(config, "LOWER_UPPER_RATIO", lower_rule.setting(isl.LOWER))


def _frame(model):
    with isl.model_catalog(model):
        return isl.model_frame(model), calibrate.load_params()


@pytest.fixture(scope="module")
def frames():
    out = {m: _frame(m) for m in ("B", "OC", "G")}
    claims = data.load()
    out["A"] = (claims[(claims.segment == "total") & (claims.metric != "share")], calibrate.load_params(with_indicators=False))
    return out


# --------------------------------------------------------------------------- 1. default off
def test_defaults_are_off():
    for f in isl.FLAGS:
        assert getattr(config, f) is None
    assert isl.VARIANTS["P1"] == {"INDICATOR_SLOPE": {"b0_sd": 0.01, "slope_sd": 0.001}}
    assert isl.VARIANTS["P1p"] == {"INDICATOR_DRIFT": 0.02} and isl.VARIANTS["P1pp"] == {"INDICATOR_BETA_SD": 0.5}
    assert isl.PREREGISTERED == ("baseline", "P1", "P1p", "P1pp")


def test_variant_context_restores():
    with isl.variant("P1"):
        assert config.INDICATOR_SLOPE == isl.SLOPE and config.INDICATOR_DRIFT is None
    for f in isl.FLAGS:
        assert getattr(config, f) is None
    with pytest.raises(ValueError):
        isl.apply_variant("P2")


@pytest.mark.parametrize("rule", [None, 1.15])
@pytest.mark.parametrize("model,vintage", [("A", "2026-08-20"), ("B", "2025-10-20"), ("B", "2026-11-09"),
                                           ("OC", "2026-09-30"), ("G", "2026-06-30")])
def test_default_reproduces_committed_model(monkeypatch, ref_model, frames, model, vintage, rule):
    monkeypatch.setattr(config, "LOWER_UPPER_RATIO", rule)
    df, p = frames[model]
    sel = data.select(df, vintage=vintage)
    if model != "A":
        assert (sel.metric == "indicator").sum() > 5
    new = Model(p).run(sel, end="2026-12-31")
    old = ref_model.Model(p).run(sel, end="2026-12-31")
    assert new.layout.n == old.layout.n and not new.layout.slopes
    np.testing.assert_array_equal(new.ms, old.ms)
    np.testing.assert_array_equal(new.Ps, old.Ps)
    assert new.loglik == old.loglik and new.iterations == old.iterations
    assert [(i.obs_id, i.mu, i.sd, i.p_tail) for i in new.innovations] == \
        [(i.obs_id, i.mu, i.sd, i.p_tail) for i in old.innovations]
    f_new, f_old = Model(p).run(sel, smooth=False), ref_model.Model(p).run(sel, smooth=False)
    np.testing.assert_array_equal(f_new.m_f, f_old.m_f)
    np.testing.assert_array_equal(f_new.P_f, f_old.P_f)
    assert f_new.loglik == f_old.loglik


def test_default_fit_indicators_reproduces_committed(ref_calibrate, frames):
    df, _ = frames["G"]
    with isl.model_catalog("G"):
        vint = df[df.published <= pd.Timestamp("2026-03-01")]
        p = backtest.unfitted_params()
        out_new, p_new = calibrate.fit_indicators(vint, p, verbose=False)
        out_old, p_old = ref_calibrate.fit_indicators(vint, p, verbose=False)
    assert list(out_new) == ["github_repo_share_claude"]
    assert out_new == out_old and p_new == p_old


# --------------------------------------------------------------------------- 2. the slope state
def _synthetic(excess_per_week=0.02, weeks=80, growth_per_month=0.05):
    """Monthly official run-rate claims on a path growing growth_per_month, and a weekly indicator that grows
    excess_per_week faster (beta 1). No noise."""
    t0 = pd.Timestamp(config.MODEL_START)
    rows = []
    for k in range(0, weeks * 7, 30):
        d = t0 + pd.Timedelta(days=k + 15)
        v = 5.0 * math.exp(growth_per_month * (k + 15) / config.DAYS_PER_MONTH)
        rows.append(dict(obs_id=f"rr{k}", origin_id=f"rr{k}", metric="run_rate", kind="point", value=v, ref_date=d,
                         t_eff=d, period_start=pd.NaT, period_end=pd.NaT, definition="instant", date_sd_days=0.0,
                         published=d, tier="official", segment="total", indicator="", is_forecast=0))
    for w in range(weeks):
        d = t0 + pd.Timedelta(days=7 * w + 3)
        lv = math.log(5.0) + growth_per_month * (7 * w + 3) / config.DAYS_PER_MONTH + excess_per_week * (w + 3 / 7)
        rows.append(dict(obs_id=f"x{w}", origin_id=f"x{w}", metric="indicator", kind="point", value=math.exp(lv + 2.0),
                         ref_date=d, t_eff=d, period_start=pd.NaT, period_end=pd.NaT, definition="instant",
                         date_sd_days=0.0, published=d, tier="altdata", segment="total", indicator="x", is_forecast=0))
    df = pd.DataFrame(rows).sort_values("t_eff").reset_index(drop=True)
    params = {"indicators": {"x": {"segment": "total", "beta": 1.0, "beta_sd": 0.25, "obs_sd": 0.02, "drift": 0.001,
                                   "prior_sd": 5.0, "role": "model", "breaks": [], "factor": ""}}}
    return df, params


def test_slope_layout_and_transition():
    df, p = _synthetic()
    with isl.variant("P1"):
        m = Model(p)
        lay = m.layout(df, None)
        assert lay.slopes == ["x"]
        ctx = m._context(lay, df)
        j, k = lay.ind("x"), lay.slp("x")
        assert ctx["slopes"] == [(j, k)] and k < lay.acc0
        assert ctx["P0"][k, k] == pytest.approx(0.01 ** 2) and ctx["m0"][k] == 0.0
        assert ctx["Q"][k, k] == pytest.approx(0.001 ** 2 / 7)          # 0.001 per week
        assert ctx["Q"][j, j] == pytest.approx(0.001 ** 2)              # eta: the indicator's drift, as before
        M, _ = m._transition(ctx, 10)
        assert M[j, k] == pytest.approx(1 / 7) and M[k, k] == 1.0 and M[k, j] == 0.0
    base = Model(p).layout(df, None)
    assert base.slopes == [] and base.n == lay.n - 1


def test_slope_learns_excess_growth():
    df, p = _synthetic(excess_per_week=0.02)
    end = df.t_eff.max()
    with isl.variant("P1"):
        r1 = Model(p).run(df, smooth=False)
        b, bsd = r1.indicator_slope(end, "x", smoothed=False)
    assert b == pytest.approx(0.02, abs=0.003) and bsd < 0.005
    # the online forecast of the next indicator reading: P1 is close, the random-walk intercept lags
    r0 = Model(p).run(df, smooth=False)
    inn1 = [i for i in r1.innovations if i.metric == "indicator"][-20:]
    inn0 = [i for i in r0.innovations if i.metric == "indicator"][-20:]
    assert abs(np.mean([i.y - i.mu for i in inn1])) < 0.2 * abs(np.mean([i.y - i.mu for i in inn0]))
    with isl.variant("P1"):
        r_none = Model(p).run(_synthetic(excess_per_week=0.0)[0], smooth=False)
        assert abs(r_none.indicator_slope(end, "x", smoothed=False)[0]) < 0.003


# --------------------------------------------------------------------------- 3. P1' and P1''
def test_fixed_drift_reaches_filter_and_refit(frames):
    df, _ = _synthetic()
    p = _synthetic()[1]
    with isl.variant("P1p"):
        lay = Model(p).layout(df, None)
        ctx = Model(p)._context(lay, df)
        assert ctx["Q"][lay.ind("x"), lay.ind("x")] == pytest.approx(0.02 ** 2) and not lay.slopes
        g, _ = frames["G"]
        with isl.model_catalog("G"):
            vint = g[g.published <= pd.Timestamp("2026-03-01")]
            out, pp = calibrate.fit_indicators(vint, backtest.unfitted_params(), verbose=False)
    assert out["github_repo_share_claude"]["drift"] == 0.02
    assert pp["indicators"]["github_repo_share_claude"]["drift"] == 0.02


def test_beta_prior_sd_reaches_refit(frames):
    g, _ = frames["G"]
    with isl.model_catalog("G"):
        vint = g[g.published <= pd.Timestamp("2026-03-01")]
        p = backtest.unfitted_params()
        assert p["indicators"]["github_repo_share_claude"]["beta_sd"] == 1.0
        out0, _ = calibrate.fit_indicators(vint, p, verbose=False)
        with isl.variant("P1pp"):
            out1, p1 = calibrate.fit_indicators(vint, p, verbose=False)
    assert p1["indicators"]["github_repo_share_claude"]["beta_sd"] == 0.5
    b0, b1 = out0["github_repo_share_claude"]["beta"], out1["github_repo_share_claude"]["beta"]
    assert b0 != b1 and abs(b1 - 1.0) < abs(b0 - 1.0)           # tighter prior: closer to the prior mean 1


# --------------------------------------------------------------------------- 4. leak hygiene for P1
def _target(df, tid):
    full = data.select(df, filings=True)
    return next(full[full.obs_id == tid].itertuples())


def _p1_forecast(df, tid, rule, p, refit):
    row = _target(df, tid)
    v = backtest.target_vintage(row, rule)
    with isl.variant("P1"):
        mu, sd, n, ev = backtest.prepublication_forecast(df, row, p, refit, 3, None, "profile", vintage=v)
    return mu, sd, ev, v, row


def _cut_after(df, v, row):
    """Every row published after the vintage deleted, except the target's own origin (the target is the
    question; prepublication_forecast reads its dates, and its origin is never evidence)."""
    return df[(df.published <= v) | (df.origin_id == row.origin_id)]


CASES = [("G", "rr_2026_09_s1a", "period_end", True, False), ("G", "rev_2026q3", "publication", True, True),
         ("G", "rr_2026_05", "publication", True, False),
         ("B", "rr_2026_05", "period_end", False, False), ("B", "rev_2026q2", "publication", False, False),
         ("B", "rr_2026_10_s1a", "publication", False, True), ("OC", "rr_2026_09_s1a", "period_end", False, True)]


@pytest.mark.parametrize("model,tid,rule,refit,filings", CASES)
def test_p1_ignores_rows_after_vintage_and_target_value(monkeypatch, iv115, frames, model, tid, rule, refit, filings):
    monkeypatch.setattr(config, "USE_FILINGS", filings)
    df, p = frames[model]
    with isl.model_catalog(model):
        mu0, sd0, ev0, v, row = _p1_forecast(df, tid, rule, p, refit)
        assert (ev0.published <= v).all() and not (ev0.origin_id == row.origin_id).any()
        assert (df.published > v).sum() > len(df[df.origin_id == row.origin_id])     # something to delete
        cut = _cut_after(df, v, row)
        mu1, sd1, ev1, _, _ = _p1_forecast(cut, tid, rule, p, refit)
        x3 = df.copy()
        x3.loc[x3.origin_id == row.origin_id, "value"] *= 3
        mu2, sd2, _, _, _ = _p1_forecast(x3, tid, rule, p, refit)
        with isl.variant("baseline"):
            mu_b, _, _, _ = backtest.prepublication_forecast(df, row, p, refit, 3, None, "profile", vintage=v)
    assert list(ev1.obs_id) == list(ev0.obs_id)
    assert mu1 == mu0 and sd1 == sd0
    assert mu2 == pytest.approx(mu0, abs=1e-12) and sd2 == pytest.approx(sd0, abs=1e-12)
    if filings and v >= pd.Timestamp("2026-09-30"):
        assert (ev0.filing != 0).any()
    assert mu_b != mu0                                    # P1 is a different model
