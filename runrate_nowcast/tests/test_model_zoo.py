"""Model zoo (model_zoo.py): no look-ahead, AUTO chosen from training pairs only, every member
fits and predicts, and the shared pieces reproduce panel_model.py.

Run: OMP_NUM_THREADS=1 python -m pytest runrate_nowcast/tests -q
"""
import math

import numpy as np
import pandas as pd
import pytest

from runrate_nowcast import backtest
from runrate_nowcast import features as F
from runrate_nowcast import model_zoo as mz
from runrate_nowcast import panel_model as pm
from runrate_nowcast.model import L

DAY = pd.Timedelta(days=1)
NOISE = {"sigma_l": 0.004, "sigma_g": 0.01, "sigma_b": 0.002, "growth_reversion": 0.1}
FIXED = {c: NOISE for c in pm.COMPANIES}


@pytest.fixture(scope="module")
def claims():
    return {c: pm.load_claims(c) for c in pm.COMPANIES}


@pytest.fixture(scope="module")
def sources():
    return F.load_sources()


def _target(company, obs_id, **override):
    t = pm.load_targets(company).set_index("obs_id").loc[obs_id]
    return pm._Row(obs_id=obs_id, **{**t.to_dict(), **override})


# --------------------------------------------------------------------------- removing later data
def _remove_late_sources(src: dict, v: pd.Timestamp) -> dict:
    """Source tables without any row published after v (publication rules restated here,
    independently of features.py)."""
    out = {k: d.copy() for k, d in src.items()}
    ind = out["indicators"]
    pub = pd.to_datetime(ind["published"], errors="coerce").fillna(pd.to_datetime(ind["date"]) + DAY)
    out["indicators"] = ind[pub <= v].reset_index(drop=True)
    g = out["github_new_repos"]
    out["github_new_repos"] = g[pd.to_datetime(g["month"]) + pd.offsets.MonthBegin(1) <= v].reset_index(drop=True)
    gw = out["gateway"]
    out["gateway"] = gw[pd.to_datetime(gw["captured_or_published"], format="%Y-%m-%d", errors="coerce") <= v]
    cl = out["cloud"]
    out["cloud"] = cl[pd.to_datetime(cl["published"]) <= v].reset_index(drop=True)
    return out


def _remove_late_claims(claims: dict, v: pd.Timestamp) -> dict:
    return {c: d[d.published <= v].reset_index(drop=True) for c, d in claims.items()}


def _rows(out):
    return pd.DataFrame(out).set_index(["pool", "model"]).sort_index()


@pytest.fixture(scope="module")
def zoo_run(claims, sources):
    """One real Anthropic target (past its last anchor), every member with one-value grids,
    both pools: with all data, and with everything published after the vintage removed."""
    tg = _target("anthropic", "rr_2026_04")
    v = backtest.target_vintage(tg, "publication")
    full, fits = mz.forecast_target(tg, v, claims, sources, params=FIXED, grids=mz.SMALL_GRIDS, return_fits=True)
    cut = mz.forecast_target(tg, v, _remove_late_claims(claims, v), _remove_late_sources(sources, v),
                             params=FIXED, grids=mz.SMALL_GRIDS)
    return {"target": tg, "vintage": v, "full": full, "fits": fits, "cut": cut}


# --------------------------------------------------------------------------- no look-ahead
def test_forecasts_unchanged_without_later_data(zoo_run):
    a, b = _rows(zoo_run["full"]), _rows(zoo_run["cut"])
    assert list(a.index) == list(b.index)
    models = set(a.index.get_level_values("model"))
    assert set(mz.MEMBERS) | set(mz.COMBOS) <= models
    assert a.loc["pooled"].gap_weeks.iloc[0] > 0           # the target lies past t0: members differ
    np.testing.assert_allclose(a.forecast.to_numpy(float), b.forecast.to_numpy(float), rtol=1e-10)
    np.testing.assert_allclose(a.cv_loss.to_numpy(float), b.cv_loss.to_numpy(float), rtol=1e-10)
    assert (a.auto_choice.fillna("") == b.auto_choice.fillna("")).all()
    assert (a.stack_weights.fillna("") == b.stack_weights.fillna("")).all()
    assert np.isfinite(a.loc[("pooled", "AUTO"), "forecast"]) and np.isfinite(a.loc[("pooled", "stack"), "forecast"])


def test_training_pairs_end_before_vintage(claims, sources, monkeypatch):
    seen = []
    real = mz.fit_zoo

    def spy(P, members=mz.MEMBERS, grids=None):
        seen.append(P.copy())
        return real(P, ("trend", "ridge"), grids)
    monkeypatch.setattr(mz, "fit_zoo", spy)
    tg = _target("openai", "oa_rr_2026_09_ft")
    v = backtest.target_vintage(tg, "period_end")
    mz.forecast_target(tg, v, claims, sources, params=FIXED, members=("trend", "ridge"))
    assert len(seen) == 1                                   # OpenAI target: pooled only
    P = seen[0]
    assert set(P.company) == {"anthropic", "openai"} and (P.t1 <= v).all()
    t0 = pm.company_labels("openai", claims["openai"], v, exclude_origin=tg.origin_id, params=NOISE).t0
    assert (P[P.company == "openai"].t1 <= t0).all()


# --------------------------------------------------------------------------- AUTO
def test_auto_choice_is_argmin_of_cv_loss():
    f = {m: mz.Fitted(m, model=object(), cv_loss=l) for m, l in
         (("trend", 0.3), ("ridge", 0.1), ("lasso", 0.1), ("gp", math.inf))}
    assert mz.auto_choice(f, ["trend", "ridge", "lasso", "gp"]) == "ridge"       # tie -> first in order
    assert mz.auto_choice(f, ["trend", "lasso", "ridge", "gp"]) == "lasso"
    g = {m: mz.Fitted(m, model=object(), cv_loss=math.inf) for m in ("ridge", "trend")}
    assert mz.auto_choice(g, ["ridge", "trend"]) == "trend"                      # no CV: the fallback
    h = {"ridge": mz.Fitted("ridge", model=None, cv_loss=0.0), "trend": mz.Fitted("trend", model=object(), cv_loss=1.0)}
    assert mz.auto_choice(h, ["ridge", "trend"]) == "trend"                      # failed fits never chosen


def test_auto_never_uses_test_targets(zoo_run, claims, sources):
    """AUTO = the member with the lowest CV loss on training pairs; the target's actual value (and
    any other test target's) cannot move it."""
    a = _rows(zoo_run["full"])
    for pool in ("pooled", "own"):
        g = a.loc[pool]
        mem = g.loc[[m for m in mz.MEMBERS if m in g.index]]
        choice = g.loc["AUTO", "auto_choice"]
        assert choice == mz.auto_choice(zoo_run["fits"][pool], list(mz.MEMBERS))
        assert mem.cv_loss.astype(float).min() == pytest.approx(float(mem.loc[choice, "cv_loss"]))
        assert g.loc["AUTO", "forecast"] == pytest.approx(mem.loc[choice, "forecast"])
    # the same target with an absurd actual value: same choice, same forecasts
    tg = zoo_run["target"]
    wild = pm._Row(**{**tg.__dict__, "value": tg.value * 100.0})
    b = _rows(mz.forecast_target(wild, zoo_run["vintage"], claims, sources, params=FIXED, grids=mz.SMALL_GRIDS))
    np.testing.assert_allclose(a.forecast.to_numpy(float), b.forecast.to_numpy(float), rtol=1e-12)
    assert (a.auto_choice.fillna("") == b.auto_choice.fillna("")).all()
    assert not np.allclose(a.log_err.to_numpy(float), b.log_err.to_numpy(float))


def test_stack_weights_nonnegative_sum_to_one(zoo_run):
    for pool in zoo_run["fits"]:
        rows = _rows(zoo_run["full"]).loc[(pool, "stack")]
        w = dict(kv.split("=") for kv in rows.stack_weights.split(";"))
        w = {k: float(x) for k, x in w.items()}
        assert all(x >= 0 for x in w.values()) and abs(sum(w.values()) - 1) < 0.01
        assert set(w) <= set(mz.MEMBERS)


# --------------------------------------------------------------------------- synthetic panel
class _FakeRes:
    def __init__(self, path: pd.Series):
        self.path = path

    def state(self, d):
        x = (self.path.index - self.path.index[0]).days.to_numpy(float)
        v = float(np.interp((pd.Timestamp(d) - self.path.index[0]).days, x, self.path.to_numpy()))
        m = np.zeros(3)
        m[L] = v
        return m, np.zeros((3, 3))


def _synthetic_company(company, fe, beta, seed, weeks=80, t0_offset=0, lagged=False):
    """Weekly log growth = fe + beta * (weekly change of OpenRouter log tokens), or with lagged=True
    a linearly decaying distributed lag of that change over 0..13 weeks."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-02", periods=weeks, freq="7D")
    dx = rng.normal(0, 0.06, weeks)
    flev = pd.DataFrame({f: np.cumsum(rng.normal(0, 0.01, weeks)) for f in pm.FEATURES}, index=idx)
    flev["openrouter_log_tokens"] = np.cumsum(dx)
    flev.iloc[:3, pm.FEATURES.index("pypi_level")] = np.nan          # some missing values
    if lagged:
        J = mz.MIDAS_LAGS
        k = (1 - np.arange(J + 1) / J) / np.sum(1 - np.arange(J + 1) / J)
        drive = np.convolve(dx, k)[:weeks]
    else:
        drive = dx
    g = fe + beta * np.r_[0.0, drive[1:]] + rng.normal(0, 0.001, weeks)
    y = pd.Series(np.cumsum(g), index=idx)
    t0 = idx[-1 - t0_offset]
    lab = pm.Labels(company, idx[-1], None, y[y.index <= t0], y[y.index <= t0],
                    pd.Series(0.05, index=y.index[y.index <= t0]), t0, idx[0], res=_FakeRes(y))
    return lab, flev, y


def _panel(lagged=False):
    la, fa, ya = _synthetic_company("anthropic", 0.03, 0.6, seed=1, t0_offset=13, lagged=lagged)
    lo, fo, _ = _synthetic_company("openai", 0.01, 0.6, seed=2, lagged=lagged)
    P = pd.concat([mz.make_pairs(la, fa), mz.make_pairs(lo, fo)], ignore_index=True)
    return P, la, fa, ya


@pytest.fixture(scope="module")
def synth():
    return _panel()


@pytest.mark.parametrize("member", mz.MEMBERS)
def test_every_member_fits_and_predicts(synth, member):
    P, la, fa, ya = synth
    folds = pm.cv_folds(P)
    assert len(folds) >= 3
    f = mz.fit_member(member, P, folds, mz.SMALL_GRIDS)
    assert f.ok, f.error
    assert np.isfinite(f.cv_loss) and f.hyper in [dict(h) for h in mz.GRIDS[member]]
    assert np.isfinite(f.oof).sum() == sum(len(va) for _, va in folds)
    tg = pm._Row(metric="run_rate", t_eff=la.t0 + 8 * pm.WEEK)
    fc = mz.path_forecast(la, fa, tg, f.model.predict)
    assert np.isfinite(fc) and fc > 0
    rev = pm._Row(metric="revenue", period_start=la.t0 - 20 * DAY, period_end=la.t0 + 60 * DAY)
    assert np.isfinite(mz.path_forecast(la, fa, rev, f.model.predict))
    # a deterministic refit gives the same model
    np.testing.assert_allclose(f.model.predict(P), mz.fit_member(member, P, folds, mz.SMALL_GRIDS).model.predict(P),
                               rtol=1e-9)


def test_linear_members_learn_the_signal(synth):
    P, la, fa, ya = synth
    fits = mz.fit_zoo(P, ("trend", "ridge", "lasso", "midas", "pls", "bayes_ridge"))
    for m in ("ridge", "lasso", "midas", "pls", "bayes_ridge"):
        assert fits[m].cv_loss < 0.5 * fits["trend"].cv_loss, m
    d = la.t0 + 8 * pm.WEEK
    fc = mz.path_forecast(la, fa, pm._Row(metric="run_rate", t_eff=d), fits["ridge"].model.predict)
    assert abs(math.log(fc / math.exp(ya[d]))) < 0.02
    w = mz.stack_weights(fits, P)
    assert abs(sum(w.values()) - 1) < 1e-9 and min(w.values()) >= 0
    assert w["trend"] < 0.05
    assert mz.auto_choice(fits) != "trend"


def test_midas_finds_a_distributed_lag():
    P, la, fa, ya = _panel(lagged=True)
    fits = mz.fit_zoo(P, ("ridge", "midas"))
    assert fits["midas"].cv_loss < 0.85 * fits["ridge"].cv_loss


def test_midas_terms_match_a_direct_sum():
    rng = np.random.default_rng(5)
    Lv = np.cumsum(rng.normal(size=(40, 3)), axis=0)
    i0, i1, n = 20, 28, 8.0
    J = mz.MIDAS_LAGS
    want = [sum((j / J) ** p * (Lv[i1 - j, f] - Lv[i0 - j, f]) / n for j in range(1, J + 1))
            for f in range(3) for p in range(mz.MIDAS_DEG + 1)]
    np.testing.assert_allclose(mz._midas_values(Lv, i0, i1, n), want, rtol=1e-12)
    assert np.isnan(mz._midas_values(Lv, 5, 9, 4.0)).all()           # lags before the first week
    # the forecast rows use the same window rule as the training pairs
    P, la, fa, _ = _panel()
    t = la.y.index[-9]
    G = mz.growth_rows(pm.Labels("anthropic", la.vintage, None, la.anchors, la.y, la.sd, t, la.start, res=la.res),
                       fa, [t + 8 * pm.WEEK])
    p = P[(P.company == "anthropic") & (P.t == t) & (P.h == 8)].iloc[0]
    cols = mz.midas_cols(pm.FEATURES) + pm.FEATURES
    np.testing.assert_allclose(G[cols].iloc[0].to_numpy(float), p[cols].to_numpy(float), rtol=1e-12)


# --------------------------------------------------------------------------- same as panel_model
def test_ridge_trend_pca_hgb_reproduce_panel_model(synth):
    P, la, fa, _ = synth
    Pn = pm._normalise(P)
    for kind, hyp in (("ridge", {"alpha": 10.0}), ("trend", {"alpha": 1.0}), ("pca", {"alpha": 10.0, "k": 2})):
        a = pm.build(kind, hyp).fit(Pn).predict(P)
        b = mz.build(kind, hyp).fit(Pn).predict(P)
        np.testing.assert_allclose(a, b, rtol=1e-8, atol=1e-12)
    hyp = {"max_depth": 2, "max_iter": 50}
    np.testing.assert_allclose(pm.build("gbm", hyp).fit(Pn).predict(P), mz.build("hgb", hyp).fit(Pn).predict(P))


def test_features_pairs_and_path_reproduce_panel_model(claims, sources):
    tg = _target("anthropic", "rev_2026q3")
    v = backtest.target_vintage(tg, "period_end")
    lab = pm.company_labels("anthropic", claims["anthropic"], v, exclude_origin=tg.origin_id, params=NOISE,
                            end=tg.period_end, queries=[(tg.period_start, tg.period_end)])
    a, b = pm.feature_levels("anthropic", v, sources), mz.feature_levels("anthropic", v, sources)
    pd.testing.assert_frame_equal(a, b)
    pa, pb = pm.make_pairs(lab, a), mz.make_pairs(lab, b)
    pd.testing.assert_frame_equal(pa, pb[pa.columns])
    m = pm.fit_model("ridge", pa)
    assert mz.path_forecast(lab, b, tg, m.predict) == pytest.approx(pm.path_forecast(lab, a, tg, m.predict), rel=1e-12)


def test_feature_spec_rule(monkeypatch):
    assert mz.feature_spec("g_spec") == pm.FEATURE_SPEC
    monkeypatch.setattr(F, "FEATURE_SETS", {"x": ["openrouter_log_tokens", "openrouter_dlog_tokens", "npm_dlog",
                                                  "ramp_adoption_chg", "cloud_bedrock_growth", "new_share",
                                                  "new_share_chg"]}, raising=False)
    s = mz.feature_spec("x")
    assert s == {"openrouter_log_tokens": ("openrouter_log_tokens", "level"), "npm_level": ("npm_dlog", "cumsum"),
                 "ramp_adoption_chg_cum": ("ramp_adoption_chg", "cumsum"),
                 "cloud_bedrock_growth_cum": ("cloud_bedrock_growth", "cumsum"), "new_share": ("new_share", "level")}
    with pytest.raises(ValueError):
        mz.feature_spec("nonexistent")
