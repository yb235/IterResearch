"""Corrected ML pipeline (ml_fixed.py): no look-ahead (including the nested-CV fold labels), the crossing
position and target dates as backtest.prepublication_forecast profiles them, ridge toward prior means,
and clipping / standardization on observed rows.

Run: OMP_NUM_THREADS=1 python -m pytest runrate_nowcast/tests -q
"""
import math

import numpy as np
import pandas as pd
import pytest

from runrate_nowcast import backtest, data, robust
from runrate_nowcast import features as F
from runrate_nowcast import ml_fixed as mf
from runrate_nowcast import panel_model as pm

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


def _remove_late_sources(src: dict, v: pd.Timestamp) -> dict:
    """Source tables without any row published after v (publication rules restated independently)."""
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
def run(claims, sources):
    """A real Anthropic target past its last anchor (a crossing claim), all models and ablations, with
    all data and with everything published after the vintage removed (caches cleared in between)."""
    tg = _target("anthropic", "rr_2026_04")
    v = backtest.target_vintage(tg, "publication")
    mf.clear_caches()
    full = mf.forecast_target(tg, v, claims, sources, params=FIXED)
    mf.clear_caches()
    cut = mf.forecast_target(tg, v, _remove_late_claims(claims, v), _remove_late_sources(sources, v), params=FIXED)
    return {"target": tg, "vintage": v, "full": full, "cut": cut}


# --------------------------------------------------------------------------- no look-ahead
def test_forecasts_unchanged_without_later_data(run):
    a, b = _rows(run["full"]), _rows(run["cut"])
    assert list(a.index) == list(b.index)
    models = set(a.index.get_level_values("model"))
    assert {mf.M1, mf.M2, mf.SS_NAME, mf.ABL_PRIOR0, mf.ABL_INNER, mf.ABL_OLDPREP, mf.ABL_RHO05, mf.ABL_PRIOR3} <= models
    assert bool(a.loc[("pooled", mf.M1), "post_t0"])               # the target lies past t0
    np.testing.assert_allclose(a.forecast.to_numpy(float), b.forecast.to_numpy(float), rtol=1e-10)
    np.testing.assert_allclose(a.lam.to_numpy(float), b.lam.to_numpy(float))
    assert (a.cv_curve.fillna("") == b.cv_curve.fillna("")).all()
    assert (a.rho.to_numpy(float) == b.rho.to_numpy(float)).all()
    assert a.loc[("pooled", mf.M1), "n_folds"] >= 1                 # the nested CV actually ran


def test_nested_fold_labels_use_only_claims_before_the_block(run, claims, sources, monkeypatch):
    """Each fold's training pairs are identical when built from claims / sources truncated at the fold
    vintage u = b0 - 1 day; they end >= 13 weeks before the block; every label run inside the folds sees
    only claims published on or before u."""
    tg, v = run["target"], run["vintage"]
    rho = mf.crossing_rho("anthropic", claims["anthropic"], v, tg.origin_id)
    P_t = mf.make_pairs(mf.labels_at("anthropic", claims["anthropic"], v, tg.origin_id, NOISE, rho=rho),
                        mf.feature_levels("anthropic", v, sources))
    seen = []
    real = pm.company_labels

    def spy(company, cl, vintage, *a, **k):
        lab = real(company, cl, vintage, *a, **k)
        seen.append((pd.Timestamp(vintage), pd.to_datetime(lab.sel.published).max() if len(lab.sel) else None))
        return lab
    monkeypatch.setattr(pm, "company_labels", spy)
    folds = mf.nested_folds(tg, P_t, claims, sources, list(pm.COMPANIES), FIXED)
    assert len(folds) == 4 and seen
    assert all(m is None or m <= u for u, m in seen)
    monkeypatch.setattr(pm, "company_labels", real)
    n_rows = 0
    for f in folds:
        assert f.u == f.b0 - DAY and f.u < v
        cut_claims, cut_src = _remove_late_claims(claims, f.u), _remove_late_sources(sources, f.u)
        mf.clear_caches()
        for c in pm.COMPANIES:
            P = f.train[c]
            n_rows += len(P)
            if len(P):
                assert (P.t1 <= f.b0 - mf.PURGE_WEEKS * pm.WEEK).all()
            ex = tg.origin_id if c == "anthropic" else None
            lab = mf.labels_at(c, cut_claims[c], f.u, ex, NOISE)
            Q = mf.make_pairs(lab, mf.feature_levels(c, f.u, cut_src))
            Q = Q[Q.t1 <= f.b0 - mf.PURGE_WEEKS * pm.WEEK].reset_index(drop=True) if len(Q) else Q
            assert len(P) == len(Q)
            if len(P):
                pd.testing.assert_frame_equal(P.reset_index(drop=True), Q.reset_index(drop=True))
    assert n_rows > 0


def test_actual_value_never_used(run, claims, sources):
    tg = run["target"]
    wild = pm._Row(**{**tg.__dict__, "value": tg.value * 100.0})
    b = _rows(mf.forecast_target(wild, run["vintage"], claims, sources, params=FIXED))
    a = _rows(run["full"])
    np.testing.assert_allclose(a.forecast.to_numpy(float), b.forecast.to_numpy(float), rtol=1e-12)
    np.testing.assert_allclose(a.lam.to_numpy(float), b.lam.to_numpy(float))


# --------------------------------------------------------------------------- crossing position
def test_rho_and_target_date_as_prepublication_forecast(run, claims):
    tg, v = run["target"], run["vintage"]
    a = data.load()
    A = a[(a.segment == "total") & (a.metric != "share")]
    rho_ref = robust.best_crossing_position(A, backtest.unfitted_params(), v, exclude_origin=tg.origin_id)
    rho = mf.crossing_rho("anthropic", claims["anthropic"], v, tg.origin_id)
    assert rho == pytest.approx(rho_ref, abs=1e-12) and abs(rho - 0.5) > 0.05
    df = data.with_crossing_position(A, rho_ref)                     # prepublication_forecast's re-dating
    ref_row = df[df.obs_id == tg.obs_id].assign(t_eff=lambda x: x.apply(data.effective_date, axis=1)).iloc[0]
    assert pd.Timestamp(mf.target_at_rho(tg, rho).t_eff) == pd.Timestamp(ref_row.t_eff)
    r = _rows(run["full"])
    assert r.loc[("pooled", mf.M1), "rho"] == pytest.approx(rho)
    assert str(r.loc[("pooled", mf.M1), "t_eff"]) == str(pd.Timestamp(ref_row.t_eff).date())
    assert str(r.loc[("pooled", mf.ABL_RHO05), "t_eff"]) != str(r.loc[("pooled", mf.M1), "t_eff"])


def test_openai_rho_profile(claims):
    v = pd.Timestamp("2026-06-30")
    rho = mf.crossing_rho("openai", claims["openai"], v)
    assert min(robust.RHOS) <= rho <= max(robust.RHOS)
    late = claims["openai"].assign(published=claims["openai"].published + pd.Timedelta(days=3650))
    both = pd.concat([claims["openai"][claims["openai"].published <= v], late[late.date_kind == "crossing"]
                      .assign(obs_id=lambda x: x.obs_id + "_late", origin_id=lambda x: x.origin_id + "_late")])
    mf.clear_caches()
    assert mf.crossing_rho("openai", both, v) == pytest.approx(rho, abs=1e-12)   # later claims ignored


# --------------------------------------------------------------------------- features, M2, pre-t0
def test_feature_levels_extend_panel_features(sources):
    v = pd.Timestamp("2026-05-27")
    for c in pm.COMPANIES:
        a, b = mf.feature_levels(c, v, sources), pm.feature_levels(c, v, sources)
        pd.testing.assert_frame_equal(a[pm.FEATURES], b[pm.FEATURES])
        price = b["log_price"] if c == "anthropic" else 0.0
        np.testing.assert_allclose(a["or_spend"], b["openrouter_log_tokens"] + price)
        np.testing.assert_allclose(a["log_gh_share"], np.log(b["github_repo_share"]))
        np.testing.assert_allclose(a["log_ramp_adoption"], np.log(b["ramp_adoption"]))


def test_contributions_add_up(run):
    r = _rows(run["full"])
    for pool in ("pooled", "own"):
        x = r.loc[(pool, mf.M1)]
        if x.lam >= 1e5:                    # only the company mean and the proxies are then non-negligible
            tot = x.rate_base + np.nansum([x[f"contrib_{k}"] for k in mf.CONTRIB])
            assert x.rate_hat == pytest.approx(tot, abs=1e-3)


def test_m2_is_the_log_average_of_m1_and_ss(run):
    r = _rows(run["full"])
    ss = r.loc[("own", mf.SS_NAME), "forecast"]
    for pool in ("pooled", "own"):
        assert r.loc[(pool, mf.M2), "forecast"] == pytest.approx(math.sqrt(r.loc[(pool, mf.M1), "forecast"] * ss))


def test_pre_t0_target_returns_the_label_path(claims, sources):
    tg = _target("anthropic", "rev_2025q2")
    v = backtest.target_vintage(tg, "period_end")
    v = max(v, pd.Timestamp("2026-04-01"))                         # a vintage with several anchors after Q2 2025
    rows = _rows(mf.forecast_target(tg, v, claims, sources, params=FIXED, ablations=False))
    assert not bool(rows.loc[("pooled", mf.M1), "post_t0"])
    assert rows.loc[("pooled", mf.M1), "forecast"] == pytest.approx(rows.loc[("own", mf.M1), "forecast"], rel=1e-12)


# --------------------------------------------------------------------------- synthetic pairs
def _pairs(n_weeks=120, beta=0.4, noise=0.002, seed=0, sparse_from=None, companies=("anthropic", "openai")):
    """Pairs-like table: y = company effect + beta * x(or_spend) + noise; every other feature is noise."""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = pd.Timestamp("2025-01-02")
    for k, c in enumerate(companies):
        fe = 0.03 if k == 0 else 0.01
        for i in range(n_weeks):
            for h in (4, 8, 13):
                x = {f: rng.normal(0, 0.01) for f in mf.FEATURES}
                x["or_spend"] = rng.normal(0, 0.05)
                if sparse_from is not None and i < sparse_from:
                    x["vercel_spend_share"] = np.nan
                rows.append({"company": c, "t": t0 + i * pm.WEEK, "t1": t0 + (i + h) * pm.WEEK, "h": h,
                             "y": fe + beta * x["or_spend"] + rng.normal(0, noise), "mom": rng.normal(fe, 0.01),
                             "w_raw": 1.0 + rng.uniform(0, 1), **x})
    return pd.DataFrame(rows)


def test_prior_recovered_when_data_uninformative():
    """Few, very noisy rows: the strongest lambda returns the prior means (1 on the proxies, 0 elsewhere),
    and a proxy without variation in the training rows keeps its prior mean at any lambda."""
    P = _pairs(n_weeks=30, beta=0.0, noise=0.5, seed=1, companies=("anthropic",))
    el = mf.PriorRidge(mf.LAMBDAS[0]).fit(P).elasticities()
    for f in mf.PRIOR_ELASTICITY:
        assert el[f] == pytest.approx(1.0, abs=0.02)
    assert abs(el["mom"]) < 0.02 and abs(el["npm_level"]) < 0.05
    Q = P.assign(log_gh_share=0.01)                                   # constant: no information at all
    for lam in (mf.LAMBDAS[0], 1.0, 0.1):
        m = mf.PriorRidge(lam).fit(Q)
        assert m.coef_[m.prep_.cols.index("log_gh_share")] == pytest.approx(m.gamma0_[m.prep_.cols.index("log_gh_share")])


def test_truth_recovered_when_data_strong():
    """Many clean rows with elasticity 0.4 (prior 1): the time-ordered CV picks a weak prior and the fit
    lands on the truth, not on the prior; the prior pulls toward 1 as lambda grows."""
    P = _pairs(n_weeks=120, beta=0.4, noise=0.002, seed=2)
    folds = mf.inner_folds(P)
    assert len(folds) >= 3
    lam = mf.choose_lambda(mf.cv_curve(folds))
    assert lam <= 100.0
    el = mf.PriorRidge(lam).fit(P).elasticities()
    assert el["or_spend"] == pytest.approx(0.4, abs=0.02)
    path = [mf.PriorRidge(l).fit(P).elasticities()["or_spend"] for l in sorted(mf.LAMBDAS)]
    assert all(b >= a - 1e-3 for a, b in zip(path, path[1:])) and path[0] < 0.45 and path[-1] > 0.9


def test_closed_form_matches_posterior_mean():
    rng = np.random.default_rng(3)
    Z, w = rng.normal(size=(50, 3)), rng.uniform(0.5, 1.5, 50)
    y = Z @ np.array([0.2, -0.1, 0.0]) + rng.normal(0, 0.1, 50)
    g0, lam = np.array([1.0, 0.0, 0.5]), 7.0
    A0, b0 = mf._normal_eq(Z, y, w)
    g = mf.solve_prior_ridge(A0, b0, lam, g0)
    from scipy.optimize import minimize
    obj = lambda c: np.sum(w * (y - Z @ c) ** 2) + lam * np.sum((c - g0) ** 2)   # noqa: E731
    np.testing.assert_allclose(g, minimize(obj, np.zeros(3), method="BFGS", options={"gtol": 1e-10}).x, atol=1e-6)


def test_choose_lambda_ties_and_default():
    assert mf.choose_lambda(None) == mf.DEFAULT_LAMBDA
    c = np.array([0.2, 0.1, 0.1, 0.3, 0.1, 0.4, 0.5, 0.6])
    assert mf.choose_lambda(c) == mf.LAMBDAS[1]                     # tie -> the stronger prior


def test_standardization_and_clipping_on_observed_rows():
    P = pm._normalise(_pairs(n_weeks=80, sparse_from=50, seed=4))     # vercel observed in 30 of 80 weeks
    pr = mf.Prep("fixed").fit(P)
    j = pr.cols.index("vercel_spend_share")
    x = P["vercel_spend_share"].to_numpy(float)
    obs = ~np.isnan(x)
    assert pr.lo[j] == pytest.approx(np.percentile(x[obs], 1)) and pr.hi[j] == pytest.approx(np.percentile(x[obs], 99))
    xc = np.clip(x, pr.lo[j], pr.hi[j])
    mu = np.array([pr.means_[c][1 + j] for c in P.company])
    for c in ("anthropic", "openai"):                                 # company means over observed rows only
        m = obs & (P.company == c).to_numpy()
        assert pr.means_[c][1 + j] == pytest.approx(np.sum(P.w.to_numpy()[m] * xc[m]) / np.sum(P.w.to_numpy()[m]))
    w = P.w.to_numpy(float)
    assert pr.scale_[j] == pytest.approx(math.sqrt(np.sum(w[obs] * (xc - mu)[obs] ** 2) / np.sum(w[obs])))
    old = mf.Prep("old").fit(P)
    assert old.scale_[old.cols.index("vercel_spend_share")] < pr.scale_[j]      # imputed zeros shrink it
    G = P.iloc[:3].copy()
    G["vercel_spend_share"] = [1e3, -1e3, np.nan]
    Z = pr.Z(G)[:, j]
    assert Z[0] == pytest.approx((pr.hi[j] - pr.means_[G.company.iloc[0]][1 + j]) / pr.scale_[j])
    assert Z[1] == pytest.approx((pr.lo[j] - pr.means_[G.company.iloc[1]][1 + j]) / pr.scale_[j])
    assert Z[2] == 0.0


def test_short_features_dropped():
    P = pm._normalise(_pairs(n_weeks=60, sparse_from=35, seed=5))      # vercel observed in 25 weeks
    pr = mf.Prep("fixed").fit(P)
    assert "vercel_spend_share" in pr.dropped and "vercel_spend_share" not in pr.cols
    assert pr.cols[0] == "mom" and "or_spend" in pr.cols
    P2 = pm._normalise(_pairs(n_weeks=60, sparse_from=34, seed=5))     # 26 weeks: kept
    assert "vercel_spend_share" in mf.Prep("fixed").fit(P2).cols
    assert "vercel_spend_share" in mf.Prep("old").fit(P).cols


def test_openai_target_when_anthropic_has_no_labels_yet(claims, sources):
    """Regression (crash in the first backtest launch, before any output): an early OpenAI target whose
    pooled pool has no Anthropic pairs (Anthropic has < 2 anchors at the vintage)."""
    tg = _target("openai", "oa_rr_2025_07_ti")
    v = backtest.target_vintage(tg, "period_end")
    assert mf.labels_at("anthropic", claims["anthropic"], v, params=NOISE).t0 is None
    rows = _rows(mf.forecast_target(tg, v, claims, sources, params=FIXED, ablations=False))
    assert np.isfinite(rows.loc[("pooled", mf.M1), "forecast"]) and np.isfinite(rows.loc[("pooled", mf.M2), "forecast"])
    assert rows.loc[("pooled", mf.M1), "n_train_pairs"] == rows.loc[("pooled", mf.M1), "n_target_pairs"]
