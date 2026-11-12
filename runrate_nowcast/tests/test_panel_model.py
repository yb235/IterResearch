"""Pooled two-company panel model (panel_model.py): no look-ahead, training-only preprocessing,
filings never used, and recovery of a known elasticity on synthetic data.

Run: python -m pytest runrate_nowcast/tests -q
"""
import math
import os

import numpy as np
import pandas as pd
import pytest

from runrate_nowcast import backtest, config, data
from runrate_nowcast import features as F
from runrate_nowcast import panel_model as pm
from runrate_nowcast.model import L, B

DAY = pd.Timedelta(days=1)
# Fixed process noise so the label tests do not refit (the refit itself is tested separately).
NOISE = {"sigma_l": 0.004, "sigma_g": 0.01, "sigma_b": 0.002, "growth_reversion": 0.1}
FIXED = {c: NOISE for c in pm.COMPANIES}


@pytest.fixture(scope="module")
def claims():
    return {c: pm.load_claims(c) for c in pm.COMPANIES}


@pytest.fixture(scope="module")
def sources():
    return F.load_sources()


def _published_sources(src: dict, v: pd.Timestamp) -> dict:
    """Source tables with every value published after v replaced by nonsense (publication
    rules restated independently of features.py, as in test_features.py)."""
    out = {k: d.copy() for k, d in src.items()}
    ind = out["indicators"]
    pub = pd.to_datetime(ind["published"]).fillna(pd.to_datetime(ind["date"]) + DAY)
    ind.loc[pub > v, "value"] = pd.to_numeric(ind.loc[pub > v, "value"], errors="coerce") * 7.0
    g = out["github_new_repos"]
    late = pd.to_datetime(g["month"]) + pd.offsets.MonthBegin(1) > v
    g.loc[late, "total_count"] = g.loc[late, "total_count"] * 7
    gw = out["gateway"]
    late = ~(pd.to_datetime(gw["captured_or_published"], format="%Y-%m-%d", errors="coerce") <= v)
    for c in ("anthropic_value", "openai_value", "google_value"):
        gw.loc[late, c] = pd.to_numeric(gw.loc[late, c], errors="coerce") * 0.3
    cl = out["cloud"]
    late = pd.to_datetime(cl["published"]) > v
    cl.loc[late, "value"] = pd.to_numeric(cl.loc[late, "value"], errors="coerce") * 3
    return out


def _cut_claims(claims: dict, v: pd.Timestamp) -> dict:
    return {c: d[d.published <= v].copy() for c, d in claims.items()}


def _perturb_late_claims(claims: dict, v: pd.Timestamp) -> dict:
    out = {}
    for c, d in claims.items():
        d = d.copy()
        d.loc[d.published > v, "value"] = d.loc[d.published > v, "value"] * 5
        out[c] = d
    return out


def _target(company, obs_id):
    t = pm.load_targets(company).set_index("obs_id").loc[obs_id]
    return pm._Row(obs_id=obs_id, **t.to_dict())


# --------------------------------------------------------------------------- no look-ahead
@pytest.mark.parametrize("company,vintage", [("anthropic", "2026-04-05"), ("openai", "2026-02-19"),
                                             ("anthropic", "2026-09-30")])
def test_labels_unchanged_without_later_claims(claims, company, vintage):
    v = pd.Timestamp(vintage)
    full = pm.company_labels(company, claims[company], v, params=NOISE)
    cut = pm.company_labels(company, _cut_claims(claims, v)[company], v, params=NOISE)
    pert = pm.company_labels(company, _perturb_late_claims(claims, v)[company], v, params=NOISE)
    for other in (cut, pert):
        assert full.t0 == other.t0
        pd.testing.assert_series_equal(full.y, other.y)
        pd.testing.assert_series_equal(full.sd, other.sd)
    assert full.t0 <= v and full.y.index.max() <= full.t0
    assert (full.sel.published <= v).all()


def test_noise_refit_sees_only_public_claims(claims, monkeypatch):
    seen = []

    def fake_fit(sel, start):
        seen.append(sel.copy())
        return dict(NOISE)
    monkeypatch.setattr(pm, "fit_noise", fake_fit)
    monkeypatch.setattr(pm, "_NOISE_CACHE", {})
    v = pd.Timestamp("2026-03-02")
    pm.company_labels("anthropic", claims["anthropic"], v, exclude_origin="anth_2025ye_9b")
    assert len(seen) == 1
    s = seen[0]
    assert (s.published <= v).all() and (s.filing == 0).all()
    assert "anth_2025ye_9b" not in set(s.origin_id)


@pytest.mark.parametrize("company,vintage", [("anthropic", "2025-12-31"), ("openai", "2026-06-30"),
                                             ("anthropic", "2026-11-09")])
def test_features_unchanged_without_later_data(sources, company, vintage):
    v = pd.Timestamp(vintage)
    a = pm.feature_levels(company, v, sources)
    b = pm.feature_levels(company, v, _published_sources(sources, v))
    pd.testing.assert_frame_equal(a, b)
    assert a.index.max() == F._week_label([v])[0]


def test_forecast_unchanged_without_later_data(claims, sources):
    """End to end: one target, both pools, every model."""
    tg = _target("anthropic", "rr_2026_04")
    v = backtest.target_vintage(tg, "publication")
    full = pm.forecast_target(tg, v, claims, sources, params=FIXED)
    late = pm.forecast_target(tg, v, _perturb_late_claims(claims, v), _published_sources(sources, v), params=FIXED)
    a = pd.DataFrame(full).set_index("model")
    b = pd.DataFrame(late).set_index("model")
    assert len(a) == 2 + len(pm.MODELS) * len(pm.POOLS)
    np.testing.assert_allclose(a.forecast.to_numpy(), b.loc[a.index].forecast.to_numpy(), rtol=1e-12)
    assert (a.n_train_pairs.fillna(0) >= 0).all()


def test_target_origin_never_used(claims):
    v = pd.Timestamp("2026-03-03")              # anth_2025ye_9b is public on this day
    sel = pm.public_claims(claims["anthropic"], v, exclude_origin="anth_2025ye_9b")
    assert "anth_2025ye_9b" not in set(sel.origin_id)
    with pytest.raises(AssertionError):
        pm._assert_public(pm.public_claims(claims["anthropic"], v), v, "anth_2025ye_9b")


# --------------------------------------------------------------------------- filings
def test_filings_never_read(tmp_path, claims, sources):
    raw = pd.read_csv(config.DATA_PATH)
    assert (raw.filing != 0).any()
    loaded = pm.load_claims("anthropic")
    assert (loaded.filing == 0).all()
    assert not set(raw.loc[raw.filing != 0, "obs_id"]) & set(loaded.obs_id)
    # filing rows changed, and a fake filing row published early: nothing moves
    alt = raw.copy()
    alt.loc[alt.filing != 0, "value"] = alt.loc[alt.filing != 0, "value"] * 10
    fake = alt[alt.obs_id == "rr_2026_04"].copy()
    for col, val in (("obs_id", "fake_filing"), ("origin_id", "fake_origin"), ("value", 500.0),
                     ("filing", 1), ("published", "2025-06-01")):
        fake[col] = val
    alt = pd.concat([alt, fake], ignore_index=True)
    path = os.path.join(tmp_path, "obs.csv")
    alt.to_csv(path, index=False)
    alt_claims = {"anthropic": pm.load_claims("anthropic", path), "openai": claims["openai"]}
    tg = _target("anthropic", "rev_2026q3")
    v = backtest.target_vintage(tg, "period_end")
    a = pd.DataFrame(pm.forecast_target(tg, v, claims, sources, models=("ridge",), params=FIXED))
    b = pd.DataFrame(pm.forecast_target(tg, v, alt_claims, sources, models=("ridge",), params=FIXED))
    np.testing.assert_allclose(a.forecast.to_numpy(), b.forecast.to_numpy(), rtol=1e-12)
    # and the guard rejects filing rows if they ever reach it
    sel = data.select(data.load(), vintage=v, filings=True)
    with pytest.raises(AssertionError, match="filing"):
        pm._assert_public(sel[sel.published <= v], v, None)


# --------------------------------------------------------------------------- training-only preprocessing
def _synthetic_pairs(n=160, seed=0, beta=0.6, fe=(0.03, 0.01), rho=0.0, noise=0.002):
    rng = np.random.default_rng(seed)
    rows = []
    for ci, c in enumerate(("anthropic", "openai")):
        t = pd.date_range("2025-01-02", periods=n // 2, freq="7D")
        for i, tt in enumerate(t):
            h = (4, 8, 13)[i % 3]
            x = rng.normal(0, 0.05, len(pm.FEATURES))
            x[pm.FEATURES.index("npm_level")] = rng.normal(0, 0.002)
            mom = fe[ci] + rng.normal(0, 0.01)
            y = fe[ci] + beta * x[0] + rho * (mom - fe[ci]) + rng.normal(0, noise)
            rows.append([c, tt, tt + h * pm.WEEK, h, y, mom, 1.0 + rng.uniform()] + list(x))
    return pd.DataFrame(rows, columns=["company", "t", "t1", "h", "y", "mom", "w_raw"] + pm.FEATURES)


@pytest.mark.parametrize("kind", ["pca", "ridge", "trend"])
def test_scaler_and_pca_fit_on_training_rows_only(kind):
    P = pm._normalise(_synthetic_pairs())
    train, test = P.iloc[:100].copy(), P.iloc[100:].copy()
    m = pm.LinearPanel(kind, alpha=1.0, k=2).fit(train)
    state = (m.scale_.copy(), {k: v.copy() for k, v in m.means_.items()}, m.ridge_.coef_.copy(),
             None if m.components_ is None else m.components_.copy())
    wild = test.copy()
    wild[pm.FEATURES] = wild[pm.FEATURES] * 1000 + 50
    m.predict(wild)
    np.testing.assert_array_equal(m.scale_, state[0])
    np.testing.assert_array_equal(m.ridge_.coef_, state[2])
    if state[3] is not None:
        np.testing.assert_array_equal(m.components_, state[3])
    # the fitted statistics are those of the training rows: refitting on train + wild test rows changes them
    m2 = pm.LinearPanel(kind, alpha=1.0, k=2).fit(pm._normalise(pd.concat([train, wild])))
    assert not np.allclose(m2.scale_, state[0])
    # company means equal the weighted means of the training rows
    w = train.w.to_numpy()
    a = train.company.to_numpy() == "anthropic"
    np.testing.assert_allclose(m.means_["anthropic"][0], np.sum(w[a] * train.y[a]) / np.sum(w[a]))


def test_cv_folds_are_time_ordered_and_purged():
    P = _synthetic_pairs()
    folds = pm.cv_folds(P)
    assert len(folds) >= 3
    for tr, va in folds:
        assert P.loc[tr, "t1"].max() < P.loc[va, "t"].min()


# --------------------------------------------------------------------------- synthetic elasticity
class _FakeRes:
    def __init__(self, path: pd.Series):
        self.path = path

    def state(self, d):
        x = (self.path.index - self.path.index[0]).days.to_numpy(float)
        v = float(np.interp((pd.Timestamp(d) - self.path.index[0]).days, x, self.path.to_numpy()))
        m = np.zeros(3)
        m[L] = v
        return m, np.zeros((3, 3))


def _synthetic_company(company, fe, beta, seed, weeks=90, t0_offset=0, factor=False):
    """A company whose weekly log growth is fe + beta * (weekly change of OpenRouter log tokens).

    factor=True: OpenRouter share and npm downloads also load on the token factor, so the
    first principal component of the standardised changes is that factor."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-02", periods=weeks, freq="7D")
    dx = rng.normal(0, 0.06, weeks)
    tok = np.cumsum(dx)
    other = {f: np.cumsum(rng.normal(0, 0.01, weeks)) for f in pm.FEATURES}
    other["openrouter_log_tokens"] = tok
    if factor:
        other["openrouter_share"] = 0.5 * tok + np.cumsum(rng.normal(0, 0.003, weeks))
        other["npm_level"] = 0.3 * tok + np.cumsum(rng.normal(0, 0.003, weeks))
    flev = pd.DataFrame(other, index=idx)[pm.FEATURES]
    g = fe + beta * np.r_[0.0, np.diff(tok)] + rng.normal(0, 0.001, weeks)
    y = pd.Series(np.cumsum(g), index=idx)
    t0 = idx[-1 - t0_offset]
    lab = pm.Labels(company, idx[-1], None, y[y.index <= t0], y[y.index <= t0],
                    pd.Series(0.05, index=y.index[y.index <= t0]), t0, idx[0], res=_FakeRes(y))
    return lab, flev, y


def test_recovers_known_elasticity():
    beta = 0.6
    la, fa, ya = _synthetic_company("anthropic", 0.03, beta, seed=1, t0_offset=13)
    lo, fo, _ = _synthetic_company("openai", 0.01, beta, seed=2)
    P = pd.concat([pm.make_pairs(la, fa), pm.make_pairs(lo, fo)], ignore_index=True)
    assert set(P.h) == {4, 8, 13}
    m = pm.fit_model("ridge", P)
    el = m.elasticities()
    assert abs(el["openrouter_log_tokens"] - beta) < 0.05
    assert max(abs(el[f]) for f in pm.FEATURES if f != "openrouter_log_tokens") < 0.1
    # forecast 8 weeks past the last label, from the label at t0 plus predicted growth
    d = la.t0 + 8 * pm.WEEK
    tg = pm._Row(metric="run_rate", t_eff=d)
    fc = pm.path_forecast(la, fa, tg, m.predict)
    truth = math.exp(ya[d])
    trend = pm.path_forecast(la, fa, tg, pm.fit_model("trend", P).predict)
    assert abs(math.log(fc / truth)) < 0.02
    assert abs(math.log(fc / truth)) < abs(math.log(trend / truth))


def test_factor_model_recovers_common_factor():
    beta = 0.6
    la, fa, ya = _synthetic_company("anthropic", 0.03, beta, seed=3, t0_offset=13, factor=True)
    lo, fo, _ = _synthetic_company("openai", 0.01, beta, seed=4, factor=True)
    P = pd.concat([pm.make_pairs(la, fa), pm.make_pairs(lo, fo)], ignore_index=True)
    mp = pm.fit_model("pca", P)
    d = la.t0 + 8 * pm.WEEK
    tg = pm._Row(metric="run_rate", t_eff=d)
    truth = math.exp(ya[d])
    err = abs(math.log(pm.path_forecast(la, fa, tg, mp.predict) / truth))
    err_trend = abs(math.log(pm.path_forecast(la, fa, tg, pm.fit_model("trend", P).predict) / truth))
    assert err < 0.06 and err < 0.5 * err_trend
    # implied effect of a one-unit move along the factor (tokens 1, share 0.5, npm 0.3) is about beta
    el = mp.elasticities()
    implied = el["openrouter_log_tokens"] + 0.5 * el["openrouter_share"] + 0.3 * el["npm_level"]
    assert abs(implied - beta) < 0.1
