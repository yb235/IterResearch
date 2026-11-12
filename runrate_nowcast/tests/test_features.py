"""No look-ahead in the as-of feature store (features.py).

The publication rules are restated here, independently of features.py, and used to cut the
source tables at a vintage. Features from the full tables must equal features from the cut
tables, and must not move when anything published after the vintage changes.

Run: python -m pytest runrate_nowcast/tests -q
"""
import builtins
import math
import os

import numpy as np
import pandas as pd
import pytest

from runrate_nowcast import config, data, features as F, pricing

DAY = pd.Timedelta(days=1)
VINTAGES = ["2025-06-15", "2026-02-05", "2026-08-31", "2026-10-15", "2026-10-20", "2026-11-12"]
ENTITIES = ("anthropic", "openai")


# --------------------------------------------------------------------------- publication rules
def _published(name: str, df: pd.DataFrame) -> pd.Series:
    """When each source row became public (NaT: never), per the project's stated rules."""
    if name == "indicators":       # stored date; daily npm/PyPI counts (no stored date) the day after
        return pd.to_datetime(df["published"]).fillna(pd.to_datetime(df["date"]) + DAY)
    if name == "github_new_repos":     # the day after the month ends
        return pd.to_datetime(df["month"]) + pd.offsets.MonthBegin(1)
    if name == "gateway":
        return pd.to_datetime(df["captured_or_published"], format="%Y-%m-%d", errors="coerce")
    if name == "cloud":
        return pd.to_datetime(df["published"])
    raise KeyError(name)


def _price_published(p: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(p["release_date"])


def _mix_published(m: pd.DataFrame) -> pd.Series:       # a measured week is public once it is over
    return pd.to_datetime(m["week_start"]) + pd.Timedelta(days=7)


VALUE_COLS = {"indicators": ["value"], "github_new_repos": ["total_count"],
              "gateway": ["anthropic_value", "openai_value", "google_value", "deepseek_value"], "cloud": ["value"]}


@pytest.fixture(scope="module")
def src():
    return F.load_sources()


@pytest.fixture(scope="module")
def full(src):
    """Outputs from the stored data at a vintage (cached: several tests reuse them)."""
    cache = {}

    def get(vintage):
        v = pd.Timestamp(vintage)
        if v not in cache:
            cache[v] = _outputs(v, src)
        return cache[v]
    return get


@pytest.fixture(scope="module")
def prices():
    return pricing.load_prices()


@pytest.fixture(scope="module")
def mix():
    return pd.read_csv(pricing.FAMILY_MIX_PATH)


def _use_price_inputs(monkeypatch, tmp_path, prices, mix):
    """Point pricing.price_index at the given price list and commit-mix table."""
    path = os.path.join(tmp_path, "mix.csv")
    mix.to_csv(path, index=False)
    monkeypatch.setattr(pricing, "load_prices", lambda path=None: prices.copy())
    monkeypatch.setattr(pricing, "FAMILY_MIX_PATH", path)


def _outputs(vintage, sources):
    out = {e: F.asof_features(vintage, e, sources=sources) for e in ENTITIES}
    out["availability"] = F.feature_availability(vintage, ENTITIES, sources=sources)
    return out


def _assert_same(a, b):
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k], check_exact=True)


# --------------------------------------------------------------------------- 1. truncation invariance
@pytest.mark.parametrize("vintage", VINTAGES)
def test_truncation_invariance(vintage, src, full, prices, mix, monkeypatch, tmp_path):
    v = pd.Timestamp(vintage)
    cut = {k: d[(_published(k, d) <= v).to_numpy()].reset_index(drop=True) for k, d in src.items()}
    assert sum(len(d) for d in cut.values()) < sum(len(d) for d in src.values())
    with monkeypatch.context() as mp:
        _use_price_inputs(mp, tmp_path, prices[_price_published(prices) <= v], mix[_mix_published(mix) <= v])
        trunc = _outputs(v, cut)
    _assert_same(full(v), trunc)


# --------------------------------------------------------------------------- 2. perturbation
def _perturb(df, after, cols, rng):
    d = df.copy()
    for c in cols:
        if c in d:
            x = d.loc[after, c].astype(float)
            d[c] = d[c].astype(float)
            d.loc[after, c] = x.fillna(1.0) * rng.uniform(0.2, 5.0, len(x)) + rng.uniform(1, 50, len(x))
    return d


@pytest.mark.parametrize("vintage", ["2026-02-05", "2026-10-15", "2026-11-12"])
def test_values_published_after_vintage_do_not_matter(vintage, src, full, prices, mix, monkeypatch, tmp_path):
    v = pd.Timestamp(vintage)
    rng = np.random.default_rng(int(v.strftime("%Y%m%d")))
    later = {k: (~(_published(k, d) <= v)).to_numpy() for k, d in src.items()}     # includes rows never published
    bent = {k: _perturb(d, later[k], VALUE_COLS[k], rng) for k, d in src.items()}
    p2 = _perturb(prices, ~(_price_published(prices) <= v).to_numpy(), ["input_usd_mtok", "output_usd_mtok"], rng)
    m2 = _perturb(mix, ~(_mix_published(mix) <= v).to_numpy(), ["total_count"], rng)
    with monkeypatch.context() as mp:
        _use_price_inputs(mp, tmp_path, p2, m2)
        moved = _outputs(v, bent)
    _assert_same(full(v), moved)


def test_perturbation_has_power(src, full):
    """Control: changing a value published on/before the vintage does change the output."""
    v = pd.Timestamp("2026-10-20")
    ind = src["indicators"]
    m = ((ind["indicator"] == "ramp_adoption_anthropic") & (ind["date"] == "2026-09-30")).to_numpy()
    bent = dict(src, indicators=ind.assign(value=np.where(m, 0.9, ind["value"])))
    assert F.asof_features(v, "openai", sources=bent).equals(full(v)["openai"])
    assert F.asof_features(v, sources=bent)["ramp_adoption"].iloc[-1] == 0.9
    assert full(v)["anthropic"]["ramp_adoption"].iloc[-1] != 0.9


# --------------------------------------------------------------------------- 3. publication dates
def _last_obs(vintage, src, feature, entity="anthropic"):
    a = F.feature_availability(vintage, (entity,), sources=src)
    return a.loc[a["feature"] == feature, "last_obs"].iloc[0]


def test_github_month_appears_the_day_after_it_ends(src, full):
    g = src["github_new_repos"].pivot_table(index="month", columns="query", values="total_count")
    sep = g.loc["2026-09-01", "claude"] / g.loc["2026-09-01", ["claude", "openai", "gemini"]].sum()
    before = full("2026-09-30")["anthropic"]["github_repo_share"]
    after = full("2026-10-01")["anthropic"]["github_repo_share"]
    assert not (before == sep).any()
    assert _last_obs("2026-09-30", src, "github_repo_share") == pd.Timestamp("2026-08-01")
    assert after.iloc[-1] == sep and after.index[-1] == pd.Timestamp("2026-10-01")
    assert _last_obs("2026-10-01", src, "github_repo_share") == pd.Timestamp("2026-09-01")
    # carried forward only to weeks ending on/after publication
    assert (after.index[after == sep] + pd.Timedelta(days=3) >= pd.Timestamp("2026-10-01")).all()


def test_ramp_month_appears_on_its_publication_date(src, full):
    ind = src["indicators"]
    sep = ind[(ind["indicator"] == "ramp_adoption_anthropic") & (ind["date"] == "2026-09-30")]["value"].iloc[0]
    for v in ("2026-09-30", "2026-10-15", "2026-10-19"):
        assert not (full(v)["anthropic"]["ramp_adoption"] == sep).any(), v
        assert _last_obs(v, src, "ramp_adoption") == pd.Timestamp("2026-08-31")
    f = full("2026-10-20")["anthropic"]
    assert f["ramp_adoption"].iloc[-1] == sep
    assert f["ramp_adoption_chg"].iloc[-1] == pytest.approx(sep - 0.4353)      # August as known then
    assert (f.index[f["ramp_adoption"] == sep] + pd.Timedelta(days=3) >= pd.Timestamp("2026-10-20")).all()


def test_openrouter_week_appears_the_day_after_it_ends(src, full):
    wk = pd.Timestamp("2026-11-05")           # week ending Sunday 2026-11-08, published Monday 11-09
    f0 = full("2026-11-08")["anthropic"]
    f1 = full("2026-11-09")["anthropic"]
    for c in ("openrouter_log_tokens", "openrouter_dlog_tokens", "openrouter_share", "npm_dlog"):
        assert np.isnan(f0.loc[wk, c]) and np.isfinite(f1.loc[wk, c]), c
    ind = src["indicators"]
    tok = ind[(ind["indicator"] == "openrouter_anthropic_tokens") & (ind["date"] == "2026-11-08")]["value"].iloc[0]
    assert f1.loc[wk, "openrouter_log_tokens"] == pytest.approx(math.log(tok))


def test_cloud_statement_carried_after_publication_only(full):
    f0 = full("2026-10-27")["anthropic"]["cloud_foundry_growth"]
    f1 = full("2026-10-28")["anthropic"]["cloud_foundry_growth"]
    assert not np.isclose(f0, math.log(3)).any()
    assert f1.iloc[-1] == pytest.approx(math.log(3))                      # 'more than tripled'
    assert full("2026-11-12")["openai"]["cloud_foundry_growth"].isna().all()


def test_gateway_revision_used_only_from_its_own_date(src):
    base = {"platform": "Vercel AI Gateway", "scope": "all traffic", "metric": "spend_share",
            "window_start": "2026-08-01", "window_end": "2026-08-31", "openai_value": np.nan,
            "google_value": np.nan, "deepseek_value": np.nan}
    gw = pd.DataFrame([
        {**base, "captured_or_published": "2026-09-17", "anthropic_value": 64.0},     # first release
        {**base, "captured_or_published": "2026-10-15", "anthropic_value": 50.0},     # revision
        {**base, "captured_or_published": np.nan, "anthropic_value": 99.0},           # no date: never used
        {**base, "captured_or_published": "2026-09", "anthropic_value": 98.0},        # month only: never used
        {**base, "scope": "all traffic (model: X)", "captured_or_published": "2026-09-01", "anthropic_value": 97.0},
    ])
    s = dict(src, gateway=gw)
    before = F.asof_features("2026-10-14", sources=s)["vercel_spend_share"]
    after = F.asof_features("2026-10-15", sources=s)["vercel_spend_share"]
    assert before.iloc[-1] == 0.64 and after.iloc[-1] == 0.50
    assert after[after.index < pd.Timestamp("2026-10-15")].dropna().eq(0.64).all()    # history not rewritten
    assert after.dropna().index[0] == pd.Timestamp("2026-09-17")
    assert F.asof_features("2026-09-16", sources=s)["vercel_spend_share"].isna().all()


def test_week_index(src, full):
    for v, last in (("2026-11-12", "2026-11-12"), ("2026-11-09", "2026-11-12"), ("2026-11-08", "2026-11-05")):
        f = full(v)["anthropic"]
        assert f.index[-1] == pd.Timestamp(last)
        assert (f.index.dayofweek == 3).all() and (np.diff(f.index.values) == np.timedelta64(7, "D")).all()
        assert list(f.columns) == F.FEATURE_COLUMNS
    p = F.asof_panel("2026-11-12", sources=src)
    assert list(p["entity"].unique()) == list(ENTITIES)
    assert p.loc[p["entity"] == "openai", "log_price"].isna().all()
    with pytest.raises(ValueError):
        F.asof_features("2026-11-12", "nobody", sources=src)


# --------------------------------------------------------------------------- 4. filings never read
def test_filing_rows_never_read(monkeypatch):
    opened = []
    real_read_csv, real_open = pd.read_csv, builtins.open

    def spy_read_csv(path, *a, **k):
        opened.append(os.path.abspath(str(path)))
        return real_read_csv(path, *a, **k)

    def spy_open(path, *a, **k):
        opened.append(os.path.abspath(str(path)))
        return real_open(path, *a, **k)

    def boom(*a, **k):
        raise AssertionError("observations.csv (filing rows) must not be loaded")

    monkeypatch.setattr(pd, "read_csv", spy_read_csv)
    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(data, "load", boom)
    monkeypatch.setattr(data, "select", boom)
    F.asof_panel("2026-11-12")
    F.feature_availability("2026-11-12")
    monkeypatch.undo()
    allowed = {os.path.abspath(p) for p in list(F.SOURCE_FILES.values()) + [pricing.PRICES_PATH, pricing.FAMILY_MIX_PATH]}
    assert opened and set(opened) <= allowed, set(opened) - allowed
    assert os.path.abspath(config.DATA_PATH) not in opened
