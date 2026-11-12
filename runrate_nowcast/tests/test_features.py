"""No look-ahead in the as-of feature store (features.py).

The publication rules are restated here, independently of features.py, and used to cut the
source tables at a vintage. Features from the full tables must equal features from the cut
tables, and must not move when anything published after the vintage changes.

The extra sources (FEATURE_SETS["all"]) get the same treatment: npm/PyPI framework and agent
packages, the OpenRouter competitor series, the GitHub model-family mix, the Ramp AI Index
history and, once its file exists, consumer_signals.csv (until then a synthetic table stands in,
so the as-of machinery for it is tested now). The original columns (FEATURE_SETS["g_spec"]) are
also compared with the committed feature store (REF_COMMIT) and must not move.

Run: python -m pytest runrate_nowcast/tests -q
"""
import builtins
import importlib.util
import math
import os
import subprocess

import numpy as np
import pandas as pd
import pytest

from runrate_nowcast import config, data, extra_sources, features as F, pricing

DAY = pd.Timedelta(days=1)
VINTAGES = ["2025-06-15", "2026-02-05", "2026-08-31", "2026-10-15", "2026-10-20", "2026-11-12"]
ENTITIES = ("anthropic", "openai")
REF_COMMIT = "9e8d419"          # the as-of feature store as reviewed (sub-task E); g_spec must match it
NEW_COLUMNS = F.FEATURE_SETS["all"][len(F.FEATURE_SETS["g_spec"]):]
CONSUMER_COLUMNS = [c for c in NEW_COLUMNS if c.startswith("consumer_")]

# Stand-in for data/consumer_signals.csv while it does not exist (same columns). Includes rows
# that must never be used: no publication date, a month-only date, a one-day window, a
# different metric (unique visitors).
_C = ["series", "entity", "metric", "period_start", "period_end", "value", "unit", "source", "published"]
SYNTH_CONSUMER = pd.DataFrame([
    ["web_claude", "Claude", "web_visits", "2026-07-01", "2026-07-31", 950, "millions of visits", "s", "2026-08-20"],
    ["web_chatgpt", "ChatGPT", "web_visits", "2026-07-01", "2026-07-31", 5800, "millions of visits", "s", "2026-08-20"],
    ["web_gemini", "Gemini", "web_visits", "2026-07-01", "2026-07-31", 1100, "millions of visits", "s", "2026-08-25"],
    ["web_claude", "Claude", "web_visits", "2026-08-01", "2026-08-31", 960, "millions of visits", "s", "2026-09-18"],
    ["web_chatgpt", "ChatGPT", "web_visits", "2026-08-01", "2026-08-31", 5700, "millions of visits", "s", "2026-09-18"],
    ["web_gemini", "Gemini", "web_visits", "2026-08", "2026-08", 1.3, "billion visits", "s", "2026-09-22"],
    ["web_claude", "anthropic", "web_visits", "2026-08-01", "2026-08-31", 980, "millions of visits", "s2", "2026-10-05"],
    ["web_claude", "Claude", "web_visits", "2026-09-01", "2026-09-30", 1000, "millions of visits", "s", np.nan],
    ["web_chatgpt", "ChatGPT", "web_visits", "2026-09-01", "2026-09-30", 6000, "millions of visits", "s", "2026-10"],
    ["web_claude", "Claude", "web_visits", "2026-10-22", "2026-10-22", 47, "millions of visits", "s", "2026-10-25"],
    ["web_claude", "Claude", "unique_visitors", "2026-08-01", "2026-08-31", 300, "millions", "s", "2026-09-10"],
    ["app_claude", "Claude", "app_downloads", "2026-04-01", "2026-04-30", 20, "million downloads", "s", "2026-05-15"],
    ["app_chatgpt", "ChatGPT", "app_downloads", "2026-04-01", "2026-04-30", 60, "million downloads", "s", "2026-05-15"],
    ["app_gemini", "Gemini", "app_downloads", "2026-04-01", "2026-04-30", 30, "million downloads", "s", "2026-05-20"],
    ["app_claude", "Claude", "app_downloads", "2026-05-01", "2026-05-31", 18, "million downloads", "s", "2026-06-10"],
    ["app_claude", "Claude", "app_revenue", "2026-05-01", "2026-05-31", 92, "USD million", "s", "2026-06-12"],
], columns=_C).assign(url="", verification="synthetic", notes="test fixture")


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
    if name == "ramp_history":         # Ramp: month end + 20 days
        return pd.to_datetime(df["date_month"]) + pd.offsets.MonthEnd(0) + 20 * DAY
    if name == "github_family":        # a sampled week (Monday-Sunday) the day after it ends
        return pd.to_datetime(df["week_start"]) + 7 * DAY
    if name == "consumer":             # the stored date, full dates only; none: never
        return pd.to_datetime(df["published"], format="%Y-%m-%d", errors="coerce")
    raise KeyError(name)


def _price_published(p: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(p["release_date"])


def _mix_published(m: pd.DataFrame) -> pd.Series:       # a measured week is public once it is over
    return pd.to_datetime(m["week_start"]) + pd.Timedelta(days=7)


VALUE_COLS = {"indicators": ["value"], "github_new_repos": ["total_count"],
              "gateway": ["anthropic_value", "openai_value", "google_value", "deepseek_value"], "cloud": ["value"],
              "ramp_history": ["value"], "github_family": ["total_count"], "consumer": ["value"]}


@pytest.fixture(scope="module")
def src():
    s = F.load_sources()
    if not len(s["consumer"]):            # consumer_signals.csv not there yet: test the machinery on a stand-in
        s["consumer"] = SYNTH_CONSUMER.copy()
    return s


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
    for c in ("openrouter_log_tokens", "openrouter_dlog_tokens", "openrouter_share", "npm_dlog",
              "openrouter_share_chg", "openrouter_google_dlog_tokens", "openrouter_google_share", "aisdk_share",
              "aisdk_share_chg", "langchain_npm_share", "langchain_pypi_share", "npm_coding_agent_dlog",
              "npm_agent_sdk_dlog", "npm_bedrock_dlog", "npm_vertex_dlog", "npm_opencode_dlog", "npm_google_genai_dlog"):
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
        assert list(f.columns) == F.FEATURE_COLUMNS == F.FEATURE_SETS["all"]
    p = F.asof_panel("2026-11-12", sources=src)
    assert list(p["entity"].unique()) == list(ENTITIES)
    assert p.loc[p["entity"] == "openai", "log_price"].isna().all()
    with pytest.raises(ValueError):
        F.asof_features("2026-11-12", "nobody", sources=src)
    with pytest.raises(ValueError):
        F.asof_features("2026-11-12", sources=src, feature_set="some")


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


# --------------------------------------------------------------------------- 5. extra sources (FEATURE_SETS["all"])
@pytest.fixture(scope="module")
def ref_module(tmp_path_factory):
    """features.py as committed in REF_COMMIT, importable next to the current package."""
    try:
        code = subprocess.run(["git", "show", f"{REF_COMMIT}:runrate_nowcast/features.py"], cwd=config.PKG_DIR,
                              capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"git or commit {REF_COMMIT} not available")
    path = tmp_path_factory.mktemp("ref") / "features_ref.py"
    path.write_text(code)
    spec = importlib.util.spec_from_file_location("runrate_nowcast._features_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("vintage", ["2025-06-15", "2026-02-05", "2026-08-31", "2026-10-20", "2026-11-12"])
def test_g_spec_matches_committed_feature_store(vintage, ref_module, full):
    assert ref_module.FEATURE_COLUMNS == F.FEATURE_SETS["g_spec"]
    ref_src = ref_module.load_sources()
    for e in ENTITIES:
        ref = ref_module.asof_features(vintage, e, sources=ref_src)
        pd.testing.assert_frame_equal(ref, full(vintage)[e][F.FEATURE_SETS["g_spec"]], check_exact=True)
        pd.testing.assert_frame_equal(ref, F.asof_features(vintage, e, feature_set="g_spec"), check_exact=True)


def test_feature_sets():
    assert F.FEATURE_SETS["all"] == F.FEATURE_COLUMNS == list(F.FEATURE_DOCS)
    assert F.FEATURE_SETS["all"][:len(F.FEATURE_SETS["g_spec"])] == F.FEATURE_SETS["g_spec"]
    assert len(set(F.FEATURE_SETS["all"])) == len(F.FEATURE_SETS["all"]) and len(F.FEATURE_SETS["g_spec"]) == 12
    for c in NEW_COLUMNS:                          # every new column states where it comes from and when it is public
        assert "public" in F.FEATURE_DOCS[c] or F.FEATURE_DOCS[c].startswith("Change in"), c


def test_new_columns_coverage(full):
    a, o = full("2026-11-12")["anthropic"], full("2026-11-12")["openai"]
    anthropic_only = {"npm_agent_sdk_dlog", "npm_bedrock_dlog", "npm_vertex_dlog", "github_opus_share",
                      "github_opus_share_chg"}
    for c in set(NEW_COLUMNS) - set(CONSUMER_COLUMNS):
        assert a[c].notna().sum() >= 25, c
        assert o[c].isna().all() if c in anthropic_only else o[c].notna().sum() >= 25, c
    for c in ("openrouter_google_dlog_tokens", "openrouter_google_share", "npm_opencode_dlog", "npm_google_genai_dlog",
              "ramp_ai_adoption_chg", "ramp_pepm_median_dlog", "ramp_pepm_top10_dlog", "ramp_ai_spend_share"):
        pd.testing.assert_series_equal(a[c], o[c])                     # market-wide controls: same for both
    for c in ("langchain_npm_share", "langchain_pypi_share"):          # two-package shares add to one
        assert np.allclose((a[c] + o[c]).dropna(), 1.0)
    s = a["openrouter_share"] + o["openrouter_share"] + a["openrouter_google_share"]
    assert np.allclose(s.dropna(), 1.0)
    for f in (a, o):                                                    # weekly series: the current week is not over
        assert f.iloc[-1][["aisdk_share", "npm_opencode_dlog", "openrouter_google_share"]].isna().all()


def _bend_public(df, mask, cols, rng):
    d = df.copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
        d.loc[mask, c] = d.loc[mask, c] * rng.uniform(1.5, 3.0, int(mask.sum()))
    return d


POWER_CASES = {   # source rows published on/before the vintage -> new columns that must move
    "downloads": ("indicators", lambda d: d["indicator"].isin(F.DOWNLOAD_SERIES),
                  ["aisdk_share", "langchain_npm_share", "langchain_pypi_share", "npm_coding_agent_dlog",
                   "npm_agent_sdk_dlog", "npm_bedrock_dlog", "npm_vertex_dlog", "npm_opencode_dlog",
                   "npm_google_genai_dlog"]),
    "openrouter_google": ("indicators", lambda d: d["indicator"] == "openrouter_google_tokens",
                          ["openrouter_google_dlog_tokens", "openrouter_google_share", "openrouter_share_chg"]),
    "ramp_history": ("ramp_history", lambda d: d["series"].notna(),
                     ["ramp_ai_adoption_chg", "ramp_pepm_median_dlog", "ramp_pepm_top10_dlog", "ramp_ai_spend_share",
                      "ramp_vendor_share", "ramp_vendor_share_chg"]),
    "github_family": ("github_family", lambda d: d["query"].notna(), ["github_opus_share", "github_opus_share_chg"]),
    "consumer": ("consumer", lambda d: d["series"].notna(),
                 ["consumer_web_visits_share", "consumer_web_visits_dlog", "consumer_app_downloads_share"]),
}


@pytest.mark.parametrize("case", list(POWER_CASES))
def test_new_sources_have_power_and_leave_g_spec_alone(case, src, full):
    """Control for the truncation/perturbation tests: changing values public at the vintage
    does move the new columns. The original columns never read the new sources."""
    v = pd.Timestamp("2026-10-20")
    key, rows, cols = POWER_CASES[case]
    s = dict(src, consumer=SYNTH_CONSUMER.copy()) if case == "consumer" else src
    base = {e: F.asof_features(v, e, sources=s) for e in ENTITIES}
    d = s[key]
    mask = (rows(d) & (_published(key, d) <= v)).to_numpy()
    bent = dict(s, **{key: _bend_public(d, mask, VALUE_COLS[key], np.random.default_rng(7))})
    g = F.FEATURE_SETS["g_spec"]
    for e in ENTITIES:
        moved = F.asof_features(v, e, sources=bent)
        for c in cols:
            if base[e][c].notna().any():
                assert not np.allclose(moved[c], base[e][c], equal_nan=True), (e, c)
        if case != "openrouter_google":                     # Google tokens are in openrouter_share's denominator
            pd.testing.assert_frame_equal(moved[g], base[e][g], check_exact=True)


# ---- publication dates of the new sources
def test_ramp_history_file_dates_every_row_by_the_rule(src):
    r = src["ramp_history"]
    rule = (pd.to_datetime(r["date_month"]) + pd.offsets.MonthEnd(0) + 20 * DAY).dt.strftime("%Y-%m-%d")
    assert (r["published"] == rule).all()
    assert (r["fetched"] == "2026-11-12").all() and r["note"].str.contains("revis").all()
    need = {"adoption_rate_pct:overall", "adoption_rate_pct:Anthropic", "adoption_rate_pct:OpenAI", "median_pepm",
            "top_10_percent_median_pepm", "ai_share_of_tpv"}
    assert need <= set(r["series"])
    assert r.groupby("series")["date_month"].min().loc[list(need)].eq("2023-07-01").all()
    # the latest month on the page (October 2026) is dated after the fetch by the rule: unused on 2026-11-12
    assert r["date_month"].max() == "2026-10-01" and r["published"].max() == "2026-11-20"


def _ramp_value(r, series, month):
    return float(r.loc[(r["series"] == series) & (r["date_month"] == month), "value"].iloc[0])


def test_ramp_history_month_appears_on_its_publication_date(src, full):
    r = src["ramp_history"]
    share = {m: _ramp_value(r, "adoption_rate_pct:Anthropic", m) / _ramp_value(r, "adoption_rate_pct:overall", m)
             for m in ("2026-07-01", "2026-08-01", "2026-09-01", "2026-10-01")}
    before, after = full("2026-10-19")["anthropic"], full("2026-10-20")["anthropic"]
    assert before["ramp_vendor_share"].iloc[-1] == pytest.approx(share["2026-08-01"])
    assert not np.isclose(before["ramp_vendor_share"], share["2026-09-01"]).any()
    assert after["ramp_vendor_share"].iloc[-1] == pytest.approx(share["2026-09-01"])
    assert after["ramp_vendor_share_chg"].iloc[-1] == pytest.approx(share["2026-09-01"] - share["2026-08-01"])
    assert after["ramp_ai_adoption_chg"].iloc[-1] == pytest.approx(
        (_ramp_value(r, "adoption_rate_pct:overall", "2026-09-01") - _ramp_value(r, "adoption_rate_pct:overall", "2026-08-01")) / 100)
    assert after["ramp_pepm_median_dlog"].iloc[-1] == pytest.approx(
        math.log(_ramp_value(r, "median_pepm", "2026-09-01") / _ramp_value(r, "median_pepm", "2026-08-01")))
    assert after["ramp_ai_spend_share"].iloc[-1] == pytest.approx(_ramp_value(r, "ai_share_of_tpv", "2026-09-01"))
    sep = after["ramp_vendor_share"]
    assert (sep.index[np.isclose(sep, share["2026-09-01"])] + pd.Timedelta(days=3) >= pd.Timestamp("2026-10-20")).all()
    now = full("2026-11-12")["anthropic"]                      # October (rule date 2026-11-20) not yet
    assert now["ramp_vendor_share"].iloc[-1] == pytest.approx(share["2026-09-01"])
    assert not np.isclose(now["ramp_vendor_share"], share["2026-10-01"]).any()
    assert _last_obs("2026-11-12", src, "ramp_vendor_share") == pd.Timestamp("2026-09-01")
    o = full("2026-10-20")["openai"]["ramp_vendor_share"].iloc[-1]
    assert o == pytest.approx(_ramp_value(r, "adoption_rate_pct:OpenAI", "2026-09-01")
                              / _ramp_value(r, "adoption_rate_pct:overall", "2026-09-01"))


def test_ramp_parser_dates_rows_by_the_rule():
    props = {"adoptionOverall": [{"date_month": "2026-09-01", "adoption_rate_pct": 55.8, "mom_change_pp": 0.6}],
             "adoptionVendor": [{"date_month": "2026-09-01", "vendor": "Anthropic", "adoption_rate_pct": 43.7}],
             "spendPerEmployee": [{"date_month": "2026-09-01", "median_pepm": 15.4,
                                   "p99_winsorized_weighted_pepm": "$undefined"}],
             "spendShareData": {"overall": [{"my_date": "2026-09-01", "ai_share_of_tpv": 0.0568}]}}
    import json
    row = '1f:["$","div",null,' + json.dumps({"children": ["$", "$L2", None, props]}) + "]\n"
    html = (f"<script>self.__next_f.push([1,{json.dumps(row)}])</script>"
            '<script>{"dateModified":"2026-11-11T16:32:43Z"}</script>')
    d = extra_sources.parse_ramp_history(html, "2026-11-12")
    assert set(d["series"]) == {"adoption_rate_pct:overall", "adoption_rate_pct:Anthropic", "median_pepm",
                                "ai_share_of_tpv"}           # "$undefined" values are not rows
    assert (d["published"] == "2026-10-20").all() and (d["fetched"] == "2026-11-12").all()
    assert d["note"].str.contains("2026-11-11T16:32:43Z").all()


def test_github_family_sample_appears_the_day_after_its_week(src, full):
    fam = src["github_family"].pivot_table(index="week_start", columns="query", values="total_count")
    share = fam["Claude Opus"] / (fam["Claude Opus"] + fam["Claude Sonnet"])
    f0, f1 = full("2026-10-11")["anthropic"], full("2026-10-12")["anthropic"]   # week of 2026-10-05 ends Sunday 10-11
    assert f0["github_opus_share"].iloc[-1] == pytest.approx(share["2026-09-07"])
    assert not np.isclose(f0["github_opus_share"], share["2026-10-05"]).any()
    assert f1["github_opus_share"].iloc[-1] == pytest.approx(share["2026-10-05"])
    assert f1["github_opus_share_chg"].iloc[-1] == pytest.approx(share["2026-10-05"] - share["2026-09-07"])
    assert _last_obs("2026-10-12", src, "github_opus_share") == pd.Timestamp("2026-10-05")
    # the first trailer weeks are too small to measure (pricing.MIN_MEASURED_COMMITS), as in log_price
    now = full("2026-11-12")["anthropic"]["github_opus_share"]
    assert not np.isclose(now, share["2025-11-03"]).any()
    assert now.first_valid_index() == pd.Timestamp("2025-12-11")             # 2025-12-01 sample, public 12-08
    assert full("2026-11-12")["openai"]["github_opus_share"].isna().all()


def test_github_family_carried_at_most_62_days(src):
    fam = src["github_family"]
    s = dict(src, github_family=fam[pd.to_datetime(fam["week_start"]) <= "2026-08-10"])    # last one public 08-17
    f = F.asof_features("2026-11-12", sources=s)["github_opus_share"]
    assert f.notna()[pd.Timestamp("2026-10-15")] and f[f.index >= pd.Timestamp("2026-10-22")].isna().all()


# ---- consumer signals (synthetic stand-in; the same rules apply to the real file)
_CONSUMER_SRC = {}


def _consumer(vintage, entity="anthropic"):
    if not _CONSUMER_SRC:
        _CONSUMER_SRC.update(F.load_sources(), consumer=SYNTH_CONSUMER.copy())
    return F.asof_features(vintage, entity, sources=_CONSUMER_SRC)


def test_consumer_rows_used_from_their_published_date_only():
    jul, aug1, aug2 = 950 / (950 + 5800 + 1100), 960 / (960 + 5700 + 1300), 980 / (980 + 5700 + 1300)
    assert _consumer("2026-08-24")["consumer_web_visits_share"].isna().all()       # Gemini July not out yet
    assert _consumer("2026-08-25")["consumer_web_visits_share"].iloc[-1] == pytest.approx(jul)
    assert _consumer("2026-09-21")["consumer_web_visits_share"].iloc[-1] == pytest.approx(jul)
    assert _consumer("2026-09-22")["consumer_web_visits_share"].iloc[-1] == pytest.approx(aug1)   # 1.3 billion
    f = _consumer("2026-10-05")
    assert f["consumer_web_visits_share"].iloc[-1] == pytest.approx(aug2)      # later figure, from its own date
    assert f.loc[pd.Timestamp("2026-09-24"), "consumer_web_visits_share"] == pytest.approx(aug1)   # not rewritten
    assert f["consumer_web_visits_dlog"].iloc[-1] == pytest.approx(math.log(980 / 950))
    assert _consumer("2026-09-18")["consumer_web_visits_dlog"].iloc[-1] == pytest.approx(math.log(960 / 950))
    assert _consumer("2026-09-18", "openai")["consumer_web_visits_dlog"].iloc[-1] == pytest.approx(math.log(5700 / 5800))
    assert _consumer("2026-10-05", "openai")["consumer_web_visits_share"].iloc[-1] == pytest.approx(5700 / 7980)
    # never used: unique visitors (09-10), a one-day window (10-25), no / month-only publication date (September)
    assert np.isnan(_consumer("2026-09-10")["consumer_web_visits_dlog"].iloc[-1])
    late = _consumer("2026-11-12")
    assert late["consumer_web_visits_dlog"].iloc[-1] == pytest.approx(math.log(980 / 950))
    assert late["consumer_web_visits_share"].iloc[-1] == pytest.approx(aug2)
    # downloads: a month's share needs all three; the change needs the month before
    f = _consumer("2026-06-10")
    assert f["consumer_app_downloads_share"].iloc[-1] == pytest.approx(20 / 110)
    assert f["consumer_app_downloads_dlog"].iloc[-1] == pytest.approx(math.log(18 / 20))
    assert f["consumer_app_revenue_share"].isna().all() and f["consumer_app_revenue_dlog"].isna().all()


def test_consumer_values_carried_at_most_62_days():
    f = _consumer("2026-12-31")["consumer_web_visits_share"]          # last figure public 2026-10-05
    assert f[pd.Timestamp("2026-12-03")] == pytest.approx(980 / 7980)  # cutoff 12-06 = 10-05 + 62 days
    assert f[f.index >= pd.Timestamp("2026-12-10")].isna().all()


def test_consumer_file_optional(tmp_path, src):
    missing = F.load_sources({"consumer": os.path.join(tmp_path, "none.csv")})
    assert list(missing["consumer"].columns) == F.OPTIONAL_SOURCES["consumer"] and not len(missing["consumer"])
    f = F.asof_features("2026-11-12", sources=missing)
    assert f[CONSUMER_COLUMNS].isna().all().all()
    bad = os.path.join(tmp_path, "bad.csv")
    pd.DataFrame({"entity": ["Claude"], "value": [1.0]}).to_csv(bad, index=False)
    with pytest.warns(UserWarning):
        assert not len(F.load_sources({"consumer": bad})["consumer"])
    real = F.SOURCE_FILES["consumer"]
    if os.path.exists(real):                                         # once it lands it must have the agreed columns
        assert set(F.OPTIONAL_SOURCES["consumer"]) <= set(pd.read_csv(real).columns)
