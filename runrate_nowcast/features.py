"""As-of feature store: weekly features built only from values public at a vintage.

Every source row has a publication date, either stored or set by the rule in FEATURE_DOCS.
At vintage v a feature uses a row only if it was published on or before v. That filter is
applied once, in `_public_obs`, right after the rows are read; everything downstream sees
public rows only. The price index is the exception: pricing.price_index is evaluated at dates
on or before v and uses only prices and commit weeks public by each date.

Index: one row per week (Monday to Sunday), labelled by its Thursday, as in ml.weekly_features.
Rows run from the week of FIRST_WEEK to the week that contains the vintage. The last row is
the current week, which is not over yet: its weekly series are NaN until the week has ended
and been published, but monthly and quarterly values already published by the vintage show.

How values are placed on weeks
  weekly series (OpenRouter, SDK downloads): row t holds week t's own value, once published.
  monthly / quarterly series: row t holds the latest value published on or before the row's
      cutoff min(end of week t, vintage). Values are carried forward after publication,
      never back to earlier weeks. A value is carried for at most MAX_CARRY after its own
      publication, so a series that stops does not stay frozen.
  price: pricing.price_index at min(Thursday of week t, vintage).

Sources read: SOURCE_FILES below, plus data/model_prices.csv and data/github_claude_family.csv
through pricing.price_index. observations.csv, which holds the filing numbers (the targets),
is never read.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from . import altdata, config, pricing, robust

DATA_DIR = os.path.join(config.PKG_DIR, "data")
SOURCE_FILES = {
    "indicators": config.INDICATORS_PATH,
    "github_new_repos": os.path.join(DATA_DIR, "github_new_repos.csv"),
    "gateway": os.path.join(DATA_DIR, "gateway_usage_snapshots.csv"),
    "cloud": os.path.join(DATA_DIR, "cloud_partner_disclosures.csv"),
}

FIRST_WEEK = config.MODEL_START                    # first row: the week containing this date
DAY = pd.Timedelta(days=1)
SDK_PUBLISH_DELAY = DAY                            # npm / pypistats show a day's count the next day
WEEKLY_PUBLISH_DELAY = DAY                         # fallback when a weekly row has no stored `published`
RAMP_PUBLISH_DELAY = pd.Timedelta(days=20)         # fallback when a Ramp row has no stored `published`
MAX_CARRY = {"monthly": pd.Timedelta(days=62), "quarterly": pd.Timedelta(days=100)}
MIN_DAYS_PER_WEEK = 5                              # SDK weekly mean needs at least this many days

ENTITIES = {
    "anthropic": {"openrouter": "openrouter_anthropic_tokens", "github": "claude",
                  "ramp": "ramp_adoption_anthropic", "gateway": "anthropic_value",
                  "npm": "npm_anthropic_sdk", "pypi": "pypi_anthropic", "price": True, "cloud": True},
    "openai": {"openrouter": "openrouter_openai_tokens", "github": "openai",
               "ramp": "ramp_adoption_openai", "gateway": "openai_value",
               "npm": "npm_openai", "pypi": "pypi_openai", "price": False, "cloud": False},
    "google": {"openrouter": "openrouter_google_tokens", "github": "gemini",
               "ramp": None, "gateway": "google_value", "npm": None, "pypi": None,
               "price": False, "cloud": False},
}
OPENROUTER_AUTHORS = ("openrouter_anthropic_tokens", "openrouter_openai_tokens", "openrouter_google_tokens")
GITHUB_QUERIES = ("claude", "openai", "gemini")
SDK_SERIES = ("npm_anthropic_sdk", "npm_openai", "pypi_anthropic", "pypi_openai")   # break-detection pairs

# Cloud partner growth statements used, as annualized log growth (Anthropic only). Left out:
# customer counts (bedrock_claude_customers, foundry_customers_using_anthropic_and_openai[_qoq]),
# AWS / Microsoft AI run-rates (they include Anthropic's own compute purchases), segment totals,
# backlog and investment marks.
CLOUD_GROWTH = {
    "bedrock_customer_spend_qoq": ("cloud_bedrock_growth", lambda x: 4 * math.log1p(x / 100)),   # % QoQ
    "bedrock_fortune500_spend_yoy": ("cloud_bedrock_growth", lambda x: math.log(x)),             # x YoY
    "foundry_revenue_yoy": ("cloud_foundry_growth", lambda x: math.log1p(x / 100)),              # % YoY, lower bound
}

# One entry per output column: source, frequency, publication rule.
FEATURE_DOCS = {
    "openrouter_log_tokens":
        "OpenRouter author-market-share (indicators.csv openrouter_<entity>_tokens): log tokens in the week. "
        "Weekly, dated at the week's Sunday; public on the stored `published` date (the Monday after). "
        "Row t = week t.",
    "openrouter_dlog_tokens":
        "Same source: log change from the previous week. NaN if either week is missing or unpublished.",
    "openrouter_share":
        "Same source: entity tokens / (anthropic + openai + google) tokens in the week. "
        "Needs all three published.",
    "github_repo_share":
        "github_new_repos.csv: new public repos mentioning the entity (claude / openai / gemini) "
        "divided by the sum of the three, monthly. Public the day after the month ends. "
        "Latest month public at the row's cutoff, carried at most 62 days.",
    "ramp_adoption":
        "Ramp AI Index (indicators.csv ramp_adoption_<entity>): share of US businesses on Ramp paying the "
        "vendor, monthly. Public on the stored `published` date (month end + 20 days). Latest month public "
        "at the row's cutoff, carried at most 62 days.",
    "ramp_adoption_chg":
        "Same source: change in that share from the month before, both months as known at the cutoff.",
    "vercel_spend_share":
        "gateway_usage_snapshots.csv: Vercel AI Gateway, spend_share, scope 'all traffic', calendar-month "
        "window; the entity's share as a fraction. Public on `captured_or_published`; rows without a full "
        "date there are never used. A later figure for the same month replaces the first one only from its "
        "own publication date. Latest month at the row's cutoff, carried at most 62 days.",
    "log_price":
        "pricing.price_index(dates, 'github'), log: list prices of models released by the date, weighted by "
        "the Opus/Sonnet commit mix of the latest measured week complete by the date. Evaluated at the "
        "week's Thursday, or at the vintage if earlier. Anthropic only.",
    "npm_dlog":
        "npm daily downloads of the entity's SDK (anthropic: @anthropic-ai/sdk, openai: openai). Weekly mean "
        "over Monday-Sunday (at least 5 days; Dec 20 - Jan 6 dropped), log change from the previous week. "
        "A day is public the next day; a week counts only once complete. Counting breaks found by "
        "robust.detect_breaks on data public at the vintage are removed. Control variable.",
    "pypi_dlog":
        "Same as npm_dlog for PyPI (anthropic / openai), pypistats without mirrors. Control variable.",
    "cloud_bedrock_growth":
        "cloud_partner_disclosures.csv, Amazon: latest Bedrock spend-growth statement as annualized log "
        "growth (bedrock_customer_spend_qoq: 4*log(1+x/100); bedrock_fortune500_spend_yoy: log(x)). "
        "Quarterly, public on `published` (the earnings date), carried at most 100 days. Anthropic only.",
    "cloud_foundry_growth":
        "cloud_partner_disclosures.csv, Microsoft: latest Foundry revenue growth (foundry_revenue_yoy, a lower "
        "bound) as log(1+x/100). Quarterly, public on `published`, carried at most 100 days. Not "
        "Claude-specific (Foundry is mostly Azure OpenAI). Anthropic only.",
}
FEATURE_COLUMNS = list(FEATURE_DOCS)


# --------------------------------------------------------------------------- loading
def load_sources(paths: dict | None = None) -> dict[str, pd.DataFrame]:
    """Raw source tables, as stored. Never includes observations.csv."""
    paths = {**SOURCE_FILES, **(paths or {})}
    return {k: pd.read_csv(p) for k, p in paths.items()}


def _entity_spec(entity: str) -> dict:
    if entity not in ENTITIES:
        raise ValueError(f"unknown entity {entity!r}; known: {sorted(ENTITIES)}")
    return ENTITIES[entity]


def _date(s) -> pd.Series:
    return pd.to_datetime(pd.Series(s), format="%Y-%m-%d", errors="coerce")


def _all_obs(src: dict[str, pd.DataFrame], entity: str) -> dict[str, pd.DataFrame]:
    """Every source row the entity's features may use, as [series, ref, published, value].

    `ref` is the date the value describes, `published` the date it became public. Nothing is
    filtered by vintage here; `_public_obs` does that.
    """
    spec = _entity_spec(entity)
    cols = ["series", "ref", "published", "value"]
    out: dict[str, pd.DataFrame] = {}

    ind = src["indicators"]
    date, pub = _date(ind["date"]), _date(ind["published"])
    val = pd.to_numeric(ind["value"], errors="coerce")

    def ind_rows(names, delay):
        m = ind["indicator"].isin(names).to_numpy() & (val > 0).to_numpy()
        return pd.DataFrame({"series": ind["indicator"][m].to_numpy(), "ref": date[m].to_numpy(),
                             "published": pub[m].fillna(date[m] + delay).to_numpy(),
                             "value": val[m].to_numpy()}, columns=cols)

    out["openrouter"] = ind_rows(OPENROUTER_AUTHORS, WEEKLY_PUBLISH_DELAY)
    out["sdk"] = ind_rows(SDK_SERIES, SDK_PUBLISH_DELAY)
    out["ramp"] = ind_rows([spec["ramp"]] if spec["ramp"] else [], RAMP_PUBLISH_DELAY)

    # GitHub: monthly counts; a month is public the day after it ends.
    g = src["github_new_repos"]
    g = g[g["query"].isin(GITHUB_QUERIES)]
    wide = g.pivot_table(index="month", columns="query", values="total_count", aggfunc="last")
    wide = wide.reindex(columns=list(GITHUB_QUERIES)).dropna()
    month = _date(wide.index)
    share = (wide[spec["github"]] / wide.sum(axis=1)).to_numpy()
    out["github"] = pd.DataFrame({"series": "github_repo_share", "ref": month.to_numpy(),
                                  "published": (month + pd.offsets.MonthBegin(1)).to_numpy(),
                                  "value": share}, columns=cols)

    # Vercel AI Gateway monthly spend share
    gw = src["gateway"]
    ws, we = _date(gw["window_start"]), _date(gw["window_end"])
    monthly = (ws.dt.day == 1).to_numpy() & (we == ws + pd.offsets.MonthEnd(0)).to_numpy()
    m = ((gw["platform"] == "Vercel AI Gateway") & (gw["metric"] == "spend_share")
         & (gw["scope"] == "all traffic")).to_numpy() & monthly
    share = pd.to_numeric(gw[spec["gateway"]], errors="coerce") / 100 if spec["gateway"] in gw \
        else pd.Series(np.nan, index=gw.index)
    out["vercel"] = pd.DataFrame({"series": "vercel_spend_share", "ref": ws[m].to_numpy(),
                                  "published": _date(gw["captured_or_published"])[m].to_numpy(),
                                  "value": share[m].to_numpy()}, columns=cols)

    # Cloud partners: growth statements, converted to annualized log growth
    rows = []
    if spec["cloud"]:
        c = src["cloud"]
        for r in c[c["metric"].isin(list(CLOUD_GROWTH))].itertuples():
            x = pd.to_numeric(r.value, errors="coerce")
            if pd.isna(x) or x <= 0:
                continue
            col, conv = CLOUD_GROWTH[r.metric]
            rows.append({"series": col, "ref": pd.Period(str(r.fiscal_period), freq="Q").end_time.normalize(),
                         "published": _date([r.published]).iloc[0], "value": conv(float(x))})
    out["cloud"] = pd.DataFrame(rows, columns=cols)

    for k, d in out.items():
        d["ref"] = pd.to_datetime(d["ref"])
        d["published"] = pd.to_datetime(d["published"])
        d["value"] = d["value"].astype(float)
        out[k] = d.dropna(subset=["ref", "value"]).reset_index(drop=True)
    return out


def _public_obs(obs: dict[str, pd.DataFrame], vintage: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """THE as-of filter: keep rows published on or before the vintage (no publication date: dropped)."""
    return {k: d[d["published"].notna() & (d["published"] <= vintage)].reset_index(drop=True)
            for k, d in obs.items()}


# --------------------------------------------------------------------------- helpers
def _vintage(vintage) -> pd.Timestamp:
    return pd.Timestamp(vintage).normalize()


def _week_index(vintage: pd.Timestamp) -> pd.DatetimeIndex:
    """Thursday labels of the weeks (ending Sunday) from FIRST_WEEK to the week containing the vintage."""
    first = pd.Timestamp(FIRST_WEEK).to_period("W-SUN").end_time.normalize()
    last = vintage.to_period("W-SUN").end_time.normalize()
    ends = pd.date_range(first, last, freq="W-SUN") if last >= first else pd.DatetimeIndex([])
    return pd.DatetimeIndex(ends - pd.Timedelta(days=3), name="week")


def _week_label(dates) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(dates)
    return d.to_period("W-SUN").to_timestamp(how="end").normalize() - pd.Timedelta(days=3)


def _latest_by_ref(o: pd.DataFrame) -> pd.DataFrame:
    """Per ref, the most recently published value (same-day duplicates averaged)."""
    o = o.groupby(["ref", "published"], as_index=False)["value"].mean()
    return o.sort_values(["ref", "published"]).drop_duplicates("ref", keep="last").set_index("ref")


def _weekly(o: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.Series:
    """A weekly-native series (one value per week, ref = a day in the week) on the week grid."""
    if not len(o):
        return pd.Series(np.nan, index=weeks)
    s = _latest_by_ref(o)["value"]
    s.index = _week_label(s.index)
    return s[~s.index.duplicated(keep="last")].reindex(weeks)


def _carry(o: pd.DataFrame, cutoffs: pd.Series, max_age: pd.Timedelta, freq: str = "M"):
    """Latest period's value public at each cutoff (carried forward after publication) and its
    change from the period before, both as known at the cutoff.

    The as-known series only changes on publication dates, so it is built once per date: each
    period keeps its most recently published value (a revision takes over from its own date;
    same-day duplicates are averaged). A value older than max_age at the cutoff is dropped.
    """
    o = o.assign(per=o["ref"].dt.to_period(freq)).groupby(["published", "per"], as_index=False)["value"].mean()
    known: dict = {}                                      # period -> (published, value)
    dates, states = [], []
    for p, day in o.groupby("published", sort=True):
        for r in day.itertuples():
            known[r.per] = (p, r.value)
        last = max(known)
        prev = known.get(last - 1)
        dates.append(p)
        states.append((known[last][0], known[last][1], known[last][1] - prev[1] if prev else np.nan))
    level = np.full(len(cutoffs), np.nan)
    chg = np.full(len(cutoffs), np.nan)
    pos = np.searchsorted(np.array(dates, dtype="datetime64[ns]"),
                          cutoffs.to_numpy(dtype="datetime64[ns]"), side="right") - 1
    for i, (c, j) in enumerate(zip(cutoffs, pos)):
        if j >= 0 and c - states[j][0] <= max_age:
            level[i], chg[i] = states[j][1], states[j][2]
    return pd.Series(level, index=cutoffs.index), pd.Series(chg, index=cutoffs.index)


def _sdk_weekly_log(sdk: pd.DataFrame, name: str, vintage: pd.Timestamp, weeks: pd.DatetimeIndex) -> pd.Series:
    """Log weekly mean of daily downloads (complete, published weeks only), break-adjusted."""
    d = _latest_by_ref(sdk[sdk["series"] == name])["value"] if name else pd.Series(dtype=float)
    if not len(d):
        return pd.Series(np.nan, index=weeks)
    md = d.index.month * 100 + d.index.day
    d = d[~((md >= altdata.HOLIDAY_START) | (md <= altdata.HOLIDAY_END))]
    g = d.resample("W-SUN")
    wk = pd.DataFrame({"v": g.mean(), "n": g.count()})
    wk = wk[(wk["n"] >= MIN_DAYS_PER_WEEK) & (wk.index + SDK_PUBLISH_DELAY <= vintage)]
    out = np.log(wk["v"])
    out.index = out.index - pd.Timedelta(days=3)
    return out.reindex(weeks)


def _break_adjusted(sdk: pd.DataFrame) -> pd.DataFrame:
    """Remove counting breaks (robust.detect_breaks rule) found in the public SDK rows: values
    before a break are scaled by (1 + step), as ml.weekly_features does."""
    if not len(sdk):
        return sdk
    raw = sdk.rename(columns={"series": "indicator", "ref": "date"})[["date", "indicator", "value"]]
    present = set(raw["indicator"])
    pairs = tuple(p for p in (("pypi_anthropic", "pypi_openai"), ("npm_anthropic_sdk", "npm_openai"))
                  if set(p) <= present)
    br = robust.detect_breaks(raw, pairs=pairs) if pairs else pd.DataFrame()
    out = sdk.copy()
    for b in br.itertuples():
        a, c = b.source_pair.split(" / ")
        for name, step in ((a, b.step_anthropic), (c, b.step_competitor)):
            m = (out["series"] == name) & (out["ref"] < b.date)
            out.loc[m, "value"] = out.loc[m, "value"] * (1 + step)
    return out


# --------------------------------------------------------------------------- public API
def asof_features(vintage, entity: str = "anthropic", freq: str = "W",
                  sources: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Weekly features for `entity` using only values published on or before `vintage`.

    Index: Thursday labels of weeks ending Sunday, up to the week containing the vintage.
    Columns: FEATURE_COLUMNS (see FEATURE_DOCS); NaN where a source has no public value.
    `sources` (from load_sources) can be passed to reuse or substitute the raw tables.
    """
    if freq != "W":
        raise ValueError("only weekly features (freq='W') are implemented")
    spec = _entity_spec(entity)
    v = _vintage(vintage)
    src = load_sources() if sources is None else sources
    pub = _public_obs(_all_obs(src, entity), v)
    weeks = _week_index(v)
    f = pd.DataFrame(index=weeks, columns=FEATURE_COLUMNS, dtype=float)
    if not len(weeks):
        return f
    cutoff = pd.Series(np.minimum(weeks + pd.Timedelta(days=3), v), index=weeks)

    # OpenRouter (weekly)
    o = pub["openrouter"]
    tok = {a: _weekly(o[o["series"] == a], weeks) for a in OPENROUTER_AUTHORS}
    mine = tok[spec["openrouter"]]
    f["openrouter_log_tokens"] = np.log(mine)
    f["openrouter_dlog_tokens"] = f["openrouter_log_tokens"].diff()
    f["openrouter_share"] = mine / sum(tok.values())

    # monthly, carried forward after publication
    f["github_repo_share"], _ = _carry(pub["github"], cutoff, MAX_CARRY["monthly"])
    f["ramp_adoption"], f["ramp_adoption_chg"] = _carry(pub["ramp"], cutoff, MAX_CARRY["monthly"])
    f["vercel_spend_share"], _ = _carry(pub["vercel"], cutoff, MAX_CARRY["monthly"])

    # price per token (Anthropic only)
    if spec["price"]:
        dates = list(np.minimum(weeks, v))
        f["log_price"] = np.log(pricing.price_index(dates, "github").to_numpy(dtype=float))

    # SDK downloads: weekly log change only
    sdk = _break_adjusted(pub["sdk"])
    f["npm_dlog"] = _sdk_weekly_log(sdk, spec["npm"], v, weeks).diff()
    f["pypi_dlog"] = _sdk_weekly_log(sdk, spec["pypi"], v, weeks).diff()

    # cloud partners (quarterly, Anthropic only)
    c = pub["cloud"]
    for col in ("cloud_bedrock_growth", "cloud_foundry_growth"):
        f[col], _ = _carry(c[c["series"] == col], cutoff, MAX_CARRY["quarterly"], freq="Q")
    return f


def asof_panel(vintage, entities=("anthropic", "openai"),
               sources: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """asof_features for several entities, stacked, with an `entity` column (index: week)."""
    src = load_sources() if sources is None else sources
    parts = [asof_features(vintage, e, sources=src).assign(entity=e) for e in entities]
    p = pd.concat(parts)
    return p[["entity"] + FEATURE_COLUMNS]


_FEATURE_INPUTS = {   # feature -> (obs key, series name or None for the entity's own)
    "openrouter_log_tokens": ("openrouter", "openrouter"), "openrouter_dlog_tokens": ("openrouter", "openrouter"),
    "openrouter_share": ("openrouter", None), "github_repo_share": ("github", None),
    "ramp_adoption": ("ramp", None), "ramp_adoption_chg": ("ramp", None), "vercel_spend_share": ("vercel", None),
    "npm_dlog": ("sdk", "npm"), "pypi_dlog": ("sdk", "pypi"),
    "cloud_bedrock_growth": ("cloud", "cloud_bedrock_growth"), "cloud_foundry_growth": ("cloud", "cloud_foundry_growth"),
}


def feature_availability(vintage, entities=("anthropic", "openai"),
                         sources: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Audit table: per entity and feature, the inputs public at the vintage and their latest
    reference date (last_obs) and publication date (last_published)."""
    v = _vintage(vintage)
    src = load_sources() if sources is None else sources
    rows = []
    for e in entities:
        spec = _entity_spec(e)
        pub = _public_obs(_all_obs(src, e), v)
        for feat, (key, ser) in _FEATURE_INPUTS.items():
            o = pub[key]
            if ser in spec:
                o = o[o["series"] == spec[ser]] if spec[ser] else o.iloc[:0]
            elif ser is not None:
                o = o[o["series"] == ser]
            rows.append({"entity": e, "feature": feat, "input": key, "n_obs": len(o),
                         "last_obs": o["ref"].max() if len(o) else pd.NaT,
                         "last_published": o["published"].max() if len(o) else pd.NaT})
        if spec["price"]:
            pr = pricing.load_prices()
            pr = pr[pr["release_date"] <= v]
            rows.append({"entity": e, "feature": "log_price", "input": "model_prices", "n_obs": len(pr),
                         "last_obs": pr["release_date"].max() if len(pr) else pd.NaT,
                         "last_published": pr["release_date"].max() if len(pr) else pd.NaT})
            mix = pd.read_csv(pricing.FAMILY_MIX_PATH) if os.path.exists(pricing.FAMILY_MIX_PATH) \
                else pd.DataFrame({"week_start": []})
            mix = mix.assign(week_start=pd.to_datetime(mix["week_start"]))
            mix = mix[mix["week_start"] + pd.Timedelta(days=7) <= v]
            rows.append({"entity": e, "feature": "log_price", "input": "github_claude_family",
                         "n_obs": int(mix["week_start"].nunique()),
                         "last_obs": mix["week_start"].max() if len(mix) else pd.NaT,
                         "last_published": mix["week_start"].max() + pd.Timedelta(days=7) if len(mix) else pd.NaT})
    out = pd.DataFrame(rows)
    out["rule"] = out["feature"].map(FEATURE_DOCS)
    return out
