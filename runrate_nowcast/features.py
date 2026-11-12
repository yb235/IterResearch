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

Feature sets: FEATURE_SETS["g_spec"] holds the original columns (sub-task E; the panel model
and the frozen Q4 forecasts use them, and their values do not change); FEATURE_SETS["all"]
adds the columns built from the extra sources (framework download shares, coding-agent and
cloud SDK downloads, OpenRouter competitor shares, the GitHub model-family mix and the Ramp
AI Index history). asof_features returns all columns.

Sources read: SOURCE_FILES below, plus data/model_prices.csv and data/github_claude_family.csv
through pricing.price_index. observations.csv, which holds the filing numbers (the targets),
is never read.
"""
from __future__ import annotations

import math
import os
import re
import warnings

import numpy as np
import pandas as pd

from . import altdata, config, pricing, robust

DATA_DIR = os.path.join(config.PKG_DIR, "data")
SOURCE_FILES = {
    "indicators": config.INDICATORS_PATH,
    "github_new_repos": os.path.join(DATA_DIR, "github_new_repos.csv"),
    "gateway": os.path.join(DATA_DIR, "gateway_usage_snapshots.csv"),
    "cloud": os.path.join(DATA_DIR, "cloud_partner_disclosures.csv"),
    # extra sources (FEATURE_SETS["all"] only)
    "ramp_history": os.path.join(DATA_DIR, "ramp_ai_index_history.csv"),
    "github_family": pricing.FAMILY_MIX_PATH,
    "consumer": os.path.join(DATA_DIR, "consumer_signals.csv"),
}
# Optional sources: a missing file (or one without these columns) reads as an empty table, so
# its columns are NaN until the file appears; it is then used with no code change.
OPTIONAL_SOURCES = {
    "consumer": ["series", "entity", "metric", "period_start", "period_end", "value", "unit", "source",
                 "published", "url", "verification", "notes"],
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
SDK_BREAK_PAIRS = (("pypi_anthropic", "pypi_openai"), ("npm_anthropic_sdk", "npm_openai"))

# --- extra sources (FEATURE_SETS["all"] only). Per entity: its own series (None: not collected).
EXTRA_ENTITIES = {
    "anthropic": {"aisdk": "npm_aisdk_anthropic", "langchain_npm": "npm_langchain_anthropic",
                  "langchain_pypi": "pypi_langchain_anthropic", "coding_agent": "npm_claude_code",
                  "agent_sdk": "npm_claude_agent_sdk", "bedrock": "npm_bedrock_sdk", "vertex": "npm_vertex_sdk",
                  "family_mix": True, "ramp_vendor": "Anthropic"},
    "openai": {"aisdk": "npm_aisdk_openai", "langchain_npm": "npm_langchain_openai",
               "langchain_pypi": "pypi_langchain_openai", "coding_agent": "npm_openai_codex",
               "agent_sdk": None, "bedrock": None, "vertex": None, "family_mix": False, "ramp_vendor": "OpenAI"},
    "google": {"aisdk": "npm_aisdk_google", "langchain_npm": None, "langchain_pypi": None, "coding_agent": None,
               "agent_sdk": None, "bedrock": None, "vertex": None, "family_mix": False, "ramp_vendor": None},
}
AISDK = ("npm_aisdk_anthropic", "npm_aisdk_openai", "npm_aisdk_google")
LANGCHAIN_NPM = ("npm_langchain_anthropic", "npm_langchain_openai")
LANGCHAIN_PYPI = ("pypi_langchain_anthropic", "pypi_langchain_openai")
DOWNLOAD_CONTROLS = {"npm_opencode_dlog": "npm_opencode", "npm_google_genai_dlog": "npm_google_genai"}
DOWNLOAD_OWN = {"npm_coding_agent_dlog": "coding_agent", "npm_agent_sdk_dlog": "agent_sdk",
                "npm_bedrock_dlog": "bedrock", "npm_vertex_dlog": "vertex"}
DOWNLOAD_SERIES = AISDK + LANGCHAIN_NPM + LANGCHAIN_PYPI + (
    "npm_claude_code", "npm_openai_codex", "npm_claude_agent_sdk", "npm_bedrock_sdk", "npm_vertex_sdk",
    "npm_opencode", "npm_google_genai")
FAMILY_QUERIES = ("Claude Opus", "Claude Sonnet")
FAMILY_SAMPLE_WEEKS = 4                            # github_claude_family.csv samples one week in four
# Ramp history file series -> (obs series, transform)
RAMP_HISTORY_SERIES = {
    "adoption_rate_pct:overall": ("ramp_ai_adoption", lambda x: x / 100),
    "median_pepm": ("ramp_log_pepm_median", np.log),
    "top_10_percent_median_pepm": ("ramp_log_pepm_top10", np.log),
    "ai_share_of_tpv": ("ramp_ai_spend_share", lambda x: x),
}
# Consumer signals (consumer_signals.csv): entity names and metric keywords -> canonical names
CONSUMER_ENTITY = {"anthropic": "anthropic", "claude": "anthropic", "openai": "openai", "chatgpt": "openai",
                   "google": "google", "gemini": "google"}
CONSUMER_METRICS = ("web_visits", "app_downloads", "app_revenue")
CONSUMER_ENTITIES = ("anthropic", "openai", "google")

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

    # ---------------- extra sources: FEATURE_SETS["all"] only ----------------
    "openrouter_share_chg":
        "OpenRouter author-market-share, as openrouter_share: change in the entity's share from the previous "
        "week. Weekly; a week is public on its stored `published` date (the Monday after it ends).",
    "openrouter_google_dlog_tokens":
        "Same source, competitor control: log change in Google's weekly tokens (openrouter_google_tokens). "
        "A week is public on its stored `published` date (the Monday after). Same values for every entity.",
    "openrouter_google_share":
        "Same source: Google tokens / (anthropic + openai + google) in the week; with openrouter_share it fixes "
        "all three shares. Needs all three public (stored `published`, the Monday after). Same values for "
        "every entity.",
    "openrouter_google_share_chg":
        "Change in openrouter_google_share from the previous week.",
    "aisdk_share":
        "npm daily downloads of the Vercel AI SDK provider packages @ai-sdk/anthropic, @ai-sdk/openai, "
        "@ai-sdk/google (indicators.csv npm_aisdk_*): the entity's package's weekly mean over the sum of the "
        "three weekly means. Weekly means as in npm_dlog (Monday-Sunday, at least 5 days, Dec 20 - Jan 6 "
        "dropped). npm rows have no stored date: a day is public the next day, so a week counts from the "
        "Monday after it ends. Level.",
    "aisdk_share_chg":
        "Change in aisdk_share from the previous week (NaN if either week is missing).",
    "langchain_npm_share":
        "npm @langchain/anthropic and @langchain/openai (npm_langchain_*): the entity's weekly mean over the sum "
        "of the two. Weekly means and publication as in aisdk_share. Level.",
    "langchain_npm_share_chg":
        "Change in langchain_npm_share from the previous week.",
    "langchain_pypi_share":
        "PyPI langchain-anthropic and langchain-openai (pypistats without mirrors, pypi_langchain_*; history "
        "from 2026-04-08 only): the entity's share of the two. The pypistats counting change of 2026-08-25 "
        "moved the two by different amounts (about -38% and -53%), so counting breaks that robust.detect_breaks "
        "finds in the pair, on data public at the vintage, are removed first (earlier values scaled by 1+step, "
        "as for pypi_dlog); until a break is detectable (about four weeks later) the share shows the jump. "
        "Weekly means and publication as in aisdk_share (a day is public the next day). Level.",
    "langchain_pypi_share_chg":
        "Change in langchain_pypi_share from the previous week.",
    "npm_coding_agent_dlog":
        "npm daily downloads of the entity's coding-agent CLI (anthropic: @anthropic-ai/claude-code, "
        "npm_claude_code; openai: @openai/codex, npm_openai_codex): log change of the weekly mean. Weekly means "
        "and publication as in npm_dlog (a day is public the next day, a week once complete); no break "
        "adjustment (no npm counting change is known).",
    "npm_agent_sdk_dlog":
        "@anthropic-ai/claude-agent-sdk (npm_claude_agent_sdk): weekly log change as npm_coding_agent_dlog "
        "(a day is public the next day, a week once complete). Anthropic only (no OpenAI counterpart collected).",
    "npm_bedrock_dlog":
        "@anthropic-ai/bedrock-sdk (npm_bedrock_sdk): weekly log change as npm_coding_agent_dlog (a day is "
        "public the next day, a week once complete). Cloud-channel proxy, Anthropic only.",
    "npm_vertex_dlog":
        "@anthropic-ai/vertex-sdk (npm_vertex_sdk): weekly log change as npm_coding_agent_dlog (a day is "
        "public the next day, a week once complete). Cloud-channel proxy, Anthropic only.",
    "npm_opencode_dlog":
        "opencode-ai (npm_opencode), a coding agent that works with every vendor: weekly log change as "
        "npm_coding_agent_dlog (a day is public the next day, a week once complete). Market-wide control, "
        "same values for every entity.",
    "npm_google_genai_dlog":
        "@google/genai (npm_google_genai): weekly log change as npm_coding_agent_dlog (a day is public the "
        "next day, a week once complete). Competitor control, same values for every entity.",
    "github_opus_share":
        "data/github_claude_family.csv: 'Claude Opus' / ('Claude Opus' + 'Claude Sonnet') counts of public "
        "commits with the Co-Authored-By: Claude <model> trailer, in one sampled week (Monday-Sunday) every 4 "
        "weeks. A sampled week is public the day after it ends (week_start + 7 days). Weeks with fewer than "
        "pricing.MIN_MEASURED_COMMITS commits in total are dropped, as in log_price. Latest sample public at "
        "the row's cutoff, carried at most 62 days. MEASURED AFTER THE FACT: every count comes from GitHub "
        "commit search run on 2026-11-11 (approximate counts, incomplete_results), not from searches run at "
        "each date; deleted or privatized repos and late indexing can make them differ from what a search on "
        "the day would have shown. Anthropic only.",
    "github_opus_share_chg":
        "Change in github_opus_share from the sample 4 weeks earlier, both as known at the cutoff "
        "(NaN if that sample is missing).",
    "ramp_ai_adoption_chg":
        "data/ramp_ai_index_history.csv (ramp.com/data/ai-index, embedded history fetched once on 2026-11-12, "
        "see extra_sources.py): change from the month before in the share of US businesses on Ramp with any AI "
        "spend (adoption_rate_pct:overall / 100), both months as known at the cutoff. Monthly; public on the "
        "stored `published` date, month end + 20 days (conservative: October 2026 was on the page 11 days "
        "after month end). Latest month carried at most 62 days. Change only: the level is an adoption curve. "
        "Same values for every entity. REVISION RISK: Ramp recomputes the whole history at each release and "
        "the file holds the 2026-11-12 values for every month (vendor adoption differed from the 2026-11-11 "
        "fetch by up to 0.10 pp), so earlier vintages see revised values, not the first-published ones.",
    "ramp_pepm_median_dlog":
        "Same file and rules as ramp_ai_adoption_chg (public month end + 20 days, carried at most 62 days; "
        "revised history): log change in the median firm's AI spend per employee per month, among firms "
        "paying for AI (median_pepm, USD). Same values for every entity.",
    "ramp_pepm_top10_dlog":
        "Same file and rules (public month end + 20 days, carried at most 62 days; revised history): log "
        "change in the median within the top 10% of those firms (top_10_percent_median_pepm). Stands in for "
        "the spend-weighted measure (p99_winsorized_weighted_pepm), which the page leaves empty. Same values "
        "for every entity.",
    "ramp_ai_spend_share":
        "Same file and rules (public month end + 20 days, carried at most 62 days; revised history): AI share "
        "of business spend excluding payroll (ai_share_of_tpv, a fraction). Level. Same values for every "
        "entity.",
    "ramp_ai_spend_share_chg":
        "Change in ramp_ai_spend_share from the month before, both as known at the cutoff.",
    "ramp_vendor_share":
        "Same file and rules (public month end + 20 days, carried at most 62 days; revised history): the entity's adoption rate (adoption_rate_pct:Anthropic or :OpenAI) over overall "
        "AI adoption, i.e. the share of AI-adopting businesses that pay the entity. A month is public once both "
        "parts are (each version uses the latest public value of both). Level. Anthropic and OpenAI only "
        "(Google is not in Ramp's vendor list).",
    "ramp_vendor_share_chg":
        "Change in ramp_vendor_share from the month before, both as known at the cutoff.",
}
_CONSUMER_LABEL = {"web_visits": "monthly web visits (claude.ai / chatgpt.com / gemini.google.com)",
                   "app_downloads": "monthly mobile app downloads", "app_revenue": "monthly mobile app revenue"}
for _m in CONSUMER_METRICS:
    FEATURE_DOCS[f"consumer_{_m}_share"] = (
        f"data/consumer_signals.csv (optional; NaN while the file is absent): {_CONSUMER_LABEL[_m]}, the "
        "entity's value over the sum for anthropic + openai + google (Claude, ChatGPT, Gemini) in the same "
        "calendar month (rows whose period_start..period_end is not one calendar month are ignored). Public on "
        "the row's `published` date (YYYY-MM-DD, the first article reporting it); rows without one are never "
        "used. A later figure for a month takes over from its own publication date; each month's share uses the "
        "latest public figure of all three, and needs all three. Latest month carried at most 62 days. Units "
        "scaled by the `unit` text (thousand/million/billion); sources are mixed if the file mixes them. Level.")
    FEATURE_DOCS[f"consumer_{_m}_dlog"] = (
        f"Same file and rules (public on `published`, carried at most 62 days): log change in the entity's own "
        f"{_CONSUMER_LABEL[_m]} from the month before, both months as known at the cutoff (NaN unless both "
        "months were reported).")

G_SPEC_COLUMNS = ["openrouter_log_tokens", "openrouter_dlog_tokens", "openrouter_share", "github_repo_share",
                  "ramp_adoption", "ramp_adoption_chg", "vercel_spend_share", "log_price", "npm_dlog",
                  "pypi_dlog", "cloud_bedrock_growth", "cloud_foundry_growth"]
FEATURE_SETS = {"g_spec": G_SPEC_COLUMNS, "all": list(FEATURE_DOCS)}
assert FEATURE_SETS["all"][:len(G_SPEC_COLUMNS)] == G_SPEC_COLUMNS
FEATURE_COLUMNS = FEATURE_SETS["all"]           # what asof_features returns


# --------------------------------------------------------------------------- loading
def load_sources(paths: dict | None = None) -> dict[str, pd.DataFrame]:
    """Raw source tables, as stored. Never includes observations.csv."""
    paths = {**SOURCE_FILES, **(paths or {})}
    out = {}
    for k, p in paths.items():
        cols = OPTIONAL_SOURCES.get(k)
        if cols is None:
            out[k] = pd.read_csv(p)
            continue
        d = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=cols)
        missing = set(cols) - set(d.columns)
        if missing:                       # never let an optional file break the original features
            warnings.warn(f"{os.path.basename(p)} lacks columns {sorted(missing)}; not used")
            d = pd.DataFrame(columns=cols)
        out[k] = d
    return out


def _entity_spec(entity: str) -> dict:
    if entity not in ENTITIES:
        raise ValueError(f"unknown entity {entity!r}; known: {sorted(ENTITIES)}")
    return ENTITIES[entity]


def _date(s) -> pd.Series:
    return pd.to_datetime(pd.Series(s), format="%Y-%m-%d", errors="coerce")


def _all_obs(src: dict[str, pd.DataFrame], entity: str, extra_sources: bool = True) -> dict[str, pd.DataFrame]:
    """Every source row the entity's features may use, as [series, ref, published, value].

    `ref` is the date the value describes, `published` the date it became public. Nothing is
    filtered by vintage here; `_public_obs` does that. extra_sources=False leaves out the
    sources that only FEATURE_SETS["all"] uses.
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

    # ---- extra sources (FEATURE_SETS["all"] only)
    extra = EXTRA_ENTITIES[entity] if extra_sources else {"family_mix": False, "ramp_vendor": None}
    out["downloads"] = ind_rows(DOWNLOAD_SERIES if extra_sources else [], SDK_PUBLISH_DELAY)
    out["family"] = _family_obs(src.get("github_family")) if extra["family_mix"] else pd.DataFrame(columns=cols)
    out["ramp_hist"] = _ramp_history_obs(src.get("ramp_history") if extra_sources else None, extra["ramp_vendor"])
    out["consumer"] = _consumer_obs(src.get("consumer") if extra_sources else None, entity)
    for k in ("downloads", "family", "ramp_hist", "consumer"):    # one datetime unit across obs tables
        out[k] = out[k].astype({"ref": _DT, "published": _DT})

    for k, d in out.items():
        d["ref"] = pd.to_datetime(d["ref"])
        d["published"] = pd.to_datetime(d["published"])
        d["value"] = d["value"].astype(float)
        out[k] = d.dropna(subset=["ref", "value"]).reset_index(drop=True)
    return out


_OBS_COLS = ["series", "ref", "published", "value"]
_DT = pd.to_datetime(pd.Series(["2000-01-01"]), format="%Y-%m-%d").dtype     # the obs tables' datetime unit


def _as_known_combos(parts: list[pd.DataFrame], combine) -> pd.DataFrame:
    """Combine several components of one quantity per ref, as they were known over time.

    One row per (ref, date on which any component was published), from each component's
    latest value public on that date (same-day values averaged); every component is needed.
    So a revision of one part enters only from its own publication date, and the combined row
    is never dated before its last part became public.
    """
    parts = [p.dropna(subset=["ref", "published"]).groupby(["ref", "published"], as_index=False)["value"].mean()
             .astype({"ref": _DT, "published": _DT, "value": float})
             .sort_values("published") for p in parts]
    keys = pd.concat([p[["ref", "published"]] for p in parts]).drop_duplicates().sort_values("published")
    if not len(parts) or not len(keys):
        return pd.DataFrame(columns=["ref", "published", "value"])
    vals = [pd.merge_asof(keys, p, on="published", by="ref", direction="backward")["value"].to_numpy()
            for p in parts]
    ok = np.logical_and.reduce([~np.isnan(x) for x in vals])
    out = keys.assign(value=np.where(ok, combine(vals), np.nan) if ok.any() else np.nan)[ok]
    return out.sort_values(["ref", "published"]).reset_index(drop=True)


def _family_obs(fam: pd.DataFrame | None) -> pd.DataFrame:
    """Opus share of Opus+Sonnet trailer commits per sampled week, public the day after the week."""
    if fam is None or not len(fam):
        return pd.DataFrame(columns=_OBS_COLS)
    g = fam[fam["query"].isin(FAMILY_QUERIES)].assign(n=lambda x: pd.to_numeric(x["total_count"], errors="coerce"))
    wide = g.pivot_table(index="week_start", columns="query", values="n", aggfunc="last")
    wide = wide.reindex(columns=list(FAMILY_QUERIES)).dropna()
    wide = wide[wide.sum(axis=1) >= pricing.MIN_MEASURED_COMMITS]       # as pricing._measured
    week = _date(wide.index)
    return pd.DataFrame({"series": "github_opus_share", "ref": week.to_numpy(),
                         "published": (week + pd.Timedelta(days=7)).to_numpy(),
                         "value": (wide[FAMILY_QUERIES[0]] / wide.sum(axis=1)).to_numpy()}, columns=_OBS_COLS)


def _ramp_history_obs(r: pd.DataFrame | None, vendor: str | None) -> pd.DataFrame:
    """National Ramp AI Index series (data/ramp_ai_index_history.csv), and the vendor's share of
    AI-adopting businesses. `published` as stored, else month end + 20 days."""
    if r is None or not len(r):
        return pd.DataFrame(columns=_OBS_COLS)
    month = _date(r["date_month"])
    pub = _date(r["published"]).fillna(month + pd.offsets.MonthEnd(0) + RAMP_PUBLISH_DELAY)
    d = pd.DataFrame({"name": r["series"].to_numpy(), "ref": month.to_numpy(), "published": pub.to_numpy(),
                      "value": pd.to_numeric(r["value"], errors="coerce").to_numpy()})
    parts = []
    for name, (series, conv) in RAMP_HISTORY_SERIES.items():
        x = d[(d["name"] == name) & (d["value"] > 0)]
        parts.append(pd.DataFrame({"series": series, "ref": x["ref"].to_numpy(), "published": x["published"].to_numpy(),
                                   "value": conv(x["value"].to_numpy(dtype=float))}, columns=_OBS_COLS))
    if vendor:
        own = d[d["name"] == f"adoption_rate_pct:{vendor}"][["ref", "published", "value"]]
        allai = d[(d["name"] == "adoption_rate_pct:overall") & (d["value"] > 0)][["ref", "published", "value"]]
        parts.append(_as_known_combos([own, allai], lambda v: v[0] / v[1]).assign(series="ramp_vendor_share"))
    return pd.concat([p[_OBS_COLS] for p in parts], ignore_index=True)


def _consumer_entity(text) -> str | None:
    hits = {v for k, v in CONSUMER_ENTITY.items() if k in str(text).lower()}
    return hits.pop() if len(hits) == 1 else None


def _consumer_metric(text) -> str | None:
    t = str(text).lower()
    if "unique" in t or "visitor" in t:          # a different count from visits
        return None
    if "visit" in t or "traffic" in t:
        return "web_visits"
    if "download" in t or "install" in t:
        return "app_downloads"
    if "revenue" in t or "spend" in t:
        return "app_revenue"
    return None


def _unit_scale(text) -> float:
    t = str(text).lower()
    for pat, k in ((r"billion|\bbn\b|\bb\b", 1e9), (r"million|\bmn\b|\bmm\b|\bm\b", 1e6), (r"thousand|\bk\b", 1e3)):
        if re.search(pat, t):
            return k
    return 1.0


def _calendar_month(start: pd.Series, end: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Month start of each row and whether its window is exactly one calendar month
    (YYYY-MM-DD or YYYY-MM; a YYYY-MM start with no end is the whole month)."""
    ps_day = pd.to_datetime(start, format="%Y-%m-%d", errors="coerce")
    ps_mon = pd.to_datetime(start, format="%Y-%m", errors="coerce")
    ps = ps_day.fillna(ps_mon)
    pe = pd.to_datetime(end, format="%Y-%m-%d", errors="coerce")
    pe = pe.fillna(pd.to_datetime(end, format="%Y-%m", errors="coerce") + pd.offsets.MonthEnd(0))
    pe = pe.fillna(ps_mon + pd.offsets.MonthEnd(0))
    return ps, (ps.dt.day == 1) & (pe == ps + pd.offsets.MonthEnd(0))


def _consumer_obs(c: pd.DataFrame | None, entity: str) -> pd.DataFrame:
    """consumer_signals.csv: the entity's log value and its share among the three entities, per
    metric and calendar month. Public on `published` (YYYY-MM-DD); rows without it never used."""
    if c is None or not len(c) or entity not in CONSUMER_ENTITIES:
        return pd.DataFrame(columns=_OBS_COLS)
    ent = c["entity"].map(_consumer_entity)
    met = pd.Series([_consumer_metric(m) or _consumer_metric(s) for m, s in zip(c["metric"], c["series"])],
                    index=c.index)
    month, monthly = _calendar_month(c["period_start"], c["period_end"])
    val = pd.to_numeric(c["value"], errors="coerce") * c["unit"].map(_unit_scale)
    pub = _date(c["published"])
    ok = monthly & ent.notna() & met.notna() & (val > 0) & pub.notna()
    d = pd.DataFrame({"entity": ent, "metric": met, "ref": month, "published": pub, "value": val})[ok]
    i = CONSUMER_ENTITIES.index(entity)
    parts = []
    for m in CONSUMER_METRICS:
        x = d[d["metric"] == m]
        comps = [x[x["entity"] == e][["ref", "published", "value"]] for e in CONSUMER_ENTITIES]
        parts.append(comps[i].assign(series=f"consumer_{m}_log", value=np.log(comps[i]["value"].astype(float))))
        parts.append(_as_known_combos(comps, lambda v: v[i] / sum(v)).assign(series=f"consumer_{m}_share"))
    return pd.concat([p[_OBS_COLS] for p in parts], ignore_index=True)


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


def _carry(o: pd.DataFrame, cutoffs: pd.Series, max_age: pd.Timedelta, freq: str = "M", lag: int = 1):
    """Latest period's value public at each cutoff (carried forward after publication) and its
    change from the period `lag` periods before, both as known at the cutoff.

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
        prev = known.get(last - lag)
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


def _weekly_logs(dl: pd.DataFrame, vintage: pd.Timestamp, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """_sdk_weekly_log for every series in dl at once (one column per series): log of the
    Monday-Sunday mean of daily values, weeks with at least MIN_DAYS_PER_WEEK days (Dec 20 -
    Jan 6 dropped) that ended at least SDK_PUBLISH_DELAY before the vintage."""
    if not len(dl):
        return pd.DataFrame(index=weeks)
    d = dl.groupby(["series", "ref", "published"], as_index=False)["value"].mean()
    d = d.sort_values(["series", "ref", "published"]).drop_duplicates(["series", "ref"], keep="last")
    md = d["ref"].dt.month * 100 + d["ref"].dt.day
    d = d[~((md >= altdata.HOLIDAY_START) | (md <= altdata.HOLIDAY_END))]
    d = d.assign(week_end=d["ref"].dt.to_period("W-SUN").dt.end_time.dt.normalize())
    g = d.groupby(["series", "week_end"])["value"].agg(["mean", "count"]).reset_index()
    g = g[(g["count"] >= MIN_DAYS_PER_WEEK) & (g["week_end"] + SDK_PUBLISH_DELAY <= vintage)]
    w = g.pivot(index="week_end", columns="series", values="mean")
    w.index = w.index - pd.Timedelta(days=3)
    return np.log(w).reindex(weeks)


def _break_adjusted(sdk: pd.DataFrame, pairs=SDK_BREAK_PAIRS) -> pd.DataFrame:
    """Remove counting breaks (robust.detect_breaks rule) found in the public SDK rows: values
    before a break are scaled by (1 + step), as ml.weekly_features does."""
    if not len(sdk):
        return sdk
    raw = sdk.rename(columns={"series": "indicator", "ref": "date"})[["date", "indicator", "value"]]
    present = set(raw["indicator"])
    pairs = tuple(p for p in pairs if set(p) <= present)
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
                  sources: dict[str, pd.DataFrame] | None = None, feature_set: str = "all") -> pd.DataFrame:
    """Weekly features for `entity` using only values published on or before `vintage`.

    Index: Thursday labels of weeks ending Sunday, up to the week containing the vintage.
    Columns: FEATURE_SETS[feature_set] (default "all" = FEATURE_COLUMNS; see FEATURE_DOCS); NaN
    where a source has no public value. feature_set="g_spec" returns the original columns only,
    with the same values, without reading the extra sources (about half the time).
    `sources` (from load_sources) can be passed to reuse or substitute the raw tables.
    """
    if freq != "W":
        raise ValueError("only weekly features (freq='W') are implemented")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature_set {feature_set!r}; known: {sorted(FEATURE_SETS)}")
    spec = _entity_spec(entity)
    v = _vintage(vintage)
    src = load_sources() if sources is None else sources
    pub = _public_obs(_all_obs(src, entity, extra_sources=feature_set != "g_spec"), v)
    weeks = _week_index(v)
    f = pd.DataFrame(index=weeks, columns=FEATURE_SETS[feature_set], dtype=float)
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

    # ---------------- extra sources: FEATURE_SETS["all"] only (the columns above do not depend on them)
    if feature_set == "g_spec":
        return f
    extra = EXTRA_ENTITIES[entity]
    # OpenRouter: own share change, Google as competitor
    f["openrouter_share_chg"] = f["openrouter_share"].diff()
    goog = tok["openrouter_google_tokens"]
    f["openrouter_google_dlog_tokens"] = np.log(goog).diff()
    f["openrouter_google_share"] = goog / sum(tok.values())
    f["openrouter_google_share_chg"] = f["openrouter_google_share"].diff()

    # downloads: weekly means of complete, published weeks (LangChain PyPI pair break-adjusted)
    dl = pub["downloads"]
    lc = dl["series"].isin(LANGCHAIN_PYPI)
    wide = _weekly_logs(pd.concat([dl[~lc], _break_adjusted(dl[lc], pairs=(LANGCHAIN_PYPI,))]), v, weeks)
    wk = {n: wide[n] if n in wide else pd.Series(np.nan, index=weeks) for n in DOWNLOAD_SERIES}
    for col, group in (("aisdk_share", AISDK), ("langchain_npm_share", LANGCHAIN_NPM),
                       ("langchain_pypi_share", LANGCHAIN_PYPI)):
        own = extra[col[:-len("_share")]]
        if own:
            f[col] = np.exp(wk[own]) / sum(np.exp(wk[n]) for n in group)
            f[col + "_chg"] = f[col].diff()
    for col, key in DOWNLOAD_OWN.items():
        if extra[key]:
            f[col] = wk[extra[key]].diff()
    for col, name in DOWNLOAD_CONTROLS.items():
        f[col] = wk[name].diff()

    # GitHub model-family mix (Anthropic only), one sampled week in four
    f["github_opus_share"], f["github_opus_share_chg"] = _carry(
        pub["family"], cutoff, MAX_CARRY["monthly"], freq="W", lag=FAMILY_SAMPLE_WEEKS)

    # Ramp AI Index history (monthly)
    rh = pub["ramp_hist"]

    def ramp(series):
        return _carry(rh[rh["series"] == series], cutoff, MAX_CARRY["monthly"])
    _, f["ramp_ai_adoption_chg"] = ramp("ramp_ai_adoption")
    _, f["ramp_pepm_median_dlog"] = ramp("ramp_log_pepm_median")
    _, f["ramp_pepm_top10_dlog"] = ramp("ramp_log_pepm_top10")
    f["ramp_ai_spend_share"], f["ramp_ai_spend_share_chg"] = ramp("ramp_ai_spend_share")
    f["ramp_vendor_share"], f["ramp_vendor_share_chg"] = ramp("ramp_vendor_share")

    # consumer signals (optional file; monthly news points)
    cs = pub["consumer"]
    for m in CONSUMER_METRICS:
        f[f"consumer_{m}_share"], _ = _carry(cs[cs["series"] == f"consumer_{m}_share"], cutoff, MAX_CARRY["monthly"])
        _, f[f"consumer_{m}_dlog"] = _carry(cs[cs["series"] == f"consumer_{m}_log"], cutoff, MAX_CARRY["monthly"])
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


def _extra_feature_inputs(entity: str) -> dict[str, tuple[str, tuple]]:
    """New feature -> (obs key, the obs series it reads) for the entity; () when it has none."""
    x = EXTRA_ENTITIES[entity]
    group = {"aisdk": AISDK, "langchain_npm": LANGCHAIN_NPM, "langchain_pypi": LANGCHAIN_PYPI}
    out = {"openrouter_share_chg": ("openrouter", OPENROUTER_AUTHORS),
           "openrouter_google_dlog_tokens": ("openrouter", ("openrouter_google_tokens",)),
           "openrouter_google_share": ("openrouter", OPENROUTER_AUTHORS),
           "openrouter_google_share_chg": ("openrouter", OPENROUTER_AUTHORS)}
    for k, g in group.items():
        out[f"{k}_share"] = out[f"{k}_share_chg"] = ("downloads", g if x[k] else ())
    for col, key in DOWNLOAD_OWN.items():
        out[col] = ("downloads", (x[key],) if x[key] else ())
    for col, name in DOWNLOAD_CONTROLS.items():
        out[col] = ("downloads", (name,))
    out["github_opus_share"] = out["github_opus_share_chg"] = ("family", ("github_opus_share",))
    for col, series in (("ramp_ai_adoption_chg", "ramp_ai_adoption"), ("ramp_pepm_median_dlog", "ramp_log_pepm_median"),
                        ("ramp_pepm_top10_dlog", "ramp_log_pepm_top10"), ("ramp_ai_spend_share", "ramp_ai_spend_share"),
                        ("ramp_ai_spend_share_chg", "ramp_ai_spend_share"), ("ramp_vendor_share", "ramp_vendor_share"),
                        ("ramp_vendor_share_chg", "ramp_vendor_share")):
        out[col] = ("ramp_hist", (series,))
    for m in CONSUMER_METRICS:
        out[f"consumer_{m}_share"] = ("consumer", (f"consumer_{m}_share",))
        out[f"consumer_{m}_dlog"] = ("consumer", (f"consumer_{m}_log",))
    assert list(out) == FEATURE_SETS["all"][len(G_SPEC_COLUMNS):]
    return out


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
        for feat, (key, names) in _extra_feature_inputs(e).items():      # FEATURE_SETS["all"] only
            o = pub[key][pub[key]["series"].isin(names)]
            rows.append({"entity": e, "feature": feat, "input": key, "n_obs": len(o),
                         "last_obs": o["ref"].max() if len(o) else pd.NaT,
                         "last_published": o["published"].max() if len(o) else pd.NaT})
    out = pd.DataFrame(rows)
    out["rule"] = out["feature"].map(FEATURE_DOCS)
    return out
