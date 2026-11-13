"""List price per token over time, and usage indicators adjusted for price.

Revenue = usage x price. SDK downloads and commits measure usage; when Anthropic cuts
prices, the same usage brings in less revenue. If an indicator i tracks usage,
    log(ind) = a + beta * (log(revenue) - log(P))
so the price-adjusted indicator log(ind) + gamma * log(P) tracks revenue, with gamma = beta
when usage does not respond to price and gamma = 0 when usage rises one-for-one with price
cuts (revenue per developer unchanged). gamma is not set by hand: the backtest compares
gamma = 0 / 0.5 / 1 and weights them by the online-filter likelihood at each vintage.

P uses only prices public on each date (model release dates in data/model_prices.csv).
Every Claude model's output price is 5x its input price, so the index does not depend on
the input/output token split; cache-read discounts are not modeled.

The model mix is the open question. Weight schemes:
  uninformed     Opus 50 / Sonnet 50. Uses no information from after any vintage.
  nov2026_mix    Opus 80 / Sonnet 20, the share of public GitHub commits co-authored by
                 Opus 5.5 vs Sonnet 5.5 in the week of 2026-11-01. HINDSIGHT for earlier
                 vintages: a sensitivity, not a forecast.
  github         measured mix at each date from data/github_claude_family.csv (when collected).

Fixed-weight schemes start at 2026-01-01 (flat before). The Opus price cut of 2025-11-24
(-67%) came with a large shift of usage from Sonnet to Opus, so its net effect on the price
per token actually paid is unknown without a measured mix; before 2026 the indicator
intercepts absorb it, as they did before this module existed. In 2026 the family prices
change on 2026-06-30 (Sonnet -33%) and 2026-09-22 (Opus -20%); Haiku and Fable are left
out (their usage share is unmeasured).

Per-model index (schemes "model" and "model_cache", added 2026-11-13; see model_counts and
model_price_level):
  P(t) = sum_m s_m(t) p_m        s_m = share of model m among public commits whose Claude Code
                                 trailer names a model (data/github_claude_models.csv, plus the
                                 family and total files), p_m = model m's price.
  "model"        p_m = list price. Every Claude model's output price is 5x its input price, so any
                 fixed input/output split gives the same index.
  "model_cache"  p_m = price of a cache-heavy agentic basket per prompt token: 0.85 cache read +
                 0.15 uncached input + 0.015 output (CACHE_BASKET; shares measured on OpenRouter's
                 standard Claude traffic in Oct-Nov 2026, where Opus/Sonnet/Fable prompts were
                 80-90% cache reads and output was 1-2% of prompt). A sensitivity: cache-read prices
                 fell more than list prices for the 5.x models.
  Real time: on date t the shares come from the latest sample week that ended by t - 1 day.
  A model released after that week takes over its family's share at once (Claude Code's
  opus / sonnet / haiku aliases move to the newest model; the samples confirm the switch
  within weeks). Before the first sample week with enough named commits, the index is
  flat at its first measured value: see FALLBACK below.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from . import config

PRICES_PATH = os.path.join(config.PKG_DIR, "data", "model_prices.csv")
FAMILY_MIX_PATH = os.path.join(config.PKG_DIR, "data", "github_claude_family.csv")
WEIGHTS = {"uninformed": {"opus": 0.5, "sonnet": 0.5}, "nov2026_mix": {"opus": 0.8, "sonnet": 0.2}}
BASE_DATE = "2026-01-01"
FIXED_START = "2026-01-01"


def load_prices(path: str = PRICES_PATH) -> pd.DataFrame:
    p = pd.read_csv(path, parse_dates=["release_date"])
    return p.sort_values("release_date").reset_index(drop=True)


def family_price(date, family: str, prices: pd.DataFrame | None = None) -> float:
    """Output price ($/Mtok) of the newest model of `family` released on or before `date`."""
    prices = load_prices() if prices is None else prices
    f = prices[(prices.family == family) & (prices.release_date <= pd.Timestamp(date))]
    return float(f.iloc[-1].output_usd_mtok) if len(f) else float("nan")


MIN_MEASURED_COMMITS = 10_000     # a week counts as measured only with this many commits (both families);
                                  # model names entered the trailer around Nov 2025 and the first weeks are tiny


def _measured(g: pd.DataFrame) -> pd.DataFrame:
    tot = g.groupby("week_start").total_count.sum()
    fams = g.groupby("week_start")["query"].nunique()
    ok = tot[(tot >= MIN_MEASURED_COMMITS) & (fams >= 2)].index
    return g[g.week_start.isin(ok)]


def _github_weights(date) -> dict | None:
    if not os.path.exists(FAMILY_MIX_PATH):
        return None
    g = _measured(pd.read_csv(FAMILY_MIX_PATH, parse_dates=["week_start"]).dropna(subset=["total_count"]))
    g = g[g.week_start + pd.Timedelta(days=7) <= pd.Timestamp(date)]       # only weeks complete by `date`
    if not len(g):
        return None
    last = g[g.week_start == g.week_start.max()]
    w = {q.split()[-1].lower(): float(n) for q, n in zip(last["query"], last.total_count)}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()} if s > 0 else None


def price_level(date, scheme: str = "uninformed", prices: pd.DataFrame | None = None) -> float:
    prices = load_prices() if prices is None else prices
    w = _github_weights(date) if scheme == "github" else WEIGHTS[scheme]
    if not w:                       # no measured mix yet at this date: uninformed weights (never a later mix)
        w, scheme = WEIGHTS["uninformed"], "uninformed"
    if scheme != "github":
        date = max(pd.Timestamp(date), pd.Timestamp(FIXED_START))
    vals = {f: family_price(date, f, prices) for f in w}
    vals = {f: v for f, v in vals.items() if np.isfinite(v)}
    tot = sum(w[f] for f in vals)
    return sum(w[f] * v for f, v in vals.items()) / tot


def _first_measured_week():
    if not os.path.exists(FAMILY_MIX_PATH):
        return None
    g = _measured(pd.read_csv(FAMILY_MIX_PATH, parse_dates=["week_start"]).dropna(subset=["total_count"]))
    return None if not len(g) else g.week_start.min() + pd.Timedelta(days=7)


def price_index(dates, scheme: str = "uninformed", base: str = BASE_DATE) -> pd.Series:
    """Price per token relative to a base. For the measured (github) mix: the unit value
    sum_f w_f(t) p_f(t) relative to the first measured week, flat before it (a constant
    before measurement starts carries no information; the indicator intercepts absorb it)."""
    prices = load_prices()
    if scheme == "github":
        t0 = _first_measured_week()
        if t0 is None:
            return pd.Series(1.0, index=pd.DatetimeIndex(dates))
        b = price_level(t0, "github", prices)
        return pd.Series([price_level(d, "github", prices) / b if pd.Timestamp(d) >= t0 else 1.0 for d in dates],
                         index=pd.DatetimeIndex(dates))
    b = price_level(base, scheme, prices)
    return pd.Series([price_level(d, scheme, prices) / b for d in dates], index=pd.DatetimeIndex(dates))


def adjust_indicators(df: pd.DataFrame, gamma: float, scheme: str = "uninformed") -> pd.DataFrame:
    """Copy of df with indicator values multiplied by P(ref_date)^gamma."""
    m = df.metric == "indicator"
    if gamma == 0 or not m.any():
        return df
    d = df.copy()
    dates = pd.to_datetime(d.loc[m, "ref_date"])
    uniq = sorted(set(dates))
    idx = price_index(uniq, scheme)
    d.loc[m, "value"] = d.loc[m, "value"].to_numpy() * idx.reindex(dates).to_numpy() ** gamma
    return d


def gamma_weights(df: pd.DataFrame, vintage, gammas=(0.0, 0.5, 1.0), scheme: str = "uninformed",
                  exclude_origin=None) -> pd.Series:
    """Posterior weight of each gamma from the online-filter likelihood of data public at the
    vintage, with unfitted parameters (flat prior over the grid)."""
    from . import backtest, data
    from .model import Model
    ll = {}
    for g in gammas:
        sel = data.select(adjust_indicators(df, g, scheme), vintage=vintage)
        if exclude_origin is not None:
            sel = sel[sel.origin_id != exclude_origin]
        ll[g] = Model(backtest.unfitted_params()).run(sel, smooth=False).loglik
    s = pd.Series(ll)
    w = np.exp(s - s.max())
    return w / w.sum()



# =========================================================================== per-model index
MODELS_PATH = os.path.join(config.PKG_DIR, "data", "github_claude_models.csv")
COMMITS_PATH = os.path.join(config.PKG_DIR, "data", "github_claude_commits.csv")
CACHE_BASKET = {"cache_read_usd_mtok": 0.85, "input_usd_mtok": 0.15, "output_usd_mtok": 0.015}
FAMILY_NAMES = ("Opus", "Sonnet", "Haiku", "Fable")
MIN_NAMED_COMMITS = 10_000      # a sample week is "measured" with at least this many commits naming a model
PUBLISH_DELAY_DAYS = 1          # a Monday-Sunday sample week is searchable on the Monday after
FALLBACK = (
    "Before the first measured sample week the index is held flat at its first measured value. "
    "Reason: from Jan 2025 to Nov 23 2025 no family's list price changed (Sonnet 3.7/4/4.5 $15 "
    "output, Opus 4/4.1 $75, Haiku 4.5 new at $5), so the price per token could only move through "
    "the model mix, which is not observed before model names enter the trailer. The one price event "
    "near the start, the Opus 4.5 cut (-67%, 2025-11-24), came with a shift of usage from Sonnet to "
    "Opus, so even its sign is unknown: with 10-20% Opus before the cut and the measured mix after "
    "it, the price per token moved between about -12% and +13%. A flat index adds no information; "
    "the indicators' random-walk intercepts absorb the level, as they did before prices were modeled.")


def _model_slug(name: str) -> str:
    """'Opus 4.5' -> 'claude-opus-4.5' (the model column of model_prices.csv)."""
    fam, ver = name.split()
    return f"claude-{fam.lower()}-{ver}"


COLLECTED_BEFORE = None          # optional UTC timestamp: ignore per-model rows collected later (pins a study's data)


def load_commit_queries(models_path: str | None = None, family_path: str | None = None,
                        totals_path: str | None = None, collected_before: str | None = None) -> pd.DataFrame:
    """All weekly commit-search counts in one long table (week_start, query, total_count, source).

    The same (week, query) can appear in several files collected on different days; GitHub's
    counts move by ~10% between collections. The per-model file (newest collection) wins, then
    the family file, then the totals file (both collected 2026-11-11). collected_before (default
    COLLECTED_BEFORE) drops per-model rows collected at or after that UTC time, so a study can be
    rerun on the data it used while the backfill continues.
    """
    cutoff = collected_before or COLLECTED_BEFORE
    parts = []
    for rank, path in enumerate((models_path or MODELS_PATH, family_path or FAMILY_MIX_PATH,
                                 totals_path or COMMITS_PATH)):
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path, parse_dates=["week_start"])
        d = d[pd.to_numeric(d.total_count, errors="coerce").notna()]
        if cutoff is not None and "collected_at" in d:
            t = pd.Timestamp(cutoff)
            t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
            d = d[pd.to_datetime(d.collected_at, utc=True) < t]
        parts.append(d[["week_start", "query", "total_count"]].assign(rank=rank, source=os.path.basename(path)))
    if not parts:
        return pd.DataFrame(columns=["week_start", "query", "total_count", "source"])
    q = pd.concat(parts, ignore_index=True)
    q["total_count"] = q.total_count.astype(float)
    q = q.sort_values("rank").drop_duplicates(["week_start", "query"], keep="first")
    return q.drop(columns="rank").sort_values(["week_start", "query"]).reset_index(drop=True)


def _released_models(week_end, prices: pd.DataFrame) -> list:
    """Model names ('Opus 4.5') released on or before week_end, from model_prices.csv."""
    r = prices[prices.release_date <= pd.Timestamp(week_end)]
    return [f"{f.capitalize()} {m.split('-')[-1]}" for f, m in zip(r.family, r.model)]


def model_counts(q: pd.DataFrame | None = None, prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Commit counts per model and sample week: family count x within-family split.

    Counts from GitHub's commit search are approximate: two runs of the same query differ by
    ~10%. Differences of overlapping counts are therefore useless (the phrase "Claude Opus 5"
    also matches "Claude Opus 5.5"), and so is a family count minus its models. Instead:
      exact count E_m    = count("Claude <Model> noreply") when <Model> is a prefix of a model
                           released by the week's end, else count("Claude <Model>")
      family count F_f   = sum of E_m when two or more models were queried (the exact queries of a
                           week were run together; the family counts of the 4-weekly windows were
                           run on another day, and the same query moves by 10-50% between days),
                           else count("Claude <Family>")
      model count        = F_f x E_m / sum of E over the family's queried models
    A family with no exact model counts goes entirely to its newest released model (most usage
    sits on the newest model; queried weeks confirm it within weeks of each release). Columns:
    one per model, 'named' (sum of family counts), 'total' (all "Co-Authored-By: Claude"
    commits), 'coverage' = named / total, and 'measured' (named >= MIN_NAMED_COMMITS).
    """
    q = load_commit_queries() if q is None else q
    prices = load_prices() if prices is None else prices
    rows = {}
    for w, g in q.groupby("week_start"):
        cnt = dict(zip(g["query"], g.total_count))
        rel = _released_models(w + pd.Timedelta(days=6), prices)
        row = {}
        for fam in FAMILY_NAMES:
            fam_models = [m for m in rel if m.startswith(fam + " ")]
            exact = {}
            for m in fam_models:
                kids = [c for c in fam_models if c.startswith(m + ".")]
                if f"Claude {m} noreply" in cnt:
                    exact[m] = cnt[f"Claude {m} noreply"]
                elif f"Claude {m}" in cnt and not kids:
                    exact[m] = cnt[f"Claude {m}"]
            if len(exact) >= 2:
                F = sum(exact.values())
            else:
                F = cnt.get(f"Claude {fam}", sum(exact.values()) if exact else None)
            if F is None or not fam_models:
                continue
            tot = sum(exact.values())
            if tot > 0:
                for m, e in exact.items():
                    row[m] = F * e / tot
            else:
                newest = max(fam_models, key=lambda m: prices.loc[prices.model == _model_slug(m), "release_date"].max())
                row[newest] = F
        if not row:
            continue
        row["total"] = cnt.get("Claude", np.nan)
        rows[w] = row
    out = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    out.index.name = "week_start"
    models = [c for c in out.columns if c != "total"]
    out[models] = out[models].fillna(0.0)
    out["named"] = out[models].sum(axis=1)
    out["coverage"] = out.named / out.total
    out["measured"] = out.named >= MIN_NAMED_COMMITS
    return out


def model_shares(counts: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of each model among named commits, measured sample weeks only."""
    c = model_counts() if counts is None else counts
    c = c[c.measured]
    models = [m for m in c.columns if m not in ("total", "named", "coverage", "measured")]
    return c[models].div(c.named, axis=0)


def model_price(name: str, basket: str = "list", prices: pd.DataFrame | None = None) -> float:
    prices = load_prices() if prices is None else prices
    r = prices[prices.model == _model_slug(name)]
    if not len(r):
        raise KeyError(name)
    r = r.iloc[0]
    if basket == "list":
        return float(r.output_usd_mtok)
    if basket == "cache":
        return float(sum(w * r[k] for k, w in CACHE_BASKET.items()))
    raise ValueError(basket)


def model_price_level(date, basket: str = "list", shares: pd.DataFrame | None = None,
                      prices: pd.DataFrame | None = None) -> float:
    """Sum_m s_m p_m on `date`, with the shares of the latest sample week public by then.

    Returns NaN before the first measured week is public (the caller applies the fallback).
    Models released after that sample week take over their family's share (successor rule).
    """
    prices = load_prices() if prices is None else prices
    shares = model_shares() if shares is None else shares
    t = pd.Timestamp(date)
    pub = shares.index + pd.Timedelta(days=6 + PUBLISH_DELAY_DAYS)
    avail = shares[pub <= t]
    if not len(avail):
        return float("nan")
    w = avail.iloc[-1]
    w = w[w > 0]
    rel = prices[prices.release_date <= t]
    level = 0.0
    for fam in FAMILY_NAMES:
        fw = w[[m for m in w.index if m.startswith(fam + " ")]]
        if not len(fw) or fw.sum() <= 0:
            continue
        sample_end = avail.index[-1] + pd.Timedelta(days=6)
        newer = rel[(rel.family == fam.lower()) & (rel.release_date > sample_end)]
        if len(newer):                       # successor rule: the newest release takes the family's share
            newest = newer.sort_values("release_date").iloc[-1]
            name = f"{fam} {newest.model.split('-')[-1]}"
            level += fw.sum() * model_price(name, basket, prices)
        else:
            level += sum(s * model_price(m, basket, prices) for m, s in fw.items())
    return level


def model_price_index(dates, basket: str = "list", shares: pd.DataFrame | None = None,
                      prices: pd.DataFrame | None = None) -> pd.Series:
    """Index relative to the first measured sample week; flat at 1 before it is public (FALLBACK)."""
    prices = load_prices() if prices is None else prices
    shares = model_shares() if shares is None else shares
    idx = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if not len(shares):
        return pd.Series(1.0, index=idx)
    t0 = shares.index[0] + pd.Timedelta(days=6 + PUBLISH_DELAY_DAYS)
    base = model_price_level(t0, basket, shares, prices)
    vals = [model_price_level(d, basket, shares, prices) / base if d >= t0 else 1.0 for d in idx]
    return pd.Series(vals, index=idx)


SPEND_INDICATORS = ("ramp_token_spend_anthropic",)     # already in dollars: never price-adjusted


def adjust_usage_indicators(df: pd.DataFrame, gamma: float, scheme: str = "model",
                            skip=SPEND_INDICATORS, index: pd.Series | None = None) -> pd.DataFrame:
    """log(indicator) + gamma * log P(ref_date) for usage indicators; claims and spend series untouched."""
    m = (df.metric == "indicator") & ~df.indicator.isin(list(skip))
    if gamma == 0 or not m.any():
        return df
    d = df.copy()
    dates = pd.to_datetime(d.loc[m, "ref_date"])
    if index is None:
        uniq = sorted(set(dates))
        if scheme in ("model", "model_cache"):
            index = model_price_index(uniq, "list" if scheme == "model" else "cache")
        else:
            index = price_index(uniq, scheme)
    d.loc[m, "value"] = d.loc[m, "value"].to_numpy() * index.reindex(dates).to_numpy() ** gamma
    return d


# =========================================================================== gamma by evidence
GAMMAS = (0.0, 0.5, 1.0)


def claim_loglik(res) -> float:
    """Online-filter log predictive density of the claims (every observation except indicators)."""
    return float(sum(i.loglik for i in res.innovations if i.metric != "indicator"))


def gamma_weights_from_loglik(ll: dict, prior: dict | None = None) -> pd.Series:
    """Posterior weights over the gamma grid: prior x exp(log-likelihood), normalized."""
    g = sorted(ll)
    prior = prior or {k: 1.0 / len(g) for k in g}
    lw = np.array([ll[k] + math.log(prior[k]) for k in g])
    w = np.exp(lw - lw.max())
    return pd.Series(w / w.sum(), index=g)


def gamma_grid(df: pd.DataFrame, params: dict, vintage, gammas=GAMMAS, scheme: str = "model",
               exclude_origin=None, prior: dict | None = None) -> pd.DataFrame:
    """Evidence for each gamma at a vintage, the same way robust.spec_grid weighs specs.

    loglik_claims: online-filter log predictive density of the claims public at the vintage
                   (filings are never in the evidence), given the price-adjusted indicators.
                   This is what decides the weights.
    loglik_joint:  the same plus the indicators' own prediction errors (sensitivity). It
                   penalizes the step a price cut puts into the adjusted indicator, so it leans
                   toward gamma = 0 whether or not revenue followed the price.
    prior: flat over the grid unless given.
    """
    from . import data
    from .model import Model
    rows = []
    for g in gammas:
        sel = data.select(adjust_usage_indicators(df, g, scheme), vintage=vintage)
        if exclude_origin is not None:
            sel = sel[sel.origin_id != exclude_origin]
        res = Model(params).run(sel, smooth=False)
        rows.append({"gamma": g, "loglik_claims": claim_loglik(res), "loglik_joint": res.loglik})
    out = pd.DataFrame(rows)
    out["weight"] = gamma_weights_from_loglik(dict(zip(out.gamma, out.loglik_claims)), prior).to_numpy()
    out["weight_joint"] = gamma_weights_from_loglik(dict(zip(out.gamma, out.loglik_joint)), prior).to_numpy()
    return out
