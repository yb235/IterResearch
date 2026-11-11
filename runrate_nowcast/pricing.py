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
"""
from __future__ import annotations

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
