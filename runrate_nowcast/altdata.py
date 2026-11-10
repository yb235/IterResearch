"""Leading indicators: catalog, fetchers, and conversion to weekly observations.

Data files
  data/indicator_catalog.csv  one row per indicator: which segment it tracks, elasticity
                              prior (beta), noise, intercept drift, lag, role, access
  data/indicators.csv         long format (date, indicator, value, published, source, note),
                              append-only, one row per indicator per day or snapshot

Roles
  model    enters the filter as an observation
  monitor  collected and charted but not in the filter (too short, or not yet validated)
  context  a competitor or market series, used for normalisation and sanity checks
  planned  not collected yet (needs a paid source, a manual step, or a host allowlist)

Known breaks (catalog column `breaks`, semicolon-separated dates) are changes in how a
source counts, e.g. pypistats on 2026-08-25, when the anthropic and openai packages both
stepped down ~35% on the same day. On a break date the indicator's intercept gets a
large variance bump, so the step is absorbed instead of read as a revenue move.

How an indicator enters the filter
  log(I_week) = a_i + beta_i * (l + s_segment)(week centre - lag) + noise
a_i is a random walk, so the level relationship between the indicator and revenue is
re-learned continuously. The indicator mostly informs changes in revenue between
audited figures; it never sets the level.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import time
import urllib.request

import pandas as pd

from . import config

UA = "runrate-nowcast/0.2 (research script)"
HOLIDAY_START, HOLIDAY_END = 1220, 106      # MMDD: Dec 20 .. Jan 6
CATALOG_COLS = ["indicator", "segment", "role", "beta", "beta_sd", "obs_sd", "drift", "prior_sd",
                "lag_days", "aggregate", "publish_delay_days", "frequency", "access", "fetcher",
                "source_url", "description", "breaks"]
RAW_COLS = ["date", "indicator", "value", "published", "source", "note"]


# --------------------------------------------------------------------------- catalog / raw
def catalog(path: str = config.CATALOG_PATH) -> pd.DataFrame:
    c = pd.read_csv(path)
    missing = set(CATALOG_COLS) - set(c.columns)
    if missing:
        raise ValueError(f"indicator_catalog.csv missing {sorted(missing)}")
    return c


def indicator_params(cat: pd.DataFrame | None = None, overrides: dict | None = None) -> dict:
    """Specs for Model(params['indicators']) from the catalog, with fitted overrides applied."""
    cat = catalog() if cat is None else cat
    out = {}
    for r in cat.itertuples():
        spec = {"segment": r.segment, "beta": float(r.beta), "beta_sd": float(r.beta_sd),
                "obs_sd": float(r.obs_sd), "drift": float(r.drift), "prior_sd": float(r.prior_sd),
                "lag_days": float(r.lag_days), "role": r.role,
                "breaks": [d for d in str(r.breaks).split(";") if d and d != "nan"]}
        ov = (overrides or {}).get(r.indicator)
        if ov:
            spec.update(ov)
            spec["fitted"] = True
        out[r.indicator] = spec
    return out


def load_raw(path: str = config.INDICATORS_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=RAW_COLS)
    d = pd.read_csv(path, parse_dates=["date", "published"])
    return d


def save_raw(new: pd.DataFrame, path: str = config.INDICATORS_PATH) -> int:
    """Merge rows into indicators.csv (newest value wins for the same date and indicator)."""
    old = load_raw(path)
    allr = pd.concat([old, new[RAW_COLS]], ignore_index=True)
    allr["date"] = pd.to_datetime(allr["date"])
    allr = allr.drop_duplicates(subset=["date", "indicator"], keep="last").sort_values(["indicator", "date"])
    allr.to_csv(path, index=False, date_format="%Y-%m-%d")
    return len(allr) - len(old)


# --------------------------------------------------------------------------- observations
def indicator_observations(roles=("model",), cat: pd.DataFrame | None = None,
                           raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """Weekly observations in the observations.csv schema, for indicators with the given roles."""
    cat = catalog() if cat is None else cat
    raw = load_raw() if raw is None else raw
    rows = []
    for spec in cat[cat.role.isin(roles)].itertuples():
        s = raw[(raw.indicator == spec.indicator) & (raw.value > 0)].copy()
        if not len(s):
            continue
        lag = pd.Timedelta(days=float(spec.lag_days))
        delay = pd.Timedelta(days=float(spec.publish_delay_days))
        if spec.aggregate == "weekly_mean":
            s = s.set_index("date").sort_index()
            # Same calendar rule as break detection: drop Dec 20 - Jan 6, whose dip and
            # rebound hit every series and say nothing about Anthropic's revenue.
            md = s.index.month * 100 + s.index.day
            s = s[~((md >= HOLIDAY_START) | (md <= HOLIDAY_END))]
            g = s["value"].resample("W-SUN")
            wk = pd.DataFrame({"value": g.mean(), "n": g.count()})
            wk = wk[wk.n >= 5]
            for week_end, r in wk.iterrows():
                centre = week_end - pd.Timedelta(days=3)
                rows.append(_obs_row(spec, centre - lag, r.value, week_end + delay, f"week ending {week_end:%Y-%m-%d}"))
        else:   # 'none': each raw point is its own observation (snapshots)
            for r in s.itertuples():
                pub = r.published if pd.notna(r.published) else r.date + delay
                centre = r.date - pd.Timedelta(days=3) if spec.frequency == "weekly" else r.date
                rows.append(_obs_row(spec, centre - lag, r.value, pub, str(r.note)))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["published"] = pd.to_datetime(out["published"])
    out["ref_date"] = pd.to_datetime(out["ref_date"])
    return out


def _obs_row(spec, ref_date, value, published, note):
    return {"obs_id": f"{spec.indicator}@{pd.Timestamp(ref_date):%Y-%m-%d}",
            "origin_id": f"{spec.indicator}@{pd.Timestamp(ref_date):%Y-%m-%d}",
            "metric": "indicator", "kind": "point", "value": float(value), "ref_date": ref_date,
            "period_start": pd.NaT, "period_end": pd.NaT, "definition": "instant",
            "date_sd_days": 0.0, "published": published, "tier": "altdata",
            "source": spec.fetcher, "url": spec.source_url, "include": 1, "notes": note,
            "segment": "total", "indicator": spec.indicator, "is_forecast": 0}


# --------------------------------------------------------------------------- fetchers
def _get(url: str, retries: int = 3) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(2 ** (i + 1))


def fetch_npm(package: str, indicator: str, start="2025-01-01", end=None) -> pd.DataFrame:
    """Daily downloads from api.npmjs.org (max 18 months per request, so chunked)."""
    end = pd.Timestamp(end or dt.date.today() - dt.timedelta(days=1))
    rows, s = [], pd.Timestamp(start)
    while s <= end:
        e = min(s + pd.DateOffset(months=17), end)
        url = f"https://api.npmjs.org/downloads/range/{s:%Y-%m-%d}:{e:%Y-%m-%d}/{package}"
        d = json.loads(_get(url))
        for r in d.get("downloads", []):
            rows.append({"date": r["day"], "indicator": indicator, "value": r["downloads"],
                         "published": pd.NaT, "source": f"npm:{package}", "note": ""})
        s = e + pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def fetch_pypistats(package: str, indicator: str) -> pd.DataFrame:
    """Daily downloads without mirrors from pypistats.org (last ~180 days only)."""
    d = json.loads(_get(f"https://pypistats.org/api/packages/{package}/overall?mirrors=false"))
    rows = [{"date": r["date"], "indicator": indicator, "value": r["downloads"], "published": pd.NaT,
             "source": f"pypistats:{package}", "note": ""}
            for r in d["data"] if r["category"] == "without_mirrors"]
    return pd.DataFrame(rows)


def fetch_ramp_snapshot() -> pd.DataFrame:
    """Parse the current and previous period from ramp.com/data/ai-index.

    Token-spend index = Anthropic share of token spend x total token spend, with the
    total for the earlier week set to 1. The page reports the total's change against
    the same week a month earlier.
    """
    import html as H
    h = _get("https://ramp.com/data/ai-index").decode("utf-8", "ignore")
    tables = []
    for t in re.findall(r"<table.*?</table>", h, flags=re.S):
        rows = []
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", t, flags=re.S):
            cells = [H.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, flags=re.S)]
            rows.append(cells)
        tables.append(rows)
    today = pd.Timestamp(dt.date.today())
    out = []

    def pct(x):
        return float(x.replace("%", "").replace(",", "")) / 100

    for rows in tables:
        if not rows or len(rows[0]) < 3:
            continue
        head = rows[0]
        d1, d0 = _ramp_date(head[1]), _ramp_date(head[2])
        vals = {r[0]: r for r in rows[1:]}
        if "AI adoption: Anthropic" in vals:
            r = vals["AI adoption: Anthropic"]
            for d, v in ((d1, r[1]), (d0, r[2])):
                out.append({"date": d, "indicator": "ramp_adoption_anthropic", "value": pct(v),
                            "published": today, "source": "ramp.com/data/ai-index", "note": head[0]})
        if "Share of token spend: Anthropic" in vals:
            share1, share0 = pct(vals["Share of token spend: Anthropic"][1]), pct(vals["Share of token spend: Anthropic"][2])
            m = re.search(r"Total token spending (rose|fell) ([0-9.]+)%", H.unescape(re.sub(r"<[^>]+>", "", h)))
            growth = (1 if m and m.group(1) == "rose" else -1) * float(m.group(2)) / 100 if m else float("nan")
            for d, v in ((d0, share0 * 1.0), (d1, share1 * (1 + growth))):
                out.append({"date": d, "indicator": "ramp_token_spend_anthropic", "value": v,
                            "published": today, "source": "ramp.com/data/ai-index",
                            "note": f"share x total-spend index (total {growth:+.1%} m/m)"})
    return pd.DataFrame(out)


def _ramp_date(label: str) -> pd.Timestamp:
    m = re.search(r"Week ending (\w+ \d+, \d{4})", label)
    if m:
        return pd.Timestamp(m.group(1))
    return pd.Timestamp(label) + pd.offsets.MonthEnd(0)    # "Sep 2026" -> month end


def fetch_github_commits(query: str, indicator: str, dates, pause: float = 8.0) -> pd.DataFrame:
    """Daily commit counts matching `query` from GitHub commit search, via the gh CLI.

    The search API rate-limits hard and reports approximate counts on large result
    sets (incomplete_results). Sample sparsely (e.g. one day a week) and pause between
    calls.
    """
    import subprocess
    rows = []
    for d in dates:
        q = f"{query} committer-date:{pd.Timestamp(d):%Y-%m-%d}"
        outp = subprocess.run(["gh", "api", "-X", "GET", "search/commits", "-f", f"q={q}", "-f", "per_page=1",
                               "--jq", ".total_count"], capture_output=True, text=True)
        if outp.returncode == 0 and outp.stdout.strip().isdigit():
            rows.append({"date": pd.Timestamp(d), "indicator": indicator, "value": int(outp.stdout.strip()),
                         "published": pd.NaT, "source": "github search/commits", "note": query})
        time.sleep(pause)
    return pd.DataFrame(rows)


FETCHERS = {
    "npm": fetch_npm,
    "pypistats": fetch_pypistats,
}


def update(start="2025-01-01", include_snapshots=True, verbose=True) -> dict:
    """Fetch every automated indicator in the catalog and merge into indicators.csv."""
    cat = catalog()
    added = {}
    for r in cat.itertuples():
        kind, _, target = str(r.fetcher).partition(":")
        try:
            if kind == "npm":
                df = fetch_npm(target, r.indicator, start)
            elif kind == "pypistats":
                df = fetch_pypistats(target, r.indicator)
            else:
                continue
            added[r.indicator] = save_raw(df)
        except Exception as e:          # keep going; report at the end
            added[r.indicator] = f"error: {e}"
    if include_snapshots:
        try:
            added["ramp"] = save_raw(fetch_ramp_snapshot())
        except Exception as e:
            added["ramp"] = f"error: {e}"
    if verbose:
        for k, v in added.items():
            print(f"  {k}: {v}")
    return added
