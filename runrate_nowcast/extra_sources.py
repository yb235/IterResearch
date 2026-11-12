"""Extra sources for the feature store: the full Ramp AI Index history.

ramp.com/data/ai-index embeds its whole monthly history in the page (a Next.js RSC payload,
`self.__next_f.push([1, "..."])`). `parse_ramp_history` decodes it and keeps the national
series:

  adoption_rate_pct:overall       % of US businesses on Ramp with any AI spend in the month
  adoption_rate_pct:<vendor>      % paying the vendor (Anthropic, OpenAI, xAI, Mistral AI, DeepSeek)
  median_pepm                     AI spend per employee per month (USD), median firm paying for AI
  top_10_percent_median_pepm      the same, median within the top 10% of those firms
  top_1_percent_median_pepm       the same, median within the top 1%
  ai_share_of_tpv                 AI share of business spend excluding payroll (fraction)

Not on the page (every value "$undefined", nationally and in all 1,600 breakdown rows when
fetched on 2026-11-12): p99_winsorized_weighted_pepm, raw_weighted_pepm, p90/p99_pepm.
Industry, size and state breakdowns are not kept.

Publication: Ramp does not date each month. `published` = month end + 20 days, the project's
rule for Ramp (features.RAMP_PUBLISH_DELAY). It is conservative: the 2026-11-12 fetch already
held October 2026 (page dateModified 2026-11-11T16:32:43Z, 11 days after month end), so the
October row's rule date (2026-11-20) falls after the fetch.

Revision risk: Ramp recomputes the whole history at each monthly release. Against the
2026-11-11 fetch stored in indicators.csv, vendor adoption moved by up to 0.10 pp, and by more
than 0.05 pp in 29 of 39 months (Anthropic). Every row here is the value as shown on the fetch
date, not as first published, so the history carries hindsight from revisions. A later fetch
should keep these rows and add only changed months, with published = that fetch date.

One request, User-Agent "runrate-nowcast research script", no personal details.
Run: python -m runrate_nowcast.extra_sources [--html saved_page.html]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import urllib.request

import pandas as pd

from . import config

RAMP_URL = "https://ramp.com/data/ai-index"
UA = "runrate-nowcast research script"
RAMP_HISTORY_PATH = os.path.join(config.PKG_DIR, "data", "ramp_ai_index_history.csv")
RAMP_PUBLISH_DELAY_DAYS = 20
COLS = ["date_month", "series", "value", "unit", "published", "fetched", "source", "note"]
PEPM_KEYS = ("median_pepm", "top_10_percent_median_pepm", "top_1_percent_median_pepm",
             "p99_winsorized_weighted_pepm", "raw_weighted_pepm", "p90_pepm", "p99_pepm")


def fetch_ramp_page() -> str:
    req = urllib.request.Request(RAMP_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "ignore")


def _rsc_rows(html: str):
    """Yield every JSON row of the page's RSC payload (rows are 'id:json' lines)."""
    chunks = re.findall(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)', html, flags=re.S)
    payload = "".join(json.loads(c) for c in chunks)
    for line in payload.split("\n"):
        m = re.match(r"^[0-9a-f]+:([\[{].*)$", line, flags=re.S)
        if not m:
            continue
        try:
            yield json.loads(m.group(1))
        except ValueError:
            continue


def _find(x, key):
    if isinstance(x, dict):
        if key in x:
            return x
        x = list(x.values())
    if isinstance(x, list):
        for v in x:
            r = _find(v, key)
            if r is not None:
                return r
    return None


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def parse_ramp_history(html: str, fetched: str) -> pd.DataFrame:
    """National monthly series from the page, one row per (month, series)."""
    props = next((p for p in (_find(r, "adoptionOverall") for r in _rsc_rows(html)) if p is not None), None)
    if props is None:
        raise ValueError("no adoptionOverall block in the page")
    m = re.search(r'"dateModified":"([^"]+)"', html.replace('\\"', '"'))
    modified = m.group(1) if m else "unknown"
    rows = []

    def add(month, series, value, unit):
        if value is not None:
            rows.append({"date_month": month, "series": series, "value": value, "unit": unit})

    for r in props["adoptionOverall"]:
        add(r["date_month"], "adoption_rate_pct:overall", _num(r.get("adoption_rate_pct")), "pct of businesses")
    for r in props["adoptionVendor"]:
        add(r["date_month"], f"adoption_rate_pct:{r['vendor']}", _num(r.get("adoption_rate_pct")), "pct of businesses")
    for r in props["spendPerEmployee"]:
        for k in PEPM_KEYS:
            add(r["date_month"], k, _num(r.get(k)), "USD per employee per month")
    for r in props["spendShareData"]["overall"]:
        add(r["my_date"], "ai_share_of_tpv", _num(r.get("ai_share_of_tpv")), "fraction of non-payroll spend")

    d = pd.DataFrame(rows)
    month = pd.to_datetime(d["date_month"], format="%Y-%m-%d")
    d["date_month"] = month.dt.strftime("%Y-%m-%d")
    d["published"] = (month + pd.offsets.MonthEnd(0)
                      + pd.Timedelta(days=RAMP_PUBLISH_DELAY_DAYS)).dt.strftime("%Y-%m-%d")
    d["fetched"] = fetched
    d["source"] = RAMP_URL
    d["note"] = (f"as shown on {fetched} (page dateModified {modified}); published assumed month end + "
                 f"{RAMP_PUBLISH_DELAY_DAYS} days; Ramp revises the whole history each release, so earlier "
                 "months are revised values, not as first published")
    d = d.drop_duplicates(["date_month", "series"], keep="last")
    return d.sort_values(["series", "date_month"])[COLS].reset_index(drop=True)


def update_ramp_history(html_path: str | None = None, fetched: str | None = None,
                        path: str = RAMP_HISTORY_PATH) -> pd.DataFrame:
    """Fetch the page once (or read a saved copy) and write data/ramp_ai_index_history.csv."""
    if html_path:
        with open(html_path, encoding="utf-8", errors="ignore") as f:
            html = f.read()
    else:
        html = fetch_ramp_page()
    d = parse_ramp_history(html, fetched or dt.date.today().isoformat())
    d.to_csv(path, index=False)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--html", help="saved copy of the page (default: fetch it)")
    ap.add_argument("--fetched", help="fetch date of the saved copy (YYYY-MM-DD)")
    a = ap.parse_args()
    out = update_ramp_history(a.html, a.fetched)
    print(out.groupby("series")["date_month"].agg(["count", "min", "max"]).to_string())
