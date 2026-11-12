"""Prediction markets on Anthropic's run-rate, revenue, valuation and IPO: fetchers and as-of helpers.

Purpose: a benchmark. For any past date t, what did prediction markets imply about the run-rate,
using only prices observable at or before t?

Data files
  data/prediction_markets_catalog.csv  markets found by web search: one row per market, plus one row per
                                       dated snapshot found in news/search summaries (evidence_source says which)
  data/prediction_markets_prices.csv   tidy price history written by `fetch_catalog` (only once the hosts
                                       are reachable; not created otherwise)

Tidy history schema (HIST_COLS), returned by every *_history fetcher
  timestamp    UTC pandas Timestamp at which the probability was observable (see per-platform notes)
  platform     manifold | polymarket | kalshi
  market       Manifold contract slug / Polymarket market slug / Kalshi market ticker
  outcome      answer text (Manifold multi-answer), groupItemTitle or question (Polymarket),
               yes_sub_title or title (Kalshi), "YES" for a plain binary
  threshold    number parsed from the outcome (or binary question) text, in USD billions (NaN if none)
  kind         'ge' (>= threshold), 'le' (<= threshold), 'between' (bucket [threshold, upper)), '' unknown
  upper        bucket upper edge for kind == 'between', else NaN
  probability  P(YES) in [0, 1]
  source       'bet' | 'initial' | 'resolution' | 'prices-history' | 'candle_close' | 'candle_mid'

When is a price "observable"? (no look-ahead)
  Manifold     each bet's probAfter at the bet's createdTime; 'initial' = the first bet's probBefore,
               dated at the answer's (or market's) createdTime; 'resolution' at resolutionTime.
  Polymarket   CLOB /prices-history points {t, p}: we treat p as known at t. Polymarket does not document
               whether a daily point is sampled at the start or end of the bucket; use
               prob_asof(..., min_lag=pd.Timedelta("1D")) to be conservative with fidelity=1440.
  Kalshi       daily candlesticks are stamped with end_period_ts (end of the candle), so the close is known
               at that time. Settlement rows use the latest of settlement_ts/expiration_time/close_time.

Field-name assumptions (checked against the public API docs as remembered, not against live
responses, because the hosts are blocked here; every access is defensive and missing fields are skipped):
  Manifold   GET /v0/slug/{slug} -> {id, slug, question, outcomeType, mechanism, createdTime(ms),
             closeTime, isResolved, resolution, resolutionTime, resolutionProbability, probability,
             shouldAnswersSumToOne, answers:[{id, text, probability, createdTime, resolution,
             resolutionTime, resolutionProbability}]}.
             GET /v0/bets?contractId=..&limit=1000&before={betId} -> newest-first list of
             {id, contractId, answerId?, createdTime(ms), probBefore, probAfter, isCancelled?}.
             Pagination: `before` = id of the oldest bet already received; stop on a short/empty page.
  Polymarket GET gamma /events?slug=.. -> [{slug, markets:[...]}]; GET gamma /markets?slug=.. -> [market].
             market: {slug, question, groupItemTitle, outcomes, outcomePrices, clobTokenIds, startDate,
             endDate, closed}; outcomes / clobTokenIds are usually JSON-encoded strings.
             GET clob /prices-history?market={token_id}&interval=max&fidelity=1440 -> {history:[{t(s), p}]}.
             For some closed markets interval=max returns nothing; we then retry with startTs/endTs.
  Kalshi     GET /markets/{ticker} -> {market:{ticker, event_ticker, title, subtitle, yes_sub_title,
             open_time, close_time, expiration_time, status, result}}; prices in integer cents or, in
             newer responses, *_dollars strings (both handled).
             GET /events/{event_ticker} -> {event:{series_ticker}}; fallback series = ticker prefix.
             GET /series/{s}/markets/{t}/candlesticks?start_ts&end_ts&period_interval=1440 ->
             {candlesticks:[{end_period_ts, price:{close}, yes_bid:{close}, yes_ask:{close}}]}.
             price.close is null when nothing traded in the period; we then use the bid/ask mid.

Network status (2026-11-12): from this environment api.manifold.markets, gamma-api.polymarket.com,
clob.polymarket.com, api.elections.kalshi.com, manifold.markets and metaculus.com are refused by the
egress proxy (HTTP 403 on CONNECT). The code and the offline tests are ready; run
    python -m runrate_nowcast.prediction_markets fetch
once the hosts are allowed.

Reading a ladder (implied_distribution)
  Input: thresholds x_i and P(V >= x_i). Probabilities are clipped to [0, 1] and made non-increasing in x
  by weighted isotonic regression (pool-adjacent-violators). Quantile q is where the survival curve
  crosses 1 - q, interpolating linearly in log(x) between adjacent thresholds.
  Beyond the ladder (survival at the lowest threshold already <= 1 - q, or at the highest still > 1 - q)
  the quantile is not identified: tail='nan' (default) returns NaN and flags it 'below_ladder' /
  'above_ladder' (so the answer is only a bound: "< lowest" / "> highest threshold"); tail='clamp'
  returns the end threshold (a bound, flagged 'clamped'); tail='extrapolate' extends the end segment's
  log-linear slope (flagged 'extrapolated'). Optional anchors `lower` (S = 1, e.g. a run-rate already
  disclosed before t for a reach-by-date ladder) and `upper` (S = 0) close the ladder.
  Caveats: "reach X by date D" ladders describe the maximum reported run-rate up to D, not the run-rate
  on D; Manifold probabilities are play-money and thinly traded; no risk-premium adjustment is made.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

UA = "runrate-nowcast research script"
MANIFOLD_API = "https://api.manifold.markets/v0"
POLY_GAMMA = "https://gamma-api.polymarket.com"
POLY_CLOB = "https://clob.polymarket.com"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(_PKG_DIR, "data", "prediction_markets_catalog.csv")
PRICES_PATH = os.path.join(_PKG_DIR, "data", "prediction_markets_prices.csv")

HIST_COLS = ["timestamp", "platform", "market", "outcome", "threshold", "kind", "upper",
             "probability", "source"]
GROUP_COLS = ("platform", "market", "outcome")

Getter = Callable[..., object]


# --------------------------------------------------------------------------- HTTP
class FetchError(RuntimeError):
    """A request failed (network refused, HTTP error, bad JSON). `status` is the HTTP code if any."""

    def __init__(self, url: str, status: int | None = None, reason: str = ""):
        self.url, self.status, self.reason = url, status, reason
        super().__init__(f"{url}: {status or ''} {reason}".strip())


def _with_params(url: str, params: dict | None) -> str:
    if not params:
        return url
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    return url + ("&" if "?" in url else "?") + q if q else url


def http_get_json(url: str, params: dict | None = None, timeout: float = 30.0, retries: int = 2,
                  pause: float = 1.5) -> object:
    """GET a JSON document with a plain User-Agent (no personal data). Raises FetchError."""
    full = _with_params(url, params)
    req = urllib.request.Request(full, headers={"User-Agent": UA, "Accept": "application/json"})
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            try:
                return json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                raise FetchError(full, None, f"invalid JSON: {e}") from e
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(pause * (attempt + 1))
                last = e
                continue
            raise FetchError(full, e.code, str(e.reason)) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # a proxy refusal ("Tunnel connection failed: 403") is not worth retrying
            if "403" in str(e) or attempt >= retries:
                raise FetchError(full, None, str(getattr(e, "reason", e))) from e
            time.sleep(pause * (attempt + 1))
            last = e
    raise FetchError(full, None, str(last))


# --------------------------------------------------------------------------- parsing helpers
def to_utc(x, unit: str | None = None) -> pd.Timestamp:
    """Timestamp in UTC. Ints/floats need `unit` ('ms' Manifold, 's' Polymarket/Kalshi). Naive -> UTC."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return pd.NaT
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return pd.Timestamp(int(x), unit=unit or "s", tz="UTC")
    t = pd.Timestamp(x)
    if t is pd.NaT:
        return t
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _float(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


_UNITS = {"trillion": 1000.0, "tn": 1000.0, "t": 1000.0, "billion": 1.0, "bn": 1.0, "b": 1.0,
          "million": 1e-3, "mm": 1e-3, "m": 1e-3, "k": 1e-6}
_U = r"(?:(trillion|billion|million|tn|bn|mm|t|b|m|k)\b)"
_N = r"(\d+(?:,\d{3})*(?:\.\d+)?)"
_RANGE = re.compile(r"(\$)?\s*" + _N + r"\s*" + _U + r"?\s*(?:-|to)\s*(\$)?\s*" + _N + r"\s*" + _U + r"?")
_AMOUNT = re.compile(r"(\$)?\s*" + _N + r"\s*" + _U + r"?")
_LE_WORDS = ("<", "under", "below", "less than", "at most", "fewer than", "lower than")


def _amount(dollar: str | None, num: str, unit: str | None, default_unit: str | None):
    """Value in billions, or None when the number has neither '$' nor a unit (e.g. a year)."""
    if not dollar and not unit:
        return None
    u = unit or default_unit
    return float(num.replace(",", "")) * _UNITS.get(u or "b", 1.0)


def parse_threshold(text: str | None) -> tuple[str, float, float]:
    """(kind, value, upper) in USD billions from an outcome label.

    '>= $80B' -> ('ge', 80, nan); 'Under $90bn' -> ('le', 90, nan); '$1.5T' -> ('ge', 1500, nan);
    '$50-60B' / '$50B to $60B' -> ('between', 50, 60); no amount -> ('', nan, nan).
    A number counts only with a '$' or a unit (so years are ignored); a bare number in a range inherits
    the other end's unit; a lone '$' amount without unit is read as billions. A single amount without
    '<'/'under'/'below'-type words is read as '>=' (threshold ladders are 'reach X' questions).
    """
    if not isinstance(text, str) or not text.strip():
        return "", math.nan, math.nan
    s = (text.lower().replace("≥", ">=").replace("≤", "<=").replace("–", "-").replace("—", "-")
         .replace("↑", ">=").replace("↓", "<="))
    for m in _RANGE.finditer(s):
        d1, n1, u1, d2, n2, u2 = m.groups()
        if (d1 or u1) or (d2 or u2):          # skip dates such as 2026-12
            lo = _amount(d1 or "$", n1, u1, u2)
            hi = _amount(d2 or "$", n2, u2, u1)
            if lo is not None and hi is not None and hi > lo:
                return "between", lo, hi
    vals = [v for v in (_amount(d, n, u, None) for d, n, u in _AMOUNT.findall(s)) if v is not None]
    if not vals:
        return "", math.nan, math.nan
    kind = "le" if any(w in s for w in _LE_WORDS) else "ge"
    return kind, vals[0], math.nan


def _tidy(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in HIST_COLS}).astype(
            {"probability": float, "threshold": float, "upper": float})
    df = pd.DataFrame(rows)
    for c in HIST_COLS:
        if c not in df:
            df[c] = math.nan if c in ("threshold", "upper") else ""
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["probability"] = pd.to_numeric(df["probability"], errors="coerce").clip(0.0, 1.0)
    df = df.dropna(subset=["timestamp", "probability"])
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df[HIST_COLS]


def _label_rows(rows: list[dict], label_for_threshold: dict[str, str]) -> list[dict]:
    """Add threshold/kind/upper parsed from the text mapped to each outcome."""
    cache: dict[str, tuple] = {}
    for r in rows:
        txt = label_for_threshold.get(r["outcome"], r["outcome"])
        if txt not in cache:
            cache[txt] = parse_threshold(txt)
        r["kind"], r["threshold"], r["upper"] = cache[txt]
    return rows


# --------------------------------------------------------------------------- Manifold
def manifold_slug(s: str) -> str:
    """'user/slug', a full URL, or a bare slug -> bare contract slug."""
    s = s.strip()
    if "manifold.markets/" in s:
        s = s.split("manifold.markets/", 1)[1]
    s = s.split("?", 1)[0].split("#", 1)[0].strip("/")
    return s.rsplit("/", 1)[-1]


def manifold_market(slug: str, get: Getter = http_get_json) -> dict:
    m = get(f"{MANIFOLD_API}/slug/{manifold_slug(slug)}")
    if not isinstance(m, dict) or "id" not in m:
        raise FetchError(f"{MANIFOLD_API}/slug/{slug}", None, "unexpected market payload")
    return m


def manifold_bets(contract_id: str, get: Getter = http_get_json, limit: int = 1000,
                  max_pages: int = 1000) -> list[dict]:
    """All bets on a contract (any order). Paginates with `before` = oldest bet id seen so far."""
    out: list[dict] = []
    seen: set = set()
    before = None
    for _ in range(max_pages):
        page = get(f"{MANIFOLD_API}/bets", {"contractId": contract_id, "limit": limit, "before": before})
        if not isinstance(page, list):
            raise FetchError(f"{MANIFOLD_API}/bets", None, "unexpected bets payload")
        new = [b for b in page if isinstance(b, dict) and b.get("id") not in seen]
        if not new:
            break
        out.extend(new)
        seen.update(b.get("id") for b in new)
        if len(page) < limit:
            break
        oldest = min(new, key=lambda b: _float(b.get("createdTime")))
        before = oldest.get("id")
    else:
        warnings.warn(f"manifold_bets: stopped after {max_pages} pages for {contract_id}; history truncated")
    return out


def _manifold_resolution_value(res, res_prob) -> float:
    if res == "YES":
        return 1.0
    if res == "NO":
        return 0.0
    if res == "MKT":
        p = _float(res_prob)
        return p / 100.0 if p > 1.0 else p
    return math.nan        # CANCEL / unknown: no information


def manifold_history(slug: str, get: Getter = http_get_json, include_resolution: bool = True,
                     market: dict | None = None, bets: list[dict] | None = None) -> pd.DataFrame:
    """Tidy probability history of a Manifold binary or multiple-choice / multi-answer market.

    Binary: one outcome 'YES'. Multi-answer: one outcome per answer text; bets carry answerId.
    In independent ladders (shouldAnswersSumToOne false) each answer is its own pool, so its bets give
    its full path. In sum-to-one markets a bet on one answer also moves the others; Manifold records
    those moves as extra bets on the other answers (same betGroupId) - if a market predates that, the
    other answers' paths miss indirect moves (compare with each answer's current `probability`).
    PSEUDO_NUMERIC markets are returned on Manifold's internal 0-1 scale (not converted).
    """
    m = market if market is not None else manifold_market(slug, get)
    if bets is None:
        bets = manifold_bets(m["id"], get)
    key = m.get("slug") or manifold_slug(slug)
    answers = {a.get("id"): a for a in (m.get("answers") or []) if isinstance(a, dict)}
    if m.get("outcomeType") == "PSEUDO_NUMERIC":
        warnings.warn(f"{key}: PSEUDO_NUMERIC probabilities are on the internal 0-1 scale")

    def outcome_of(b):
        if not answers:
            return "YES"
        a = answers.get(b.get("answerId"))
        return a.get("text", str(b.get("answerId"))) if a else None

    rows: list[dict] = []
    first: dict[str, dict] = {}
    for b in sorted((b for b in bets if isinstance(b, dict)), key=lambda b: _float(b.get("createdTime"))):
        if b.get("isCancelled"):
            continue
        out = outcome_of(b)
        t, p = b.get("createdTime"), _float(b.get("probAfter"))
        if out is None or t is None or math.isnan(p):
            continue
        rows.append({"timestamp": to_utc(t, "ms"), "platform": "manifold", "market": key,
                     "outcome": out, "probability": p, "source": "bet"})
        first.setdefault(out, b)

    # opening probability, valid from creation until the first bet
    for out, b in first.items():
        p0 = _float(b.get("probBefore"))
        created = None
        if answers:
            a = next((a for a in answers.values() if a.get("text") == out), None)
            created = a.get("createdTime") if a else None
        created = created or m.get("createdTime")
        if created is not None and not math.isnan(p0) and _float(created) <= _float(b.get("createdTime")):
            rows.append({"timestamp": to_utc(created, "ms"), "platform": "manifold", "market": key,
                         "outcome": out, "probability": p0, "source": "initial"})

    if include_resolution:
        if answers:
            for a in answers.values():
                v = _manifold_resolution_value(a.get("resolution"), a.get("resolutionProbability"))
                if not math.isnan(v) and a.get("resolutionTime"):
                    rows.append({"timestamp": to_utc(a["resolutionTime"], "ms"), "platform": "manifold",
                                 "market": key, "outcome": a.get("text"), "probability": v,
                                 "source": "resolution"})
            # older sum-to-one markets resolve at market level: resolution = winning answer id
            res = m.get("resolution")
            if m.get("isResolved") and m.get("resolutionTime") and res in answers:
                done = {r["outcome"] for r in rows if r["source"] == "resolution"}
                for aid, a in answers.items():
                    if a.get("text") not in done:
                        rows.append({"timestamp": to_utc(m["resolutionTime"], "ms"), "platform": "manifold",
                                     "market": key, "outcome": a.get("text"),
                                     "probability": 1.0 if aid == res else 0.0, "source": "resolution"})
        elif m.get("isResolved") and m.get("resolutionTime"):
            v = _manifold_resolution_value(m.get("resolution"), m.get("resolutionProbability"))
            if not math.isnan(v):
                rows.append({"timestamp": to_utc(m["resolutionTime"], "ms"), "platform": "manifold",
                             "market": key, "outcome": "YES", "probability": v, "source": "resolution"})

    labels = {} if answers else {"YES": m.get("question", "")}
    return _tidy(_label_rows(rows, labels))


# --------------------------------------------------------------------------- Polymarket
def _jsonish(v) -> list:
    """Gamma returns outcomes / outcomePrices / clobTokenIds as JSON-encoded strings (sometimes lists)."""
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return [x.strip() for x in v.strip("[]").split(",") if x.strip()]
    return list(v) if isinstance(v, (list, tuple)) else [v]


def polymarket_markets(slug: str, get: Getter = http_get_json) -> list[dict]:
    """Markets behind an event slug (tried first) or a market slug, normalised.

    Each item: slug, question, label (groupItemTitle or question), outcome_name, token_id (the 'Yes'
    token, else the first outcome's token), start, end, closed.
    """
    raw: list[dict] = []
    try:
        ev = get(f"{POLY_GAMMA}/events", {"slug": slug})
    except FetchError as e:
        if e.status not in (404, 422):
            raise
        ev = []
    for e in (ev if isinstance(ev, list) else [ev] if isinstance(ev, dict) else []):
        raw.extend(m for m in (e.get("markets") or []) if isinstance(m, dict))
    if not raw:
        mk = get(f"{POLY_GAMMA}/markets", {"slug": slug})
        raw = [m for m in (mk if isinstance(mk, list) else [mk]) if isinstance(m, dict)]
    out = []
    for m in raw:
        outcomes = [str(o) for o in _jsonish(m.get("outcomes"))]
        tokens = [str(t) for t in _jsonish(m.get("clobTokenIds"))]
        idx = next((i for i, o in enumerate(outcomes) if o.strip().lower() == "yes"), 0)
        out.append({
            "slug": m.get("slug") or slug,
            "question": m.get("question") or "",
            "label": m.get("groupItemTitle") or m.get("question") or m.get("slug") or slug,
            "outcome_name": outcomes[idx] if idx < len(outcomes) else "Yes",
            "token_id": tokens[idx] if idx < len(tokens) else None,
            "start": m.get("startDate") or m.get("createdAt"),
            "end": m.get("endDate") or m.get("closedTime"),
            "closed": bool(m.get("closed")),
        })
    return out


def polymarket_prices(token_id: str, get: Getter = http_get_json, fidelity: int = 1440,
                      start_ts: int | None = None, end_ts: int | None = None) -> list[tuple[int, float]]:
    """[(unix seconds, price)] from the CLOB. interval=max unless a start/end window is given."""
    params: dict = {"market": token_id, "fidelity": fidelity}
    if start_ts is None and end_ts is None:
        params["interval"] = "max"
    else:
        params.update(startTs=start_ts, endTs=end_ts)
    js = get(f"{POLY_CLOB}/prices-history", params)
    hist = js.get("history", []) if isinstance(js, dict) else (js if isinstance(js, list) else [])
    pts = []
    for h in hist:
        if isinstance(h, dict) and h.get("t") is not None:
            p = _float(h.get("p"))
            if not math.isnan(p):
                pts.append((int(h["t"]), p))
    return pts


def polymarket_history(slug: str, get: Getter = http_get_json, fidelity: int = 1440) -> pd.DataFrame:
    """Tidy daily price history (YES token) for every market under an event or market slug."""
    rows: list[dict] = []
    labels: dict[str, str] = {}
    for m in polymarket_markets(slug, get):
        if not m["token_id"]:
            continue
        pts = polymarket_prices(m["token_id"], get, fidelity)
        if not pts and m["start"] and m["end"]:
            s, e = to_utc(m["start"]), to_utc(m["end"])
            if s is not pd.NaT and e is not pd.NaT:
                pts = polymarket_prices(m["token_id"], get, fidelity, int(s.timestamp()), int(e.timestamp()))
        out = m["label"] if m["outcome_name"].lower() == "yes" else f"{m['label']}: {m['outcome_name']}"
        labels[out] = m["label"]
        for t, p in pts:
            rows.append({"timestamp": to_utc(t, "s"), "platform": "polymarket", "market": m["slug"],
                         "outcome": out, "probability": p, "source": "prices-history"})
    return _tidy(_label_rows(rows, labels))


# --------------------------------------------------------------------------- Kalshi
def _kprice(d: dict | None, key: str) -> float:
    """Price in dollars (0-1) from Kalshi's `{key}_dollars` (string dollars) or `{key}` (int cents)."""
    if not isinstance(d, dict):
        return math.nan
    v = _float(d.get(f"{key}_dollars"))
    if not math.isnan(v):
        return v
    v = _float(d.get(key))
    return v / 100.0 if not math.isnan(v) else math.nan


def kalshi_market(ticker: str, get: Getter = http_get_json) -> dict:
    js = get(f"{KALSHI_API}/markets/{ticker}")
    m = js.get("market", js) if isinstance(js, dict) else None
    if not isinstance(m, dict) or not m.get("ticker"):
        raise FetchError(f"{KALSHI_API}/markets/{ticker}", None, "unexpected market payload")
    return m


def kalshi_event_markets(event_ticker: str, get: Getter = http_get_json, max_pages: int = 50) -> list[dict]:
    out, cursor = [], None
    for _ in range(max_pages):
        js = get(f"{KALSHI_API}/markets", {"event_ticker": event_ticker, "limit": 200, "cursor": cursor})
        page = js.get("markets", []) if isinstance(js, dict) else []
        out.extend(m for m in page if isinstance(m, dict))
        cursor = js.get("cursor") if isinstance(js, dict) else None
        if not cursor or not page:
            break
    return out


def kalshi_series_ticker(market: dict, get: Getter = http_get_json) -> str:
    ev = market.get("event_ticker")
    if ev:
        try:
            js = get(f"{KALSHI_API}/events/{ev}")
            s = (js.get("event") or {}).get("series_ticker") if isinstance(js, dict) else None
            if s:
                return s
        except FetchError:
            pass
    return str(ev or market.get("ticker", "")).split("-")[0]


def kalshi_candles(series: str, ticker: str, start_ts: int, end_ts: int, get: Getter = http_get_json,
                   period: int = 1440, chunk_periods: int = 4000) -> list[dict]:
    """Candlesticks in chunks (the API caps the number of periods per request)."""
    out: list[dict] = []
    step = chunk_periods * period * 60
    s = start_ts
    while s < end_ts:
        e = min(end_ts, s + step)
        js = get(f"{KALSHI_API}/series/{series}/markets/{ticker}/candlesticks",
                 {"start_ts": s, "end_ts": e, "period_interval": period})
        out.extend(c for c in (js.get("candlesticks", []) if isinstance(js, dict) else []) if isinstance(c, dict))
        s = e
    return out


def kalshi_history(ticker: str, get: Getter = http_get_json, period: int = 1440,
                   now: pd.Timestamp | None = None, include_resolution: bool = True) -> pd.DataFrame:
    """Tidy daily history for a Kalshi market ticker, or for every market of an event ticker."""
    try:
        markets = [kalshi_market(ticker, get)]
    except FetchError as e:
        if e.status != 404:
            raise
        markets = kalshi_event_markets(ticker, get)
        if not markets:
            raise
    now = to_utc(now) if now is not None else pd.Timestamp.now(tz="UTC")
    rows: list[dict] = []
    labels: dict[str, str] = {}
    for m in markets:
        tk = m["ticker"]
        out = m.get("yes_sub_title") or m.get("subtitle") or m.get("title") or tk
        labels[out] = out if parse_threshold(out)[0] else (m.get("title") or out)
        open_t, close_t = to_utc(m.get("open_time")), to_utc(m.get("close_time"))
        if open_t is pd.NaT:
            continue
        end = min(close_t, now) if close_t is not pd.NaT else now
        series = kalshi_series_ticker(m, get)
        for c in kalshi_candles(series, tk, int(open_t.timestamp()), int(end.timestamp()), get, period):
            t = c.get("end_period_ts")
            p, src = _kprice(c.get("price"), "close"), "candle_close"
            if math.isnan(p):
                bid, ask = _kprice(c.get("yes_bid"), "close"), _kprice(c.get("yes_ask"), "close")
                p, src = ((bid + ask) / 2, "candle_mid") if not (math.isnan(bid) or math.isnan(ask)) else (p, src)
            if t is None or math.isnan(p):
                continue
            rows.append({"timestamp": to_utc(t, "s"), "platform": "kalshi", "market": tk, "outcome": out,
                         "probability": p, "source": src})
        res = str(m.get("result") or "").lower()
        if include_resolution and res in ("yes", "no"):
            times = [to_utc(m.get(k)) for k in ("settlement_ts", "expiration_time", "close_time")]
            times = [x for x in times if x is not pd.NaT]
            if times:
                rows.append({"timestamp": max(times), "platform": "kalshi", "market": tk, "outcome": out,
                             "probability": 1.0 if res == "yes" else 0.0, "source": "resolution"})
    return _tidy(_label_rows(rows, labels))


# --------------------------------------------------------------------------- as-of helpers
def prob_asof(history: pd.DataFrame, ts, min_lag: pd.Timedelta | None = None,
              max_age: pd.Timedelta | None = None, by: Iterable[str] = GROUP_COLS) -> pd.DataFrame:
    """Last probability of each (platform, market, outcome) at or before `ts`.

    Strictly no later data: rows with timestamp > ts - min_lag are dropped before anything else.
    Groups with no observation by then are absent. `age` = ts - timestamp of the row used; rows older
    than `max_age` are dropped (stale). A naive `ts` is read as UTC.
    """
    ts = to_utc(ts)
    cutoff = ts - (min_lag if min_lag is not None else pd.Timedelta(0))
    if history is None or len(history) == 0:
        return pd.DataFrame(columns=list(HIST_COLS) + ["age"])
    h = history.copy()
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    h = h[h["timestamp"] <= cutoff]
    if h.empty:
        return pd.DataFrame(columns=list(h.columns) + ["age"])
    h = h.sort_values("timestamp", kind="mergesort")
    last = h.groupby(list(by), sort=False, dropna=False).tail(1).copy()
    last["age"] = ts - last["timestamp"]
    if max_age is not None:
        last = last[last["age"] <= max_age]
    return last.sort_values(list(by), kind="mergesort").reset_index(drop=True)


def buckets_to_ladder(kinds, lo, hi, probs) -> tuple[np.ndarray, np.ndarray]:
    """Mutually exclusive buckets -> survival points P(V >= lower edge).

    kinds: 'between' [lo, hi), 'le' (-inf, lo], 'ge' [lo, inf). Bucket probabilities are renormalised
    to sum to 1 (Manifold sum-to-one markets are close to 1 already)."""
    k = np.asarray(kinds, dtype=object)
    lo = np.asarray(lo, dtype=float)
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    if p.sum() <= 0:
        raise ValueError("bucket probabilities sum to zero")
    p = p / p.sum()
    edges = np.where(k == "le", -np.inf, lo)          # lower edge of each bucket; 'le' has none
    xs = np.unique(lo[k != "le"])
    surv = np.array([p[edges >= x].sum() for x in xs])
    return xs, surv


def ladder_asof(history: pd.DataFrame, ts, market: str | None = None,
                **kw) -> tuple[np.ndarray, np.ndarray]:
    """(thresholds, P(V >= threshold)) from one market's outcomes as of `ts` (see prob_asof for kw).

    Ladder of binaries: 'ge' outcomes are used as is, 'le' outcomes as 1 - p (continuous variable).
    If any outcome is a 'between' bucket the market is treated as mutually exclusive buckets."""
    h = history if market is None else history[history["market"] == market]
    if h["market"].nunique() > 1:
        raise ValueError("ladder_asof needs a single market; pass market=")
    snap = prob_asof(h, ts, **kw)
    snap = snap[snap["kind"].isin(["ge", "le", "between"]) & snap["threshold"].notna()]
    if snap.empty:
        return np.array([]), np.array([])
    if (snap["kind"] == "between").any():
        return buckets_to_ladder(snap["kind"], snap["threshold"], snap["upper"], snap["probability"])
    p = np.where(snap["kind"] == "le", 1.0 - snap["probability"], snap["probability"])
    return snap["threshold"].to_numpy(float), p.astype(float)


# --------------------------------------------------------------------------- implied distribution
def isotonic_decreasing(y, w=None) -> np.ndarray:
    """Weighted least-squares non-increasing fit (pool-adjacent-violators)."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    blocks: list[list[float]] = []                    # [mean of -y, weight, count]
    for yi, wi in zip(-y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            blocks.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, c1 + c2])
    return -np.concatenate([np.full(int(c), v) for v, _, c in blocks]) if blocks else np.array([])


@dataclass
class ImpliedDistribution:
    thresholds: np.ndarray            # sorted, unique (anchors included)
    survival: np.ndarray              # monotone P(V >= x)
    raw: np.ndarray                   # input probabilities after clipping/averaging duplicates
    median: float
    q10: float
    q90: float
    flags: dict = field(default_factory=dict)     # 'ok' | 'below_ladder' | 'above_ladder' | 'clamped' | 'extrapolated'
    max_adjustment: float = 0.0       # largest |survival - raw| from the monotone fix
    scale: str = "log"
    tail: str = "nan"

    def quantile(self, q: float) -> tuple[float, str]:
        return survival_quantile(self.thresholds, self.survival, q, self.scale, self.tail)


def survival_quantile(x, s, q: float, scale: str = "log", tail: str = "nan") -> tuple[float, str]:
    """Value v with P(V >= v) = 1 - q on a monotone ladder; returns (value, flag)."""
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    if len(x) == 0:
        return math.nan, "empty"
    f = np.log if scale == "log" else (lambda v: v)
    finv = np.exp if scale == "log" else (lambda v: v)
    target = 1.0 - q
    below = np.nonzero(s <= target)[0]
    if len(below) and below[0] > 0:
        i = below[0]
        frac = (s[i - 1] - target) / (s[i - 1] - s[i])
        return float(finv(f(x[i - 1]) + frac * (f(x[i]) - f(x[i - 1])))), "ok"
    side = "below_ladder" if len(below) else "above_ladder"
    if len(below) and s[0] == target:
        return float(x[0]), "ok"
    if tail == "clamp":
        return float(x[0] if side == "below_ladder" else x[-1]), "clamped"
    if tail == "extrapolate" and len(x) >= 2:
        j0, j1 = (0, 1) if side == "below_ladder" else (len(x) - 2, len(x) - 1)
        ds = s[j1] - s[j0]
        if ds < 0:
            v = f(x[j0]) + (target - s[j0]) / ds * (f(x[j1]) - f(x[j0]))
            return float(finv(v)), "extrapolated"
    return math.nan, side


def implied_distribution(thresholds, probs, weights=None, lower: float | None = None,
                         upper: float | None = None, scale: str = "log",
                         tail: str = "nan") -> ImpliedDistribution:
    """Median and 10%/90% quantiles from a ladder of P(V >= threshold).

    Steps: drop NaNs; clip to [0, 1]; average duplicate thresholds (weighted); add optional anchors
    (lower, 1.0) and (upper, 0.0) with dominant weight; weighted isotonic regression to a non-increasing
    survival curve; read quantiles by interpolation (linear in log(x) by default; scale='linear' for
    linear). Beyond the ladder: see module docstring and `tail`.
    """
    x = np.asarray(thresholds, dtype=float)
    p = np.asarray(probs, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    if not (len(x) == len(p) == len(w)):
        raise ValueError("thresholds, probs and weights must have the same length")
    ok = ~(np.isnan(x) | np.isnan(p) | np.isnan(w)) & (w > 0)
    x, p, w = x[ok], np.clip(p[ok], 0.0, 1.0), w[ok]
    big = 1e6 * max(1.0, float(w.sum()) if len(w) else 1.0)
    if lower is not None:
        x, p, w = np.append(x, lower), np.append(p, 1.0), np.append(w, big)
    if upper is not None:
        x, p, w = np.append(x, upper), np.append(p, 0.0), np.append(w, big)
    if scale == "log" and np.any(x <= 0):
        raise ValueError("log interpolation needs positive thresholds; use scale='linear'")
    if len(x) == 0:
        return ImpliedDistribution(x, x, x, math.nan, math.nan, math.nan,
                                   {"q10": "empty", "median": "empty", "q90": "empty"}, 0.0, scale, tail)
    df = pd.DataFrame({"x": x, "p": p, "w": w})
    df["pw"] = df.p * df.w
    g = df.groupby("x", sort=True).agg(pw=("pw", "sum"), w=("w", "sum"))
    xs = g.index.to_numpy(float)
    ws = g["w"].to_numpy(float)
    raw = (g["pw"] / g["w"]).to_numpy(float)
    surv = np.clip(isotonic_decreasing(raw, ws), 0.0, 1.0)
    res = {k: survival_quantile(xs, surv, q, scale, tail) for k, q in (("q10", 0.1), ("median", 0.5), ("q90", 0.9))}
    return ImpliedDistribution(
        thresholds=xs, survival=surv, raw=raw,
        median=res["median"][0], q10=res["q10"][0], q90=res["q90"][0],
        flags={k: v[1] for k, v in res.items()},
        max_adjustment=float(np.max(np.abs(surv - raw))) if len(raw) else 0.0,
        scale=scale, tail=tail)


def distribution_path(history: pd.DataFrame, dates, market: str | None = None,
                      lower: float | Callable | None = None, upper: float | None = None,
                      tail: str = "nan", **asof_kw) -> pd.DataFrame:
    """Implied median / q10 / q90 of one market's ladder at each date (as-of, no look-ahead).

    `lower` may be a callable date -> float (e.g. the last run-rate disclosed before that date) for
    reach-by-date ladders."""
    out = []
    for d in dates:
        xs, ps = ladder_asof(history, d, market=market, **asof_kw)
        lo = lower(d) if callable(lower) else lower
        if len(xs) == 0 and lo is None and upper is None:
            out.append({"date": to_utc(d), "n_thresholds": 0, "median": math.nan, "q10": math.nan,
                        "q90": math.nan, "flag_median": "empty", "flag_q10": "empty", "flag_q90": "empty",
                        "max_adjustment": math.nan})
            continue
        dist = implied_distribution(xs, ps, lower=lo, upper=upper, tail=tail)
        out.append({"date": to_utc(d), "n_thresholds": int(len(xs)), "median": dist.median, "q10": dist.q10,
                    "q90": dist.q90, "flag_median": dist.flags["median"], "flag_q10": dist.flags["q10"],
                    "flag_q90": dist.flags["q90"], "max_adjustment": dist.max_adjustment})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- catalog-driven fetch
FETCHERS = {"manifold": manifold_history, "polymarket": polymarket_history, "kalshi": kalshi_history}


def catalog_targets(path: str = CATALOG_PATH, platforms: Iterable[str] | None = None,
                    metrics: Iterable[str] | None = None) -> list[tuple[str, str]]:
    """Unique (platform, id) pairs from the catalog that a fetcher can handle."""
    c = pd.read_csv(path, dtype=str).fillna("")
    c = c[c["platform"].isin(list(platforms or FETCHERS)) & (c["market_id_or_slug"] != "")]
    if metrics:
        c = c[c["metric"].isin(list(metrics))]
    c = c[~c["market_id_or_slug"].str.startswith("event/")]      # Manifold event pages are not markets
    return list(dict.fromkeys(zip(c["platform"], c["market_id_or_slug"])))


def fetch_catalog(path: str = CATALOG_PATH, out: str | None = PRICES_PATH, get: Getter = http_get_json,
                  platforms: Iterable[str] | None = None, metrics: Iterable[str] | None = None,
                  verbose: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch every catalogued market; write the combined tidy history to `out` only if non-empty.

    Returns (history, status) where status lists {platform, id, ok, rows, error}."""
    frames, status = [], []
    fetched_at = pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds")
    for platform, mid in catalog_targets(path, platforms, metrics):
        try:
            h = FETCHERS[platform](mid, get=get)
            h["fetched_at"] = fetched_at
            frames.append(h)
            status.append({"platform": platform, "id": mid, "ok": True, "rows": len(h), "error": ""})
        except (FetchError, KeyError, ValueError, TypeError) as e:
            status.append({"platform": platform, "id": mid, "ok": False, "rows": 0, "error": str(e)[:200]})
        if verbose:
            s = status[-1]
            print(f"{platform:10s} {mid[:60]:60s} {'ok' if s['ok'] else 'FAIL'} {s['rows']:6d} {s['error']}",
                  file=sys.stderr)
    hist = pd.concat(frames, ignore_index=True) if frames else _tidy([])
    if out and len(hist):
        hist.to_csv(out, index=False, quoting=csv.QUOTE_MINIMAL)
    return hist, status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m runrate_nowcast.prediction_markets")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="fetch every catalogued market into data/prediction_markets_prices.csv")
    f.add_argument("--catalog", default=CATALOG_PATH)
    f.add_argument("--out", default=PRICES_PATH)
    f.add_argument("--platform", action="append", choices=sorted(FETCHERS))
    f.add_argument("--metric", action="append")
    a = sub.add_parser("asof", help="implied median / 10% / 90% of one market's ladder on given dates")
    a.add_argument("--prices", default=PRICES_PATH)
    a.add_argument("--market", required=True)
    a.add_argument("--date", action="append", required=True)
    a.add_argument("--lower", type=float)
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        hist, status = fetch_catalog(args.catalog, args.out, platforms=args.platform, metrics=args.metric)
        n_ok = sum(s["ok"] for s in status)
        print(f"{n_ok}/{len(status)} markets fetched, {len(hist)} rows"
              + (f" -> {args.out}" if len(hist) else " (nothing written)"))
        return 0 if n_ok else 1
    hist = pd.read_csv(args.prices)
    print(distribution_path(hist, args.date, market=manifold_slug(args.market) if "/" in args.market else args.market,
                            lower=args.lower).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
