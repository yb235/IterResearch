"""OpenRouter capacity and provider mix for Claude models (data plan, task C).

altdata.fetch_openrouter_authors already takes weekly tokens by model author. This module
adds, for Anthropic's models: daily usage per model, how each model's traffic splits across
the providers that serve it (a proxy for the cloud channel), OpenRouter spend share, and
capacity signals (uptime, latency, throughput) per provider endpoint.

Every fetcher returns a tidy frame with TIDY_COLS:
  date       day (or hour) the value refers to
  model      OpenRouter permaslug, e.g. anthropic/claude-opus-5.5-20260921
             ("Others" = the chart's residual bucket, "(total)" = sum over all models,
              "anthropic/*" = sum over the author's models)
  variant    standard | batch | all
  provider   provider family: anthropic, claude-on-aws, amazon-bedrock, google-vertex, azure,
             or "all" for model-level values
  endpoint   provider endpoint, e.g. amazon-bedrock/us-east-1, anthropic/fast; "all" if none
  metric     see the parse_* functions
  value      float
  published  first day the value could have been seen (see "Dating" below)
  fetched    day this run downloaded it
  source     short endpoint name

Endpoints (public, no key). permaslug = `canonical_slug` from /api/v1/models; variant =
standard or batch. The /api/frontend/v1/rankings/* paths and their parameters are in the
site's JS bundles; the /api/frontend/v1/stats/* paths are called server-side (they are not
in the bundles) and were found by probing names; their only parameters are permaslug and
variant (effective-pricing also takes range=1w|1m|3m|all; uptime-hourly takes id=<endpoint id>).

  endpoint                                       content                                       history
  /api/v1/models/{slug}/endpoints                per endpoint: price, status,                   current only
                                                 uptime_last_5m / 30m / 1d (latency and
                                                 throughput fields are null here)
  /api/frontend/v1/stats/endpoint                per endpoint: p50-p99 latency and throughput,  current only, 30-minute window
                                                 request_count in the window, price, status
  /api/frontend/v1/stats/effective-pricing       per endpoint: tokens over a recent window      summary: current only;
                                                 (undocumented, roughly one day, rolling),      daily price chart: since the
                                                 effective $/M input and output, cache-hit      model's launch with range=all
                                                 rate; daily effective input $/M per endpoint   (8 days by default)
  /api/frontend/v1/stats/model-activity          per model variant, daily: prompt, completion,  rolling 31 days, today partial
                                                 reasoning, cached and BYOK tokens, requests,
                                                 tool calls (range is ignored)
  /api/frontend/v1/stats/uptime-recent           per endpoint, daily uptime % and judged mins   3 days, today partial
  /api/frontend/v1/stats/uptime-hourly?id=       one endpoint, hourly uptime %                  ~72 hours
  /api/frontend/v1/stats/throughput-comparison   per endpoint, daily p50 throughput (tok/s)     8 days, today partial
  /api/frontend/v1/stats/latency-comparison      per endpoint, daily p50 latency (ms)           8 days, today partial
  /api/frontend/v1/rankings/models               every model variant over the last day / 7 /    one rolling window ending at
      ?view=day|week|month                       30 complete days: tokens, BYOK tokens,         the last complete UTC day, plus
      &metric=tokens_excl_byok|spend             requests, change vs the previous window;       `change` vs the window before
                                                 BYOK fields are 0 unless metric=tokens_excl_byok;
                                                 metric=spend -> share of all OpenRouter spend
  /api/frontend/v1/rankings/performance          per model: p50 latency and throughput,         current only (about one day)
                                                 requests, provider count
  /provider/{slug}   (HTML, RSC "chartData")     daily tokens on that provider by model,        rolling 90 days, today partial
                                                 top 9 per day + Others
  /{author}          (HTML, RSC "data")          daily tokens of the author's models,           rolling 91 days, today partial
                                                 top 9 per day + Others

Dating, revisions and look-ahead
  * Completed periods look final: every complete week of author-market-share and
    model-rankings-chart was identical in the 2026-11-11 and 2026-11-12 snapshots. (Two
    fetches of model-activity and the provider pages 30 minutes apart were identical too,
    but those responses are cached for hours, so that says little.) Completed periods get
    published = period end + 1 day. Rows older than `fetched` - 1 day were backfilled from a
    rolling window: a strict backtest that distrusts "no revisions" should use `fetched`.
    Compare consecutive daily snapshots to keep checking this.
  * Partial periods are dropped (keep_partial=False): today's row in every daily series, and
    the current hour in uptime-hourly. Partial values grow during the day and lag (cache).
  * The OLDEST bucket of a rolling window can be truncated: author-market-share's first week
    (2025-11-10) fell 22% between the 11-11 and 11-12 snapshots because the window lost a
    day (altdata.fetch_openrouter_authors keeps that week; it should drop it). The daily
    charts here start at 00:00 and their first day is in line with the next ones.
  * rankings/performance intermittently returns an empty list (full at 16:55 UTC, empty
    from 17:20 on 2026-11-12); fetch_rankings_performance retries once.
  * Top 9 + Others (provider and author charts): a model's own series has gaps on days it
    falls into Others. Provider totals (sum including Others) are complete. On google-vertex
    and azure, Claude models compete with Gemini / GPT for the top 9, so Claude tokens there
    are a lower bound; on anthropic and claude-on-aws nearly everything is Claude.
  * Provider charts and model-activity include BYOK tokens (customers' own provider keys).
  * What the provider split measures: OpenRouter's routing of its own customers' Claude
    traffic (price, latency, uptime, capacity and its contracts decide it), plus BYOK. It is
    a capacity / routing signal, not the cloud channel mix of Anthropic's revenue. Example:
    claude-on-aws carried ~5bn tokens/day in mid-August and ~600bn/day in November.
  * effective-pricing token totals use an undocumented rolling window (~1 day; 805 -> 822bn
    for Opus 5.5 within 15 minutes). Use them for provider shares, not levels.
  * stats/endpoint request counts cover the last 30 minutes: strong time-of-day effects.
  * Endpoint ids are stable UUIDs, but endpoints come and go (regional endpoints, the
    anthropic/fast tier), and the display names' ordinals ("Amazon Bedrock (2)") are not
    stable. Use provider_slug, as here.
  * rankings/models: each row's `date` is that model's last active day; the window end is
    the latest date in the response. New models enter mid-window (partial first window).
  * Nothing here goes back further than 91 days; history must be built by running
    snapshot() daily (weekly at minimum for the rankings windows).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from . import config

BASE = "https://openrouter.ai"
UA = "runrate-nowcast research script"
PAUSE = 1.0                                   # seconds between requests
OUT_DIR = os.path.join(config.PKG_DIR, "data", "openrouter")
TIDY_COLS = ["date", "model", "variant", "provider", "endpoint", "metric", "value", "published",
             "fetched", "source"]
CLAUDE_PROVIDERS = ("anthropic", "claude-on-aws", "amazon-bedrock", "google-vertex", "azure")
# Who sells the tokens. claude-on-aws ("Claude Platform on AWS") is Anthropic's own platform
# billed through AWS; amazon-bedrock is AWS's service.
CHANNEL = {"anthropic": "anthropic_direct", "claude-on-aws": "anthropic_on_aws",
           "amazon-bedrock": "aws_bedrock", "google-vertex": "google_vertex", "azure": "microsoft_azure"}
DAY = pd.Timedelta(days=1)


# --------------------------------------------------------------------------- HTTP
_last_request = [0.0]


def _get(url: str, retries: int = 3, pause: float = PAUSE) -> bytes:
    """GET with a fixed User-Agent, at most one request per `pause` seconds, retry on 429/5xx."""
    for i in range(retries):
        wait = _last_request[0] + pause - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request[0] = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or i == retries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if i == retries - 1:
                raise
        time.sleep(2 ** (i + 1))
    raise RuntimeError("unreachable")


def _get_json(path: str, **params) -> dict:
    q = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None}, safe="/")
    return json.loads(_get(BASE + path + ("?" + q if q else "")))


def _get_text(path: str) -> str:
    return _get(BASE + path).decode("utf-8", "replace")


# --------------------------------------------------------------------------- helpers
def _day(s) -> pd.Timestamp:
    return pd.Timestamp(str(s)[:10])


def _ts(x) -> pd.Timestamp:
    return pd.Timestamp(x).normalize()


def provider_key(slug: str) -> str:
    """'amazon-bedrock/us-east-1' -> 'amazon-bedrock', 'anthropic/fast' -> 'anthropic'."""
    return str(slug).split("/")[0]


def _rows_to_df(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=TIDY_COLS)
    if len(df):
        df["value"] = df["value"].astype(float)
    return df


def _r(date, model, variant, provider, endpoint, metric, value, published, fetched, source):
    return {"date": date, "model": model, "variant": variant, "provider": provider, "endpoint": endpoint,
            "metric": metric, "value": value, "published": published, "fetched": fetched, "source": source}


def _per_m(price) -> float | None:
    try:
        return float(price) * 1e6
    except (TypeError, ValueError):
        return None


def _is_author(model: str, author: str) -> bool:
    return str(model).startswith(author + "/")


# --------------------------------------------------------------------------- model list
def parse_claude_models(js: dict, author: str = "anthropic") -> pd.DataFrame:
    """/api/v1/models -> slug, permaslug, variant for the author's models."""
    rows = []
    for m in js["data"]:
        if not _is_author(m["id"], author):
            continue
        slug, _, var = m["id"].partition(":")
        rows.append({"slug": m["id"], "base_slug": slug, "permaslug": m.get("canonical_slug") or slug,
                     "variant": var or "standard", "created": pd.Timestamp(m.get("created", 0), unit="s"),
                     "price_prompt_per_m": _per_m(m["pricing"].get("prompt")),
                     "price_completion_per_m": _per_m(m["pricing"].get("completion"))})
    return pd.DataFrame(rows)


def claude_models(author: str = "anthropic") -> pd.DataFrame:
    return parse_claude_models(_get_json("/api/v1/models"), author)


# --------------------------------------------------------------------------- per-endpoint, current
def parse_v1_endpoints(js: dict, fetched, model: str | None = None, variant: str = "standard") -> pd.DataFrame:
    """/api/v1/models/{slug}/endpoints: price, status and uptime per endpoint (current)."""
    fetched = _ts(fetched)
    d = js["data"]
    rows = []
    for e in d["endpoints"]:
        m = model or e.get("model_id") or d.get("id")
        ep = e.get("tag") or e.get("provider_name")
        vals = {"price_prompt_per_m": _per_m(e["pricing"].get("prompt")),
                "price_completion_per_m": _per_m(e["pricing"].get("completion")),
                "price_cache_read_per_m": _per_m(e["pricing"].get("input_cache_read")),
                "status": e.get("status"),
                "uptime_last_5m": e.get("uptime_last_5m"), "uptime_last_30m": e.get("uptime_last_30m"),
                "uptime_last_1d": e.get("uptime_last_1d"),
                "latency_last_30m": e.get("latency_last_30m"), "throughput_last_30m": e.get("throughput_last_30m")}
        for k, v in vals.items():
            if v is not None:
                rows.append(_r(fetched, m, variant, provider_key(ep), ep, k, v, fetched, fetched, "v1/models/endpoints"))
    return _rows_to_df(rows)


def fetch_v1_endpoints(slug: str, permaslug: str | None = None, fetched=None) -> pd.DataFrame:
    base, _, var = slug.partition(":")
    js = _get_json(f"/api/v1/models/{slug}/endpoints")
    return parse_v1_endpoints(js, fetched or dt.date.today(), permaslug or base, var or "standard")


def endpoint_labels(stats_js: dict) -> dict:
    """endpoint id -> provider_slug (e.g. 'amazon-bedrock/us-east-1') from stats/endpoint."""
    return {e["id"]: e.get("provider_slug") or e.get("provider_name") for e in stats_js.get("data", [])}


def parse_endpoint_stats(js: dict, fetched) -> pd.DataFrame:
    """/api/frontend/v1/stats/endpoint: latency / throughput percentiles and request count in
    the last `window_minutes` (30), plus price and status, per endpoint (current)."""
    fetched = _ts(fetched)
    rows = []
    for e in js["data"]:
        m, var = e.get("model_variant_permaslug"), e.get("variant", "standard")
        ep = e.get("provider_slug") or e.get("provider_name")
        vals = {"price_prompt_per_m": _per_m((e.get("pricing") or {}).get("prompt")),
                "price_completion_per_m": _per_m((e.get("pricing") or {}).get("completion")),
                "status": e.get("status")}
        s = e.get("stats") or {}
        w = int(s.get("window_minutes") or 30)
        for k in ("p50_latency", "p90_latency", "p99_latency", "p50_throughput", "p90_throughput", "mean_latency"):
            vals[k + ("_ms" if "latency" in k else "")] = s.get(k)
        vals[f"requests_{w}m"] = s.get("request_count")
        for k, v in vals.items():
            if v is not None:
                rows.append(_r(fetched, m, var, provider_key(ep), ep, k, v, fetched, fetched, "stats/endpoint"))
    return _rows_to_df(rows)


def fetch_endpoint_stats(permaslug: str, variant: str = "standard", fetched=None) -> tuple[pd.DataFrame, dict]:
    """Returns (tidy frame, id -> provider_slug labels for the other per-endpoint fetchers)."""
    js = _get_json("/api/frontend/v1/stats/endpoint", permaslug=permaslug, variant=variant)
    return parse_endpoint_stats(js, fetched or dt.date.today()), endpoint_labels(js)


def _label(eid: str, labels: dict | None, fallback_provider: dict | None = None) -> tuple[str, str]:
    ep = (labels or {}).get(eid)
    if ep:
        return provider_key(ep), ep
    if eid == "openrouter":
        return "all", "openrouter"
    return (fallback_provider or {}).get(eid, "unknown"), eid


def parse_effective_pricing(js: dict, model: str, variant: str, fetched, labels: dict | None = None,
                            keep_partial: bool = False) -> pd.DataFrame:
    """/api/frontend/v1/stats/effective-pricing.

    Current (date = fetched): per endpoint tokens_recent (rolling ~1-day window), effective
    input / output $/M and cache-hit rate; model-level weighted prices and cache-hit rate.
    Daily: effective_input_price_per_m per endpoint (history length set by range=)."""
    fetched = _ts(fetched)
    d = js["data"]
    slugs = d.get("endpointProviderSlugs") or {}
    rows = []
    for s in d.get("providerSummaries", []):
        prov, ep = _label(s["endpointId"], labels, slugs)
        for k, src in (("tokens_recent", "totalTokens"), ("effective_input_price_per_m", "effectiveInputPrice"),
                       ("effective_output_price_per_m", "effectiveOutputPrice"), ("cache_hit_rate", "cacheHitRate")):
            if s.get(src) is not None:
                rows.append(_r(fetched, model, variant, prov, ep, k, s[src], fetched, fetched, "stats/effective-pricing"))
    for k, src in (("weighted_input_price_per_m", "weightedInputPrice"),
                   ("weighted_output_price_per_m", "weightedOutputPrice"), ("weighted_cache_hit_rate", "weightedCacheHitRate")):
        if d.get(src) is not None:
            rows.append(_r(fetched, model, variant, "all", "all", k, d[src], fetched, fetched, "stats/effective-pricing"))
    for pt in d.get("inputChartData", []):
        day = _day(pt["x"])
        if day >= fetched and not keep_partial:
            continue
        for eid, v in pt["y"].items():
            prov, ep = _label(eid, labels, slugs)
            rows.append(_r(day, model, variant, prov, ep, "effective_input_price_per_m_daily", v,
                           day + DAY if day < fetched else fetched, fetched, "stats/effective-pricing"))
    return _rows_to_df(rows)


def fetch_effective_pricing(permaslug: str, variant: str = "standard", labels=None, range_: str = "1w",
                            fetched=None, keep_partial=False) -> pd.DataFrame:
    js = _get_json("/api/frontend/v1/stats/effective-pricing", permaslug=permaslug, variant=variant, range=range_)
    return parse_effective_pricing(js, permaslug, variant, fetched or dt.date.today(), labels, keep_partial)


# --------------------------------------------------------------------------- per-model daily usage
ACTIVITY_METRICS = {"completion_tokens": ("total_completion_tokens",), "cached_tokens": ("total_native_tokens_cached",),
                    "byok_tokens": ("total_byok_prompt_tokens", "total_byok_completion_tokens"), "requests": ("count",)}
ACTIVITY_EXTRA = {"prompt_tokens": ("total_prompt_tokens",), "reasoning_tokens": ("total_native_tokens_reasoning",),
                  "tool_calls": ("total_tool_calls",), "tool_call_errors": ("requests_with_tool_call_errors",)}
ACTIVITY_FIELDS = ("date", "model_permaslug", "variant", "variant_permaslug", "count", "total_prompt_tokens",
                   "total_completion_tokens", "total_native_tokens_reasoning", "total_native_tokens_cached",
                   "total_byok_prompt_tokens", "total_byok_completion_tokens", "total_tool_calls",
                   "requests_with_tool_call_errors", "num_media_prompt")


def parse_model_activity(js: dict, fetched, keep_partial: bool = False, full: bool = False) -> pd.DataFrame:
    """/api/frontend/v1/stats/model-activity: daily usage of one model variant (31 days).

    Metrics: tokens (= prompt + completion, the same definition as the rankings), plus
    ACTIVITY_METRICS (and ACTIVITY_EXTRA if full). Completed days are published the next
    day. The response is cached for hours (data.cachedAt), so 'today' lags."""
    fetched = _ts(fetched)
    rows = []
    for r in js["data"]["analytics"]:
        day = _day(r["date"])
        if day >= fetched and not keep_partial:
            continue
        pub = day + DAY if day < fetched else fetched
        m, var = r.get("variant_permaslug") or r["model_permaslug"], r.get("variant", "standard")
        vals = {"tokens": r.get("total_prompt_tokens", 0) + r.get("total_completion_tokens", 0)}
        for k, cols in {**ACTIVITY_METRICS, **(ACTIVITY_EXTRA if full else {})}.items():
            vals[k] = sum(r.get(c) or 0 for c in cols)
        for k, v in vals.items():
            rows.append(_r(day, m, var, "all", "all", k, v, pub, fetched, "stats/model-activity"))
    return _rows_to_df(rows)


def fetch_model_activity(permaslug: str, variant: str = "standard", fetched=None, keep_partial=False) -> pd.DataFrame:
    js = _get_json("/api/frontend/v1/stats/model-activity", permaslug=permaslug, variant=variant)
    return parse_model_activity(js, fetched or dt.date.today(), keep_partial)


# --------------------------------------------------------------------------- per-endpoint daily capacity
def parse_uptime_recent(js: dict, model: str, variant: str, fetched, labels: dict | None = None,
                        keep_partial: bool = False) -> pd.DataFrame:
    """/api/frontend/v1/stats/uptime-recent: daily uptime % and judged minutes per endpoint
    (3 days). Key 'openrouter' is reported as provider 'all', endpoint 'openrouter'."""
    fetched = _ts(fetched)
    rows = []
    for eid, pts in js["data"].items():
        prov, ep = _label(eid, labels)
        for p in pts:
            day = _day(p["date"])
            if day >= fetched and not keep_partial:
                continue
            pub = day + DAY if day < fetched else fetched
            if p.get("uptime") is not None:
                rows.append(_r(day, model, variant, prov, ep, "uptime_pct", p["uptime"], pub, fetched, "stats/uptime-recent"))
            rows.append(_r(day, model, variant, prov, ep, "uptime_judged_minutes", p.get("judgedMinutes") or 0, pub,
                           fetched, "stats/uptime-recent"))
    return _rows_to_df(rows)


def fetch_uptime_recent(permaslug: str, variant: str = "standard", labels=None, fetched=None, keep_partial=False):
    js = _get_json("/api/frontend/v1/stats/uptime-recent", permaslug=permaslug, variant=variant)
    return parse_uptime_recent(js, permaslug, variant, fetched or dt.date.today(), labels, keep_partial)


def parse_uptime_hourly(js: dict, model: str, variant: str, endpoint: str, fetched, keep_partial: bool = False):
    """/api/frontend/v1/stats/uptime-hourly?id=: hourly uptime of one endpoint (~72 hours).
    The latest hour is partial and dropped unless keep_partial."""
    fetched = _ts(fetched)
    hist = js["data"]["history"]
    latest = max((pd.Timestamp(p["date"]) for p in hist), default=None)
    rows = []
    for p in hist:
        t = pd.Timestamp(p["date"])
        if t == latest and not keep_partial:
            continue
        if p.get("uptime") is not None:
            rows.append(_r(t, model, variant, provider_key(endpoint), endpoint, "uptime_pct_hourly", p["uptime"],
                           t + pd.Timedelta(hours=1), fetched, "stats/uptime-hourly"))
    return _rows_to_df(rows)


def fetch_uptime_hourly(endpoint_id: str, model: str, variant: str, endpoint: str, fetched=None, keep_partial=False):
    js = _get_json("/api/frontend/v1/stats/uptime-hourly", id=endpoint_id)
    return parse_uptime_hourly(js, model, variant, endpoint, fetched or dt.date.today(), keep_partial)


def parse_comparison(js: dict, model: str, variant: str, fetched, metric: str, labels: dict | None = None,
                     keep_partial: bool = False) -> pd.DataFrame:
    """throughput-comparison / latency-comparison: daily p50 per endpoint (8 days)."""
    fetched = _ts(fetched)
    rows = []
    for pt in js["data"]:
        day = _day(pt["x"])
        if day >= fetched and not keep_partial:
            continue
        pub = day + DAY if day < fetched else fetched
        for eid, v in pt["y"].items():
            prov, ep = _label(eid, labels)
            if v is not None:
                rows.append(_r(day, model, variant, prov, ep, metric, v, pub, fetched, f"stats/{metric.split('_')[1]}-comparison"))
    return _rows_to_df(rows)


def fetch_comparison(permaslug: str, kind: str = "throughput", variant: str = "standard", labels=None,
                     fetched=None, keep_partial=False) -> pd.DataFrame:
    """kind = 'throughput' (metric p50_throughput_daily, tok/s) or 'latency' (p50_latency_ms_daily)."""
    js = _get_json(f"/api/frontend/v1/stats/{kind}-comparison", permaslug=permaslug, variant=variant)
    metric = "p50_throughput_daily" if kind == "throughput" else "p50_latency_ms_daily"
    return parse_comparison(js, permaslug, variant, fetched or dt.date.today(), metric, labels, keep_partial)


# --------------------------------------------------------------------------- rankings
def parse_rankings_models(js: dict, view: str, fetched, metric: str | None = None,
                          author: str = "anthropic", end=None) -> pd.DataFrame:
    """/api/frontend/v1/rankings/models?view=day|week|month[&metric=spend].

    date = window end (latest date in the response = last complete UTC day). Per model
    variant of the author and summed over them (model 'anthropic/*'):
      metric=tokens_excl_byok (use this): {view}_tokens, {view}_byok_tokens, {view}_requests,
                      {view}_tokens_excl_byok, {view}_tokens_excl_byok_change (vs previous window)
      no metric:      the same token totals, but the BYOK fields are all 0 in this mode, so
                      byok is not reported; {view}_tokens_change
      metric=spend:   {view}_spend_share (share of all OpenRouter spend in the window)."""
    fetched = _ts(fetched)
    data = js["data"]
    if not data:
        return _rows_to_df([])
    end = _ts(end) if end is not None else max(_day(r["date"]) for r in data)
    pub = end + DAY
    rows, tot = [], {}
    for r in data:
        m = r["model_permaslug"]
        if not _is_author(m, author):
            continue
        var = r.get("variant", "standard")
        if metric == "spend":
            vals = {f"{view}_spend_share": r.get("rankingMetricValue")}
        else:
            vals = {f"{view}_tokens": r["total_prompt_tokens"] + r["total_completion_tokens"],
                    f"{view}_requests": r.get("count")}
            if metric == "tokens_excl_byok":
                vals[f"{view}_byok_tokens"] = r.get("total_byok_prompt_tokens", 0) + r.get("total_byok_completion_tokens", 0)
                vals[f"{view}_tokens_excl_byok"] = r.get("rankingMetricValue")
        for k, v in vals.items():
            if v is None:
                continue
            rows.append(_r(end, m, var, "all", "all", k, v, pub, fetched, "rankings/models"))
            tot[k] = tot.get(k, 0.0) + float(v)
        if metric != "spend" and r.get("change") is not None:
            rows.append(_r(end, m, var, "all", "all", f"{view}_{metric or 'tokens'}_change", r["change"], pub, fetched,
                           "rankings/models"))
    for k, v in tot.items():
        rows.append(_r(end, f"{author}/*", "all", "all", "all", k, v, pub, fetched, "rankings/models"))
    return _rows_to_df(rows)


def fetch_rankings_models(view: str = "week", metric: str | None = None, fetched=None,
                          author: str = "anthropic") -> tuple[pd.DataFrame, dict]:
    js = _get_json("/api/frontend/v1/rankings/models", view=view, metric=metric)
    return parse_rankings_models(js, view, fetched or dt.date.today(), metric, author), js


def parse_rankings_performance(js: dict, fetched, author: str = "anthropic") -> pd.DataFrame:
    """/api/frontend/v1/rankings/performance: p50 latency / throughput, requests and
    provider count per model (current, roughly the last day)."""
    fetched = _ts(fetched)
    rows = []
    for r in js["data"]:
        if r.get("author") != author and not _is_author(r.get("id", ""), author):
            continue
        for k, src in (("p50_latency_ms", "p50_latency"), ("p50_throughput", "p50_throughput"),
                       ("requests_recent", "request_count"), ("provider_count", "provider_count")):
            if r.get(src) is not None:
                rows.append(_r(fetched, r["id"], "standard", "all", "all", k, r[src], fetched, fetched, "rankings/performance"))
    return _rows_to_df(rows)


def fetch_rankings_performance(fetched=None, author="anthropic", retry_wait: float = 30.0) -> tuple[pd.DataFrame, dict]:
    """The endpoint intermittently returns {"data": []} (seen 2026-11-12 17:20 UTC, an hour
    after a full response): retry once after retry_wait seconds."""
    js = _get_json("/api/frontend/v1/rankings/performance")
    if not js.get("data") and retry_wait:
        time.sleep(retry_wait)
        js = _get_json("/api/frontend/v1/rankings/performance")
    return parse_rankings_performance(js, fetched or dt.date.today(), author), js


# --------------------------------------------------------------------------- HTML charts (RSC payload)
_PUSH = re.compile(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)')


def rsc_payload(html: str) -> str:
    """Concatenate the React Server Components flight data embedded in a Next.js page."""
    return "".join(json.loads(s) for s in _PUSH.findall(html))


def parse_rsc_series(html: str, key: str) -> list:
    """First `"key":[{"x":"YYYY-MM-DD..."` array of {x, ys} points in the page's RSC payload."""
    t = rsc_payload(html)
    m = re.search(r'"%s":(?=\[\{"x":"\d{4}-\d{2}-\d{2})' % re.escape(key), t)
    if not m:
        raise ValueError(f"no {key!r} x/ys series in page")
    arr, _ = json.JSONDecoder().raw_decode(t[m.end():])
    return arr


def parse_daily_chart(series: list, fetched, provider: str, source: str, keep_partial: bool = False) -> pd.DataFrame:
    """{x, ys} daily token chart -> tidy rows: one per model (incl. 'Others') and '(total)'.
    A '_total' key (saved snapshots trimmed to one author) is used as the total."""
    fetched = _ts(fetched)
    rows = []
    for pt in series:
        day = _day(pt["x"])
        if day >= fetched and not keep_partial:
            continue
        pub = day + DAY if day < fetched else fetched
        total = 0.0
        for m, v in pt["ys"].items():
            if m == "_total":
                continue
            rows.append(_r(day, m, "all", provider, "all", "tokens", v, pub, fetched, source))
            total += float(v)
        total = float(pt["ys"].get("_total", total))
        rows.append(_r(day, "(total)", "all", provider, "all", "tokens", total, pub, fetched, source))
    return _rows_to_df(rows)


def fetch_provider_chart(provider_slug: str, fetched=None, keep_partial=False) -> tuple[pd.DataFrame, list]:
    """Daily tokens by model on one provider (/provider/{slug}, top 9 + Others, 90 days)."""
    series = parse_rsc_series(_get_text(f"/provider/{provider_slug}"), "chartData")
    return parse_daily_chart(series, fetched or dt.date.today(), provider_slug, "provider page", keep_partial), series


def fetch_author_chart(author: str = "anthropic", fetched=None, keep_partial=False) -> tuple[pd.DataFrame, list]:
    """Daily tokens of the author's models (/{author}, top 9 + Others, 91 days)."""
    series = parse_rsc_series(_get_text(f"/{author}"), "data")
    return parse_daily_chart(series, fetched or dt.date.today(), "all", "author page", keep_partial), series


# --------------------------------------------------------------------------- derived: provider mix
def provider_shares(df: pd.DataFrame, metric: str = "tokens", source: str = "provider page",
                    author: str = "anthropic", providers=CLAUDE_PROVIDERS) -> pd.DataFrame:
    """Share of each provider in a Claude model's traffic, per date.

    Works on provider-page rows (metric 'tokens'), effective-pricing rows ('tokens_recent')
    or stats/endpoint rows ('requests_30m'). Also returns the author total over all its
    models (model 'anthropic/*'). Output metric: '{metric}_provider_share'; provider
    google-vertex and azure are lower bounds for provider-page data (Claude in 'Others')."""
    d = df[(df.metric == metric) & (df.source == source) & df.provider.isin(providers)
           & df.model.map(lambda m: _is_author(m, author))]
    if not len(d):
        return _rows_to_df([])
    g = d.groupby(["date", "model", "provider"], as_index=False).agg(
        value=("value", "sum"), published=("published", "max"), fetched=("fetched", "max"))
    allm = g.groupby(["date", "provider"], as_index=False).agg(
        value=("value", "sum"), published=("published", "max"), fetched=("fetched", "max"))
    allm["model"] = f"{author}/*"
    g = pd.concat([g, allm], ignore_index=True)
    g["value"] = g["value"] / g.groupby(["date", "model"])["value"].transform("sum")
    g["variant"], g["endpoint"], g["metric"], g["source"] = "all", "all", f"{metric}_provider_share", source
    return g[TIDY_COLS]


# --------------------------------------------------------------------------- snapshot
def _trim_endpoint(e: dict) -> dict:
    keep = ("id", "name", "model_variant_permaslug", "provider_name", "provider_slug", "provider_region",
            "variant", "derived_service_tier",
            "quantization", "status", "is_disabled", "is_deranked", "is_byok_only", "context_length",
            "max_completion_tokens", "limit_rpm", "provider_model_id", "created_at", "stats",
            "tag", "service_tier", "uptime_last_5m", "uptime_last_30m", "uptime_last_1d",
            "latency_last_30m", "throughput_last_30m")
    out = {k: e[k] for k in keep if k in e}
    out["pricing"] = {k: v for k, v in (e.get("pricing") or {}).items()
                      if k in ("prompt", "completion", "input_cache_read", "input_cache_write", "discount")}
    return out


def _trim_rankings(js: dict, author: str) -> list:
    return [r for r in js["data"] if _is_author(r["model_permaslug"], author)]


def _trim_chart(series: list, author: str) -> list:
    """Keep the author's models, 'Others' and '_total' (sum of all models) per point."""
    out = []
    for pt in series:
        ys = {k: v for k, v in pt["ys"].items() if _is_author(k, author) or k == "Others"}
        ys["_total"] = sum(pt["ys"].values())
        out.append({"x": pt["x"], "ys": ys})
    return out


def _keep_rows(df: pd.DataFrame, author: str) -> pd.DataFrame:
    """Provider-page rows of the author's models, 'Others' and '(total)' only."""
    return df[df.model.map(lambda m: _is_author(m, author) or m in ("Others", "(total)"))]


def select_models(rank_week: pd.DataFrame, top_n: int = 10, author: str = "anthropic") -> list:
    """Author's standard-variant permaslugs ordered by tokens in the last week."""
    w = rank_week[(rank_week.metric == "week_tokens") & (rank_week.variant == "standard")
                  & rank_week.model.map(lambda m: _is_author(m, author))]
    return list(w.sort_values("value", ascending=False).model.head(top_n))


RAW_FILES = {"stats": "claude-endpoint-stats", "pricing": "claude-effective-pricing",
             "activity": "claude-model-activity", "capacity": "claude-capacity",
             "provider": "provider-daily-tokens", "author": "author-daily-tokens",
             "rankings": "rankings-models-anthropic", "performance": "rankings-performance-anthropic"}


def fetch_raw(fetched, models: list | None = None, top_n: int = 10, providers=CLAUDE_PROVIDERS,
              author: str = "anthropic", price_range: str = "1w", hourly: bool = False,
              log=print) -> dict:
    """Download everything for the author's main models; returns the trimmed raw responses
    (the content of the snapshot's JSON files). About 8 requests per model plus ~12."""
    fetched = _ts(fetched)
    raw = {k: {} for k in RAW_FILES}
    cat = parse_claude_models(_get_json("/api/v1/models"), author)
    rank = []
    for view, metric in (("day", "tokens_excl_byok"), ("week", "tokens_excl_byok"), ("month", "tokens_excl_byok"),
                         ("day", "spend"), ("week", "spend")):
        df, js = fetch_rankings_models(view, metric, fetched, author)
        rank.append(df)
        end = max((_day(r["date"]) for r in js["data"]), default=None)
        raw["rankings"][view + ("_" + metric if metric else "")] = {
            "window_end": f"{end:%Y-%m-%d}" if end is not None else None, "data": _trim_rankings(js, author)}
    df, js = fetch_rankings_performance(fetched, author)
    raw["performance"] = [r for r in js["data"] if r.get("author") == author]
    if not raw["performance"]:
        log("warning: rankings/performance returned no rows")
    if models is None:
        models = select_models(pd.concat(rank, ignore_index=True), top_n, author)
    log(f"models: {models}")

    for pm in models:
        rec = cat[cat.permaslug == pm]
        st = _get_json("/api/frontend/v1/stats/endpoint", permaslug=pm, variant="standard")
        labels = endpoint_labels(st)
        raw["stats"][pm] = {"stats_endpoint": [_trim_endpoint(e) for e in st["data"]]}
        std = rec[rec.variant == "standard"]
        if len(std):
            v1 = _get_json(f"/api/v1/models/{std.slug.iloc[0]}/endpoints")
            raw["stats"][pm]["v1_endpoints"] = [_trim_endpoint(e) for e in v1["data"]["endpoints"]]
        raw["pricing"][pm] = _get_json("/api/frontend/v1/stats/effective-pricing", permaslug=pm,
                                       variant="standard", range=price_range)["data"]
        raw["activity"][pm] = {}
        for var in ["standard"] + (["batch"] if (rec.variant == "batch").any() else []):
            ma = _get_json("/api/frontend/v1/stats/model-activity", permaslug=pm, variant=var)["data"]
            raw["activity"][pm][var] = {"cachedAt": ma.get("cachedAt"),
                                        "analytics": [{k: r.get(k) for k in ACTIVITY_FIELDS} for r in ma["analytics"]]}
        cap = {"endpoint_labels": labels,
               "uptime_recent": _get_json("/api/frontend/v1/stats/uptime-recent", permaslug=pm, variant="standard")["data"]}
        for kind in ("throughput", "latency"):
            cap[kind] = _get_json(f"/api/frontend/v1/stats/{kind}-comparison", permaslug=pm, variant="standard")["data"]
        if hourly:
            cap["uptime_hourly"] = {eid: _get_json("/api/frontend/v1/stats/uptime-hourly", id=eid)["data"] for eid in labels}
        raw["capacity"][pm] = cap
        log(f"  {pm}: {len(labels)} endpoints")

    for p in providers:
        raw["provider"][p] = _trim_chart(parse_rsc_series(_get_text(f"/provider/{p}"), "chartData"), author)
    raw["author"][author] = parse_rsc_series(_get_text(f"/{author}"), "data")
    return raw


def tidy_from_raw(raw: dict, fetched, author: str = "anthropic", providers=CLAUDE_PROVIDERS,
                  share_models: int = 3) -> pd.DataFrame:
    """Tidy frame (TIDY_COLS) from fetch_raw()'s output or a saved snapshot's JSON files.
    Provider-page shares are kept for the author total and the first `share_models` models."""
    fetched = _ts(fetched)
    frames = []
    for key, obj in raw.get("rankings", {}).items():
        view, _, metric = key.partition("_")
        rows, end = (obj["data"], obj.get("window_end")) if isinstance(obj, dict) else (obj, None)
        frames.append(parse_rankings_models({"data": rows}, view, fetched, metric or None, author, end))
    frames.append(parse_rankings_performance({"data": raw.get("performance") or []}, fetched, author))
    models = list(raw.get("stats", {}))
    for pm, st in raw.get("stats", {}).items():
        eps = [{"model_variant_permaslug": pm, **e} for e in st.get("stats_endpoint", [])]
        frames.append(parse_endpoint_stats({"data": eps}, fetched))
        if st.get("v1_endpoints"):
            v1 = parse_v1_endpoints({"data": {"endpoints": st["v1_endpoints"]}}, fetched, pm, "standard")
            frames.append(v1[v1.metric.str.startswith(("uptime", "latency", "throughput"))])
        cap = raw.get("capacity", {}).get(pm, {})
        labels = cap.get("endpoint_labels") or endpoint_labels({"data": eps})
        if pm in raw.get("pricing", {}):
            frames.append(parse_effective_pricing({"data": raw["pricing"][pm]}, pm, "standard", fetched, labels))
        for var, ma in raw.get("activity", {}).get(pm, {}).items():
            frames.append(parse_model_activity({"data": ma}, fetched))
        if "uptime_recent" in cap:
            frames.append(parse_uptime_recent({"data": cap["uptime_recent"]}, pm, "standard", fetched, labels))
        for kind, metric in (("throughput", "p50_throughput_daily"), ("latency", "p50_latency_ms_daily")):
            if kind in cap:
                frames.append(parse_comparison({"data": cap[kind]}, pm, "standard", fetched, metric, labels))
        for eid, h in (cap.get("uptime_hourly") or {}).items():
            frames.append(parse_uptime_hourly({"data": h}, pm, "standard", labels.get(eid, eid), fetched))
    for p, series in raw.get("provider", {}).items():
        frames.append(_keep_rows(parse_daily_chart(series, fetched, p, "provider page"), author))
    for a, series in raw.get("author", {}).items():
        frames.append(parse_daily_chart(series, fetched, "all", "author page"))

    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    page = provider_shares(out, "tokens", "provider page", author, providers)
    shares = [page[page.model.isin(models[:share_models] + [f"{author}/*"])],
              provider_shares(out, "tokens_recent", "stats/effective-pricing", author, providers),
              provider_shares(out, "requests_30m", "stats/endpoint", author, providers)]
    out = pd.concat([out] + [x for x in shares if len(x)], ignore_index=True)
    return out.sort_values(["source", "model", "provider", "endpoint", "metric", "date"]).reset_index(drop=True)


def save_snapshot(raw: dict, df: pd.DataFrame, fetched, out_dir: str = OUT_DIR) -> list:
    """Write the raw JSON files and the tidy CSV, prefixed with the fetch date."""
    tag = f"{_ts(fetched):%Y-%m-%d}"
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for key, name in RAW_FILES.items():
        path = os.path.join(out_dir, f"{tag}_{name}.json")
        with open(path, "w") as f:
            json.dump({"fetched": tag, "source": BASE, "data": raw.get(key, {})}, f, separators=(",", ":"))
        paths.append(path)
    path = os.path.join(out_dir, f"{tag}_claude_endpoints.csv")
    save_csv(df, path)
    return paths + [path]


def load_raw(date, out_dir: str = OUT_DIR) -> dict:
    """Read a saved snapshot's JSON files back into fetch_raw()'s structure."""
    tag = f"{_ts(date):%Y-%m-%d}"
    raw = {}
    for key, name in RAW_FILES.items():
        path = os.path.join(out_dir, f"{tag}_{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                raw[key] = json.load(f)["data"]
    return raw


def rebuild(date, out_dir: str = OUT_DIR, save: bool = False, **kw) -> pd.DataFrame:
    """Re-parse a saved snapshot without network access (e.g. after a parser change)."""
    df = tidy_from_raw(load_raw(date, out_dir), date, **kw)
    if save:
        save_csv(df, os.path.join(out_dir, f"{_ts(date):%Y-%m-%d}_claude_endpoints.csv"))
    return df


def snapshot(asof=None, out_dir: str = OUT_DIR, models: list | None = None, top_n: int = 10,
             providers=CLAUDE_PROVIDERS, author: str = "anthropic", price_range: str = "1w",
             hourly: bool = False, save: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Fetch everything above for the author's main models and save a dated snapshot.

    Writes to out_dir, with prefix YYYY-MM-DD_ (the fetch date):
      claude-endpoint-stats.json   stats/endpoint + v1 endpoints per model (trimmed)
      claude-effective-pricing.json, claude-model-activity.json (standard + batch),
      claude-capacity.json         uptime-recent, throughput- and latency-comparison
                                   (+ uptime-hourly per endpoint if hourly=True)
      provider-daily-tokens.json   provider pages' 90-day charts (author's models, Others, _total)
      author-daily-tokens.json     author page's 91-day chart
      rankings-models-anthropic.json, rankings-performance-anthropic.json (author's rows)
      claude_endpoints.csv         tidy rows (TIDY_COLS): provider pages trimmed to the author's
                                   models + Others + (total); provider-page shares for the author
                                   total and the top 3 models (provider_shares() gives the rest)
    Models default to the author's top_n standard variants by tokens in the last week.
    About 8 requests per model plus ~12, one per second. Returns the tidy frame."""
    fetched = _ts(asof or dt.date.today())
    log = print if verbose else (lambda *a, **k: None)
    raw = fetch_raw(fetched, models, top_n, providers, author, price_range, hourly, log)
    out = tidy_from_raw(raw, fetched, author, providers)
    if save:
        paths = save_snapshot(raw, out, fetched, out_dir)
        log(f"saved {len(out)} rows and {len(paths) - 1} raw files to {out_dir}")
    return out


def save_csv(df: pd.DataFrame, path: str) -> None:
    d = df.copy()
    for c in ("date", "published"):
        t = pd.to_datetime(d[c])
        d[c] = t.dt.strftime("%Y-%m-%d").where(t == t.dt.normalize(), t.dt.strftime("%Y-%m-%d %H:%M"))
    d["fetched"] = pd.to_datetime(d["fetched"]).dt.strftime("%Y-%m-%d")
    d[TIDY_COLS].to_csv(path, index=False, float_format="%.6g")


def load_csv(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    for c in ("date", "published", "fetched"):
        d[c] = pd.to_datetime(d[c], format="mixed")
    return d


if __name__ == "__main__":
    snapshot()
