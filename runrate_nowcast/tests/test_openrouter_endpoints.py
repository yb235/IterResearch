"""Offline tests for openrouter_endpoints: parsers on trimmed real responses, and a full
snapshot run against a fake server.

Run: python -m pytest runrate_nowcast/tests/test_openrouter_endpoints.py -q
"""
import json
import os

import pandas as pd
import pytest

from runrate_nowcast import openrouter_endpoints as oe

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "openrouter_endpoints.json")
F = pd.Timestamp("2026-11-12")
PM = "anthropic/claude-opus-5.5-20260921"
A, BED, VTX = "043a453c-8450-4da8-82c1-cf6cc19cba9e", "6d76a7f6-28bc-4af6-9ae1-9b12caeb9549", "a24cd509-61a3-47ae-acea-41b4b40e6645"


@pytest.fixture(scope="module")
def fx():
    with open(FIX) as f:
        return json.load(f)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access in an offline test")
    monkeypatch.setattr(oe, "_get", refuse)


@pytest.fixture
def labels(fx):
    return oe.endpoint_labels(fx["stats_endpoint"])


def one(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    assert len(df) == 1, (kw, len(df))
    return df.iloc[0]


def test_tidy_columns_everywhere(fx, labels):
    frames = [oe.parse_v1_endpoints(fx["v1_endpoints"], F, PM), oe.parse_endpoint_stats(fx["stats_endpoint"], F),
              oe.parse_effective_pricing(fx["effective_pricing"], PM, "standard", F, labels),
              oe.parse_model_activity(fx["model_activity"], F),
              oe.parse_uptime_recent(fx["uptime_recent"], PM, "standard", F, labels),
              oe.parse_comparison(fx["throughput_comparison"], PM, "standard", F, "p50_throughput_daily", labels),
              oe.parse_rankings_models(fx["rankings_models_week"], "week", F, "tokens_excl_byok"),
              oe.parse_rankings_performance(fx["rankings_performance"], F)]
    for df in frames:
        assert list(df.columns) == oe.TIDY_COLS
        assert len(df) and df.value.notna().all()
        assert (df.published >= df.date).all() and (df.published <= df.fetched).all()


def test_claude_models(fx):
    m = oe.parse_claude_models(fx["models"])
    assert set(m.slug) == {"anthropic/claude-opus-5.5", "anthropic/claude-opus-5.5:batch"}
    assert set(m.permaslug) == {PM} and set(m.variant) == {"standard", "batch"}
    assert m.price_prompt_per_m.iloc[0] == pytest.approx(4.0)


def test_provider_key():
    assert oe.provider_key("amazon-bedrock/us-east-1") == "amazon-bedrock"
    assert oe.provider_key("anthropic/fast") == "anthropic"
    assert oe.provider_key("claude-on-aws") == "claude-on-aws"


def test_v1_endpoints(fx):
    df = oe.parse_v1_endpoints(fx["v1_endpoints"], F, PM)
    assert set(df.provider) == {"anthropic", "amazon-bedrock", "azure"}
    assert one(df, endpoint="amazon-bedrock/us-east-1", metric="price_prompt_per_m").value == pytest.approx(4.4)
    assert one(df, endpoint="anthropic", metric="uptime_last_1d").value == 100
    # null uptime / latency are skipped, not stored as NaN
    assert df[(df.endpoint == "azure/us") & df.metric.str.startswith("uptime")].empty
    assert not (df.metric == "latency_last_30m").any()
    assert (df.date == F).all() and (df.published == F).all()


def test_endpoint_stats(fx, labels):
    df = oe.parse_endpoint_stats(fx["stats_endpoint"], F)
    assert one(df, endpoint="anthropic", metric="p50_latency_ms").value == 2629
    assert one(df, endpoint="anthropic", metric="requests_30m").value == 83331
    assert one(df, endpoint="google-vertex/global", metric="requests_30m").provider == "google-vertex"
    assert set(df.model) == {PM}
    assert labels[BED] == "amazon-bedrock/us-east-1"


def test_effective_pricing(fx, labels):
    df = oe.parse_effective_pricing(fx["effective_pricing"], PM, "standard", F, labels)
    tok = df[df.metric == "tokens_recent"]
    assert one(tok, endpoint="anthropic").value == 194697190217
    assert (tok.date == F).all()
    # an endpoint missing from stats/endpoint falls back to endpointProviderSlugs
    unlabeled = tok[~tok.endpoint.isin(labels.values())]
    assert list(unlabeled.provider) == ["azure"]
    daily = df[df.metric == "effective_input_price_per_m_daily"]
    assert sorted(daily.date.unique()) == [pd.Timestamp("2026-11-10"), pd.Timestamp("2026-11-11")]   # today dropped
    assert (daily.published == daily.date + pd.Timedelta(days=1)).all()
    assert one(df, metric="weighted_input_price_per_m").provider == "all"
    kept = oe.parse_effective_pricing(fx["effective_pricing"], PM, "standard", F, labels, keep_partial=True)
    assert kept[kept.metric == "effective_input_price_per_m_daily"].date.max() == F


def test_model_activity(fx):
    rows = fx["model_activity"]["data"]["analytics"]
    df = oe.parse_model_activity(fx["model_activity"], F)
    assert sorted(df.date.unique()) == [pd.Timestamp("2026-11-10"), pd.Timestamp("2026-11-11")]
    r = [x for x in rows if x["date"].startswith("2026-11-11")][0]
    assert one(df, date=pd.Timestamp("2026-11-11"), metric="tokens").value == r["total_prompt_tokens"] + r["total_completion_tokens"]
    assert one(df, date=pd.Timestamp("2026-11-11"), metric="byok_tokens").value == \
        r["total_byok_prompt_tokens"] + r["total_byok_completion_tokens"]
    assert one(df, date=pd.Timestamp("2026-11-11"), metric="requests").published == F
    assert set(oe.parse_model_activity(fx["model_activity"], F, full=True).metric) >= {"prompt_tokens", "tool_calls"}
    part = oe.parse_model_activity(fx["model_activity"], F, keep_partial=True)
    assert one(part, date=F, metric="tokens").published == F


def test_uptime(fx, labels):
    df = oe.parse_uptime_recent(fx["uptime_recent"], PM, "standard", F, labels)
    assert df.date.max() == pd.Timestamp("2026-11-11")
    assert one(df, endpoint="openrouter", metric="uptime_pct", date=pd.Timestamp("2026-11-11")).provider == "all"
    # an endpoint with no judged minutes has no uptime value but keeps its zero minutes
    dead = df[df.provider == "unknown"]
    assert set(dead.metric) == {"uptime_judged_minutes"} and (dead.value == 0).all()
    h = oe.parse_uptime_hourly(fx["uptime_hourly"], PM, "standard", "anthropic", F)
    assert list(h.date) == [pd.Timestamp("2026-11-12 15:00"), pd.Timestamp("2026-11-12 14:00")]   # 16:00 partial
    assert (h.published == h.date + pd.Timedelta(hours=1)).all()


def test_comparison(fx, labels):
    df = oe.parse_comparison(fx["latency_comparison"], PM, "standard", F, "p50_latency_ms_daily", labels)
    assert set(df.endpoint) == {"anthropic", "amazon-bedrock/us-east-1", "google-vertex/global"}
    assert df.date.max() == pd.Timestamp("2026-11-11") and set(df.source) == {"stats/latency-comparison"}


def test_rankings_models(fx):
    df = oe.parse_rankings_models(fx["rankings_models_week"], "week", F, "tokens_excl_byok")
    assert (df.date == pd.Timestamp("2026-11-11")).all()       # window end, not the row's own last day
    assert not df.model.str.startswith("openai/").any()
    rows = [r for r in fx["rankings_models_week"]["data"] if r["model_permaslug"].startswith("anthropic/")]
    tot = sum(r["total_prompt_tokens"] + r["total_completion_tokens"] for r in rows)
    assert one(df, model="anthropic/*", metric="week_tokens").value == pytest.approx(tot)
    assert one(df, model=PM, variant="standard", metric="week_byok_tokens").value == 2234525478646
    assert one(df, model=PM, variant="standard", metric="week_tokens_excl_byok_change").value == pytest.approx(-0.10353, abs=1e-4)
    plain = oe.parse_rankings_models(fx["rankings_models_week"], "week", F)
    assert not plain.metric.str.contains("byok").any()
    sp = oe.parse_rankings_models(fx["rankings_models_week_spend"], "week", F, "spend")
    assert one(sp, model=PM, variant="standard", metric="week_spend_share").value == pytest.approx(0.152355, rel=1e-4)
    assert one(sp, model="anthropic/*").published == F
    assert oe.parse_rankings_models({"data": []}, "week", F).empty


def test_rankings_performance(fx):
    df = oe.parse_rankings_performance(fx["rankings_performance"], F)
    assert set(df.model) == {PM}
    assert one(df, metric="p50_throughput").value == 189


def test_rsc_charts(fx):
    s = oe.parse_rsc_series(fx["provider_page_html"], "chartData")     # payload split across two pushes
    assert [p["x"][:10] for p in s] == ["2026-11-10", "2026-11-11", "2026-11-12"]
    df = oe.parse_daily_chart(s, F, "amazon-bedrock", "provider page")
    d = df[df.date == pd.Timestamp("2026-11-10")]
    assert one(d, model="(total)").value == pytest.approx(d[d.model != "(total)"].value.sum())
    assert df.date.max() == pd.Timestamp("2026-11-11")
    trimmed = oe._trim_chart(s, "anthropic")
    assert "openai/gpt-6-luna-20260922" not in trimmed[0]["ys"]
    t2 = oe.parse_daily_chart(trimmed, F, "amazon-bedrock", "provider page")
    assert one(t2[t2.date == pd.Timestamp("2026-11-10")], model="(total)").value == one(d, model="(total)").value
    a = oe.parse_rsc_series(fx["author_page_html"], "data")             # skips "data":[] without x/ys
    assert len(a) == 3 and PM in a[0]["ys"]
    with pytest.raises(ValueError):
        oe.parse_rsc_series("<html></html>", "chartData")


def test_provider_shares(fx):
    s = oe.parse_rsc_series(fx["provider_page_html"], "chartData")
    rows = []
    for prov, scale in (("amazon-bedrock", 1.0), ("anthropic", 3.0)):
        rows.append(oe.parse_daily_chart(s, F, prov, "provider page").assign(value=lambda d, k=scale: d.value * k))
    sh = oe.provider_shares(pd.concat(rows))
    assert set(sh.model) == {PM, "anthropic/claude-haiku-5.5-20261007", "anthropic/*"}
    assert sh.groupby(["date", "model"]).value.sum().round(9).eq(1).all()
    assert one(sh, model=PM, provider="anthropic", date=pd.Timestamp("2026-11-10")).value == pytest.approx(0.75)


def _fake_server(fx, calls):
    def get_json(path, **p):
        calls.append(path)
        if path == "/api/v1/models":
            return fx["models"]
        if path == "/api/frontend/v1/rankings/models":
            return fx["rankings_models_week_spend" if p.get("metric") == "spend" else "rankings_models_week"]
        if path.startswith("/api/v1/models/"):
            return fx["v1_endpoints"]
        if path.endswith("/model-activity") and p.get("variant") == "batch":
            ma = json.loads(json.dumps(fx["model_activity"]))
            for r in ma["data"]["analytics"]:
                r["variant"], r["total_prompt_tokens"] = "batch", r["total_prompt_tokens"] // 100
            return ma
        key = path.rsplit("/", 1)[-1].replace("-", "_")
        return fx[{"performance": "rankings_performance", "endpoint": "stats_endpoint"}.get(key, key)]

    def get_text(path):
        calls.append(path)
        return fx["provider_page_html"] if path.startswith("/provider/") else fx["author_page_html"]
    return get_json, get_text


def test_snapshot_offline(fx, monkeypatch, tmp_path):
    calls = []
    gj, gt = _fake_server(fx, calls)
    monkeypatch.setattr(oe, "_get_json", gj)
    monkeypatch.setattr(oe, "_get_text", gt)
    df = oe.snapshot(asof="2026-11-12", out_dir=str(tmp_path), top_n=1, providers=("amazon-bedrock", "anthropic"),
                     hourly=True, verbose=False)
    files = sorted(os.listdir(tmp_path))
    assert len(files) == 9 and all(f.startswith("2026-11-12_") for f in files)
    assert "2026-11-12_claude_endpoints.csv" in files
    # top_n=1 picks the model with most weekly tokens; batch activity fetched too
    assert set(df[df.metric == "p50_latency_ms"].model) == {PM}
    assert set(df[df.source == "stats/model-activity"].variant) == {"standard", "batch"}
    assert {"tokens_provider_share", "tokens_recent_provider_share", "requests_30m_provider_share",
            "uptime_pct_hourly", "week_spend_share", "p50_throughput_daily"} <= set(df.metric)
    # provider rows keep only Anthropic models, Others and (total)
    pp = df[df.source == "provider page"]
    assert pp.model.map(lambda m: m.startswith("anthropic/") or m in ("Others", "(total)")).all()
    assert calls.count("/api/frontend/v1/stats/uptime-hourly") == 3
    # CSV round trip and offline rebuild from the raw JSON files give the same rows
    back = oe.load_csv(os.path.join(tmp_path, "2026-11-12_claude_endpoints.csv"))
    assert len(back) == len(df) and list(back.columns) == oe.TIDY_COLS
    re = oe.rebuild("2026-11-12", out_dir=str(tmp_path))
    key = ["source", "model", "variant", "provider", "endpoint", "metric", "date"]
    m = df.merge(re, on=key, suffixes=("", "_re"))
    assert len(m) == len(df) == len(re)
    assert (m.value - m.value_re).abs().max() == 0
    assert back.value.sub(df.value).abs().div(df.value.abs().clip(lower=1)).max() < 1e-5
