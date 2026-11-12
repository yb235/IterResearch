"""Offline tests for prediction_markets.py: mocked API JSON, as-of logic, ladder -> distribution.

Run: python -m pytest runrate_nowcast/tests/test_prediction_markets.py -q
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from runrate_nowcast import prediction_markets as pm

MS = 1000
T0 = 1_767_225_600 * MS          # 2026-01-01T00:00Z in ms


class FakeGet:
    """Stands in for http_get_json. Each route: (url_suffix, params_subset or None, response or Exception)."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, params=None):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        self.calls.append((url, params))
        for suffix, want, resp in self.routes:
            if url.endswith(suffix) and all(params.get(k) == v for k, v in (want or {}).items()):
                if isinstance(resp, Exception):
                    raise resp
                return json.loads(json.dumps(resp))       # fresh copy, JSON types only
        raise pm.FetchError(url, 404, "no route")


def _ts(ms):
    return pd.Timestamp(ms, unit="ms", tz="UTC")


# --------------------------------------------------------------------------- parse_threshold
@pytest.mark.parametrize("text,expected", [
    (">= $80B", ("ge", 80.0)),
    ("≥$100B", ("ge", 100.0)),
    ("Will Anthropic Annual Recurring Revenue be under $90bn end of 2026?", ("le", 90.0)),
    ("Anthropic 60B revenue annual run rate on July 6, 2026?", ("ge", 60.0)),
    ("$1.5T", ("ge", 1500.0)),
    ("↓ $1.025T", ("le", 1025.0)),
    ("$500M", ("ge", 0.5)),
    ("Above 1,200 billion", ("ge", 1200.0)),
    ("Anthropic highest annualized revenue run rate by end of June 2026", ("", math.nan)),
])
def test_parse_threshold_single(text, expected):
    kind, val, upper = pm.parse_threshold(text)
    assert kind == expected[0]
    assert (math.isnan(val) and math.isnan(expected[1])) or val == pytest.approx(expected[1])
    assert math.isnan(upper)


def test_parse_threshold_ranges_skip_dates():
    assert pm.parse_threshold("$50-60B") == ("between", 50.0, 60.0)
    assert pm.parse_threshold("$50B to $60B") == ("between", 50.0, 60.0)
    assert pm.parse_threshold("2026-12-31: $1T-1.5T") == ("between", 1000.0, 1500.0)


# --------------------------------------------------------------------------- Manifold
def _manifold_binary_routes():
    market = {"id": "c1", "slug": "anthropic-60b-revenue-on-july-6-202", "outcomeType": "BINARY",
              "question": "Anthropic 60B revenue annual run rate on July 6, 2026?", "createdTime": T0,
              "isResolved": True, "resolution": "YES", "resolutionTime": T0 + 50 * 86400 * MS}
    b1 = {"id": "b1", "contractId": "c1", "createdTime": T0 + 1 * 86400 * MS, "probBefore": 0.40, "probAfter": 0.50}
    b2 = {"id": "b2", "contractId": "c1", "createdTime": T0 + 2 * 86400 * MS, "probBefore": 0.50, "probAfter": 0.60}
    b3 = {"id": "b3", "contractId": "c1", "createdTime": T0 + 3 * 86400 * MS, "probBefore": 0.60, "probAfter": 0.70}
    bx = {"id": "bx", "contractId": "c1", "createdTime": T0 + 4 * 86400 * MS, "probBefore": 0.7,
          "probAfter": 0.1, "isCancelled": True}
    return [
        ("/v0/slug/anthropic-60b-revenue-on-july-6-202", None, market),
        ("/v0/bets", {"contractId": "c1", "before": "b2"}, [b1]),
        ("/v0/bets", {"contractId": "c1"}, [bx, b3, b2]),          # newest first, full page (limit=3)
    ]


def test_manifold_binary_paginates_and_adds_initial_and_resolution():
    get = FakeGet(_manifold_binary_routes())
    h = pm.manifold_history("https://manifold.markets/Bayesian/anthropic-60b-revenue-on-july-6-202?r=x",
                            get=get, bets=pm.manifold_bets("c1", get=get, limit=3),
                            market=pm.manifold_market("Bayesian/anthropic-60b-revenue-on-july-6-202", get=get))
    assert list(h.columns) == pm.HIST_COLS
    assert list(h["source"]) == ["initial", "bet", "bet", "bet", "resolution"]
    assert list(h["probability"]) == pytest.approx([0.40, 0.50, 0.60, 0.70, 1.0])   # cancelled bet ignored
    assert h["timestamp"].iloc[0] == _ts(T0)
    assert (h["kind"] == "ge").all() and (h["threshold"] == 60.0).all()
    assert (h["market"] == "anthropic-60b-revenue-on-july-6-202").all()
    # second page requested with before = oldest bet id of the first page
    assert ("https://api.manifold.markets/v0/bets", {"contractId": "c1", "limit": 3, "before": "b2"}) in get.calls


def test_manifold_bets_stops_on_repeated_page():
    page = [{"id": "b1", "createdTime": T0, "probAfter": 0.5}]
    get = FakeGet([("/v0/bets", None, page)])
    assert [b["id"] for b in pm.manifold_bets("c1", get=get, limit=1)] == ["b1"]
    assert len(get.calls) == 2


def _manifold_ladder():
    a80 = {"id": "a80", "text": ">= $80B", "createdTime": T0, "resolution": "YES",
           "resolutionTime": T0 + 30 * 86400 * MS}
    a100 = {"id": "a100", "text": ">= $100B", "createdTime": T0 + 5 * 86400 * MS}
    a120 = {"id": "a120", "text": ">= $120B", "createdTime": T0}
    market = {"id": "c2", "slug": "will-anthropic-achieve-a-revenue-ru", "outcomeType": "MULTIPLE_CHOICE",
              "shouldAnswersSumToOne": False, "question": "Anthropic annualized revenue run rate by end of 2026?",
              "createdTime": T0, "answers": [a80, a100, a120]}
    d = 86400 * MS
    bets = [
        {"id": "x1", "answerId": "a80", "createdTime": T0 + 1 * d, "probBefore": 0.50, "probAfter": 0.70},
        {"id": "x2", "answerId": "a120", "createdTime": T0 + 2 * d, "probBefore": 0.20, "probAfter": 0.25},
        {"id": "x3", "answerId": "a100", "createdTime": T0 + 6 * d, "probBefore": 0.40, "probAfter": 0.45},
        {"id": "x4", "answerId": "a80", "createdTime": T0 + 10 * d, "probBefore": 0.70, "probAfter": 0.90},
        {"id": "x5", "answerId": "a120", "createdTime": T0 + 12 * d, "probBefore": 0.25, "probAfter": 0.30},
        {"id": "x6", "answerId": "gone", "createdTime": T0 + 13 * d, "probBefore": 0.1, "probAfter": 0.2},
    ]
    return market, bets


def test_manifold_multi_answer_ladder():
    market, bets = _manifold_ladder()
    h = pm.manifold_history("will-anthropic-achieve-a-revenue-ru", market=market, bets=bets)
    assert set(h["outcome"]) == {">= $80B", ">= $100B", ">= $120B"}      # unknown answer id dropped
    init = h[h["source"] == "initial"].set_index("outcome")
    assert init.loc[">= $100B", "timestamp"] == _ts(T0 + 5 * 86400 * MS)  # answer's own creation time
    assert init.loc[">= $100B", "probability"] == pytest.approx(0.40)
    res = h[h["source"] == "resolution"]
    assert list(res["outcome"]) == [">= $80B"] and res["probability"].iloc[0] == 1.0
    assert dict(zip(h["outcome"], h["threshold"])) == {">= $80B": 80.0, ">= $100B": 100.0, ">= $120B": 120.0}


def test_manifold_sum_to_one_market_level_resolution():
    market = {"id": "c3", "slug": "s", "outcomeType": "MULTIPLE_CHOICE", "createdTime": T0, "isResolved": True,
              "resolution": "a2", "resolutionTime": T0 + 9 * MS,
              "answers": [{"id": "a1", "text": "$50-60B"}, {"id": "a2", "text": "$60-70B"}]}
    bets = [{"id": "z", "answerId": "a1", "createdTime": T0 + MS, "probBefore": 0.5, "probAfter": 0.4}]
    h = pm.manifold_history("s", market=market, bets=bets)
    res = h[h["source"] == "resolution"].set_index("outcome")["probability"]
    assert res.to_dict() == {"$50-60B": 0.0, "$60-70B": 1.0}
    assert set(h["kind"]) == {"between"}


# --------------------------------------------------------------------------- Polymarket
def test_polymarket_event_with_string_encoded_fields_and_window_fallback():
    ev = [{"slug": "will-anthropics-valuation-hit-by-december-31", "markets": [
        {"slug": "val-2t", "question": "Will Anthropic's valuation hit $2T by December 31?", "groupItemTitle": "$2T",
         "outcomes": "[\"Yes\", \"No\"]", "clobTokenIds": "[\"tok2y\", \"tok2n\"]",
         "startDate": "2026-06-01T00:00:00Z", "endDate": "2026-12-31T00:00:00Z"},
        {"slug": "val-3t", "question": "Will Anthropic's valuation hit $3T by December 31?", "groupItemTitle": "$3T",
         "outcomes": ["No", "Yes"], "clobTokenIds": ["tok3n", "tok3y"],
         "startDate": "2026-06-01T00:00:00Z", "endDate": "2026-12-31T00:00:00Z"},
    ]}]
    get = FakeGet([
        ("/events", {"slug": "will-anthropics-valuation-hit-by-december-31"}, ev),
        ("/prices-history", {"market": "tok2y", "interval": "max"},
         {"history": [{"t": 1780000000, "p": 0.46}, {"t": 1780086400, "p": "0.48"}]}),
        ("/prices-history", {"market": "tok3y", "interval": "max"}, {"history": []}),
        ("/prices-history", {"market": "tok3y"}, {"history": [{"t": 1780000000, "p": 0.2}]}),
    ])
    h = pm.polymarket_history("will-anthropics-valuation-hit-by-december-31", get=get)
    assert len(h) == 3
    two = h[h["outcome"] == "$2T"]
    assert list(two["probability"]) == pytest.approx([0.46, 0.48])
    assert (two["threshold"] == 2000.0).all() and (two["kind"] == "ge").all()
    assert h[h["outcome"] == "$3T"]["probability"].iloc[0] == pytest.approx(0.2)   # Yes token, not first token
    window = [p for u, p in get.calls if u.endswith("/prices-history") and "startTs" in p]
    assert window and window[0]["market"] == "tok3y" and "interval" not in window[0]
    assert h["timestamp"].iloc[0] == pd.Timestamp(1780000000, unit="s", tz="UTC")


def test_polymarket_market_slug_fallback_and_named_outcomes():
    get = FakeGet([
        ("/events", None, []),
        ("/markets", {"slug": "anthropic-vs-openai"}, [{"slug": "anthropic-vs-openai", "question": "Higher valuation?",
                                                       "outcomes": "[\"Anthropic\", \"OpenAI\"]",
                                                       "clobTokenIds": "[\"tA\", \"tO\"]"}]),
        ("/prices-history", {"market": "tA"}, {"history": [{"t": 1780000000, "p": 0.86}]}),
    ])
    h = pm.polymarket_history("anthropic-vs-openai", get=get)
    assert list(h["outcome"]) == ["Higher valuation?: Anthropic"]
    assert h["probability"].iloc[0] == pytest.approx(0.86)


# --------------------------------------------------------------------------- Kalshi
def test_kalshi_market_candles_mid_fallback_and_settlement():
    mk = {"market": {"ticker": "KXIPOANTHROPIC-DATE-26DEC01", "event_ticker": "KXIPOANTHROPIC-DATE",
                     "title": "Anthropic IPO before Dec 1?", "open_time": "2026-09-01T00:00:00Z",
                     "close_time": "2026-09-04T00:00:00Z", "expiration_time": "2026-09-05T00:00:00Z",
                     "result": "no"}}
    candles = {"candlesticks": [
        {"end_period_ts": 1788307200, "price": {"close": 55}, "yes_bid": {"close": 50}, "yes_ask": {"close": 60}},
        {"end_period_ts": 1788393600, "price": {"close": None}, "yes_bid": {"close_dollars": "0.40"},
         "yes_ask": {"close_dollars": "0.50"}},
        {"end_period_ts": 1788480000, "price": {"close_dollars": "0.30"}},
    ]}
    get = FakeGet([
        ("/markets/KXIPOANTHROPIC-DATE-26DEC01", None, mk),
        ("/events/KXIPOANTHROPIC-DATE", None, {"event": {"series_ticker": "KXIPOANTHROPIC"}}),
        ("/series/KXIPOANTHROPIC/markets/KXIPOANTHROPIC-DATE-26DEC01/candlesticks", {"period_interval": 1440}, candles),
    ])
    h = pm.kalshi_history("KXIPOANTHROPIC-DATE-26DEC01", get=get, now="2026-11-12")
    assert list(h["source"]) == ["candle_close", "candle_mid", "candle_close", "resolution"]
    assert list(h["probability"]) == pytest.approx([0.55, 0.45, 0.30, 0.0])
    assert h["timestamp"].iloc[0] == pd.Timestamp(1788307200, unit="s", tz="UTC")     # end of candle
    assert h["timestamp"].iloc[-1] == pd.Timestamp("2026-09-05", tz="UTC")             # latest of close/expiry


def test_kalshi_event_ticker_falls_back_to_event_markets():
    m = {"ticker": "EVT-A", "event_ticker": "EVT", "yes_sub_title": "Above $100B",
         "open_time": "2026-09-01T00:00:00Z", "close_time": "2026-09-02T00:00:00Z"}
    get = FakeGet([
        ("/markets/EVT", None, pm.FetchError("x", 404, "not found")),
        ("/markets", {"event_ticker": "EVT"}, {"markets": [m], "cursor": ""}),
        ("/events/EVT", None, {"event": {}}),                                   # no series -> ticker prefix
        ("/series/EVT/markets/EVT-A/candlesticks", None,
         {"candlesticks": [{"end_period_ts": 1788307200, "price": {"close": 62}}]}),
    ])
    h = pm.kalshi_history("EVT", get=get, now="2026-11-12")
    assert h["threshold"].iloc[0] == 100.0 and h["kind"].iloc[0] == "ge"
    assert h["probability"].iloc[0] == pytest.approx(0.62)


def test_kalshi_non_404_error_propagates():
    get = FakeGet([("/markets/T", None, pm.FetchError("x", None, "Tunnel connection failed: 403"))])
    with pytest.raises(pm.FetchError):
        pm.kalshi_history("T", get=get)


# --------------------------------------------------------------------------- prob_asof
def _toy_history():
    rows = [
        ("2026-01-01T00:00:00Z", ">= $80B", 0.50),
        ("2026-01-05T12:00:00Z", ">= $80B", 0.70),
        ("2026-01-10T00:00:00Z", ">= $80B", 0.95),          # must never be visible before Jan 10
        ("2026-01-03T00:00:00Z", ">= $100B", 0.30),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "outcome", "probability"])
    df["platform"], df["market"], df["source"] = "manifold", "m", "bet"
    df["kind"], df["threshold"], df["upper"] = "ge", df["outcome"].str.extract(r"(\d+)")[0].astype(float), np.nan
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[pm.HIST_COLS]


def test_prob_asof_uses_only_past_rows():
    h = _toy_history()
    snap = pm.prob_asof(h, "2026-01-09T23:59:59").set_index("outcome")       # naive -> UTC
    assert snap.loc[">= $80B", "probability"] == 0.70
    assert snap.loc[">= $100B", "probability"] == 0.30
    assert snap["timestamp"].max() <= pd.Timestamp("2026-01-09T23:59:59", tz="UTC")
    at = pm.prob_asof(h, "2026-01-10T00:00:00Z").set_index("outcome")       # 'at or before' includes ts
    assert at.loc[">= $80B", "probability"] == 0.95
    assert pm.prob_asof(h, "2025-12-31").empty
    lag = pm.prob_asof(h, "2026-01-10T00:00:00Z", min_lag=pd.Timedelta("1D")).set_index("outcome")
    assert lag.loc[">= $80B", "probability"] == 0.70
    stale = pm.prob_asof(h, "2026-01-09", max_age=pd.Timedelta("5D"))
    assert set(stale["outcome"]) == {">= $80B"}                                # $100B row is 6 days old
    assert pm.prob_asof(h.sample(frac=1, random_state=0), "2026-01-09").equals(pm.prob_asof(h, "2026-01-09"))


def test_prob_asof_reads_csv_strings(tmp_path):
    p = tmp_path / "h.csv"
    _toy_history().to_csv(p, index=False)
    snap = pm.prob_asof(pd.read_csv(p), pd.Timestamp("2026-01-06", tz="UTC")).set_index("outcome")
    assert snap.loc[">= $80B", "probability"] == 0.70


# --------------------------------------------------------------------------- implied_distribution
def test_implied_distribution_log_interpolation_and_tails():
    x, p = [50, 80, 100, 120], [0.99, 0.90, 0.60, 0.30]
    d = pm.implied_distribution(x, p)
    assert d.median == pytest.approx(100 * 1.2 ** (1 / 3))      # S = 0.5 one third of the way 100 -> 120 in log
    assert d.q10 == pytest.approx(80.0)                          # S(80) = 0.9 exactly
    assert math.isnan(d.q90) and d.flags["q90"] == "above_ladder"
    assert d.flags["median"] == "ok" and d.max_adjustment == 0.0
    assert pm.implied_distribution(x, p, tail="clamp").q90 == 120.0
    assert pm.implied_distribution(x, p, tail="extrapolate").q90 == pytest.approx(100 * 1.2 ** (5 / 3))
    lo = pm.implied_distribution([80, 100], [0.4, 0.2])          # median < 80 and q90 > 100: bounds only
    assert math.isnan(lo.median) and math.isnan(lo.q10) and math.isnan(lo.q90)
    assert lo.flags == {"q10": "below_ladder", "median": "below_ladder", "q90": "above_ladder"}
    assert pm.implied_distribution([80, 100], [0.4, 0.2], tail="extrapolate").median < 80
    lin = pm.implied_distribution(x, p, scale="linear")
    assert lin.median == pytest.approx(100 + 20 / 3)


def test_implied_distribution_fixes_non_monotone_ladder():
    d = pm.implied_distribution([80, 120, 100, 60, 100], [0.9, 0.2, 0.5, 0.95, 0.7])
    # duplicates at 100 average to 0.6; then [0.95, 0.9, 0.6, 0.2] is already monotone
    assert list(d.thresholds) == [60, 80, 100, 120]
    assert d.raw == pytest.approx([0.95, 0.9, 0.6, 0.2])
    v = pm.implied_distribution([80, 100, 120, 150], [0.9, 0.5, 0.6, 0.2], weights=[1, 1, 1, 1])
    assert v.survival == pytest.approx([0.9, 0.55, 0.55, 0.2])
    assert np.all(np.diff(v.survival) <= 1e-12)
    assert v.max_adjustment == pytest.approx(0.05)
    w = pm.implied_distribution([80, 100, 120], [0.9, 0.5, 0.6], weights=[1, 3, 1])
    assert w.survival[1] == pytest.approx(0.525) and w.survival[2] == pytest.approx(0.525)
    assert np.all((v.survival >= 0) & (v.survival <= 1))


def test_implied_distribution_anchors_close_the_ladder():
    d = pm.implied_distribution([80, 100], [0.8, 0.4], lower=65, upper=200)
    assert d.q10 == pytest.approx(math.sqrt(65 * 80))             # halfway 65 (S=1) -> 80 (S=0.8) in log
    assert d.median == pytest.approx(80 * 1.25 ** 0.75)           # S from 0.8 to 0.4: 0.5 at 3/4
    assert d.q90 == pytest.approx(100 * 2 ** 0.75)
    assert set(d.flags.values()) == {"ok"}
    # an anchor dominates a contradicting market price below it
    e = pm.implied_distribution([60, 80], [0.7, 0.6], lower=65)
    assert e.survival[0] == pytest.approx(1.0) and e.survival[1] == pytest.approx(1.0, abs=1e-5)


def test_implied_distribution_input_checks():
    with pytest.raises(ValueError):
        pm.implied_distribution([0, 10], [0.9, 0.1])
    with pytest.raises(ValueError):
        pm.implied_distribution([1, 2], [0.5])
    e = pm.implied_distribution([], [])
    assert math.isnan(e.median) and e.flags["median"] == "empty"
    d = pm.implied_distribution([80, np.nan, 100], [1.2, 0.5, -0.1])     # clipped, NaN dropped
    assert d.raw == pytest.approx([1.0, 0.0])


def test_buckets_to_ladder():
    labels = ["< $50B", "$50-80B", "$80-100B", "> $100B"]
    parsed = [pm.parse_threshold(t) for t in labels]
    kinds = [k for k, _, _ in parsed]
    lo = [v for _, v, _ in parsed]
    hi = [u for _, _, u in parsed]
    xs, s = pm.buckets_to_ladder(kinds, lo, hi, [0.1, 0.3, 0.4, 0.2])
    assert list(xs) == [50, 80, 100]
    assert s == pytest.approx([0.9, 0.6, 0.2])
    _, s2 = pm.buckets_to_ladder(kinds, lo, hi, [0.2, 0.6, 0.8, 0.4])     # renormalised
    assert s2 == pytest.approx([0.9, 0.6, 0.2])


# --------------------------------------------------------------------------- end to end on a mocked ladder
def test_ladder_asof_and_distribution_path_no_lookahead():
    market, bets = _manifold_ladder()
    h = pm.manifold_history("will-anthropic-achieve-a-revenue-ru", market=market, bets=bets)
    d = 86400 * MS
    xs, ps = pm.ladder_asof(h, _ts(T0 + 3 * d))
    assert dict(zip(xs, ps)) == {80.0: 0.70, 120.0: 0.25}        # $100B answer not created yet
    path = pm.distribution_path(h, [_ts(T0 + 3 * d), _ts(T0 + 11 * d), _ts(T0 + 40 * d)])
    # day 11: 80 -> 0.90 (bet day 10), 100 -> 0.45, 120 -> 0.25 (day-12 bet not visible)
    exp = 80 * (100 / 80) ** ((0.90 - 0.5) / (0.90 - 0.45))
    assert path["median"].iloc[1] == pytest.approx(exp)
    assert path["n_thresholds"].tolist() == [2, 3, 3]
    # day 40: $80B resolved YES (1.0); median between 80 and 100
    assert 80 < path["median"].iloc[2] < 100
    floor = pm.distribution_path(h, [_ts(T0 + 3 * d)], lower=lambda t: 47.0)
    assert floor["flag_q10"].iloc[0] == "ok"


def test_ladder_asof_converts_le_outcomes():
    h = _toy_history()
    h.loc[h["outcome"] == ">= $100B", ["outcome", "kind"]] = ["under $100B", "le"]
    xs, ps = pm.ladder_asof(h, "2026-01-06")
    assert dict(zip(xs, ps)) == pytest.approx({80.0: 0.70, 100.0: 0.70})


# --------------------------------------------------------------------------- catalog
def test_catalog_file_is_well_formed():
    c = pd.read_csv(pm.CATALOG_PATH, dtype=str).fillna("")
    cols = ["platform", "market_id_or_slug", "url", "question", "metric", "threshold", "unit", "resolution_date",
            "resolution_criteria", "created_date", "close_date", "status_or_resolution", "snapshot_probability",
            "snapshot_date", "evidence_source", "notes"]
    assert list(c.columns) == cols
    assert set(c["metric"]) <= {"run_rate", "revenue", "valuation", "ipo", "other"}
    probs = pd.to_numeric(c["snapshot_probability"].replace("", np.nan))
    assert probs.dropna().between(0, 1).all()
    for col in ("snapshot_date", "resolution_date", "created_date", "close_date"):
        vals = c.loc[c[col] != "", col]
        assert pd.to_datetime(vals, format="%Y-%m-%d").notna().all()
    assert (c.loc[probs.notna(), "evidence_source"] != "").all()
    assert (c["metric"] == "run_rate").sum() >= 3


def test_fetch_catalog_records_failures_without_writing(tmp_path):
    cat = tmp_path / "cat.csv"
    pd.DataFrame([
        {"platform": "manifold", "market_id_or_slug": "u/some-market", "metric": "run_rate"},
        {"platform": "manifold", "market_id_or_slug": "u/some-market", "metric": "run_rate"},   # duplicate
        {"platform": "manifold", "market_id_or_slug": "event/x", "metric": "valuation"},          # not a market
        {"platform": "metaculus", "market_id_or_slug": "38405", "metric": "other"},              # no fetcher
        {"platform": "kalshi", "market_id_or_slug": "", "metric": "ipo"},
    ]).to_csv(cat, index=False)
    assert pm.catalog_targets(str(cat)) == [("manifold", "u/some-market")]
    blocked = FakeGet([("", None, pm.FetchError("https://api.manifold.markets/v0/slug/some-market", None,
                                                "Tunnel connection failed: 403 Forbidden"))])
    out = tmp_path / "prices.csv"
    hist, status = pm.fetch_catalog(str(cat), str(out), get=blocked, verbose=False)
    assert hist.empty and not out.exists()
    assert status == [{"platform": "manifold", "id": "u/some-market", "ok": False, "rows": 0,
                       "error": status[0]["error"]}]
    assert "403" in status[0]["error"]


def test_with_params_drops_none():
    assert pm._with_params("https://x/y", {"a": 1, "b": None}) == "https://x/y?a=1"
    assert pm._with_params("https://x/y?z=2", {"a": "s p"}) == "https://x/y?z=2&a=s+p"
