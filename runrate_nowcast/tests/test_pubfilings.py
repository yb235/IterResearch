"""Sub-task O: filing numbers used as evidence once public ("pub" rule, config.USE_FILINGS = True).

1. Unchanged targets are identical: a target whose vintage precedes every filing row's publication gets
   the same forecast with and without filings (live, model A frame, the run_backtest call), and in the
   saved reports every pub target without a filing row in its evidence reproduces sub-task N's mean.
2. Filing rows in the evidence are never published after the vintage (nor with or after the target),
   for every target and both vintage rules; rr_2026_09_s1a and rr_2026_10_s1a (same S-1/A) never see
   each other. The rows recorded in the saved pub reports satisfy the same check.
3. Answer independence: tripling the target's value leaves the pub forecast's mean and sd unchanged.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from runrate_nowcast import backtest, calibrate, config, data, lower_rule

ASOF = "2026-11-13"
FIRST_FILING_PUBLIC = pd.Timestamp("2026-09-30")       # Fortune's draft-S-1 report
S1A2 = ("rr_2026_09_s1a", "rr_2026_10_s1a")             # both from S-1/A No. 2, published 2026-11-10
MODELS = ("A", "B", "E", "G", "R")


@pytest.fixture
def pub(monkeypatch):
    """The pub evidence rule with the primary lower-claim rule, restored after the test."""
    monkeypatch.setattr(config, "USE_FILINGS", True)
    monkeypatch.setattr(config, "LOWER_UPPER_RATIO", lower_rule.setting("iv115"))


@pytest.fixture(scope="module")
def claims_a():
    df = data.load_all(indicators=False)
    return df[(df.segment == "total") & (df.metric != "share")]


@pytest.fixture(scope="module")
def params():
    return calibrate.load_params()


def _targets(df):
    full = data.select(df, filings=True)
    return full[full.tier.isin(backtest.TARGET_TIERS) & (full.kind == "point")
                & full.metric.isin(["run_rate", "revenue"]) & (full.is_forecast == 0) & (full.segment == "total")]


def _forecast(df, tid, rule, p):
    t = _targets(df)
    row = next(t[t.obs_id == tid].itertuples())
    v = backtest.target_vintage(row, rule)
    mu, sd, n, ev = backtest.prepublication_forecast(df, row, p, True, 3, None, "profile", vintage=v)
    return mu, sd, ev


# --------------------------------------------------------------------------- 1. unchanged targets
def test_targets_do_not_depend_on_the_evidence_rule(monkeypatch, claims_a):
    a = set(_targets(claims_a).obs_id)
    monkeypatch.setattr(config, "USE_FILINGS", True)
    assert set(_targets(claims_a).obs_id) == a


@pytest.mark.parametrize("tid,rule", [("rr_2026_05", "publication"), ("rev_2026q2", "period_end")])
def test_pre_filing_target_identical(monkeypatch, claims_a, params, tid, rule):
    monkeypatch.setattr(config, "LOWER_UPPER_RATIO", lower_rule.setting("iv115"))
    monkeypatch.setattr(config, "USE_FILINGS", False)
    mu0, sd0, ev0 = _forecast(claims_a, tid, rule, params)
    monkeypatch.setattr(config, "USE_FILINGS", True)
    mu1, sd1, ev1 = _forecast(claims_a, tid, rule, params)
    assert (ev1.filing != 0).sum() == 0
    assert list(ev0.obs_id) == list(ev1.obs_id)
    assert mu1 == mu0 and sd1 == sd0


def _report(prefix, model, variant, rule):
    f = os.path.join(config.OUTPUT_DIR, f"{prefix}_{model}_{variant}_{rule}_{ASOF}.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


@pytest.mark.parametrize("variant", ["iv115", "point"])
@pytest.mark.parametrize("rule", ["publication", "period_end"])
@pytest.mark.parametrize("model", MODELS)
def test_reports_unchanged_targets_match_no_filing_run(model, variant, rule):
    nf, pb = _report("forecast_interval", model, variant, rule), _report("forecast_pubfilings", model, variant, rule)
    if nf is None or pb is None:
        pytest.skip("report not written yet")
    assert set(nf.target) == set(pb.target)
    j = nf.merge(pb, on="target", suffixes=("_nf", "_pub"))
    same = j[j.n_filing_evidence == 0]
    assert len(same) >= 6
    assert (same.forecast_pub == same.forecast_nf).all()            # bit-identical mean
    # run-rate intervals reproduce exactly; revenue sds move only by the uncommitted model.py fix (made after
    # N's runs): noise (s y)^2 -> (s mu)^2, s = gaap log_sd, y = actual, mu = forecast
    rr, rev = same[same.metric_nf == "run_rate"], same[same.metric_nf == "revenue"]
    assert (rr.q10_pub == rr.q10_nf).all() and (rr.q90_pub == rr.q90_nf).all()
    s = config.TIERS["gaap"]["log_sd"]
    sd_nf, sd_pub = (rev.q90_nf - rev.q10_nf) / 2.5632, (rev.q90_pub - rev.q10_pub) / 2.5632
    pred = (sd_nf ** 2 + s ** 2 * (rev.forecast_nf ** 2 - rev.actual_nf ** 2)) ** 0.5
    assert ((sd_pub / pred - 1).abs() < 1e-9).all()
    assert (j[j.n_filing_evidence > 0].vintage_pub >= str(FIRST_FILING_PUBLIC.date())).all()


# --------------------------------------------------------------------------- 2. no filing row after the vintage
@pytest.mark.parametrize("rule", ["publication", "period_end"])
def test_filing_rows_in_evidence_public_by_vintage(monkeypatch, rule):
    monkeypatch.setattr(config, "USE_FILINGS", True)
    df = data.load_all(indicators=False)
    raw = data.load().set_index("obs_id")
    seen = 0
    for tg in _targets(df).itertuples():
        v = backtest.target_vintage(tg, rule)
        ev = data.select(df, vintage=v)
        ev = ev[ev.origin_id != tg.origin_id]
        backtest.check_no_leak(ev, v, tg)
        fil = raw.loc[ev[ev.filing != 0].obs_id]
        assert (fil.published <= v).all()
        assert (raw.loc[ev.obs_id].published < raw.loc[tg.obs_id, "published"]).all()
        if v < FIRST_FILING_PUBLIC:
            assert not len(fil)
        if tg.obs_id in S1A2:
            other = [x for x in S1A2 if x != tg.obs_id][0]
            assert other not in set(ev.obs_id)
        seen += len(fil)
    assert seen > 0                                       # the rule did let filing rows in


def test_reports_recorded_filing_rows_public_by_vintage():
    raw = data.load().set_index("obs_id")
    files = [f for f in os.listdir(config.OUTPUT_DIR) if f.startswith("forecast_pubfilings_") and f.endswith(f"_{ASOF}.csv")]
    if not files:
        pytest.skip("no pub reports yet")
    n = 0
    for f in files:
        for r in pd.read_csv(os.path.join(config.OUTPUT_DIR, f)).itertuples():
            if pd.isna(r.filing_evidence):
                continue
            for item in str(r.filing_evidence).split(";"):
                oid, pubd = item.split("@")
                assert raw.loc[oid, "filing"] != 0
                assert raw.loc[oid, "published"] == pd.Timestamp(pubd) <= pd.Timestamp(r.vintage)
                assert raw.loc[oid, "published"] < raw.loc[r.target, "published"]
                assert not (r.target in S1A2 and oid in S1A2)
                n += 1
    assert n > 0


# --------------------------------------------------------------------------- 3. answer independence
@pytest.mark.parametrize("tid,rule", [("rr_2026_10_s1a", "publication"), ("rev_2026q3", "publication")])
def test_pub_forecast_ignores_target_value(pub, claims_a, params, tid, rule):
    mu0, sd0, ev0 = _forecast(claims_a, tid, rule, params)
    assert (ev0.filing != 0).any()                      # a target the pub rule changes
    x3 = claims_a.copy()
    x3.loc[x3.obs_id == tid, "value"] *= 3
    mu1, sd1, _ = _forecast(x3, tid, rule, params)
    assert mu1 == pytest.approx(mu0, abs=1e-12)
    assert sd1 == pytest.approx(sd0, abs=1e-12)
