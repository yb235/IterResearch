"""Frozen Q4 2026 forecasts (scripts/register_q4.py): frozen model list, filings never in the
evidence, combination rule, quantiles on the registry's convention, and no registry write by default.

Offline. The slow parts are run with the fast settings (no refit, no crossing profile, fixed panel
noise), except one production-path check on model A.
Run: python -m pytest runrate_nowcast/tests -q
"""
import hashlib
import importlib.util
import math
import os

import numpy as np
import pandas as pd
import pytest

from runrate_nowcast import altdata, calibrate, config, data, registry
from runrate_nowcast import panel_model as pm
from runrate_nowcast.model import Model

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "register_q4.py")
_spec = importlib.util.spec_from_file_location("register_q4", SCRIPT)
rq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rq)

RR_1231 = "runrate_2026-12-31"
S1A_OCT = "rr_2026_10_s1a"                        # the S-1/A $81B: a filing row, never an input
NOISE = {"sigma_l": 0.004, "sigma_g": 0.01, "sigma_b": 0.002, "growth_reversion": 0.1}
EXPECTED_INDICATORS = {"A": set(), "B": {"npm_anthropic_sdk", "pypi_anthropic", "ramp_token_spend_anthropic"},
                       "E": {"openrouter_anthropic_tokens"}, "G": {"github_repo_share_claude"},
                       "R": {"ramp_adoption_anthropic"}}


def _fake_claim(filing: int, value: float = 500.0, **kw) -> pd.DataFrame:
    """A run-rate claim published the day before the vintage, far from every other claim."""
    row = dict(obs_id="fake_s1", origin_id="fake_s1", metric="run_rate", kind="point", value=value,
               ref_date=pd.Timestamp("2026-11-01"), period_start=pd.NaT, period_end=pd.NaT, definition="t28x13",
               date_sd_days=0.0, published=pd.Timestamp("2026-11-11"), tier="official", source="", url="",
               include=1, notes="test", segment="total", indicator="", is_forecast=0, filing=filing,
               date_lo=pd.NaT, date_hi=pd.NaT, date_kind="")
    row.update(kw)
    return pd.DataFrame([row])


@pytest.fixture(scope="module")
def table_a():
    with rq.variant_catalog("A"):
        return rq.variant_data("A")


@pytest.fixture(scope="module")
def fast_a(table_a):
    df, p = table_a
    return rq.statespace_forecasts("A", df=df, params=p, refit=False, crossing=None)


def _same(a: dict, b: dict) -> bool:
    """Records (or {quantity: record}) equal, NaN == NaN."""
    return pd.DataFrame(a if isinstance(next(iter(a.values())), dict) else {"r": a}).equals(
        pd.DataFrame(b if isinstance(next(iter(b.values())), dict) else {"r": b}))


def _fake_results(point_only=("P1", "P2", "N", "C")) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for m in rq.MODELS:
        out[m] = {}
        for q in rq.QUANTITIES:
            v = float(rng.uniform(20, 200))
            r = rq.point_record(q, v)
            if m not in point_only:
                r.update(q10=v * 0.7, q90=v * 1.4, stat="mean" if q == rq.REVENUE else "median")
            out[m][q] = r
    return out


# --------------------------------------------------------------------------- frozen specification
def test_frozen_model_list():
    assert list(rq.MODELS) == ["A", "B", "E", "G", "R", "P1", "P2", "N", "C"]
    assert rq.STATE_SPACE == ("A", "B", "E", "G", "R")
    assert rq.PANEL == {"P1": "pca", "P2": "ridge"}
    assert (rq.NAIVE, rq.COMBINED) == ("N", "C")
    assert rq.COMBINATION == ("A", "E", "G", "R", "P2")
    assert set(rq.STATE_SPACE) | set(rq.PANEL) | {rq.NAIVE, rq.COMBINED} == set(rq.MODELS)
    assert set(rq.VARIANT_INDICATORS) == set(rq.STATE_SPACE)
    assert set(rq.JOBS) == set(rq.STATE_SPACE) | {"panel"}
    assert rq.QUANTITIES == ("revenue_q4_2026", "runrate_2026-11-30", "runrate_2026-12-31")
    assert rq.VINTAGE == pd.Timestamp("2026-11-12")
    assert rq.Q4_PERIOD == "2026-10-01..2026-12-31"
    assert rq.NOTE == "frozen 2026-11-12, filings excluded"
    assert not config.USE_FILINGS


def test_compute_all_returns_exactly_the_frozen_models(monkeypatch):
    fake = _fake_results()

    def run_job(name):
        if name == "panel":
            return name, {m: fake[m] for m in rq.PANEL}, 0.0
        if name == "A":
            return name, {"A": fake["A"], "N": fake["N"]}, 0.0
        return name, {name: fake[name]}, 0.0

    monkeypatch.setattr(rq, "run_job", run_job)
    res, _ = rq.compute_all(jobs=1, verbose=False)
    assert list(res) == list(rq.MODELS)
    assert all(list(res[m]) == list(rq.QUANTITIES) for m in rq.MODELS)
    assert _same(res["C"], rq.combine(fake))


def test_target_rows():
    t = rq.target_rows().set_index("obs_id")
    rev = t.loc[rq.TARGET_IDS[rq.REVENUE]]
    assert (rev.metric, rev.tier, rev.period_start, rev.period_end) == (
        "revenue", "gaap", pd.Timestamp("2026-10-01"), pd.Timestamp("2026-12-31"))
    for d in rq.RUNRATE_DATES:
        r = t.loc[rq.TARGET_IDS[f"runrate_{d}"]]
        assert (r.metric, r.definition, r.tier, r.ref_date) == ("run_rate", "t28x13", "official", pd.Timestamp(d))
        assert data.effective_date(r) == pd.Timestamp(d) - pd.Timedelta(days=14)
    assert (t.include == 0).all() and (t.published > rq.VINTAGE).all() and (t.filing == 0).all()
    tgt = rq.panel_target(RR_1231)
    assert tgt.company == "anthropic" and tgt.t_eff == pd.Timestamp("2026-12-17")


# --------------------------------------------------------------------------- variant data
def _openrouter_patch(c):              # scripts/forecast_openrouter.py, variant or_price
    c = c.copy()
    c.loc[c.indicator == "openrouter_anthropic_tokens", "role"] = "model"
    c.loc[c.indicator.isin(["npm_anthropic_sdk", "pypi_anthropic", "ramp_token_spend_anthropic"]), "role"] = "monitor"
    return c


def _ramp_patch(c, use):               # scripts/forecast_ramp.py, variants gh_share and ramp_only
    c = c.copy()
    c.loc[c.role == "model", "role"] = "monitor"
    c.loc[c.indicator.isin(use), "role"] = "model"
    return c


def test_variant_catalogs_match_the_variant_scripts():
    orig = altdata.catalog
    cat = orig()
    expected = {"A": cat, "B": cat, "E": _openrouter_patch(cat), "G": _ramp_patch(cat, ["github_repo_share_claude"]),
                "R": _ramp_patch(cat, ["ramp_adoption_anthropic"])}
    for m, exp in expected.items():
        with rq.variant_catalog(m):
            pd.testing.assert_frame_equal(altdata.catalog(), exp)
            assert {k for k, v in calibrate.load_params()["indicators"].items() if v["role"] == "model"} \
                == set(exp.loc[exp.role == "model", "indicator"])
        assert altdata.catalog is orig
    with pytest.raises(RuntimeError):
        with rq.variant_catalog("E"):
            raise RuntimeError
    assert altdata.catalog is orig


@pytest.mark.parametrize("model", ["A", "B", "E", "G", "R"])
def test_variant_data_and_evidence_hold_no_filing_rows(model):
    with rq.variant_catalog(model):
        df, _ = rq.variant_data(model)
    assert set(df.loc[df.metric == "indicator", "indicator"]) == EXPECTED_INDICATORS[model]
    tbl = rq.with_targets(df)
    assert S1A_OCT in set(tbl.obs_id)                       # the filing rows are in the table ...
    ev = data.select(tbl, vintage=rq.VINTAGE)
    rq.assert_clean_evidence(ev)                             # ... but never in the evidence
    assert S1A_OCT not in set(ev.obs_id)
    assert not set(ev.obs_id) & rq.filing_obs_ids()
    if model == "A":
        assert set(df.segment) == {"total"} and "share" not in set(df.metric)


def test_openrouter_tokens_are_price_adjusted():
    with rq.variant_catalog("E"):
        df, _ = rq.variant_data("E")
        raw = data.load_all()
    m = df.metric == "indicator"
    ratio = df.loc[m, "value"].to_numpy() / raw.loc[raw.metric == "indicator", "value"].to_numpy()
    assert np.nanmax(np.abs(ratio - 1)) > 0.01              # the price index moved after the first measured week


def test_assert_clean_evidence_rejects_filings_late_rows_and_targets(table_a):
    df, _ = table_a
    ev = data.select(df, vintage=rq.VINTAGE)
    rq.assert_clean_evidence(ev)
    with pytest.raises(AssertionError):
        rq.assert_clean_evidence(pd.concat([ev, df[df.obs_id == S1A_OCT]], ignore_index=True))
    with pytest.raises(AssertionError):
        rq.assert_clean_evidence(pd.concat([ev, _fake_claim(0, published=pd.Timestamp("2026-11-13"))], ignore_index=True))
    with pytest.raises(AssertionError):
        rq.assert_clean_evidence(pd.concat([ev, rq.target_rows()], ignore_index=True))


# --------------------------------------------------------------------------- filings never enter
def test_filing_row_never_changes_the_production_forecast(table_a):
    """Model A with the frozen settings (refit at the vintage, crossing position profiled): a wild
    claim flagged as a filing changes nothing; the same claim unflagged does move the forecast."""
    df, p = table_a
    q = (RR_1231,)
    base, _ = rq.statespace_forecasts("A", df=df, params=p, quantities=q)
    filed, ev = rq.statespace_forecasts("A", df=pd.concat([df, _fake_claim(1)], ignore_index=True), params=p, quantities=q)
    assert _same(filed[RR_1231], base[RR_1231])
    assert "fake_s1" not in set(ev[RR_1231].obs_id) and S1A_OCT not in set(ev[RR_1231].obs_id)
    fast = dict(params=p, refit=False, crossing=None, quantities=q)
    b0, _ = rq.statespace_forecasts("A", df=df, **fast)
    moved, _ = rq.statespace_forecasts("A", df=pd.concat([df, _fake_claim(0)], ignore_index=True), **fast)
    assert abs(moved[RR_1231]["log_mu"] - b0[RR_1231]["log_mu"]) > 0.05


def test_panel_claims_hold_no_filing_rows():
    claims = rq.panel_claims()
    for c, d in claims.items():
        assert (d.filing.fillna(0).astype(int) == 0).all(), c
    assert not set(claims["anthropic"].obs_id) & rq.filing_obs_ids()
    inj = pd.concat([claims["anthropic"], _fake_claim(1, company="anthropic")], ignore_index=True)
    sel = pm.public_claims(inj, rq.VINTAGE)
    assert "fake_s1" not in set(sel.obs_id)
    rq.assert_clean_evidence(sel)


def test_panel_forecasts_plumbing():
    out = rq.panel_forecasts(noise={c: NOISE for c in pm.COMPANIES}, quantities=(RR_1231,))
    assert set(out) == {"P1", "P2"}
    for m in out:
        r = out[m][RR_1231]
        assert np.isfinite(r["point"]) and r["point"] > 0 and np.isnan(r["q10"])
        assert r["detail"].startswith("t0 ")


# --------------------------------------------------------------------------- quantities
def test_quantiles_are_the_models_own_uncertainty(table_a, fast_a):
    """Registered numbers = the smoothed state at the vintage (registry.register's convention):
    the target tier's measurement noise added by observation_predictive is removed."""
    df, p = table_a
    recs, _ = fast_a
    ev = data.select(rq.with_targets(df), vintage=rq.VINTAGE)
    res = Model(p).run(ev, end=rq.Q4_END, queries=[(rq.Q4_START, rq.Q4_END)])
    for d in rq.RUNRATE_DATES:
        r = recs[f"runrate_{d}"]
        mu, sd = res.log_runrate(d, "t28x13")
        assert r["log_mu"] == pytest.approx(mu, abs=1e-6) and r["log_sd"] == pytest.approx(sd, abs=1e-6)
        assert r["point"] == pytest.approx(math.exp(mu), rel=1e-6)
        assert r["q10"] < r["point"] < r["q90"]
    pr = res.period_revenue(rq.Q4_PERIOD)
    r = recs[rq.REVENUE]
    for k, kk in (("point", "mean"), ("sd_usd_b", "sd"), ("q10", "q10"), ("q90", "q90")):
        assert r[k] == pytest.approx(pr[kk], rel=1e-6)


def test_naive_is_the_last_non_filing_point_claim(fast_a):
    _, evs = fast_a
    n = rq.naive_forecasts(evs)
    ev = evs[RR_1231]
    rr = ev[(ev.metric == "run_rate") & (ev.kind == "point") & (ev.segment == "total") & (ev.is_forecast == 0)]
    last = rr.sort_values("ref_date").iloc[-1]
    assert int(last.filing) == 0
    for d in rq.RUNRATE_DATES:
        assert n[f"runrate_{d}"]["point"] == pytest.approx(last.value)
        assert np.isnan(n[f"runrate_{d}"]["q10"])
    assert n[rq.REVENUE]["point"] == pytest.approx(last.value * 92 / 365)


def test_combination_is_the_mean_of_member_log_forecasts():
    res = _fake_results()
    c = rq.combine(res)
    for q in rq.QUANTITIES:
        logs = [math.log(res[m][q]["point"]) for m in rq.COMBINATION]
        assert math.log(c[q]["point"]) == pytest.approx(np.mean(logs), rel=1e-12)
        assert c[q]["log_mu"] == pytest.approx(np.mean(logs), rel=1e-12)
        assert np.isnan(c[q]["q10"])
    other = {m: dict(v) for m, v in res.items()}
    for m in ("B", "P1", "N"):                            # non-members do not matter
        other[m] = {q: rq.point_record(q, 1e6) for q in rq.QUANTITIES}
    assert _same(rq.combine(other), c)
    missing = {m: dict(v) for m, v in res.items()}
    missing["P2"] = dict(missing["P2"], **{RR_1231: rq.point_record(RR_1231, np.nan)})
    cm = rq.combine(missing)
    assert np.isnan(cm[RR_1231]["point"]) and cm[rq.REVENUE]["point"] == c[rq.REVENUE]["point"]


# --------------------------------------------------------------------------- registry
def test_registry_summaries_round_trip(tmp_path):
    res = _fake_results()
    path = str(tmp_path / "registry.jsonl")
    rq.register_all(res, path=path)
    entries = registry.load_registry(path)
    assert [e["model"] for e in entries] == [f"q4_frozen_{m}" for m in rq.MODELS]
    for e, m in zip(entries, rq.MODELS):
        assert e["asof"] == "2026-11-12" and e["note"].startswith(rq.NOTE)
        assert set(e["runrate_points"]) == set(rq.RUNRATE_DATES) and list(e["revenue"]) == [rq.Q4_PERIOD]
        for d in rq.RUNRATE_DATES:
            pt, r = e["runrate_points"][d], res[m][f"runrate_{d}"]
            assert pt["median"] == pytest.approx(r["point"], abs=1e-3)
        rv = e["revenue"][rq.Q4_PERIOD]
        assert rv["mean"] == pytest.approx(res[m][rq.REVENUE]["point"], abs=1e-4)
        if m in ("P1", "P2", "N", "C"):
            assert e["note"] == rq.NOTE + rq.POINT_ONLY_NOTE
            assert rv["sd"] == 0 and rv["q10"] == rv["q90"]
            assert all(v["log_sd"] == 0 for v in e["runrate_points"].values())
        else:
            assert e["note"] == rq.NOTE and rv["sd"] > 0
            assert all(v["log_sd"] > 0 for v in e["runrate_points"].values())
    assert registry.score(data.load(), calibrate.load_params(), path=path) is not None


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def test_default_is_a_dry_run_that_never_touches_the_registry(tmp_path, monkeypatch):
    fake = _fake_results()
    monkeypatch.setattr(rq, "compute_all", lambda jobs=1, verbose=True: (fake, {}))

    def boom(*a, **k):
        raise AssertionError("dry run wrote to the registry")

    monkeypatch.setattr(registry, "register_summary", boom)
    before = _sha(config.REGISTRY_PATH)
    out = tmp_path / "q4.csv"
    for argv in ([], ["--dry-run"]):
        assert rq.main(argv + ["--out", str(out), "--jobs", "1"]) == 0
    assert _sha(config.REGISTRY_PATH) == before
    frame = pd.read_csv(out)
    assert list(frame.model.unique()) == list(rq.MODELS) and len(frame) == len(rq.MODELS) * len(rq.QUANTITIES)
    with pytest.raises(SystemExit):
        rq.main(["--dry-run", "--register"])


def test_register_refuses_dirty_tree_or_second_registration(tmp_path, monkeypatch):
    def no_compute(*a, **k):
        raise AssertionError("computed before the checks")

    monkeypatch.setattr(rq, "compute_all", no_compute)
    monkeypatch.setattr(registry, "register_summary", no_compute)
    monkeypatch.setattr(rq, "already_registered", lambda path=None: [])
    monkeypatch.setattr(rq, "uncommitted", lambda: ["?? runrate_nowcast/scripts/register_q4.py"])
    with pytest.raises(SystemExit):
        rq.main(["--register", "--out", str(tmp_path / "q4.csv")])
    monkeypatch.setattr(rq, "uncommitted", lambda: [])
    monkeypatch.setattr(rq, "already_registered", lambda path=None: ["q4_frozen_A"])
    with pytest.raises(SystemExit):
        rq.main(["--register", "--out", str(tmp_path / "q4.csv")])
