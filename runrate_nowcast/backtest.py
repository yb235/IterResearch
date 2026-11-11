"""Vintage backtest: forecast each official figure using only what was public the day before.

For each target (GAAP revenue or official run-rate):
  vintage = target.published - 1 day
  evidence = rows published on/before the vintage, excluding the target's own origin
  forecast = predictive distribution of the target claim (the smoothed state plus the
             target tier's measurement noise)
It reports the z-score, PIT and whether the 50%/80% intervals cover the actual value,
next to a naive benchmark. A calibrated model's 80% interval should cover about 80%
of targets.

Caveat: process-noise parameters are fit on the full sample, so this is not a strict
real-time test. Pass refit=True to refit them at every vintage (slower).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import calibrate, config, data
from .model import Model, observation_predictive

TARGET_TIERS = ("gaap", "official")


def _naive(row, ev: pd.DataFrame) -> float | None:
    """Benchmark an analyst could compute by hand from the same evidence.

    Run-rate target: the latest point run-rate claim (random walk), in log units.
    Revenue target: join the point run-rate claims log-linearly, hold flat outside
    them, and integrate over the period: sum of daily rate / 365, in $B.
    """
    rr = ev[(ev.metric == "run_rate") & (ev.kind == "point") & (ev.segment == "total")
            & (ev.is_forecast == 0)].sort_values("ref_date")
    if not len(rr):
        return None
    if row.metric == "run_rate":
        prior = rr[rr.ref_date <= row.ref_date]
        return None if not len(prior) else math.log(float(prior.iloc[-1].value))
    x = (rr.ref_date - pd.Timestamp("2000-01-01")).dt.days.to_numpy(float)
    y = np.log(rr.value.to_numpy(float))
    days = pd.date_range(row.period_start, row.period_end, freq="D")
    xd = (days - pd.Timestamp("2000-01-01")).days.to_numpy(float)
    return float(np.exp(np.interp(xd, x, y)).sum() / config.DAYS_PER_YEAR)


def unfitted_params() -> dict:
    """Defaults plus catalog indicator priors: nothing estimated from data, so safe at any vintage."""
    from . import altdata
    return {"indicators": altdata.indicator_params()}


def target_vintage(row, rule: str = "publication") -> pd.Timestamp:
    """Information date for a forecast of `row`.

    publication  the day before the number was published (nowcast just before release)
    period_end   the date the number refers to: end of the revenue period, or the run-rate
                 date. A forecast made then cannot see later claims about the same period.
                 Never later than the day before publication.
    """
    pub = pd.Timestamp(row.published) - pd.Timedelta(days=1)
    if rule == "publication":
        return pub
    if rule == "period_end":
        d = row.period_end if row.metric == "revenue" else row.ref_date
        return min(pd.Timestamp(d), pub)
    raise ValueError(rule)


def check_no_leak(ev: pd.DataFrame, vintage, row) -> None:
    """Hard guard run on every forecast: evidence must be public by the vintage, never a
    filing (unless config.USE_FILINGS), and never the target's own origin."""
    late = ev[ev.published > pd.Timestamp(vintage)]
    if len(late):
        raise AssertionError(f"look-ahead for {row.obs_id}: {late.obs_id.tolist()[:5]}")
    if not config.USE_FILINGS and "filing" in ev and (ev.filing.fillna(0).astype(int) != 0).any():
        raise AssertionError(f"filing rows in evidence for {row.obs_id}: {ev[ev.filing != 0].obs_id.tolist()}")
    if (ev.origin_id == row.origin_id).any():
        raise AssertionError(f"target origin in evidence for {row.obs_id}")


def prepublication_forecast(df: pd.DataFrame, row, params: dict | None = None, refit: bool = False,
                            min_evidence: int = 3, forecasts: bool | None = None, crossing: str | float | None = None,
                            vintage=None):
    """Forecast one claim using only rows public at the vintage, excluding its own origin.

    vintage defaults to the day before publication. Returns (mu, sd, n_evidence, evidence_df).
    mu and sd are in log units for run-rate claims and in $B for revenue claims. Returns None
    if there is too little evidence.
    """
    pub = pd.Timestamp(row.published) - pd.Timedelta(days=1)
    vintage = pub if vintage is None else min(pd.Timestamp(vintage), pub)
    ev0 = data.select(df, vintage=vintage, forecasts=forecasts)
    if (ev0[ev0.origin_id != row.origin_id].metric != "indicator").sum() < min_evidence:
        return None                         # too little evidence (checked before any fitting)
    if crossing is not None:
        # "profile": crossing position from data public at the vintage, with unfitted parameters
        # (full-sample fits would carry information from after the vintage)
        from . import robust
        rho = robust.best_crossing_position(df, unfitted_params(), vintage,
                                            exclude_origin=row.origin_id) if crossing == "profile" else float(crossing)
        df = data.with_crossing_position(df, rho)
        row = df[df.obs_id == row.obs_id].assign(t_eff=lambda x: x.apply(data.effective_date, axis=1)).iloc[0]
    ev = data.select(df, vintage=vintage, forecasts=forecasts)
    ev = ev[ev.origin_id != row.origin_id]
    check_no_leak(ev, vintage, row)
    # Count claims only: indicator weeks alone are not enough evidence to score a target.
    if (ev.metric != "indicator").sum() < min_evidence or pd.Timestamp(row.t_eff) < pd.Timestamp(config.MODEL_START):
        return None
    p = dict(params or {})
    if refit:
        # Strict real-time test: every estimated parameter is refit on data public before the vintage.
        vint = df[df.published <= vintage]
        p.update({k: v for k, v in calibrate.fit_process_noise(vint, p).items()
                  if k in ("sigma_l", "sigma_g", "sigma_b", "growth_reversion")})
        if (vint.metric == "indicator").any() and p.get("indicators"):
            from . import altdata
            p["indicators"] = altdata.indicator_params()     # restart from catalog priors, not full-sample fits
            _, p = calibrate.fit_indicators(vint, p, verbose=False)
    model = Model(p)
    queries = [(row.period_start, row.period_end)] if row.metric == "revenue" else None
    end = max(pd.Timestamp(row.t_eff), vintage)
    seg = getattr(row, "segment", "total")
    res = model.run(ev, end=end, queries=queries, segments=[seg] if seg != "total" else None)
    mu, sd = observation_predictive(res, row, model)
    return mu, sd, len(ev), ev


def run_backtest(df: pd.DataFrame, params: dict | None = None, refit: bool = False,
                 min_evidence: int = 3, segment_targets: bool = False,
                 forecasts: bool | None = None, target_ids=None, crossing=None,
                 vintage_rule: str = "publication") -> pd.DataFrame:
    """Targets: point claims from GAAP/official sources, excluding forecasts.

    By default only total-company run-rate and revenue are targets, so variants with and
    without the segment layer are scored on the same set.
    """
    full = data.select(df, filings=True)          # filings are targets, never evidence
    targets = full[full.tier.isin(TARGET_TIERS) & (full.kind == "point")
                   & full.metric.isin(["run_rate", "revenue"]) & (full.is_forecast == 0)]
    if not segment_targets:
        targets = targets[targets.segment == "total"]
    if target_ids is not None:            # fixed target set, e.g. for sensitivity runs
        targets = full[full.obs_id.isin(list(target_ids))].copy()
        targets["kind"] = "point"
    rows = []
    for tg in targets.itertuples():
        vint = target_vintage(tg, vintage_rule)
        out = prepublication_forecast(df, tg, params, refit, min_evidence, forecasts, crossing, vintage=vint)
        if out is None:
            continue
        mu, sd, n_ev, ev = out
        y = math.log(tg.value) if tg.metric == "run_rate" else float(tg.value)
        z = (y - mu) / sd
        nv = _naive(tg, ev)
        if tg.metric == "run_rate":
            err_model, err_naive = y - mu, (None if nv is None else y - nv)
            lo, hi = math.exp(mu - 1.2816 * sd), math.exp(mu + 1.2816 * sd)
            point = math.exp(mu)
            naive_pt = None if nv is None else math.exp(nv)
        else:
            err_model = math.log(y / mu)
            err_naive = None if nv is None else math.log(y / nv)
            lo, hi = mu - 1.2816 * sd, mu + 1.2816 * sd
            point, naive_pt = mu, nv
        rows.append({
            "target": tg.obs_id, "metric": tg.metric,
            "vintage": vint.date(),
            "n_evidence": n_ev, "actual": tg.value, "forecast": point, "q10": lo, "q90": hi,
            "z": z, "pit": float(norm.cdf(z)), "in50": abs(z) < 0.6745, "in80": abs(z) < 1.2816,
            "log_err_model": err_model, "naive": naive_pt, "log_err_naive": err_naive,
        })
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> dict:
    if not len(bt):
        return {}
    ok = bt.dropna(subset=["log_err_naive"])
    return {
        "n": int(len(bt)),
        "coverage50": float(bt.in50.mean()),
        "coverage80": float(bt.in80.mean()),
        "mean_z": float(bt.z.mean()),
        "sd_z": float(bt.z.std(ddof=0)),
        "mae_log_model": float(np.mean(np.abs(bt.log_err_model))),
        "mae_log_naive_same_targets": float(np.mean(np.abs(ok.log_err_naive))) if len(ok) else float("nan"),
        "mae_log_model_same_targets": float(np.mean(np.abs(ok.log_err_model))) if len(ok) else float("nan"),
    }


def layer_ablation(df_all: pd.DataFrame, params: dict | None = None, refit: bool = False,
                   n_boot: int = 10000, seed: int = 0):
    """Backtest the same targets while adding one layer at a time, with paired uncertainty.

    A  total-company claims only (run-rates, GAAP revenue)
    B  + segment claims (shares, Claude Code run-rates)
    C  + leading indicators            <- production default
    D  + forecast claims (company targets, investor/third-party forecasts)

    Returns (summary table, paired comparisons). Each paired comparison bootstraps the
    per-target difference in |log error| over the shared targets (90% interval) and
    counts how many targets improved. With ~13 targets, read an interval that crosses
    zero as "no evidence either way".
    """
    import numpy as np
    base = df_all[df_all.metric != "indicator"]
    variants = [
        ("A", "A total-level claims", base[(base.segment == "total") & (base.metric != "share")], False),
        ("B", "B + segment claims", base, False),
        ("C", "C + leading indicators", df_all, False),
        ("D", "D + forecast claims", df_all, True),
    ]
    rows, errs = [], {}
    for key, name, df, fc in variants:
        bt = run_backtest(df, params, forecasts=fc, refit=refit)
        s = summarize(bt)
        errs[key] = bt.set_index("target").log_err_model.abs()
        rows.append({"variant": name, "n": s["n"], "coverage80": s["coverage80"], "coverage50": s["coverage50"],
                     "mean_z": s["mean_z"], "sd_z": s["sd_z"], "mae_log": s["mae_log_model"],
                     "mae_log_naive": s["mae_log_naive_same_targets"]})
    rng = np.random.default_rng(seed)
    pairs = []
    for a, b in (("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")):
        common = errs[a].index.intersection(errs[b].index)
        d = (errs[b][common] - errs[a][common]).to_numpy()
        boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, [5, 95])
        pairs.append({"comparison": f"{b} vs {a}", "n": len(d), "mean_diff_abs_err": d.mean(),
                      "ci90_lo": lo, "ci90_hi": hi, "targets_improved": int((d < 0).sum()),
                      "targets_worse": int((d > 0).sum())})
    return pd.DataFrame(rows), pd.DataFrame(pairs)
