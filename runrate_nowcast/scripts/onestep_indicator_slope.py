# Run from the repo root:
#   PYTHONPATH=. python runrate_nowcast/scripts/onestep_indicator_slope.py <model> <variant> <evidence>
"""Sub-task P: one-step-ahead prediction of every point claim as it arrives (does the bias persist?).

Claims: every point claim of total run-rate or revenue that is not a forecast (all tiers, filings included
as claims to predict: they are the answers), in order of publication. For a claim published on day p:
  vintage   p - 1 (the day before it arrived); evidence = rows public at the vintage minus its own origin,
            exactly backtest.prepublication_forecast (check_no_leak on, crossing position profiled at the
            vintage); config.USE_FILINGS per the evidence set
  params    real time, no refit here: the parameters refit at the latest backtest vintage on or before the
            claim's vintage, from scripts/forecast_indicator_slope.py's params_<model>_<variant>_<evidence>_*.json
            (both vintage rules pooled); before the first refit, backtest.unfitted_params() (catalog priors,
            package defaults)
Output: reports/indicator_slope_2026-11-13/onestep_<model>_<variant>_<evidence>.csv with z = (y - mu) / sd
(standardized one-step-ahead innovation) and log_err = log(actual / forecast) (+ = forecast too low).

Also: the online filter's own innovations of the same point claims, in the order the filter meets them
(effective date), from one forward pass over all rows public on 2026-11-12 with the last refit parameters
(columns kind "filter"): innov_<model>_<variant>_<evidence>.csv.
"""
import warnings; warnings.simplefilter("ignore")
import glob, json, math, os, sys
import pandas as pd
from runrate_nowcast import backtest, calibrate, config, data, lower_rule
from runrate_nowcast import indicator_slope as isl
from runrate_nowcast.model import Model

OUTDIR = os.environ.get("ISL_OUTDIR") or os.path.join(config.OUTPUT_DIR, f"indicator_slope_{isl.ASOF}")
LAST = pd.Timestamp("2026-11-12")


def point_claims(df):
    full = data.select(df, filings=True)
    pc = full[full.metric.isin(["run_rate", "revenue"]) & (full.kind == "point") & (full.is_forecast == 0)
              & (full.segment == "total")]
    return pc.sort_values(["published", "t_eff", "obs_id"]).reset_index(drop=True)


def fits(model, variant, evidence):
    out = []
    for f in glob.glob(os.path.join(OUTDIR, f"params_{model}_{variant}_{evidence}_*.json")):
        for tg, r in json.load(open(f)).items():
            out.append((pd.Timestamp(r["vintage"]), tg, r["params"]))
    return sorted(out, key=lambda x: (x[0], x[1]))


def main(model, variant, evidence):
    isl.apply_variant(variant)
    lower_rule.apply(isl.LOWER)
    config.USE_FILINGS = isl.EVIDENCE[evidence]
    isl.use_model(model)
    df = isl.model_frame(model)
    F = fits(model, variant, evidence)
    if not F:
        raise SystemExit(f"no params_{model}_{variant}_{evidence}_*.json yet")
    rows = []
    for row in point_claims(df).itertuples():
        v = pd.Timestamp(row.published) - pd.Timedelta(days=1)
        prior = [f for f in F if f[0] <= v]
        p, src = (prior[-1][2], f"{prior[-1][1]}@{prior[-1][0].date()}") if prior else (backtest.unfitted_params(), "unfitted")
        out = backtest.prepublication_forecast(df, row, p, False, 3, None, "profile", vintage=v)
        if out is None:
            continue
        mu, sd, n, ev = out
        y = math.log(row.value) if row.metric == "run_rate" else float(row.value)
        fc = math.exp(mu) if row.metric == "run_rate" else mu
        err = (y - mu) if row.metric == "run_rate" else (math.log(y / mu) if mu > 0 else float("nan"))
        rows.append({"model": model, "variant": variant, "evidence": evidence, "obs_id": row.obs_id,
                     "metric": row.metric, "tier": row.tier, "filing": int(row.filing), "published": row.published.date(),
                     "vintage": v.date(), "t_eff": pd.Timestamp(row.t_eff).date(), "actual": row.value,
                     "forecast": fc, "mu": mu, "sd": sd, "z": (y - mu) / sd, "log_err": err, "n_evidence": n,
                     "params_from": src})
    os.makedirs(OUTDIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, f"onestep_{model}_{variant}_{evidence}.csv"), index=False)

    # the online filter's own innovations (effective-date order), all rows public on LAST
    p = F[-1][2]
    rho = __import__("runrate_nowcast.robust", fromlist=["x"]).best_crossing_position(df, backtest.unfitted_params(), LAST)
    sel = data.select(data.with_crossing_position(df, rho), vintage=LAST)
    res = Model(p).run(sel, smooth=False)
    ids = set(point_claims(df).obs_id)
    inn = [{"model": model, "variant": variant, "evidence": evidence, "obs_id": i.obs_id, "date": i.date.date(),
            "metric": i.metric, "z": i.z, "resid": i.y - i.mu} for i in res.innovations if i.obs_id in ids]
    pd.DataFrame(inn).to_csv(os.path.join(OUTDIR, f"innov_{model}_{variant}_{evidence}.csv"), index=False)
    d = pd.DataFrame(rows)
    print(model, variant, evidence, len(d), "mean z", round(d.z.mean(), 3), "share z<0", round((d.z < 0).mean(), 2),
          "mean log err", round(d.log_err.mean(), 3), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
