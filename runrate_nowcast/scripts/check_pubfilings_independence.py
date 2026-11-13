# Run from the repo root:
#   PYTHONPATH=. python runrate_nowcast/scripts/check_pubfilings_independence.py
"""Sub-task O, leak check 3: answer independence under the "pub" evidence rule.

For one run-rate target (rr_2026_10_s1a) and one revenue target (rev_2026q3), models A and B, both
vintage rules, the lower-claim rule iv115 and config.USE_FILINGS = True (as forecast_pubfilings.py):
forecast the target with backtest.prepublication_forecast exactly as run_backtest calls it (refit=True,
crossing="profile", vintage from backtest.target_vintage), then again with the target's own value
multiplied by 3. The predictive mean and sd must not change. Also checks that the unmodified result
equals the row of reports/forecast_pubfilings_<model>_iv115_<rule>_2026-11-13.csv.

Output: reports/pubfilings_answer_independence_2026-11-13.csv (not overwritten if it exists).
"""
import warnings; warnings.simplefilter("ignore")
import math, os, time
import pandas as pd
from runrate_nowcast import backtest, calibrate, config, data, lower_rule

ASOF = "2026-11-13"
TARGETS = ("rr_2026_10_s1a", "rev_2026q3")
MODELS = ("A", "B")
RULES = ("publication", "period_end")


def frame(model: str) -> pd.DataFrame:
    df = data.load_all()
    if model == "A":
        base = df[df.metric != "indicator"]
        df = base[(base.segment == "total") & (base.metric != "share")]
    return df


def forecast(df: pd.DataFrame, target: str, rule: str, p: dict):
    """(mu, sd, n_evidence): the call run_backtest makes for this target."""
    full = data.select(df, filings=True)
    tg = next(full[full.obs_id == target].itertuples())
    vint = backtest.target_vintage(tg, rule)
    mu, sd, n, _ = backtest.prepublication_forecast(df, tg, p, True, 3, None, "profile", vintage=vint)
    return mu, sd, n, vint


def main():
    out = os.path.join(config.OUTPUT_DIR, f"pubfilings_answer_independence_{ASOF}.csv")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; not overwriting")
    assert not config.USE_FILINGS
    config.USE_FILINGS = True
    lower_rule.apply("iv115")
    p = calibrate.load_params()
    rows = []
    for model in MODELS:
        df = frame(model)
        for target in TARGETS:
            df3 = df.copy()
            df3.loc[df3.obs_id == target, "value"] *= 3
            for rule in RULES:
                t = time.time()
                mu0, sd0, n0, vint = forecast(df, target, rule, p)
                mu3, sd3, n3, _ = forecast(df3, target, rule, p)
                fc0 = math.exp(mu0) if target.startswith("rr_") else mu0
                rows.append({"model": model, "target": target, "rule": rule, "vintage": vint.date(),
                             "mu": mu0, "sd": sd0, "mu_x3": mu3, "sd_x3": sd3, "n_ev": n0, "n_ev_x3": n3,
                             "abs_diff_mu": abs(mu3 - mu0), "abs_diff_sd": abs(sd3 - sd0), "forecast": fc0,
                             "seconds": round(time.time() - t)})
                print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    # the unmodified forecast must equal the row of the main pub run (read last: those runs may finish later)
    fc_csv = []
    for r in d.itertuples():
        f = os.path.join(config.OUTPUT_DIR, f"forecast_pubfilings_{r.model}_iv115_{r.rule}_{ASOF}.csv")
        csv = pd.read_csv(f) if os.path.exists(f) else None
        fc_csv.append(csv[csv.target == r.target].forecast.iloc[0] if csv is not None else float("nan"))
    d["forecast_csv"] = fc_csv
    d["rel_diff_vs_csv"] = (d.forecast / d.forecast_csv - 1).abs()
    d.to_csv(out, index=False)
    print(d.to_string())


if __name__ == "__main__":
    main()
