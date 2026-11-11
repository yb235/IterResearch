# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/<this file> [args]
"""No-filing forecasts: state-space (claims only, + indicators) and ML, at two vintage rules."""
import warnings; warnings.simplefilter("ignore")
import os, sys
import pandas as pd
from runrate_nowcast import data, calibrate, backtest, config, ml
assert not config.USE_FILINGS
df_all = data.load_all(); p = calibrate.load_params()
base = df_all[df_all.metric != "indicator"]
A = base[(base.segment == "total") & (base.metric != "share")]
for rule in ("publication", "period_end"):
    out = {}
    for key, df in (("claims_only", A), ("with_indicators", df_all)):
        out[key] = backtest.run_backtest(df, p, refit=True, crossing="profile", vintage_rule=rule)
        print(rule, key, "done", len(out[key]), flush=True)
    m = out["claims_only"].merge(out["with_indicators"], on=["target", "metric", "vintage", "actual"],
                                 how="outer", suffixes=("_claims_only", "_with_indicators"))
    m.to_csv(os.path.join(config.OUTPUT_DIR, f"forecast_backtest_{rule}_2026-11-10.csv"), index=False)
    res = ml.run(df_all, params=p, verbose=False, vintage_rule=rule)
    res.to_csv(os.path.join(config.OUTPUT_DIR, f"forecast_ml_{rule}_2026-11-10.csv"), index=False)
    print(rule, "ml done", len(res), flush=True)
print("all done", flush=True)
