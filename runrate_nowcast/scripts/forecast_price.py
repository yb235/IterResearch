# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/<this file> [args]
"""Price-adjusted indicators (variant C): backtests for one (scheme, gamma) at the given rules."""
import warnings; warnings.simplefilter("ignore")
import os, sys
from runrate_nowcast import data, calibrate, backtest, config, pricing
scheme, gamma = sys.argv[1], float(sys.argv[2]); rules = sys.argv[3].split(",")
df_all = data.load_all(); p = calibrate.load_params()
adj = pricing.adjust_indicators(df_all, gamma, scheme)
for rule in rules:
    bt = backtest.run_backtest(adj, p, refit=True, crossing="profile", vintage_rule=rule)
    bt.to_csv(os.path.join(config.OUTPUT_DIR, f"forecast_price_{scheme}_g{gamma:g}_{rule}_2026-11-10.csv"), index=False)
    print(scheme, gamma, rule, "done", flush=True)
