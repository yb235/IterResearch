# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/<this file> [args]
import warnings; warnings.simplefilter("ignore")
import os
from runrate_nowcast import data, calibrate, bayes, config
df_all = data.load_all(); p = calibrate.load_params()
for rule in ("publication", "period_end"):
    bt = bayes.bayes_backtest(df_all, p, crossing="profile", vintage_rule=rule)
    bt.to_csv(os.path.join(config.OUTPUT_DIR, f"forecast_bayes_{rule}_2026-11-10.csv"), index=False)
    print(rule, "bayes done", flush=True); print(bayes.summarize(bt).round(3).to_string(), flush=True)
print("all done", flush=True)
