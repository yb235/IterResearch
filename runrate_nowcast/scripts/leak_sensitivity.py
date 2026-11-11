# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/<this file> [args]
import warnings; warnings.simplefilter("ignore")
import os
import pandas as pd
from runrate_nowcast import data, calibrate, leakcheck, config
df = data.load_all(); p = calibrate.load_params()
out = []
for rule in ("publication", "period_end"):
    s = leakcheck.post_filing_sensitivity(df, p, rule=rule)
    s["rule"] = rule
    out.append(s); print(s.round(2).to_string(index=False), flush=True)
pd.concat(out).to_csv(os.path.join(config.OUTPUT_DIR, "leakcheck_sensitivity_2026-11-10.csv"), index=False)
print("done", flush=True)
