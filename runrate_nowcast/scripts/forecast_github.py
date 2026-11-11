# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/<this file> [args]
"""Variant D: add public GitHub commits co-authored by Claude as a usage indicator.

The catalog prior is replaced in this process only by a wide one (beta 1 +/- 1, obs_sd 0.15),
since the indicator is new and its counts are approximate; nothing is tuned on targets.
"""
import warnings; warnings.simplefilter("ignore")
import os, sys
import pandas as pd
from runrate_nowcast import altdata, data, calibrate, backtest, pricing, config
variant, rules = sys.argv[1], sys.argv[2].split(",")
cat0 = altdata.catalog()
def cat_patched(path=config.CATALOG_PATH):
    c = cat0.copy()
    m = c.indicator == "github_claude_commits"
    for k, v in {"role": "model", "beta": 1.0, "beta_sd": 1.0, "obs_sd": 0.15, "drift": 0.01,
                 "aggregate": "none", "frequency": "weekly", "publish_delay_days": 1}.items():
        c.loc[m, k] = v
    return c
altdata.catalog = cat_patched
g = pd.read_csv(os.path.join(config.PKG_DIR, "data", "github_claude_commits.csv"), parse_dates=["week_start"])
g = g[(g["query"] == "Claude") & g.total_count.notna()]
raw_g = pd.DataFrame({"date": g.week_start + pd.Timedelta(days=6), "indicator": "github_claude_commits",
                      "value": g.total_count.astype(float), "published": g.week_start + pd.Timedelta(days=7),
                      "source": "github search", "note": "weekly public commits with Claude co-author trailer"})
raw = pd.concat([altdata.load_raw(), raw_g], ignore_index=True)
ind = altdata.indicator_observations(roles=("model",), cat=cat_patched(), raw=raw)
df = pd.concat([data.load(), ind], ignore_index=True)
print("github weeks:", int((ind.indicator == "github_claude_commits").sum()), flush=True)
if variant == "price":
    df = pricing.adjust_indicators(df, 1.0, "uninformed")
p = calibrate.load_params()
for rule in rules:
    bt = backtest.run_backtest(df, p, refit=True, crossing="profile", vintage_rule=rule)
    name = "forecast_price_uninformed_g1_github" if variant == "price" else "forecast_price_none_g0_github"
    bt.to_csv(os.path.join(config.OUTPUT_DIR, f"{name}_{rule}_2026-11-10.csv"), index=False)
    print(variant, rule, "done", flush=True)
