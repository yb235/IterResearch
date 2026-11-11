# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/forecast_all.py <variant> <rules>
"""Variant F: every real-time signal at once in the state-space model (no filings as inputs).

  all_price   SDK downloads (npm, PyPI), Ramp, OpenRouter Anthropic tokens, GitHub co-authored
              commits, OpenCode downloads; every usage signal x measured price per token (gamma 1)
  all_raw     the same signals without the price adjustment
  core_price  SDK downloads, Ramp, OpenRouter tokens, x price. SELECTED USING THE BACKTEST (it drops
              the two signals that failed alone), so its score is optimistic.
  ml_all      ML (ridge, gbm, trend) on all signals, interpolated and smoother labels
Each indicator keeps its own drifting intercept and noise, refit at every vintage; new signals get
the catalog's wide prior (beta 1 +/- 1).
"""
import warnings; warnings.simplefilter("ignore")
import os, sys
import pandas as pd
from runrate_nowcast import altdata, data, calibrate, backtest, pricing, config, ml
variant, rules = sys.argv[1], sys.argv[2].split(",")
ALL = ["npm_anthropic_sdk", "pypi_anthropic", "ramp_token_spend_anthropic", "openrouter_anthropic_tokens",
       "github_claude_commits", "npm_opencode"]
CORE = ["npm_anthropic_sdk", "pypi_anthropic", "ramp_token_spend_anthropic", "openrouter_anthropic_tokens"]
use = CORE if variant == "core_price" else ALL
cat0 = altdata.catalog()
def cat_patched(path=config.CATALOG_PATH):
    c = cat0.copy()
    c.loc[c.role == "model", "role"] = "monitor"
    c.loc[c.indicator.isin(use), "role"] = "model"
    m = c.indicator == "github_claude_commits"
    for k, v in {"beta": 1.0, "beta_sd": 1.0, "obs_sd": 0.15, "drift": 0.01, "aggregate": "none",
                 "frequency": "weekly", "publish_delay_days": 1}.items():
        c.loc[m, k] = v
    return c
altdata.catalog = cat_patched
g = pd.read_csv(os.path.join(config.PKG_DIR, "data", "github_claude_commits.csv"), parse_dates=["week_start"])
g = g[(g["query"] == "Claude") & g.total_count.notna()]
raw_g = pd.DataFrame({"date": g.week_start + pd.Timedelta(days=6), "indicator": "github_claude_commits",
                      "value": g.total_count.astype(float), "published": g.week_start + pd.Timedelta(days=7),
                      "source": "github search", "note": ""})
raw = pd.concat([altdata.load_raw(), raw_g], ignore_index=True)
raw["date"] = pd.to_datetime(raw["date"])
p = calibrate.load_params()
if variant == "ml_all":
    df = data.load_all()
    for rule in rules:
        res = ml.run(df, raw=raw, params=p, verbose=False, vintage_rule=rule, feature_set="all")
        res.to_csv(os.path.join(config.OUTPUT_DIR, f"forecast_mlall_{rule}_2026-11-10.csv"), index=False)
        print(variant, rule, "done", len(res), flush=True)
    sys.exit()
ind = altdata.indicator_observations(roles=("model",), cat=cat_patched(), raw=raw)
df = pd.concat([data.load(), ind], ignore_index=True)
print(variant, ind.indicator.value_counts().to_dict(), flush=True)
if variant in ("all_price", "core_price"):
    df = pricing.adjust_indicators(df, 1.0, "github")
for rule in rules:
    bt = backtest.run_backtest(df, p, refit=True, crossing="profile", vintage_rule=rule)
    bt.to_csv(os.path.join(config.OUTPUT_DIR, f"forecast_all_{variant}_{rule}_2026-11-10.csv"), index=False)
    print(variant, rule, "done", flush=True)
