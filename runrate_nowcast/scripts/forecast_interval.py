# Run from the repo root:
#   PYTHONPATH=. python runrate_nowcast/scripts/forecast_interval.py <model> <variant> <rules>
#   e.g. ... forecast_interval.py A iv115 publication,period_end
"""Sub-task N: the no-filing backtests rerun with a lower-claim rule (lower_rule.py, DATA_PLAN 子任务 N).

Each model mirrors its original script exactly; only config.LOWER_UPPER_RATIO differs.
  A      claims only            forecast_statespace_ml.py (claims_only frame)
  B      + catalog indicators   forecast_statespace_ml.py (with_indicators: data.load_all())
  E      OpenRouter x price     forecast_openrouter.py or_price
  G      GitHub new-repo share  forecast_ramp.py gh_share
  R      Ramp adoption          forecast_ramp.py ramp_only
  bayes  Laplace mixture        forecast_bayes.py
A, B, E, G, R: backtest.run_backtest(df, calibrate.load_params(), refit=True, crossing="profile",
vintage_rule=rule). bayes: bayes.bayes_backtest(df_all, p, crossing="profile", vintage_rule=rule).
variant: baseline (one-sided, the original rule), point, iv110, iv115 (primary), iv125.
Output: reports/forecast_interval_<model>_<variant>_<rule>_2026-11-13.csv, with the columns
run_backtest (or bayes_backtest) returns, plus `variant`. Existing reports are never overwritten.
"""
import warnings; warnings.simplefilter("ignore")
import os, sys, time
from runrate_nowcast import altdata, backtest, bayes, calibrate, config, data, lower_rule, pricing

ASOF = "2026-11-13"
model, variant, rules = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
assert not config.USE_FILINGS
lower_rule.apply(variant)

# ---- catalog patches, copied from the original scripts
cat0 = altdata.catalog()
if model == "E":                                  # forecast_openrouter.py, variant or_price
    def cat_patched(path=config.CATALOG_PATH):
        c = cat0.copy()
        c.loc[c.indicator == "openrouter_anthropic_tokens", "role"] = "model"
        c.loc[c.indicator.isin(["npm_anthropic_sdk", "pypi_anthropic", "ramp_token_spend_anthropic"]), "role"] = "monitor"
        return c
    altdata.catalog = cat_patched
elif model in ("G", "R"):                         # forecast_ramp.py, variants gh_share / ramp_only
    use = {"R": ["ramp_adoption_anthropic"], "G": ["github_repo_share_claude"]}[model]
    def cat_patched(path=config.CATALOG_PATH):
        c = cat0.copy()
        c.loc[c.role == "model", "role"] = "monitor"
        c.loc[c.indicator.isin(use), "role"] = "model"
        return c
    altdata.catalog = cat_patched
elif model not in ("A", "B", "bayes"):
    raise SystemExit(f"unknown model {model}")

df = data.load_all()
if model == "A":
    base = df[df.metric != "indicator"]
    df = base[(base.segment == "total") & (base.metric != "share")]
elif model == "E":
    df = pricing.adjust_indicators(df, 1.0, "github")
p = calibrate.load_params()
print(model, variant, lower_rule.describe(variant), df[df.metric == "indicator"].indicator.value_counts().to_dict(), flush=True)
for rule in rules:
    out = os.path.join(config.OUTPUT_DIR, f"forecast_interval_{model}_{variant}_{rule}_{ASOF}.csv")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; not overwriting")
    t = time.time()
    if model == "bayes":
        bt = bayes.bayes_backtest(df, p, crossing="profile", vintage_rule=rule)
    else:
        bt = backtest.run_backtest(df, p, refit=True, crossing="profile", vintage_rule=rule)
    bt.insert(0, "variant", variant)
    bt.to_csv(out, index=False)
    print(model, variant, rule, "done", len(bt), f"{time.time() - t:.0f}s", flush=True)
    if model == "bayes":
        print(bayes.summarize(bt).round(3).to_string(), flush=True)
