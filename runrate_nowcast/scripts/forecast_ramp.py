# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/forecast_ramp.py <variant> <rules>
"""Ramp monthly vendor adoption (share of businesses paying Anthropic) as an indicator. No filings as inputs.
  ramp_only     claims + Ramp Anthropic adoption
  ramp_or       claims + Ramp adoption + OpenRouter tokens x price per token
  gh_share      claims + share of new GitHub repos mentioning Claude (vs OpenAI, Gemini)
"""
import warnings; warnings.simplefilter("ignore")
import os, sys
from runrate_nowcast import altdata, data, calibrate, backtest, pricing, config
variant, rules = sys.argv[1], sys.argv[2].split(",")
use = {"ramp_only": ["ramp_adoption_anthropic"], "ramp_or": ["ramp_adoption_anthropic", "openrouter_anthropic_tokens"],
       "gh_share": ["github_repo_share_claude"]}[variant]
cat0 = altdata.catalog()
def cat_patched(path=config.CATALOG_PATH):
    c = cat0.copy()
    c.loc[c.role == "model", "role"] = "monitor"
    c.loc[c.indicator.isin(use), "role"] = "model"
    return c
altdata.catalog = cat_patched
df = data.load_all()
print(variant, df[df.metric == "indicator"].indicator.value_counts().to_dict(), flush=True)
if variant == "ramp_or":     # price-adjust the token series only (adoption counts businesses, not tokens)
    m = df.indicator == "openrouter_anthropic_tokens"
    adj = pricing.adjust_indicators(df[m | (df.metric != "indicator")], 1.0, "github")
    df = __import__("pandas").concat([adj, df[(df.metric == "indicator") & ~m]], ignore_index=True)
p = calibrate.load_params()
for rule in rules:
    bt = backtest.run_backtest(df, p, refit=True, crossing="profile", vintage_rule=rule)
    bt.to_csv(os.path.join(config.OUTPUT_DIR, f"forecast_openrouter_{variant}_{rule}_2026-11-10.csv"), index=False)
    print(variant, rule, "done", flush=True)
