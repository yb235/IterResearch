# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/forecast_openrouter.py <variant> <rules>
"""Variant E: OpenRouter weekly Anthropic tokens as a usage indicator (no filings as inputs).

  or_only     claims + OpenRouter Anthropic tokens (SDK downloads and Ramp left out)
  or_price    or_only with tokens x measured price per token (pricing.py, GitHub model mix): a spend proxy
  b_plus_or   claims + SDK downloads + Ramp + OpenRouter tokens
  opencode    claims + OpenCode npm downloads (vendor-agnostic agentic-coding demand)
The catalog roles are switched in this process only; priors come from the catalog (wide, beta 1 +/- 1).
"""
import warnings; warnings.simplefilter("ignore")
import os, sys
from runrate_nowcast import altdata, data, calibrate, backtest, pricing, config
variant, rules = sys.argv[1], sys.argv[2].split(",")
cat0 = altdata.catalog()
def cat_patched(path=config.CATALOG_PATH):
    c = cat0.copy()
    if variant == "opencode":
        c.loc[c.indicator == "npm_opencode", "role"] = "model"
    else:
        c.loc[c.indicator == "openrouter_anthropic_tokens", "role"] = "model"
    if variant in ("or_only", "or_price", "opencode"):
        c.loc[c.indicator.isin(["npm_anthropic_sdk", "pypi_anthropic", "ramp_token_spend_anthropic"]), "role"] = "monitor"
    return c
altdata.catalog = cat_patched
df = data.load_all()
print(variant, df[df.metric == "indicator"].indicator.value_counts().to_dict(), flush=True)
if variant == "or_price":
    df = pricing.adjust_indicators(df, 1.0, "github")
p = calibrate.load_params()
for rule in rules:
    bt = backtest.run_backtest(df, p, refit=True, crossing="profile", vintage_rule=rule)
    bt.to_csv(os.path.join(config.OUTPUT_DIR, f"forecast_openrouter_{variant}_{rule}_2026-11-10.csv"), index=False)
    print(variant, rule, "done", flush=True)
