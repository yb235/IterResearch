# Run from the repo root:
#   PYTHONPATH=. python runrate_nowcast/scripts/forecast_pubfilings.py <model> <variant> <rules>
#   e.g. ... forecast_pubfilings.py A iv115 publication,period_end
"""Sub-task O: the backtests of sub-task N with filing numbers allowed once public (DATA_PLAN 子任务 O).

Evidence rule "pub": at each vintage every row published on or before the vintage is evidence, including
rows with filing = 1 (S-1, S-1/A, press reports of the draft S-1) and filing = 2 (third-party numbers
published after a filing). The target's own origin stays out (backtest.prepublication_forecast,
bayes.bayes_backtest) and backtest.check_no_leak stays on.

This script mirrors scripts/forecast_interval.py line for line (same catalog patches, same frames, same
calls: backtest.run_backtest(df, calibrate.load_params(), refit=True, crossing="profile",
vintage_rule=rule), bayes.bayes_backtest(df, p, crossing="profile", vintage_rule=rule)). The one change
is config.USE_FILINGS = True. Every evidence path reads that flag through data.select(filings=None):
the evidence rows (backtest.prepublication_forecast, bayes.bayes_backtest / fit_posterior), the
per-vintage process-noise and indicator refits (calibrate.fit_process_noise / fit_indicators via
data.select(df)) and the crossing-position profile (robust.crossing_profile). Target selection is
unchanged: run_backtest and bayes_backtest select targets with data.select(df, filings=True).

Two recorders wrap existing functions without changing what they return:
  backtest.check_no_leak  also records the filing rows (filing != 0) in each target's evidence and
                          asserts, on the raw table, that every evidence row was published on/before the
                          vintage and strictly before the target itself (so no row from the target's own
                          filing document can enter).
  backtest._naive         also records which claim the naive benchmark carries forward.
They add the columns filing_evidence ("obs_id@published;..."), n_filing_evidence and naive_from.

Output: reports/forecast_pubfilings_<model>_<variant>_<rule>_2026-11-13.csv (the columns of
run_backtest / bayes_backtest, plus variant, evidence = "pub" and the recorder columns).
Existing reports are never overwritten.
"""
import warnings; warnings.simplefilter("ignore")
import os, sys, time
import pandas as pd
from runrate_nowcast import altdata, backtest, bayes, calibrate, config, data, lower_rule, pricing

ASOF = "2026-11-13"
PREFIX = "forecast_pubfilings"
RAW = data.load().set_index("obs_id")          # raw rows: each row's own published date and filing flag

EVIDENCE_LOG: dict = {}                        # target obs_id -> {"vintage", "filing_rows"}
NAIVE_LOG: dict = {}                           # target obs_id -> description of the naive's input


def check_no_leak_logged(ev, vintage, row, _orig=backtest.check_no_leak):
    _orig(ev, vintage, row)                    # the package's own guard (filing check off when USE_FILINGS)
    v = pd.Timestamp(vintage)
    claims = ev[ev.metric != "indicator"]
    raw = RAW.loc[claims.obs_id]
    late = raw[raw.published > v]
    if len(late):
        raise AssertionError(f"{row.obs_id}: rows published after the vintage {v.date()}: {late.index.tolist()}")
    tpub = RAW.loc[row.obs_id, "published"]
    same_or_later = raw[raw.published >= tpub]
    if len(same_or_later):
        raise AssertionError(f"{row.obs_id}: rows published with or after the target: {same_or_later.index.tolist()}")
    fil = raw[raw.filing != 0]
    EVIDENCE_LOG[row.obs_id] = {
        "vintage": v.date(),
        "filing_rows": [(i, r.published.date(), int(r.filing), float(r.value)) for i, r in fil.sort_values("published").iterrows()],
    }


def naive_logged(row, ev, _orig=backtest._naive):
    out = _orig(row, ev)
    rr = ev[(ev.metric == "run_rate") & (ev.kind == "point") & (ev.segment == "total")
            & (ev.is_forecast == 0)].sort_values("ref_date")
    if out is None or not len(rr):
        NAIVE_LOG[row.obs_id] = ""
    elif row.metric == "run_rate":
        last = rr[rr.ref_date <= row.ref_date].iloc[-1]
        NAIVE_LOG[row.obs_id] = f"{last.obs_id}={last.value:g}@{pd.Timestamp(last.ref_date).date()}"
    else:                                      # revenue: interpolation of every point run-rate claim
        before = rr[rr.ref_date < row.period_start].tail(1)
        inside = rr[(rr.ref_date >= row.period_start) & (rr.ref_date <= row.period_end)]
        after = rr[rr.ref_date > row.period_end].head(1)
        use = pd.concat([before, inside, after])
        NAIVE_LOG[row.obs_id] = "interp " + ";".join(f"{r.obs_id}={r.value:g}@{pd.Timestamp(r.ref_date).date()}"
                                                     for r in use.itertuples())
    return out


def main(model, variant, rules):
    assert not config.USE_FILINGS, "expected the package default config.USE_FILINGS = False"
    config.USE_FILINGS = True                  # the only change from forecast_interval.py: evidence rule "pub"
    lower_rule.apply(variant)
    backtest.check_no_leak = check_no_leak_logged
    backtest._naive = naive_logged

    # ---- catalog patches, copied from scripts/forecast_interval.py
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
    print(model, variant, lower_rule.describe(variant), "evidence: pub (filings once public)",
          df[df.metric == "indicator"].indicator.value_counts().to_dict(), flush=True)
    for rule in rules:
        out = os.path.join(config.OUTPUT_DIR, f"{PREFIX}_{model}_{variant}_{rule}_{ASOF}.csv")
        if os.path.exists(out):
            raise SystemExit(f"{out} exists; not overwriting")
        EVIDENCE_LOG.clear(); NAIVE_LOG.clear()
        t = time.time()
        if model == "bayes":
            bt = bayes.bayes_backtest(df, p, crossing="profile", vintage_rule=rule)
        else:
            bt = backtest.run_backtest(df, p, refit=True, crossing="profile", vintage_rule=rule)
        bt.insert(0, "variant", variant)
        bt.insert(1, "evidence", "pub")
        for tg in bt.target:
            if str(EVIDENCE_LOG[tg]["vintage"]) != str(bt[bt.target == tg].vintage.iloc[0]):
                raise AssertionError(f"{tg}: recorded vintage differs")
        bt["n_filing_evidence"] = [len(EVIDENCE_LOG[t]["filing_rows"]) for t in bt.target]
        bt["filing_evidence"] = [";".join(f"{i}@{d}" for i, d, _, _ in EVIDENCE_LOG[t]["filing_rows"]) for t in bt.target]
        if model != "bayes":
            bt["naive_from"] = [NAIVE_LOG.get(t, "") for t in bt.target]
        bt.to_csv(out, index=False)
        print(model, variant, rule, "done", len(bt), f"{time.time() - t:.0f}s", flush=True)
        if model == "bayes":
            print(bayes.summarize(bt).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(","))
