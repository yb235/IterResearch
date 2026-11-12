# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/forecast_panel.py [--jobs 4] [--report-only]
"""Pooled two-company panel model (panel_model.py): backtest at both vintage rules, then a report.

Writes reports/forecast_panel_<rule>_2026-11-10.csv (both companies, `company` column) and
reports/forecast_panel_2026-11-10.md (comparison with the existing variants, read through
forecast_eval.collect(), and the naive last claim from forecast_backtest_<rule>_2026-11-10.csv).
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from runrate_nowcast import config, data, forecast_eval
from runrate_nowcast import features as F
from runrate_nowcast import panel_model as pm

warnings.simplefilter("ignore")
ASOF = "2026-11-10"
RULES = ("publication", "period_end")
COLS = ["target", "metric", "vintage", "actual", "forecast", "log_err", "model", "n_train_pairs",
        "company", "pool", "rule", "t0", "gap_weeks", "n_anchors", "n_target_pairs", "hyper", "cv_folds"]
EXISTING = ["A claims only", "E OpenRouter tokens x price", "G GitHub new-repo share", "ML all signals (smoother)"]
NAIVE = "naive last claim"
PANEL = [f"panel {k} ({p})" for p in pm.POOLS for k in pm.MODELS]
PRIMARY = "panel pca (pooled)"


def csv_path(rule):
    return os.path.join(config.OUTPUT_DIR, f"forecast_panel_{rule}_{ASOF}.csv")


def run(jobs: int):
    for rule in RULES:
        t = time.time()
        res = pm.run_backtest(rule, n_jobs=jobs)
        res = res.reindex(columns=COLS)
        res.to_csv(csv_path(rule), index=False)
        print(f"{rule}: {len(res)} rows, {res.target.nunique()} targets, {time.time() - t:.0f}s", flush=True)


# --------------------------------------------------------------------------- report
def anthropic_table() -> pd.DataFrame:
    """Existing variants (forecast_eval.collect), the naive last claim, and the panel models."""
    obs = data.load().set_index("obs_id")
    ex = forecast_eval.collect(ASOF)
    ex = ex[ex.model.isin(EXISTING)][["rule", "model", "target", "metric", "vintage", "actual", "forecast", "log_err"]]
    parts = [ex]
    for rule in RULES:
        b = pd.read_csv(os.path.join(config.OUTPUT_DIR, f"forecast_backtest_{rule}_{ASOF}.csv"))
        b = b.dropna(subset=["log_err_naive_claims_only"])
        parts.append(pd.DataFrame({"rule": rule, "model": NAIVE, "target": b.target, "metric": b.metric,
                                   "vintage": b.vintage, "actual": b.actual, "forecast": b.naive_claims_only,
                                   "log_err": b.log_err_naive_claims_only}))
        p = pd.read_csv(csv_path(rule))
        p = p[(p.company == "anthropic") & p.model.isin(PANEL + ["SS claims only (panel labels)"])]
        parts.append(p[["rule", "model", "target", "metric", "vintage", "actual", "forecast", "log_err"]])
    d = pd.concat(parts, ignore_index=True).dropna(subset=["log_err"])
    d["filing_target"] = d.target.map(lambda t: int(obs.loc[t, "filing"]) == 1)
    d["pct_err"] = 100 * (d.forecast / d.actual - 1)
    return d


def summary(g: pd.DataFrame, order) -> pd.DataFrame:
    keep = g.groupby("target").model.nunique()
    common = keep[keep == g.model.nunique()].index
    rows = []
    for m in order:
        h = g[g.model == m]
        if not len(h):
            continue
        hc = h[h.target.isin(common)]
        rows.append({"model": m, "n": len(h), "mae_log": h.log_err.abs().mean(), "bias_log": h.log_err.mean(),
                     f"n_common": len(hc), "mae_log_common": hc.log_err.abs().mean(),
                     "mae_log_h2_2026": h[h.target.isin(forecast_eval.H2_TARGETS)].log_err.abs().mean()})
    return pd.DataFrame(rows)


def pairs_table(d, rule, comps):
    out = [forecast_eval.paired(d, a, b, rule) for a, b in comps]
    return pd.DataFrame([p for p in out if p])


def elasticity_table() -> pd.DataFrame:
    """What the pooled and own-company ridge / pca learned at the last Anthropic vintage."""
    claims = {c: pm.load_claims(c) for c in pm.COMPANIES}
    src = F.load_sources()
    tg = pm.load_targets("anthropic").set_index("obs_id").loc["rr_2026_10_s1a"]
    tg = pm._Row(obs_id="rr_2026_10_s1a", **tg.to_dict())
    from runrate_nowcast import backtest
    v = backtest.target_vintage(tg, "publication")
    rows = []
    for pool in pm.POOLS:
        fm = pm.fitted_models(tg, v, claims, src, kinds=("trend", "pca", "ridge"), pool=pool)
        for k in ("trend", "pca", "ridge"):
            rows.append({"vintage": v.date(), "pool": pool, "model": k, "hyper": str(fm[k].hyper_),
                         "n_pairs": len(fm["pairs"]), **{c: round(x, 3) for c, x in fm[k].elasticities().items()}})
    return pd.DataFrame(rows)


def report() -> str:
    d = anthropic_table()
    order = PANEL + ["SS claims only (panel labels)"] + EXISTING + [NAIVE]
    out = [f"# Pooled two-company panel model: backtest ({ASOF} data)", "",
           "Specification: the docstring of `panel_model.py`, fixed before the first run. Labels: the claims-only",
           "state-space path of each company, refit on claims public at the vintage; features: changes in",
           "`features.asof_features(v)`; targets: weekly log growth over 4, 8 and 13 weeks; forecast: label at the",
           "last anchor public at v plus predicted growth. Filing rows are never inputs. Error = log(actual / forecast),",
           "negative = forecast too high. `pooled` = Anthropic + OpenAI training pairs, `own` = target company only",
           "(the pre-declared control). Existing variants are read through `forecast_eval.collect()`; the naive last",
           "claim is `backtest._naive` from `forecast_backtest_<rule>_2026-11-10.csv`.", ""]
    # headline: mean |log error| by rule
    head = []
    for rule in RULES:
        s = summary(d[d.rule == rule], order).set_index("model")
        head.append(s[["n", "mae_log", "mae_log_common", "mae_log_h2_2026"]].add_prefix(f"{rule}: "))
    out += ["## Anthropic targets: mean |log error| by rule", "",
            "`mae_log`: each model's own targets; `mae_log_common`: targets every listed model forecast;",
            "`mae_log_h2_2026`: rev_2026q3, rr_2026_09_s1a, rr_2026_10_s1a.", "",
            pd.concat(head, axis=1).round(3).to_markdown(), ""]
    for rule in RULES:
        g = d[d.rule == rule]
        out += [f"## Vintage rule: {rule}", "", summary(g, order).round(3).to_markdown(index=False), "",
                "H2 2026 targets, % error (+ = forecast too high):", "",
                g[g.target.isin(forecast_eval.H2_TARGETS)].pivot_table(index="model", columns="target", values="pct_err")
                .reindex([m for m in order if m in set(g.model)]).round(1).to_markdown(), "",
                "Per-target log error:", "",
                g.pivot_table(index="target", columns="model", values="log_err")
                .reindex(columns=[m for m in ["panel trend (pooled)", PRIMARY, "panel ridge (pooled)", "panel gbm (pooled)",
                                              "panel pca (own)", "SS claims only (panel labels)"] + EXISTING + [NAIVE]
                                  if m in set(g.model)]).round(3).to_markdown(), ""]
        comps = [(f"panel {k} (own)", f"panel {k} (pooled)") for k in pm.MODELS]
        comps += [("panel trend (pooled)", f"panel {k} (pooled)") for k in ("pca", "ridge", "gbm")]
        comps += [("panel trend (own)", f"panel {k} (own)") for k in ("pca", "ridge", "gbm")]
        comps += [(m, PRIMARY) for m in ["A claims only", "E OpenRouter tokens x price", "G GitHub new-repo share", NAIVE,
                                         "ML all signals (smoother)", "SS claims only (panel labels)"]]
        out += ["Paired differences in |log error| (negative = second model better; 90% bootstrap interval):", "",
                pairs_table(d, rule, comps).round(3).to_markdown(index=False), ""]
    # OpenAI secondary set
    oa = []
    for rule in RULES:
        p = pd.read_csv(csv_path(rule))
        oa.append(p[p.company == "openai"])
    oa = pd.concat(oa, ignore_index=True).dropna(subset=["log_err"])
    o_order = PANEL + ["SS claims only (panel labels)", NAIVE]
    out += ["## Secondary test set: OpenAI claims", "",
            "OpenAI's included, non-forecast point claims (all tiers), each forecast from data public at its vintage",
            "with its own origin excluded. OpenAI's basis (4 weeks x 12, net of cloud partners) is never mixed with",
            "Anthropic's: each company keeps its own level, only growth is pooled.", ""]
    for rule in RULES:
        g = oa[oa.rule == rule]
        out += [f"### {rule}", "", summary(g, o_order).drop(columns=["mae_log_h2_2026"]).round(3).to_markdown(index=False), "",
                pairs_table(g, rule, [(f"panel {k} (own)", f"panel {k} (pooled)") for k in pm.MODELS]
                            + [("SS claims only (panel labels)", PRIMARY), (NAIVE, PRIMARY)]).round(3).to_markdown(index=False), ""]
    out += ["## Run notes", "",
            "- Bugs fixed while running (no change to the specification):",
            "  1. GBM crashed when a feature column had no value in a CV training fold (sklearn's binning rejects an",
            "     all-missing column). Found by the unit tests before any backtest; such columns are now dropped inside",
            "     `GBMPanel.fit` (they cannot split, so the model is the same).",
            "  2. The first full run crashed in the period_end rule at vintages with no anchor at all (the empty anchor",
            "     series had a non-date index). Fixed in `anchors`; the publication results, computed before the crash,",
            "     are unchanged, and no period_end result existed before the fix.",
            "  3. Runtime only: the four worker processes oversubscribed BLAS/OpenMP threads; run with",
            "     OMP_NUM_THREADS=1 (about 3 minutes for both rules).",
            "- Before t0 (the last anchor public at the vintage) every panel model returns the label path, so",
            "  pooled and own forecasts can differ only on targets past t0 (about half the Anthropic set).",
            "- The naive last claim here comes from `forecast_backtest_*`; for revenue targets it differs by up to 10%",
            "  from the one the panel computes itself (`forecast_panel_*.csv`), because that backtest re-dates crossing",
            "  claims per vintage (`crossing='profile'`), while the panel uses the neutral position 0.5.", ""]
    out += ["## What the linear models learned (last Anthropic vintage, 2026-11-09)", "",
            "Coefficients in original units: weekly log growth per unit of weekly feature change (log or share",
            "points); `mom` = weight on the previous weeks' label growth. pca: coefficients implied by the kept components.", "",
            elasticity_table().to_markdown(index=False), ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        run(a.jobs)
    md = report()
    with open(os.path.join(config.OUTPUT_DIR, f"forecast_panel_{ASOF}.md"), "w") as f:
        f.write(md)
    print("report written", flush=True)
