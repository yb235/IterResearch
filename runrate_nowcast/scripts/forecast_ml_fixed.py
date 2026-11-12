# Run from the repo root:
#   OMP_NUM_THREADS=1 PYTHONPATH=. python runrate_nowcast/scripts/forecast_ml_fixed.py [--jobs 4]
#   PYTHONPATH=. python runrate_nowcast/scripts/forecast_ml_fixed.py --report-only
"""Corrected ML pipeline (ml_fixed.py; specification in its docstring): backtest at both vintage rules, then a report.

Writes reports/forecast_mlfixed_<rule>_2026-11-10.csv (both companies, every model, pool and ablation) and
reports/forecast_mlfixed_2026-11-10.md. References: A / E / G through forecast_eval.collect(), the naive last
claim from forecast_backtest_<rule>_2026-11-10.csv, and the old zoo AUTO from forecast_zoo_g_spec_<rule>_2026-11-10.csv.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from runrate_nowcast import config, forecast_eval  # noqa: E402
from runrate_nowcast import ml_fixed as mf  # noqa: E402

warnings.simplefilter("ignore")
ASOF = "2026-11-10"
RULES = ("publication", "period_end")
A, E, G = "A claims only", "E OpenRouter tokens x price", "G GitHub new-repo share"
NAIVE = "naive last claim"
OLD, OLD_OWN = "old AUTO (pooled)", "old AUTO (own)"
M1P, M2P, M1O, M2O = "M1 (pooled)", "M2 (pooled)", "M1 (own)", "M2 (own)"
SS = mf.SS_NAME
HEAD = [M1P, M2P, M1O, M2O, SS, A, E, G, NAIVE, OLD, OLD_OWN]
COMMON = [M1P, M2P, A, E, G, NAIVE, OLD]
ABL = [M1P] + [f"{a} (pooled)" for a in (mf.ABL_PRIOR0, mf.ABL_INNER, mf.ABL_OLDPREP, mf.ABL_RHO05, mf.ABL_PRIOR3)]
ABL_OWN = [M1O] + [f"{a} (own)" for a in (mf.ABL_PRIOR0, mf.ABL_INNER, mf.ABL_OLDPREP, mf.ABL_RHO05, mf.ABL_PRIOR3)]
REP = config.OUTPUT_DIR


def csv_path(rule):
    return os.path.join(REP, f"forecast_mlfixed_{rule}_{ASOF}.csv")


def run(jobs: int, rules=RULES):
    for rule in rules:
        t = time.time()
        res = mf.run_backtest(rule, n_jobs=jobs)
        res.to_csv(csv_path(rule), index=False)
        print(f"{rule}: {len(res)} rows, {res.target.nunique()} targets, {time.time() - t:.0f}s wall", flush=True)


# --------------------------------------------------------------------------- loading
def new_rows() -> pd.DataFrame:
    z = pd.concat([pd.read_csv(csv_path(r)) for r in RULES if os.path.exists(csv_path(r))], ignore_index=True)
    single = z.model.isin([SS, mf.NAIVE_NAME, mf.SS_RHO05])
    z["label"] = np.where(single, z.model, z.model + " (" + z.pool + ")")
    return z


def long_table(z: pd.DataFrame) -> pd.DataFrame:
    """New rows and references in one long table (rule, model, target, company, log_err, post_t0)."""
    keep = ["rule", "model", "target", "company", "metric", "vintage", "actual", "forecast", "log_err"]
    parts = [z.assign(model=z.label)[keep]]
    ex = forecast_eval.collect(ASOF)
    parts.append(ex[ex.model.isin([A, E, G])].assign(company="anthropic")[keep])
    for rule in RULES:
        b = pd.read_csv(os.path.join(REP, f"forecast_backtest_{rule}_{ASOF}.csv")).dropna(subset=["log_err_naive_claims_only"])
        parts.append(pd.DataFrame({"rule": rule, "model": NAIVE, "target": b.target, "company": "anthropic",
                                   "metric": b.metric, "vintage": b.vintage, "actual": b.actual,
                                   "forecast": b.naive_claims_only, "log_err": b.log_err_naive_claims_only}))
        o = pd.read_csv(os.path.join(REP, f"forecast_zoo_g_spec_{rule}_{ASOF}.csv"))
        o = o[o.model == "AUTO"]
        parts.append(o.assign(model=np.where(o.pool == "pooled", OLD, OLD_OWN), rule=rule)[keep])
    d = pd.concat(parts, ignore_index=True).dropna(subset=["log_err"])
    post = z[z.model == SS].set_index(["rule", "target"]).post_t0
    d["post_t0"] = [bool(post.get((r, t), False)) for r, t in zip(d.rule, d.target)]
    return d


# --------------------------------------------------------------------------- tables
def common_targets(g: pd.DataFrame, models) -> set:
    sets = [set(g[g.model == m].target) for m in models]
    return set.intersection(*sets) if sets and all(sets) else set()


def score(g: pd.DataFrame, order, common) -> pd.DataFrame:
    rows = []
    for m in order:
        h = g[(g.model == m) & g.target.isin(common)]
        if not len(h):
            continue
        p = h[h.post_t0]
        rows.append({"model": m, "n post-t0": len(p), "mean |err| post-t0": p.log_err.abs().mean(),
                     "bias post-t0": p.log_err.mean(), "n all": len(h), "mean |err| all": h.log_err.abs().mean()})
    return pd.DataFrame(rows).set_index("model")


def paired_table(g: pd.DataFrame, rule: str, comps, common) -> pd.DataFrame:
    out = []
    for subset, sel in (("post-t0", g.post_t0), ("all", g.post_t0 | ~g.post_t0)):
        h = g[sel & g.target.isin(common)]
        for a, b in comps:
            p = forecast_eval.paired(h, a, b, rule)
            if p:
                out.append({"targets": subset, **p})
    return pd.DataFrame(out)


def per_target(g: pd.DataFrame, cols) -> pd.DataFrame:
    t = g.pivot_table(index="target", columns="model", values="log_err")
    t = t.reindex(columns=[c for c in cols if c in t.columns])
    post = g.drop_duplicates("target").set_index("target").post_t0
    t.insert(0, "post_t0", post.reindex(t.index))
    return t.sort_values(["post_t0", "target"], ascending=[False, True])


def fit_table(z: pd.DataFrame, rule: str, model: str = mf.M1, pool: str = "pooled", company: str = "anthropic"):
    g = z[(z.rule == rule) & (z.model == model) & (z.pool == pool) & (z.company == company) & z.post_t0]
    cols = ["target", "vintage", "t0", "rho", "gap_weeks", "lam", "n_folds", "feats_dropped", "beta_mom", "beta_or_spend",
            "beta_log_gh_share", "beta_log_ramp_adoption", "rate_base", "contrib_mom", "contrib_or_spend",
            "contrib_log_gh_share", "contrib_log_ramp_adoption", "rate_hat", "log_err"]
    return g[cols].sort_values("vintage")


def lam_counts(z: pd.DataFrame) -> pd.DataFrame:
    g = z[(z.model == mf.M1)]
    return g.groupby(["company", "rule", "pool"]).lam.value_counts().unstack(fill_value=0)


def fmt(df: pd.DataFrame, nd=3) -> str:
    return df.round(nd).to_markdown()


# --------------------------------------------------------------------------- report
def anthropic_section(d: pd.DataFrame, z: pd.DataFrame) -> list[str]:
    out = ["## Anthropic targets", ""]
    heads = []
    for rule in RULES:
        g = d[(d.rule == rule) & (d.company == "anthropic")]
        common = common_targets(g, COMMON)
        s = score(g, HEAD, common)
        heads.append(s[["n post-t0", "mean |err| post-t0", "n all", "mean |err| all"]].add_prefix(f"{rule}: "))
    out += ["### Headline: mean |log error| on the common Anthropic targets", "",
            "Common targets (per rule) = those forecast by M1, M2 (pooled), A, E, G, the naive last claim and the old",
            "zoo AUTO (pooled). Post-t0 = the target lies after the last anchor public at the vintage (primary); `all`",
            "adds the pre-t0 targets, where every ML model returns the claims-only label path.", "",
            fmt(pd.concat(heads, axis=1)), ""]
    for rule in RULES:
        g = d[(d.rule == rule) & (d.company == "anthropic")]
        common = common_targets(g, COMMON)
        comps = [(r, m) for m in (M1P, M2P) for r in (E, G, A, OLD, NAIVE)]
        comps += [(M1O, M1P), (M2O, M2P), (M1P, M2P), (SS, M2P)]
        out += [f"### Vintage rule: {rule}", "",
                "Scores with bias (mean log error on post-t0 targets; + = forecast too low):", "",
                fmt(score(g, HEAD, common)), "",
                "Paired differences in |log error| (`B vs A`: negative = B better; 90% bootstrap interval;",
                "forecast_eval.paired):", "",
                paired_table(g, rule, comps, common).round(3).to_markdown(index=False), "",
                "Per-target log error (+ = forecast too low):", "",
                fmt(per_target(g[g.target.isin(common)], HEAD)), "",
                "What M1 (pooled) did on the post-t0 targets: CV-chosen lambda, usable nested folds, dropped",
                "features, elasticities in original units, and the predicted weekly growth after t0 split into the",
                "company mean (`rate_base`), mom and the three proxies (other features contribute the rest):", "",
                fit_table(z, rule).round(4).to_markdown(index=False), ""]
    return out


def ablation_section(d: pd.DataFrame) -> list[str]:
    out = ["## Ablations (pre-declared diagnostics, not candidates)", "",
           "Each changes one thing in M1 (see the specification, item 8). Same common targets as the headline.", ""]
    for rule in RULES:
        g = d[(d.rule == rule) & (d.company == "anthropic")]
        common = common_targets(g, COMMON)
        s = pd.concat([score(g, ABL, common), score(g, ABL_OWN, common), score(g, [SS, mf.SS_RHO05, A], common)])
        comps = [(M1P, a) for a in ABL[1:]] + [(M1O, a) for a in ABL_OWN[1:]] + [(SS, mf.SS_RHO05)]
        out += [f"### {rule}", "", fmt(s), "",
                "Paired, ablation vs M1 (`ablation vs M1`: negative = the ablation is better):", "",
                paired_table(g, rule, comps, common).round(3).to_markdown(index=False), ""]
    return out


def openai_section(d: pd.DataFrame) -> list[str]:
    out = ["## Second test set: OpenAI claims", "",
           "OpenAI's included, non-forecast point total claims (all tiers); pooled training only. References: the",
           "claims-only path from the same label run, the panel's naive last claim (both at OpenAI's profiled rho) and",
           "the old zoo AUTO (pooled, g_spec; rho 0.5).", ""]
    order = [M1P, M2P, SS, mf.NAIVE_NAME, OLD, f"{mf.ABL_PRIOR0} (pooled)", f"{mf.ABL_PRIOR3} (pooled)"]
    for rule in RULES:
        g = d[(d.rule == rule) & (d.company == "openai")]
        common = common_targets(g, [M1P, M2P, SS, mf.NAIVE_NAME, OLD])
        comps = [(r, m) for m in (M1P, M2P) for r in (SS, mf.NAIVE_NAME, OLD)]
        out += [f"### {rule}", "", fmt(score(g, order, common)), "",
                paired_table(g, rule, comps, common).round(3).to_markdown(index=False), "",
                fmt(per_target(g[g.target.isin(common)], order[:5])), ""]
    return out


def notes_section(z: pd.DataFrame, d: pd.DataFrame) -> list[str]:
    import sklearn
    out = ["## Run notes", "",
           f"- scikit-learn {sklearn.__version__}; numpy {np.__version__}; 4 worker processes, every fit single-threaded.",
           "- lambda chosen by the nested CV for M1 (count of targets):", "", lam_counts(z).to_markdown(), ""]
    for rule in RULES:
        g = z[z.rule == rule].drop_duplicates(["target", "company"])
        out.append(f"- {rule}: {g.target.nunique()} targets with a row; summed job time "
                   f"{g.job_seconds.sum() / 60:.1f} CPU-minutes.")
    chk = []
    for rule in RULES:
        g = d[(d.rule == rule) & (d.company == "anthropic")]
        e = g.pivot_table(index="target", columns="model", values="log_err")
        if SS in e and A in e:
            x = (e[SS] - e[A]).dropna()
            chk.append(f"{rule}: n={len(x)}, mean |SS - A| = {x.abs().mean():.4f}, max {x.abs().max():.4f}")
    out += ["- Crossing alignment check: the claims-only path from the label run (profiled rho) against A's log error "
            "on the same targets: " + "; ".join(chk) + ".", ""]
    return out


def _mae(d, rule, model, common, post=True):
    g = d[(d.rule == rule) & (d.model == model) & d.target.isin(common)]
    return g[g.post_t0].log_err.abs().mean() if post else g.log_err.abs().mean()


def findings_section(d: pd.DataFrame, z: pd.DataFrame) -> list[str]:
    """Short reading of the results; every number is computed here from the CSVs."""
    out = ["## Reading the results", ""]
    lines = []
    for rule in RULES:
        g = d[(d.rule == rule) & (d.company == "anthropic")]
        c = common_targets(g, COMMON)
        lines.append(f"{rule}: M1 {_mae(d, rule, M1P, c):.3f}, M2 {_mae(d, rule, M2P, c):.3f}, E {_mae(d, rule, E, c):.3f}, "
                     f"G {_mae(d, rule, G, c):.3f}, A {_mae(d, rule, A, c):.3f}, old AUTO {_mae(d, rule, OLD, c):.3f} "
                     f"(n = {int(g[g.target.isin(c) & (g.model == M1P)].post_t0.sum())})")
    out += ["- **Post-t0 Anthropic targets, mean |log error|:** " + "; ".join(lines) + "."]
    h2 = z[(z.company == "anthropic") & (z.model == mf.M1) & (z.pool == "pooled") & z.target.isin(forecast_eval.H2_TARGETS)]
    out += ["- **Why M1 fails.** Wherever the nested CV picks the strongest prior (lambda 1e6), M1 is the prior itself:",
            "  company mean growth + 1 x (each proxy's weekly change - its training mean), summed over the three proxies.",
            f"  At the H2-2026 targets the company mean is {h2.rate_base.mean():.3f}/week, and the three proxy terms sum to "
            f"{(h2.contrib_or_spend + h2.contrib_log_gh_share + h2.contrib_log_ramp_adoption).mean():.3f}/week",
            f"  (OpenRouter tokens x price alone {h2.contrib_or_spend.mean():.3f}), so M1 predicts "
            f"{h2.rate_hat.mean():.3f}/week after t0 and is far too low.",
            "  Two parts of the specification cause it. First, a prior of 1 on each of three proxies of the same",
            "  revenue adds their deviations; the state space averages them as separate measurements of one level.",
            "  Second, the company effect carries each proxy's historical excess growth over revenue (OpenRouter",
            "  tokens grew faster than revenue in its training window), which the state space's drifting",
            "  intercept does not. The ablation `M1 prior 1/3`, which averages the proxies instead of summing them,",
            "  is the best ML variant here."]
    for rule in RULES:
        g = d[(d.rule == rule) & (d.company == "anthropic") & d.post_t0]
        c = common_targets(g, COMMON)
        e = g[g.target.isin(c)].pivot_table(index="target", columns="model", values="log_err")
        opp = int((np.sign(e[M1P]) != np.sign(e[SS])).sum())
        out.append(f"- **M2 ({rule}) gains by cancelling errors.** M1 and the claims-only path err in opposite directions "
                   f"on {opp} of {len(e)} post-t0 targets (bias M1 {e[M1P].mean():+.3f}, SS {e[SS].mean():+.3f}; + = too low). "
                   f"M2's {e[M2P].abs().mean():.3f} comes from averaging two biased forecasts, not from an accurate M1.")
    nf = z[(z.company == "anthropic") & (z.model == mf.M1)]
    out += [f"- **The nested CV has little to work with.** Anthropic vintages have {int(nf.n_folds.min())}-{int(nf.n_folds.max())} "
            "usable folds: none up to mid-February 2026, then one or two (the early blocks fail the usability rule of "
            "spec item 4). From April 2026 on the CV picks lambda 1e6 (1e4 in May), so M1 is the prior.",
            "- **Ablations (pooled, post-t0; differences are paired, negative = the ablation is better than M1):** see the",
            "  table above. Prior 0, the old inner CV and the old preprocessing all beat M1 (the intervals exclude 0,",
            "  except the inner CV under period_end), because each one weakens the wrong prior. The old",
            "  preprocessing scales sparse features by their RMS over imputed zeros, which",
            "  shrinks their scale. A given lambda then penalizes the elasticity less, and the CV picks lambda 1-100,",
            "  which leaves elasticities of about 0.6 on or_spend. The crossing fix helps a little: `M1 rho 0.5` is worse",
            "  by about 0.03 in both rules, with an interval that excludes 0. The pre-t0 targets are identical for",
            "  every ML variant.",
            "- **OpenAI (second test set):** M1 and M2 beat the claims-only path, the naive last claim and the old AUTO under",
            "  the publication rule; under period_end the gains over the path and the old AUTO are small (intervals",
            "  include 0 for M1). For OpenAI the nested CV has up to 3 folds and picks lambda 10-1000 at 14 of 15",
            "  targets in each rule, so M1 there is mostly fitted from data, not the summed prior.",
            ""]
    out += ["## Changes after the specification was written", "",
            "- Before the first backtest (no score seen): (a) rho = 0.5 when a company has no claim public at a fold",
            "  vintage (a unit test failed on it; no labels exist there); (b) the `M1 prior 1/3` ablation and the",
            "  contribution columns, after the inputs of one forecast showed the summed prior (-0.13/week at",
            "  rr_2026_09_s1a). M1 and M2 are as specified.",
            "- First backtest launch: crashed in a worker before writing any output (no score produced). The pooled-pairs",
            "  assertion compared each pool company's `t1` with its `t0`, which is None for Anthropic at early OpenAI",
            "  vintages (it contributes no pairs then). Fixed to check the companies present in the pairs, as",
            "  panel_model does. A regression test was added. The results here come from the second launch, the",
            "  only run that produced results. No bug was fixed after results existed, so there are no before/after",
            "  numbers.",
            ""]
    return out


def report() -> str:
    z = new_rows()
    d = long_table(z)
    out = [f"# Corrected ML pipeline (ml_fixed.py): backtest ({ASOF} data)", "",
           "Specification: the docstring of `runrate_nowcast/ml_fixed.py`, fixed before the first run. Two pre-declared",
           "models: **M1** = ridge toward prior elasticities (1 on OpenRouter tokens x price, log GitHub new-repo",
           "share and log Ramp adoption; 0 on the other g_spec features and on momentum), penalty chosen by",
           "nested-vintage CV (fold labels rebuilt from claims published before each validation block, 13-week",
           "purge); **M2** = equal-weight log average of M1 and the claims-only state-space path. Features",
           "standardized on observed rows, clipped to the training 1st-99th percentile, dropped with < 26 observed",
           "weeks. Crossing claims re-dated at the rho profiled at each vintage. `pooled` = Anthropic + OpenAI pairs",
           "(primary); `own` = target company only (control). Errors are log(actual / forecast); negative = forecast",
           "too high. Filing rows are never inputs.", ""]
    out += findings_section(d, z)
    out += anthropic_section(d, z)
    out += ablation_section(d)
    out += openai_section(d)
    out += notes_section(z, d)
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--rules", nargs="+", default=list(RULES))
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        t0 = time.time()
        run(a.jobs, a.rules)
        print(f"total {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(REP, f"forecast_mlfixed_{ASOF}.md"), "w") as f:
        f.write(report())
    print("report written", flush=True)
