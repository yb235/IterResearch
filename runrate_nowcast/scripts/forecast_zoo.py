# Run from the repo root:
#   OMP_NUM_THREADS=1 PYTHONPATH=. python runrate_nowcast/scripts/forecast_zoo.py [--feature-set g_spec] [--jobs 4]
#   OMP_NUM_THREADS=1 PYTHONPATH=. python runrate_nowcast/scripts/forecast_zoo.py --feature-set all
#   PYTHONPATH=. python runrate_nowcast/scripts/forecast_zoo.py --report-only
"""Model zoo (model_zoo.py; protocol in its docstring): backtest at both vintage rules, then a report.

Writes reports/forecast_zoo_<feature_set>_<rule>_2026-11-10.csv (both companies) and
reports/forecast_zoo_2026-11-10.md, which covers every feature set whose CSVs exist. References:
A / E / G through forecast_eval.collect(), the naive last claim from forecast_backtest_<rule>_2026-11-10.csv
(as in forecast_panel.py), and the earlier panel models (forecast_panel_<rule>) for context.
"""
import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from runrate_nowcast import config, data, forecast_eval  # noqa: E402
from runrate_nowcast import model_zoo as mz  # noqa: E402

warnings.simplefilter("ignore")
ASOF = "2026-11-10"
RULES = ("publication", "period_end")
COLS = ["target", "company", "metric", "vintage", "actual", "forecast", "log_err", "model", "feature_set", "pool",
        "rule", "auto_choice", "stack_weights", "hyper", "cv_loss", "cv_folds", "n_train_pairs", "n_target_pairs",
        "n_anchors", "t0", "gap_weeks", "job_seconds"]
REFS = ["A claims only", "E OpenRouter tokens x price", "G GitHub new-repo share"]
NAIVE = "naive last claim"
PANEL_OLD = ["P panel pca (pooled)", "P panel ridge (pooled)"]
SS = mz.SS_NAME
AUTO, STACK, AUTO_OWN, STACK_OWN = "AUTO (pooled)", "stack (pooled)", "AUTO (own)", "stack (own)"
HEAD = [AUTO, STACK, AUTO_OWN, STACK_OWN]


def csv_path(fs, rule):
    return os.path.join(config.OUTPUT_DIR, f"forecast_zoo_{fs}_{rule}_{ASOF}.csv")


def run(fs: str, jobs: int, rules=RULES):
    mz.feature_spec(fs)                       # fails fast if the feature set is not defined
    for rule in rules:
        t = time.time()
        res = mz.run_backtest(rule, feature_set=fs, n_jobs=jobs)
        res = res.reindex(columns=COLS)
        res.to_csv(csv_path(fs, rule), index=False)
        print(f"{fs} {rule}: {len(res)} rows, {res.target.nunique()} targets, {time.time() - t:.0f}s wall", flush=True)


# --------------------------------------------------------------------------- loading
def feature_sets_present() -> list[str]:
    fs = set()
    for f in glob.glob(os.path.join(config.OUTPUT_DIR, f"forecast_zoo_*_publication_{ASOF}.csv")):
        fs.add(re.match(r".*forecast_zoo_(.+)_publication_", f).group(1))
    return sorted(fs, key=lambda s: (s != "g_spec", s))


def zoo_rows(fs: str) -> pd.DataFrame:
    parts = []
    for rule in RULES:
        if os.path.exists(csv_path(fs, rule)):
            parts.append(pd.read_csv(csv_path(fs, rule)))
    z = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLS)
    z = z.dropna(subset=["log_err"]).copy()
    z["label"] = np.where(z.model.isin([SS, mz.NAIVE_NAME]), z.model, z.model + " (" + z.pool + ")")
    return z


def anthropic_table(z: pd.DataFrame) -> pd.DataFrame:
    """Zoo rows for Anthropic targets plus the references, in forecast_eval's long format."""
    obs = data.load().set_index("obs_id")
    keep = ["rule", "model", "target", "metric", "vintage", "actual", "forecast", "log_err"]
    za = z[(z.company == "anthropic") & (z.model != mz.NAIVE_NAME)].assign(model=lambda x: x.label)[keep]
    ex = forecast_eval.collect(ASOF)
    ex = ex[ex.model.isin(REFS + PANEL_OLD)][keep]
    parts = [za, ex]
    for rule in RULES:
        b = pd.read_csv(os.path.join(config.OUTPUT_DIR, f"forecast_backtest_{rule}_{ASOF}.csv"))
        b = b.dropna(subset=["log_err_naive_claims_only"])
        parts.append(pd.DataFrame({"rule": rule, "model": NAIVE, "target": b.target, "metric": b.metric,
                                   "vintage": b.vintage, "actual": b.actual, "forecast": b.naive_claims_only,
                                   "log_err": b.log_err_naive_claims_only}))
    d = pd.concat(parts, ignore_index=True).dropna(subset=["log_err"])
    d["filing_target"] = d.target.map(lambda t: int(obs.loc[t, "filing"]) == 1)
    d["pct_err"] = 100 * (d.forecast / d.actual - 1)
    return d


# --------------------------------------------------------------------------- tables
def common_targets(g: pd.DataFrame, models) -> set:
    sets = [set(g[g.model == m].target) for m in models if m in set(g.model)]
    return set.intersection(*sets) if sets else set()


def score_table(g: pd.DataFrame, order, common_models) -> pd.DataFrame:
    common = common_targets(g, common_models)
    rows = []
    for m in order:
        h = g[g.model == m]
        if not len(h):
            continue
        hc = h[h.target.isin(common)]
        rows.append({"model": m, "n": len(h), "mae_log": h.log_err.abs().mean(), "n_common": len(hc),
                     "mae_log_common": hc.log_err.abs().mean(), "bias_log_common": hc.log_err.mean(),
                     "mae_log_h2_2026": h[h.target.isin(forecast_eval.H2_TARGETS)].log_err.abs().mean()})
    return pd.DataFrame(rows)


def member_table(g: pd.DataFrame, pools=("pooled", "own")) -> pd.DataFrame:
    """Every member's mean |log error| (on the targets AUTO forecast), per pool, ranked."""
    out = []
    for pool in pools:
        auto = g[g.model == f"AUTO ({pool})"]
        if not len(auto):
            continue
        tg = set(auto.target)
        for m in mz.MEMBERS + ("AUTO", "stack"):
            h = g[(g.model == f"{m} ({pool})") & g.target.isin(tg)]
            if len(h):
                out.append({"pool": pool, "model": m, "n": len(h), "mae_log": h.log_err.abs().mean(),
                            "bias_log": h.log_err.mean(),
                            "mae_log_h2_2026": h[h.target.isin(forecast_eval.H2_TARGETS)].log_err.abs().mean()})
    t = pd.DataFrame(out)
    if not len(t):
        return t
    wide = t.pivot_table(index="model", columns="pool", values="mae_log")
    wide = wide.reindex(columns=[p for p in pools if p in wide.columns])
    for p in wide.columns:
        mem = wide.loc[[m for m in mz.MEMBERS if m in wide.index], p]
        wide[f"rank_{p}"] = mem.rank(method="min").reindex(wide.index)
    sort_col = "pooled" if "pooled" in wide.columns else wide.columns[0]
    return wide.sort_values(sort_col)


def spread_table(g: pd.DataFrame, pools=("pooled", "own")) -> pd.DataFrame:
    rows = []
    for pool in pools:
        auto = g[g.model == f"AUTO ({pool})"]
        if not len(auto):
            continue
        tg = set(auto.target)
        mae = pd.Series({m: g[(g.model == f"{m} ({pool})") & g.target.isin(tg)].log_err.abs().mean()
                         for m in mz.MEMBERS}).dropna()
        a = auto.log_err.abs().mean()
        s = g[(g.model == f"stack ({pool})") & g.target.isin(tg)].log_err.abs().mean()
        rows.append({"pool": pool, "n_targets": len(tg), "n_members": len(mae), "best member (optimistic)": mae.idxmin(),
                     "best": mae.min(), "p25": mae.quantile(0.25), "median": mae.median(), "p75": mae.quantile(0.75),
                     "worst": mae.max(), "worst member": mae.idxmax(), "AUTO": a,
                     "AUTO rank among members": int((mae < a - 1e-12).sum()) + 1, "stack": s,
                     "LOO best-of-zoo": loo_best(g, pool, tg)})
    return pd.DataFrame(rows)


def loo_best(g: pd.DataFrame, pool: str, tg: set) -> float:
    """Selection-optimism check: for each target, the member with the lowest mean |log error| on
    the OTHER targets, scored on this one (uses later targets, so not a real-time model)."""
    e = {m: g[(g.model == f"{m} ({pool})") & g.target.isin(tg)].set_index("target").log_err.abs()
         for m in mz.MEMBERS}
    E = pd.DataFrame(e).dropna(axis=1, how="any")
    if E.shape[0] < 2 or E.shape[1] == 0:
        return np.nan
    out = []
    for t in E.index:
        best = E.drop(index=t).mean().idxmin()
        out.append(E.loc[t, best])
    return float(np.mean(out))


def picks_table(z: pd.DataFrame, rule: str, company: str) -> pd.DataFrame:
    g = z[(z.rule == rule) & (z.company == company) & (z.model == "AUTO")]
    if not len(g):
        return pd.DataFrame()
    w = g.pivot_table(index=["target", "vintage", "gap_weeks"], columns="pool", values="auto_choice", aggfunc="first")
    le = g.pivot_table(index=["target", "vintage", "gap_weeks"], columns="pool", values="log_err")
    w = w.join(le.add_prefix("log_err AUTO ").round(3))
    return w.reset_index().sort_values("vintage")


def pick_counts(z: pd.DataFrame, company: str) -> pd.DataFrame:
    g = z[(z.company == company) & (z.model == "AUTO")]
    return g.groupby(["rule", "pool"]).auto_choice.value_counts().unstack(fill_value=0)


def stack_weight_table(z: pd.DataFrame, company: str) -> pd.DataFrame:
    g = z[(z.company == company) & (z.model == "stack")].dropna(subset=["stack_weights"])
    rows = []
    for r in g.itertuples():
        w = {k: float(v) for k, v in (kv.split("=") for kv in str(r.stack_weights).split(";") if "=" in kv)}
        rows.append({"rule": r.rule, "pool": r.pool, "target": r.target, **w})
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows).fillna(0.0)
    mem = [m for m in mz.MEMBERS if m in t.columns]
    avg = t.groupby(["rule", "pool"])[mem].mean()
    return avg.loc[:, (avg > 0.005).any()].round(3)


def pairs_table(d, rule, comps):
    out = [forecast_eval.paired(d, a, b, rule) for a, b in comps]
    return pd.DataFrame([p for p in out if p])


# --------------------------------------------------------------------------- report
def anthropic_section(fs: str, z: pd.DataFrame) -> list[str]:
    d = anthropic_table(z)
    members_p = [f"{m} (pooled)" for m in mz.MEMBERS]
    order = HEAD + [SS] + REFS + [NAIVE] + PANEL_OLD
    out = []
    head = []
    for rule in RULES:
        s = score_table(d[d.rule == rule], order, HEAD + REFS + [NAIVE]).set_index("model")
        head.append(s[["n_common", "mae_log_common", "mae_log_h2_2026"]].add_prefix(f"{rule}: "))
    out += [f"### Headline: mean |log error| on common Anthropic targets ({fs})", "",
            "Common targets = those forecast by AUTO, stack (both pools), A, E, G and the naive last claim.",
            "AUTO (pooled) is the pre-declared headline model.", "",
            pd.concat(head, axis=1).round(3).to_markdown(), ""]
    for rule in RULES:
        g = d[d.rule == rule]
        if not len(g):
            continue
        out += [f"### Vintage rule: {rule} ({fs})", "",
                "All models, own targets and common targets (bias: mean log error, + = forecast too low):", "",
                score_table(g, order, HEAD + REFS + [NAIVE]).round(3).to_markdown(index=False), "",
                "Every member, mean |log error| on the targets AUTO forecast. The best member is selected after",
                "the fact and therefore optimistic; ranks are among the members only:", "",
                member_table(g).round(3).to_markdown(), "",
                "Spread of member scores, AUTO's rank, and a leave-one-target-out best-of-zoo (pick the member",
                "with the lowest error on the other targets, score it on the held-out one):", "",
                spread_table(g).round(3).to_markdown(index=False), "",
                "AUTO's pick and log error per target:", "",
                picks_table(z, rule, "anthropic").to_markdown(index=False), "",
                "H2 2026 targets, % error (+ = forecast too high):", "",
                g[g.target.isin(forecast_eval.H2_TARGETS)].pivot_table(index="model", columns="target", values="pct_err")
                .reindex([m for m in order + members_p if m in set(g.model)]).round(1).to_markdown(), ""]
        comps = [(r, AUTO) for r in REFS + [NAIVE, SS] + PANEL_OLD]
        comps += [(r, STACK) for r in REFS + [NAIVE, SS]]
        comps += [(AUTO_OWN, AUTO), (STACK_OWN, STACK), (STACK, AUTO)]
        out += ["Paired differences in |log error| (`B vs A`: negative = B better; 90% bootstrap interval):", "",
                pairs_table(d, rule, comps).round(3).to_markdown(index=False), "",
                "Per-target log error:", "",
                g.pivot_table(index="target", columns="model", values="log_err")
                .reindex(columns=[m for m in HEAD + [SS] + REFS + [NAIVE] if m in set(g.model)]).round(3).to_markdown(), ""]
    out += ["### Stack weights (mean over targets; members with any mean weight > 0.005)", "",
            stack_weight_table(z, "anthropic").to_markdown(), "",
            "### AUTO's picks (number of Anthropic targets)", "", pick_counts(z, "anthropic").to_markdown(), ""]
    return out


def openai_section(fs: str, z: pd.DataFrame) -> list[str]:
    o = z[z.company == "openai"].assign(model=lambda x: x.label)
    if not len(o):
        return []
    naive = mz.NAIVE_NAME
    order = [AUTO, STACK, SS, naive]
    out = [f"## Second test set: OpenAI claims ({fs})", "",
           "OpenAI's included, non-forecast point total claims (all tiers), pooled training only. The naive last",
           f"claim here is the panel's own (`{naive}`, neutral crossing position).", ""]
    for rule in RULES:
        g = o[o.rule == rule]
        if not len(g):
            continue
        out += [f"### {rule}", "", score_table(g, order, order).drop(columns=["mae_log_h2_2026"]).round(3)
                .to_markdown(index=False), "",
                "Members (pooled), spread and AUTO's rank:", "",
                member_table(g, pools=("pooled",)).round(3).to_markdown(), "",
                spread_table(g, pools=("pooled",)).round(3).to_markdown(index=False), "",
                pairs_table(g, rule, [(SS, AUTO), (naive, AUTO), (SS, STACK), (naive, STACK), (STACK, AUTO)])
                .round(3).to_markdown(index=False), ""]
    out += ["AUTO's picks (number of OpenAI targets):", "", pick_counts(z, "openai").to_markdown(), ""]
    return out


def _r3(x):
    return f"{x:.3f}" if isinstance(x, (float, np.floating)) and np.isfinite(x) else x


def side_by_side(zs: dict) -> list[str]:
    """Every feature set next to the others: headline models, member spread, members, pairs, OpenAI."""
    if len(zs) < 2:
        return []
    fss = list(zs)
    members_p = [f"{m} (pooled)" for m in mz.MEMBERS]
    parts, refs = [], None
    for fs, z in zs.items():
        a = anthropic_table(z)
        parts.append(a[a.model.isin(HEAD + members_p)].assign(model=lambda x, fs=fs: x.model + " [" + fs + "]"))
        if refs is None:
            refs = a[a.model.isin([SS] + REFS + [NAIVE])]
    d = pd.concat(parts + [refs], ignore_index=True)
    out = ["## Feature sets side by side", "",
           "Same protocol, same zoo, same AUTO rule and controls; only the feature columns differ (g_spec: the 8",
           "features of sub-task G; all: every column of `features.FEATURE_SETS[\"all\"]`, mapped by the protocol's",
           "rule). Mean |log error| on the Anthropic targets that every model below forecast (per rule). Members:",
           "pooled. `best member` is chosen after seeing the test errors and is optimistic.", ""]
    rows = {}
    for rule in RULES:
        g = d[d.rule == rule]
        heads = [f"{m} [{fs}]" for fs in fss for m in HEAD] + [SS] + REFS + [NAIVE]
        common = common_targets(g, heads)
        g = g[g.target.isin(common)]
        mae = g.groupby("model").log_err.apply(lambda x: x.abs().mean())
        for fs in fss:
            col = f"{rule}: {fs} (n={len(common)})"
            mem = mae.reindex([f"{m} (pooled) [{fs}]" for m in mz.MEMBERS]).dropna()
            vals = {m: mae.get(f"{m} [{fs}]", np.nan) for m in HEAD}
            vals.update({"best member (optimistic)": mem.min(), "median member": mem.median(), "worst member": mem.max(),
                         "best member is": mem.idxmin().split(" (")[0] if len(mem) else ""})
            vals.update({r: mae.get(r, np.nan) for r in [SS] + REFS + [NAIVE]})
            rows[col] = vals
    t = pd.DataFrame(rows).map(_r3)
    out += ["### Anthropic targets", "", t.to_markdown(), ""]
    mt = {}
    for rule in RULES:
        g = d[d.rule == rule]
        common = common_targets(g, [f"{m} [{fs}]" for fs in fss for m in HEAD])
        g = g[g.target.isin(common)]
        for fs in fss:
            mt[f"{rule}: {fs}"] = pd.Series({m: g[g.model == f"{m} (pooled) [{fs}]"].log_err.abs().mean()
                                             for m in mz.MEMBERS})
    out += ["Every member (pooled), mean |log error| on the targets both AUTO versions forecast:", "",
            pd.DataFrame(mt).round(3).to_markdown(), ""]
    comps = [(f"{m} [g_spec]", f"{m} [{fs}]") for fs in fss if fs != "g_spec" for m in HEAD]
    for rule in RULES:
        out += [f"Paired, {rule} (`B vs A`: negative = B better; 90% bootstrap interval):", "",
                pairs_table(d, rule, comps).round(3).to_markdown(index=False), ""]
    # OpenAI
    orows = {}
    for rule in RULES:
        for fs, z in zs.items():
            o = z[(z.company == "openai") & (z.rule == rule)].assign(model=lambda x: x.label)
            names = [AUTO, STACK, SS, mz.NAIVE_NAME]
            common = common_targets(o, names)
            o = o[o.target.isin(common)]
            mem = pd.Series({m: o[o.model == f"{m} (pooled)"].log_err.abs().mean() for m in mz.MEMBERS}).dropna()
            vals = {m: o[o.model == m].log_err.abs().mean() for m in names}
            vals.update({"best member (optimistic)": mem.min(), "median member": mem.median(),
                         "best member is": mem.idxmin() if len(mem) else ""})
            orows[f"{rule}: {fs} (n={len(common)})"] = vals
    out += ["### OpenAI targets (pooled)", "", pd.DataFrame(orows).map(_r3).to_markdown(), ""]
    return out


def run_notes(zs: dict) -> list[str]:
    import sklearn
    vers = [f"scikit-learn {sklearn.__version__}"]
    for mod in ("lightgbm", "xgboost"):
        try:
            vers.append(f"{mod} {__import__(mod).__version__}")
        except Exception:      # noqa: BLE001
            vers.append(f"{mod} not installed (member dropped)")
    out = ["## Run notes", "", f"- Packages: {', '.join(vers)}. Members dropped: {', '.join(mz.DROPPED) or 'none'}.",
           "- Every member single-threaded (OMP_NUM_THREADS=1); 4 worker processes across targets."]
    for fs, z in zs.items():
        for rule in RULES:
            g = z[z.rule == rule].drop_duplicates(["target", "company"])
            if len(g):
                out.append(f"- {fs} / {rule}: {g.target.nunique()} targets forecast; summed job time "
                           f"{g.job_seconds.sum() / 60:.1f} CPU-minutes (about a quarter of that in wall time on 4 workers).")
    for fs in zs:
        if fs != "g_spec":
            src = mz.F.FEATURE_SETS.get(fs, [])
            out.append(f"- Feature set `{fs}`: {len(src)} columns of `features.FEATURE_SETS['{fs}']`, "
                       f"{len(mz.feature_spec(fs))} features after the protocol's mapping (duplicate change columns "
                       "dropped; changes and rates cumulated); each also gets the three MIDAS terms.")
    if "all" in zs:
        out += ["- The first `all` run was stopped and discarded: `features.py` was being finalised by another sub-task",
                "  while it ran, and its workers had loaded an earlier draft. The `all` results here are a rerun on the",
                "  committed feature store (commit fd0c449). The g_spec columns of that store equal those of the original",
                "  store (checked at 9 vintages for both companies), so the earlier g_spec run stands.",
                "- The look-ahead guarantee of the extra `all` columns rests on `features.py`'s as-of filter and its own",
                "  tests; this module's end-to-end no-look-ahead test covers the g_spec run."]
    out += ["- The MIDAS member was revised once before the first backtest (see the protocol in `model_zoo.py`): the",
            "  first draft put lags 0..13 under a single Almon polynomial, which cannot represent a contemporaneous-only",
            "  effect; a synthetic unit test showed this. No test target had been forecast at that point.",
            "- The naive last claim for Anthropic comes from `forecast_backtest_*` (crossing claims re-dated per vintage);",
            "  the panel's own naive (neutral crossing position) is in the CSVs as `naive last claim (panel)`.",
            "- Before t0 (the last anchor public at the vintage) every member returns the label path, so members can",
            "  differ only on targets past t0; with few such targets, differences between members are noisy.", ""]
    return out


def report() -> str:
    fss = feature_sets_present()
    zs = {fs: zoo_rows(fs) for fs in fss}
    out = [f"# Model zoo on the pooled two-company panel: backtest ({ASOF} data)", "",
           "Protocol: the docstring of `model_zoo.py`, fixed before the first run. Labels, training pairs, forecasts",
           "and targets are those of `panel_model.py`. 18 members (" + ", ".join(mz.MEMBERS) + ");",
           "hyperparameters by purged time-ordered CV inside the training pairs public at each vintage.",
           "**AUTO** (declared in advance as the headline) = at each vintage, the member with the lowest inner-CV",
           "error on training pairs. **stack** = non-negative weights summing to one on the members' out-of-fold",
           "predictions. `pooled` = Anthropic + OpenAI pairs; `own` = Anthropic only (pre-declared control).",
           "Errors are log(actual / forecast); negative = forecast too high. Filing rows are never inputs.",
           f"Feature sets in this report: {', '.join(fss)}.", ""]
    out += side_by_side(zs)
    for fs, z in zs.items():
        out += [f"## Anthropic targets ({fs})", ""] + anthropic_section(fs, z)
    for fs, z in zs.items():
        out += openai_section(fs, z)
    out += run_notes(zs)
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="g_spec")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--rules", nargs="+", default=list(RULES))
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        t0 = time.time()
        run(a.feature_set, a.jobs, a.rules)
        print(f"total {time.time() - t0:.0f}s", flush=True)
    md = report()
    with open(os.path.join(config.OUTPUT_DIR, f"forecast_zoo_{ASOF}.md"), "w") as f:
        f.write(md)
    print("report written", flush=True)
