# Run from the repo root: PYTHONPATH=. python runrate_nowcast/scripts/eval_pubfilings.py
"""Sub-task O: score the "pub" evidence rule (filing numbers used once public) against sub-task N's
no-filing runs under the same lower-claim rule (DATA_PLAN 子任务 O).

Inputs (reports/):
  forecast_interval_<model>_<variant>_<rule>_2026-11-13.csv    no-filing runs of sub-task N
  forecast_pubfilings_<model>_<variant>_<rule>_2026-11-13.csv  pub runs (scripts/forecast_pubfilings.py)
  pubfilings_answer_independence_2026-11-13.csv              scripts/check_pubfilings_independence.py
  eval_interval_long_2026-11-13.csv                          N's post-t0 flags (kept as N defined them)
Models A, B, E, G, R (run_backtest) and bayes (laplace_mixture rows, as N). Primary lower-claim rule
iv115; point is a sensitivity (A, B, E, G, R only: N ran no bayes point).
Naive = backtest._naive, the last point run-rate claim (revenue: those claims joined log-linearly and
integrated over the period), taken from model A's iv115 file of the same evidence rule, so the pub naive
may carry forward a filing = 2 run-rate number. Errors are log(actual / forecast): + = forecast too low.

Target sets per vintage rule:
  all      targets every model x evidence rule forecast (common targets)
  post-t0  N's flag (eval_interval_long_2026-11-13.csv), unchanged so the subsets match N's
  H2-2026  rev_2026q3, rr_2026_09_s1a, rr_2026_10_s1a
  changed  targets with at least one filing row in their pub evidence
Paired comparisons: forecast_eval.paired (difference in |log error|, 10000 bootstrap draws, 90% interval).

Outputs: reports/eval_pubfilings_2026-11-13.md, reports/eval_pubfilings_long_2026-11-13.csv.
"""
import warnings; warnings.simplefilter("ignore")
import math, os
import numpy as np
import pandas as pd

from runrate_nowcast import config, data, forecast_eval

ASOF = "2026-11-13"
REP = config.OUTPUT_DIR
RULES = ("publication", "period_end")
MODELS = ("A", "B", "E", "G", "R", "bayes")
LABELS = {"A": "A claims only", "B": "B + indicators", "E": "E OpenRouter tokens x price",
          "G": "G GitHub new-repo share", "R": "R Ramp adoption", "bayes": "Bayes + indicators"}
NAIVE = "N naive last claim"
PRIMARY, SENS = "iv115", "point"
H2 = forecast_eval.H2_TARGETS
Z80 = 1.2816


# Written after the runs, from the tables this script prints (the numbers in the tables are authoritative).
READING = [
    "## Reading (written after the runs)", "",
    "- Publication rule, 9 changed targets: using the public filing numbers lowers mean |log error| for A (0.076 ->",
    "  0.031), B (0.144 -> 0.060), E (0.098 -> 0.046) and Bayes (0.123 -> 0.059); the paired 90% intervals exclude",
    "  zero for all four. G (0.034 -> 0.033) and R (0.055 -> 0.044) barely move: they were already close.",
    "- H2 2026 (3 targets, publication): every model improves (A 0.157 -> 0.043, B 0.336 -> 0.114, Bayes 0.276 -> 0.111),",
    "  but none beats the naive last claim, which also improves (0.056 -> 0.041). B and Bayes are still worse than the",
    "  naive (paired +0.07, intervals above zero); A, E, G, R are level with it. B still puts the Oct 31 run-rate at",
    "  90.7 (actual 81, outside its 80% interval) with Q3 GAAP, Sacra ~70, SemiAnalysis ~74 and the WSJ '>80' in hand.",
    "- Period_end rule, 3 changed targets: mixed. All of the gain is rr_2026_10_s1a (A 112 -> 76.7, B 159 -> 115,",
    "  E 116 -> 89.4, Bayes 145 -> 122). rev_2026q3 and rr_2026_09_s1a, whose only new input is Fortune's Q1-2026",
    "  revenue (4.73), get slightly worse for most models (G rev_2026q3 +0.023 -> +0.085; Bayes rr_2026_09_s1a",
    "  -0.439 -> -0.488). Only E's interval excludes zero on the improving side; G's excludes zero on the worse side.",
    "- The six S-1 revenue targets forecast on 2026-10-25 see one new row, Fortune's Q1-2026 4.73 (2026-09-30): the",
    "  error shrinks on 5 of 6 for A, B and E, 4 of 6 for R and Bayes, 3 of 6 for G. The moves are small (at most 0.016",
    "  log for A, 0.053 for G).",
    "- 80% coverage: publication A 0.73 -> 0.80, B 0.60 -> 0.67, E 0.87 -> 1.00, Bayes 0.80 -> 0.93, G and R",
    "  unchanged; period_end unchanged for every model.",
    "- Point sensitivity (A B E G R): same direction. On the publication rule's changed targets B (-0.066), E (-0.050)",
    "  and R (-0.011) gain about as much as under iv115; A gains half as much (-0.021, interval touches zero) because",
    "  A's no-filing point run was already closer; G is flat (+0.005).",
    "- Every target without a filing row in its pub evidence reproduces N's forecast mean bit for bit (check below).",
    "- Caveats: (a) rr_2026_10_semi (filing = 2) may be Q3 GAAP annualized (its notes say so); with rev_2026q3 also in",
    "  the evidence for the run-rate targets that double-counts one number, and it is what the pub naive carries",
    "  forward for rr_2026_10_s1a (74). (b) rev_2026q1's value 4.729 is noted as derived with S-1/A No. 1 numbers",
    "  (2026-11-02) but dated 2026-10-26, so it enters rev_2026q3's evidence (vintage 2026-11-01). Replacing it with the",
    "  10-26-derivable 16.3 - 11.555 = 4.745 moves model A's rev_2026q3 forecast from 17.831 to 17.833 and the run-rate",
    "  forecasts by < 0.01%: immaterial, but the row's date is generous. (c) 9 / 3 targets, one per number, strongly",
    "  correlated across models: the intervals are descriptive, not evidence of a general effect.", ""]


def path(model, variant, rule, evidence):
    pre = "forecast_interval" if evidence == "nofiling" else "forecast_pubfilings"
    return os.path.join(REP, f"{pre}_{model}_{variant}_{rule}_{ASOF}.csv")


def load_one(model, variant, rule, evidence) -> list:
    f = path(model, variant, rule, evidence)
    if not os.path.exists(f):
        return []
    d = pd.read_csv(f)
    rows = []
    if model == "bayes":
        d = d[d.method == "laplace_mixture"]
    for r in d.itertuples():
        rr = r.metric == "run_rate"
        if model == "bayes":
            mu, sd = r.forecast_mu, r.forecast_sd
            fc = math.exp(mu) if rr else mu
            lo, hi = (math.exp(mu - Z80 * sd), math.exp(mu + Z80 * sd)) if rr else (mu - Z80 * sd, mu + Z80 * sd)
            err = r.log_err
        else:
            fc, lo, hi, err = r.forecast, r.q10, r.q90, r.log_err_model
            mu = math.log(fc) if rr else fc
            sd = (math.log(hi) - math.log(lo)) / (2 * Z80) if rr else (hi - lo) / (2 * Z80)
        fil = getattr(r, "filing_evidence", np.nan)
        rows.append(dict(rule=rule, model=LABELS[model], key=model, variant=variant, evidence=evidence, target=r.target,
                         metric=r.metric, vintage=str(r.vintage), actual=r.actual, forecast=fc, q10=lo, q90=hi, mu=mu,
                         sd=sd, log_err=err, z=r.z, in80=bool(r.in80), in50=bool(r.in50),
                         filing_evidence="" if pd.isna(fil) else str(fil), naive_from=""))
        if model == "A" and variant == PRIMARY and pd.notna(r.log_err_naive):
            nf = getattr(r, "naive_from", "")
            rows.append(dict(rule=rule, model=NAIVE, key="naive", variant="-", evidence=evidence, target=r.target,
                             metric=r.metric, vintage=str(r.vintage), actual=r.actual, forecast=r.naive, q10=np.nan,
                             q90=np.nan, mu=np.nan, sd=np.nan, log_err=r.log_err_naive, z=np.nan, in80=np.nan,
                             in50=np.nan, filing_evidence="", naive_from="" if pd.isna(nf) else str(nf)))
    return rows


def load_long() -> pd.DataFrame:
    rows = []
    for m in MODELS:
        for v in (PRIMARY, SENS):
            for rule in RULES:
                for ev in ("nofiling", "pub"):
                    rows += load_one(m, v, rule, ev)
    d = pd.DataFrame(rows)
    n = pd.read_csv(os.path.join(REP, f"eval_interval_long_{ASOF}.csv")).drop_duplicates(["rule", "target"])
    flags = {(r, t): bool(p) for r, t, p in zip(n.rule, n.target, n.post_t0)}
    d["post_t0"] = [flags.get((r, t), False) for r, t in zip(d.rule, d.target)]
    changed = d[(d.evidence == "pub") & (d.filing_evidence != "")].groupby("rule").target.apply(set).to_dict()
    d["changed"] = [t in changed.get(r, set()) for r, t in zip(d.rule, d.target)]
    return d


# --------------------------------------------------------------------------- checks
def identity(d: pd.DataFrame) -> pd.DataFrame:
    """Per model x variant x rule: pub vs no-filing on targets with no filing row in the pub evidence."""
    rows = []
    for (m, v, rule), g in d[d.model != NAIVE].groupby(["model", "variant", "rule"]):
        a, b = g[g.evidence == "nofiling"].set_index("target"), g[g.evidence == "pub"].set_index("target")
        if not len(a) or not len(b):
            continue
        un = b[b.filing_evidence == ""].index
        common = un.intersection(a.index)
        rel_fc = (b.loc[common, "forecast"] / a.loc[common, "forecast"] - 1).abs()
        rel_sd = (b.loc[common, "sd"] / a.loc[common, "sd"] - 1).abs()
        flips = [t for t in common if a.loc[t, "in80"] != b.loc[t, "in80"]]
        # what the model.py fix predicts for N's revenue sd: base_var (s y)^2 -> (s mu)^2, s = gaap log_sd
        s_g = config.TIERS["gaap"]["log_sd"]
        rev = [t for t in common if a.loc[t, "metric"] == "revenue"]
        pred = {t: math.sqrt(a.loc[t, "sd"] ** 2 + s_g ** 2 * (a.loc[t, "forecast"] ** 2 - a.loc[t, "actual"] ** 2)) for t in rev}
        resid = max([abs(b.loc[t, "sd"] / pred[t] - 1) for t in rev], default=np.nan)
        resid_rr = max([abs(b.loc[t, "sd"] / a.loc[t, "sd"] - 1) for t in common if t not in rev], default=np.nan)
        ch = b[b.filing_evidence != ""].index.intersection(a.index)
        same_ch = [t for t in ch if abs(b.loc[t, "forecast"] / a.loc[t, "forecast"] - 1) < 1e-12]
        rows.append({"rule": rule, "model": m, "variant": v,
                     "targets nofiling/pub": f"{len(a)}/{len(b)}",
                     "only in one": ", ".join(sorted(set(a.index) ^ set(b.index))) or "-",
                     "n unchanged-evidence": len(common),
                     "max rel diff forecast": rel_fc.max() if len(common) else np.nan,
                     "max rel diff sd": rel_sd.max() if len(common) else np.nan,
                     "sd diff on": ", ".join(t for t in common if rel_sd[t] > 1e-12) or "-",
                     "revenue sd vs fix-predicted (max rel)": resid,
                     "run-rate sd (max rel diff)": resid_rr,
                     "in80 flips": ", ".join(flips) or "-",
                     "n with filing rows": len(ch),
                     "with filing rows but same forecast": ", ".join(same_ch) or "-"})
    return pd.DataFrame(rows)


def leak_table(d: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Every filing row recorded in every pub file: published <= vintage < target publication."""
    raw = data.load().set_index("obs_id")
    rows, bad = [], []
    for f in sorted(os.listdir(REP)):
        if not (f.startswith("forecast_pubfilings_") and f.endswith(f"_{ASOF}.csv")):
            continue
        x = pd.read_csv(os.path.join(REP, f))
        for r in x.itertuples():
            ids = [] if pd.isna(r.filing_evidence) else [s.split("@") for s in str(r.filing_evidence).split(";")]
            v = pd.Timestamp(r.vintage)
            tpub = raw.loc[r.target, "published"]
            for oid, pub in ids:
                pub_raw = raw.loc[oid, "published"]
                ok = pd.Timestamp(pub) == pub_raw and pub_raw <= v and pub_raw < tpub and oid != r.target \
                    and raw.loc[oid, "origin_id"] != raw.loc[r.target, "origin_id"]
                if not ok:
                    bad.append((f, r.target, oid, pub))
                rows.append({"file": f, "target": r.target, "vintage": v.date(), "target_published": tpub.date(),
                             "filing_row": oid, "filing": int(raw.loc[oid, "filing"]), "published": pub_raw.date(),
                             "value": raw.loc[oid, "value"], "ok": ok})
    return pd.DataFrame(rows), bad


# --------------------------------------------------------------------------- tables
def common(d: pd.DataFrame) -> pd.DataFrame:
    """Rows whose (rule, target) every (model, evidence) of the frame forecast."""
    out = []
    for rule, g in d.groupby("rule"):
        combos = g[["model", "evidence"]].drop_duplicates()
        cnt = g.groupby("target").apply(lambda h: len(h[["model", "evidence"]].drop_duplicates()))
        out.append(g[g.target.isin(cnt[cnt == len(combos)].index)])
    return pd.concat(out, ignore_index=True)


def summary(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = [LABELS[k] for k in MODELS] + [NAIVE]
    for (rule, model, ev), h in c.groupby(["rule", "model", "evidence"]):
        post, h2, ch = h[h.post_t0], h[h.target.isin(H2)], h[h.changed]
        rows.append({"rule": rule, "model": model, "evidence": ev,
                     "n_all": len(h), "mae_all": h.log_err.abs().mean(), "bias_all": h.log_err.mean(),
                     "n_post": len(post), "mae_post": post.log_err.abs().mean(),
                     "n_h2": len(h2), "mae_h2": h2.log_err.abs().mean(),
                     "n_chg": len(ch), "mae_chg": ch.log_err.abs().mean(), "bias_chg": ch.log_err.mean(),
                     "cov80": h.in80.dropna().astype(float).mean() if model != NAIVE else np.nan,
                     "cov80_chg": ch.in80.dropna().astype(float).mean() if model != NAIVE else np.nan,
                     "cov50": h.in50.dropna().astype(float).mean() if model != NAIVE else np.nan})
    s = pd.DataFrame(rows)
    s["o"] = s.model.map(order.index) * 2 + (s.evidence == "pub")
    return s.sort_values(["rule", "o"]).drop(columns="o").reset_index(drop=True)


def paired(c: pd.DataFrame) -> pd.DataFrame:
    dd = c.assign(model=c.model + " [" + c.evidence + "]")
    out = []
    for rule in RULES:
        for subset, cc in (("all", dd), ("post-t0", dd[dd.post_t0]), ("H2-2026", dd[dd.target.isin(H2)]),
                           ("changed", dd[dd.changed])):
            for k in MODELS:
                m = LABELS[k]
                for a, b in ((f"{m} [nofiling]", f"{m} [pub]"), (f"{NAIVE} [pub]", f"{m} [pub]"),
                             (f"{NAIVE} [nofiling]", f"{m} [nofiling]")):
                    p = forecast_eval.paired(cc, a, b, rule)
                    if p:
                        out.append({"subset": subset, **p})
            p = forecast_eval.paired(cc, f"{NAIVE} [nofiling]", f"{NAIVE} [pub]", rule)
            if p:
                out.append({"subset": subset, **p})
    return pd.DataFrame(out)


def _by_date(items) -> str:
    """["a@2026-10-26", "b@2026-10-26", "c@2026-09-30"] -> "2026-09-30: c; 2026-10-26: a, b"."""
    g = {}
    for x in items:
        oid, day = x.split("@")
        g.setdefault(day, []).append(oid)
    return "; ".join(f"{day}: {', '.join(sorted(ids))}" for day, ids in sorted(g.items()))


def per_target(d: pd.DataFrame, rule: str, variant: str):
    g = d[(d.rule == rule) & d.changed & ((d.variant == variant) | (d.model == NAIVE))]
    if not len(g):
        return None, None
    targets = g.drop_duplicates("target").sort_values(["vintage", "target"])
    pub_fil = g[(g.evidence == "pub") & (g.model != NAIVE)].groupby("target").filing_evidence.apply(
        lambda s: sorted(set(x for v in s for x in v.split(";") if x)))
    a_fil = g[(g.evidence == "pub") & (g.model == LABELS["A"])].set_index("target").filing_evidence
    info, fc = [], []
    for t in targets.itertuples():
        nv = g[(g.model == NAIVE) & (g.target == t.target)].set_index("evidence")
        allf = pub_fil.get(t.target, [])
        notA = [x for x in allf if x not in set(str(a_fil.get(t.target, "")).split(";"))]
        info.append({"target": t.target, "vintage": t.vintage, "actual": t.actual,
                     "filing rows in the pub evidence, by publication date": _by_date([x for x in allf if x not in notA])
                     + (f" (B E G R bayes also: {_by_date(notA)})" if notA else ""),
                     "naive no-filing": f"{nv.loc['nofiling', 'forecast']:.3g}" if "nofiling" in nv.index else "",
                     "naive pub (input)": f"{nv.loc['pub', 'forecast']:.3g} ({nv.loc['pub', 'naive_from']})"
                     if "pub" in nv.index else ""})
        row = {"target": t.target, "actual": t.actual}
        for k in MODELS:
            h = g[(g.model == LABELS[k]) & (g.target == t.target)].set_index("evidence")
            if {"nofiling", "pub"} <= set(h.index):
                a_, b_ = h.loc["nofiling"], h.loc["pub"]
                row[k] = (f"{a_.forecast:.3g} -> {b_.forecast:.3g} ({a_.log_err:+.3f} -> {b_.log_err:+.3f}; "
                          f"in80 {'Y' if a_.in80 else 'N'}{'Y' if b_.in80 else 'N'})")
        fc.append(row)
    return pd.DataFrame(info), pd.DataFrame(fc)


def fmt(df, nd=3):
    return df.round(nd).to_markdown(index=False)


def main():
    d = load_long()
    d.to_csv(os.path.join(REP, f"eval_pubfilings_long_{ASOF}.csv"), index=False)
    idt = identity(d)
    leaks, bad = leak_table(d)
    prim = d[(d.variant == PRIMARY) | (d.model == NAIVE)]
    c = common(prim)
    s = summary(c)
    p = paired(c)
    sens = d[(d.variant == SENS) | (d.model == NAIVE)]
    cs = common(sens) if len(sens[sens.model != NAIVE]) else sens
    ss = summary(cs) if len(cs) else pd.DataFrame()
    ps = paired(cs) if len(cs) else pd.DataFrame()

    L = [f"# Filing numbers as evidence once public (sub-task O): backtest ({ASOF})", "",
         "Pre-registered in DATA_PLAN.md (子任务 O) before any run. Evidence rule **pub**: at each vintage every row",
         "published on or before the vintage is evidence, including filing = 1 (S-1, S-1/A, press reports of the draft",
         "S-1) and filing = 2 (third-party numbers published after a filing). The target's own origin stays out and",
         "backtest.check_no_leak stays on. Everything else mirrors sub-task N (scripts/forecast_interval.py): same",
         "frames, refit at every vintage, crossing position profiled, lower-claim rule iv115 (point as a sensitivity).",
         "Comparator: N's no-filing files `forecast_interval_<model>_<variant>_<rule>_2026-11-13.csv`.",
         "Errors are log(actual / forecast); + = forecast too low.", "",
         "**The sample is tiny.** Only targets forecast after 2026-09-30 can change: 9 under the publication rule (the six",
         "S-1 revenue numbers forecast on 2026-10-25, which can see only the Fortune draft-S-1 leak, plus rev_2026q3,",
         "rr_2026_09_s1a, rr_2026_10_s1a) and 3 under the period_end rule. Every interval below over 3-9 targets, one",
         "per target and highly correlated across models, is close to uninformative.", ""]
    for rule in RULES:
        g = c[c.rule == rule]
        L.append(f"- {rule}: {g.target.nunique()} common targets; changed (filing rows in the pub evidence): "
                 f"{', '.join(sorted(g[g.changed].target.unique()))}; post-t0 (N's flags): "
                 f"{', '.join(sorted(g[g.post_t0].target.unique()))}.")
    L += ["- Naive: backtest._naive on each run's own evidence (taken from model A's iv115 file), so under pub it may",
          "  carry forward a filing = 2 run-rate claim (semi74 / sacra70); for revenue it joins point run-rate claims and",
          "  never reads a filing revenue number. N's eval used the naive of A [baseline]; here both naives come from",
          "  A [iv115] (they differ from N's by the crossing position only: at most 0.021 in a revenue forecast).", ""]

    L += READING
    L += ["## Mean |log error| by model x evidence x rule, iv115 (common targets)", "",
          fmt(s[["rule", "model", "evidence", "n_all", "mae_all", "bias_all", "n_post", "mae_post", "n_h2", "mae_h2",
                 "n_chg", "mae_chg", "bias_chg", "cov80", "cov80_chg", "cov50"]]), "",
          "## Paired differences in |log error|, iv115 (`B vs A`: negative = B better; 90% bootstrap interval)", "",
          "On `all` and `post-t0` the unchanged targets contribute exact zeros to `pub vs nofiling`; `changed` is the",
          "informative subset.", ""]
    for rule in RULES:
        L += [f"### {rule}", "", fmt(p[p.rule == rule].drop(columns="rule")), ""]

    for rule in RULES:
        info, fc = per_target(d, rule, PRIMARY)
        if info is None:
            continue
        L += [f"## Changed targets, {rule}, iv115", "", info.to_markdown(index=False), "",
              "Forecast no-filing -> pub (log error no-filing -> pub; in 80% interval no-filing/pub):", "",
              fc.to_markdown(index=False), ""]

    if len(ss):
        L += [f"## Sensitivity: lower claims as points ({SENS}), A B E G R (N ran no bayes point)", "",
              fmt(ss[["rule", "model", "evidence", "n_all", "mae_all", "n_post", "mae_post", "n_h2", "mae_h2",
                      "n_chg", "mae_chg", "cov80", "cov80_chg"]]), "",
              fmt(ps[(ps.subset == "changed")]), ""]
        for rule in RULES:
            info, fc = per_target(d, rule, SENS)
            if fc is not None:
                L += [f"Changed targets, {rule}, {SENS}: forecast no-filing -> pub", "", fc.to_markdown(index=False), ""]

    L += ["## Check: unchanged targets identical to N's no-filing run", "",
          "Targets with no filing row in their pub evidence must reproduce N exactly. `max rel diff sd` on revenue",
          "targets comes from the orchestrator's uncommitted model.py fix (observation_predictive scales revenue noise",
          "by the forecast mean instead of the target value), which postdates N's runs; it never touches the mean.",
          "`revenue sd vs fix-predicted` applies that change to N's sd (sqrt(sd^2 + 0.003^2 (forecast^2 - actual^2)))",
          "and compares it with the pub run: ~1e-9 means the whole sd difference is the fix.", "",
          idt.to_markdown(index=False, floatfmt=".2e"), ""]

    ok = leaks[leaks.ok] if len(leaks) else leaks
    L += ["## Leak checks", "",
          f"1. Filing rows in the evidence, all {leaks.file.nunique() if len(leaks) else 0} pub files: "
          f"{len(leaks)} (file, target, row) entries, {len(bad)} violations of published <= vintage < target publication "
          "(dates re-read from observations.csv, not the deduplicated table). Per target and vintage (union over models):", ""]
    if len(leaks):
        rows = []
        for (t, v, tp), h in leaks.groupby(["target", "vintage", "target_published"]):
            rows.append({"target": t, "vintage": v, "target published": tp,
                         "filing rows (filing flag) by publication date": _by_date(
                             sorted({f"{r.filing_row} ({r.filing})@{r.published}" for r in h.itertuples()})),
                         "latest published": max(h.published), "all <= vintage": bool((h.published <= v).all()),
                         "pub files": h.file.nunique()})
        L += [pd.DataFrame(rows).sort_values(["vintage", "target"]).to_markdown(index=False), "",
              "The two seg_* share rows enter B, E, G, R and bayes only (model A's frame has no share rows). Same-day "
              "rule: under period_end the Fortune row (published 2026-09-30) enters forecasts dated 2026-09-30; the "
              "package's convention is published <= vintage for every row, filing or not.", ""]
    s1a = leaks[leaks.target.isin(["rr_2026_09_s1a", "rr_2026_10_s1a"]) &
                leaks.filing_row.isin(["rr_2026_09_s1a", "rr_2026_10_s1a"])] if len(leaks) else leaks
    L += [f"2. rr_2026_09_s1a and rr_2026_10_s1a (both S-1/A No. 2, published 2026-11-10): entries of one in the other's "
          f"evidence across all pub files: {len(s1a)}. Their latest vintage is 2026-11-09 (publication rule), and the "
          "recorder in forecast_pubfilings.py asserts on every forecast that no evidence row was published on or after "
          "the target's own publication date.", ""]
    ind = os.path.join(REP, f"pubfilings_answer_independence_{ASOF}.csv")
    if os.path.exists(ind):
        x = pd.read_csv(ind)
        fc_csv = []                                  # re-read: the check may have finished before a main run
        for r in x.itertuples():
            f = path(r.model, PRIMARY, r.rule, "pub")
            y = pd.read_csv(f) if os.path.exists(f) else None
            fc_csv.append(y[y.target == r.target].forecast.iloc[0] if y is not None else np.nan)
        x["forecast_csv"] = fc_csv
        x["rel_diff_vs_csv"] = (x.forecast / x.forecast_csv - 1).abs()
        L += ["3. Answer independence: target value x3, same call as run_backtest (refit, profiled crossing, iv115, pub).", "",
              x[["model", "target", "rule", "vintage", "mu", "sd", "mu_x3", "sd_x3", "abs_diff_mu", "abs_diff_sd",
                 "rel_diff_vs_csv"]].to_markdown(index=False, floatfmt=".3g"), ""]
    else:
        L += ["3. Answer independence: not run (scripts/check_pubfilings_independence.py).", ""]
    with open(os.path.join(REP, f"eval_pubfilings_{ASOF}.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
